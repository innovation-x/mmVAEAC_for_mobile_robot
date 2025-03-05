from torch.nn import Module
from torch.distributions import kl_divergence
import torch
from videovae.pretrained_VAEAC import normal_parse_params
from torch import nn,optim

squence_length = 16
half_seq_len = int(squence_length/2)

class VideoFixedSequenceGenerator:
    """
    The fixed mask that masks actions and observations after timestep 8, which is consistent with the paper. 
    """
    def __init__(self, t1=half_seq_len, t2=squence_length, x1=0, y1=0, x2=64, y2=64):
        self.t1 = t1
        self.t2 = t2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
    def __call__(self, batch):
        mask = torch.zeros_like(batch)
        mask[:, self.t1:self.t2] = 1
        return mask
    
mask_generator = VideoFixedSequenceGenerator()

class mmVAEAC(Module):
    """
    Variational Autoencoder with Arbitrary Conditioning core model.
    It is rather flexible, but have several assumptions:
    + The batch of objects and the mask of unobserved features
      have the same shape.
    + The prior and proposal distributions in the latent space
      are component-wise independent Gaussians.
    The constructor takes
    + Prior and proposal network which take as an input the concatenation
      of the batch of objects and the mask of unobserved features
      and return the parameters of Gaussians in the latent space.
      The range of neural networks outputs should not be restricted.
    + Generative network takes latent representation as an input
      and returns the parameters of generative distribution
      p_theta(x_b | z, x_{1 - b}, b), where b is the mask
      of unobserved features. The information about x_{1 - b} and b
      can be transmitted to generative network from prior network
      through nn_utils.MemoryLayer. It is guaranteed that for every batch
      prior network is always executed before generative network.
    + Reconstruction log probability. rec_log_prob is a callable
      which takes (groundtruth, distr_params, mask) as an input
      and return vector of differentiable log probabilities
      p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
    + Sigma_mu and sigma_sigma are the coefficient of the regularization
      in the hidden space. The default values correspond to a very weak,
      almost disappearing regularization, which is suitable for all
      experimental setups the model was tested on.
    """
    def __init__(self, rec_log_prob, mobilenet_enc_proposal, mobilenet_enc_prior,
                 mot_proposal_network, mot_prior_network, MLP_network_proposal, MLP_network_prior, MLP_network_vid, mot_generative_network, mobilenet_generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.mobilenet_enc_proposal = mobilenet_enc_proposal
        self.mobilenet_enc_prior = mobilenet_enc_prior

        self.mot_proposal_network = mot_proposal_network
        self.mot_prior_network = mot_prior_network

        self.MLP_network_proposal = MLP_network_proposal
        self.MLP_network_prior = MLP_network_prior

        self.MLP_network_vid = MLP_network_vid
        self.generative_network = mobilenet_generative_network
        self.mot_generative_network = mot_generative_network

        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma
        self.dropout = nn.Dropout(0.5)

    def make_observed(self, batch, mask):
        """
        Copy batch of objects and zero unobserved features.
        """
        observed = torch.tensor(batch)
        observed[mask.byte()] = 0
        return observed

    def make_latent_distributions(self, batch, mot, mask_batch, mask_motion, no_proposal=False):
        """
        Make latent distributions for the given batch and mask.
        No no_proposal is True, return None instead of proposal distribution.
        """
        B = batch.size(0)
        observed = self.make_observed(batch, mask_batch)
        
        batch = batch.permute(0,4,1,2,3).reshape(B,-1, 64,64)
        mask_batch = mask_batch.permute(0,4,1,2,3).reshape(B,-1, 64,64)
        observed_vid = observed.permute(0,4,1,2,3).reshape(B, -1, 64,64)
        observed_mot = self.make_observed(mot, mask_motion)

        observed_vid = torch.cat([observed_vid,mask_batch], 1)
        observed_mot = torch.cat([observed_mot,mask_motion], 2).flatten(1)

        if no_proposal:
            proposal = None
        else:
            full_info_vid = torch.cat([batch, mask_batch], 1)
            full_info_mot = torch.cat([mot, mask_motion], 2).flatten(1)

            proposal_params_vid = self.mobilenet_enc_proposal(full_info_vid)
            proposal_params_mot = self.mot_proposal_network(full_info_mot)

            latent_proposal = torch.cat([proposal_params_vid, proposal_params_mot], 1) # (bs, 1256)
            latent_proposal = normal_parse_params(self.MLP_network_proposal(latent_proposal), 1e-3) # (bs, 1024) --> (bs, 512)

        prior_params_vid = self.mobilenet_enc_prior(observed_vid)
        prior_params_mot = self.mot_prior_network(observed_mot)

        latent_prior = torch.cat([prior_params_vid, prior_params_mot], 1) # (bs, 1256)
        latent_prior = normal_parse_params(self.MLP_network_prior(latent_prior), 1e-3) # (bs, 1024) --> (bs, 512)

        return latent_proposal, latent_prior

    def prior_regularization(self, prior):
        """
        The prior distribution regularization in the latent space.
        Though it saves prior distribution parameters from going to infinity,
        the model usually doesn't diverge even without this regularization.
        It almost doesn't affect learning process near zero with default
        regularization parameters which are recommended to be used.
        """
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def batch_vlb(self, batch, mot, mask_batch, mask_motion):
        """
        Compute differentiable lower bound for the given batch of objects
        and mask.
        """
        latent_proposal, latent_prior = self.make_latent_distributions(batch, mot, mask_batch, mask_motion)
        bs = batch.size(0)
        prior_regularization = self.prior_regularization(latent_prior)
        
        latent = latent_proposal.rsample() # 512
        # latent_vid = self.MLP_network_vid(latent).reshape(bs,128,1,4,4) #2048 --> 128*4*4
        
        rec_params_vid = self.generative_network(latent).reshape(bs,18,16,64,64)
        rec_params_mot = self.mot_generative_network(latent) # (32)

        rec_loss = self.rec_log_prob(batch, mot, rec_params_vid, rec_params_mot)
        
        kl = kl_divergence(latent_proposal, latent_prior).view(batch.shape[0], -1).sum(-1)
        
        return rec_loss, kl, prior_regularization
    
    def forward(self, batch, mot):
        mask_batch = mask_generator(batch)
        mask_motion = mask_generator(mot)
        latent_vid_proposal, latent_mot_proposal, latent_vid_prior, latent_mot_prior = self.make_latent_distributions(batch, mask_batch, mask_motion)

        latent_vid = latent_vid_prior.rsample() # 128
        latent_mot = latent_mot_prior.rsample().flatten(1) # 64
        rec_params_vid = self.pretrain_video_dec(latent_vid)
        rec_params_mot = self.mot_generative_network(latent_mot)
        
        return rec_params_vid, rec_params_mot
    
    def generate_samples_params(self, batch, mot, mask_batch, mask_motion, K=1):
        """
        Generate parameters of generative distributions for samples
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        It is better to use it inside torch.no_grad in order to save memory.
        With torch.no_grad the method doesn't require extra memory
        except the memory for the result.
        """
        latent_proposal, latent_prior = self.make_latent_distributions(batch, mot, mask_batch, mask_motion)
  
        bs = batch.size(0)
        
        latent = latent_prior.rsample() # 512
        # latent_vid = self.MLP_network_vid(latent).reshape(bs,128,1,4,4) #2048 --> 128*4*4
        rec_params_vid = self.generative_network(latent).reshape(bs,18,16,64,64)
        rec_params_mot = self.mot_generative_network(latent) # (32)

        return rec_params_vid, rec_params_mot
