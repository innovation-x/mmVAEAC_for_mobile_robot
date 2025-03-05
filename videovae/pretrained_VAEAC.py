from torch.distributions import Normal
from torch.nn.functional import softplus
from torch.nn import Module
from torch import nn,optim
import torch

from typing import Callable, Optional, List

import torch
from torch import Tensor
from torch import nn

from torchvision.ops.misc import ConvNormActivation
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv2 import InvertedResidual
import torch.nn.functional as F

######### Loss Function ###############################
def sampler(params):
    return normal_parse_params(params).mean
def normal_parse_params(params, min_sigma=0):
    """
    Take a Tensor (e. g. neural network output) and return
    torch.distributions.Normal distribution.
    This Normal distribution is component-wise independent,
    and its dimensionality depends on the input shape.
    First half of channels is mean of the distribution,
    the softplus of the second half is std (sigma), so there is
    no restrictions on the input tensor.
    min_sigma is the minimal value of sigma. I. e. if the above
    softplus is less than min_sigma, then sigma is clipped
    from below with value min_sigma. This regularization
    is required for the numerical stability and may be considered
    as a neural network architecture choice without any change
    to the probabilistic model.
    """
    n = params.shape[0]
    d = params.shape[1]
    mu = params[:, :d // 2]
    sigma_params = params[:, d // 2:]
    sigma = softplus(sigma_params)
    sigma = sigma.clamp(min=min_sigma)
    distr = Normal(mu, sigma)
    return distr

class GaussianLoss(Module):
    """
    Compute reconstruction log probability of groundtruth given
    a tensor of Gaussian distribution parameters and a mask.
    Gaussian distribution parameters are output of a neural network
    without any restrictions, the minimal sigma value is clipped
    from below to min_sigma (default: 1e-2) in order not to overfit
    network on some exact pixels.
    The first half of channels corresponds to mean, the second half
    corresponds to std. See normal_parse_parameters for more info.
    This layer doesn't work with NaNs in the data, it is used for
    inpainting. Roughly speaking, this loss is similar to L2 loss.
    Returns a vector of log probabilities for each object of the batch.
    """
    def __init__(self, min_sigma=1e-2):
        super().__init__()
        self.min_sigma = min_sigma

    def forward(self, groundtruth, mot, distr_params, motion_pred, squence_length = 16):
        distr_params = sampler(distr_params)
        # B,T,H,W,C --> B,C,T,H,W
        groundtruth_am = torch.argmax(groundtruth,4)

        loss = nn.CrossEntropyLoss()
        recon_loss_vid = loss(distr_params, groundtruth_am)

        b,n = motion_pred.shape
        motion_pred = motion_pred.reshape(b, squence_length, 2)
        motion_pred_lin = motion_pred[:,:,0]
        motion_pred_ang = motion_pred[:,:,1]

        motion_groundtruth_li = mot[:,:,0]
        motion_groundtruth_ang = mot[:,:,1] + 0.5

        loss = nn.MSELoss()
        recon_loss_motion_lin = loss(motion_pred_lin, motion_groundtruth_li)
        recon_loss_motion_ang = loss(motion_pred_ang, motion_groundtruth_ang)
        
        recon_loss = 0.5*recon_loss_vid + 0.2*recon_loss_motion_lin + 0.3*recon_loss_motion_ang
        return recon_loss
    
################################ pretrained VAEAC ########################################
from torch.distributions import kl_divergence
class VAEAC(Module):
    def __init__(self, rec_log_prob, mobilenet_enc_proposal, mobilenet_enc_prior,
                 MLP_network_proposal, MLP_network_prior, MLP_network_vid, generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.mobilenet_enc_proposal = mobilenet_enc_proposal
        self.mobilenet_enc_prior = mobilenet_enc_prior
        self.MLP_network_proposal = MLP_network_proposal
        self.MLP_network_prior = MLP_network_prior

        self.MLP_network_vid = MLP_network_vid
        self.generative_network = generative_network

        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma
        
    def make_observed(self, batch, mask):
        """
        Copy batch of objects and zero unobserved features.
        """
        observed = torch.tensor(batch)
        observed[mask.byte()] = 0
        return observed
    
    def make_latent_distributions(self, batch, mask, no_proposal=False):
        """
        Make latent distributions for the given batch and mask.
        No no_proposal is True, return None instead of proposal distribution.
        """
        B = batch.size(0)
        observed = self.make_observed(batch, mask)
        # print(batch.permute(0,4,1,2,3).shape)
        batch = batch.permute(0,4,1,2,3).reshape(B,-1, 64,64)
        mask = mask.permute(0,4,1,2,3).reshape(B,-1, 64,64)

        observed = observed.permute(0,4,1,2,3).reshape(B, -1, 64,64)

        if no_proposal:
            proposal = None
        else:
            full_info = torch.cat([batch, mask], 1)
            # print(full_info.shape)
            proposal = normal_parse_params(self.MLP_network_proposal(self.mobilenet_enc_proposal(full_info)), 1e-3) # 1024 --> 512
            # print(proposal_params.shape)

        prior_params = self.mobilenet_enc_prior(torch.cat([observed, mask], 1)) #1000
        prior_params = self.MLP_network_prior(prior_params)
        prior = normal_parse_params(prior_params, 1e-3) # 512
        return proposal, prior
    
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
    
    def batch_vlb(self, batch, mask):
        """
        Compute differentiable lower bound for the given batch of objects
        and mask.
        """
        B= batch.size(0)
        proposal, prior = self.make_latent_distributions(batch, mask)
        prior_regularization = self.prior_regularization(prior)
        latent = proposal.rsample() # 512

        # rec_params = self.MLP_network_vid(latent).reshape(B,128,1,4,4) # 2048
        rec_params = self.generative_network(latent).reshape(B,18,16,64,64) # B, 18, 16, 64, 64 
        # print(rec_params.shape)
        rec_loss = self.rec_log_prob(batch, rec_params, mask)
        kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
        return rec_loss, kl, prior_regularization
    

    def generate_samples_params(self, batch, mask, K=1):
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
        B = batch.size(0)
        _, prior = self.make_latent_distributions(batch, mask)
        samples_params = []
        for i in range(K):
            latent = prior.rsample() # 500

            # rec_params = self.MLP_network_vid(latent).reshape(B,128,1,4,4) # 2048
            sample_params = self.generative_network(latent).reshape(B,18,16,64,64) # B, 18, 16, 64, 64 

            samples_params.append(sample_params.unsqueeze(1))
        return torch.cat(samples_params, 1)

    def generate_reconstructions_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for reconstructions
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
        _, prior = self.make_latent_distributions(batch, mask)
        reconstructions_params = []
        for i in range(K):
            latent = prior.rsample()
            rec_params = self.generative_network(latent)
            reconstructions_params.append(rec_params.unsqueeze(1))
        return torch.cat(reconstructions_params, 1)

########################  Encoder ########################################
# modified from PyTorch torchvision library
class MobileNetEncoder(nn.Module):
    def __init__(
        self,
        num_images: int = 1,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_images (int): number of images stacked in the input tensor
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features: List[nn.Module] = [
            ConvNormActivation(
                num_images * 3,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

################# Decoder ########################################
class MobileNetDecoder(nn.Module):
    def __init__(
        self,
        num_images: int = 1,
        num_classes: int = 512,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_images (int): number of images stacked in the input tensor
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [6, 320, 1, 1],
                [6, 160, 3, 2],
                [6, 96, 3, 2],
                [6, 64, 4, 2],
                [6, 32, 3, 2],
                [6, 24, 2, 2],
                [1, 16, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )
        
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        # # building first classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_classes, self.last_channel),
        )
        # building first layer
        features: List[nn.Module] = [
            ConvNormActivation(
                self.last_channel,
                input_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        ]
        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        1,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                features.append(
                    nn.Upsample(scale_factor=stride),
                )
                input_channel = output_channel
        
        features.append(
            ConvNormActivation(
                input_channel,
                num_images * 3,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.classifier(x)
        B = x.size(0)
        x = x.reshape(B,1280,1,1)
        x = F.upsample(x, scale_factor=2)
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
