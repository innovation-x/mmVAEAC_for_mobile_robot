import argparse
import os.path

import torch
import numpy as np
from torch import nn,optim
from videovae import VAEs
from torch import nn
from torch.utils.data import DataLoader  #Gives easier data management by creating mini batches etc.
from tqdm import tqdm
import matplotlib.pyplot as plt

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# The parser setting
parser = argparse.ArgumentParser()
parser = VAEs.add_model_specific_args(parser)
# parser.add_argument('--data_path', type=str, default='./datasets/livingroom/')
parser.add_argument('--data_path', type=str, default='./datasets/ATR_1F_3F_2024-12-25_withtest')

parser.add_argument('--sequence_length', type=int, default=16)
parser.add_argument('--resolution', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--LR_RATE', type=float, default=5e-5)
parser.add_argument('--NUM_EPOCHS', type=int, default=100)
args = parser.parse_args()
squence_length = 16
half_seq_len = int(squence_length/2)


##################### Data Loading #####################
from datasets.video_dataset import VideoDataset
# Load the Dataset
Dataset = VideoDataset
dataset = Dataset(args.data_path, args.sequence_length, train=True, resolution=args.resolution)

# dataloader for the trainning
dataloader_train = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
    shuffle=True
)

# dataloader for the evaluation
dataset = Dataset(args.data_path, args.sequence_length, train=False, resolution=args.resolution)
dataloader_val = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
    shuffle=False
)

##################### Pretrained Model Loading #####################
from videovae.pretrained_VAEAC import GaussianLoss, MobileNetDecoder, MobileNetEncoder
reonstruction_log_prob = GaussianLoss()
mobilenet_enc_proposal = MobileNetEncoder(96)
mobilenet_enc_prior = MobileNetEncoder(96)

# improve train computational stability by dividing the loss by this scale factor right before backpropagation
vlb_scale_factor = 64 ** 2 * 16

MLP_network_proposal = nn.Sequential(
    nn.Linear(1000, 1024)
)

MLP_network_prior = nn.Sequential(
    nn.Linear(1000, 1024)
)

MLP_network_vid = nn.Sequential(
    nn.Linear(512, 1024),

    nn.Linear(1024, 2048),
)

generative_network = MobileNetDecoder(96)

from videovae.pretrained_VAEAC import VAEAC
MobileNet_pretrain = VAEAC(
    reonstruction_log_prob, mobilenet_enc_proposal, mobilenet_enc_prior,
                 MLP_network_proposal, MLP_network_prior, MLP_network_vid, generative_network,
).to(DEVICE)
# model = nn.DataParallel(model)
# modelData = './checkpoints/pretrained_mobilenet/600e_seg_4_2.pth.tar'
# MobileNet_pretrain.load_state_dict(torch.load(modelData)['model_state_dict'])


########################## mmVAEAC Model Loading ##########################
reonstruction_log_prob = GaussianLoss()
mobilenet_enc_proposal = MobileNet_pretrain.mobilenet_enc_proposal
mobilenet_enc_prior = MobileNet_pretrain.mobilenet_enc_prior

# improve train computational stability by dividing the loss by this scale factor right before backpropagation
vlb_scale_factor = 64 ** 2 * 16

input_mot_channel = squence_length*4
mot_proposal_network = nn.Sequential(
    nn.Linear(input_mot_channel,input_mot_channel),
    nn.ReLU(),
    nn.Linear(input_mot_channel,128),
    nn.ReLU(),
    nn.Linear(128,256),
)

mot_prior_network = nn.Sequential(
    nn.Linear(input_mot_channel,input_mot_channel),
    nn.ReLU(),
    nn.Linear(input_mot_channel,128),
    nn.ReLU(),
    nn.Linear(128,256),
)

MLP_network_proposal = nn.Sequential(
    nn.Linear(1000 + 256, 1024),
    nn.Linear(1024, 1024),
)

MLP_network_prior = nn.Sequential(
    nn.Linear(1000 + 256, 1024),
    nn.Linear(1024, 1024),
)

MLP_network_vid = nn.Sequential(
    nn.Linear(512, 1024),
    nn.Linear(1024, 2048),

)

mot_generative_network = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 32),
    # nn.Sigmoid()
)

mobilenet_generative_network = MobileNet_pretrain.generative_network


from videovae.mmVAEAC import mmVAEAC
batch_size = args.batch_size
eval_losses = []
validation_iwae = []
checkpoint_path = os.path.join('./checkpoints/pretrained_mmVAEAC/')


model = mmVAEAC(
    reonstruction_log_prob, mobilenet_enc_proposal, mobilenet_enc_prior,
    mot_proposal_network, mot_prior_network, MLP_network_proposal, MLP_network_prior, MLP_network_vid, mot_generative_network, mobilenet_generative_network
).to(DEVICE)

############################################ Continue from the checkpoints ############################################
# modelData = './checkpoints/pretrained_mmVAEAC/lvr_go3_60e.pth.tar'
# checkpoint = torch.load(modelData)
# model.load_state_dict(checkpoint['model_state_dict'])

############################################ Train the Model ############################################

from torch.utils.tensorboard import SummaryWriter
from videovae.utils import extend_batch, extend_motion
from videovae.mmVAEAC import VideoFixedSequenceGenerator

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="mmVAEAC",

    # track hyperparameters and run metadata
    config={
    "learning_rate": args.LR_RATE,
    "epochs": args.NUM_EPOCHS,
    }
)
wandb.define_metric("Epoch")
wandb.define_metric("mean_loss", step_metric="Epoch")
wandb.define_metric("mean_recon_loss", step_metric="Epoch")
wandb.define_metric("mean_kl_div", step_metric="Epoch")

mask_generator = VideoFixedSequenceGenerator()
optimizer = optim.Adam(model.parameters(), lr=args.LR_RATE)

#start Training
for epoch in range(args.NUM_EPOCHS):
    vlb_losses = []
    recon_losses = []
    KL_losses = []
    pr_losses = []
    model.train()
    loop = tqdm(dataloader_train)
    for x in loop:
        loop.set_description(f"Epoch {epoch}")
        batch, _, _ = x
        batch, motion = dict.values(batch)
         # (BCTHW)
        batch = batch.to(DEVICE)
        motion = motion.permute(0,2,1).to(DEVICE)
        # print(motion.shape)
        # print(motion.mean(dim=4).mean(dim=3))
        batch = extend_batch(batch, dataloader_train, batch_size)
        motion = extend_motion(motion, dataloader_train, batch_size)
        mask_batch = mask_generator(batch)
        mask_motion = mask_generator(motion)
        # print(mask_motion.shape)
        # Make the shape of motion be the same with video
        if (motion.shape[0]!= args.batch_size):
            # print(1)
            continue
        # print(batch.shape, motion.shape)

        optimizer.zero_grad()
        recon_loss, kl_div, pr_loss = model.batch_vlb(batch, motion, mask_batch, mask_motion)
        recon_loss, kl_div, pr_loss = recon_loss, kl_div.mean()/vlb_scale_factor, pr_loss.mean()/ vlb_scale_factor
        vlb = - recon_loss - kl_div + pr_loss
        loss = - vlb

        loss.backward()
        optimizer.step()

        loop.set_postfix(vlbloss=loss.item(), recon_loss=recon_loss.item(), kl_div=kl_div.item(), pr_loss=pr_loss.item())

    if ((epoch+1) % 5 == 0):
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path + '/20250310_atr_data_no_pre_epoch_'  + str(epoch+1) + '.pth.tar')

    # log metrics to wandb
    log_dict = {
        "Epoch":epoch,
        "mean_loss": loss,
        "mean_recon_loss": recon_loss,
        "mean_kl_div": kl_div,
    }
    wandb.log(log_dict)

# Save the last checkpoint
torch.save({'model_state_dict': model.state_dict()}, checkpoint_path + '/m'  + str('%2f' % loss) + '.pth.tar')
# torch.save(model, checkpoint_path + '/m'  + str('%2f' % loss) + '.pth.tar')
