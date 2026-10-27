import torch
import glob
import warnings
import joblib
import pickle
import math

import os.path as osp
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips
import torch.nn.functional as F

import argparse

import torch
import torch.nn.functional as F
import torch.utils.data as data

import math
import pickle
import warnings
import os.path as osp
import glob
from torchvision.datasets.video_utils import VideoClips

import joblib

class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, rgb=False, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.rgb = rgb
        camera_motions = []
        camera_mot = []
        # screen all folders under the directory

        folder = osp.join(data_folder, 'train' if train else 'test')
        # files = sorted(sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
        #              for ext in self.exts], []))

        # motion_files = sorted(glob.glob(osp.join(folder, '**', f'*.csv'), recursive=True))

        files = sorted(sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], []), key=lambda x: (osp.basename(osp.dirname(x)), int(osp.splitext(osp.basename(x))[0])))

        motion_files = sorted(glob.glob(osp.join(folder, '**', f'*.csv'), recursive=True),
                              key=lambda x:(osp.basename(osp.dirname(x)), int(osp.splitext(osp.basename(x))[0])))
        # print(files)
        # print(motion_files)
        for motion in motion_files:
            mot = torch.load(motion)
            if mot.shape[0] != 2:
                mot = mot.permute(1,0)
            camera_mot.append(mot)
        # create or open the metadata
        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        cache_file_mot = osp.join(folder, f"metadata_motion_{sequence_length}.pkl")
        # Given a list of video files, computes all consecutive subvideos of size `clip_length_in_sequence_length`
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=4, frames_between_clips=1)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
            # cut the motion into clips
            for i in range(len(motion_files)):
                motion_clips = camera_mot[i].float()
                # Normalize the motion
                motion_clips[0] = abs(motion_clips[0])*0.7
                motion_clips[1] = torch.round(motion_clips[1]*0.45*10)/10

                segment_poses = motion_clips
                #(137,2,64)
                segment_poses = segment_poses.unfold(1,sequence_length,1).permute(1,0,2)
                camera_motion = segment_poses.split(1,dim=0)
                camera_motion_squeese = [tensor.squeeze(0) for tensor in camera_motion]

                camera_motions += camera_motion_squeese
            metadata_mot = {'motion':camera_motions}
            joblib.dump(metadata_mot, cache_file_mot)

        else:
            print("metadata exists in:", cache_file)
            metadata = pickle.load(open(cache_file, 'rb'))
            # print(metadata)
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)
            metadata_mot = joblib.load(cache_file_mot)
            camera_motions = metadata_mot['motion']

        self._clips = clips
        self._motions = camera_motions
        self.files=files
        self.motion_files = motion_files

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        # get_clips return video(tensor) audio(tensor) info(Dict) video_idx (int): index of the video in `video_paths`
        # print(len(self._clips))
        # print(len(self._motions))
        video, _, _, vid_idx = self._clips.get_clip(idx)
        motion_clips = self._motions[idx]
        return dict(video=preprocess(video, resolution, self.rgb, self.sequence_length), motion=motion_clips), vid_idx, self.files[vid_idx]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))

def preprocess(video, resolution, rgb, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]
    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW
    if not rgb:
        video = video[0, :, :, :]
        # one hot encoding
        t,h,w = video.shape
        # print(video[0,10])
        flat_tensor = video.view(-1)
        class_tensor = torch.round(flat_tensor*10).to(torch.long)
        video_onehot = torch.nn.functional.one_hot(class_tensor,num_classes=9)
        video_onehot = video_onehot.reshape(t,h,w,9)
        video_onehot = video_onehot.to(torch.float)
    else:
        video_onehot = video.permute(1,2,3,0)

    return video_onehot




# class VideoDataset(data.Dataset):
#     """ Generic dataset for videos files stored in folders
#     Returns BCTHW videos in the range [-0.5, 0.5] """
#     exts = ['avi', 'mp4', 'webm']

#     def __init__(self, data_folder, sequence_length, train=True, resolution=64):
#         """
#         Args:
#             data_folder: path to the folder with videos. The folder
#                 should contain a 'train' and a 'test' directory,
#                 each with corresponding videos stored
#             sequence_length: length of extracted video sequences
#         """
#         super().__init__()
#         self.train = train
#         self.sequence_length = sequence_length
#         self.resolution = resolution
#         camera_motions = []
#         camera_mot = []
#         # screen all folders under the directory
#         folder = osp.join(data_folder, 'train' if train else 'test')
#         files = sorted(sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
#                      for ext in self.exts], []))

#         motion_files = sorted(glob.glob(osp.join(folder, '**', f'*.csv'), recursive=True))

#         for motion in motion_files:
#             camera_mot.append(torch.load(motion))
#         # create or open the metadata
#         warnings.filterwarnings('ignore')
#         cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
#         cache_file_mot = osp.join(folder, f"metadata_motion_{sequence_length}.pkl")
#         # Given a list of video files, computes all consecutive subvideos of size `clip_length_in_sequence_length`
#         if not osp.exists(cache_file):
#             clips = VideoClips(files, sequence_length, num_workers=4)
#             pickle.dump(clips.metadata, open(cache_file, 'wb'))
#             # cut the motion into clips
#             for i in range(len(motion_files)):
#                 motion_clips = camera_mot[i].float()
#                 # Normalize the motion
#                 motion_clips[0] = abs(motion_clips[0])*0.7
#                 motion_clips[1] = torch.round(motion_clips[1]*0.45*10)/10

#                 segment_poses = motion_clips
#                 #(137,2,64)
#                 segment_poses = segment_poses.unfold(1,sequence_length,1).permute(1,0,2)
#                 camera_motion = segment_poses.split(1,dim=0)
#                 camera_motion_squeese = [tensor.squeeze(0) for tensor in camera_motion]

#                 camera_motions += camera_motion_squeese
#             metadata_mot = {'motion':camera_motions}
#             joblib.dump(metadata_mot, cache_file_mot)

#         else:
#             print("metadata exists in:", cache_file)
#             metadata = pickle.load(open(cache_file, 'rb'))
#             # print(metadata)
#             clips = VideoClips(files, sequence_length,
#                                _precomputed_metadata=metadata)
#             metadata_mot = joblib.load(cache_file_mot)
#             camera_motions = metadata_mot['motion']

#         self._clips = clips
#         self._motions = camera_motions
#         self.files=files
#         self.motion_files = motion_files

#     @property
#     def n_classes(self):
#         return len(self.classes)

#     def __len__(self):
#         return self._clips.num_clips()

#     def __getitem__(self, idx):
#         resolution = self.resolution
#         # get_clips return video(tensor) audio(tensor) info(Dict) video_idx (int): index of the video in `video_paths`
#         video, _, _, _ = self._clips.get_clip(idx)
#         motion_clips = self._motions[idx]
#         return dict(video=preprocess(video, resolution, self.sequence_length), motion=motion_clips)



# def get_parent_dir(path):
#     return osp.basename(osp.dirname(path))

# def preprocess(video, resolution, sequence_length=None):
#     # video: THWC, {0, ..., 255}
#     video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
#     t, c, h, w = video.shape

#     # temporal crop
#     if sequence_length is not None:
#         assert sequence_length <= t
#         video = video[:sequence_length]
#     # scale shorter side to resolution
#     scale = resolution / min(h, w)
#     if h < w:
#         target_size = (resolution, math.ceil(w * scale))
#     else:
#         target_size = (math.ceil(h * scale), resolution)
#     video = F.interpolate(video, size=target_size, mode='bilinear',
#                           align_corners=False)

#     # center crop
#     t, c, h, w = video.shape
#     w_start = (w - resolution) // 2
#     h_start = (h - resolution) // 2
#     video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
#     video = video.permute(1, 0, 2, 3).contiguous() # CTHW

#     video = video[0, :, :, :]

#     # one hot encoding
#     t,h,w = video.shape
#     flat_tensor = video.view(-1)
#     class_tensor = torch.floor(flat_tensor*10).to(torch.long)
#     video_onehot = torch.nn.functional.one_hot(class_tensor,num_classes=9)
#     video_onehot = video_onehot.reshape(t,h,w,9)
#     video_onehot = video_onehot.to(torch.float)

#     return video_onehot
