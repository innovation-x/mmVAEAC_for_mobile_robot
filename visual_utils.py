import torch
from torchvision.io import read_video, write_video

def save_video(name, video, fps):
    # get the shape of video
    B, C, T, H, W = video.shape
    video = video.view(B * C, T, H, W)
    # Convert from float32 to uint8, use global minimum and global maximum - this is not the best solution
    min_val = video.min()
    max_val = video.max()
    video_as_uint8 = ((video - min_val) * 255 / (max_val - min_val)).to(torch.uint8)
    # Reorder the axes to be ordered as [T, H, W, C]
    vid_arr = torch.permute(video_as_uint8, (1, 2, 3, 0))
    vid_arr = vid_arr.cpu()
    write_video(name, vid_arr, fps=fps)
