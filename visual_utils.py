import torch
from torchvision.io import read_video, write_video
import cv2
import numpy as np

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


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_gray_frames(video):
    """Convert labels or image frames to ``[T, H, W]`` uint8 grayscale frames."""
    frames = _as_numpy(video)
    is_label_map = frames.ndim == 3
    if frames.ndim == 4 and frames.shape[-1] > 3:
        frames = np.argmax(frames, axis=-1)
        is_label_map = True

    if frames.ndim == 3:
        if is_label_map:
            # The model uses nine semantic channels. Keeping this fixed range
            # makes origin and generated clips directly comparable.
            return np.rint(np.clip(frames, 0, 8) * (255.0 / 8.0)).astype(np.uint8)
        return np.rint(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)

    if frames.ndim != 4:
        raise ValueError("video must have shape [T, H, W] or [T, H, W, C]")

    if frames.shape[-1] == 1:
        frames = frames[..., 0]
    elif frames.shape[-1] == 3:
        frames = (
            0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
        )
    else:
        raise ValueError("video frames must have one, three, or one-hot channels")

    if np.issubdtype(frames.dtype, np.floating):
        frames = np.clip(frames, 0.0, 1.0) * 255.0
    return np.rint(frames).astype(np.uint8)


def _as_motion_sequence(motion, frame_count):
    values = _as_numpy(motion).squeeze()
    if values.ndim != 2:
        raise ValueError("motion must have shape [T, 2], [1, T, 2], or [2, T]")
    if values.shape == (2, frame_count):
        values = values.T
    if values.shape != (frame_count, 2):
        raise ValueError(
            f"motion shape {values.shape} does not match {frame_count} video frames"
        )
    return values.astype(np.float32)


def save_motion_video(name, video, motion, fps=5, title=None):
    """Write a grayscale video with a green motion arrow on every frame.

    Args:
        name: Output ``.mp4`` path.
        video: Semantic labels ``[T, H, W]`` or image/one-hot frames
            ``[T, H, W, C]``. All frames are exported in grayscale.
        motion: Per-frame ``[linear_velocity, angular_velocity]`` values.
        fps: Output frame rate.
        title: Retained for backwards compatibility; no text is drawn.
    """
    frames = _as_gray_frames(video)
    motions = _as_motion_sequence(motion, len(frames))
    _, height, width = frames.shape
    annotated_frames = []

    for frame, (linear_velocity, angular_velocity) in zip(frames, motions):
        canvas = np.repeat(frame[..., np.newaxis], 3, axis=-1)
        origin = (width // 2, height - 7)
        horizontal = int(np.clip(angular_velocity, -1.0, 1.0) * (width * 0.32))
        vertical = int(np.clip(linear_velocity, -1.0, 1.0) * (height * 0.45))
        endpoint = (
            int(np.clip(origin[0] + horizontal, 3, width - 4)),
            int(np.clip(origin[1] - vertical, 3, height - 4)),
        )
        if endpoint == origin:
            endpoint = (origin[0], max(3, origin[1] - 8))
        cv2.arrowedLine(canvas, origin, endpoint, (0, 255, 0), 2, tipLength=0.28)
        cv2.circle(canvas, origin, 2, (0, 255, 0), -1)
        annotated_frames.append(canvas)

    write_video(name, torch.from_numpy(np.stack(annotated_frames)), fps=fps)
