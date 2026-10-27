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


_SEMANTIC_PALETTE = np.array(
    [
        (30, 30, 30),
        (220, 70, 70),
        (70, 180, 90),
        (80, 120, 230),
        (240, 190, 60),
        (180, 90, 200),
        (70, 210, 210),
        (235, 235, 235),
        (150, 150, 150),
    ],
    dtype=np.uint8,
)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_rgb_frames(video):
    """Convert ``[T, H, W]`` labels or ``[T, H, W, C]`` frames to RGB."""
    frames = _as_numpy(video)
    if frames.ndim == 4 and frames.shape[-1] > 3:
        frames = np.argmax(frames, axis=-1)

    if frames.ndim == 3:
        labels = np.rint(frames).astype(np.int64)
        return _SEMANTIC_PALETTE[np.clip(labels, 0, len(_SEMANTIC_PALETTE) - 1)]

    if frames.ndim != 4:
        raise ValueError("video must have shape [T, H, W] or [T, H, W, C]")

    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif frames.shape[-1] != 3:
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


def save_motion_video(name, video, motion, fps=5, title="Motion"):
    """Write an annotated video with a motion arrow below every frame.

    Args:
        name: Output ``.mp4`` path.
        video: Semantic labels ``[T, H, W]`` or RGB/one-hot frames
            ``[T, H, W, C]``.
        motion: Per-frame ``[linear_velocity, angular_velocity]`` values.
        fps: Output frame rate.
        title: Prefix displayed in the bottom motion panel.
    """
    frames = _as_rgb_frames(video)
    motions = _as_motion_sequence(motion, len(frames))
    _, height, width, _ = frames.shape
    panel_height = max(48, height // 2)
    annotated_frames = []

    for frame, (linear_velocity, angular_velocity) in zip(frames, motions):
        canvas = np.full((height + panel_height, width, 3), 18, dtype=np.uint8)
        canvas[:height] = frame
        cv2.line(canvas, (0, height), (width, height), (235, 235, 235), 1)

        origin = (width // 2, height + panel_height - 9)
        horizontal = int(np.clip(angular_velocity, -1.0, 1.0) * (width * 0.32))
        vertical = max(8, int(np.clip(abs(linear_velocity), 0.0, 1.0) * (panel_height * 0.65)))
        endpoint = (origin[0] + horizontal, origin[1] - vertical)
        cv2.arrowedLine(canvas, origin, endpoint, (80, 220, 255), 2, tipLength=0.28)
        cv2.circle(canvas, origin, 2, (255, 255, 255), -1)
        label = f"{title}  v={linear_velocity:+.2f}  w={angular_velocity:+.2f}"
        cv2.putText(
            canvas,
            label,
            (4, height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        annotated_frames.append(canvas)

    write_video(name, torch.from_numpy(np.stack(annotated_frames)), fps=fps)
