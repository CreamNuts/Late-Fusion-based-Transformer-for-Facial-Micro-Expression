import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import asnumpy, rearrange

from .tim import tim


def rgb2gray(func):
    """
    Decorator for converting RGB numpy array to grayscale numpy array.
    If RGB array is input, (H, W, 3) is converted to (H, W, 1).
    If grayscale array is input, (H, W) is converted to (H, W, 1).
    """

    def rgb2gray(*args, **kwargs):
        newargs = []
        for data in args:
            if data.shape[-1] == 3:
                newargs.append(cv2.cvtColor(data, cv2.COLOR_RGB2GRAY))
            else:
                newargs.append(data.squeeze(-1))
        return func(*newargs, **kwargs)

    return rgb2gray


@rgb2gray
def calcOpticalFlow(prev: np.array, curr: np.array):
    """
    Calculate optical flow from two consecutive frames.
    Args:
        prev(np.array): shape is (H, W, 1), type is uint8 np array
        curr(np.array): shape is (H, W, 1), type is uint8 np array
    Return:
        np array of optical flow. Shape is (H, W, 3), type is float32
    """
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    opt = np.stack([mag, dx, dy], axis=-1)
    return opt


def getOFVideo(video: torch.Tensor) -> torch.Tensor:
    """
    Calculate optical flow from video.
    Args:
        video(torch.Tensor): shape is (S, C, H, W)
    """
    OF = []
    video = asnumpy(rearrange(video, "s c h w -> s h w c"))
    prev = video[0]
    for frame in video[1:]:
        OF.append(calcOpticalFlow(prev, frame))
        prev = frame
    OF = torch.from_numpy(rearrange(OF, "s h w c -> s c h w"))
    return OF


class FixSequenceInterpolation:
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def __call__(self, x) -> None:
        """
        Arrgs:
            x(torch.ByteTensor): shape is (S, C, H, W)
        Return:
            torch byte tensor of desired length's video. Shape is (S_desired, C, H, W)
        """
        with torch.no_grad():
            s, _, h, w = x.shape
            if s == self.sequence_length:
                return x
            else:
                x = x.float()
                x = rearrange(x, "s c h w -> 1 c s h w")
                x = F.interpolate(
                    x, size=(self.sequence_length, h, w), mode="trilinear", align_corners=False
                )
                x = rearrange(x, "1 c s h w -> s c h w")
                x = x.byte()
                return x

    def __str__(self):
        return str(self.sequence_length)


class FixSequenceTIM:
    def __init__(self, sequence_length: int) -> None:
        """
        Args:
            sequence_length(int): desired sequence length
        """
        self.sequence_length = sequence_length

    def __call__(self, x: torch.ByteTensor) -> torch.Tensor:
        """
        Args:
            x(torch.ByteTensor): shape is (S, C, H, W)
        Return:
            torch byte tensor of desired length's video. Shape is (S_desired, C, H, W)
        """
        with torch.no_grad():
            if x.shape[0] == self.sequence_length:
                return x
            else:
                x = x.numpy().astype(np.float32)
                x = tim(x, self.sequence_length)
                x = np.where(x <= 255, x, 255)
                x = np.where(x >= 0, x, 0)
                x = x.astype(np.uint8)
                return torch.from_numpy(x)
