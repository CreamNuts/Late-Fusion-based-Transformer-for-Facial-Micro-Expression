import random
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import asnumpy, rearrange


def size2tuple(list: List):
    if len(list) == 1:
        size = tuple(list) * 2
    else:
        size = tuple(list)
    return size


def fix_seed():
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_font_scale(text, width):
    for scale in reversed(range(0, 100, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale / 100, thickness=1
        )
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 100
    return 1


def puttext_video(video: np.array, frame_size, upper_text, lower_text):
    video_with_text = []
    for frame in video:
        frame = frame.copy()
        for text, h_mul in [(upper_text, 1), (lower_text, 9)]:
            cv2.putText(
                frame,
                text,
                (0, int(frame_size[0] / 10) * h_mul),
                cv2.FONT_HERSHEY_SIMPLEX,
                get_font_scale(text, frame_size[1]),
                (255, 255, 255),
                1,
            )
        video_with_text.append(frame)
    return video_with_text


class Visualizer:
    def __init__(self, label_map: Dict):
        self.int2str = label_map

    def __call__(self, model, input, label, rgb):
        """
        input: (S, C, H, W)
        output: (S, C, H, W)
        """
        _, s, _, h, w = input.data.size()
        model.eval()
        logit = asnumpy(F.softmax(model(input.cuda()), dim=-1))[0]
        lw_text = ""
        for i, v in enumerate(logit):
            lw_text += f"{self.int2str[i]}: {v:.2f} "
        vis = puttext_video(
            rearrange(asnumpy(rgb), "S C H W -> S H W C"),
            (h, w),
            upper_text=f"Answer-{self.int2str[label]}",
            lower_text=f"Pred-{lw_text}",
        )
        vis = rearrange(vis, "s h w c -> 1 s c h w") / 255
        vis = torch.from_numpy(vis)
        return vis

    def make_videos(self, model, videos):
        return torch.cat(
            [self(model, video[0].unsqueeze(0), video[1], video[2]) for video in videos],
            dim=0,
        )
