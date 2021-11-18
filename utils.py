import random
from typing import List

import numpy as np
import torch
from einops import rearrange


def string2bool(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def size2tuple(list: List):
    if len(list) == 1:
        size = tuple(list) * 2
    else:
        size = tuple(list)
    return size


def unnormalize(input, mean, std):
    mean = rearrange(np.array(mean), "c -> () c () ()")
    std = rearrange(np.array(std), "c -> () c () ()")
    unnormal = input * std + mean
    return unnormal


def fix_seed():
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
