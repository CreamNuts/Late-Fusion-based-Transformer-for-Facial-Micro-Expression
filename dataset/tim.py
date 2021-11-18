from typing import Dict

import numpy as np
from einops import rearrange, repeat


def getPathWeightMatrix(n):
    adj = np.zeros((n, n))
    for i in range(0, n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def getLaplacian(matrix):
    d = np.zeros((matrix.shape[0], matrix.shape[0]))
    d[np.diag_indices(matrix.shape[0])] = np.sum(matrix, axis=1)
    return d - matrix


def trainPGM(x: np.array) -> Dict:
    """
    Input
        x(np.array): (n, d) flattend video, type is float
    Output
        dict: model's parameters
    """
    n, _ = x.shape
    feature = rearrange(x, "n d-> d n")
    # if np.linalg.matrix_rank(feature) < n:
    #     print(f"Warning: video's rank is {np.linalg.matrix_rank(feature)} < {n}")
    #     return False
    mu = np.mean(feature, axis=1)
    feature = feature - repeat(mu, "m -> m n", n=n)
    u, s, vh = np.linalg.svd(feature, full_matrices=False)
    u, s, vh = u[:, :-1], np.diag(s[:-1]), vh[:, :-1]
    q = s @ vh.T
    g = getPathWeightMatrix(n)
    l = getLaplacian(g)
    _, v = np.linalg.eig(l)
    v = v[:, 1:]
    w = np.linalg.solve(q @ q.T, q @ v)
    m = np.array(
        [
            q[:, 1].T @ w[:, i] / np.sin(1 / n * i * np.pi + np.pi * (n - i) / (2 * n))
            for i in range(n - 1)
        ]
    )
    return {"mu": mu, "u": u, "w": w, "m": m, "n": n}


def synPGM(model: dict, pos: np.array) -> np.array:
    """
    Input
        model(dict): PGM's parameters
        pos(np.array): array representing positions of each data at the same distance on the curve from 0 to 1
    Output
        np.array: (len(pos), d) interpolated video
    """
    n = model["n"]
    # rescale [0, 1] to [1/n, 1]
    pos = pos * (1 - 1 / n) + 1 / n
    video = np.zeros((model["u"].shape[0], len(pos)))
    for i, p in enumerate(pos):
        v = np.array([np.sin(p * j * np.pi + np.pi * (n - j) / (2 * n)) for j in range(1, n)])
        video[:, i] = model["u"] @ (np.linalg.solve(model["w"].T, v * model["m"])) + model["mu"]
    return video


def tim(video: np.array, target_frame: int) -> np.array:
    """
    Input
        video(np.array): (n, c, h, w) float video
        target_frame(int): num of interpolated frame
    Output
        np.array: (t, c, h, w) interpolated float video
    """
    n, c, h, w = video.shape
    video = rearrange(video, "n c h w -> n (c h w)")
    model = trainPGM(video)
    ratio = n / target_frame
    pos = np.arange(ratio * 1 / n, 1 + ratio * 1 / n, ratio * 1 / n)
    if len(pos) > target_frame:
        pos = pos[:-1]
    intp_video = synPGM(model, pos)
    intp_video = rearrange(intp_video, f"(c h w) t -> t c h w", h=h, w=w, c=c)
    return intp_video
