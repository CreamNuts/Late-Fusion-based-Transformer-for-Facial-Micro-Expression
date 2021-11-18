from abc import *
from collections import defaultdict
from pathlib import Path
from typing import Any, List

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from einops import rearrange
from torch.utils.data import Dataset

from .utils import *


class Facial(Dataset):
    def __init__(
        self,
        data_path: str,
        interpol: bool,
        feature: str,
        total_frame: int,
        *,
        transform: List,
        subject: List = [],
    ) -> None:
        """
        Args:
            data_path(str): path to dataset
            interpol(bool): whether to interpolate video
            feature(str): feature of video
            total_frame(int): the number of frames which you want to fix in video
        Kwargs:
            subject(list): index of subject needed to include
            transform(list): list of transform
        """
        super(Facial, self).__init__()
        self.feature = feature
        self.interpol = interpol in ["linear", "tim"]

        if "OF" in self.feature:
            total_frame += 1
        if interpol == "linear":
            self.interpolation = FixSequenceInterpolation(total_frame)
        elif interpol == "tim":
            self.interpolation = FixSequenceTIM(total_frame)
        # Make pandas dataframe composed of video path and label(int)
        self.dataset = self.get_dataset(data_path, subject)

        self.transform = A.ReplayCompose(
            [
                *transform,
                ToTensorV2(),
            ]
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        video_path = self.dataset["Paths"].iloc[index]
        video = self.path2tensor(video_path)
        if self.interpol:
            video = self.interpolation(video)
        video = self.video2of(video)
        return video, self.dataset["Label"].iloc[index]

    def __len__(self):
        return len(self.dataset["Label"])

    def get_labels(self):
        return [self[i][1] for i in range(len(self))]

    def path2tensor(self, video_path: List[str]) -> torch.Tensor:
        """
        By using video path, open video and convert it to torch tensor
        Return: (S, C, H, W) torch.Tensor
        """
        video = []
        replay = None  # for applying same augmentation to all frames in video
        for frame_path in video_path:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY if "RGB" not in self.feature else cv2.COLOR_BGR2RGB
            )
            if not replay:
                frame = self.transform(image=frame)
                replay = frame["replay"]
            else:
                frame = A.ReplayCompose.replay(replay, image=frame)
            video.append(frame["image"])
        video = rearrange(video, "s c h w -> s c h w")
        return video

    def video2of(self, video):
        if self.feature == "ONLY_OF":
            video = getOFVideo(video)
        elif "OF" in self.feature:
            video = torch.cat([getOFVideo(video), video[:-1] / 255], axis=1)
        else:
            video = video / 255
        return video

    def get_sample(self):
        """
        Get data w.r.t label
        Return: List of tuples (tensor(S, C, H, W), label)
        """
        label_set = set(self.dataset["Label"])
        idx_list = []
        for idx, row in self.dataset.iterrows():
            if (label := row["Label"]) in label_set:
                idx_list.append((label, idx))
                label_set.remove(label)
            if not label_set:
                break
        idx_list.sort()
        return [self[idx] for _, idx in idx_list]

    def get_dataset(self) -> pd.DataFrame:
        """
        Make DataFrame consisting of subject, paths, label
        Return:
            DataFrame
                |  idx  |   Subject    |    Paths     |    Label   |
                |  int  |     int      | List of str  |     int    |
        """
        raise NotImplementedError


def get_dataset(mixin, *args, **kwargs):
    class Data(mixin, Facial):
        pass

    return Data(*args, **kwargs)


class SMICMixIn:
    str2int = {"ne": 0, "po": 1, "sur": 2}
    subject_list = [i for i in range(1, 21) if i not in [7, 10, 16, 17]]
    num_classes = 3

    def get_dataset(self, data_path: str, subject: List) -> pd.DataFrame:
        # Get all img's path
        path = Path(data_path) / "SMIC_CROP" / "HS"
        datadict = defaultdict(list)
        for img_path in path.rglob("micro/**/*.bmp"):
            datadict[img_path.parent.name].append(str(img_path))

        # Make DataFrame
        dataset = pd.DataFrame(columns=["Subject", "Paths", "Label"])
        for key in datadict.keys():
            info = key.split("_")  # Ex) s12_ne_01
            if (sub := int(info[0][1:])) in subject:
                datadict[key].sort()
                dataset = dataset.append(
                    {
                        "Subject": sub,
                        "Paths": datadict[key],
                        "Label": self.str2int[info[1]],
                    },
                    ignore_index=True,
                )
        dataset = dataset.sort_values(by="Subject").reset_index(drop=True)
        return dataset


class SAMMMixIn:
    str2int = {
        "Anger": 0,  # 35.84%/57 samples
        "Contempt": 1,  # 7.54%/12 samples
        "Happiness": 2,  # 16.35%/26 samples
        "Surprise": 3,  # 9.43%/15 samples
        "Others": 4,  # 16.35%/26 samples
        # 'Disgust': 5,  # 5.66%/9 samples
        # 'Fear': 6,  # 5.03%/8 samples
        # 'Sadness': 7,  # 3.77%/6 samples
    }
    subject_list = [i for i in range(6, 38) if i not in [8, 27, 29]]
    num_classes = 5

    def get_dataset(self, data_path: str, subject: List) -> pd.DataFrame:
        # Read xlsx
        samm_path = Path(data_path) / "SAMM" / "SAMM_CROP"
        xlsx_path = samm_path / "SAMM_Micro_FACS_Codes_v2.xlsx"
        dataset = pd.read_excel(xlsx_path, skiprows=range(13))[
            ["Subject", "Filename", "Estimated Emotion"]
        ]
        new_dataset = pd.DataFrame(columns=["Subject", "Paths", "Label"])
        # Get Video paths
        for idx, video in dataset.iterrows():
            if (emotion := video["Estimated Emotion"]) in self.str2int.keys() and (
                sub := video["Subject"]
            ) in subject:
                frame_paths = (samm_path / f"{sub:0>3}/{video['Filename']}").glob("*")
                video_path = sorted([str(path) for path in frame_paths])
                new_dataset.loc[idx] = [
                    video["Subject"],
                    video_path,
                    self.str2int[emotion],
                ]
        return new_dataset


class CASME2MixIn:
    str2int = {
        "disgust": 0,
        "repression": 1,
        # "sadness": 2,
        "happiness": 2,
        "others": 3,
        # 'fear': 5,
        "surprise": 4,
    }
    subject_list = list(range(1, 27))
    num_classes = 5

    def get_dataset(self, data_path: str, subject: List) -> pd.DataFrame:
        # Read xlsx
        casme2_path = Path(data_path) / "CASME2"
        xlsx_path = casme2_path / "CASME2-coding-20190701.xlsx"
        dataset = pd.read_excel(xlsx_path)[["Subject", "Filename", "Estimated Emotion"]]
        new_dataset = pd.DataFrame(columns=["Subject", "Paths", "Label"])
        # Get Video paths
        for idx, video in dataset.iterrows():
            if (emotion := video["Estimated Emotion"]) in self.str2int.keys() and (
                sub := video["Subject"]
            ) in subject:
                frame_paths = (casme2_path / "Cropped" / f"sub{sub:0>2}/{video['Filename']}").glob(
                    "*"
                )
                video_path = sorted([str(path) for path in frame_paths])
                new_dataset.loc[idx] = [
                    video["Subject"],
                    video_path,
                    self.str2int[emotion],
                ]
        return new_dataset
