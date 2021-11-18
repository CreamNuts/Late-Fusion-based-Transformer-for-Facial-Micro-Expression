import argparse
import warnings

import albumentations as A
import cv2
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsampler import ImbalancedDatasetSampler

from dataset import *
from meter import TotalMeter
from model import create_model
from train import trainval
from utils import size2tuple

warnings.filterwarnings(action="ignore")


def _parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "MODEL",
        type=str,
        default="ch3_resnext",
        help="Like ch{Num}_{Backbone}. You can choose {Num} in [1, 3, 4, 6] and {Backbone} in [resnext, swin, timesformer].",
    )
    parser.add_argument(
        "DATA_PATH",
        type=str,
        help="Dataset directory except dataset name. Ex) /home/data not /home/data/SMIC",
    )
    parser.add_argument(
        "DATASET",
        type=str,
        default="SMIC",
        help="Choose your dataset in ['SMIC', 'CASME2', 'SAMM']",
    )
    parser.add_argument(
        "-tb", type=str, default=".", dest="TENSOR_BOARD", help="Tensorboad dir name"
    )
    parser.add_argument(
        "-imgsize",
        type=int,
        default=[224, 224],
        nargs="+",
        dest="IMG_SIZE",
        help="1 int or 2 int like 120(H) 64(W). For 1 int, resize width and height same",
    )

    parser.add_argument(
        "-feature",
        type=str,
        default="RGB",
        choices=["RGB", "GRAY", "GRAY_OF", "ONLY_OF", "RGB_OF"],
        dest="FEATURE",
        help="Choose image mode or use OF with grayscale",
    )
    parser.add_argument(
        "-interpol",
        type=str,
        default="linear",
        choices=["linear", "tim"],
        dest="INTERPOL",
        help="Choose using interpolation for fixed sequence length",
    )
    parser.add_argument(
        "-num_frames",
        type=int,
        default=16,
        dest="NUM_FRAMES",
        help="Video length in frames",
    )
    parser.add_argument("-bs", type=int, default=1, dest="BATCH_SIZE", help="Train batch size")
    parser.add_argument("-lr", type=float, default=1e-5, dest="LR", help="Learning rate")
    parser.add_argument("-ep", type=int, default=50, dest="EPOCH", help="Num of epochs")
    parser.add_argument(
        "-gpu",
        type=int,
        default=[0],
        nargs="+",
        dest="GPU",
        help="GPU number to use. If not available, use CPU",
    )
    parser.add_argument(
        "--imbalanced_sampler",
        action="store_true",
        default=False,
        dest="IMBALANCED",
        help="Flag to determine whether to use imbalanced sampler for balancing dataset with respect to class",
    )
    parser.add_argument(
        "--subtb",
        action="store_true",
        default=False,
        dest="SUB_TB",
        help="Flag to determine whether to track each subject metric with Tensorboard",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parser_args()
    img_size = size2tuple(args.IMG_SIZE)

    if "SAMM" in args.DATASET:
        mixin = SAMMMixIn
    elif "SMIC" in args.DATASET:
        mixin = SMICMixIn
    elif "CASME2" in args.DATASET:
        mixin = CASME2MixIn
    else:
        raise NotImplementedError
    subject_list = mixin.subject_list
    num_classes = mixin.num_classes
    str2int = mixin.str2int

    if args.IMBALANCED:
        sampler = ImbalancedDatasetSampler
    else:
        sampler = None

    train_transform = [
        A.Resize(height=int(img_size[0]), width=int(img_size[1])),  # for Random Crop, *1.145
        A.ShiftScaleRotate(
            border_mode=cv2.BORDER_REFLECT, p=0.5, shift_limit=0, scale_limit=0.1, rotate_limit=10
        ),
        A.HorizontalFlip(p=0.5),
    ]

    val_transform = [
        A.Resize(height=int(img_size[0]), width=int(img_size[1])),
    ]

    device = torch.device(f"cuda:{args.GPU[0]}" if torch.cuda.is_available() else "cpu")
    tb = f"{args.TENSOR_BOARD}"
    totalmeter = TotalMeter(f"{tb}/total/")
    for val_idx in subject_list:
        trainset = get_dataset(
            mixin,
            args.DATA_PATH,
            args.INTERPOL,
            args.FEATURE,
            args.NUM_FRAMES,
            subject=[idx for idx in subject_list if idx != val_idx],
            transform=train_transform,
        )
        valset = get_dataset(
            mixin,
            args.DATA_PATH,
            args.INTERPOL,
            args.FEATURE,
            args.NUM_FRAMES,
            subject=[val_idx],
            transform=val_transform,
        )

        if sampler:
            trainloader = DataLoader(
                trainset, batch_size=args.BATCH_SIZE, sampler=sampler(trainset), num_workers=8
            )
        else:
            trainloader = DataLoader(
                trainset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8
            )

        valloader = DataLoader(valset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=8)
        model = create_model(
            args.MODEL, image_size=img_size, num_frames=args.NUM_FRAMES, num_classes=num_classes
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=args.LR, betas=(0.9, 0.999), weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
        print(f"Subject {val_idx} Out")
        if args.SUB_TB:
            writer = SummaryWriter(f"{tb}/{val_idx}_{len(trainset)}|{len(valset)}/")
        else:
            writer = None
        subjmeter = trainval(
            device,
            args.EPOCH,
            model,
            optimizer,
            criterion,
            trainloader,
            valloader,
            writer,
        )
        if writer:
            writer.close()
        totalmeter.save(subjmeter)
    totalmeter.tensorboard(
        str2int=str2int,
        num_classes=num_classes,
        model=args.MODEL,
        imbalanced=args.IMBALANCED,
        interpolation=args.INTERPOL,
        feature=args.FEATURE,
        lr=args.LR,
        bsize=args.BATCH_SIZE,
        img_size=img_size,
    )
