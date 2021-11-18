from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import reduce
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


def confusion2uf1(confusion):
    """
    confusion:
        ith row: true class
        jth col: predicted class
    """
    tp = np.diag(confusion)
    fp = reduce(confusion, "i j -> j", "sum") - tp
    fn = reduce(confusion, "i j -> i", "sum") - tp
    f1 = np.divide(2 * tp, denom := (2 * tp + fp + fn), where=denom != 0)
    uf1 = reduce(f1, "i -> ", "mean")
    return uf1


def confusion2uar(confusion):
    tp = np.diag(confusion)
    num_per_class = reduce(confusion, "i j -> i", "sum")
    ar = np.divide(tp, num_per_class, where=num_per_class != 0)
    uar = reduce(ar, "i -> ", "mean")
    return uar


def confusion2acc(confusion):
    tp = np.sum(np.diag(confusion))
    total = np.sum(confusion)
    return tp / total


def confusion2score(confusion) -> tuple[float, float, float]:
    acc = confusion2acc(confusion)
    uar = confusion2uar(confusion)
    uf1 = confusion2uf1(confusion)
    return acc, uar, uf1


def confusion2fig(train: np.ndarray, val: np.ndarray, label_map: List):
    train_denom = reduce(train, "i j -> i ()", "sum")
    train = np.divide(train, train_denom, where=train_denom != 0)
    train_cm = pd.DataFrame(train, label_map, label_map)
    val_denom = reduce(val, "i j -> i ()", "sum")
    val = np.divide(val, val_denom, where=val_denom != 0)
    val_cm = pd.DataFrame(val, label_map, label_map)
    sns.set(font_scale=1.4)
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))
    axs[0].set_title("Train"), axs[1].set_title("Val")
    for ax, cm in zip(axs, [train_cm, val_cm]):
        sns.heatmap(
            cm,
            vmin=0,
            vmax=1,
            annot=True,
            square=True,
            fmt="3.2g",
            annot_kws={"size": 32 / np.sqrt(len(cm))},
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig


class Meter:
    def __init__(self) -> None:
        self.confusion = 0
        self.loss = 0
        self.uars = []
        self.uf1s = []
        self.accs = []
        self.confusions = []
        self.losses = []
        self.best_confusion = 0
        self.num_of_subject = 0

    def __call__(self, target: np.ndarray, pred: np.ndarray, target_range: int):
        self.num_of_subject = 1  # Performance of model on current subject is recorded
        self.confusion += confusion_matrix(target, pred, labels=range(target_range))

    def __len__(self):
        return len(self.losses)

    def __add__(self, other):
        """
        In case of summing two meters, the result is the sum of the two meters.
        """
        if self.num_of_subject:
            new_meter = Meter()
            new_meter.num_of_subject = self.num_of_subject + other.num_of_subject
            new_meter.losses = [
                s - (s - o) / new_meter.num_of_subject for s, o in zip(self.losses, other.losses)
            ]
            new_meter.confusions = [s + o for s, o in zip(self.confusions, other.confusions)]
            new_meter.best_confusion = self.best_confusion + other.best_confusion
            return new_meter
        else:
            return other

    def save_best(self):
        idx = self.uf1s.index(max(self.uf1s))
        self.best_confusion = self.confusions[idx]

    def reset(self):
        """
        When the end of one epoch is reached, record the performance of the model and reset the meter.
        """
        if self.num_of_subject:
            self.confusions.append(self.confusion)
            self.losses.append(self.loss)
            acc, uar, uf1 = confusion2score(self.confusion)
            self.accs.append(acc)
            self.uars.append(uar)
            self.uf1s.append(uf1)
            self.save_best()
        self.confusion = 0
        self.loss = 0


class SubjMeter:
    def __init__(self, writer=None) -> None:
        self.train = Meter()
        self.val = Meter()
        self.writer = writer

    def __len__(self):
        return len(self.train)

    def reset(self):
        self.train.reset()
        self.val.reset()
        if self.writer:
            self.tensorboard()

    def tensorboard(self):
        if len(self.train):
            self.writer.add_scalars(
                "Acc", {"train": self.train.accs[-1], "val": self.val.accs[-1]}, len(self.train)
            )
            self.writer.add_scalars(
                "Uar", {"train": self.train.uars[-1], "val": self.val.uars[-1]}, len(self.train)
            )
            self.writer.add_scalars(
                "F1", {"train": self.train.uf1s[-1], "val": self.val.uf1s[-1]}, len(self.train)
            )
            self.writer.add_scalars(
                "Loss",
                {"train": self.train.losses[-1], "val": self.val.losses[-1]},
                len(self.train),
            )


class TotalMeter:
    def __init__(self, tensorboard_dir):
        self.tb_dir = tensorboard_dir
        self.train = Meter()
        self.val = Meter()

    def save(self, subjmeter: SubjMeter):
        self.train += subjmeter.train
        self.val += subjmeter.val

    def tensorboard(
        self,
        *,
        str2int,
        num_classes,
        model,
        imbalanced,
        interpolation,
        feature,
        lr,
        bsize,
        img_size,
    ):
        writer = SummaryWriter(log_dir=self.tb_dir)
        train_score = []
        val_score = []
        for idx in range(len(self.train)):
            train_score.append(confusion2score(self.train.confusions[idx]))
            val_score.append(confusion2score(self.val.confusions[idx]))
            writer.add_scalars(
                "AvgLoss", {"train": self.train.losses[idx], "val": self.val.losses[idx]}, idx
            )
            writer.add_scalars("ACC", {"train": train_score[-1][0], "val": val_score[-1][0]}, idx)
            writer.add_scalars("UAR", {"train": train_score[-1][1], "val": val_score[-1][1]}, idx)
            writer.add_scalars("UF1", {"train": train_score[-1][2], "val": val_score[-1][2]}, idx)
            writer.add_figure(
                "Confusion",
                confusion2fig(
                    self.train.confusions[idx],
                    self.val.confusions[idx],
                    list(str2int.keys())[:num_classes],
                ),
                idx,
            )

        writer.add_figure(
            "BestConfusion",
            confusion2fig(
                self.train.best_confusion,
                self.val.best_confusion,
                list(str2int.keys())[:num_classes],
            ),
            0,
        )
        best_acc, best_uar, best_uf1 = confusion2score(self.val.best_confusion)
        val_uf1s = [uf1 for _, _, uf1 in val_score]
        max_uf1_idx = val_uf1s.index(max(val_uf1s))
        writer.add_hparams(
            {
                "model": model,
                "imbalanced sampler": imbalanced,
                "interpolation": interpolation,
                "feature": feature,
                "lr": lr,
                "bsize": bsize,
                "imgsize": img_size[0],
            },
            {
                "train_avgBestUf1": train_score[-1][2],
                "val_bestAcc": best_acc,
                "val_bestUar": best_uar,
                "val_bestUf1": best_uf1,
                "val_avgBestAcc": val_score[max_uf1_idx][0],
                "val_avgBestUar": val_score[max_uf1_idx][1],
                "val_avgBestUf1": val_score[max_uf1_idx][2],
            },
            run_name="Hparam",
        )
        writer.close()
