import torch
import torch.nn as nn
from einops import rearrange

from .create_model import create_model, register_model


class Fusion(nn.Module):
    """Some Information about Fusion"""

    def __init__(self, backbone1, backbone2, num_classes=3):
        super(Fusion, self).__init__()
        self.backbone1 = backbone1
        self.backbone1.reset_classifier(num_classes=0)
        self.backbone2 = backbone2
        self.backbone2.reset_classifier(num_classes=0)
        self.linear_in_features = (
            self.backbone1.linear_in_features + self.backbone2.linear_in_features
        )
        self.classifier = nn.Linear(self.linear_in_features, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        of, orig = torch.split(x, 3, dim=2)
        feature1 = self.backbone1(of)
        feature2 = self.backbone2(orig)
        if feature1.dim() == 5:
            feature1 = rearrange(self.avg_pool(feature1), "B C 1 1 1 -> B C")
            feature2 = rearrange(self.avg_pool(feature2), "B C 1 1 1 -> B C")
        feature = torch.cat([feature1, feature2], axis=1)
        out = self.classifier(feature)
        return out

    def load_checkpoint(self, path=""):
        checkpoint = torch.load(path)
        backbone = {k: v for k, v in checkpoint["state_dict"].items() if "backbone" in k}
        self.load_state_dict(backbone, strict=False)


def fusion(backbone_type, ch1, ch2, **kwargs):
    assert backbone_type in [
        "cnn",
        "cnn-like",
        "transformer",
    ], "backbone_type must be one of cnn, cnn-like, transformer"
    if backbone_type == "cnn":
        backbone1 = create_model(f"ch{ch1}_resnext", **kwargs)
        backbone2 = create_model(f"ch{ch2}_resnext", **kwargs)
    elif backbone_type == "cnn-like":
        backbone1 = create_model(f"ch{ch1}_swin", **kwargs)
        backbone2 = create_model(f"ch{ch2}_swin", **kwargs)
    else:
        backbone1 = create_model(f"ch{ch1}_timesformer", **kwargs)
        backbone2 = create_model(f"ch{ch2}_timesformer", **kwargs)
    return Fusion(backbone1, backbone2, num_classes=kwargs["num_classes"])


@register_model
def ch3_1_fusion_cnn(**kwargs):
    return fusion("cnn", 3, 1, **kwargs)


@register_model
def ch3_1_fusion_cnn_like(**kwargs):
    return fusion("cnn-like", 3, 1, **kwargs)


@register_model
def ch3_1_fusion_transformer(**kwargs):
    return fusion("transformer", 3, 1, **kwargs)


@register_model
def ch3_3_fusion_cnn(**kwargs):
    return fusion("cnn", 3, 3, **kwargs)


@register_model
def ch3_3_fusion_cnn_like(**kwargs):
    return fusion("cnn-like", 3, 3, **kwargs)


@register_model
def ch3_3_fusion_transformer(**kwargs):
    return fusion("transformer", 3, 3, **kwargs)
