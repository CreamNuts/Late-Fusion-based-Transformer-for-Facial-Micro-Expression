import cv2
import numpy as np
import torch
from einops import asnumpy
from facenet_pytorch import MTCNN

from preprocess.model import MobileNet_GDConv

mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])


def drawLandmark(img, landmark):
    """
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    """
    img_ = img.copy()
    for x, y in landmark:
        cv2.circle(img_, (int(x), int(y)), 3, (0, 255, 0), -1)
    return img_


class BBox:
    def __init__(self, bbox, width, height) -> None:
        x1, y1, x2, y2 = bbox
        self.orig_left = int(x1)
        self.orig_right = int(x2)
        self.orig_top = int(y1)
        self.orig_bottom = int(y2)
        w = x2 - x1
        h = y2 - y1
        size = int(min([w, h]) * 1.2)
        cx = x1 + w / 2
        cy = y1 + h / 2
        x1 = cx - size / 2
        x2 = x1 + size
        y1 = cy - size / 2
        y2 = y1 + size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        self.left = new_bbox[0]
        self.right = new_bbox[1]
        self.top = new_bbox[2]
        self.bottom = new_bbox[3]
        self.w = new_bbox[1] - new_bbox[0]
        self.h = new_bbox[3] - new_bbox[2]

    def reprojectLandmark(self, landmark):
        landmark_ = np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.left
            y = point[1] * self.h + self.top
            landmark_[i] = (x, y)
        return landmark_


class Preprocess:
    def __init__(
        self,
        device: torch.device,
        ckp_dir: str,
    ) -> None:
        self.detector = MTCNN(device=device)
        self.detector.eval()
        self.classifier = torch.nn.DataParallel(MobileNet_GDConv(136), device_ids=[device.index])
        checkpoint = torch.load(ckp_dir)
        self.classifier.load_state_dict(checkpoint["state_dict"])
        self.classifier.to(device)
        self.classifier.eval()
        self.device = device

    def __call__(self, img, landmark_img=False):
        height, width, _ = img.shape
        box = self.detect(img)
        bbox = BBox(box[0], width, height)
        cropped = img[bbox.orig_top : bbox.orig_bottom, bbox.orig_left : bbox.orig_right]
        landmark = img[bbox.top : bbox.bottom, bbox.left : bbox.right]
        landmark = self.landmark(landmark)
        if landmark_img:
            landmark_ = bbox.reprojectLandmark(landmark)
            crp_w_lnd = drawLandmark(img, landmark_)
            return cropped, landmark, crp_w_lnd
        return cropped, landmark

    def detect(self, img):
        bbox_list, _ = self.detector.detect(img)
        return bbox_list

    def landmark(self, img):
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = (img - mean) / std
        img = img.transpose((2, 0, 1))
        img = img.reshape((1,) + img.shape)
        img = torch.from_numpy(img).float().to(self.device)
        landmark = asnumpy(self.classifier(img))
        landmark = landmark.reshape(-1, 2)
        return landmark
