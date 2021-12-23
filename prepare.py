import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from preprocess import Preprocess

SMIC = "/home/data/SMIC/"
# from https://github.com/cunjian/pytorch_face_landmark, MobileNetV2_ExternalData (224×224)
WEIGHT = "./preprocess/weights/mobilenet_224_model_best_gdconv_external.pth.tar"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", default=SMIC, type=str, dest="DIR", help="Dataset directory. Ex) data/SMIC"
    )
    parser.add_argument(
        "--gpu", default=0, dest="GPU", help="GPU number to use. If not available, use CPU"
    )
    parser.add_argument(
        "--ext", default="jpg", dest="EXT", help="Choose image file extention in dataset"
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.DIR)
    crop_base = data_path.with_stem(f"{data_path.stem}_CROP")
    landmark_base = data_path.with_stem(f"{data_path.stem}_LANDMARK")

    preprocess = Preprocess(device=device, ckp_dir=WEIGHT)
    box = None
    for img_path in tqdm(list(data_path.rglob(f"*.{args.EXT}"))):
        orig = cv2.imread(str(img_path))
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        crop_path = crop_base / img_path.relative_to(data_path)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        landmark_path = landmark_base / img_path.relative_to(data_path)
        landmark_path.parent.mkdir(parents=True, exist_ok=True)
        numpy_path = landmark_path.with_suffix(".npy")
        try:
            box, cropped, landmark, crp_w_lnd = preprocess(orig, box, landmark_img=True)
            cv2.imwrite(str(crop_path), cropped)
            cv2.imwrite(str(landmark_path), crp_w_lnd)
            np.save(numpy_path, landmark)
        except:
            print(f"{img_path} failed")
            break
    print("Complete")
