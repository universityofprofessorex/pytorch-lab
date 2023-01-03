from types import NoneType
import cv2
import numpy as np
import torch
import albumentations as A
from typing import Optional, Union


def opencv_read_and_convert_image(path: str, cvt=cv2.COLOR_BGR2RGB) -> np.ndarray:
    img = cv2.imread(f"{path}")
    #  convert color image into RGB image
    img = cv2.cvtColor(img, cvt)
    return img


def convert_image_numpy_array_to_tensor(img: np.ndarray):
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    return img

def safe_read_image(path: str):
    img = opencv_read_and_convert_image(path)
    img = convert_image_numpy_array_to_tensor(img)
    return img

def load_and_transform_image_for_prediction(path: str, transform: Union[A.Compose, NoneType] = None, img_size = 140):
    img = safe_read_image(path)

    if transform is not None:
        img_transform: A.Compose = A.Compose(
            [A.Resize(img_size, img_size)],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels"]
            ),
        )

    return img, img_transform


if __name__ == "__main__":
    path = "/Users/malcolm/Downloads/IMG_6400.PNG"

    image = opencv_read_and_convert_image(path)

    # https://www.dropbox.com/s/rzkwy02hz2j3ath/screencropnet_best_model.pt?dl=1
    # https://www.dropbox.com/s/ot2ijn5u5d5v93n/test_crop_twitter_IMG_6400.PNG?dl=1
