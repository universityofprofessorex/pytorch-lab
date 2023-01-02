import cv2
import numpy as np
import torch


def opencv_read_and_convert_image(path: str, cvt=cv2.COLOR_BGR2RGB) -> np.ndarray:
    img = cv2.imread(f"{path}")
    #  convert color image into RGB image
    img = cv2.cvtColor(img, cvt)
    return img


def convert_image_numpy_array_to_tensor(img: np.ndarray):
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    return img


if __name__ == "__main__":
    path = "/Users/malcolm/Downloads/IMG_6400.PNG"

    image = opencv_read_and_convert_image(path)

    # https://www.dropbox.com/s/rzkwy02hz2j3ath/screencropnet_best_model.pt?dl=1
    # https://www.dropbox.com/s/ot2ijn5u5d5v93n/test_crop_twitter_IMG_6400.PNG?dl=1
