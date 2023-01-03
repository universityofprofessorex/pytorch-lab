from __future__ import annotations

import sys

sys.path.append("../")
import cv2
import torch
import pandas as pd
from typing import Union, Optional
from torchvision import datasets, transforms
from torch import Tensor
from icecream import ic
import numpy as np

from ml_types import ImageNdarrayBGR, ImageNdarrayHWC, TensorCHW


class ObjLocDataset(torch.utils.data.Dataset):

    """Localization Dataset."""

    def __init__(self, df: pd.DataFrame, transform=None, root_dir: str = ""):
        """
        Args:
            df (pd.DataFrame): dataframe containing infromation about images and their bounding boxes
            transform (callable, optional): Optional transform to be applied
                on a sample.. Defaults to None.
            root_dir (str, optional): Directory with all the images.. Defaults to "".
        """
        self.df = df
        self.transform = transform

        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[TensorCHW, torch.Tensor]:
        row = self.df.iloc[idx]

        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.ymax

        bbox = [[xmin, ymin, xmax, ymax]]
        img_path: str = self.root_dir + row.img_path
        img = cv2.imread(img_path)
        img: ImageNdarrayHWC
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            # print(f"[ObjLocDataset] Applying transform {self.transform} on img_path -> {img_path}")
            # import bpdb
            # NOTE: img.shape before any transforms etc (2532, 1170, 3) type -> <class 'numpy.ndarray'>

            data = self.transform(image=img, bboxes=bbox, class_labels=[None])
            img: ImageNdarrayHWC
            img = data["image"]
            bbox = data["bboxes"][0]
            # NOTE: AFTER transform img.shape is (140, 140, 3) type -> <class 'numpy.ndarray'>
            # -----------------------------------------------
            # Example array data:
            # -----------------------------------------------
            # {'image': array([[[23, 31, 42],
            # [23, 31, 42],
            # [23, 31, 42],
            # ...,
            # [23, 31, 42],
            # [23, 31, 42],
            # [23, 31, 42]],
            # .......

            # ic(img.shape)
            # ic(bbox)

        # bpdb.set_trace()
        img_tensor: TensorCHW = (
            torch.from_numpy(img).permute(2, 0, 1) / 255.0
        )  # (h,w,c) -> (c,h,w)
        # NOTE: after transform to tensor, looks like this
        # (BPdb) img.shape
        # torch.Size([3, 140, 140])
        # (BPdb)
        # -----------------------------------------------
        # Example array data:
        # -----------------------------------------------
        # [[0.1216, 0.1216, 0.1216,  ..., 0.1216, 0.1216, 0.1216],
        # [0.1216, 0.1216, 0.1216,  ..., 0.1216, 0.1216, 0.1216],
        # [0.1216, 0.1216, 0.1216,  ..., 0.1216, 0.1216, 0.1216],
        # ...,
        # [0.1216, 0.1216, 0.1216,  ..., 0.1216, 0.1216, 0.1216],
        # [0.1216, 0.1216, 0.1216,  ..., 0.1216, 0.1216, 0.1216],
        # [0.1216, 0.1216, 0.1216,  ..., 0.1216, 0.1216, 0.1216]],

        # print(f"ObjLocDataset converted to tensor and normalized img_path -> {img_path}")
        # NOTE: Before converting to tensor, bbox looks like the following:
        # (BPdb) bbox
        # (4.786324786324785, 40.75039494470775, 140.0, 123.5781990521327)
        # (BPdb) type(bbox)
        # <class 'tuple'>
        bbox_tensor: Tensor = torch.Tensor(bbox)

        return img_tensor, bbox_tensor
