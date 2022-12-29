import cv2
import torch
import pandas as pd
from typing import Union, Optional
from torchvision import datasets, transforms
from torch import Tensor


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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.ymax

        bbox = [[xmin, ymin, xmax, ymax]]
        img_path: str = self.root_dir + row.img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            data = self.transform(image=img, bboxes=bbox, class_labels=[None])
            img = data["image"]
            bbox = data["bboxes"][0]

        img: Tensor = (
            torch.from_numpy(img).permute(2, 0, 1) / 255.0
        )  # (h,w,c) -> (c,h,w)
        bbox: Tensor = torch.Tensor(bbox)

        return img, bbox
