#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

# NOTE: https://github.com/alexattia/SimpsonRecognition/tree/fa65cc3124ed606e0ad6456ae49c734a2685db52
# NOTE: https://github.com/ngduyanhece/object_localization/blob/master/label_pointer.py
# NOTE: https://github.com/thtang/CheXNet-with-localization/blob/master/preprocessing.py
# 

Files description
label_data.py : tools functions for notebooks + script to name characters from frames from .avi videos
label_pointer.py : point with mouse clicks to save bounding box coordinates on annotations text file (from already labeled pictures)
train.py : training simple convnet
train_frcnn.py -p annotation.txt : training Faster R-CNN with data from the annotation text file
test_frcnn.py -p path/test_data/ : testing Faster R-CNN


HOW TO RUN:

./preprocessing_data_loader.py [path of images folder] [path to data_entry] [path to bbox_list_path] [path to train_txt] [path to valid_txt] [path of preprocessed output (folder)]
"""

import glob
import os
import pathlib
import numpy as np
from fastai.vision.all import *
# from icecream import ic
from icecream import install
install()

from prompt_toolkit.completion import Completer, WordCompleter, merge_completers

# **********************************************************************************************************
# CONSTANTS - START
# **********************************************************************************************************
ROOT_DIR = os.path.dirname(__file__)

DATASET_FOLDER = f"{ROOT_DIR}/demo/datasets/twitter_facebook_tiktok_screenshots"
TRAIN_DIR = f"{DATASET_FOLDER}/train"
TEST_DIR = f"{DATASET_FOLDER}/test"

# videos_folder = f"{DATASET_FOLDER}/videos"

MAP_LABELS = {
    0: "facebook",
    1: "tiktok",
    2: "twitter"
}

LABELS_FOLDER = TEST_DIR

# Best size of images
IMG_SIZE = (80, 80)
# Since we don't require color in our images, set this to 1, grayscale
CHANNELS = 1

PIC_SIZE = 80
BATCH_SIZE = 32
EPOCHS = 200
NUM_CLASSES = len(MAP_LABELS)
PICTURES_PER_CLASS = 1000
TEST_SIZE = 0.15

LABEL_LIST_FROM_FOLDER = [
    pathlib.Path(f"{k}").stem for k in glob.glob(f"{LABELS_FOLDER}/*")
]
LABEL_LIST_FROM_FOLDER.append("no")

NAME_COMPLETER = WordCompleter(
    LABEL_LIST_FROM_FOLDER,
    ignore_case=True,
)

YES_NO_COMPLETER = WordCompleter(
    ["No", "Right", "Left", "Full", "Stop"], ignore_case=True
)
# **********************************************************************************************************
# CONSTANTS - END
# **********************************************************************************************************

def np_array_to_npy_file(np_array_data: np.ndarray, folder_path: str, npy_filename: str):
    """Take ndarry and save it to disk as a .npy file
    
    SEE: https://towardsdatascience.com/what-is-npy-files-and-why-you-should-use-them-603373c78883

    Args:
        np_array_data (np.ndarray): _description_
        folder_path (str): _description_
        npy_filename (str): _description_
    """
    np.save(os.path.join(folder_path,f"{npy_filename}.npy"), np_array_data)


test_image_paths = get_image_files(f"{DATASET_FOLDER}/test")
ic(test_image_paths)