"""

# NOTE: https://github.com/alexattia/SimpsonRecognition/tree/fa65cc3124ed606e0ad6456ae49c734a2685db52
# NOTE: https://github.com/ngduyanhece/object_localization/blob/master/label_pointer.py

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

from prompt_toolkit.completion import Completer, WordCompleter, merge_completers

# mkdir -p ~/dev/bossjones/practical-python-and-opencv-case-studies/deeplearning_data/{autogenerate,characters,models}/{captain_kobayashi,eiko_yamano,izana_shinatose,kouichi_tsuruuchi,lalah_hiyama,mozuku_kunato,nagate_tanikaze,norio_kunato,ochiai,samari_ittan,shizuka_hoshijiro,yuhata_midorikawa}/{edited,non_filtered,pic_video}

ROOT_DIR = os.path.dirname(__file__)

movies_path = "/Users/malcolm/Downloads/farming/anime/knights_of_sidonia"
dataset_folder = "/Users/malcolm/dev/bossjones/practical-python-and-opencv-case-studies/deeplearning_data"

videos_folder = f"{dataset_folder}/videos"

map_characters = {
    0: "twitter",
    1: "tiktok",
    2: "facebook"
}

characters_folder = f"{dataset_folder}/characters"

# Best size of images
IMG_SIZE = (80, 80)
# Since we don't require color in our images, set this to 1, grayscale
channels = 1

# pic_size = 64
pic_size = 80
batch_size = 32
# epochs = 200
epochs = 200
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15

path_to_best_model = f"{ROOT_DIR}/models/weights.best_6conv2.hdf5"

chars_list_from_folder = [
    pathlib.Path(f"{k}").stem for k in glob.glob(f"{characters_folder}/*")
]
chars_list_from_folder.append("no")

name_completer = WordCompleter(
    chars_list_from_folder,
    ignore_case=True,
)

yes_no_completer = WordCompleter(
    ["No", "Right", "Left", "Full", "Stop"], ignore_case=True
)
