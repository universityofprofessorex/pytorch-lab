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
from random import shuffle
import time
import cv2
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseEvent
from matplotlib.image import AxesImage
import matplotlib.patches as patches
from icecream import install
install()

from prompt_toolkit.completion import Completer, WordCompleter, merge_completers
# from prompt_toolkit.eventloop.defaults import create_event_loop
from prompt_toolkit.eventloop.inputhook import (
    new_eventloop_with_inputhook,
    set_eventloop_with_inputhook,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession, input_dialog, prompt
import rich

# **********************************************************************************************************
# CONSTANTS - START
# **********************************************************************************************************
CLOSE_FLAG = 0
TO_CROP = []
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

KEYBINDINGS = KeyBindings()


@KEYBINDINGS.add("c-space")
def _(event):
    """
    Start auto completion. If the menu is showing already, select the next
    completion.
    """
    b = event.app.current_buffer
    if b.complete_state:
        b.complete_next()
    else:
        b.start_completion(select_first=False)



# SOURCE: https://github.com/GamestonkTerminal/GamestonkTerminal/blob/e7e49538b03e6271e1709c5229f99b5c6f4b494d/gamestonk_terminal/menu.py
def inputhook(inputhook_contex):
    while not inputhook_contex.input_is_ready():
        # print("not inputhook_contex.input_is_ready")
        try:
            # Run the GUI event loop for interval seconds.
            # If there is an active figure, it will be updated and displayed before the pause, and the GUI event loop (if any) will run during the pause.
            # This can be used for crude animation. For more complex animation use matplotlib.animation.
            # If there is no active figure, sleep for interval seconds instead.
            pyplot.pause(0.5)
            # img_axes.figure.canvas.flush_events()
        # pylint: disable=unused-variable
        except Exception:  # noqa: F841
            continue
    return False


def np_array_to_npy_file(np_array_data: np.ndarray, folder_path: str, npy_filename: str):
    """Take ndarry and save it to disk as a .npy file
    
    SEE: https://towardsdatascience.com/what-is-npy-files-and-why-you-should-use-them-603373c78883

    Args:
        np_array_data (np.ndarray): _description_
        folder_path (str): _description_
        npy_filename (str): _description_
    """
    np.save(os.path.join(folder_path,f"{npy_filename}.npy"), np_array_data)

# **********************************************************************************************************
# Matplotlib event handlers - START
# **********************************************************************************************************
def line_select_callback(eclick: MouseEvent, erelease: MouseEvent):
    # 'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata # start position
    x2, y2 = erelease.xdata, erelease.ydata  # end position
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    
    # crop_img = settings.img[x1:x2,y1:y2]
    
    # SOURCE: https://stackoverflow.com/questions/56313235/dynamic-interaction-between-rectangle-selector-and-a-matplotlib-figure
    # gcf get current figure. 
    plt.gcf().canvas.draw()


def toggle_selector(event: MouseEvent):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    if event.key == 'enter' and toggle_selector.RS.active:
        print(' Enter pressed.')
        # cnt.append(1)
        center = toggle_selector.RS.center  # xy coord, units same as plot axes
        extents = toggle_selector.RS.extents  # Return (xmin, xmax, ymin, ymax)
        # rich.inspect(toggle_selector.RS, all=True)
        ic(center)
        ic(extents)
        # rect_selection_coords = toggle_selector.RS.extents
        # ic(rect_selection_coords)
        x1, x2, y1, y2 = extents
        
        data = {
            "center": center,
            "extents": extents
        }
        
        TO_CROP[f"{fname}"] = data
        ic(TO_CROP)
        plt.close()
    if event.key == "escape" and toggle_selector.RS.active:
        print(' Escape pressed.')
        plt.close()
        
# to handle close event.
def handle_close(evt):
    global CLOSE_FLAG # should be global variable to change the outside CLOSE_FLAG.
    CLOSE_FLAG = 1
    print('Closed Figure!')

# def select_rectangle(fname: str):
#     """Return location of interactive user click on image.
#     Parameters
#     ----------
#     image : AdornedImage or 2D numpy array.
#     Returns
#     -------
#     center, extents
#           Rectangle center and extents.
#           Coordinates are in x, y format & real space units.
#           (Units are the same as the matplotlib figure axes.)
#           Rectangle extents are given as: (xmin, xmax, ymin, ymax).
#     """
    
#     img = mpim.imread(f"{fname}")
#     ic(img)
    
#     if interactive:
#         plt.ion()
    
#     # img_axes: AxesImage
#     img_axes = plt.imshow(img)  # Display data as an image, i.e., on a 2D regular raster.
#     ic(img_axes)
#     ic(type(img_axes))
#     ic(img_axes.axes)

#     toggle_selector.RS = RectangleSelector(img_axes.axes, line_select_callback,
#                            useblit=True,
#                            button=[1, 3],  # don't use middle button
#                            minspanx=5, minspany=5,
#                            spancoords='pixels',
#                            interactive=True,
#                            props=dict(facecolor='black', 
#                                               edgecolor = 'black',
#                                               alpha=1.,
#                                               fill=None))

#     plt.connect('key_press_event', toggle_selector)
#     plt.show()
#     # rich.inspect(toggle_selector.RS, all=True)
#     rect_selection_coords = toggle_selector.RS.extents
#     ic(rect_selection_coords)
#     x1, x2, y1, y2 = rect_selection_coords
    
#     fig, ax = quick_plot(image)
#     # Here are the docs fir the matplotlib RectangleSelector
#     # https://matplotlib.org/3.1.0/api/widgets_api.html#matplotlib.widgets.RectangleSelector
#     toggle_selector.RS = RectangleSelector(ax, _select_rectangle_callback,
#                                            drawtype='box', useblit=True,
#                                            # don't use middle button
#                                            button=[1, 3],
#                                            minspanx=5, minspany=5,
#                                            spancoords='pixels',
#                                            interactive=True)
#     plt.show()
#     center = toggle_selector.RS.center  # xy coord, units same as plot axes
#     extents = toggle_selector.RS.extents  # Return (xmin, xmax, ymin, ymax)
#     return center, extents


# **********************************************************************************************************
# Matplotlib event handlers - END
# **********************************************************************************************************

# SOURCE: https://github.com/bossjones/practical-python-and-opencv-case-studies/blob/main/practical_python_and_opencv_case_studies/dataset_builder/label_data.py
def labelized_data_from_images(to_shuffle=False, interactive=False):
    """
    Interactive labeling data with the possibility to crop the picture shown : full picture,
    left part, right part. Manually labeling data from .avi videos in the same folder. Analzying
    frame (randomly chosen) of each video and then save the picture into the right character
    folder.
    :param interactive: boolean to label from terminal
    """
    test_image_paths = get_image_files(f"{DATASET_FOLDER}/test")
    if to_shuffle:
        shuffle(test_image_paths)
    ic(test_image_paths[::-1])
    ic(test_image_paths[-1])
    for fname in test_image_paths[::-1]:
        try:
            while CLOSE_FLAG == 0:
                img = mpim.imread(f"{fname}")
                # ic(img)
                
                if interactive:
                    plt.ion()
                
                # img_axes: AxesImage
                img_axes = plt.imshow(img)  # Display data as an image, i.e., on a 2D regular raster.
                ic(img_axes)
                ic(type(img_axes))
                ic(img_axes.axes)
            
                toggle_selector.RS = RectangleSelector(img_axes.axes, line_select_callback,
                                       useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True,
                                       props=dict(facecolor='black', 
                                                          edgecolor = 'black',
                                                          alpha=1.,
                                                          fill=None))
                                                          
                plt.connect('key_press_event', toggle_selector)
                plt.connect('close_event', handle_close)
                plt.show()
                
                if CLOSE_FLAG == 1:
                    break
            
            
        except Exception as e:
            if e == KeyboardInterrupt:
                return
            else:
                continue

        # try:
        #     # NOTE: IMREAD_UNCHANGED - If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
        #     img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # cv2.cvtColor() method is used to convert an image from one color space to another.
        #     while True:
        #         if interactive:
        #             plt.ion()
                
        #         img_axes: AxesImage
        #         img_axes = plt.imshow(img)  # Display data as an image, i.e., on a 2D regular raster.
        #         ic(img_axes)
        #         # img_axes.figure.canvas.flush_events()
        #         # plt.show()
        #         # where = prompt(
        #         #     message="Where is the bounding box ? Please type one of the following [No,Right,Left,Full] :",
        #         #     completer=constants.yes_no_completer,
        #         #     complete_while_typing=True,
        #         #     key_bindings=kb,
        #         # )
                
        #         # drawtype is 'box' or 'line' or 'none'
        #         toggle_selector.RS = RectangleSelector(img_axes, line_select_callback,
        #                                drawtype='box', useblit=True,
        #                                button=[1, 3],  # don't use middle button
        #                                minspanx=5, minspany=5,
        #                                spancoords='pixels',
        #                                interactive=True,
        #                                rectprops=dict(facecolor='black', 
        #                                                   edgecolor = 'black',
        #                                                   alpha=1.,
        #                                                   fill=None))
                                                          
        #         plt.connect('key_press_event', toggle_selector)
        #         plt.show()
                
        # except Exception as e:
        #     if e == KeyboardInterrupt:
        #         return
        #     else:
        #         continue
                
    # plt.imshow(img)
    #         m, s = np.random.randint(0, 3), np.random.randint(0, 59)
    #         print(f"fname = {fname}")
    #         cap = cv2.VideoCapture(fname)  # video_name is the video being called
    #         fps = cap.get(cv2.CAP_PROP_FPS)
    #         cap.set(1, fps * (m * 60 + s))  # Where frame_no is the frame you want
    #         i = 0
    #         while True:
    #             i += 1
    #             ret, frame = cap.read()  # Read the frame
    #             # Resizing HD pictures (we don't need HD)
    #             if np.min(frame.shape[:2]) > 900:
    #                 frame = cv2.resize(
    #                     frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
    #                 )
    #             if i % np.random.randint(100, 250) == 0:
    #                 if interactive:
    #                     plt.ion()
    #                 img_axes = plt.imshow(frame)
    #                 img_axes.figure.canvas.flush_events()
    #                 plt.show()
    #                 where = prompt(
    #                     message="Where is the character ? Please type one of the following [No,Right,Left,Full] :",
    #                     completer=constants.yes_no_completer,
    #                     complete_while_typing=True,
    #                     key_bindings=kb,
    #                 )

    #                 if where.lower() == "stop":
    #                     raise

    #                 elif where.lower() in ["left", "l"]:
    #                     plt.close()
    #                     img_axes = plt.imshow(frame[:, : int(frame.shape[1] / 2)])
    #                     img_axes.figure.canvas.flush_events()
    #                     plt.show()

    #                     name = prompt(
    #                         message="Name ? Please type one of the following [Name or No] :",
    #                         completer=constants.name_completer,
    #                         complete_while_typing=True,
    #                         key_bindings=kb,
    #                         # complete_in_thread=True
    #                     )
    #                     plt.close()
    #                     if name.lower() not in ["no", "n", ""]:
    #                         name_char = get_character_name(name)
    #                         name_new_pic = "pic_{:04d}.jpg".format(
    #                             len(
    #                                 glob.glob(
    #                                     f"{constants.characters_folder}/%s/*"
    #                                     % name_char
    #                                 )
    #                             )
    #                         )
    #                         title = f"{constants.characters_folder}/%s/%s" % (
    #                             name_char,
    #                             name_new_pic,
    #                         )
    #                         cv2.imwrite(title, frame[:, : int(frame.shape[1] / 2)])
    #                         print("Saved at %s" % title)
    #                         print(
    #                             "%s : %d photos labeled"
    #                             % (
    #                                 name_char,
    #                                 len(
    #                                     glob.glob(
    #                                         f"{constants.characters_folder}/%s/*"
    #                                         % name_char
    #                                     )
    #                                 ),
    #                             )
    #                         )

    #                 elif where.lower() in ["right", "r"]:
    #                     plt.close()
    #                     img_axes = plt.imshow(frame[:, int(frame.shape[1] / 2) :])
    #                     img_axes.figure.canvas.flush_events()
    #                     plt.show()
    #                     name = prompt(
    #                         message="Name ? Please type one of the following [Name or No] :",
    #                         completer=constants.name_completer,
    #                         complete_while_typing=True,
    #                         key_bindings=kb,
    #                         # complete_in_thread=True
    #                     )
    #                     plt.close()
    #                     if name.lower() not in ["no", "n", ""]:
    #                         name_char = get_character_name(name)
    #                         name_new_pic = "pic_{:04d}.jpg".format(
    #                             len(
    #                                 glob.glob(
    #                                     f"{constants.characters_folder}/%s/*"
    #                                     % name_char
    #                                 )
    #                             )
    #                         )
    #                         title = f"{constants.characters_folder}/%s/%s" % (
    #                             name_char,
    #                             name_new_pic,
    #                         )
    #                         cv2.imwrite(title, frame[:, int(frame.shape[1] / 2) :])
    #                         print("Saved at %s" % title)
    #                         print(
    #                             "%s : %d photos labeled"
    #                             % (
    #                                 name_char,
    #                                 len(
    #                                     glob.glob(
    #                                         f"{constants.characters_folder}/%s/*"
    #                                         % name_char
    #                                     )
    #                                 ),
    #                             )
    #                         )

    #                 elif where.lower() in ["full", "f"]:

    #                     name = prompt(
    #                         message="Name ? Please type one of the following [Name or No] :",
    #                         completer=constants.name_completer,
    #                         complete_while_typing=True,
    #                         key_bindings=kb,
    #                         # complete_in_thread=True
    #                     )
    #                     plt.close()
    #                     if name.lower() not in ["no", "n", ""]:
    #                         name_char = get_character_name(name)
    #                         name_new_pic = "pic_{:04d}.jpg".format(
    #                             len(
    #                                 glob.glob(
    #                                     f"{constants.characters_folder}/%s/*"
    #                                     % name_char
    #                                 )
    #                             )
    #                         )
    #                         title = f"{constants.characters_folder}/%s/%s" % (
    #                             name_char,
    #                             name_new_pic,
    #                         )
    #                         cv2.imwrite(title, frame)
    #                         print("Saved at %s" % title)
    #                         print(
    #                             "%s : %d photos labeled"
    #                             % (
    #                                 name_char,
    #                                 len(
    #                                     glob.glob(
    #                                         f"{constants.characters_folder}/%s/*"
    #                                         % name_char
    #                                     )
    #                                 ),
    #                             )
    #                         )
    #     except Exception as e:
    #         if e == KeyboardInterrupt:
    #             return
    #         else:
    #             continue



# test_image_paths = get_image_files(f"{DATASET_FOLDER}/test")
# ic(test_image_paths)

labelized_data_from_images()

# # https://gist.github.com/GenevieveBuckley/aa46f72cb64637ae2a9d8c7d88aac588
# def main():
#     import fibsem
#     rect_coords = select_rectangle(fibsem.data.embryo_adorned())
#     print(rect_coords)


# if __name__ == '__main__':
#     main()