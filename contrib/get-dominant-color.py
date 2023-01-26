# NOTE: For more examples tqdm + aiofile, search https://github.com/search?l=Python&q=aiofile+tqdm&type=Code

from __future__ import annotations

import concurrent.futures

import asyncio
import time
# import aiohttp
import os
import errno
from hashlib import md5
import os
import shutil
import tempfile
import ssl
import certifi
import rich
import uritools

# import aiofile
import pathlib
import functools
import gc

# import aiorwlock
import requests
from tqdm.auto import tqdm
from icecream import ic
import argparse
from typing import List, Union, Optional, Tuple

# import aiosqlite
from datetime import datetime
from dateutil.parser import parse as parse_date
import mimetypes
import os
import re
from subprocess import Popen, PIPE
import imghdr
import os
import shutil
from hashlib import md5
from io import StringIO

from PIL import Image

# from tqdm.asyncio import tqdm
from urllib.request import urlretrieve
import pytz


import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd

utc = pytz.utc

parser = argparse.ArgumentParser(description="extract dominant color")
parser.add_argument("-u", "--urls", metavar="URL", nargs="*", help="urls to download. ")

def get_all_corners_color(urls):
    pbar = tqdm(urls)
    pixels = []
    pixel_data = {"top_left": "", "top_right": "", "bottom_left": "", "bottom_right": ""}
    for url in pbar:
        img = Image.open(url).convert("RGB")
        # breakpoint()
        width, height = img.size
        pixel_layout = img.load()
        pixel_data["top_left"] = top_left = pixel_layout[0,0]
        pixel_data["top_right"] = top_right = pixel_layout[width-1, 0]
        pixel_data["bottom_left"] = bottom_left = pixel_layout[0, height-1]
        pixel_data["bottom_right"] = bottom_right = pixel_layout[width-1, height-1]
        rich.print(pixel_data)
    return pixel_data

def main():
    args = parser.parse_args()
    print()
    ic(args)
    print()
    return args

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)


if __name__ == "__main__":
    start_time = time.time()
    # /Users/malcolm/Downloads/chrome_downloader/pinterest_memes/This-Could-Be-Us-But-You-Playin-Meme.jpg
    args = main()


    corner_pixels = get_all_corners_color(args.urls)

    if corner_pixels["top_left"] == corner_pixels["top_right"] == corner_pixels["bottom_left"] == corner_pixels["bottom_right"]:
        r, g, b = corner_pixels["top_left"]
        background_color = rgb2hex(r, g, b)
        ic(background_color)
    else:
        r, g, b = corner_pixels["top_right"]
        background_color = rgb2hex(r, g, b)
        ic(background_color)


    duration = time.time() - start_time
    print(f"Calculated 1 image in {duration} seconds")
