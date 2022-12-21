#!/usr/bin/env python

# import bpdb; bpdb.set_trace()
# import pdbr
# import pdb
# pdb = pdb.pdb

import os
import os.path
import pathlib
import pandas as pd
# from rich_dataframe import prettify

# ---------------------------------------------------------------------------
# Import rich and whatever else we need
# %load_ext rich
# %matplotlib inline
import sys

import bpdb
import platform

extra_modules_path_api = pathlib.Path("../going_modular")
extra_modules_path = os.path.abspath(str(extra_modules_path_api))
# print(extra_modules_path)

# sys.path.insert(1, extra_modules_path)
sys.path.append(extra_modules_path)
sys.path.append("../")
# import better_exceptions
import better_exceptions

# from rich.traceback import install
# install(show_locals=True)
from icecream import ic
import rich
from rich import inspect, print
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
import torch
import torchvision
from torchvision import datasets, transforms

import devices  # pylint: disable=import-error

better_exceptions.hook()

console = Console()
# ---------------------------------------------------------------------------


assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert (
    int(torchvision.__version__.split(".")[1]) >= 13
), "torchvision version should be 0.13+"
# print(f"torch version: {torch.__version__}")
# print(f"torchvision version: {torchvision.__version__}")
# ---------------------------------------------------------------------------

# breakpoint()
from going_modular import data_setup, engine  # pylint: disable=no-name-in-module

# Continue with regular imports
import matplotlib.pyplot as plt
import mlxtend
import torch
from torch import nn
from torchinfo import summary
import torchmetrics
import torchvision
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work


# print(f"mlxtend version: {mlxtend.__version__}")
assert (
    int(mlxtend.__version__.split(".")[1]) >= 19
), "mlxtend verison should be 0.19.0 or higher"

import argparse
from enum import Enum
from itertools import product
import os
from pathlib import Path
import random
import shutil
from timeit import default_timer as timer
from urllib.parse import urlparse
import warnings
import zipfile

from typing import List, Tuple, Optional, Union

from PIL import Image

# Import accuracy metric
from helper_functions import (
    accuracy_fn,
    plot_loss_curves,
)  # Note: could also use torchmetrics.Accuracy()
import matplotlib
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import numpy.typing as npt
import requests

# SOURCE: https://github.com/rasbt/deeplearning-models/blob/35aba5dc03c43bc29af5304ac248fc956e1361bf/pytorch_ipynb/helper_evaluate.py
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import Subset
import torch.utils.data.distributed
from torchmetrics import ConfusionMatrix
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from watermark import watermark


def download_and_predict(
    url: str,
    model: torch.nn.Module,
    data_path: pathlib.PosixPath,
    class_names: List[str],
    device: torch.device = None,
):
    # Download custom image
    urlparse(url).path
    fname = Path(urlparse(url).path).name

    # Setup custom image path
    custom_image_path = data_path / fname

    print(f"fname: {custom_image_path}")

    # Download the image if it doesn't already exist
    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get(url)
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")

    # Predict on custom image
    pred_and_plot_image(
        model=model,
        image_path=custom_image_path,
        class_names=class_names,
        device=device,
    )


def show_confusion_matrix_helper(
    cmat: np.ndarray,
    class_names: List[str],
    to_disk: bool = True,
    fname: str = "plot.png",
):
    # boss: function via https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb#scrollTo=7aed6d76-ad1c-429e-b8e0-c80572e3ebf4
    fig, ax = plot_confusion_matrix(
        conf_mat=cmat,
        class_names=class_names,
        norm_colormap=matplotlib.colors.LogNorm()
        # normed colormaps highlight the off-diagonals
        # for high-accuracy models better
    )

    if to_disk:
        ic("Writing confusion matrix to disk ...")
        ic(plt.savefig(fname))
    else:
        plt.show()


def compute_accuracy(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: str
):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_epoch_loss(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: str
):
    model.eval()
    curr_loss, num_examples = 0.0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = F.cross_entropy(logits, targets, reduction="sum")
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def compute_confusion_matrix(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device
):

    all_targets, all_predictions = [], []
    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to("cpu"))
            all_predictions.extend(predicted_labels.to("cpu"))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_targets, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat


def run_confusion_matrix(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
):

    cmat = compute_confusion_matrix(model, test_dataloader, device)

    # cmat, type(cmat)

    show_confusion_matrix_helper(cmat, class_names)


def run_validate(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
):
    print(" Running in evaluate mode ...")

    start_time = timer()
    # Setup testing and save the results
    test_loss, test_acc = engine.test_step(
        model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total testing time: {end_time-start_time:.3f} seconds")
    ic(test_loss)
    ic(test_acc)


def run_train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    batch_size: int
):
    print("No other options selected so we are training this model....")
    # ic(model)
    ic(train_dataloader)
    ic(test_dataloader)
    ic(loss_fn)
    ic(optimizer)
    ic(epochs)
    ic(device)

    start_time = timer()
    # Setup training and save the results
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
    )

    # End the timer and print out how long it took
    end_time = timer()
    # print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    # Print out timer and results
    total_train_time = print_train_time(start=start_time,
                                        end=end_time,
                                        device=device,
                                        machine="silicontop")

    dataset_name="twitter_facebook_tiktok"

    write_results_to_csv("silicontop", device, dataset_name=dataset_name, num_epochs=epochs, batch_size=batch_size, image_size=(224, 224), train_data=train_dataloader.dataset, test_data=test_dataloader.dataset, total_train_time=total_train_time, model=model)

    results_df = inspect_csv_results()
    ic("Plot performance benchmarks")
    # Get names of devices
    machine_and_device_list = [row[1][0] + " (" + row[1][1] + ")" for row in results_df[["machine", "device"]].iterrows()]

    # Plot and save figure
    plt.figure(figsize=(10, 7))
    plt.style.use('fivethirtyeight')
    plt.bar(machine_and_device_list, height=results_df.time_per_epoch)
    plt.title(f"PyTorch ScreenNetV1 Training on {dataset_name} with batch size {batch_size} and image size {(224, 224)}", size=16)
    plt.xlabel("Machine (device)", size=14)
    plt.ylabel("Seconds per epoch (lower is better)", size=14);
    save_path = f"results/{model.__class__.__name__}_{dataset_name}_benchmark_with_batch_size_{batch_size}_image_size_{(224, 224)[0]}.png"
    print(f"Saving figure to '{save_path}'")
    plt.savefig(save_path)

    ic("Plot the loss curves of our model")
    plot_loss_curves(results, to_disk=True)

# SOURCE: https://github.com/mrdbourke/pytorch-apple-silicon/blob/main/01_cifar10_tinyvgg.ipynb
def write_results_to_csv(MACHINE, device, dataset_name="", num_epochs="", batch_size="", image_size="", train_data="", test_data="", total_train_time="", model=""):
    # Create results dict
    results = {
    "machine": MACHINE,
    "device": device,
    "dataset_name": dataset_name,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "image_size": image_size[0],
    "num_train_samples": len(train_data),
    "num_test_samples": len(test_data),
    "total_train_time": round(total_train_time, 3),
    "time_per_epoch": round(total_train_time/num_epochs, 3),
    "model": model.__class__.__name__
    }

    results_df = pd.DataFrame(results, index=[0])

    # Write CSV to file
    if not os.path.exists("results/"):
        os.makedirs("results/")

    results_df.to_csv(f"results/{MACHINE.lower().replace(' ', '_')}_{device}_{dataset_name}_image_size.csv",
                      index=False)

def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table

def inspect_csv_results():
    results_paths = list(Path("results").glob("*.csv"))

    df_list = []
    for path in results_paths:
        df_list.append(pd.read_csv(path))
    results_df = pd.concat(df_list).reset_index(drop=True)
    # prettify(results_df)

    # Initiate a Table instance to be modified
    table = Table(show_header=True, header_style="bold magenta")

    # Modify the table instance to have the data from the DataFrame
    table = df_to_table(results_df, table)

    # Update the style of the table
    table.row_styles = ["none", "dim"]
    table.box = box.SIMPLE_HEAD

    console.print(table)
    return results_df

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


def clean_dir_images(image_path):
    for f in Path(image_path).glob("*.jpg"):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def clean_dirs_in_dir(image_path):
    try:
        shutil.rmtree(image_path)
    except OSError as e:
        print("Error: %s : %s" % (image_path, e.strerror))


def setup_workspace(data_path: pathlib.PosixPath, image_path: pathlib.PosixPath):

    # Setup path to data folder
    # data_path = Path("data/")
    # image_path = data_path / "twitter_facebook_tiktok"

    # NOTE: Use this if you need to delete folders again
    # clean_dir_images(image_path)
    # clean_dirs_in_dir(image_path)
    # os.rmdir(image_path)
    # os.unlink("data/twitter_facebook_tiktok.zip")

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download twitter, facebook, tiktok data
        with open(data_path / "twitter_facebook_tiktok.zip", "wb") as f:
            request = requests.get(
                "https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1"
            )
            print("Downloading twitter, facebook, tiktok data...")
            f.write(request.content)

        # Unzip twitter, facebook, tiktok data
        with zipfile.ZipFile(data_path / "twitter_facebook_tiktok.zip", "r") as zip_ref:
            print("Unzipping twitter, facebook, tiktok data...")
            zip_ref.extractall(image_path)


# boss: use this to instantiate a new model class with all the proper setup as before
def setup_efficientnet_model(device: str, class_names: List[str]) -> torch.nn.Module:
    """Create an instance of pretrained model EfficientNet_B0, freeze all base layers and define classifier. Return model class

    Args:
        device (str): _description_
        class_names (List[str]): _description_

    Returns:
        _type_: _description_
    """
    # NEW: Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
    weights = (
        torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )  # .DEFAULT = best available weights
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(device)

    return model


def get_model_summary(
    model: torch.nn.Module,
    input_size: tuple = (32, 3, 224, 224),
    verbose: int = 0,
    col_names: List[str] = ["input_size", "output_size", "num_params", "trainable"],
    col_width: int = 20,
    row_settings: List[str] = ["var_names"],
):
    print(f"Getting model summary for -> {model}")
    # # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
    summary(
        model,
        input_size=input_size,  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=verbose,
        col_names=col_names,
        col_width=col_width,
        row_settings=row_settings,
    )


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
# print(model_names)

shared_datasets_path_api = pathlib.Path(os.path.expanduser("~/Downloads/datasets"))
shared_datasets_path = os.path.abspath(str(shared_datasets_path_api))
# print(f"shared_datasets_path - {shared_datasets_path}")
DEFAULT_DATASET_DIR = Path(f"{shared_datasets_path}")

# --------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="PyTorch ScreenNet Training")
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default=f"{DEFAULT_DATASET_DIR}",
    help=f"path to dataset (default: {DEFAULT_DATASET_DIR})",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="efficientnet_b0",
    choices=model_names,
    help="model architecture: "
    + " | ".join(model_names)
    + " (default: 'efficientnet_b0)",
)
parser.add_argument(
    "--model-weights",
    metavar="Model Weights",
    default="EfficientNet_B0_Weights",
    choices=model_names,
    help="model weight: "
    + " | ".join(model_names)
    + " (default: 'EfficientNet_B0_Weights)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=5, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 32), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)

parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--predict",
    default="",
    type=str,
    metavar="PREDICT_PATH",
    help="path to image to run prediction on (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--info",
    dest="info",
    action="store_true",
    help="info about this build",
)
parser.add_argument(
    "--download-and-predict",
    default="",
    type=str,
    metavar="DOWNLOAD_PREDICT_PATH",
    help="url to image to run prediction on (default: none)",
)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    action="store_true",
    default=True,
    help="use pre-trained model",
)
parser.add_argument(
    "--summary",
    dest="summary",
    action="store_true",
    help="Get model summary output",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument("--dummy", action="store_true", help="use fake data to benchmark")

best_acc1 = 0


def main():
    args = parser.parse_args()
    ic(args)
    # rich.inspect(args)

    if args.seed is not None:
        validate_seed(args.seed)
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        # cudnn.benchmark = False
        # warnings.warn(
        #     "You have chosen to seed training. "
        #     "This will turn on the CUDNN deterministic setting, "
        #     "which can slow down your training considerably! "
        #     "You may see unexpected behavior when restarting "
        #     "from checkpoints."
        # )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    global best_acc1
    args.gpu = gpu

    y_preds = []
    y_pred_tensor = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    if args.pretrained:
        ic("=> using pre-trained model '{}'".format(args.arch))
        # breakpoint()
        device = devices.get_optimal_device(args)
        # models.__dict__[args.model_weights].DEFAULT = device
        weights = models.__dict__[args.model_weights].DEFAULT
        auto_transforms = weights.transforms()
        model = models.__dict__[args.arch](weights=weights).to(device)
    else:
        ic("=> creating model '{}'".format(args.arch))
        # breakpoint()
        # weights = models.__dict__[args.model_weights].DEFAULT.to(device)
        # auto_transforms = weights.transforms()
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        ic("using CPU, this will be slow")
    elif args.distributed:
        ic("distributed mode enabled")
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu]
                )
                ic(f"Using GPU devices with DistributedDataParallel {[args.gpu]}")
            else:
                ic("Attempting to use single gpu device with DistributedDataParallel")
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        ic("GPU enabled")
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        ic("MPS mode enabled")
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            ic(f"Using alexnet or vgg -> {args.arch}")
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device("cuda:{}".format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # BOSSNEW
    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(
            1281167, (3, 224, 224), 1000, transforms.ToTensor()
        )
        val_dataset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor()
        )
    else:

        # Setup path to data folder
        data_path = Path(args.data)
        image_path = data_path / "twitter_facebook_tiktok"
        train_dir = image_path / "train"
        test_dir = image_path / "test"

        train_dataloader: torch.utils.data.DataLoader
        test_dataloader: torch.utils.data.DataLoader
        class_names: List[str]

        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=auto_transforms,  # perform same data transforms on our own data as the pretrained model
            batch_size=args.batch_size,
            pin_memory=True,
        )  # set mini-batch size to 32

        # get datasets for confusion matrix
        # Use ImageFolder to create dataset(s)
        train_dataset = train_data = datasets.ImageFolder(
            train_dir, transform=auto_transforms
        )
        val_dataset = test_data = datasets.ImageFolder(
            test_dir, transform=auto_transforms
        )

    # -----------------------------
    # BOSSNEW
    # Print a summary using torchinfo (uncomment for actual output)
    # print('Do a summary *before* freezing the features and changing the output classifier layer (uncomment for actual output)')
    # summary(model=model,
    #         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
    #         # col_names=["input_size"], # uncomment for smaller output
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"]
    # )

    # BOSSNEW
    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(device)

    ic(next(model.parameters()).device)

    # print('Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)')
    # summary(model,
    #         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
    #         verbose=0,
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"]
    # )

    # define loss function (criterion), optimizer, and learning rate scheduler
    # criterion = nn.CrossEntropyLoss().to(device)
    # Define loss and optimizer
    # BOSSNEW
    # loss_fn = nn.CrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # # define loss function (criterion), optimizer, and learning rate scheduler
    # # criterion = nn.CrossEntropyLoss().to(device)
    # # Define loss and optimizer
    # loss_fn = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None


    if args.info:
        info(args, dataset_root_dir=image_path)
        return

    if args.summary:
        print(" Running model summary ...")
        get_model_summary(
            model=model,
            input_size=(
                32,
                3,
                224,
                224,
            ),  # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )

        get_model_named_params(model)

        return

    if args.evaluate:
        # validate(val_loader, model, criterion, args)
        ic(run_validate(model, test_dataloader, device, loss_fn))
        return

    if args.download_and_predict:
        print(" Running download and predict command ...")
        download_and_predict(
            args.download_and_predict,
            model,
            Path(args.data),
            class_names=class_names,
            device=device,
        )
        return

    if args.predict:
        print(" Running predict command ...")
        ic(
            pred_and_plot_image(
                model,
                image_path,
                class_names,
                image_size=(224, 224),
                # transform: torchvision.transforms = None,
                device=device,
            )
        )
        # validate(val_loader, model, criterion, args)
        return

    ic(
        run_train(
            model,
            train_dataloader,
            test_dataloader,
            loss_fn,
            optimizer,
            args.epochs,
            device,
            args.batch_size
        )
    )
    print("No other options selected so we are training this model....")

    path_to_model = save_model_to_disk("ScreenNetV1", model)

    loaded_model_for_inference = run_get_model_for_inference(
        model, device, class_names, path_to_model
    )

    cmat = compute_confusion_matrix(model, test_dataloader, device)
    show_confusion_matrix_helper(
        cmat, class_names, to_disk=True, fname="confusion-matrix.png"
    )

    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)

    #     # train for one epoch
    #     ic(train(train_loader, model, criterion, optimizer, epoch, device, args))

    #     # evaluate on validation set
    #     acc1, test_loss, test_acc = validate(val_loader, model, criterion, args)

    #     # scheduler.step()

    #     # remember best acc@1 and save checkpoint
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)

    #     if not args.multiprocessing_distributed or (
    #         args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    #     ):
    #         save_checkpoint(
    #             {
    #                 "epoch": epoch + 1,
    #                 "arch": args.arch,
    #                 "state_dict": model.state_dict(),
    #                 "best_acc1": best_acc1,
    #                 "optimizer": optimizer.state_dict(),
    #                 # "scheduler": scheduler.state_dict(),
    #             },
    #             is_best,
    #         )


# def train(
#     train_loader: torch.utils.data.DataLoader,
#     model: torch.nn.Module,
#     criterion: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     epoch: int,
#     device: torch.device,
#     args: argparse.Namespace,
# ) -> Tuple[float, float]:
#     batch_time = AverageMeter("Time", ":6.3f")
#     data_time = AverageMeter("Data", ":6.3f")
#     losses = AverageMeter("Loss", ":.4e")
#     top1 = AverageMeter("Acc@1", ":6.2f")
#     top5 = AverageMeter("Acc@5", ":6.2f")
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1, top5],
#         prefix="Epoch: [{}]".format(epoch),
#     )

#     # switch to train mode
#     model.train()

#     # Setup train loss and train accuracy values
#     train_loss, train_acc = 0, 0

#     end = time.time()
#     for i, (images, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)

#         # Send data to target device
#         # move data to the same device as model
#         images: torch.Tensor
#         target: torch.Tensor
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         ic(next(model.parameters()).device)

#         # 1. Forward pass (logits)
#         y_pred: torch.Tensor
#         output: torch.Tensor
#         y_pred = output = model(images)

#         # 2. Calculate  and accumulate loss
#         # compute output
#         loss: torch.Tensor
#         loss = criterion(output, target)

#         # measure accuracy and record loss
#         acc1, acc5 = ic(accuracy(output, target, topk=(1, 5)))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))
#         top5.update(acc5[0], images.size(0))

#         # compute gradient and do SGD step
#         # 3. Optimizer zero grad
#         optimizer.zero_grad()

#         # 4. Loss backward
#         loss.backward()

#         # 5. Optimizer step
#         optimizer.step()

#         # Calculate and accumulate accuracy metric across all batches
#         y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
#         train_acc += (y_pred_class == target).sum().item() / len(y_pred)

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0:
#             progress.display(i + 1)

#     # Adjust metrics to get average loss and accuracy per batch
#     train_loss = train_loss / len(train_loader)
#     train_acc = train_acc / len(train_loader)
#     return train_loss, train_acc


# def validate(
#     val_loader: torch.utils.data.DataLoader,
#     model: torch.nn.Module,
#     criterion: torch.nn.Module,
#     args: argparse.Namespace,
# ) -> Tuple[int, float, float]:
#     def run_validate(loader, base_progress=0):

#         # Setup test loss and test accuracy values
#         test_loss, test_acc = 0, 0

#         # Turn on no grad context manager
#         # with torch.no_grad():
#         with torch.inference_mode():
#             end = time.time()
#             # Loop through DataLoader batches
#             for i, (images, target) in enumerate(loader):
#                 i = base_progress + i
#                 if args.gpu is not None and torch.cuda.is_available():
#                     images = images.cuda(args.gpu, non_blocking=True)
#                     ic("GPU mode")
#                 if torch.backends.mps.is_available():
#                     images = images.to("mps")
#                     target = target.to("mps")
#                     ic("MPS mode")
#                 if torch.cuda.is_available():
#                     target = target.cuda(args.gpu, non_blocking=True)
#                     ic("Cuda mode")

#                 # compute output
#                 # 1. Forward pass
#                 test_pred_logits = output = model(images)

#                 # 2. Calculate and accumulate loss
#                 loss = criterion(output, target)
#                 test_loss += loss.item()

#                 # measure accuracy and record loss
#                 acc1, acc5 = accuracy(output, target, topk=(1, 5))
#                 losses.update(loss.item(), images.size(0))
#                 top1.update(acc1[0], images.size(0))
#                 top5.update(acc5[0], images.size(0))

#                 # Calculate and accumulate accuracy
#                 test_pred_labels = test_pred_logits.argmax(dim=1)
#                 test_acc += (test_pred_labels == target).sum().item() / len(
#                     test_pred_labels
#                 )

#                 # measure elapsed time
#                 batch_time.update(time.time() - end)
#                 end = time.time()

#                 if i % args.print_freq == 0:
#                     progress.display(i + 1)

#     batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
#     losses = AverageMeter("Loss", ":.4e", Summary.NONE)
#     top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
#     top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
#     progress = ProgressMeter(
#         len(val_loader)
#         + (
#             args.distributed
#             and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))
#         ),
#         [batch_time, losses, top1, top5],
#         prefix="Test: ",
#     )

#     # switch to evaluate mode
#     model.eval()

#     run_validate(val_loader)
#     if args.distributed:
#         top1.all_reduce()
#         top5.all_reduce()

#     if args.distributed and (
#         len(val_loader.sampler) * args.world_size < len(val_loader.dataset)
#     ):
#         aux_val_dataset = Subset(
#             val_loader.dataset,
#             range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),
#         )
#         aux_val_loader = torch.utils.data.DataLoader(
#             aux_val_dataset,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.workers,
#             pin_memory=True,
#         )
#         run_validate(aux_val_loader, len(val_loader))

#     progress.display_summary()

#     # Adjust metrics to get average loss and accuracy per batch
#     test_loss = test_loss / len(val_loader)
#     test_acc = test_acc / len(val_loader)
#     # return test_loss, test_acc

#     return top1.avg, test_loss, test_acc


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# https://discuss.pytorch.org/t/pred-output-topk-maxk-1-true-true-runtimeerror-selected-index-k-out-of-range/126940
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # with torch.no_grad():
    with torch.inference_mode():
        maxk = max(topk)
        ic(maxk)
        ic(output.shape)
        batch_size = target.size(0)
        ic(batch_size)

        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_random_images_from_dataset(
    model: torch.nn.Module,
    test_dir: pathlib.PosixPath,
    class_names: List[str],
    num_images_to_plot: int = 3,
    device: torch.device = None,
    y_preds: List[torch.Tensor] = [],
    y_pred_tensor: torch.Tensor = None,
):

    # Get a random list of image paths from test set
    import random

    num_images_to_plot = 3
    test_image_path_list = list(
        Path(test_dir).glob("*/*.jpg")
    )  # get list all image paths from test data
    test_image_path_sample = random.sample(
        population=test_image_path_list,  # go through all of the test image paths
        k=num_images_to_plot,
    )  # randomly select 'k' image paths to pred and plot

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
            image_size=(224, 224),
            device=device,
            y_preds=y_preds,
            y_pred_tensor=y_pred_tensor,
        )


# y_preds = []
# y_pred_tensor = None

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = None,
    y_preds: List[torch.Tensor] = [],
    y_pred_tensor: torch.Tensor = None,
):

    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # 7. Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # boss: Put predictions on CPU for evaluation
    # source: https://www.learnpytorch.io/03_pytorch_computer_vision/#11-save-and-load-best-performing-model
    ic(target_image_pred_probs)
    y_preds.append(target_image_pred_probs.cpu())
    # boss: Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plot_image_with_predicted_label(
        to_disk=True,
        img=img,
        target_image_pred_label=target_image_pred_label,
        target_image_pred_probs=target_image_pred_probs,
        class_names=class_names,
        fname=f"example.png",
    )


# wrapper function of common code
def run_save_model_for_inference(model: torch.nn.Module) -> Tuple[pathlib.PosixPath]:
    """Save model to disk

    Args:
        model (torch.nn.Module): _description_

    Returns:
        Tuple[pathlib.PosixPath]: _description_
    """
    ic("Saving model to disk ...")
    path_to_model = save_model_to_disk("ScreenNetV1", model)
    ic(path_to_model)
    return path_to_model


# wrapper function of common code
def run_get_model_for_inference(
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    path_to_model: pathlib.PosixPath,
) -> torch.nn.Module:
    """wrapper function to load model .pth file from disk

    Args:
        model (torch.nn.Module): _description_
        device (torch.device): _description_
        class_names (List[str]): _description_

    Returns:
        Tuple[pathlib.PosixPath, torch.nn.Module]: _description_
    """
    loaded_model_for_inference = load_model_for_inference(
        path_to_model, device, class_names
    )
    # rich.inspect(loaded_model_for_inference, all=True)
    return loaded_model_for_inference


# SOURCE: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
# Saving & Loading Model for Inference
def save_model_to_disk(my_model_name: str, model: torch.nn.Module):
    # Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(
        parents=True,  # create parent directories if needed
        exist_ok=True,  # if models directory already exists, don't error
    )

    # Create model save path
    MODEL_NAME = f"{my_model_name}.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(
        obj=model.state_dict(),  # only saving the state_dict() only saves the learned parameters
        f=MODEL_SAVE_PATH,
    )
    print("Model saved to path {} successfully.".format(MODEL_SAVE_PATH))
    return MODEL_SAVE_PATH


# NOTE: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
def load_model_for_inference(
    save_path: str, device: str, class_names: List[str]
) -> nn.Module:
    model = setup_efficientnet_model(device, class_names)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print("Model loaded from path {} successfully.".format(save_path))
    # get_model_summary(model)
    return model


# SOURCE: https://github.com/a-sasikumar/image_caption_errors/blob/d583dc77cfa9938bb15297b3096a959fe6084b66/models/model.py
def load_model_from_disk(save_path: str, empty_model: nn.Module) -> nn.Module:
    # Loading Model for Inference
    empty_model.load_state_dict(torch.load(save_path))
    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
    empty_model.eval()
    print("Model loaded from path {} successfully.".format(save_path))
    return empty_model


def plot_image_with_predicted_label(
    to_disk: bool = True,
    img: Image = None,
    target_image_pred_label: torch.Tensor = None,
    target_image_pred_probs: torch.Tensor = None,
    class_names: List[str] = None,
    fname: str = "plot.png",
):
    # 10. Plot image with predicted label and probability
    if not to_disk:
        plt.ion()

    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)

    if to_disk:
        plt.imsave(fname, img)
        # ic(plt.savefig(fname))


def validate_seed(seed: int):
    ic(seed, type(seed))
    devices.seed_everything(seed)

    # Nevermind, the unexpected behaviour is from cpu not mps. I was using from_numpy which the document indicate it will create a tensor buffer that shared memory space (I didn't catch this at first). This shared memory space seems to only be valid for cpu version, and nor mps version

    # I tried switching cpu, and mps order around without using copy(), and got expected result, but the base_tensor value is modified after cpu computation. So in my conclusion, using copy() with from_numpy is needed to have consistent behaviour on both cpu, and mps

    # https://github.com/pytorch/pytorch/issues/77988
    # Test seed set correctly
    coeff = 0.5
    base_tensor = np.random.rand(100, 100).astype(np.float32)
    grad_tensor = np.random.rand(100, 100).astype(np.float32)

    device = "cpu"
    cpu_tensor = torch.from_numpy(base_tensor.copy()).to(device)  # Change this line
    cpu_tensor.requires_grad = True
    cpu_tensor.grad = torch.from_numpy(grad_tensor.copy()).to(
        device
    )  # Change this line

    with torch.no_grad():
        cpu_tensor.add_(-coeff * cpu_tensor.grad)

    if torch.backends.mps.is_available():
        device = "mps"
        mps_tensor = torch.from_numpy(base_tensor.copy()).to(device)  # Change this line
        mps_tensor.requires_grad = True
        mps_tensor.grad = torch.from_numpy(grad_tensor.copy()).to(
            device
        )  # Change this line

        with torch.no_grad():
            mps_tensor.add_(-coeff * mps_tensor.grad)

        print(cpu_tensor.detach().cpu().numpy() - mps_tensor.detach().cpu().numpy())


def info(args, dataset_root_dir=""):
    platform.platform()
    print(
        watermark(
            packages="torch,pytorch_lightning,torchmetrics,torchvision,matplotlib,rich,PIL,numpy,mlxtend"
        )
    )
    devices.mps_check()
    validate_seed(args.seed)
    walk_through_dir(dataset_root_dir)
    sys.exit(0)


# func to save model checkpoint
# SOURCE: https://github.com/PineAppleUser/CVprojects/blob/ad49656a0a69354c134554a93d90e07913aa0dab/segmentationLungs/utils.py
def save_checkpoint(state, filename="saved_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# func to load model checkpoint
# SOURCE: https://github.com/PineAppleUser/CVprojects/blob/ad49656a0a69354c134554a93d90e07913aa0dab/segmentationLungs/utils.py
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# #function to save prediction as an image
# # SOURCE: https://github.com/PineAppleUser/CVprojects/blob/ad49656a0a69354c134554a93d90e07913aa0dab/segmentationLungs/utils.py
# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="mps"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

#     model.train()


def get_model_named_params(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(name, ":", param.requires_grad)

# SOURCE: https://github.com/mrdbourke/pytorch-apple-silicon/blob/main/01_cifar10_tinyvgg.ipynb
def print_train_time(start, end, device=None, machine=None):
    """Prints difference between start and end time.
    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    if device:
        print(f"\nTrain time on {machine} using PyTorch device {device}: {total_time:.3f} seconds\n")
    else:
        print(f"\nTrain time: {total_time:.3f} seconds\n")
    return round(total_time, 3)

if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as ex:

        print(str(ex))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.TracebackException(exc_type, exc_value, exc_traceback)
        traceback_str = "".join(tb.format_exception_only())
        print("Error Class: {}".format(str(ex.__class__)))

        output = "[{}] {}: {}".format("UNEXPECTED", type(ex).__name__, ex)
        print(output)
        print("exc_type: {}".format(exc_type))
        print("exc_value: {}".format(exc_value))
        traceback.print_tb(exc_traceback)
        bpdb.pm()
