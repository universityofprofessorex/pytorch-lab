#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Import rich and whatever else we need
# %load_ext rich
# %matplotlib inline
import sys
import os

import os.path
import pathlib

extra_modules_path_api = pathlib.Path("../going_modular")
extra_modules_path = os.path.abspath(str(extra_modules_path_api))
print(extra_modules_path)

# sys.path.insert(1, extra_modules_path)
sys.path.append(extra_modules_path)
sys.path.append("../")
import rich
from rich import inspect, print
from rich.console import Console
from icecream import ic

console = Console()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
import torch
import torchvision
from torchvision import datasets, transforms

assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert (
    int(torchvision.__version__.split(".")[1]) >= 13
), "torchvision version should be 0.13+"
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
# ---------------------------------------------------------------------------

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work

from torchinfo import summary

# breakpoint()
from going_modular import data_setup, engine

import torchmetrics, mlxtend

print(f"mlxtend version: {mlxtend.__version__}")
assert (
    int(mlxtend.__version__.split(".")[1]) >= 19
), "mlxtend verison should be 0.19.0 or higher"

# Import accuracy metric
from helper_functions import accuracy_fn  # Note: could also use torchmetrics.Accuracy()

# # Setup device agnostic code
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device

import argparse
import os
import random
import shutil
import time
import warnings

from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from timeit import default_timer as timer
from pathlib import Path
import shutil
import requests
import zipfile
from pathlib import Path
from typing import List, Tuple
from PIL import Image


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


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


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
print(model_names)

shared_datasets_path_api = pathlib.Path(os.path.expanduser("~/Downloads/datasets"))
shared_datasets_path = os.path.abspath(str(shared_datasets_path_api))
print(f"shared_datasets_path - {shared_datasets_path}")
DEFAULT_DATASET_DIR = Path(f"{shared_datasets_path}")

# # Setup path to data folder
# data_path = Path(f"{shared_datasets_path}")
# image_path = data_path / "twitter_facebook_tiktok"
# train_dir = image_path / "twitter_facebook_tiktok" / "train"
# test_dir = image_path / "twitter_facebook_tiktok" / "test"


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
# parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
# parser.add_argument(
#     "--wd",
#     "--weight-decay",
#     default=1e-4,
#     type=float,
#     metavar="W",
#     help="weight decay (default: 1e-4)",
#     dest="weight_decay",
# )
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
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
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
    rich.inspect(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

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


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

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
        print("=> using pre-trained model '{}'".format(args.arch))
        # breakpoint()
        weights = models.__dict__[args.model_weights].DEFAULT
        auto_transforms = weights.transforms()
        model = models.__dict__[args.arch](pretrained=True, weights=weights)
    else:
        print("=> creating model '{}'".format(args.arch))
        # breakpoint()
        weights = models.__dict__[args.model_weights].DEFAULT
        auto_transforms = weights.transforms()
        model = models.__dict__[args.arch](weights=weights)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
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
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
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
    # define loss function (criterion), optimizer, and learning rate scheduler
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = loss_fn = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

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
        train_dir = image_path / "twitter_facebook_tiktok" / "train"
        test_dir = image_path / "twitter_facebook_tiktok" / "test"

        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transform=auto_transforms,  # perform same data transforms on our own data as the pretrained model
            batch_size=args.batch_size,
        )  # set mini-batch size to 32

        # get datasets for confusion matrix
        # Use ImageFolder to create dataset(s)
        train_dataset = train_data = datasets.ImageFolder(
            train_dir, transform=auto_transforms
        )
        val_dataset = test_data = datasets.ImageFolder(
            test_dir, transform=auto_transforms
        )

        # traindir = os.path.join(args.data, "train")
        # valdir = os.path.join(args.data, "val")
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )

        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose(
        #         [
        #             transforms.RandomResizedCrop(224),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             normalize,
        #         ]
        #     ),
        # )

        # val_dataset = datasets.ImageFolder(
        #     valdir,
        #     transforms.Compose(
        #         [
        #             transforms.Resize(256),
        #             transforms.CenterCrop(224),
        #             transforms.ToTensor(),
        #             normalize,
        #         ]
        #     ),
        # )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=(train_sampler is None),
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=train_sampler,
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=val_sampler,
    # )

    train_loader = train_dataloader
    val_loader = test_dataloader

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

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.predict:
        # pred_and_plot_image(model: torch.nn.Module,
        #                 image_path: str,
        #                 class_names: List[str],
        #                 image_size: Tuple[int, int] = (224, 224),
        #                 transform: torchvision.transforms = None,
        #                 device: torch.device=device)
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

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        ic(train(train_loader, model, criterion, optimizer, epoch, device, args))

        # evaluate on validation set
        acc1, test_loss, test_acc = validate(val_loader, model, criterion, args)

        # scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                    # "scheduler": scheduler.state_dict(),
                },
                is_best,
            )


def train(
    train_loader, model, criterion, optimizer, epoch, device, args
) -> Tuple[float, float]:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Send data to target device
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 1. Forward pass
        y_pred = output = model(images)

        # 2. Calculate  and accumulate loss
        # compute output
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == target).sum().item() / len(y_pred)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)
    return train_loss, train_acc


def validate(val_loader, model, criterion, args) -> Tuple[int, float, float]:
    def run_validate(loader, base_progress=0):

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0

        # Turn on no grad context manager
        with torch.no_grad():
            end = time.time()
            # Loop through DataLoader batches
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to("mps")
                    target = target.to("mps")
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                # 1. Forward pass
                test_pred_logits = output = model(images)

                # 2. Calculate and accumulate loss
                loss = criterion(output, target)
                test_loss += loss.item()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += (test_pred_labels == target).sum().item() / len(
                    test_pred_labels
                )

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader)
        + (
            args.distributed
            and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))
        ),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (
        len(val_loader.sampler) * args.world_size < len(val_loader.dataset)
    ):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(val_loader)
    test_acc = test_acc / len(val_loader)
    # return test_loss, test_acc

    return top1.avg, test_loss, test_acc


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    target_image_pred_probs
    # y_preds.append(target_image_pred_probs.cpu())
    # boss: Concatenate list of predictions into a tensor
    # y_pred_tensor = torch.cat(y_preds)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plt.ion()
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)


if __name__ == "__main__":
    main()
