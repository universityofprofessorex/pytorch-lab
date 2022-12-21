# SOURCE: https://github.com/socialhourmobile/SD-hassan-ns/blob/3b6b266b17e0fd0a9b17374cd2afbf4c59b7c245/modules/devices.py
import argparse
import contextlib

from typing import Optional, Union

import torch

import errors
from icecream import ic


# has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
# check `getattr` and try it for compatibility
def has_mps() -> bool:
    if not getattr(torch, "has_mps", False):
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False


def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def get_optimal_device(args: argparse.Namespace):
    if torch.cuda.is_available():
        # from modules import shared
        device_id: Optional[Union[int, None]]
        device_id = args.gpu

        if device_id is not None:
            cuda_device = f"cuda:{device_id}"
            return torch.device(cuda_device)
        else:
            return torch.device("cuda")

    if has_mps():
        return torch.device("mps")

    return cpu


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu = torch.device("cpu")
device = (
    device_interrogate
) = (
    device_gfpgan
) = device_swinir = device_esrgan = device_scunet = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16


def randn(seed: int, shape: int) -> torch.Tensor:
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == "mps":
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape: int):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == "mps":
        generator = torch.Generator(device=cpu)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    return torch.randn(shape, device=device)


# SOURCE: https://github.com/socialhourmobile/SD-hassan-ns/blob/3b6b266b17e0fd0a9b17374cd2afbf4c59b7c245/modules/shared.py#L42
def autocast(disable=False, precision: str = "autocast"):
    """_summary_

    Args:
        precision (str): Options include ["full", "autocast"]
        disable (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


# MPS workaround for https://github.com/pytorch/pytorch/issues/79383
def mps_contiguous(input_tensor: torch.Tensor, device: torch.device):
    """Returns a contiguous in memory tensor containing the same data as self tensor. If self tensor is already in the specified memory format, this function returns the self tensor.

    Args:
        input_tensor (torch.Tensor): _description_
        device (torch.device): _description_

    Returns:
        _type_: _description_
    """
    return input_tensor.contiguous() if device.type == "mps" else input_tensor


def mps_contiguous_to(input_tensor: torch.Tensor, device: torch.device):
    return mps_contiguous(input_tensor, device).to(device)


def mps_check():
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

    else:
        ic(torch.has_mps)
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print(x)
        else:
            print("MPS device not found.")

        mps_device = torch.device("mps")

        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        # Or
        x = torch.ones(5, device="mps")

        # Any operation happens on the GPU
        y = x * 2


# SOURCE: https://github.com/pytorch/pytorch/issues/77988
def seed_everything(seed: int):
    # Ref: https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
