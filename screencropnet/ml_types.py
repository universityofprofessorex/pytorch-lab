import numpy as np
from typing import NewType
import torch

ImageNdarrayBGR = NewType('ImageBGR', np.ndarray)
ImageNdarrayHWC = NewType('ImageHWC', np.ndarray)
TensorCHW = NewType('TensorCHW', torch.Tensor)
