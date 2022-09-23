import os
import warnings

import hydra
import pytorch_lightning as pl
from torchmetrics.utilities.data import get_num_classes
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pywt
from src.utils.technical_utils import load_obj, flatten_omegaconf, convert_to_jit
from src.utils.utils import set_seed, save_useful_info
import cv2

im=cv2.imread('D:/workspace/SatPose/workspace/datasets/cubesat/images/img000004.jpg')
print(im)