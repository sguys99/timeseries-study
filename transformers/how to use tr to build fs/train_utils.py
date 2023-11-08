import json
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader



def split_df(
    df: pd.DataFrame, split: str, history_size: int = 120, horizon_size: int = 30
):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows

    :param df:
    :param split:
    :param history_size:
    :param horizon_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(horizon_size + 1, df.shape[0] - horizon_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    label_index = end_index - horizon_size
    start_index = max(0, label_index - history_size)

    history = df[start_index:label_index]
    targets = df[label_index:end_index]

    return history, targets


def pad_arr(arr: np.ndarray, expected_size: int = 120):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def df_to_np(df):
    arr = np.array(df)
    arr = pad_arr(arr)
    return arr


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, features, target):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        src, trg = split_df(df, split=self.split)

        src = src[self.features + [self.target]]

        src = df_to_np(src)

        trg_in = trg[self.features + [f"{self.target}_lag_1"]]

        trg_in = np.array(trg_in)
        trg_out = np.array(trg[self.target])

        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src, trg_in, trg_out