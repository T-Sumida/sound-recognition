# coding: utf-8
import os
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def set_seed(seed: int = 1234):
    """乱数を固定する

    Args:
        seed (int, optional): シード値. Defaults to 1234.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    """引数を取得する

    Returns:
        argparse.Namespace: 引数情報
    """
    parser = argparse.ArgumentParser(
        prog="train.py", usage="training sound classifier.",
        description="set config file path and experiment name",
        add_help=True
    )
    parser.add_argument("config_path", type=str, help="config file path")
    parser.add_argument("exp_name", type=str, help="mlflow experience name")
    return parser.parse_args()


def plot_log(log_path: str, save_dir: str):
    log_df = pd.read_csv(log_path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(log_df['epoch'], log_df['loss'], label='loss')
    ax.plot(log_df['epoch'], log_df['val_loss'], label='val_loss')
    plt.savefig(os.path.join(save_dir, "loss.png"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(log_df['epoch'], log_df['train_f1'], label='F1')
    ax.plot(log_df['epoch'], log_df['val_f1'], label='val_F1')
    plt.savefig(os.path.join(save_dir, "f1.png"))
