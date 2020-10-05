# coding: utf-8

import random

import librosa
import numpy as np


def stretch(data: np.array) -> np.array:
    """伸長拡張する

    Args:
        data (np.array): 信号データ

    Returns:
        np.array: 拡張後信号データ
    """
    rate = random.uniform(0.9, 1.1)
    input_length = data.shape[0]
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        return np.array(data[:input_length])
    else:
        return np.pad(data, (0, max(0, input_length - len(data))), "constant")


def add_white_noise(data: np.array, rate: float = 0.005) -> np.array:
    """ホワイトノイズ拡張

    Args:
        data (np.array): 信号データ
        rate (float, optional): ノイズの加算係数. Defaults to 0.005.

    Returns:
        np.array: 拡張後信号データ
    """
    return data + rate*np.random.randn(data.shape[0])


def pitch_shift(data: np.array) -> np.array:
    """ピッチシフト拡張

    Args:
        data (np.array): 信号データ

    Returns:
        np.array: 拡張後信号データ
    """
    rate = random.randrange(start=-4, stop=4, step=1)
    return librosa.effects.pitch_shift(data, 32000, n_steps=rate)


def change_volume(x: np.array) -> np.array:
    """ボリューム変化拡張

    Args:
        x (np.array): 信号データ

    Returns:
        np.array: 拡張後信号データ
    """
    coef = random.uniform(0.7, 1.2)
    return x * coef
