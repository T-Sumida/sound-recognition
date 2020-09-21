# coding: utf-8

import random

import librosa
import numpy as np


def stretch(data):
    rate = random.uniform(0.9, 1.1)
    input_length = data.shape[0]
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        return np.array(data[:input_length])
    else:
        return np.pad(data, (0, max(0, input_length - len(data))), "constant")


def add_white_noise(data, rate=0.005):
    return data + rate*np.random.randn(data.shape[0])


def pitch_shift(data):
    rate = random.randrange(start=-4, stop=4, step=1)
    return librosa.effects.pitch_shift(data, 32000, n_steps=rate)


def change_volume(x):
    coef = random.uniform(0.7, 1.2)
    return x * coef
