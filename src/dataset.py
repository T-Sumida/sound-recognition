# coding: utf-8

from typing import List, Dict

import cv2
import librosa
import numpy as np
import torch.utils.data as data

LABEL_CODE = {
    "A": 0, "B": 1, "C": 2
}
INV_LABEL_CODE = {v: k for k, v in LABEL_CODE.items()}


def normalize(X: np.ndarray, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6) -> np.ndarray:
    """画像の正規化

    Args:
        X (np.ndarray): 3chの画像
        mean ([type], optional): 平均. Defaults to None.
        std ([type], optional): 標準偏差. Defaults to None.
        norm_max ([type], optional): 正規化最大値. Defaults to None.
        norm_min ([type], optional): 正規化最小値. Defaults to None.
        eps ([type], optional): 下駄. Defaults to 1e-6.

    Returns:
        np.ndarray: 正規化された3ch画像
    """
    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


class SpectrogramDataset(data.Dataset):
    def __init__(
            self, file_list: List, settings: Dict,
            waveform_transforms=None, spectrogram_transforms=None) -> None:
        """コンストラクタ

        Args:
            file_list (List): ファイルリスト
            img_size (int): 変換後の画像サイズ
            target_sr (int): SR
            waveform_transforms ([type], optional): 波形データに対する変換関数. Defaults to None.
            spectrogram_transforms ([type], optional): スペクトログラムに対する変換関数. Defaults to None.
        """
        print('ok')
        self.file_list = file_list
        self.img_size = settings['img_size']
        self.target_sr = settings['target_sr']
        self.period = settings['period']
        self.n_mels = settings['n_mels']
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        file_path, label_code = self.file_list[idx]

        y, _ = librosa.core.load(file_path, sr=self.sr, mono=True)

        len_y = len(y)
        effective_length = self.sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.signal_augment:
            y = self.signal_augment(y)

        image = self.create_melspec_image(y)
        labels = np.zeros(len(LABEL_CODE), dtype="f")
        labels[LABEL_CODE[label_code]] = 1

        if self.spectrogram_transforms:
            image = self.spectrogram_transforms(image)

        return image, labels

    def create_melspec_image(self, y):
        melspec1 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y, sr=self.target_sr, n_mels=self.n_mels, fmin=20, fmax=16000,
                n_fft=self.target_sr//40, hop_length=self.target_sr//100
            ), ref=np.max
        ).astype(np.float32)

        melspec2 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y, sr=self.target_sr, n_mels=self.n_mels, fmin=20, fmax=16000,
                n_fft=self.target_sr//20, hop_length=self.target_sr//40
            ), ref=np.max
        ).astype(np.float32)

        melspec3 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y, sr=self.target_sr, n_mels=self.n_mels, fmin=20, fmax=16000,
                n_fft=self.target_sr//10, hop_length=self.target_sr//20
            ), ref=np.max
        ).astype(np.float32)

        melspec1 = cv2.resize(melspec1, (250, 128))
        melspec2 = cv2.resize(melspec2, (250, 128))
        melspec3 = cv2.resize(melspec3, (250, 128))

        melspec = np.stack([melspec1, melspec2, melspec3], axis=-1)
        return normalize(melspec)
