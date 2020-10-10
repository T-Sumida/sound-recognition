# -*- coding: utf-8 -*-

import cv2
import librosa
import numpy as np


class Signal2ImageTransformer:
    def __init__(self, sr: int, n_mels: int) -> None:
        """コンストラクタ

        Args:
            sr (int): 入力信号のサンプリングレート
            n_mels (int): メルスペクトログラムの係数
        """
        self.target_sr = sr
        self.n_mels = n_mels
        pass

    def transform(self, X: np.array) -> np.array:
        """信号データからメルスペクトログラム画像を生成する

        Args:
            y (np.array): 信号データ

        Returns:
            np.array: 画像データ
        """
        melspec1 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                X, sr=self.target_sr, n_mels=self.n_mels, fmin=20, fmax=16000,
                n_fft=self.target_sr//40, hop_length=self.target_sr//100
            ), ref=np.max
        ).astype(np.float32)

        melspec2 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                X, sr=self.target_sr, n_mels=self.n_mels, fmin=20, fmax=16000,
                n_fft=self.target_sr//20, hop_length=self.target_sr//40
            ), ref=np.max
        ).astype(np.float32)

        melspec3 = librosa.power_to_db(
            librosa.feature.melspectrogram(
                X, sr=self.target_sr, n_mels=self.n_mels, fmin=20, fmax=16000,
                n_fft=self.target_sr//10, hop_length=self.target_sr//20
            ), ref=np.max
        ).astype(np.float32)

        melspec1 = cv2.resize(melspec1, (250, 128))
        melspec2 = cv2.resize(melspec2, (250, 128))
        melspec3 = cv2.resize(melspec3, (250, 128))

        melspec = np.stack([melspec1, melspec2, melspec3], axis=-1)
        return self.__normalize(melspec)

    def __normalize(self, X: np.ndarray, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6) -> np.ndarray:
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
