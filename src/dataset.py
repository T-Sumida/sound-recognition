# coding: utf-8

from typing import List, Dict, Tuple

import librosa
import numpy as np
import torch.utils.data as data

import augment
from transformer import Signal2ImageTransformer


class SpectrogramDataset(data.Dataset):
    def __init__(
            self, file_list: List, settings: Dict, labels: List,
            waveform_transforms=None, spectrogram_transforms=None) -> None:
        """コンストラクタ

        Args:
            file_list (List): ファイルリスト
            settings (Dict): 設定情報
            labels (List): ラベル情報
            waveform_transforms ([type], optional): 波形データに対する変換関数. Defaults to None.
            spectrogram_transforms ([type], optional): スペクトログラムに対する変換関数. Defaults to None.
        """
        self.file_list = file_list
        self.img_size = settings['img_size']
        self.period = settings['period']
        self.labels = labels
        self.transforms = Signal2ImageTransformer(
            settings['target_sr'], settings['n_mels'])
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms

    def __len__(self) -> int:
        """データセットのサイズを返す

        Returns:
            int: データセットのサイズ
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """

        Args:
            idx (int): batch数

        Returns:
            Tuple[np.array, np.array]: [画像データ, ラベルデータ]
        """
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

        if self.waveform_transforms:
            for func_name in self.waveform_transforms:
                try:
                    y = getattr(augment, func_name)(y)
                except AttributeError as e:
                    print(f"{func_name} is None. {e}")
        image = self.transforms(y)
        labels = np.zeros(len(self.labels), dtype="f")
        labels[self.labels[label_code]] = 1

        if self.spectrogram_transforms:
            image = self.spectrogram_transforms(image)

        return image, labels
