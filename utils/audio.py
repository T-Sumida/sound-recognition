# coding: utf-8
import sys
import queue
import numpy as np
import sounddevice as sd

from typing import Tuple


class Audio(object):
    ERROR = -1
    SUCCESS = 1
    WAIT = 0

    def __init__(self, cfg, dev_id=1, channels=1, audio_file=None) -> None:
        """コンストラクタ
        Arguments:
            cfg {json} -- パラメータ情報
        Keyword Arguments:
            audio_file {str} -- 音楽ファイルパス (default: {None})
        """
        self.cfg = cfg
        self.audio_file = audio_file
        if self.audio_file is not None:
            # ローカル用初期化
            self.__init_local()
            self.get = self.__get_from_local
        else:
            # リアルタイム用初期化
            self.channels = channels
            self.__init_realtime(dev_id)
            self.get = self.__get_from_input

    def __init_local(self) -> None:
        """ローカル用初期化"""
        import librosa
        self.data, sr = librosa.core.load(
            self.audio_file,
            sr=self.cfg['sampling_rate'],
            mono=True
        )
        self.max_counter = range(
            0, self.data.shape[0] - self.cfg['audio_length'],
            sr
        )[-1]
        self.counter = -self.cfg['block_size']

    def __init_realtime(self, dev_id) -> None:
        """リアルタイム用初期化"""
        self.buffer = np.zeros(self.cfg['audio_length'])

        info = sd.query_devices(device=int(dev_id), kind='input')
        print(info)
        # todo デバイスとチャンネルいじれるように
        self.stream = sd.InputStream(
            device=dev_id, channels=max([i+1 for i in range(info['max_input_channels'])]),
            samplerate=self.cfg['sampling_rate'],
            blocksize=self.cfg['block_size'],
            callback=self.__audio_callback
        )
        self.q = queue.Queue()
        self.status = Audio.SUCCESS

    def start(self) -> None:
        """ストリーミング開始"""
        self.q.queue.clear()
        self.stream.start()

    def stop(self) -> None:
        """ストリーミング停止"""
        self.stream.stop()

    def __get_from_local(self) -> Tuple[int, np.array]:
        """音楽ファイルから指定されたサイズ・スライド幅でデータを返す
        Returns:
            Tuple[int, np.array]: (ステータスコード, 信号)
        """
        self.counter += self.cfg['block_size']
        if self.counter > self.max_counter:
            return Audio.ERROR, 0
        return Audio.SUCCESS, self.data[self.counter:self.counter+self.cfg['audio_length']]

    def __get_from_input(self) -> Tuple[int, np.array]:
        """マイク入力から指定されたサイズ・スライド幅でデータを返す

        Returns:
            Tuple[int, np.array]: (ステータスコード, 信号)
        """
        try:
            data = self.q.get_nowait()
        except:
            return Audio.WAIT, None
        self.buffer = np.roll(self.buffer, -data.shape[0], axis=0)
        self.buffer[-data.shape[0]:] = data
        return self.status, self.buffer

    def __audio_callback(self, indata, frames, time, status) -> None:
        """信号入力用コールバック
        Arguments:
            indata {numpy.array} -- 信号
            frames {int} -- 信号のサイズ
            time {CData} -- ADCキャプチャ時間
            status {CallbackFlags} -- エラー収集用のフラグ
        """
        if status:
            print("[audio callback error] {}".format(status))
            print(status, file=sys.stderr)
            # self.status = Audio.ERROR
        self.q.put(indata[:, self.channels-1])