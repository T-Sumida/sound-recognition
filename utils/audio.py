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

    def __init__(self, cfg) -> None:
        """コンストラクタ
        Arguments:
            cfg {json} -- パラメータ情報
        Keyword Arguments:
            audio_file {str} -- 音楽ファイルパス (default: {None})
        """
        self.cfg = cfg

        # リアルタイム用初期化
        self.channels = self.cfg['device_ch']
        self.__init_realtime(self.cfg['device_id'])
        self.get = self.__get_from_input

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