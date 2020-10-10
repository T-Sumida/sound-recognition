# -*- coding :utf-8 -*-

"""
Usage
実行すると以下のような画面がコンソールに出力される。
```
==================================================
   0 Microsoft サウンド マッパー - Input, MME (2 in, 0 out)
>  1 マイク (Realtek Audio), MME (2 in, 0 out)
   2 Microsoft サウンド マッパー - Output, MME (0 in, 2 out)
<  3 EX-LDGC251T (NVIDIA High Defini, MME (0 in, 2 out)
   4 スピーカー / ヘッドホン (Realte, MME (0 in, 2 out)
   5 プライマリ サウンド キャプチャ ドライバー, Windows DirectSound (2 in, 0 out)
   6 マイク (Realtek Audio), Windows DirectSound (2 in, 0 out)
   7 プライマリ サウンド ドライバー, Windows DirectSound (0 in, 2 out)
   8 EX-LDGC251T (NVIDIA High Definition Audio), Windows DirectSound
   9 スピーカー / ヘッドホン (Realtek Audio), Windows DirectSound (0 in, 2 out)
  10 Realtek ASIO, ASIO (2 in, 2 out)
  11 スピーカー / ヘッドホン (Realtek Audio), Windows WASAPI (0 in, 2 out)
  12 EX-LDGC251T (NVIDIA High Definition Audio), Windows WASAPI (0 in, 2 out)
  13 マイク (Realtek Audio), Windows WASAPI (2 in, 0 out)
  14 マイク (Realtek HD Audio Mic input), Windows WDM-KS (2 in, 0 out)
  15 Speakers (Realtek HD Audio output), Windows WDM-KS (0 in, 2 out)
  16 ステレオ ミキサー (Realtek HD Audio Stereo input), Windows WDM-KS (2 in, 0 out)
  17 ライン入力 (Realtek HD Audio Line input), Windows WDM-KS (2 in, 0 out)
  18 Output (), Windows WDM-KS (0 in, 2 out)
==================================================
********** PLEASE INPUT DEVICE ID (quit > 'q') ********
>
```
プロンプトに試したいデバイスID（例：1 マイク... の場合、1）を入力すると、信号がリアルタイムに描画される。
終了時はプロンプトに'q'キーを入力しエンターすればよい。
"""

import queue
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import List

q = queue.Queue()
mapping = []
COLOR_LIST = [
    'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'pink', 'tab:purple'
]


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, device_id, info) -> None:
        """コンストラクタ
        Arguments:
            device_id {int} -- INPUT機器のデバイスID
            info {dict} -- INPUT機器の情報
        """
        global mapping
        samplerate = info['default_samplerate']
        length = int(200 * samplerate / (1000 * 10))
        self.plotdata = np.zeros((length, info['max_input_channels']))

        fig, ax = plt.subplots(info['max_input_channels'], 1)
        self.all_lines = []
        for i in range(info['max_input_channels']):
            self.all_lines.append(
                ax[i].plot(
                    self.plotdata[:, i],
                    label="Channel:{}".format(i + 1),
                    color=COLOR_LIST[i]
                )
            )
            ax[i].axis((0, len(self.plotdata), -1, 1))
            ax[i].set_yticks([0])
            ax[i].yaxis.grid(True)
            ax[i].tick_params(bottom=False, top=False, labelbottom=False,
                              right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)
        fig.legend()
        self.stream = sd.InputStream(
            device=int(device_id),
            channels=max([i+1 for i in range(info['max_input_channels'])]),
            samplerate=samplerate,
            callback=audio_callback
        )
        self.stream.start()
        animation.TimedAnimation.__init__(self, fig, interval=10, blit=True)

    def _draw_frame(self, framedata) -> None:
        """描画処理
        Arguments:
            framedata {list} -- 未使用
        """
        while True:
            try:
                data = q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data

        for i, lines in enumerate(self.all_lines):
            for column, line in enumerate(lines):
                line.set_ydata(self.plotdata[:, i])

    def new_frame_seq(self) -> List[int]:
        """次フレーム用の処理
        Returns:
            list -- 未使用（故に適当なListを返す）
        """
        return iter(range(3))


def print_devices() -> None:
    """デバイスリストを表示する"""
    print("==================================================")
    print(sd.query_devices())
    print("==================================================")


def check_device(device_id) -> List[str]:
    """入力されたデバイスIDが適切かどうかをチェックする
    Arguments:
        device_id {str} -- デバイスID
    Returns:
        list or None -- 適切だった場合はデバイス情報を返す
    """
    if not device_id.isdecimal():
        return None
    try:
        info = sd.query_devices(device=int(device_id), kind='input')
        return info
    except Exception as e:
        print(f"[device list get error : {e}")
        return None


def audio_callback(indata, frames, time, status):
    """信号入力用コールバック
    Arguments:
        indata {numpy.array} -- 信号
        frames {int} -- 信号のサイズ
        time {CData} -- ADCキャプチャ時間
        status {CallbackFlags} -- エラー収集用のフラグ
    """
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::10, :])


if __name__ == "__main__":
    # global mapping

    while True:
        print_devices()
        print("********** PLEASE INPUT DEVICE ID (quit > \'q\') ********")
        print("> ", end="")
        line = input()

        if line == 'q':
            print('exit')
            break
        info = check_device(line)
        if info is None:
            print("----- device_id {} doesn't have INPUT -----".format(line))
            continue

        ani = SubplotAnimation(int(line), info)
        plt.show()
