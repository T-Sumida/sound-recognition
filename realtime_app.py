# -*- coding: utf-8 -*-
import argparse
import yaml

import torch
import numpy as np

import ml.my_model
from ml.transformer import Signal2ImageTransformer
from utils.audio import Audio


class SoundRecognitionApp():
    def __init__(self, cfg) -> None:
        self.transformer = Signal2ImageTransformer(**cfg['transforms'])
        self.audio = Audio(cfg['audio'])
        self.load_model(cfg['model'])
        pass

    def run(self):
        print("============= REALTIME START ==============")
        self.audio.start()
        self.flag = True

        try:
            while self.flag:
                status, data = self.audio.get()
                if status == Audio.ERROR:
                    print('[error]')
                    break
                elif status == Audio.WAIT:
                    continue
                mel_spec = self.preprocess(data)
                result = self.inference(mel_spec)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            self.audio.stop()
        print("============= REALTIME FINISH ==============")

    def preprocess(self, signal):
        return np.expand_dims(self.transformer.transform(signal), axis=0)

    def inference(self, X):
        image = torch.from_numpy(X.astype(np.float32)).clone()
        image.to(self.device).float()
        prob = self.model(image)['multilabel_proba'].detach().cpu().numpy()
        return prob

    def load_model(self, cfg):
        try:
            self.device = torch.device(cfg["device"])
            self.model = getattr(ml.my_model, cfg['name'])(**cfg['params'])
            self.model.load_state_dict(torch.load(cfg['path']))
            self.model.to(self.device)
        except AttributeError as e:
            print(f"Model {cfg['name']} is None. {e}")
            exit(1)
        except FileNotFoundError as e:
            print(f"{e}")
            exit(1)
        except Exception as e:
            print(f"{e}")
            exit(1)


def parse_arg():
    parser = argparse.ArgumentParser(
        prog="realtime_app.py",
        description="real time autdio recognition application.",
        add_help=True
    )

    parser.add_argument(
        "config_file", type=str,
        help="deploy.yml path"
    )
    return parser.parse_args()


def main():
    args = parse_arg()

    with open(args.config_file, 'r') as f:
        cfg = yaml.load(f)
    print(cfg)


if __name__ == "__main__":
    main()
