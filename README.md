# sound-recognition
A template for your own music recognition machine learning project.

# Overview
This repository is a template for a machine learning project for multi-class classification of music files.

It is implemented so that you can try out various experimental conditions (model types, dataset creation, etc.) by rewriting the configuration file.

In addition, [MLFlow](https://mlflow.org/) is used to manage the experimental results.

However, this repository has only basic functionality and will be expanded in the future.

# Approach
The approach currently being implemented is as follows.

1. Clipping any time interval from a music file.
2. Perform augmentation on the signal data.
   - Time Stretch
   - Additional White Noise
   - Pitch Shift
   - Change Volume
3. From the signal data, a Mel-Spectrogram made of three different parameters is generated as an image.
4. The study is performed as an image classification problem using ResNet and other models.

# Environment
- Windows10 Home 64-bit
- Anaconda

```
$conda create -n {env_name} python=3.7.2
$activate {env_name}
$pip install -r requirements.txt
```

**It is possible to build an environment using nvidia-docker in wsl2.**

**Check Dockerfile.**

# Usage
### Start Experiment
```
$cd src
$python train.py settings.yaml {experiment name}
```

# License
Copyright © 2020 T_Sumida Distributed under the MIT License.