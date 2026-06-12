# IAEC

Official implementation of **IAEC-DepressNet**, a multimodal depression detection framework based on identity-adaptive feature purification and cross-modal emotional consistency modeling.

## Overview

Depression detection based on multimodal signals has attracted increasing attention in affective computing and mental health analysis. However, automatic depression detection still faces several challenges, including identity-related interference in visual representations, insufficient temporal emotional alignment between audio and visual modalities, and unstable multimodal fusion.

This project implements **IAEC-DepressNet**, which aims to improve multimodal depression detection by combining visual feature purification, cross-modal emotional consistency modeling, and multimodal semantic fusion.

## Repository Structure

```text
IAEC/
├── model/                 # Model definitions
├── preprocess/            # Data preprocessing scripts
├── pretrain/              # Pretraining-related scripts
├── tools/                 # Utility functions
├── config.yaml            # Dataset paths and experimental configuration
├── dataloader.py          # Data loading for AVEC2014
├── dataloader_2017.py     # Data loading for AVEC2017/2019
├── model_test.py          # Testing script
├── move.py                # move model script
├── train_attention.py     # Multimodal attention/fusion training script
├── train_feature.py       # Feature-level training script
```

## Requirements

A typical environment includes:

```bash
python >= 3.8
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
tqdm
opencv-python
pyyaml
```

## Dataset Preparation

This repository is designed for depression detection experiments on AVEC datasets, including:

- AVEC2014
- AVEC2017
- AVEC2019

Due to dataset license restrictions, the original datasets are not included in this repository. Please obtain the datasets from the official AVEC/DAIC-WOZ providers and organize them according to your own local directory structure.

The dataset paths should be configured in `config.yaml`. For example:

```yaml
dataset: "AVEC2014"
# dataset: "AVEC2017"
# dataset: "AVEC2019"

AVEC2014:
  train_path: "/path/to/AVEC2014/train"
  dev_path: "/path/to/AVEC2014/dev"
  test_path: "/path/to/AVEC2014/test"

AVEC2017:
  base_root: "/path/to/AVEC2017"
  train_path: "/path/to/AVEC2017/train_label.csv"
  dev_path: "/path/to/AVEC2017/dev_label.csv"
  test_path: "/path/to/AVEC2017/test_label.csv"

AVEC2019:
  base_root: "/path/to/AVEC2019"
  train_path: "/path/to/AVEC2019/train_label.csv"
  dev_path: "/path/to/AVEC2019/dev_label.csv"
  test_path: "/path/to/AVEC2019/test_label.csv"
```

## Configuration

Before training or testing, please modify `config.yaml` according to your dataset and experimental setting.

Important options include:

```yaml
dataset: "AVEC2014"

stage: 2
mode: "multimodal"

w_o_ID: "1"
w_o_SE: "1"
w_o_SIM: "1"
w_o_Video_Guide: "1"
w_o_fs: "1"

audio_feature_type: "audio_npy"
```

The flags are used to control different modules or ablation settings:

- `w_o_ID`: whether to include the identity-related module
- `w_o_SE`: whether to include the SE-based attention module
- `w_o_SIM`: whether to include the cross-modal similarity consistency constraint
- `w_o_Video_Guide`: whether to include the video-guided fusion setting
- `w_o_fs`: whether to include the frame-select module setting

## Training

### 1. Train the emonet model(Stage One)

```
python pretrain/train.py
```

### 2. Train the  feature model(Stage Two)

```bash
python train_feature.py
```

This script trains the feature-level model and saves checkpoints according to the paths defined in `config.yaml` and the script arguments.

### 3. Train the fusion model(Stage three)

```bash
python train_attention.py
```

This script loads the trained feature model and trains the multimodal fusion/attention module.

## Testing

After training, run:

```bash
python model_test.py
```

## Notes

1. The current code contains local absolute paths. Please modify them before running the project.
2. Some scripts may depend on the local package name or directory structure. If import errors occur, please check `sys.path` and package names.
3. The repository is intended for academic research only.

