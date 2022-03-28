# SNet Backbone

This directory contains the implementation of the SNet146 backbone. We use the code from [ShuffleNetV2](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2).

## Requirements

Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Usage

Train:

```shell
python train.py --model-size thunder --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```

Eval:

```shell
python train.py --eval --eval-resume YOUR_WEIGHT_PATH --model-size thunder --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```

## Trained Models

The training log is put in `log`. And the pretrained weights file is `../weights/snet146-3000000.pth.tar`.

## Results

| Model | FLOPs | Top-1 | Top-5 |
| :---: | :---: | :---: | :---: |
| SNet146 | 146M | 32.2 | 12.0 |
