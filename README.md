# Advanced convolutions and data augmentations
> Use advanced convolutions and image transformations to train a CNN with CIFAR-10
- Dataset: [CIFAR-10](https://paperswithcode.com/dataset/cifar-10)
- Data augmentation: [Albumentations](https://albumentations.ai/)
  - Cutout
  - Horizontal flip
  - Affine transformations: translation, rotation, scaling, shear
- Model: fully [convolutional neural network](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
  - [Dilated convolution](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/dilation.gif)
  - [Depthwise separable convolution](https://animatedai.github.io/media/depthwise-separable-convolution-animation-3x3-kernel.gif)

## Installation
If you want to use our models, dataloaders, training engine, and other utilities, please run the following command:
```console
git clone https://github.com/woncoh1/era1a9.git
```
And then import the modules in Python:
```python
from era1a9 import data, models, engine, utils
```

## Targets
Acheive all of the followings using modular code organization:
- Test accuracy > 85.0 %
- Number of parameters < 200,000
- Number of epochs: unlimited

## Results
- Best train accuracy = 81.40 %
- Best test accuracy = 87.04 %
- Number of parameters = 163,744
- Number of epochs = 30

## Sample images
![si](https://github.com/woncoh1/era1a9/assets/12987758/05063be9-9214-4c69-bc22-2b330e5cf905)

## Model summary
```
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
Net (Net)                                [128, 10]                 --
├─Sequential (conv0)                     [128, 16, 32, 32]         --
│    └─Conv2d (0)                        [128, 16, 32, 32]         432
│    └─BatchNorm2d (1)                   [128, 16, 32, 32]         32
│    └─Dropout2d (2)                     [128, 16, 32, 32]         --
│    └─ReLU (3)                          [128, 16, 32, 32]         --
├─SkipBlock (conv1)                      [128, 32, 32, 32]         --
│    └─Sequential (conv1)                [128, 16, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 16, 32, 32]         2,304
│    │    └─BatchNorm2d (1)              [128, 16, 32, 32]         32
│    │    └─Dropout2d (2)                [128, 16, 32, 32]         --
│    │    └─ReLU (3)                     [128, 16, 32, 32]         --
│    └─Sequential (conv2)                [128, 32, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 16, 32, 32]         144
│    │    └─BatchNorm2d (1)              [128, 16, 32, 32]         32
│    │    └─Dropout2d (2)                [128, 16, 32, 32]         --
│    │    └─ReLU (3)                     [128, 16, 32, 32]         --
│    │    └─Conv2d (4)                   [128, 32, 32, 32]         512
│    │    └─BatchNorm2d (5)              [128, 32, 32, 32]         64
│    │    └─Dropout2d (6)                [128, 32, 32, 32]         --
│    │    └─ReLU (7)                     [128, 32, 32, 32]         --
│    └─Conv2d (conv3)                    [128, 32, 32, 32]         9,216
│    └─BatchNorm2d (norm3)               [128, 32, 32, 32]         64
│    └─Dropout2d (drop3)                 [128, 32, 32, 32]         --
│    └─Conv2d (downsampler)              [128, 32, 32, 32]         512
├─SkipBlock (conv2)                      [128, 32, 32, 32]         --
│    └─Sequential (conv1)                [128, 32, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 32, 32, 32]         9,216
│    │    └─BatchNorm2d (1)              [128, 32, 32, 32]         64
│    │    └─Dropout2d (2)                [128, 32, 32, 32]         --
│    │    └─ReLU (3)                     [128, 32, 32, 32]         --
│    └─Sequential (conv2)                [128, 32, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 32, 32, 32]         288
│    │    └─BatchNorm2d (1)              [128, 32, 32, 32]         64
│    │    └─Dropout2d (2)                [128, 32, 32, 32]         --
│    │    └─ReLU (3)                     [128, 32, 32, 32]         --
│    │    └─Conv2d (4)                   [128, 32, 32, 32]         1,024
│    │    └─BatchNorm2d (5)              [128, 32, 32, 32]         64
│    │    └─Dropout2d (6)                [128, 32, 32, 32]         --
│    │    └─ReLU (7)                     [128, 32, 32, 32]         --
│    └─Conv2d (conv3)                    [128, 32, 32, 32]         9,216
│    └─BatchNorm2d (norm3)               [128, 32, 32, 32]         64
│    └─Dropout2d (drop3)                 [128, 32, 32, 32]         --
├─SkipBlock (conv3)                      [128, 64, 32, 32]         --
│    └─Sequential (conv1)                [128, 32, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 32, 32, 32]         9,216
│    │    └─BatchNorm2d (1)              [128, 32, 32, 32]         64
│    │    └─Dropout2d (2)                [128, 32, 32, 32]         --
│    │    └─ReLU (3)                     [128, 32, 32, 32]         --
│    └─Sequential (conv2)                [128, 64, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 32, 32, 32]         288
│    │    └─BatchNorm2d (1)              [128, 32, 32, 32]         64
│    │    └─Dropout2d (2)                [128, 32, 32, 32]         --
│    │    └─ReLU (3)                     [128, 32, 32, 32]         --
│    │    └─Conv2d (4)                   [128, 64, 32, 32]         2,048
│    │    └─BatchNorm2d (5)              [128, 64, 32, 32]         128
│    │    └─Dropout2d (6)                [128, 64, 32, 32]         --
│    │    └─ReLU (7)                     [128, 64, 32, 32]         --
│    └─Conv2d (conv3)                    [128, 64, 32, 32]         36,864
│    └─BatchNorm2d (norm3)               [128, 64, 32, 32]         128
│    └─Dropout2d (drop3)                 [128, 64, 32, 32]         --
│    └─Conv2d (downsampler)              [128, 64, 32, 32]         2,048
├─SkipBlock (conv4)                      [128, 64, 32, 32]         --
│    └─Sequential (conv1)                [128, 64, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 64, 32, 32]         36,864
│    │    └─BatchNorm2d (1)              [128, 64, 32, 32]         128
│    │    └─Dropout2d (2)                [128, 64, 32, 32]         --
│    │    └─ReLU (3)                     [128, 64, 32, 32]         --
│    └─Sequential (conv2)                [128, 64, 32, 32]         --
│    │    └─Conv2d (0)                   [128, 64, 32, 32]         576
│    │    └─BatchNorm2d (1)              [128, 64, 32, 32]         128
│    │    └─Dropout2d (2)                [128, 64, 32, 32]         --
│    │    └─ReLU (3)                     [128, 64, 32, 32]         --
│    │    └─Conv2d (4)                   [128, 64, 32, 32]         4,096
│    │    └─BatchNorm2d (5)              [128, 64, 32, 32]         128
│    │    └─Dropout2d (6)                [128, 64, 32, 32]         --
│    │    └─ReLU (7)                     [128, 64, 32, 32]         --
│    └─Conv2d (conv3)                    [128, 64, 32, 32]         36,864
│    └─BatchNorm2d (norm3)               [128, 64, 32, 32]         128
│    └─Dropout2d (drop3)                 [128, 64, 32, 32]         --
├─Sequential (trans)                     [128, 10]                 --
│    └─AdaptiveAvgPool2d (0)             [128, 64, 1, 1]           --
│    └─Conv2d (1)                        [128, 10, 1, 1]           640
│    └─Flatten (2)                       [128, 10]                 --
│    └─LogSoftmax (3)                    [128, 10]                 --
==========================================================================================
Total params: 163,744
Trainable params: 163,744
Non-trainable params: 0
Total mult-adds (G): 21.20
==========================================================================================
Input size (MB): 1.57
Forward/backward pass size (MB): 1543.51
Params size (MB): 0.65
Estimated Total Size (MB): 1545.74
==========================================================================================
```

## Training log
```
  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.01350, Accuracy = 35.72%, Epoch = 1
Test : Loss = 0.01174, Accuracy = 45.37%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.01150, Accuracy = 46.42%, Epoch = 2
Test : Loss = 0.01023, Accuracy = 51.40%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.01044, Accuracy = 52.20%, Epoch = 3
Test : Loss = 0.01385, Accuracy = 48.17%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00973, Accuracy = 55.58%, Epoch = 4
Test : Loss = 0.00887, Accuracy = 61.17%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00921, Accuracy = 57.99%, Epoch = 5
Test : Loss = 0.00803, Accuracy = 64.68%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00873, Accuracy = 60.63%, Epoch = 6
Test : Loss = 0.01046, Accuracy = 56.59%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00829, Accuracy = 62.47%, Epoch = 7
Test : Loss = 0.00764, Accuracy = 67.90%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00792, Accuracy = 64.48%, Epoch = 8
Test : Loss = 0.00721, Accuracy = 69.30%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00762, Accuracy = 65.65%, Epoch = 9
Test : Loss = 0.00894, Accuracy = 60.94%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00738, Accuracy = 67.25%, Epoch = 10
Test : Loss = 0.00790, Accuracy = 66.51%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00712, Accuracy = 68.06%, Epoch = 11
Test : Loss = 0.00647, Accuracy = 73.85%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00686, Accuracy = 69.37%, Epoch = 12
Test : Loss = 0.00753, Accuracy = 68.70%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00669, Accuracy = 70.09%, Epoch = 13
Test : Loss = 0.00636, Accuracy = 73.25%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00655, Accuracy = 70.74%, Epoch = 14
Test : Loss = 0.00623, Accuracy = 73.89%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00633, Accuracy = 71.67%, Epoch = 15
Test : Loss = 0.00585, Accuracy = 75.73%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00619, Accuracy = 72.29%, Epoch = 16
Test : Loss = 0.00528, Accuracy = 77.83%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00606, Accuracy = 72.90%, Epoch = 17
Test : Loss = 0.00698, Accuracy = 72.92%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00589, Accuracy = 73.82%, Epoch = 18
Test : Loss = 0.00472, Accuracy = 79.30%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00574, Accuracy = 74.43%, Epoch = 19
Test : Loss = 0.00452, Accuracy = 81.06%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00557, Accuracy = 75.14%, Epoch = 20
Test : Loss = 0.00453, Accuracy = 81.34%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00542, Accuracy = 75.88%, Epoch = 21
Test : Loss = 0.00452, Accuracy = 80.78%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00530, Accuracy = 76.44%, Epoch = 22
Test : Loss = 0.00446, Accuracy = 81.28%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00511, Accuracy = 77.25%, Epoch = 23
Test : Loss = 0.00443, Accuracy = 82.20%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00498, Accuracy = 77.85%, Epoch = 24
Test : Loss = 0.00371, Accuracy = 84.43%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00481, Accuracy = 78.45%, Epoch = 25
Test : Loss = 0.00369, Accuracy = 84.55%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00463, Accuracy = 79.50%, Epoch = 26
Test : Loss = 0.00337, Accuracy = 85.45%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00444, Accuracy = 80.27%, Epoch = 27
Test : Loss = 0.00334, Accuracy = 86.01%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00432, Accuracy = 80.64%, Epoch = 28
Test : Loss = 0.00315, Accuracy = 86.50%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00420, Accuracy = 81.35%, Epoch = 29
Test : Loss = 0.00314, Accuracy = 86.83%

  0%|          | 0/391 [00:00<?, ?it/s]
Train: Loss = 0.00418, Accuracy = 81.40%, Epoch = 30
Test : Loss = 0.00309, Accuracy = 87.04%
```

## Learning curves
![lc](https://github.com/woncoh1/era1a9/assets/12987758/525f7a63-70ef-43e7-a189-65b2b338b97c)

## Incorrect predictions
![ip](https://github.com/woncoh1/era1a9/assets/12987758/0d947f3f-4dfb-4d46-853a-63ea9e11668f)

## TODO
- [ ] Sample raw images
- [ ] Receptive field calculations
- [ ] Demo live predictions
