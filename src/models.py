import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self,
        drop:float=0, # probability of dropout
        padm:str='zeros', # padding_mode
    ) -> None:
        super().__init__()
        self.conv0 = convblock( 3, 16, drop, padm)
        self.conv1 = SkipBlock(16, 32, drop, padm)
        self.conv2 = SkipBlock(32, 32, drop, padm)
        self.conv3 = SkipBlock(32, 64, drop, padm)
        self.conv4 = SkipBlock(64, 64, drop, padm)
        self.trans = predblock(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.trans(x)
        return x


class SkipBlock(nn.Module):
    def __init__(self,
        i:int, # in_channels
        o:int, # out_channels
        d:float, # probability of dropout
        pm:str, # padding_mode
    ) -> None:
        super().__init__()
        self.i = i
        self.o = o
        self.conv1 = convblock(i, i, d, pm)
        self.conv2 = dwscblock(i, o, d, pm)
        # Use dliated convolution instead of max pooling or strided convolution
        self.conv3 = nn.Conv2d(o, o, 3, padding=2, padding_mode=pm, dilation=2, bias=False)
        self.norm3 = nn.BatchNorm2d(o)
        self.drop3 = nn.Dropout2d(p=d)
        if i != o: self.downsampler = nn.Conv2d(i, o, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.drop3(out)
        if self.i != self.o: identity = self.downsampler(identity)
        out = out + identity
        out = nn.functional.relu(out)
        return out


def convblock( # Convolution block: 3x3 convolution to extract features
    i:int, # in_channels
    o:int, # out_channels
    d:float, # probability of dropout
    pm:str, # padding_mode
    p:int=1, # padding
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=p, padding_mode=pm, bias=False),
        nn.BatchNorm2d(o),
        nn.Dropout2d(p=d),
        nn.ReLU(),
    )


def dwscblock( # Depthwise separable convolution block
    i:int, # in_channels
    o:int, # out_channels
    d:float, # probability of dropout
    pm:str, # padding_mode
    p:int=1, # padding
) -> nn.Sequential:
    # https://www.youtube.com/watch?v=vVaRhZXovbw
    # https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV1.py#L15-L26
    # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L42-L53
    return nn.Sequential(
        # Depth-wise convolution
        nn.Conv2d(i, i, 3, padding=p, padding_mode=pm, groups=i, bias=False),
        nn.BatchNorm2d(num_features=i),
        nn.Dropout2d(p=d),
        nn.ReLU(),
        # Point-wise convolution
        nn.Conv2d(i, o, 1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=o),
        nn.Dropout2d(p=d),
        nn.ReLU(),
    )


def predblock( # Prediction block = GAP + softmax
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    return nn.Sequential(
        # [-1, i, s, s]
        nn.AdaptiveAvgPool2d(output_size=1),
        # [-1, i, 1, 1]
        nn.Conv2d(i, o, 1, padding=0, bias=False),
        # [-1, o, 1, 1]
        nn.Flatten(),
        # [-1, o]
        nn.LogSoftmax(dim=1), # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
        # [-1, o]
    )