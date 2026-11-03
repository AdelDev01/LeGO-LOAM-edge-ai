import torch.nn as nn

class Convolution_Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        conv_out = self.conv(x)
        norm = self.bn(conv_out)
        relu_out = self.relu(norm)
        return relu_out
        