import convolution_block as cb
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = cb.Convolution_Block(1,16,stride=(1, 2))
        self.block2 = cb.Convolution_Block(16, 32, stride=(1, 2))
        self.block3 = cb.Convolution_Block(32, 64, stride=(1, 2))
        self.block4 = cb.Convolution_Block(64, 128, stride=(1, 2))


    def forward(self, x):

        block_list = []

        block1_out = self.block1(x)
        block_list.append(block1_out)
        
        block2_out = self.block2(block1_out)
        block_list.append(block2_out)

        block3_out = self.block3(block2_out)
        block_list.append(block3_out)

        block4_out = self.block4(block3_out)
        block_list.append(block4_out)

        return block_list