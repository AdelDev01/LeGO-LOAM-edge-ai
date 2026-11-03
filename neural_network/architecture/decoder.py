import convolution_block as cb
import torch.nn as nn
import torch


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 0))
        self.conv1 = cb.Convolution_Block(128,64)

        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1))
        self.conv2 = cb.Convolution_Block(64, 32)

        self.upsample3 = nn.ConvTranspose2d(32, 16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1))
        self.conv3 = cb.Convolution_Block(32, 16)

        self.upsample4 = nn.ConvTranspose2d(16, 16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1))
        self.conv4 = cb.Convolution_Block(16, 16)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=(1))


    def forward(self, encoder_outputs):

        block1_out, block2_out, block3_out, block4_out = encoder_outputs

        up1 = self.upsample1(block4_out)
        skip1 = torch.cat([up1, block3_out], dim=1)
        dec1 = self.conv1(skip1)

        up2 = self.upsample2(dec1)
        skip2 = torch.cat([up2, block2_out], dim=1)
        dec2 = self.conv2(skip2)

        up3 = self.upsample3(dec2)
        skip3 = torch.cat([up3, block1_out], dim=1)
        dec3 = self.conv3(skip3)

        up4 = self.upsample4(dec3)
        dec4 = self.conv4(up4)

        final_output = self.final_conv(dec4)

        return final_output

