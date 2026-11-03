import torch.nn as nn
import torch
import convolution_block as cb
import encoder as enc
import decoder as dec

class NewGroundSegmentation(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = enc.Encoder()
        self.decoder = dec.Decoder()


    def forward(self, x):

        output_list = self.encoder(x)
        final_result = self.decoder(output_list)

        return final_result


