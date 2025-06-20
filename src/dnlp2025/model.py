# Model Implementation

import torch
import torch.nn as nn
from torch.types import FileLike

from dnlp2025.encoder_decoder import EncoderDecoder

class AIAYNModel(nn.Module):
    def __init__(self, layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1) -> None:
        super(AIAYNModel, self).__init__()
        self.encoder_decoder = EncoderDecoder(layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1)

    def forward(self, in_encoder, in_decoder):
        # TODO add mask
        self.encoder_decoder(in_encoder, in_decoder)

        #TODO define forward pass
        #TODO remove placeholder linear impl
        return self.linear(x)

    def save(self, path: FileLike) -> None:
        torch.save(self.state_dict(), path)
        pass

    def load(self, path: FileLike) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))

