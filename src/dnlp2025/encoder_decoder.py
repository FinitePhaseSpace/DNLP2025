import torch
import torch.nn as nn

from dnlp2025.decoder_layer import DecoderLayer
from dnlp2025.encoder_layer import EncoderLayer

class EncoderDecoder(nn.Module):
    def __init__(self, layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.ModuleList([
            EncoderLayer(m_dim=dimension, ffn_dim=ffn_dim, heads=heads, dropout=dropout) for i in range(layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(m_dim=dimension, ffn_dim=ffn_dim, heads=heads, dropout=dropout) for i in range(layers)
        ])
        # TODO linear + softmax
        pass

    def forward(self, in_encoder, in_decoder):
        for encoder in self.encoder:
            #TODO add mask
            in_encoder = self.encoder(in_encoder)

        for decoder in self.decoder:
            # TODO add mask
            in_decoder = self.decoder(x=in_decoder, encoder_out=in_encoder)

        return in_decoder