import torch
import torch.nn as nn

from src.dnlp2025.decoder_layer import DecoderLayer
from src.dnlp2025.encoder_layer import EncoderLayer


class EncoderDecoder(nn.Module):
    def __init__(
        self, layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1
    ) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    m_dim=dimension, ffn_dim=ffn_dim, heads=heads, dropout=dropout
                )
                for i in range(layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    m_dim=dimension, ffn_dim=ffn_dim, heads=heads, dropout=dropout
                )
                for i in range(layers)
            ]
        )
        pass

    # def forward(self, in_encoder, mask_encoder, in_decoder, mask_decoder):
    def forward(
        self,
        in_encoder,
        mask_encoder,
        in_decoder,
        mask_decoder,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    ):

        for encoder in self.encoder:
            in_encoder = encoder(in_encoder, mask_encoder)

        for decoder in self.decoder:
            # in_decoder = decoder(x=in_decoder, encoder_out=in_encoder, mask_encoder=mask_encoder, mask_decoder=mask_decoder)
            in_decoder = decoder(
                x=in_decoder,
                encoder_out=in_encoder,
                tgt_mask=mask_decoder,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return in_decoder
