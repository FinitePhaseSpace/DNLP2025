import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    # def __init__(self, layers=6, m_dim=512, ffn_dim=2048, heads=8, dropout=0.1) -> None: #nr of layers is misleading, decoder layer represents only one transformer layer
    def __init__(self, layers=6, m_dim=512, ffn_dim=2048, heads=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # MHA1 - decoder self attention
        self.mha1_key_projection = nn.Linear(m_dim, m_dim)
        self.mha1_query_projection = nn.Linear(m_dim, m_dim)
        self.mha1_value_projection = nn.Linear(m_dim, m_dim)
        self.mha1 = nn.MultiheadAttention(
            embed_dim=m_dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.mha1_norm = nn.LayerNorm(m_dim)
        # MHA2 - encoder-decoder attention
        self.mha2_key_projection = nn.Linear(m_dim, m_dim)
        self.mha2_query_projection = nn.Linear(m_dim, m_dim)
        self.mha2_value_projection = nn.Linear(m_dim, m_dim)
        self.mha2 = nn.MultiheadAttention(
            embed_dim=m_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.mha2_norm = nn.LayerNorm(m_dim)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(m_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, m_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(m_dim)
        # self.ffn_hidden = nn.Linear(in_features=m_dim, out_features=ffn_dim)
        # self.relu = nn.ReLU()
        # self.ffn_output = nn.Linear(in_features=ffn_dim, out_features=m_dim)
        # self.ffn_norm = nn.LayerNorm(m_dim)
        # self.ffn_dropout = nn.Dropout(dropout)

    # def forward(self, x, encoder_out, mask_encoder=None, mask_decoder=None):
    def forward(
        self, x, encoder_out, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask
    ):
        # In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
        # and the memory keys and values come from the output of the encoder.
        # *** MHA 1
        residual = x
        # output dim =512
        key = self.mha1_key_projection(x)
        query = self.mha1_query_projection(x)
        value = self.mha1_value_projection(x)
        # multi_head_out, mha1_weights = self.mha1(query, key, value, attn_mask=mask_decoder)
        multi_head_out, mha1_weights = self.mha1(
            query, key, value, attn_mask=tgt_mask, 
            key_padding_mask=tgt_key_padding_mask
        )
        # add residual + norm
        mha1_out = self.mha1_norm(multi_head_out + residual)

        # *** MHA 2
        residual = mha1_out
        # output dim =512
        # use encoder output for key and values
        key = self.mha2_key_projection(encoder_out)
        query = self.mha2_query_projection(mha1_out)
        value = self.mha2_value_projection(encoder_out)
        # multi_head_out, mha2_weights = self.mha2(query, key, value, attn_mask=mask_encoder)
        multi_head_out, mha2_weights = self.mha2(
            query, key, value, key_padding_mask=memory_key_padding_mask
        )
        # add residual + norm
        mha2_out = self.mha2_norm(multi_head_out + residual)

        # *** FFN
        residual = mha2_out
        layer2out = self.ffn(mha2_out)
        normed = self.ffn_norm(layer2out + residual)
        # # hidden layer
        # layer1out = self.ffn_hidden(mha2_out)
        # layer1relu = self.relu(layer1out)
        # # output layer
        # layer2out = self.ffn_output(layer1relu)
        # layer2dropout = self.ffn_dropout(layer2out)
        # # add residual + norm
        # normed = self.ffn_norm(layer2dropout + residual)

        return normed
