import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        # The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df_f = 2048
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        #MHA
        self.key_projection = nn.Linear(512, 512)
        self.query_projection = nn.Linear(512, 512)
        self.value_projection = nn.Linear(512, 512)
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=dropout)
        self.mha_norm = nn.LayerNorm(512)
        #FFN
        self.sub2_linear_hidden = nn.Linear(in_features=512, out_features=2048)
        self.relu = nn.ReLU()
        self.sub2_linear_output = nn.Linear(in_features=2048, out_features=512)
        self.sub2_norm = nn.LayerNorm(512)
        self.sub2_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # positional encoding has to happen before!
        # *** Sub Layer 1
        residual = x
        #output dim =512
        key = self.key_projection(x)
        query = self.query_projection(x)
        value = self.value_projection(x)
        multi_head_out = self.mha(query, key, value, attn_mask=mask)
        #add residual + norm
        sublayer1_out = self.mha_norm(multi_head_out + residual)

        # *** Sub Layer 2
        residual = sublayer1_out
        # hidden layer
        layer1out = self.sub2_linear_hidden(sublayer1_out)
        layer1relu = self.relu(layer1out)
        # output layer
        layer2out = self.sub2_linear_output(layer1relu)
        layer2dropout = self.sub2_dropout(layer2out)
        #add residual + norm
        normed = self.sub2_norm(layer2dropout + residual)
        return normed