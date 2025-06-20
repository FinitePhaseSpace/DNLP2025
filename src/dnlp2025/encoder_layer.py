import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self,m_dim=512, ffn_dim=2048, heads=8, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()
        # The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df_f = 2048
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        #MHA
        self.key_projection = nn.Linear(m_dim, m_dim)
        self.query_projection = nn.Linear(m_dim, m_dim)
        self.value_projection = nn.Linear(m_dim, m_dim)
        self.mha = nn.MultiheadAttention(embed_dim=m_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(m_dim)
        #FFN
        self.sub2_linear_hidden = nn.Linear(in_features=m_dim, out_features=ffn_dim)
        self.relu = nn.ReLU()
        self.sub2_linear_output = nn.Linear(in_features=ffn_dim, out_features=m_dim)
        self.sub2_norm = nn.LayerNorm(m_dim)
        self.sub2_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # positional encoding has to happen before!
        # *** Sub Layer 1
        residual = x
        #output dim =512
        key = self.key_projection(x)
        query = self.query_projection(x)
        value = self.value_projection(x)
        multi_head_out, mha_weights = self.mha(query, key, value, attn_mask=mask)
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