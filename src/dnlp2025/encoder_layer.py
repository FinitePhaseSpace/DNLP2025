import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super(EncoderLayer, self).__init__()
        #TODO define layers

        # The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df_f = 2048
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        self.sub2_linear_hidden = nn.Linear(in_features=512, out_features=2048)
        self.relu = nn.ReLU()
        self.sub2_linear_output = nn.Linear(in_features=2048, out_features=512)
        self.sub2_norm = nn.LayerNorm(512)

    def forward(self, x):
        # positional encoding has to happen before!

        # *** Sub Layer 1
        residual = x
        #output dim =512
        #TODO actual multi head attention
        multi_head_out = x
        #add residual + norm
        sublayer1_out = nn.LayerNorm(512)(multi_head_out + residual)

        # *** Sub Layer 2
        residual = sublayer1_out
        # hidden layer
        layer1out = self.sub2_linear_hidden(sublayer1_out)
        layer1relu = self.relu(layer1out)
        # output layer
        layer2out = self.sub2_linear_output(layer1relu)
        #add residual + norm
        normed = self.sub2_norm(layer2out + residual)
        return normed