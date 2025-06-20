# Model Implementation
import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.types import FileLike

from encoder_decoder import EncoderDecoder


# TODO share wight matrix between embeddings and linear out?!
class AIAYNModel(nn.Module):
    def __init__(self, vocab_size, layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1) -> None:
        super(AIAYNModel, self).__init__()
        self.embedding_in = nn.Embedding(vocab_size, dimension)
        self.embedding_in_drop = nn.Dropout(dropout)
        self.embedding_out_drop = nn.Dropout(dropout)
        self.embedding_out = nn.Embedding(vocab_size, dimension)
        # todo set to 5000, ok?
        self.pe = positional_encoding(5000, dimension)
        self.encoder_decoder = EncoderDecoder(layers=layers, dimension=dimension, ffn_dim=ffn_dim, heads=heads, dropout=dropout)
        self.linear = nn.Linear(dimension, vocab_size)

    def forward(self, x):
        # TODO add mask
        #add dropout!
        in_encoding = self.embedding_in(x)
        in_encoding = in_encoding + self.pe[:, : x.size(1)].requires_grad_(False)
        in_encoding =self.embedding_in_drop(in_encoding)

        in_decoding = self.embedding_out(x)
        in_decoding = in_decoding + self.pe[:, : x.size(1)].requires_grad_(False)
        in_decoding =self.embedding_out_drop(in_decoding)

        x = self.encoder_decoder(in_encoding, in_decoding)

        return log_softmax(self.linear(x), dim=-1)

    def save(self, path: FileLike) -> None:
        torch.save(self.state_dict(), path)
        pass

    def load(self, path: FileLike) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))



def positional_encoding(max_len, d_model):
    """
    Generate positional encoding for a given maximum length and model dimension.

    Args:
        max_len (int): Maximum length of the sequence.
        d_model (int): Dimension of the model.

    Returns:
        torch.Tensor: Positional encoding tensor of shape (max_len, d_model).
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    #Change dim ? TODO is this correct? probably, we get batched inputs and need to add the batch dim
    pe.unsqueeze_(0)

    return pe


# Test code
def test_model():
    vocab_size = 100
    seq_len = 10
    batch_size = 2
    model = AIAYNModel(vocab_size=vocab_size)
    model.eval()

    # Generate a random batch of token IDs
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Expected: (batch_size, seq_len, vocab_size)
    print(f"Sample output (logits):\n{output[0]}")


if __name__ == "__main__":
    test_model()