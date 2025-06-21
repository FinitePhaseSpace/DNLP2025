# Model Implementation
import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.types import FileLike

from src.dnlp2025.encoder_decoder import EncoderDecoder


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

    def forward(self, enc_input, enc_mask, dec_input, dec_mask):
        #add dropout!
        enc_in = self.embedding_in(enc_input)
        enc_in = enc_in + self.pe[:, : enc_input.size(1)].requires_grad_(False)
        enc_in =self.embedding_in_drop(enc_in)

        dec_in = self.embedding_out(dec_input)
        dec_in = dec_in + self.pe[:, : dec_input.size(1)].requires_grad_(False)
        dec_in =self.embedding_out_drop(dec_in)

        x = self.encoder_decoder(enc_in, enc_mask, dec_in, dec_mask)

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
    #manally switch to cuda, model.to(device) wont set this for some reason
    return pe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subseq_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subseq_mask == 0

# Test code
def try_model():
    vocab_size = 10
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

    print(f"Predicted token id:{output[0].argmax(dim=-1)}")


def generate_sequence(model, input_seq, max_new_tokens, vocab_size):
    """
    Generate sequences token by token.

    Args:
        model: transformer model (expects [batch_size, seq_len] input)
        input_seq: Tensor of shape [batch_size, seq_len] → initial sequence
        max_new_tokens: Number of tokens to generate
        vocab_size: Size of vocabulary (optional, for clarity)

    Returns:
        Generated sequences → shape [batch_size, seq_len + max_new_tokens]
    """
    model.eval()

    # Start with the provided input
    generated = input_seq.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Forward pass → get logits
            output = model(generated)

            # Get logits for the *last* position → shape: [batch_size, vocab_size]
            next_token_logits = output[:, -1, :]

            # Greedy decoding → pick the most probable token
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [batch_size, 1]

            # Append the predicted token to the sequence
            generated = torch.cat([generated, next_token], dim=1)  # [batch_size, seq_len + 1]

    return generated


def try_sequence_gen():
    vocab_size = 10
    seq_len = 10
    batch_size = 2
    model = AIAYNModel(vocab_size=vocab_size)
    model.eval()
    # Generate a random batch of token IDs
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"input:{input_tensor}")
    print(f"seq{generate_sequence(model, input_tensor, 5, vocab_size)}")


if __name__ == "__main__":
    print("Tests are broken, they were implemented before correcting the network to use masks and input +output")
    try_model()

    try_sequence_gen()


