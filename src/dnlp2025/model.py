# Model Implementation
import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.types import FileLike

from src.dnlp2025.encoder_decoder import EncoderDecoder


# TODO share wight matrix between embeddings and linear out?!
class AIAYNModel(nn.Module):
    def __init__(
        self, vocab_size, layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1, max_seq_len=4000
    ) -> None:
        super(AIAYNModel, self).__init__()
        self.embedding_in = nn.Embedding(vocab_size, dimension)
        self.embedding_out = nn.Embedding(vocab_size, dimension)
        self.embedding_in_drop = nn.Dropout(dropout)
        self.embedding_out_drop = nn.Dropout(dropout)
        # self.scale = math.sqrt(dimension)
        # todo set to 5000, ok? yes
        self.pe = positional_encoding(max_seq_len, dimension)
        self.encoder_decoder = EncoderDecoder(
            layers=layers,
            dimension=dimension,
            ffn_dim=ffn_dim,
            heads=heads,
            dropout=dropout,
        )
        self.linear = nn.Linear(dimension, vocab_size)

    # def forward(self, enc_input, enc_mask, dec_input, dec_mask):
    def forward(
        self,
        enc_input,
        enc_mask,
        dec_input,
        dec_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    ):
        # add dropout!
        enc_in = self.embedding_in(enc_input)
        # enc_in = enc_in + self.pe[:, : enc_input.size(1)].requires_grad_(False)
        enc_in = enc_in + self.pe[:, : enc_input.size(1)].detach()
        enc_in = self.embedding_in_drop(enc_in)

        dec_in = self.embedding_out(dec_input)
        # dec_in = dec_in + self.pe[:, : dec_input.size(1)].requires_grad_(False)
        dec_in = dec_in + self.pe[:, : dec_input.size(1)].detach()
        dec_in = self.embedding_out_drop(dec_in)

        # x = self.encoder_decoder(enc_in, enc_mask, dec_in, dec_mask)
        x = self.encoder_decoder(
            in_encoder=enc_in,
            mask_encoder=enc_mask,
            in_decoder=dec_in,
            mask_decoder=dec_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.linear(x)

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
    # Change dim ? TODO is this correct? probably, we get batched inputs and need to add the batch dim
    pe.unsqueeze_(0)
    # manally switch to cuda, model.to(device) wont set this for some reason
    return pe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def subsequent_mask(size: int) -> torch.Tensor:
    """Return a [T, T] mask where subsequent positions are masked (upper triangular)."""
    return torch.triu(torch.ones(size, size), diagonal=1).bool()  # [T, T]


# Test code
def try_model():
    vocab_size = 10
    seq_len = 10
    batch_size = 2
    model = AIAYNModel(vocab_size=vocab_size)
    model.eval()

    # Generate a random batch of token IDs
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))
    enc_mask = None  # not used explicitly
    dec_input = input_tensor.clone()
    dec_mask = subsequent_mask(seq_len).to(input_tensor.device)
    tgt_key_padding_mask = dec_input == 0
    memory_key_padding_mask = input_tensor == 0

    with torch.no_grad():
        output = model(
            input_tensor,
            enc_mask,
            dec_input,
            dec_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )

    # with torch.no_grad():
    #     output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(
        f"Output shape: {output.shape}"
    )  # Expected: (batch_size, seq_len, vocab_size)
    print(f"Sample output (logits):\n{output[0]}")

    print(f"Predicted token id:{output[0].argmax(dim=-1)}")


def generate_sequence(model, encoder_input, max_new_tokens, vocab_size, bos_token_id=2, eos_token_id=1, pad_token_id=0):
    """
    Generate sequences token by token using encoder-decoder architecture.

    Args:
        model: transformer model (encoder-decoder)
        encoder_input: Tensor of shape [batch_size, src_seq_len] → source sequence
        max_new_tokens: Number of tokens to generate
        vocab_size: Size of vocabulary (optional, for clarity)
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID

    Returns:
        Generated sequences → shape [batch_size, generated_seq_len]
    """
    model.eval()
    batch_size = encoder_input.size(0)
    device = encoder_input.device

    # Start decoder with BOS token
    decoder_input = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Create masks
        dec_seq_len = decoder_input.size(1)
        dec_mask = subsequent_mask(dec_seq_len).to(device)
        tgt_key_padding_mask = decoder_input == pad_token_id
        memory_key_padding_mask = encoder_input == pad_token_id

        with torch.no_grad():
            output = model(
                encoder_input,
                None,  # encoder mask
                decoder_input,
                dec_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
            
            # Get logits for the last position
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append the predicted token
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Stop if all sequences have generated EOS token
            if (next_token == eos_token_id).all():
                break

    return decoder_input


def try_sequence_gen():
    vocab_size = 10
    seq_len = 10
    batch_size = 2
    model = AIAYNModel(vocab_size=vocab_size)
    model.eval()
    # Generate a random batch of token IDs
    encoder_input = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid pad token
    print(f"encoder input:{encoder_input}")
    generated = generate_sequence(model, encoder_input, 5, vocab_size, bos_token_id=2, eos_token_id=1, pad_token_id=0)
    print(f"generated sequence:{generated}")


if __name__ == "__main__":
    print(
        "Tests are broken, they were implemented before correcting the network to use masks and input +output"
    )
    try_model()

    try_sequence_gen()
