import os
import torch
from torch import nn, optim

# from src.dnlp2025.model import AIAYNModel
# from src.dnlp2025.dataset import create_dataloader
# from src.dnlp2025.download_datasets import download_wmt14_de_en
from model import AIAYNModel
from dataset import create_dataloader
from download_datasets import download_wmt14_de_en
from tokenizers import Tokenizer
from check import get_device


class TrainState:
    """Track steps, examples, and tokens processed"""

    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.accum_step = 0
        self.samples = 0
        self.tokens = 0

    def to_dict(self):
        return self.__dict__

    def load_state(self, state_dict):
        self.__dict__.update(state_dict)


def save_checkpoint(model, optimizer, train_state, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_state": train_state.to_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, train_state, path):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_state.load_state(checkpoint["train_state"])
        return True
    return False


def train():
    # --- Config ---
    device = get_device()
    epochs = 10
    max_tokens_per_batch = 20000
    learning_rate = 3e-4
    checkpoint_path = "checkpoints/aiayn.pt"
    os.makedirs("checkpoints", exist_ok=True)

    # --- Tokenizer ---
    tokenizer = Tokenizer.from_file("tokenizers/de_en_tokenizer.json")
    vocab_size = len(tokenizer.get_vocab())
    pad_token_id = tokenizer.token_to_id("<pad>")

    # --- Data ---
    dataset = download_wmt14_de_en()
    train_loader = create_dataloader(
        dataset_split=dataset["train"],
        dataset_split_name="train",
        tokenizer=tokenizer,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=True,
        source_lang="de",
        target_lang="en",
        num_workers=0,
    )

    val_loader = create_dataloader(
        dataset_split=dataset["validation"],
        dataset_split_name="validation",
        tokenizer=tokenizer,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=False,
        source_lang="de",
        target_lang="en",
        num_workers=0,
    )

    # --- Model ---
    model = AIAYNModel(layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1).to(
        device
    )

    output_projection = nn.Linear(512, vocab_size).to(device)

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # --- Train State ---
    train_state = TrainState()

    # --- Resume if possible ---
    load_checkpoint(model, optimizer, train_state, checkpoint_path)

    # --- Training Loop ---
    for epoch in range(train_state.epoch, epochs):
        print(f"\nEpoch {epoch + 1}")
        train_state.epoch = epoch
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            for key in batch:
                batch[key] = batch[key].to(device)

            encoder_input = batch["encoder_input_ids"]
            decoder_input = batch["decoder_input_ids"]
            labels = batch["labels"]

            # Forward
            output = model(encoder_input, decoder_input)
            logits = output_projection(output)  # [B, T, vocab_size]

            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Update train state
            train_state.step += 1
            train_state.samples += encoder_input.size(0)
            train_state.tokens += encoder_input.numel()

            total_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"[Step {i}] Loss: {loss.item():.4f} | "
                    f"Tokens: {train_state.tokens} | Samples: {train_state.samples}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, train_state, checkpoint_path)


if __name__ == "__main__":
    train()
