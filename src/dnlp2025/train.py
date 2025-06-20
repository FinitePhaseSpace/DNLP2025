import torch
from torch import nn, optim
from src.dnlp2025.model import AIAYNModel
from src.dnlp2025.dataset import create_dataloader
from src.dnlp2025.download_datasets import download_wmt14_de_en
from tokenizers import Tokenizer
from src.dnlp2025.check import get_device
import os


def train():
    # --- Config ---
    device = get_device()
    epochs = 10
    max_tokens_per_batch = 20000
    learning_rate = 3e-4
    save_path = "checkpoints/aiayn.pt"
    os.makedirs("checkpoints", exist_ok=True)

    # --- Tokenizer ---
    tokenizer = Tokenizer.from_file("tokenizers/de_en_tokenizer.json")

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

    vocab_size = len(tokenizer.get_vocab())
    pad_token_id = tokenizer.token_to_id("<pad>")
    output_projection = nn.Linear(512, vocab_size).to(device)

    # --- Training Setup ---
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            for key in batch:
                batch[key] = batch[key].to(device)

            # Forward pass
            encoder_input = batch["encoder_input_ids"]
            decoder_input = batch["decoder_input_ids"]
            labels = batch["labels"]

            encoder_out = encoder_input
            decoder_out = decoder_input

            transformer_out = model(encoder_out, decoder_out)  # shape: [B, T, D]
            logits = output_projection(transformer_out)  # shape: [B, T, vocab_size]

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
