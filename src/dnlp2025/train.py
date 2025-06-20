import torch
from torch import nn, optim
from src.model import AIAYNModel
from src.dataset import create_dataloader
from src.download_datasets import download_wmt14_de_en
from tokenizers import Tokenizer
from check import get_device
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
        source_lang="de",  # adjust based on direction
        target_lang="en",
        num_workers=0
    )

    val_loader = create_dataloader(
        dataset_split=dataset["validation"],
        dataset_split_name="validation",
        tokenizer=tokenizer,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=False,
        source_lang="de",
        target_lang="en",
        num_workers=0
    )

    # --- Model ---
    model = AIAYNModel(
        vocab_size=len(tokenizer.get_vocab()),
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=512,
        pad_token_id=tokenizer.token_to_id("<pad>")
    ).to(device)

    # --- Training Setup ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            for key in batch:
                batch[key] = batch[key].to(device)

            outputs = model(
                batch["encoder_input_ids"],
                batch["decoder_input_ids"],
                batch["source_key_padding_mask"],
                batch["target_mask"],
                batch["target_key_padding_mask"]
            )

            loss = criterion(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()
