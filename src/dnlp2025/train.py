import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# from src.dnlp2025.model import AIAYNModel
# from src.dnlp2025.dataset import create_dataloader
# from src.dnlp2025.download_datasets import download_wmt14_de_en
from src.dnlp2025.model import AIAYNModel, subsequent_mask
from src.dnlp2025.dataset import create_dataloader
from src.dnlp2025.download_datasets import download_wmt14_de_en
from src.dnlp2025.losses import LabelSmoothingLoss
from tokenizers import Tokenizer
from src.dnlp2025.check import get_device


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
    print(f"Trying to load checkpoint from {path}")
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_state.load_state(checkpoint["train_state"])
        return True
    return False

def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (model_size ** -0.5 * min(step ** -0.5, step * warmup ** -1.5))


def train(model_size=512, factor=1.0, warmup=4000):
    # --- Config ---
    device = get_device()
    epochs = 10
    max_tokens_per_batch = 25000
    gradient_accumulation_steps = 5  # Accumulate over 5 smaller batches
    learning_rate = 1.0 #3e-4
    checkpoint_dir = os.path.join(os.path.expanduser("~"), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "aiayn.pt")
    dataset_percentage = 1


    # --- Tokenizer ---
    print(f"Loading Tokenizer")
    # tokenizer = Tokenizer.from_file("tokenizers/de_en_tokenizer.json")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tokenizer_path = os.path.join(project_root, "tokenizers", "de_en_tokenizer.json")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    vocab_size = len(tokenizer.get_vocab())
    pad_token_id = tokenizer.token_to_id("<pad>")

    train_loader = None
    max_seq_len = 128

    if os.path.isfile("dataloader/de_en.pth"):
        print("\n>>>>>     Loading Dataloader from File!!!     <<<<<<\n")
        train_loader = torch.load("dataloader/de_en.pth", weights_only=False)

    else:
        
        # --- Data ---
        print(f"Loading and Tokenizing Dataset")
        
        dataset = download_wmt14_de_en()
        full_dataset = dataset["train"]
        subset_size = int(len(full_dataset) * dataset_percentage)
        
        # Optional: shuffle for random subset
        full_dataset = full_dataset.shuffle(seed=42)
        subset_size = int(len(full_dataset) * dataset_percentage)
        subset_dataset = full_dataset.select(range(subset_size))
        
        t0 = time.time()
        train_loader = create_dataloader(
            dataset_split=subset_dataset,#dataset["train"],
            dataset_split_name="train",
            tokenizer=tokenizer,
            max_tokens_per_batch=max_tokens_per_batch,
            shuffle=True,
            source_lang="de",
            target_lang="en",
            num_workers=0,
            max_seq_len=max_seq_len
        )
        
        # Save max_seq_len for future use
        os.makedirs("dataloader", exist_ok=True)
        with open("dataloader/max_seq_len.txt", "w") as f:
            f.write(str(max_seq_len))

        t1 = time.time()
        print(f"Total df time: {t1 - t0} seconds")
        print(f"Finished and Tokenizing Dataset: Train")


        #print(f"Loading and Tokenizing Dataset: Validation")
        #val_loader = create_dataloader(
        #    dataset_split=dataset["validation"],
        #    dataset_split_name="validation",
        #    tokenizer=tokenizer,
        #    max_tokens_per_batch=max_tokens_per_batch,
        #    shuffle=False,
        #    source_lang="de",
        #    target_lang="en",
        #    num_workers=0,
        #)
        #print(f"Finished and Tokenizing Dataset: Validation")

    # --- Model ---
    model = AIAYNModel(vocab_size=vocab_size, layers=6, dimension=512, ffn_dim=2048, heads=8, dropout=0.1, max_seq_len=max_seq_len).to(
        device
    )

    # output_projection = nn.Linear(512, vocab_size).to(device)

    # --- Optimizer ---
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step, model_size, factor, warmup))


    # --- Loss ---
    label_smoothing = 0.1  # You can tweak this value
    criterion = LabelSmoothingLoss(
        label_smoothing=label_smoothing,
        vocab_size=vocab_size,
        ignore_index=pad_token_id
    )
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
            # target_mask = batch["target_mask"]
            labels = batch["labels"]
            # src_mask = batch["source_key_padding_mask"]

            # Forward
            #TODO add masks (currently they dont work (wrong shape) and i think the values in there are wrong! (debug))
            # TODO target mask is also wrong! it should be: mask all the padding, mask all subsequent tokens
            pad_mask_enc = encoder_input == pad_token_id
            pad_mask_dec = decoder_input == pad_token_id

            # Create subsequent mask [1, T, T] → broadcastable in MultiheadAttention
            seq_len = decoder_input.size(1)
            T = decoder_input.size(1)
            subsequent = subsequent_mask(T).to(device)  # shape [T, T]


            
            output = model(
                encoder_input,
                None,  # enc_mask if needed
                decoder_input,
                subsequent,
                tgt_key_padding_mask=pad_mask_dec,
                memory_key_padding_mask=pad_mask_enc,
            )
            #output = model(encoder_input, None, decoder_input, None)
            # --> the model already outputs [B, T, vocab_size] TODO verify
            #logits = output_projection(output)  # [B, T, vocab_size]

            # Compute loss and scale by accumulation steps
            loss = criterion(output, labels)
            loss = loss / gradient_accumulation_steps  # Scale loss
            loss.backward()

            # Only update weights every gradient_accumulation_steps
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Clear VRAM cache periodically
                if (i + 1) % (gradient_accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache()

            # Update train state
            train_state.step += 1
            train_state.samples += encoder_input.size(0)
            train_state.tokens += encoder_input.numel()

            total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging

            if i % 50 == 0:
                print(
                    f"[Step {i}] Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                    f"Tokens: {train_state.tokens} | Samples: {train_state.samples}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, train_state, checkpoint_path)


if __name__ == "__main__":
    os.chdir("../../") # hack so direct file execution works
    train()
