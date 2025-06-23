import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

import random
import os


class TokenBatchSampler(Sampler[list[int]]):
    """
    Creates batches of indices where each batch aims to not exceed specified
    total token counts for source and target sequences.

    It first sorts examples by length to minimize padding.
    """

    def __init__(
        self,
        dataset,  # Hugging Face dataset with 'source_lengths' and 'target_lengths' columns
        max_src_tokens_per_batch: int,
        max_tgt_tokens_per_batch: int,
        shuffle_batches: bool = True,
        drop_last: bool = False,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.max_src_tokens_per_batch = max_src_tokens_per_batch
        self.max_tgt_tokens_per_batch = max_tgt_tokens_per_batch
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last

        # Get lengths and original indices
        # This might load all lengths into memory. Might be problematic for the french dataset.
        self.lengths_and_indices = []
        print(f"TokenBatchSampler get lengths and indices (dataset size: {len(dataset)})")
        print(f"Extracting source Lengths")

        t0 = time.time()
        source_lengths = torch.tensor([dataset[i]["source_length"] for i in range(len(dataset))])

        print(f"Extracting target Lengths")
        target_lengths = torch.tensor([dataset[i]["target_length"] for i in range(len(dataset))])

        mask = (source_lengths != 0) & (target_lengths != 0)
        print(f"Extracting indices")
        indices = torch.arange(len(dataset))[mask]
        print(f"Adding indices and src/tar lengths to map")
        self.lengths_and_indices = [
            {"id": int(i), "source_length": int(src), "target_length": int(tgt)}
            for i, src, tgt in zip(indices, source_lengths[mask], target_lengths[mask])
        ]

        print(f"Done computing len_ind")

        # Sort by the specified key (e.g., source length)
        print(f"Sorting")
        self.lengths_and_indices.sort(key=lambda x: x["source_length"])
        print(f"Creating Batches")
        self.batches = self._create_batches()

        del lengths_and_indices

        print(f"Batches created")
        t1 = time.time()
        print(f"TokenBatchSampler took {t1 - t0} seconds")

    def _create_batches(self) -> list[list[int]]:
        batches = []
        current_batch_indices = []
        current_batch_src_tokens = 0
        current_batch_tgt_tokens = 0

        for item_info in self.lengths_and_indices:
            idx = item_info["id"]
            src_len = item_info["source_length"]
            tgt_len = item_info["target_length"]

            # Check if this sentence can be added to the current batch
            can_add = True
            if current_batch_src_tokens + src_len > self.max_src_tokens_per_batch:
                can_add = False
            if current_batch_tgt_tokens + tgt_len > self.max_tgt_tokens_per_batch:
                can_add = False

            # If adding this sentence exceeds limits, finalize current batch and start a new one
            if not can_add and len(current_batch_indices) > 0:
                batches.append(current_batch_indices)
                current_batch_indices = []
                current_batch_src_tokens = 0
                current_batch_tgt_tokens = 0

            # Add to (potentially new) current batch, ensuring even single large sentences form a batch
            current_batch_indices.append(idx)
            current_batch_src_tokens += src_len
            current_batch_tgt_tokens += tgt_len

            # Eagerly finalize if limits are met after adding the current sentence
            # This handles the case where the very first sentence in a new batch might already be large
            if (
                current_batch_src_tokens >= self.max_src_tokens_per_batch
                or current_batch_tgt_tokens >= self.max_tgt_tokens_per_batch
            ):
                batches.append(current_batch_indices)
                current_batch_indices = []
                current_batch_src_tokens = 0
                current_batch_tgt_tokens = 0

        # Add the last batch if it has any items and drop_last is False
        if len(current_batch_indices) > 0:
            if not self.drop_last or (
                current_batch_src_tokens > 0 and current_batch_tgt_tokens > 0
            ):  # Ensure we don't add empty batches
                batches.append(current_batch_indices)

        return batches

    def __iter__(self):
        if self.shuffle_batches:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class TranslationBatchCollator:
    """
    Collator for the translation dataset. Used to pad sequences and create masks.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Assumes each item in 'batch' is a dictionary:
        # {'encoder_input_ids': tensor, 'decoder_input_ids': tensor, 'labels': tensor}
        encoder_input_ids_list = [item["encoder_input_ids"] for item in batch]
        decoder_input_ids_list = [item["decoder_input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        encoder_input_ids_padded = pad_sequence(
            encoder_input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        decoder_input_ids_padded = pad_sequence(
            decoder_input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = pad_sequence(
            labels_list, batch_first=True, padding_value=self.pad_token_id
        )

        # --- MASK CREATION ---
        src_key_padding_mask = encoder_input_ids_padded == self.pad_token_id
        tgt_key_padding_mask = decoder_input_ids_padded == self.pad_token_id
        tgt_len = decoder_input_ids_padded.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(  # Causal mask
            tgt_len, device=decoder_input_ids_padded.device
        )
        return {
            "encoder_input_ids": encoder_input_ids_padded,
            "decoder_input_ids": decoder_input_ids_padded,
            "labels": labels_padded,
            #TODO we need to change the shape here?
            #TODO we need to change the shape here?
            "source_key_padding_mask": src_key_padding_mask,  # For encoder
            "target_key_padding_mask": tgt_key_padding_mask,  # For decoder self-attention
            "target_mask": tgt_mask,  # For decoder self-attention (causal)
        }


def create_dataloader(
    dataset_split,
    dataset_split_name,
    tokenizer,
    max_tokens_per_batch=25000,
    shuffle=True,
    source_lang="en",
    target_lang="de",
    num_workers=0,
    max_seq_len=128,
):
    """
    Create a DataLoader for the translation dataset.

    Args:
        dataset_split (Dataset): The dataset split to use.
        dataset_split_name (str): Name of the dataset split (e.g., 'train', 'validation', 'test').
        tokenizer: The tokenizer to use for encoding text.
        max_tokens_per_batch (int): Maximum number of tokens per batch.
        shuffle (bool): Whether to shuffle the dataset.
        source_lang (str): Source language code. "en" for English, "de" for German, etc.
        target_lang (str): Target language code. "de" for German, "en" for English, etc.
        num_workers (int): Number of worker processes to use for data loading. If greater than zero, then tokenization will
            also be parallzelized with all available cpu cores.
    """

    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>")
    pad_token_id = tokenizer.token_to_id("<pad>")

    if None in [bos_token_id, eos_token_id, pad_token_id]:
        raise ValueError("Tokenizer must have <s>, </s>, and <pad> tokens defined.")

    def preprocess_fn(examples):
        source_texts = [ex[source_lang] for ex in examples["translation"]]
        target_texts = [ex[target_lang] for ex in examples["translation"]]

        # <s> and </s> tokens are added by the tokenizer's post-processor
        source_encodings = tokenizer.encode_batch(source_texts)
        target_encodings = tokenizer.encode_batch(target_texts)

        encoder_input_ids = [enc.ids for enc in source_encodings]
        decoder_input_ids = [enc.ids[:-1] for enc in target_encodings]
        labels = [enc.ids[1:] for enc in target_encodings]

        # Total number of tokens to be used
        source_lengths = [len(ids) for ids in encoder_input_ids]
        target_lengths = [len(ids) for ids in decoder_input_ids]

        return {
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "source_length": source_lengths,
            "target_length": target_lengths,
        }

    num_proc = os.cpu_count() if num_workers > 0 else 1
    print(f"dataset.py: Create tokenized Dataset")
    tokenized_dataset = dataset_split.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset_split.column_names,
        num_proc=num_proc,
        desc="Tokenizing dataset",
    )
    print(f"dataset.py: Tokenization Done")

    # Filter out examples longer than max_seq_len
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: x["source_length"] <= max_seq_len and x["target_length"] <= max_seq_len
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=[
            "encoder_input_ids",
            "decoder_input_ids",
            "labels",
            "source_length",
            "target_length",
        ],
    )

    token_batch_sampler = TokenBatchSampler(
        tokenized_dataset,
        max_src_tokens_per_batch=max_tokens_per_batch,
        max_tgt_tokens_per_batch=max_tokens_per_batch,
        shuffle_batches=shuffle,
        drop_last=(dataset_split_name == "train"),  # Drop last incomplete batch
    )

    collator = TranslationBatchCollator(pad_token_id=pad_token_id)

    print(f"Creating DataLoader using Batch TokenBatchSampler and TranslationBatchCollator")
    data_loader = DataLoader(
        tokenized_dataset,
        batch_sampler=token_batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
    )
    print(f"Done")

    Path("dataloader/").mkdir(parents=True, exist_ok=True)
    data_loader_path = "dataloader/" + source_lang + "_" + target_lang + ".pth"
    print(f"Saving DataLoader: {data_loader_path}")
    torch.save(data_loader, data_loader_path)
    return data_loader
