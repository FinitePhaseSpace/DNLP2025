from src.dnlp2025.download_datasets import download_wmt14_de_en, download_wmt14_fr_en

from tokenizers import Tokenizer, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import os

def train_tokenizer(download_func, tokenizer_save_path, source_lang, target_lang):
    dataset = download_func()

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    total_length = (len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])) * 2

    def iterator():
        for split in ["train", "validation", "test"]:
            for example in dataset[split]:
                yield example["translation"][source_lang]
                yield example["translation"][target_lang]

    trainer = BpeTrainer(special_tokens=["<pad>", "<s>", "</s>"])

    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=total_length)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    tokenizer.save(tokenizer_save_path)

    return tokenizer


def main():
    os.makedirs("tokenizers", exist_ok=True)

    print("Training de-en tokenizer...")
    de_en_tokenizer = train_tokenizer(download_wmt14_de_en, "tokenizers/de_en_tokenizer.json", "de", "en")
    print("Finished training de-en tokenizer...")

    # print("Training fr-en tokenizer...")
    # fr_en_tokenizer = train_tokenizer(download_wmt14_fr_en, "tokenizers/fr_en_tokenizer.json", "fr", "en")
    # print("Finished training fr-en tokenizer...")

    print("Testing de-en tokenizer...")
    print(de_en_tokenizer.encode("Guten tag. Wilkommen. Wie geht es Ihnen?").tokens)
    # print("Testing fr-en tokenizer...")
    # print(fr_en_tokenizer.encode("Bonjour, comment ça va?").tokens)

if __name__ == "__main__":
    main()