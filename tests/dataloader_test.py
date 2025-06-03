import unittest

from unittest.mock import MagicMock
from dnlp2025.dataset import create_dataloader
from datasets import Dataset
from tokenizers import Tokenizer

import torch

def get_mock_split_small():
    return Dataset.from_list([
        { "translation": { "de": "Guten morgen.", "en": "Good morning." }},
        { "translation": { "de": "Guten tag.", "en": "Good day." }},
        { "translation": { "de": "Wilkommen.", "en": "Welcome." }},
        { "translation": { "de": "Wie geht es Ihnen?", "en": "How are you?" }}
    ])

def get_mock_split_big():
    return Dataset.from_list([
        { "translation": { "de": "Guten morgen.", "en": "Good morning." }},
        { "translation": { "de": "Guten tag.", "en": "Good day." }},
        { "translation": { "de": "Wilkommen.", "en": "Welcome." }},
        { "translation": { "de": "Wie geht es Ihnen?", "en": "How are you?" }},
        { "translation": { "de": "Das ist ein Buch.", "en": "That is a book." }},
        { "translation": { "de": "Ich denke, dass ich einen Kaffee brauche.", "en": "I think I need a coffee." }},
        { "translation": { "de": "Ich bin ein bisschen m dde.", "en": "I am a bit tired." }},
        { "translation": { "de": "Ich verstehe das nicht.", "en": "I don t understand this." }},
        { "translation": { "de": "Ich brauche noch ein bisschen mehr Zeit.", "en": "I need a little more time." }},
        { "translation": { "de": "Ich denke, dass ich das verstehe.", "en": "I think I understand this." }},
    ])


class TestCreateDataLoader(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer.from_file('tokenizers/de_en_tokenizer.json')

        if self.tokenizer is None:
            raise ValueError("Tokenizer not found. Please ensure the tokenizer is trained and saved correctly.")
        
        self.source_lang = 'en'
        self.target_lang = 'de'
        self.shuffle = True

    def test_create_dataloader(self):
        dataset_split = get_mock_split_small()
        dataset_split_name = 'train'

        max_tokens_per_batch = 250

        dataloader = create_dataloader(dataset_split, dataset_split_name, self.tokenizer, max_tokens_per_batch, self.shuffle, self.source_lang, self.target_lang)

        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        self.assertGreater(len(dataloader.dataset), 0, "Dataloader should not be empty")

        for batch in dataloader:
            self.assertIn("encoder_input_ids", batch)
            self.assertEqual(batch["encoder_input_ids"].shape[0], 4)
    
    def test_dataloader_multiple_batches(self):
        dataset_split = get_mock_split_big()
        
        # Test split to prevent removing last batch
        dataset_split_name = 'test'

        # Super small tokens per batch to ensure there's more than one batch
        max_tokens_per_batch = 20

        dataloader = create_dataloader(dataset_split, dataset_split_name, self.tokenizer, max_tokens_per_batch, self.shuffle, self.source_lang, self.target_lang)

        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        self.assertGreater(len(dataloader.dataset), 0, "Dataloader should not be empty")

        batch_num = 0

        for batch in dataloader:
            self.assertIn("encoder_input_ids", batch)
            self.assertGreaterEqual(batch["encoder_input_ids"].shape[0], 0)
            batch_num += 1
        
        self.assertGreater(batch_num, 1)


if __name__ == '__main__':
    unittest.main()
