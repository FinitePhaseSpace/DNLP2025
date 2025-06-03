import os
import datasets


def download_wmt14_de_en():
    """Download the WMT14 German-English dataset."""
    dataset = datasets.load_dataset("wmt14", "de-en")
    return dataset


def download_wmt14_fr_en():
    """Download the WMT14 French-English dataset. It's significantly bigger than the German-English dataset."""
    dataset = datasets.load_dataset("wmt14", "fr-en")
    return dataset
