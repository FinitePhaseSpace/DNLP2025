# import torch
# print("CUDA available" if torch.cuda.is_available() else "WARNING! CUDA is NOT available!")

import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device
