import torch
print("CUDA available" if torch.cuda.is_available() else "WARNING! CUDA is NOT available!")
