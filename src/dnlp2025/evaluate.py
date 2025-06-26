"""
Evaluation script for the AIAYN translation model.
Loads a trained model from checkpoint and evaluates it on given text.
"""
import os
import torch
from tokenizers import Tokenizer

from src.dnlp2025.model import AIAYNModel, generate_sequence
from src.dnlp2025.check import get_device


def load_model_from_checkpoint(checkpoint_path, vocab_size, max_seq_len=128):
    """Load model from checkpoint file."""
    device = get_device()
    
    # Initialize model with same parameters as training
    model = AIAYNModel(
        vocab_size=vocab_size, 
        layers=6, 
        dimension=512, 
        ffn_dim=2048, 
        heads=8, 
        dropout=0.1, 
        max_seq_len=max_seq_len
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully")
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


def translate_text(model, tokenizer, source_text, max_new_tokens=50):
    """
    Translate a single source text using the trained model.
    
    Args:
        model: Trained AIAYN model
        tokenizer: Tokenizer used during training
        source_text: Source text to translate (string)
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        str: Translated text
    """
    device = next(model.parameters()).device
    
    # Get special token IDs
    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>")
    pad_token_id = tokenizer.token_to_id("<pad>")
    
    # Tokenize source text
    source_encoding = tokenizer.encode(source_text)
    encoder_input = torch.tensor([source_encoding.ids], dtype=torch.long, device=device)
    
    print(f"Source: {source_text}")
    print(f"Source tokens: {source_encoding.tokens}")
    print(f"Encoder input shape: {encoder_input.shape}")
    
    # Generate translation
    with torch.no_grad():
        generated_ids = generate_sequence(
            model=model, 
            encoder_input=encoder_input, 
            max_new_tokens=max_new_tokens,
            vocab_size=len(tokenizer.get_vocab()),
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
    
    print(f"Generated token IDs: {generated_ids[0]}")
    
    # Decode generated tokens (skip BOS token)
    generated_tokens = generated_ids[0][1:].tolist()  # Remove BOS token
    
    # Remove EOS token if present
    if eos_token_id in generated_tokens:
        eos_idx = generated_tokens.index(eos_token_id)
        generated_tokens = generated_tokens[:eos_idx]
    
    # Decode to text
    translated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"Generated tokens: {[tokenizer.id_to_token(id) for id in generated_tokens]}")
    print(f"Translation: {translated_text}")
    
    return translated_text


def evaluate_model():
    """Main evaluation function."""
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tokenizer_path = os.path.join(project_root, "tokenizers", "de_en_tokenizer.json")
    checkpoint_dir = os.path.join(os.path.expanduser("~"), "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "aiayn.pt")
    
    # Load max_seq_len from dataloader info if available
    max_seq_len = 128
    max_seq_len_file = "dataloader/max_seq_len.txt"
    if os.path.exists(max_seq_len_file):
        with open(max_seq_len_file, "r") as f:
            max_seq_len = int(f.read().strip())
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    print(f"Vocab size: {vocab_size}")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, vocab_size, max_seq_len)
    model.eval()
    
    # Test sentences (German to English)
    test_sentences = [
        "Guten Morgen.",
        "Wie geht es dir?",
        "Das ist ein schönes Buch.",
        "Ich möchte einen Kaffee.",
        "Die Katze schläft auf dem Sofa."
    ]
    
    print("\n" + "="*50)
    print("TRANSLATION EVALUATION")
    print("="*50)
    
    for i, german_text in enumerate(test_sentences, 1):
        print(f"\n--- Example {i} ---")
        try:
            translation = translate_text(model, tokenizer, german_text, max_new_tokens=30)
            print(f"✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
        print("-" * 30)

def main():
    # Change to project root for relative paths to work
    os.chdir("../../")
    evaluate_model()

if __name__ == "__main__":
    main()