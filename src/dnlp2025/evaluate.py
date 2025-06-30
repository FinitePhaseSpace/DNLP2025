"""
Evaluation script for the AIAYN translation model.
Loads a trained model from checkpoint and evaluates it on given text.
"""
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import evaluate
from datasets import load_dataset
import math

from src.dnlp2025.model import AIAYNModel, generate_sequence
from src.dnlp2025.check import get_device
from src.dnlp2025.download_datasets import download_wmt14_de_en


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


def evaluate_bleu_on_test_dataset(model, tokenizer, max_samples=1000):
    """
    Evaluate BLEU score on the WMT14 test dataset.
    
    Args:
        model: Trained AIAYN model
        tokenizer: Tokenizer used during training
        max_samples: Maximum number of test samples to evaluate (for speed)
        
    Returns:
        dict: BLEU score results
    """
    print("\n" + "="*50)
    print("BLEU SCORE EVALUATION ON TEST DATASET")
    print("="*50)
    
    # Load WMT14 de-en test dataset
    dataset = download_wmt14_de_en()
    test_dataset = dataset['test']
    
    # Limit samples for faster evaluation
    if len(test_dataset) > max_samples:
        test_dataset = test_dataset.select(range(max_samples))
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    # Initialize BLEU metric
    bleu_metric = evaluate.load("bleu")
    
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_dataset)} samples...")
            
            # Extract source (English) and reference (German) texts
            source_text = example['translation']['en']
            reference_text = example['translation']['de']
            
            try:
                # Generate translation
                prediction = translate_text(model, tokenizer, source_text, max_new_tokens=50)
                
                predictions.append(prediction)
                references.append(reference_text)
                
            except Exception as e:
                print(f"Error translating sample {i}: {e}")
                # Skip this sample
                continue
    
    # Calculate BLEU score
    if predictions and references:
        # BLEU expects references as list of lists (multiple references per prediction)
        references_formatted = [[ref] for ref in references]
        bleu_results = bleu_metric.compute(
            predictions=predictions, 
            references=references_formatted
        )
        
        print(f"\nBLEU Score Results:")
        print(f"BLEU: {bleu_results['bleu']:.4f}")
        print(f"BLEU-1: {bleu_results['precisions'][0]:.4f}")
        print(f"BLEU-2: {bleu_results['precisions'][1]:.4f}")
        print(f"BLEU-3: {bleu_results['precisions'][2]:.4f}")
        print(f"BLEU-4: {bleu_results['precisions'][3]:.4f}")
        print(f"Brevity Penalty: {bleu_results['brevity_penalty']:.4f}")
        print(f"Length Ratio: {bleu_results['length_ratio']:.4f}")
        print(f"Translation Length: {bleu_results['translation_length']}")
        print(f"Reference Length: {bleu_results['reference_length']}")
        
        # Show some example translations
        print(f"\nExample Translations:")
        for i in range(min(5, len(predictions))):
            print(f"\nExample {i+1}:")
            print(f"Source (EN): {test_dataset[i]['translation']['en']}")
            print(f"Reference (DE): {references[i]}")
            print(f"Prediction (DE): {predictions[i]}")
        
        return bleu_results
    else:
        print("No successful translations generated!")
        return None


def calculate_perplexity(model, tokenizer, test_dataset, max_samples=1000):
    """
    Calculate perplexity on the test dataset according to the "Attention is All You Need" paper.
    Perplexity is calculated per-wordpiece (subword token).
    
    Args:
        model: Trained AIAYN model
        tokenizer: Tokenizer used during training
        test_dataset: Test dataset to evaluate on
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        float: Perplexity score
    """
    print("\n" + "="*50)
    print("PERPLEXITY CALCULATION")
    print("="*50)
    
    device = next(model.parameters()).device
    
    # Get special token IDs
    pad_token_id = tokenizer.token_to_id("<pad>")
    
    # Limit samples for faster evaluation
    if len(test_dataset) > max_samples:
        test_dataset = test_dataset.select(range(max_samples))
    
    print(f"Calculating perplexity on {len(test_dataset)} test samples...")
    
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_dataset)} samples...")
            
            try:
                # Get source (English) and target (German) texts
                source_text = example['translation']['en']
                target_text = example['translation']['de']
                
                # Tokenize source and target
                source_encoding = tokenizer.encode(source_text)
                target_encoding = tokenizer.encode(target_text)
                
                # Convert to tensors
                encoder_input = torch.tensor([source_encoding.ids], dtype=torch.long, device=device)
                decoder_input = torch.tensor([target_encoding.ids[:-1]], dtype=torch.long, device=device)  # Remove last token
                target_output = torch.tensor([target_encoding.ids[1:]], dtype=torch.long, device=device)   # Remove first token (BOS)
                
                # Create masks as done in the training code
                # Source key padding mask
                src_key_padding_mask = (encoder_input == pad_token_id).float()
                
                # Target key padding mask
                tgt_key_padding_mask = (decoder_input == pad_token_id).float()
                
                # Target causal mask (subsequent mask)
                tgt_len = decoder_input.size(1)
                target_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    tgt_len, device=device
                )
                
                # Forward pass through the model with all required arguments
                logits = model(
                    encoder_input,           # enc_input
                    None,                    # enc_mask (not used in training)
                    decoder_input,           # dec_input
                    target_mask,             # dec_mask (causal mask)
                    tgt_key_padding_mask,    # tgt_key_padding_mask
                    src_key_padding_mask     # memory_key_padding_mask
                )
                
                # Calculate cross-entropy loss for each token
                # logits shape: [batch_size, seq_len, vocab_size]
                # target_output shape: [batch_size, seq_len]
                
                # Reshape for loss calculation
                logits_flat = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
                target_flat = target_output.view(-1)  # [batch_size * seq_len]
                
                # Calculate log probabilities
                log_probs = F.log_softmax(logits_flat, dim=-1)
                
                # Get log probabilities for actual tokens (excluding padding)
                for j, target_token in enumerate(target_flat):
                    if target_token != pad_token_id:  # Skip padding tokens
                        token_log_prob = log_probs[j, target_token]
                        total_log_likelihood += token_log_prob.item()
                        total_tokens += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    if total_tokens > 0:
        # Calculate perplexity: exp(-1/N * sum(log P(token)))
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = math.exp(-avg_log_likelihood)
        
        print(f"\nPerplexity Results:")
        print(f"Total tokens evaluated: {total_tokens}")
        print(f"Average log-likelihood: {avg_log_likelihood:.6f}")
        print(f"Perplexity (per-wordpiece): {perplexity:.4f}")
        
        return perplexity
    else:
        print("No valid tokens found for perplexity calculation!")
        return None


def evaluate_model():
    """Main evaluation function."""
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tokenizer_path = os.path.join(project_root, "tokenizers", "de_en_tokenizer.json")
    checkpoint_dir = os.path.join(os.path.expanduser("~"), "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "aiayn-en-de-3.pt")
    
    # Load max_seq_len from dataloader info if available
    max_seq_len = 128
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    print(f"Vocab size: {vocab_size}")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, vocab_size, max_seq_len)
    model.eval()
    
    # Test sentences (English to German)
    test_sentences = [
        "Good morning.",
        "How are you?",
        "This is a beautiful book.",
        "I would like a coffee.",
        "The cat sleeps on the sofa."
    ]
    
    print("\n" + "="*50)
    print("TRANSLATION EVALUATION")
    print("="*50)
    
    for i, english_text in enumerate(test_sentences, 1):
        print(f"\n--- Example {i} ---")
        try:
            translation = translate_text(model, tokenizer, english_text, max_new_tokens=30)
            print(f"✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
        print("-" * 30)
    
    # Load test dataset for both BLEU and perplexity evaluation
    dataset = download_wmt14_de_en()
    test_dataset = dataset['test']
    
    #Evaluate BLEU score on test dataset
    try:
        bleu_results = evaluate_bleu_on_test_dataset(model, tokenizer, max_samples=500)
        if bleu_results:
            print(f"\n{'='*50}")
            print(f"FINAL BLEU SCORE: {bleu_results['bleu']:.4f}")
            print(f"{'='*50}")
    except Exception as e:
        print(f"\n✗ Error during BLEU evaluation: {e}")
        print("Continuing without BLEU score...")
    
    # Evaluate perplexity on test dataset
    try:
        perplexity = calculate_perplexity(model, tokenizer, test_dataset, max_samples=500)
        if perplexity:
            print(f"\n{'='*50}")
            print(f"FINAL PERPLEXITY (per-wordpiece): {perplexity:.4f}")
            print(f"{'='*50}")
    except Exception as e:
        print(f"\n✗ Error during perplexity evaluation: {e}")
        print("Continuing without perplexity score...")

def main():
    # Change to project root for relative paths to work
    os.chdir("../../")
    evaluate_model()

if __name__ == "__main__":
    main()