
import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock

# Ensure we can import the script
sys.path.append(os.getcwd())

# Import the function to test
from inference.inference_ALTO import process_text_with_sliding_window

class MockTokenizer:
    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self.vocab = {c: i+1000 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    
    def __call__(self, words, is_split_into_words=True, add_special_tokens=False, return_tensors=None):
        # Simple character-level tokenization for testing
        # "A B" -> [1000, 1001]
        input_ids = []
        _word_ids = []
        for i, word in enumerate(words):
            if word in self.vocab:
                input_ids.append(self.vocab[word])
                _word_ids.append(i)
            else:
                # Handle unknown
                input_ids.append(999)
                _word_ids.append(i)
                
        # Return a mock encoding object
        encoding = MagicMock()
        encoding.__getitem__.return_value = input_ids # encoding['input_ids']
        encoding.word_ids.return_value = _word_ids
        return encoding

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.num_labels = 4
    
    def forward(self, input_ids, attention_mask=None):
        # Create deterministic logits based on input_ids
        # This ensures that if the input ID is correct, the output is consistent
        # regardless of padding or batching.
        
        # Shape: [batch, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # We'll generate logits where the value depends on the token ID
        # So we can track if 'A' always yields the same probs.
        # logits shape: [batch, seq_len, 4]
        
        logits = torch.zeros(batch_size, seq_len, 4)
        
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = input_ids[b, s]
                if token_id == 0: # Pad
                    # irrelevant, but let's keep it zero
                    pass
                else:
                    # Deterministic values
                    val = float(token_id) / 1000.0
                    logits[b, s, 0] = val
                    logits[b, s, 1] = val * 2
                    logits[b, s, 2] = val * 3
                    logits[b, s, 3] = val * 4
                    
        return MagicMock(logits=logits)

def run_verification():
    print("🧪 Starting Batching Verification...")
    
    # Setup
    text = "A B C D E F G H I J K L M N O P Q R S T" # 20 words
    tokenizer = MockTokenizer()
    model = MockModel()
    device = "cpu"
    
    # Parameters
    stride = 2
    window_size = 6 # Small window: [CLS, 1, 2, 3, 4, SEP] -> 4 tokens per window
    
    print(f"Text length: {len(text.split())} words")
    print(f"Window size: {window_size} (max 4 real tokens)")
    print(f"Stride: {stride}")
    
    # Run 1: Sequential (Batch Size = 1)
    print("\n1️⃣  Running Sequential (Batch Size = 1)...")
    _, _, _, probs_seq, _ = process_text_with_sliding_window(
        text, model, tokenizer, device, 
        stride=stride, window_size=window_size, batch_size=1
    )
    
    # Run 2: Batched (Batch Size = 4)
    print("\n2️⃣  Running Batched (Batch Size = 4)...")
    _, _, _, probs_batch, _ = process_text_with_sliding_window(
        text, model, tokenizer, device, 
        stride=stride, window_size=window_size, batch_size=4
    )
    
    # Run 3: Batched with odd size (Batch Size = 3) to test padding/residuals
    print("\n3️⃣  Running Batched (Batch Size = 3)...")
    _, _, _, probs_batch_3, _ = process_text_with_sliding_window(
        text, model, tokenizer, device, 
        stride=stride, window_size=window_size, batch_size=3
    )
    
    # Compare
    print("\n🔍 Comparing Results...")
    
    # Check lengths
    if len(probs_seq) != len(probs_batch):
        print(f"❌ Length mismatch! Seq: {len(probs_seq)}, Batch: {len(probs_batch)}")
        return
        
    # Check values
    max_diff = 0.0
    for i in range(len(probs_seq)):
        p_seq = probs_seq[i]
        p_batch = probs_batch[i]
        p_batch3 = probs_batch_3[i]
        
        diff = np.abs(p_seq - p_batch).max()
        diff3 = np.abs(p_seq - p_batch3).max()
        
        if diff > 1e-6:
            print(f"❌ Mismatch at token {i}:")
            print(f"   Seq:   {p_seq}")
            print(f"   Batch: {p_batch}")
            return
            
        if diff3 > 1e-6:
            print(f"❌ Mismatch at token {i} (Batch=3):")
            print(f"   Seq:   {p_seq}")
            print(f"   Batch3: {p_batch3}")
            return
            
        max_diff = max(max_diff, diff)
        
    print(f"✅ PASSED! Max difference: {max_diff:.9f}")
    print("   The batched implementation produces mathematically identical results to sequential execution.")

if __name__ == "__main__":
    run_verification()
