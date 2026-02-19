import os
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm


def count_with_two_tokenizers(
    tokenizer1: PreTrainedTokenizerBase,
    tokenizer2: PreTrainedTokenizerBase,
    text_path: str,
    chunk_size_bytes: int = 1_000_000,  # ~1MB per batch
) -> tuple[int, int, int]:
    """
    Count words and tokens for two tokenizers in one pass.
    
    Returns:
        (total_words, total_tokens_tokenizer1, total_tokens_tokenizer2)
    """
    total_words = 0
    total_tokens_1 = 0
    total_tokens_2 = 0
    file_size = os.path.getsize(text_path)
    
    tok_kwargs = dict(
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )
    
    with open(text_path, "r", encoding="utf-8") as f, tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Processing"
    ) as pbar:
        buffer = []
        buffer_bytes = 0
        
        for line in f:
            # Store cleaned line
            cleaned = line.rstrip("\n\r")
            buffer.append(cleaned)
            
            # Track bytes for progress
            line_bytes = len(line.encode("utf-8"))
            buffer_bytes += line_bytes
            pbar.update(line_bytes)
            
            # Process batch when threshold reached
            if buffer_bytes >= chunk_size_bytes:
                # Tokenize batch
                enc1 = tokenizer1.batch_encode_plus(buffer, **tok_kwargs)
                enc2 = tokenizer2.batch_encode_plus(buffer, **tok_kwargs)
                
                total_tokens_1 += sum(len(ids) for ids in enc1["input_ids"])
                total_tokens_2 += sum(len(ids) for ids in enc2["input_ids"])
                
                # Count words from the SAME text being tokenized
                total_words += sum(len(line.split()) for line in buffer)
                
                buffer.clear()
                buffer_bytes = 0
        
        # Process remaining buffer
        if buffer:
            enc1 = tokenizer1.batch_encode_plus(buffer, **tok_kwargs)
            enc2 = tokenizer2.batch_encode_plus(buffer, **tok_kwargs)
            
            total_tokens_1 += sum(len(ids) for ids in enc1["input_ids"])
            total_tokens_2 += sum(len(ids) for ids in enc2["input_ids"])
            total_words += sum(len(line.split()) for line in buffer)
    
    return total_words, total_tokens_1, total_tokens_2


def main():
    mbert_tokenizer_name = "google-bert/bert-base-multilingual-cased"
    cpt_mbert_tokenizer_name = "OMRIDRORI/mbert-tibetan-continual-wylie-final"
    text_path = "tibetan_bert_ready_512.txt"
    
    print("Loading tokenizers...")
    mbert = AutoTokenizer.from_pretrained(mbert_tokenizer_name, use_fast=True)
    cpt = AutoTokenizer.from_pretrained(cpt_mbert_tokenizer_name, use_fast=True)
    
    print("\nCounting words and tokens for both tokenizers in one pass...")
    n_words, n_tok_mbert, n_tok_cpt = count_with_two_tokenizers(mbert, cpt, text_path)
    
    if n_words == 0:
        print("No words found.")
        return
    
    # Calculate metrics
    comp_mbert = n_words / n_tok_mbert if n_tok_mbert else float("nan")
    comp_cpt = n_words / n_tok_cpt if n_tok_cpt else float("nan")
    fert_mbert = n_tok_mbert / n_words
    fert_cpt = n_tok_cpt / n_words
    
    # Display results
    print("\n=== Results ===")
    print(f"Words: {n_words:,}")
    print(f"\nmBERT:")
    print(f"  Tokens: {n_tok_mbert:,}")
    print(f"  Compression (words/token): {comp_mbert:.6f}")
    print(f"  Fertility (tokens/word): {fert_mbert:.6f}")
    print(f"\ncpt-mBERT:")
    print(f"  Tokens: {n_tok_cpt:,}")
    print(f"  Compression (words/token): {comp_cpt:.6f}")
    print(f"  Fertility (tokens/word): {fert_cpt:.6f}")
    
    # Show improvement
    if n_tok_cpt < n_tok_mbert:
        improvement = (1 - n_tok_cpt / n_tok_mbert) * 100
        print(f"\n✓ cpt-mBERT uses {improvement:.2f}% fewer tokens")
    elif n_tok_cpt > n_tok_mbert:
        increase = (n_tok_cpt / n_tok_mbert - 1) * 100
        print(f"\n✗ cpt-mBERT uses {increase:.2f}% more tokens")


if __name__ == "__main__":
    main()