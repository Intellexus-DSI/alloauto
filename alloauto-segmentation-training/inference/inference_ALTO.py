"""
Inference code for Tibetan code-switching detection
Reads .docx files and outputs CSV predictions
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from docx import Document
from tqdm import tqdm


def read_docx_file(docx_path):
    """
    Read text from a .docx file
    """
    doc = Document(docx_path)
    # Extract all text from paragraphs
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
    return text


def read_txt_file(txt_path):
    """
    Read text from a .txt file
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def process_text_with_sliding_window(text, model, tokenizer, device, stride=200, window_size=512, batch_size=16):
    """
    Process text using sliding window inference and return word-level predictions
    with averaged probabilities across windows.
    """
    model.eval()
    words = text.split()

    if not words:
        return [], [], [], [], 0

    # 1. Tokenize the entire text without truncation/padding first
    #    to get the global sequence of tokens and word mappings.
    full_encoding = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        return_tensors=None  # Get lists
    )
    
    all_input_ids = full_encoding['input_ids']
    all_word_ids = full_encoding.word_ids()
    
    total_tokens = len(all_input_ids)

    print(f"Total tokens in text: {total_tokens}")
    
    # Store accumulated probabilities for each token: dictionary mapping token_idx -> list of prob vectors
    token_probs_accumulator = {i: [] for i in range(total_tokens)}
    
    # 2. Sliding Window Inference
    #    Max tokens per window = window_size - 2 (for [CLS] and [SEP])
    max_tokens = window_size - 2
    
    # Create windows
    windows = list(range(0, total_tokens, stride))
    
    print(f"Processing {len(windows)} windows (stride={stride}, batch_size={batch_size})...")

    # Helper function to run inference on a batch
    def run_batch(batch_input_ids, batch_start_indices):
        if not batch_input_ids:
            return

        # Pad sequences to max length in this batch
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        
        # Use tokenizer's pad token if available, else 0
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids = ids + [pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
            
        input_tensor = torch.tensor(padded_input_ids).to(device)
        mask_tensor = torch.tensor(attention_masks).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=mask_tensor)
            logits = outputs.logits  # Shape: [batch, seq_len, num_labels]
            probs_batch = F.softmax(logits, dim=-1) # Shape: [batch, seq_len, num_labels]
            
        # Process each item in the batch
        for i, start_idx in enumerate(batch_start_indices):
            # Get real length of this sequence (unpadded)
            real_len = len(batch_input_ids[i])
            
            # Slice off padding from probs
            # Shape: [real_len, num_labels]
            seq_probs = probs_batch[i, :real_len, :]
            
            # Remove special tokens predictions ([CLS] is at 0, [SEP] is at -1 of the real sequence)
            # Valid predictions correspond to the original chunk_ids
            # valid_probs = seq_probs[1:-1] would be [1 : real_len-1]
            if real_len > 2:
                valid_probs = seq_probs[1:real_len-1] # Shape: [len(chunk_ids), num_labels]
                
                # Map back to global token indices and accumulate
                for j, prob_vec in enumerate(valid_probs):
                    global_idx = start_idx + j
                    if global_idx < total_tokens:
                        token_probs_accumulator[global_idx].append(prob_vec.cpu().numpy())

    # Collect batches
    batch_input_ids = []
    batch_start_indices = []

    for start_idx in tqdm(windows, desc="Inference Progress", unit="window"):
        end_idx = min(start_idx + max_tokens, total_tokens)
        
        # Extract chunk
        chunk_ids = all_input_ids[start_idx:end_idx]
        
        if not chunk_ids:
            continue
            
        # Add special tokens
        input_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
        
        batch_input_ids.append(input_ids)
        batch_start_indices.append(start_idx)
        
        # If batch is full, run it
        if len(batch_input_ids) >= batch_size:
            run_batch(batch_input_ids, batch_start_indices)
            batch_input_ids = []
            batch_start_indices = []
            
    # Process remaining items in the last batch
    if batch_input_ids:
        run_batch(batch_input_ids, batch_start_indices)
                
    # 3. Aggregate Probabilities per Token
    final_token_probs = []
    final_token_preds = []
    final_token_confs = []
    
    for i in range(total_tokens):
        probs_list = token_probs_accumulator[i]
        if not probs_list:
            # Should not happen if strides cover everything
            avg_probs = np.zeros(model.config.num_labels)
        else:
            # Average the probabilities
            avg_probs = np.mean(probs_list, axis=0)
            
        final_token_probs.append(avg_probs)
        pred_label = np.argmax(avg_probs)
        final_token_preds.append(pred_label)
        final_token_confs.append(avg_probs[pred_label])
        
    # 4. Align with Words
    #    We use the prediction of the FIRST token of the word.
    aligned_preds = []
    aligned_confs = []
    aligned_probs = [] # Store the full prob vector for the chosen token
    
    # Group predictions by word
    for word_idx in range(len(words)):
        # Find the FIRST token position for this word in the global sequence
        try:
            first_token_pos = all_word_ids.index(word_idx)
            
            pred = final_token_preds[first_token_pos]
            conf = final_token_confs[first_token_pos]
            probs = final_token_probs[first_token_pos]
            
            aligned_preds.append(pred)
            aligned_confs.append(conf)
            aligned_probs.append(probs)
            
        except ValueError:
            # Word might not have any tokens (e.g. if it was just special chars stripped by tokenizer?)
            # Or if word_ids logic is tricky. Fallback.
            aligned_preds.append(0)
            aligned_confs.append(0.0)
            aligned_probs.append(np.zeros(model.config.num_labels))

    return words, aligned_preds, aligned_confs, aligned_probs, total_tokens


def get_label_name(label_id):
    """
    Convert label ID to readable name
    """
    label_names = {
        0: 'Non-switch Auto',
        1: 'Non-switch Allo',
        2: 'Switch→Auto',
        3: 'Switch→Allo'
    }
    return label_names.get(label_id, 'Unknown')


def save_predictions_to_csv(words, predictions, confidences, probs, output_csv_path):
    """
    Save predictions to CSV file with detailed probabilities
    """
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = [
            'Word_Index', 'Word', 
            'Prob_0_NonSwitchAuto', 'Prob_1_NonSwitchAllo', 'Prob_2_SwitchAuto', 'Prob_3_SwitchAllo',
            'Decided_Prediction_ID', 'Decided_Prediction_Label', 'Decided_Confidence'
        ]
        writer.writerow(header)

        # Write data
        for i, (word, pred, conf, prob_vec) in enumerate(zip(words, predictions, confidences, probs)):
            row = [
                i,
                word,
                f"{prob_vec[0]:.4f}", f"{prob_vec[1]:.4f}", f"{prob_vec[2]:.4f}", f"{prob_vec[3]:.4f}",
                pred,
                get_label_name(pred),
                f"{conf:.4f}"
            ]
            writer.writerow(row)

    print(f"✅ Saved predictions to: {output_csv_path}")


def process_file(file_path, model, tokenizer, device, output_dir='./inference_results', stride=200, window_size=512, batch_size=4):
    """
    Process a file (.docx or .txt) and save predictions to CSV
    """
    print(f"\n{'=' * 80}")
    print(f"PROCESSING: {file_path}")
    print(f"{'=' * 80}")

    # Read the file
    try:
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.docx':
            text = read_docx_file(file_path)
        elif file_ext == '.txt':
            text = read_txt_file(file_path)
        else:
            print(f"❌ Error: Unsupported file extension '{file_ext}'. Supported: .docx, .txt")
            return None
            
        print(f"✅ Successfully read file")
        print(f"Total characters: {len(text)}")
        print(f"Total words: {len(text.split())}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

    if not text.strip():
        print("⚠️ File is empty!")
        return None

    # Process with model using sliding window
    words, predictions, confidences, probs, total_tokens = process_text_with_sliding_window(
        text, model, tokenizer, device, stride=stride, window_size=window_size, batch_size=batch_size
    )

    # Statistics
    pred_counts = {i: predictions.count(i) for i in range(4)}
    switch_count = pred_counts.get(2, 0) + pred_counts.get(3, 0)

    print(f"\n📊 Prediction Statistics:")
    print(f"  Non-switch Auto: {pred_counts.get(0, 0)}")
    print(f"  Non-switch Allo: {pred_counts.get(1, 0)}")
    print(f"  Switch→Auto: {pred_counts.get(2, 0)}")
    print(f"  Switch→Allo: {pred_counts.get(3, 0)}")
    print(f"  Total switches: {switch_count}")
    print(f"  Switch rate: {switch_count / len(predictions) * 100:.2f}%")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate output CSV filename
    input_filename = Path(file_path).stem  # Get filename without extension
    output_csv_path = Path(output_dir) / f"{input_filename}_predictions.csv"

    # Save to CSV
    save_predictions_to_csv(words, predictions, confidences, probs, output_csv_path)

    return {
        'file': file_path,
        'words': words,
        'predictions': predictions,
        'confidences': confidences,
        'pred_counts': pred_counts,
        'output_csv': str(output_csv_path)
    }


def run_inference_on_files(files, model_path, output_dir='./inference_results', stride=200, batch_size=4):
    """
    Run inference on multiple files and save results to CSV
    """
    print("\n" + "=" * 80)
    print("CODE-SWITCHING INFERENCE ON FILES")
    print("=" * 80)

    # Load model
    print("\n🔍 Checking Compute Device...")
    
    # Check for various accelerators
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if has_cuda:
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {device_name}")
        print(f"   Backend: {'ROCm (AMD)' if torch.version.hip else 'CUDA (NVIDIA)'}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif has_mps:
        device = torch.device('mps')
        print(f"✅ Apple Silicon GPU Detected (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU detected. Running on CPU.")
        print("   - If you have an AMD GPU, ensure ROCm drivers are installed and")
        print("     that you are using a ROCm-compatible PyTorch version.")
        print("   - If you have an NVIDIA GPU, check your CUDA drivers.")
    
    print(f"   Active Device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        print(f"   Model: {model_path}")
        print(f"   Number of labels: {model.config.num_labels}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Process each file
    all_results = []
    for file_path in files:
        result = process_file(file_path, model, tokenizer, device, output_dir, stride=stride, batch_size=batch_size)
        if result:
            all_results.append(result)

    # Overall summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)

    total_words = 0
    total_switches = 0

    for result in all_results:
        switches = result['pred_counts'].get(2, 0) + result['pred_counts'].get(3, 0)
        words_count = len(result['words'])
        total_switches += switches
        total_words += words_count

        print(f"\n📄 {Path(result['file']).name}:")
        print(f"   Words: {words_count}")
        print(f"   Switches: {switches}")
        print(f"   Switch rate: {switches / words_count * 100:.2f}%")
        print(f"   Output: {result['output_csv']}")

    print(f"\n📊 Overall Statistics:")
    print(f"   Total files processed: {len(all_results)}")
    print(f"   Total words: {total_words}")
    print(f"   Total switches: {total_switches}")
    print(f"   Overall switch rate: {total_switches / total_words * 100:.2f}%")

    if total_switches == 0:
        print("\n⚠️ WARNING: NO SWITCHES DETECTED IN ANY FILE!")
        print("   This may indicate the model needs retraining or adjustment.")

    return all_results


# Main execution
if __name__ == "__main__":
    # Your files
    files = [
        # 'alloauto-segmentation-training/data/D3818_for_ALTO_TEST.docx',
        # 'alloauto-segmentation-training/data/D496_for_ALTO_TEST.docx',
        # 'alloauto-segmentation-training/data/data_auto_Orna/auto with auto citation theg mchog 4.docx',
        # 'alloauto-segmentation-training/data/data_auto_Orna/rTsags Dar ma rgyal po - gSang ldan gyi rgya cher bshad pa 2.docx'
        # 'alloauto-segmentation-training/data/Sonam_inference_24_10.docx'
        'Jan_31_case_study/Tb. 1.txt'
    ]

    # Your model path
    model_path = 'levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_MUL_SEG_RUNI'
    
    # Inference parameters
    stride = 50
    batch_size = 8

    # Output directory for CSV files
    output_dir = f'./results/stride_{stride}'

    # Run inference
    results = run_inference_on_files(files, model_path, output_dir, stride=stride, batch_size=batch_size)

    print("\n✅ Inference complete!")
    print(f"📁 Results saved in: {output_dir}/")




























