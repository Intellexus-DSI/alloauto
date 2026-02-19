"""
Inference code for Tibetan code-switching detection
Reads .docx files and outputs CSV predictions
FIXED: Handles long documents by processing in chunks
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from docx import Document


def read_docx_file(docx_path):
    """
    Read text from a .docx file
    """
    doc = Document(docx_path)
    # Extract all text from paragraphs
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
    return text


def process_text_with_model_fixed(text, model, tokenizer, device, max_words_per_chunk=400):
    """
    Process text and return word-level predictions with better alignment
    Handles long texts by processing in chunks
    """
    model.eval()
    words = text.split()

    if not words:
        return [], [], []

    all_predictions = []
    all_confidences = []

    # Process in chunks with overlap
    chunk_size = max_words_per_chunk  # Process 400 words at a time
    overlap = 50  # Overlap between chunks to maintain context

    for start_idx in range(0, len(words), chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]

        # Tokenize chunk with proper settings
        encoding = tokenizer(
            chunk_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=False
        )

        inputs = {k: v.to(device) for k, v in encoding.items()}

        # Get predictions for this chunk
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            probabilities = F.softmax(logits, dim=-1)

        # Align with words in this chunk
        word_ids = encoding.word_ids()
        chunk_preds = []
        chunk_confs = []

        # Group predictions by word
        for word_idx in range(len(chunk_words)):
            # Find all token positions for this word
            token_positions = [i for i, wid in enumerate(word_ids) if wid == word_idx]

            if token_positions:
                # Use FIRST token's prediction
                first_pos = token_positions[0]
                pred = predictions[0][first_pos].item()
                conf = probabilities[0][first_pos][pred].item()

                chunk_preds.append(pred)
                chunk_confs.append(conf)
            else:
                # Default fallback (should rarely happen)
                chunk_preds.append(0)
                chunk_confs.append(0.0)

        # Add chunk predictions to overall results
        # Skip overlap region for subsequent chunks (except the first chunk)
        if start_idx == 0:
            all_predictions.extend(chunk_preds)
            all_confidences.extend(chunk_confs)
        else:
            # Skip the overlap portion that was already processed
            skip_overlap = min(overlap, len(chunk_preds))
            all_predictions.extend(chunk_preds[skip_overlap:])
            all_confidences.extend(chunk_confs[skip_overlap:])

        print(f"  Processed chunk: words {start_idx + 1}-{end_idx} of {len(words)}")

    # Verify alignment
    assert len(all_predictions) == len(words), f"Alignment error: {len(all_predictions)} != {len(words)}"

    return words, all_predictions, all_confidences


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


def save_predictions_to_csv(words, predictions, confidences, output_csv_path):
    """
    Save predictions to CSV file
    """
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['Word_Index', 'Word', 'Prediction_ID', 'Prediction_Label', 'Confidence'])

        # Write data
        for i, (word, pred, conf) in enumerate(zip(words, predictions, confidences)):
            writer.writerow([
                i,
                word,
                pred,
                get_label_name(pred),
                f"{conf:.4f}"
            ])

    print(f"✅ Saved predictions to: {output_csv_path}")


def process_docx_file(docx_path, model, tokenizer, device, output_dir='./inference_results'):
    """
    Process a .docx file and save predictions to CSV
    """
    print(f"\n{'=' * 80}")
    print(f"PROCESSING: {docx_path}")
    print(f"{'=' * 80}")

    # Read the docx file
    try:
        text = read_docx_file(docx_path)
        print(f"✅ Successfully read file")
        print(f"Total characters: {len(text)}")
        print(f"Total words: {len(text.split())}")

        # Check if chunking will be needed
        if len(text.split()) > 400:
            print(f"⚠️ Long document detected - will process in chunks")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

    if not text.strip():
        print("⚠️ File is empty!")
        return None

    # Process with model (now handles chunking automatically)
    words, predictions, confidences = process_text_with_model_fixed(
        text, model, tokenizer, device
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
    input_filename = Path(docx_path).stem  # Get filename without extension
    output_csv_path = Path(output_dir) / f"{input_filename}_predictions.csv"

    # Save to CSV
    save_predictions_to_csv(words, predictions, confidences, output_csv_path)

    return {
        'file': docx_path,
        'words': words,
        'predictions': predictions,
        'confidences': confidences,
        'pred_counts': pred_counts,
        'output_csv': str(output_csv_path)
    }


def run_inference_on_docx_files(docx_files, model_path, output_dir='./inference_results'):
    """
    Run inference on multiple .docx files and save results to CSV
    """
    print("\n" + "=" * 80)
    print("CODE-SWITCHING INFERENCE ON DOCX FILES")
    print("=" * 80)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
    for docx_file in docx_files:
        result = process_docx_file(docx_file, model, tokenizer, device, output_dir)
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
    if total_words > 0:
        print(f"   Overall switch rate: {total_switches / total_words * 100:.2f}%")

    if total_switches == 0:
        print("\n⚠️ WARNING: NO SWITCHES DETECTED IN ANY FILE!")
        print("   This may indicate the model needs retraining or adjustment.")

    return all_results


# Main execution
if __name__ == "__main__":
    # Your .docx files
    docx_files = [
        'alloauto-segmentation-training/data/D3818_for_ALTO_TEST.docx',
        'alloauto-segmentation-training/data/D496_for_ALTO_TEST.docx'
    ]

    # Your model path
    model_path = './alloauto-segmentation-training/fine_tuned_ALTO_models/ALTO_combined_data_with_allo_and_auto_balanced_22_10/final_model'

    # Output directory for CSV files
    output_dir = './inference_results_Orna_22_10_MUL_ALTO_with_more_allo_auto_balanced_data_correct_segmentation'

    # Run inference
    results = run_inference_on_docx_files(docx_files, model_path, output_dir)

    print("\n✅ Inference complete!")
    print(f"📁 Results saved in: {output_dir}/")