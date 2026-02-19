import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
print(f"Number of GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from typing import Dict, List, Tuple


class CodeSwitchingDataset(Dataset):
    """Dataset for token-level code-switching classification."""

    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row['tokens'].split()
        labels = list(map(int, row['labels'].split(',')))

        # Tokenize and align labels
        encoding = self.tokenize_and_align_labels(tokens, labels)

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(encoding['labels'])
        }

    def tokenize_and_align_labels(self, tokens, labels):
        """
        Tokenize words and align labels with subword tokens.
        """
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        word_ids = tokenized_inputs.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                aligned_labels.append(labels[word_idx] if word_idx < len(labels) else -100)
            else:
                # Other tokens of the same word get -100
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs['labels'] = aligned_labels
        return tokenized_inputs


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten for metrics
    flat_true = [item for sublist in true_labels for item in sublist]
    flat_pred = [item for sublist in true_predictions for item in sublist]

    # Calculate metrics
    accuracy = accuracy_score(flat_true, flat_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average='weighted', zero_division=0
    )

    # Per-class metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_0_f1': class_f1[0],
        'class_1_f1': class_f1[1],  # Switch to auto
        'class_2_f1': class_f1[2],  # Switch to allo
    }


def train_code_switching_model(
        train_csv='train_sequences.csv',
        val_csv='val_sequences.csv',
        model_name='bert-base-multilingual-cased',  # Good for Tibetan
        output_dir='./code_switching_model',
        num_epochs=5,
        batch_size=16,
        learning_rate=2e-5
):
    """
    Train BERT model for code-switching detection.
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,  # 0: no switch, 1: to auto, 2: to allo
        label2id={'no_switch': 0, 'to_auto': 1, 'to_allo': 2},
        id2label={0: 'no_switch', 1: 'to_auto', 2: 'to_allo'}
    )

    # Create datasets
    train_dataset = CodeSwitchingDataset(train_csv, tokenizer)
    val_dataset = CodeSwitchingDataset(val_csv, tokenizer)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        warmup_steps=500,
        save_total_limit=3,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the model
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')

    # Evaluate
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()
    print(eval_results)

    return trainer, model, tokenizer


def predict_code_switching(text, model, tokenizer, return_probs=False):
    """
    Fixed prediction function that handles word_ids properly.
    """
    # Get device from model
    device = next(model.parameters()).device

    # Tokenize
    tokens = text.split()

    # Keep the tokenizer output to access word_ids
    tokenizer_output = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Get word_ids BEFORE converting to device
    word_ids = tokenizer_output.word_ids()

    # Now move to device
    inputs = {key: val.to(device) for key, val in tokenizer_output.items()}

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        probs = torch.softmax(outputs.logits, dim=2)

    # Align predictions with original tokens
    aligned_predictions = []
    aligned_probs = []

    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            pred = predictions[0][idx].item()
            aligned_predictions.append(pred)
            if return_probs:
                aligned_probs.append(probs[0][idx].cpu().numpy())
        previous_word_idx = word_idx

    # Create output
    results = []
    for i, (token, pred) in enumerate(zip(tokens[:len(aligned_predictions)], aligned_predictions)):
        result = {
            'token': token,
            'prediction': pred,
            'label': ['no_switch', 'to_auto', 'to_allo'][pred]
        }
        if return_probs and i < len(aligned_probs):
            result['probabilities'] = {
                'no_switch': float(aligned_probs[i][0]),
                'to_auto': float(aligned_probs[i][1]),
                'to_allo': float(aligned_probs[i][2])
            }
        results.append(result)

    return results

def analyze_predictions(predictions):
    """
    Analyze prediction results to extract code-switching segments.
    """
    segments = []
    current_segment = {'type': 'unknown', 'tokens': [], 'start_idx': 0}

    for i, pred in enumerate(predictions):
        if pred['prediction'] == 1:  # Switch to auto
            if current_segment['tokens']:
                current_segment['end_idx'] = i - 1
                segments.append(current_segment)
            current_segment = {'type': 'auto', 'tokens': [pred['token']], 'start_idx': i}
        elif pred['prediction'] == 2:  # Switch to allo
            if current_segment['tokens']:
                current_segment['end_idx'] = i - 1
                segments.append(current_segment)
            current_segment = {'type': 'allo', 'tokens': [pred['token']], 'start_idx': i}
        else:  # No switch
            current_segment['tokens'].append(pred['token'])

    # Add final segment
    if current_segment['tokens']:
        current_segment['end_idx'] = len(predictions) - 1
        segments.append(current_segment)

    return segments


# Example usage and testing
if __name__ == "__main__":
    # Train the model
    trainer, model, tokenizer = train_code_switching_model(
        train_csv='train_sequences.csv',
        val_csv='val_sequences.csv',
        model_name='bert-base-multilingual-cased',
        output_dir='./tibetan_cs_model',
        num_epochs=5
    )

    # Test on new text (without tags)
    test_text = "dngos po kun gyi rang bzhin mchog dngos po kun gyi rang bzhin"
    predictions = predict_code_switching(test_text, model, tokenizer, return_probs=True)

    print("\n=== Predictions ===")
    for pred in predictions[:10]:  # Show first 10
        print(f"Token: {pred['token']:<20} Prediction: {pred['label']:<10}")
        if 'probabilities' in pred:
            print(f"  Probabilities: {pred['probabilities']}")

    # Analyze segments
    segments = analyze_predictions(predictions)
    print("\n=== Detected Segments ===")
    for seg in segments:
        print(f"Type: {seg['type']}, Tokens: {' '.join(seg['tokens'][:5])}...")


if __name__ == "__main__":
    """
    Simple usage example for Tibetan code-switching detection
    """

    import pandas as pd
    from organize_allo_auto_code_switching_3_classes import process_tibetan_3class, prepare_for_bert_training

    # Step 1: Process your .docx file
    print("Step 1: Processing .docx file...")
    token_df, seq_df = process_tibetan_3class(
        'classify_allo_auto/data/Nicola_Bajetta_rNam_gsum_bshad_pa_Auto_vs_Allo_signals_alo_and_auto_cleaned.docx',  # Replace with your file
        'classify_allo_auto/data/tokens_3class.csv',
        'classify_allo_auto/data/sequences_3class.csv',
        sequence_length=512
    )

    # Step 2: Create train/validation split
    print("\nStep 2: Creating train/validation split...")
    train_df, val_df = prepare_for_bert_training('classify_allo_auto/data/sequences_3class.csv')

    # Step 3: Train the model
    print("\nStep 3: Training BERT model...")
    trainer, model, tokenizer = train_code_switching_model(
        train_csv='train_sequences.csv',
        val_csv='val_sequences.csv',
        model_name='bert-base-multilingual-cased',
        output_dir='./tibetan_cs_model',
        num_epochs=3,  # Start with fewer epochs for testing
        batch_size=16,
        learning_rate=2e-5
    )

    # Step 4: Test on new text (without <allo>/<auto> tags)
    print("\nStep 4: Testing on new text...")
    # Example test text - replace with your actual test data
    test_texts = [
        "dngos po kun gyi rang bzhin mchog dngos po kun gyi",
        "zhes gsungs pa lta bu dang rnam rig sna tshogs",
        # Add more test sentences
    ]

    for test_text in test_texts:
        print(f"\nInput: {test_text}")
        predictions = predict_code_switching(test_text, model, tokenizer)

        # Show predictions
        switches_found = []
        for i, pred in enumerate(predictions):
            if pred['prediction'] > 0:  # Found a switch
                switches_found.append({
                    'position': i,
                    'token': pred['token'],
                    'switch_to': 'auto' if pred['prediction'] == 1 else 'allo'
                })

        if switches_found:
            print("Switches detected:")
            for switch in switches_found:
                print(f"  - Position {switch['position']}: '{switch['token']}' → {switch['switch_to']}")
        else:
            print("No switches detected")

    # Step 5: Save predictions for evaluation
    print("\nStep 5: Saving test predictions...")
    all_predictions = []
    for test_text in test_texts:
        preds = predict_code_switching(test_text, model, tokenizer, return_probs=True)
        all_predictions.extend(preds)

    # Convert to DataFrame for analysis
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv('test_predictions.csv', index=False)

    print("\nDone! Check the following files:")
    print("- tokens_3class.csv: Token-level annotations")
    print("- train_sequences.csv & val_sequences.csv: Training data")
    print("- ./tibetan_cs_model/: Trained model")
    print("- test_predictions.csv: Predictions on test data")

    # Quick stats
    print("\n=== Quick Statistics ===")
    print(f"Total training sequences: {len(train_df)}")
    print(f"Total validation sequences: {len(val_df)}")
    print(f"Sequences with switches in training: {train_df['contains_switch'].sum()}")
    print(f"Sequences with switches in validation: {val_df['contains_switch'].sum()}")
    import ipdb
    ipdb.set_trace()