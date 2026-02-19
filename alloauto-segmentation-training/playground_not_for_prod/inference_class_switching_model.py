import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class CodeSwitchingInference:
    def __init__(self, model_path='./proximity_cs_model_with_test/final_model'):
        """Load the trained model and tokenizer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Label mappings
        self.id2label = {0: 'no_switch', 1: 'to_auto', 2: 'to_allo'}

    def process_tagged_text(self, text):
        """
        Process text with <auto> and <allo> tags to create tokens and labels.
        Returns tokens and their positions where switches occur.
        """
        tokens = []
        expected_switches = []  # For comparison with predictions
        current_mode = 'auto'  # Assume starting with auto

        # Split by tags
        parts = re.split(r'(<auto>|<allo>)', text)

        for i, part in enumerate(parts):
            if part == '<auto>':
                if current_mode != 'auto':
                    expected_switches.append((len(tokens), 'to_auto'))
                current_mode = 'auto'
            elif part == '<allo>':
                if current_mode != 'allo':
                    expected_switches.append((len(tokens), 'to_allo'))
                current_mode = 'allo'
            elif part.strip():
                # Tokenize the text part
                words = part.strip().split()
                tokens.extend(words)

        return tokens, expected_switches

    def predict(self, tokens):
        """Run inference on tokenized input."""
        # Tokenize for BERT
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probs = torch.softmax(outputs.logits, dim=2)

        # Align predictions with original tokens
        word_ids = encoding.word_ids()
        aligned_predictions = []
        aligned_probs = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                pred = predictions[0][i].item()
                prob = probs[0][i].cpu().numpy()
                aligned_predictions.append(pred)
                aligned_probs.append(prob)
            previous_word_idx = word_idx

        # Trim to match token length
        aligned_predictions = aligned_predictions[:len(tokens)]
        aligned_probs = aligned_probs[:len(tokens)]

        return aligned_predictions, aligned_probs

    def analyze_text(self, tagged_text, confidence_threshold=0.5):
        """
        Full analysis pipeline for tagged text.
        """
        # Process text
        tokens, expected_switches = self.process_tagged_text(tagged_text)

        # Get predictions
        predictions, probs = self.predict(tokens)

        # Find predicted switches
        predicted_switches = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            if pred > 0:  # Switch detected
                confidence = prob[pred]
                if confidence >= confidence_threshold:
                    predicted_switches.append({
                        'position': i,
                        'token': tokens[i] if i < len(tokens) else '',
                        'type': self.id2label[pred],
                        'confidence': float(confidence)
                    })

        # Create result summary
        result = {
            'tokens': tokens,
            'expected_switches': expected_switches,
            'predicted_switches': predicted_switches,
            'predictions': predictions,
            'comparison': self.compare_switches(expected_switches, predicted_switches, tokens)
        }

        return result

    def compare_switches(self, expected, predicted, tokens):
        """Compare expected vs predicted switches."""
        comparison = []

        # Convert expected to dict for easier lookup
        expected_dict = {pos: switch_type for pos, switch_type in expected}

        # Check predictions
        for pred in predicted:
            pos = pred['position']
            is_correct = False
            is_close = False

            # Check exact match
            if pos in expected_dict:
                is_correct = (pred['type'] == expected_dict[pos].replace('_', ' '))

            # Check proximity (within 5 tokens)
            for exp_pos, exp_type in expected_dict.items():
                if abs(pos - exp_pos) <= 5:
                    is_close = True
                    break

            comparison.append({
                'position': pos,
                'token': tokens[pos] if pos < len(tokens) else '',
                'predicted': pred['type'],
                'is_correct': is_correct,
                'is_close': is_close,
                'confidence': pred['confidence']
            })

        return comparison

    def print_results(self, result):
        """Pretty print the results."""
        print("\n" + "=" * 60)
        print("CODE-SWITCHING ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\nTotal tokens: {len(result['tokens'])}")
        print(f"Expected switches: {len(result['expected_switches'])}")
        print(f"Predicted switches: {len(result['predicted_switches'])}")

        print("\n--- EXPECTED SWITCHES ---")
        for pos, switch_type in result['expected_switches']:
            token = result['tokens'][pos] if pos < len(result['tokens']) else '[END]'
            print(f"  Position {pos}: {switch_type} at '{token}'")

        print("\n--- PREDICTED SWITCHES ---")
        for pred in result['predicted_switches']:
            status = "✓" if any(
                c['is_correct'] for c in result['comparison'] if c['position'] == pred['position']) else "✗"
            print(
                f"  {status} Position {pred['position']}: {pred['type']} at '{pred['token']}' (conf: {pred['confidence']:.2f})")

        print("\n--- PERFORMANCE ---")
        if result['comparison']:
            correct = sum(1 for c in result['comparison'] if c['is_correct'])
            close = sum(1 for c in result['comparison'] if c['is_close'] and not c['is_correct'])
            print(f"  Exact matches: {correct}/{len(result['predicted_switches'])}")
            print(f"  Close matches (±5 tokens): {close}/{len(result['predicted_switches'])}")


# Usage example
if __name__ == "__main__":
    # Initialize inference
    inferencer = CodeSwitchingInference('levshechter/tibetan-code-switching-detector')
    # inferencer = CodeSwitchingInference('./classify_allo_auto/proximity_cs_model_with_test/final_model')

    # Your tagged text
    text = """<auto>thams cad la yang shes par bya'o de la ji ltar de chos thams cad kyi rang bzhin nam bdag nyid yin zhe na 'di nyid las gsungs pa
    <allo>dngos po kun gyi rang bzhin mchog dngos po kun gyi rang bzhin 'dzin skye med chos te sna tshogs ston chos kun ngo bo nyid 'chang ba
    <auto>zhes gsungs pa lta bu dang
    <allo>rnam rig sna tshogs gzugs don can rnam shes sna tshogs rgyud dang ldan"""

    # Analyze
    result = inferencer.analyze_text(text, confidence_threshold=0.5)

    # Print results
    inferencer.print_results(result)

    # Get raw predictions for further processing
    print("\n\nRaw predictions (first 50):", result['predictions'][:50])