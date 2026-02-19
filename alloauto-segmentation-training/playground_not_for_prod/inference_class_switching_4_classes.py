import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class CodeSwitchingInference4Class:
    def __init__(self, model_path='./classify_allo_auto/combined_model_4class/final_model'):
        """Load the trained 4-class model and tokenizer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=False)
        self.model.to(self.device)
        self.model.eval()

        # Updated label mappings for 4-class system
        self.id2label = {
            0: 'non_switch_auto',
            1: 'non_switch_allo',
            2: 'switch_to_auto',
            3: 'switch_to_allo'
        }

    def robust_preprocess_text(self, text_content):
        """Preprocess text to handle various tag formats."""
        # Handle all possible tag variations
        tag_patterns = [
            (r'<\s*ALLO\s*>', '<allo>'),
            (r'<\s*allo\s*>', '<allo>'),
            (r'<\s*Allo\s*>', '<allo>'),
            (r'<\s*AUTO\s*>', '<auto>'),
            (r'<\s*auto\s*>', '<auto>'),
            (r'<\s*Auto\s*>', '<auto>'),
        ]

        for pattern, replacement in tag_patterns:
            text_content = re.sub(pattern, replacement, text_content, flags=re.IGNORECASE)

        # Clean up whitespace
        text_content = text_content.replace('\n', ' ')
        text_content = text_content.replace('\r', ' ')
        text_content = text_content.replace('\t', ' ')
        text_content = re.sub(r'\s+', ' ', text_content)

        # Ensure spaces around tags
        text_content = re.sub(r'<allo>', ' <allo> ', text_content)
        text_content = re.sub(r'<auto>', ' <auto> ', text_content)
        text_content = re.sub(r'\s+', ' ', text_content)

        return text_content.strip()

    def process_tagged_text(self, text):
        """
        Process text with <auto> and <allo> tags to create tokens and expected labels.
        Now handles 4-class system.
        """
        # Preprocess text first
        text = self.robust_preprocess_text(text)

        tokens = []
        expected_labels = []
        expected_switches = []
        current_mode = None  # Start with unknown mode

        # Split by tags
        parts = re.split(r'(<auto>|<allo>)', text)

        for i, part in enumerate(parts):
            if part == '<auto>':
                next_mode = 'auto'
            elif part == '<allo>':
                next_mode = 'allo'
            elif part.strip():
                words = part.strip().split()

                for word_idx, word in enumerate(words):
                    tokens.append(word)

                    # Determine expected label (same logic as training)
                    if word_idx == 0 and i > 0:  # First word after a tag
                        prev_part = parts[i - 1]
                        if prev_part == '<auto>':
                            if current_mode == 'allo':
                                expected_labels.append(2)  # Switch TO auto
                                expected_switches.append((len(tokens) - 1, 'switch_to_auto'))
                            else:
                                expected_labels.append(0)  # Continue in auto
                            current_mode = 'auto'
                        elif prev_part == '<allo>':
                            if current_mode == 'auto':
                                expected_labels.append(3)  # Switch TO allo
                                expected_switches.append((len(tokens) - 1, 'switch_to_allo'))
                            else:
                                expected_labels.append(1)  # Continue in allo
                            current_mode = 'allo'
                        else:
                            # Continue current mode
                            if current_mode == 'auto':
                                expected_labels.append(0)
                            elif current_mode == 'allo':
                                expected_labels.append(1)
                            else:
                                expected_labels.append(0)  # Default to auto
                    else:
                        # Not a switch point
                        if current_mode == 'auto':
                            expected_labels.append(0)
                        elif current_mode == 'allo':
                            expected_labels.append(1)
                        else:
                            expected_labels.append(0)  # Default

        return tokens, expected_labels, expected_switches

    def predict(self, tokens, confidence_threshold=0.5):
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

    def filter_predictions_with_confidence(self, predictions, probs, tokens,
                                           confidence_threshold=0.8,
                                           context_window=10):
        """
        Filter predictions to reduce false positives.
        Only keep switch predictions with high confidence and proper context.
        """
        filtered_predictions = []

        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            if pred in [2, 3]:  # Switch classes
                # Check confidence
                confidence = prob[pred]
                if confidence < confidence_threshold:
                    # Revert to most likely non-switch class
                    filtered_predictions.append(0 if prob[0] > prob[1] else 1)
                else:
                    # Check context - do we have consistent modes before/after?
                    before = predictions[max(0, i - context_window):i]
                    after = predictions[i + 1:min(len(predictions), i + context_window)]

                    # If context is too mixed, likely a false positive
                    if len(before) > 5:
                        before_mode = max(set(b for b in before if b in [0, 1]),
                                          key=before.count) if before else None
                    else:
                        before_mode = None

                    if len(after) > 5:
                        after_mode = max(set(a for a in after if a in [0, 1]),
                                         key=after.count) if after else None
                    else:
                        after_mode = None

                    # Keep switch only if modes are different and consistent
                    if before_mode is not None and after_mode is not None:
                        if (pred == 2 and before_mode == 1 and after_mode == 0) or \
                                (pred == 3 and before_mode == 0 and after_mode == 1):
                            filtered_predictions.append(pred)
                        else:
                            filtered_predictions.append(before_mode if before_mode is not None else 0)
                    else:
                        # Not enough context, keep if high confidence
                        if confidence > 0.9:
                            filtered_predictions.append(pred)
                        else:
                            filtered_predictions.append(0 if prob[0] > prob[1] else 1)
            else:
                filtered_predictions.append(pred)

        return filtered_predictions

    def analyze_text(self, tagged_text, confidence_threshold=0.7, use_filtering=True):
        """
        Full analysis pipeline for tagged text with 4-class system.
        Modified to return label names with their probabilities.
        """
        # Process text
        tokens, expected_labels, expected_switches = self.process_tagged_text(tagged_text)

        # Get predictions
        predictions, probs = self.predict(tokens)

        # Optionally filter predictions
        if use_filtering:
            filtered_predictions = self.filter_predictions_with_confidence(
                predictions, probs, tokens, confidence_threshold
            )
        else:
            filtered_predictions = predictions

        # Convert numeric labels to readable strings with probabilities
        expected_labels_readable = [self.id2label[label] for label in expected_labels]

        # Create predictions with probabilities
        predictions_with_probs = []
        raw_predictions_with_probs = []

        for i, (pred, prob_array) in enumerate(zip(filtered_predictions, probs)):
            # Get the probability for the predicted class
            pred_prob = prob_array[pred]
            predictions_with_probs.append({
                'label': self.id2label[pred],
                'probability': float(pred_prob),
                'all_probs': {
                    'non_switch_auto': float(prob_array[0]),
                    'non_switch_allo': float(prob_array[1]),
                    'switch_to_auto': float(prob_array[2]),
                    'switch_to_allo': float(prob_array[3])
                }
            })

        for i, (pred, prob_array) in enumerate(zip(predictions, probs)):
            pred_prob = prob_array[pred]
            raw_predictions_with_probs.append({
                'label': self.id2label[pred],
                'probability': float(pred_prob),
                'all_probs': {
                    'non_switch_auto': float(prob_array[0]),
                    'non_switch_allo': float(prob_array[1]),
                    'switch_to_auto': float(prob_array[2]),
                    'switch_to_allo': float(prob_array[3])
                }
            })

        # Find predicted switches
        predicted_switches = []
        for i, (pred, prob) in enumerate(zip(filtered_predictions, probs)):
            if pred in [2, 3]:  # Switch classes
                confidence = prob[pred]
                predicted_switches.append({
                    'position': i,
                    'token': tokens[i] if i < len(tokens) else '',
                    'type': self.id2label[pred],
                    'confidence': float(confidence),
                    'current_mode': 'auto' if pred == 2 else 'allo'
                })

        # Create result summary with readable labels and probabilities
        result = {
            'tokens': tokens,
            'expected_labels': expected_labels_readable,
            'expected_switches': expected_switches,
            'predicted_switches': predicted_switches,
            'predictions': predictions_with_probs,  # Now contains dicts with label and probability
            'raw_predictions': raw_predictions_with_probs,  # Now contains dicts with label and probability
            'filtered': use_filtering,
            'comparison': self.compare_switches(expected_switches, predicted_switches, tokens)
        }

        return result

    def convert_predictions_to_labels(self, predictions):
        """Convert numeric predictions to readable labels."""
        return [self.id2label[pred] for pred in predictions]

    def visualize_predictions(self, result, max_tokens_per_line=15):
        """Visualize tokens with their predicted labels in a readable format."""
        tokens = result['tokens']

        # Extract labels from the dictionary structure
        pred_labels = [pred_dict['label'] for pred_dict in result['predictions']]
        expected_labels = result['expected_labels']  # These are already strings

        print("\n" + "=" * 80)
        print("TOKEN-LEVEL PREDICTIONS")
        print("=" * 80)

        # Process in chunks for better readability
        for i in range(0, len(tokens), max_tokens_per_line):
            chunk_tokens = tokens[i:i + max_tokens_per_line]
            chunk_preds = pred_labels[i:i + max_tokens_per_line]
            chunk_expected = expected_labels[i:i + max_tokens_per_line]

            print(f"\nTokens {i}-{min(i + max_tokens_per_line - 1, len(tokens) - 1)}:")
            print("-" * 60)

            # Print tokens
            print("Tokens:   ", end="")
            for token in chunk_tokens:
                print(f"{token[:8]:<10}", end="")
            print()

            # Print expected labels with color coding
            print("Expected: ", end="")
            for label in chunk_expected:
                if 'switch' in label:
                    print(f"\033[91m{label[:10]:<10}\033[0m", end="")  # Red for switches
                elif label == 'non_switch_allo':
                    print(f"\033[94m{'allo':<10}\033[0m", end="")  # Blue for allo
                else:
                    print(f"\033[92m{'auto':<10}\033[0m", end="")  # Green for auto
            print()

            # Print predicted labels with color coding
            print("Predicted:", end="")
            for j, label in enumerate(chunk_preds):
                if 'switch' in label:
                    print(f"\033[91m{label[:10]:<10}\033[0m", end="")  # Red for switches
                elif label == 'non_switch_allo':
                    print(f"\033[94m{'allo':<10}\033[0m", end="")  # Blue for allo
                else:
                    print(f"\033[92m{'auto':<10}\033[0m", end="")  # Green for auto
            print()

            # Mark mismatches
            print("Match:    ", end="")
            for j in range(len(chunk_preds)):
                if chunk_preds[j] == chunk_expected[j]:
                    print(f"{'✓':<10}", end="")
                else:
                    print(f"\033[93m{'✗':<10}\033[0m", end="")  # Yellow for mismatches
            print()

    def create_readable_summary(self, result):
        """Create a summary with readable labels instead of numbers."""
        tokens = result['tokens']

        # Extract labels from the dictionary structure
        pred_labels = [pred_dict['label'] for pred_dict in result['predictions']]
        expected_labels = result['expected_labels']  # Already strings

        # Count occurrences
        pred_counts = {}
        exp_counts = {}

        for label in self.id2label.values():
            pred_counts[label] = pred_labels.count(label)
            exp_counts[label] = expected_labels.count(label)

        # Create DataFrame for nice display
        import pandas as pd

        summary_data = {
            'Label': list(self.id2label.values()),
            'Expected Count': [exp_counts[label] for label in self.id2label.values()],
            'Predicted Count': [pred_counts[label] for label in self.id2label.values()],
            'Difference': [pred_counts[label] - exp_counts[label] for label in self.id2label.values()]
        }

        df = pd.DataFrame(summary_data)

        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False))

        # Show accuracy by class
        correct_by_class = {label: 0 for label in self.id2label.values()}
        total_by_class = {label: 0 for label in self.id2label.values()}

        for pred, exp in zip(pred_labels, expected_labels):
            total_by_class[exp] += 1
            if pred == exp:
                correct_by_class[exp] += 1

        print("\n" + "=" * 60)
        print("ACCURACY BY CLASS")
        print("=" * 60)

        for label in self.id2label.values():
            if total_by_class[label] > 0:
                acc = correct_by_class[label] / total_by_class[label] * 100
                print(f"{label:<20}: {correct_by_class[label]}/{total_by_class[label]} ({acc:.1f}%)")
            else:
                print(f"{label:<20}: No examples")
    # Updated print_results method
    def print_results_enhanced(self, result):
        """Enhanced print results with readable labels."""
        print("\n" + "=" * 60)
        print("4-CLASS CODE-SWITCHING ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\nTotal tokens: {len(result['tokens'])}")
        print(f"Expected switches: {len(result['expected_switches'])}")
        print(f"Predicted switches: {len(result['predicted_switches'])}")
        print(f"Filtering: {'Enabled' if result['filtered'] else 'Disabled'}")

        # Create readable summary
        self.create_readable_summary(result)

        # Visualize predictions
        self.visualize_predictions(result)

        # Show switch analysis
        print("\n" + "=" * 60)
        print("SWITCH ANALYSIS")
        print("=" * 60)

        print("\nExpected Switches:")
        for pos, switch_type in result['expected_switches']:
            token = result['tokens'][pos] if pos < len(result['tokens']) else '[END]'
            # Show context around switch
            context_before = ' '.join(result['tokens'][max(0, pos - 3):pos])
            context_after = ' '.join(result['tokens'][pos + 1:min(len(result['tokens']), pos + 4)])
            print(f"  Pos {pos:3d}: {switch_type:<20} | ...{context_before} [{token}] {context_after}...")

        if result['predicted_switches']:
            print("\nPredicted Switches:")
            for pred in result['predicted_switches']:
                pos = pred['position']
                token = result['tokens'][pos] if pos < len(result['tokens']) else '[END]'
                context_before = ' '.join(result['tokens'][max(0, pos - 3):pos])
                context_after = ' '.join(result['tokens'][pos + 1:min(len(result['tokens']), pos + 4)])
                print(f"  Pos {pos:3d}: {pred['type']:<20} (conf: {pred['confidence']:.3f}) | "
                      f"...{context_before} [{token}] {context_after}...")
        else:
            print("\nNo switches predicted!")

        # Performance metrics
        comp = result['comparison']
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Exact matches:    {comp['exact_matches']}")
        print(f"Close matches:    {comp['close_matches']} (within ±5 tokens)")
        print(f"False positives:  {comp['false_positives']}")
        print(f"Missed switches:  {comp['missed_switches']}")

        total_predicted = len(result['predicted_switches'])
        total_expected = len(result['expected_switches'])

        if total_predicted > 0:
            precision = (comp['exact_matches'] + comp['close_matches']) / total_predicted
            print(f"Precision:        {precision:.3f}")
        else:
            print(f"Precision:        N/A (no predictions)")

        if total_expected > 0:
            recall = (comp['exact_matches'] + comp['close_matches']) / total_expected
            print(f"Recall:           {recall:.3f}")

            if total_predicted > 0 and (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"F1 Score:         {f1:.3f}")
        else:
            print(f"Recall:           N/A (no expected switches)")

    def compare_switches(self, expected, predicted, tokens, tolerance=5):
        """Compare expected vs predicted switches."""
        comparison = {
            'exact_matches': 0,
            'close_matches': 0,
            'false_positives': 0,
            'missed_switches': 0,
            'details': []
        }

        matched_expected = set()
        matched_predicted = set()

        # Check each predicted switch
        for j, pred in enumerate(predicted):
            pos = pred['position']
            found_match = False

            for i, (exp_pos, exp_type) in enumerate(expected):
                distance = abs(pos - exp_pos)

                if distance == 0 and pred['type'] == exp_type:
                    # Exact match
                    comparison['exact_matches'] += 1
                    matched_expected.add(i)
                    matched_predicted.add(j)
                    found_match = True
                    comparison['details'].append({
                        'type': 'exact_match',
                        'position': pos,
                        'token': tokens[pos] if pos < len(tokens) else '',
                        'predicted': pred['type'],
                        'expected': exp_type
                    })
                    break
                elif distance <= tolerance and pred['type'] == exp_type:
                    # Close match
                    comparison['close_matches'] += 1
                    matched_expected.add(i)
                    matched_predicted.add(j)
                    found_match = True
                    comparison['details'].append({
                        'type': 'close_match',
                        'position': pos,
                        'token': tokens[pos] if pos < len(tokens) else '',
                        'predicted': pred['type'],
                        'expected': exp_type,
                        'distance': distance
                    })
                    break

            if not found_match:
                # False positive
                comparison['false_positives'] += 1
                comparison['details'].append({
                    'type': 'false_positive',
                    'position': pos,
                    'token': tokens[pos] if pos < len(tokens) else '',
                    'predicted': pred['type']
                })

        # Check for missed switches
        for i, (exp_pos, exp_type) in enumerate(expected):
            if i not in matched_expected:
                comparison['missed_switches'] += 1
                comparison['details'].append({
                    'type': 'missed',
                    'position': exp_pos,
                    'token': tokens[exp_pos] if exp_pos < len(tokens) else '',
                    'expected': exp_type
                })

        return comparison

    def print_results(self, result):
        """Pretty print the results for 4-class system."""
        print("\n" + "=" * 60)
        print("4-CLASS CODE-SWITCHING ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\nTotal tokens: {len(result['tokens'])}")
        print(f"Expected switches: {len(result['expected_switches'])}")
        print(f"Predicted switches: {len(result['predicted_switches'])}")
        print(f"Filtering: {'Enabled' if result['filtered'] else 'Disabled'}")

        # Print mode distribution in predictions
        # Now predictions are dicts, so we need to extract the labels
        mode_counts = {
            'non_switch_auto': 0,
            'non_switch_allo': 0,
            'switch_to_auto': 0,
            'switch_to_allo': 0
        }

        for pred_dict in result['predictions']:
            label = pred_dict['label']
            mode_counts[label] = mode_counts.get(label, 0) + 1

        print("\n--- MODE DISTRIBUTION ---")
        print(f"  Non-switch Auto: {mode_counts['non_switch_auto']}")
        print(f"  Non-switch Allo: {mode_counts['non_switch_allo']}")
        print(f"  Switch to Auto: {mode_counts['switch_to_auto']}")
        print(f"  Switch to Allo: {mode_counts['switch_to_allo']}")

        print("\n--- EXPECTED SWITCHES ---")
        for pos, switch_type in result['expected_switches']:
            token = result['tokens'][pos] if pos < len(result['tokens']) else '[END]'
            print(f"  Position {pos}: {switch_type} at '{token}'")

        print("\n--- PREDICTED SWITCHES ---")
        if result['predicted_switches']:
            for pred in result['predicted_switches']:
                print(f"  Position {pred['position']}: {pred['type']} at '{pred['token']}' "
                      f"(conf: {pred['confidence']:.3f})")
        else:
            print("  No switches predicted")

        print("\n--- PERFORMANCE SUMMARY ---")
        comp = result['comparison']
        print(f"  Exact matches: {comp['exact_matches']}")
        print(f"  Close matches (±5 tokens): {comp['close_matches']}")
        print(f"  False positives: {comp['false_positives']}")
        print(f"  Missed switches: {comp['missed_switches']}")

        # Calculate metrics
        total_predicted = len(result['predicted_switches'])
        total_expected = len(result['expected_switches'])

        if total_predicted > 0:
            precision = (comp['exact_matches'] + comp['close_matches']) / total_predicted
            print(f"  Precision: {precision:.3f}")

        if total_expected > 0:
            recall = (comp['exact_matches'] + comp['close_matches']) / total_expected
            print(f"  Recall: {recall:.3f}")

            if total_predicted > 0 and (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"  F1 Score: {f1:.3f}")

# Usage example
if __name__ == "__main__":
    # Initialize inference with your trained model
    inferencer = CodeSwitchingInference4Class('levshechter/tibetan-code-switching-detector')
    # Or use from HuggingFace:
    # inferencer = CodeSwitchingInference4Class('levshechter/tibetan-code-switching-detector')

    # Your tagged text
    text = """<auto>thams cad la yang shes par bya'o de la ji ltar de chos thams cad kyi rang bzhin nam bdag nyid yin zhe na 'di nyid las gsungs pa
    <allo>dngos po kun gyi rang bzhin mchog dngos po kun gyi rang bzhin 'dzin skye med chos te sna tshogs ston chos kun ngo bo nyid 'chang ba
    <auto>zhes gsungs pa lta bu dang
    <allo>rnam rig sna tshogs gzugs don can rnam shes sna tshogs rgyud dang ldan"""

    # Analyze with filtering enabled
    print("WITH FILTERING:")
    result = inferencer.analyze_text(text, confidence_threshold=0.7, use_filtering=True)
    inferencer.print_results(result)
    inferencer.print_results_enhanced(result)

    # Analyze without filtering to see raw predictions
    print("\n\nWITHOUT FILTERING:")
    result = inferencer.analyze_text(text, confidence_threshold=0.7, use_filtering=False)
    inferencer.print_results(result)
    import ipdb
    ipdb.set_trace()