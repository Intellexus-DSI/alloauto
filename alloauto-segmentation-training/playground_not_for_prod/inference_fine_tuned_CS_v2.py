import torch
import torch.nn.functional as F
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class CodeSwitchingInference4Class:
    def __init__(self, model_path='./classify_allo_auto/tibetan_code_switching_constrained_model/final_model'):
        """Load the trained 4-class model and tokenizer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Updated label mappings for 4-class system
        self.id2label = {
            0: 'non_switch_auto',
            1: 'non_switch_allo',
            2: 'switch_to_auto',
            3: 'switch_to_allo'
        }

    def clean_and_normalize_text(self, text_content):
        """
        EXACTLY match the preprocessing from training
        """
        # First normalize all tag variations
        text_content = re.sub(r'<\s*AUTO\s*>', '<auto>', text_content, flags=re.IGNORECASE)
        text_content = re.sub(r'<\s*ALLO\s*>', '<allo>', text_content, flags=re.IGNORECASE)

        # Remove ALL newlines and replace with spaces (matching training)
        text_content = text_content.replace('\n', ' ')
        text_content = text_content.replace('\r', ' ')
        text_content = text_content.replace('\t', ' ')

        # Remove any separator lines (multiple dashes, underscores, etc)
        text_content = re.sub(r'[-_=]{3,}', ' ', text_content)

        # Clean up multiple spaces
        text_content = re.sub(r'\s+', ' ', text_content)

        # Ensure proper spacing around tags for splitting
        text_content = re.sub(r'<auto>', ' <auto> ', text_content)
        text_content = re.sub(r'<allo>', ' <allo> ', text_content)

        # Final cleanup of multiple spaces
        text_content = re.sub(r'\s+', ' ', text_content)

        return text_content.strip()

    def apply_transition_constraints(self, predictions, current_mode_start=0):
        """
        Apply logical constraints to predictions (from training code):
        - If in Auto mode (0), can only switch to Allo (3)
        - If in Allo mode (1), can only switch to Auto (2)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        corrected_predictions = predictions.copy()
        current_mode = current_mode_start  # Start with provided mode (0=Auto by default)

        for i in range(len(predictions)):
            pred = predictions[i]

            if pred == -100:  # Skip padding
                continue

            # Check for invalid transitions
            if pred == 2:  # Switch to Auto
                if current_mode == 0:  # Already in Auto - INVALID
                    # Change to continuation in Auto
                    corrected_predictions[i] = 0
                else:  # Was in Allo - VALID
                    current_mode = 0

            elif pred == 3:  # Switch to Allo
                if current_mode == 1:  # Already in Allo - INVALID
                    # Change to continuation in Allo
                    corrected_predictions[i] = 1
                else:  # Was in Auto - VALID
                    current_mode = 1

            elif pred == 0:  # Continue in Auto
                current_mode = 0

            elif pred == 1:  # Continue in Allo
                current_mode = 1

        return corrected_predictions

    def process_tagged_text(self, text):
        """
        Process text with <auto> and <allo> tags to create tokens and expected labels.
        ENSURES first token gets a label.
        """
        # Preprocess text using training preprocessing
        text = self.clean_and_normalize_text(text)

        tokens = []
        expected_labels = []
        expected_switches = []
        current_mode = 'auto'  # DEFAULT START MODE: AUTO (matching training)

        # Split by tags
        parts = re.split(r'(<auto>|<allo>)', text)

        for i, part in enumerate(parts):
            if part == '<auto>':
                continue
            elif part == '<allo>':
                continue
            elif part.strip():
                words = part.strip().split()

                # Determine what mode this segment should be in
                segment_mode = None
                for j in range(i - 1, -1, -1):
                    if parts[j] == '<auto>':
                        segment_mode = 'auto'
                        break
                    elif parts[j] == '<allo>':
                        segment_mode = 'allo'
                        break

                if segment_mode is None:
                    segment_mode = 'auto'  # Default to auto (matching training)

                for word_idx, word in enumerate(words):
                    tokens.append(word)

                    # Determine label based on position and mode change
                    if word_idx == 0 and current_mode != segment_mode:
                        # This is a switch point
                        if segment_mode == 'auto':
                            expected_labels.append(2)  # Switch to auto
                            expected_switches.append((len(tokens) - 1, 'switch_to_auto'))
                        else:
                            expected_labels.append(3)  # Switch to allo
                            expected_switches.append((len(tokens) - 1, 'switch_to_allo'))
                    else:
                        # Continuation
                        if segment_mode == 'auto':
                            expected_labels.append(0)
                        else:
                            expected_labels.append(1)

                current_mode = segment_mode

        return tokens, expected_labels, expected_switches

    def predict(self, tokens, apply_constraints=True, confidence_threshold=0.5):
        """
        Run inference on tokenized input with optional constraint application.
        """
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

        # Apply transition constraints if enabled
        if apply_constraints:
            # Determine starting mode from first prediction if available
            start_mode = 0  # Default to auto
            if len(aligned_predictions) > 0:
                if aligned_predictions[0] in [1, 3]:
                    start_mode = 1  # Start in allo if first token suggests it

            aligned_predictions = self.apply_transition_constraints(
                aligned_predictions,
                current_mode_start=start_mode
            )

        return aligned_predictions, aligned_probs

    def filter_predictions_with_confidence(self, predictions, probs, tokens,
                                           confidence_threshold=0.8,
                                           proximity_tolerance=5):
        """
        Filter predictions using confidence and proximity logic from training.
        """
        filtered_predictions = predictions.copy()

        # Find all predicted switches
        switch_positions = []
        for i, pred in enumerate(predictions):
            if pred in [2, 3]:  # Switch classes
                switch_positions.append(i)

        # For each predicted switch, check confidence and proximity
        for i in switch_positions:
            confidence = probs[i][predictions[i]]

            if confidence < confidence_threshold:
                # Low confidence - revert to continuation
                if predictions[i] == 2:  # Was switch to auto
                    filtered_predictions[i] = 0  # Continue auto
                else:  # Was switch to allo
                    filtered_predictions[i] = 1  # Continue allo
            else:
                # Check if there's another switch too close
                for j in switch_positions:
                    if i != j and abs(i - j) <= proximity_tolerance:
                        # Too close to another switch - keep only the higher confidence one
                        if probs[j][predictions[j]] > confidence:
                            # Other switch has higher confidence, remove this one
                            if predictions[i] == 2:
                                filtered_predictions[i] = 0
                            else:
                                filtered_predictions[i] = 1
                            break

        return filtered_predictions

    def analyze_text(self, tagged_text, confidence_threshold=0.7,
                     use_filtering=True, apply_constraints=True):
        """
        Full analysis pipeline for tagged text with 4-class system.
        Now includes constraint application matching training.
        """
        # Process text
        tokens, expected_labels, expected_switches = self.process_tagged_text(tagged_text)

        if len(tokens) == 0:
            print("Warning: No tokens found after processing!")
            return None

        # Get predictions with optional constraints
        predictions, probs = self.predict(tokens, apply_constraints=apply_constraints)

        # Store raw predictions before filtering
        raw_predictions = predictions.copy()

        # Optionally filter predictions
        if use_filtering:
            predictions = self.filter_predictions_with_confidence(
                predictions, probs, tokens,
                confidence_threshold=confidence_threshold,
                proximity_tolerance=5  # Match training
            )

        # Convert numeric labels to readable strings
        expected_labels_readable = [self.id2label[label] for label in expected_labels]

        # Create predictions with probabilities
        predictions_with_probs = []
        raw_predictions_with_probs = []

        for i, (pred, prob_array) in enumerate(zip(predictions, probs)):
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

        for i, (pred, prob_array) in enumerate(zip(raw_predictions, probs)):
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

        # Find predicted switches (with proximity tolerance)
        predicted_switches = []
        for i, pred in enumerate(predictions):
            if pred in [2, 3]:  # Switch classes
                confidence = probs[i][pred]
                predicted_switches.append({
                    'position': i,
                    'token': tokens[i] if i < len(tokens) else '',
                    'type': self.id2label[pred],
                    'confidence': float(confidence),
                    'switch_direction': 'to_auto' if pred == 2 else 'to_allo'
                })

        # Create result summary
        result = {
            'tokens': tokens,
            'expected_labels': expected_labels_readable,
            'expected_switches': expected_switches,
            'predicted_switches': predicted_switches,
            'predictions': predictions_with_probs,
            'raw_predictions': raw_predictions_with_probs,
            'filtered': use_filtering,
            'constraints_applied': apply_constraints,
            'comparison': self.compare_switches_with_proximity(
                expected_switches, predicted_switches, tokens
            )
        }

        return result

    def compare_switches_with_proximity(self, expected, predicted, tokens, tolerance=5):
        """
        Compare switches with proximity tolerance (matching training evaluation).
        Switch types must match exactly.
        """
        comparison = {
            'exact_matches': 0,
            'proximity_matches': 0,
            'false_positives': 0,
            'missed_switches': 0,
            'details': []
        }

        matched_expected = set()
        matched_predicted = set()

        # Group expected switches by type
        expected_to_auto = [(i, pos) for i, (pos, stype) in enumerate(expected)
                            if stype == 'switch_to_auto']
        expected_to_allo = [(i, pos) for i, (pos, stype) in enumerate(expected)
                            if stype == 'switch_to_allo']

        # Check each predicted switch
        for j, pred in enumerate(predicted):
            pos = pred['position']
            pred_type = pred['type']
            found_match = False

            # Only check against same type
            if pred_type == 'switch_to_auto':
                check_list = expected_to_auto
            else:
                check_list = expected_to_allo

            for idx, exp_pos in check_list:
                distance = abs(pos - exp_pos)

                if distance == 0:
                    # Exact match
                    comparison['exact_matches'] += 1
                    matched_expected.add(idx)
                    matched_predicted.add(j)
                    found_match = True
                    break
                elif distance <= tolerance:
                    # Proximity match
                    if idx not in matched_expected:  # Don't double-match
                        comparison['proximity_matches'] += 1
                        matched_expected.add(idx)
                        matched_predicted.add(j)
                        found_match = True
                        break

            if not found_match:
                comparison['false_positives'] += 1

        # Check for missed switches
        comparison['missed_switches'] = len(expected) - len(matched_expected)

        return comparison

    def visualize_predictions(self, result, max_tokens_per_line=15):
        """Enhanced visualization with constraint indicators."""
        tokens = result['tokens']
        pred_labels = [pred_dict['label'] for pred_dict in result['predictions']]
        expected_labels = result['expected_labels']

        print("\n" + "=" * 80)
        print("TOKEN-LEVEL PREDICTIONS")
        if result.get('constraints_applied'):
            print("(With Logical Transition Constraints Applied)")
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
                    print(f"\033[91m{label[0:10]:<10}\033[0m", end="")  # Red for switches
                elif label == 'non_switch_allo':
                    print(f"\033[94m{'allo':<10}\033[0m", end="")  # Blue for allo
                else:
                    print(f"\033[92m{'auto':<10}\033[0m", end="")  # Green for auto
            print()

            # Print predicted labels with color coding
            print("Predicted:", end="")
            for j, label in enumerate(chunk_preds):
                if 'switch' in label:
                    print(f"\033[91m{label[0:10]:<10}\033[0m", end="")  # Red for switches
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

    def print_results(self, result):
        """Enhanced results printing with proximity metrics."""
        print("\n" + "=" * 60)
        print("4-CLASS CODE-SWITCHING ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\nTotal tokens: {len(result['tokens'])}")
        print(f"Expected switches: {len(result['expected_switches'])}")
        print(f"Predicted switches: {len(result['predicted_switches'])}")
        print(f"Filtering: {'Enabled' if result['filtered'] else 'Disabled'}")
        print(f"Constraints: {'Applied' if result.get('constraints_applied') else 'Disabled'}")

        # Visualize predictions
        self.visualize_predictions(result)

        # Show switch analysis
        print("\n" + "=" * 60)
        print("SWITCH ANALYSIS (5-token proximity tolerance)")
        print("=" * 60)

        print("\nExpected Switches:")
        for pos, switch_type in result['expected_switches']:
            token = result['tokens'][pos] if pos < len(result['tokens']) else '[END]'
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

        # Performance metrics with proximity
        comp = result['comparison']
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS (Proximity-Aware)")
        print("=" * 60)
        print(f"Exact matches:       {comp['exact_matches']}")
        print(f"Proximity matches:   {comp['proximity_matches']} (within ±5 tokens)")
        print(f"Total correct:       {comp['exact_matches'] + comp['proximity_matches']}")
        print(f"False positives:     {comp['false_positives']}")
        print(f"Missed switches:     {comp['missed_switches']}")

        total_predicted = len(result['predicted_switches'])
        total_expected = len(result['expected_switches'])

        if total_predicted > 0:
            precision = (comp['exact_matches'] + comp['proximity_matches']) / total_predicted
            print(f"Precision:           {precision:.3f}")
        else:
            print(f"Precision:           N/A (no predictions)")

        if total_expected > 0:
            recall = (comp['exact_matches'] + comp['proximity_matches']) / total_expected
            print(f"Recall:              {recall:.3f}")

            if total_predicted > 0:
                precision = (comp['exact_matches'] + comp['proximity_matches']) / total_predicted
                if (precision + recall) > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    print(f"F1 Score:            {f1:.3f}")
        else:
            print(f"Recall:              N/A (no expected switches)")

    # Add these methods to your existing class:

    def process_unlabeled_text(self, text):
        """Process text WITHOUT tags - just tokenize it."""
        # Clean text (remove any accidental tags if present)
        text = re.sub(r'<[^>]+>', '', text)

        # Basic cleaning
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'[-_=]{3,}', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Tokenize
        tokens = text.split()
        return tokens

    def detect_if_labeled(self, text):
        """Detect if text has <auto> or <allo> tags."""
        auto_pattern = r'<\s*[Aa][Uu][Tt][Oo]\s*>'
        allo_pattern = r'<\s*[Aa][Ll][Ll][Oo]\s*>'

        has_auto = bool(re.search(auto_pattern, text))
        has_allo = bool(re.search(allo_pattern, text))

        return has_auto or has_allo

    def get_simple_results(self, text, confidence_threshold=0.7,
                           use_filtering=True, apply_constraints=True):
        """
        Returns a simple list of tuples for both labeled and unlabeled data:
        - If labeled: (word, true_label, pred_label, probability)
        - If unlabeled: (word, None, pred_label, probability)
        """
        # Detect if text is labeled
        is_labeled = self.detect_if_labeled(text)

        if is_labeled:
            # Use your existing analyze_text method
            result = self.analyze_text(text, confidence_threshold, use_filtering)

            tokens = result['tokens']
            expected_labels = result['expected_labels']
            predictions = result['predictions']

            output_list = []
            for i in range(len(tokens)):
                word = tokens[i]
                true_label = expected_labels[i]
                pred_dict = predictions[i]
                pred_label = pred_dict['label']
                probability = pred_dict['probability']

                output_list.append((word, true_label, pred_label, probability))
        else:
            # Process as unlabeled text
            tokens = self.process_unlabeled_text(text)

            if len(tokens) == 0:
                return []

            # Get predictions
            predictions_numeric, probs = self.predict(tokens)

            # Apply filtering if requested
            if use_filtering:
                predictions_numeric = self.filter_predictions_with_confidence(
                    predictions_numeric, probs, tokens, confidence_threshold
                )

            # Build tuple list with None for true labels
            output_list = []
            for i in range(len(tokens)):
                word = tokens[i]
                true_label = None  # No true label for unlabeled data
                pred_label = self.id2label[predictions_numeric[i]]
                probability = float(probs[i][predictions_numeric[i]])

                output_list.append((word, true_label, pred_label, probability))

        return output_list


# Usage example
if __name__ == "__main__":
    # Initialize inference with your trained model
    inferencer = CodeSwitchingInference4Class(
        "levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie")

    # Your tagged text
    # text = """<auto>thams cad la yang shes par bya'o de la ji ltar de chos thams cad kyi rang bzhin nam bdag nyid yin zhe na 'di nyid las gsungs pa
    # <allo>dngos po kun gyi rang bzhin mchog dngos po kun gyi rang bzhin 'dzin skye med chos te sna tshogs ston chos kun ngo bo nyid 'chang ba
    # <auto>zhes gsungs pa lta bu dang
    # <allo>rnam rig sna tshogs gzugs don can rnam shes sna tshogs rgyud dang ldan"""
    text = """nyid las skyes pa yang yin pa'i phyir te/ tshul bcu po 'di ni/ 'khor ba dang mya ngan las 'das pa'i chos thams cad la yang shes par bya'o// de la ji ltar/ de chos thams cad kyi rang bzhin nam bdag nyid yin zhe na/ 'di nyid las gsungs pa/ <allo>dngos po kun gyi rang bzhin mchog// dngos po kun gyi rang bzhin 'dzin// skye med chos te sna tshogs ston// chos kun ngo bo nyid 'chang ba//<auto>zhes gsungs pa lta bu dang /<allo>rnam rig sna tshogs gzugs don can// rnam shes sna tshogs rgyud dang ldan// zhes gsungs pa dang / <allo>sdug bsngal gsum gyi sdug bsngal zhi////<auto> zhes pa dang / <allo>rnam pa bcu po don bcu'i don// <auto> zhes pa dang // <allo>don 'grub bsam pa grub pa ste// kun tu rtog pa thams cad spangs// <auto>shes bya ba la sos pa lta bu'i skabs su bshad pa/ dngos po thams cad kyi rang bzhin dang / dngos po med pa'i chos kyi rang bzhin dang / chos thams cad kyi ngo bo nyid """

    print("=" * 80)
    print("TESTING WITH ALL FEATURES ENABLED")
    print("=" * 80)

    # Analyze with constraints and filtering (like training)
    result = inferencer.analyze_text(
        text,
        confidence_threshold=0.7,
        use_filtering=True,
        apply_constraints=True  # Apply logical constraints
    )
    inferencer.print_results(result)

    print("\n\n" + "=" * 80)
    print("TESTING WITHOUT CONSTRAINTS (RAW PREDICTIONS)")
    print("=" * 80)

    # Compare without constraints to see difference
    result_raw = inferencer.analyze_text(
        text,
        confidence_threshold=0.7,
        use_filtering=False,
        apply_constraints=False
    )
    inferencer.print_results(result_raw)

    # Show constraint effect
    if result and result_raw:
        print("\n" + "=" * 60)
        print("CONSTRAINT EFFECT ANALYSIS")
        print("=" * 60)

        constrained_switches = len(result['predicted_switches'])
        raw_switches = len(result_raw['predicted_switches'])

        print(f"Raw predictions:        {raw_switches} switches")
        print(f"After constraints:      {constrained_switches} switches")
        print(f"Removed by constraints: {raw_switches - constrained_switches}")

    simple_results = inferencer.get_simple_results(text)

    # Print first few to verify
    for i, (word, true_label, pred_label, prob) in enumerate(simple_results):
        print(f"{i}: ('{word}', '{true_label}', '{pred_label}', {prob:.3f})")

    import ipdb

    ipdb.set_trace()