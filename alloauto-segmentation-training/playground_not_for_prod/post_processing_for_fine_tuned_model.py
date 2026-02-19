"""
Post-processing module for code-switching predictions
Enforces logical consistency rules for switch points
"""

import numpy as np
from typing import Union, List


def apply_post_processing_rules(predictions: Union[np.ndarray, List[int]],
                                min_tokens_between_switches: int = 2) -> Union[np.ndarray, List[int]]:
    """
    Apply post-processing rules to enforce logical consistency in predictions.

    Rules enforced:
    1. No adjacent switches (switches cannot be back-to-back)
    2. Switch→Allo (label 3) must follow Non-switch Auto (label 0)
    3. Switch→Auto (label 2) must follow Non-switch Allo (label 1)
    4. Minimum tokens between switches (default: 2)
       - After Switch→Allo: at least 2 Allo tokens before next switch
       - After Switch→Auto: at least 2 Auto tokens before next switch

    Label mapping:
        0: Non-switch Auto
        1: Non-switch Allo
        2: Switch→Auto
        3: Switch→Allo

    Args:
        predictions: Array or list of predicted labels
        min_tokens_between_switches: Minimum number of tokens required between switches

    Returns:
        Corrected predictions in same format as input
    """
    if len(predictions) == 0:
        return predictions

    # Convert to list for easier manipulation
    is_numpy = isinstance(predictions, np.ndarray)
    preds = predictions.tolist() if is_numpy else list(predictions)

    corrected = []
    current_mode = 0  # Start in Auto mode (0)
    tokens_since_last_switch = min_tokens_between_switches

    for i, pred in enumerate(preds):
        # Skip padding tokens (used in batched processing)
        if pred == -100:
            corrected.append(pred)
            continue

        # Identify if current prediction is a switch
        is_switch = pred in [2, 3]

        if is_switch:
            # RULE 1: No adjacent switches
            if tokens_since_last_switch == 0:
                # Previous token was a switch - convert to continuation
                corrected_label = current_mode  # Continue in current mode
                corrected.append(corrected_label)
                tokens_since_last_switch += 1
                continue

            # RULE 4: Minimum tokens between switches
            if tokens_since_last_switch < min_tokens_between_switches:
                # Too soon to switch - need more tokens in current mode
                corrected_label = current_mode  # Continue in current mode
                corrected.append(corrected_label)
                tokens_since_last_switch += 1
                continue

            # RULE 2 & 3: Mode consistency check
            if pred == 2:  # Switch→Auto
                if current_mode != 1:
                    # INVALID: Can only switch to Auto from Allo mode
                    # Convert to continuation in current mode
                    corrected.append(current_mode)
                    tokens_since_last_switch += 1
                else:
                    # VALID: Switching from Allo to Auto
                    corrected.append(2)
                    current_mode = 0  # Now in Auto mode
                    tokens_since_last_switch = 0  # Reset counter

            elif pred == 3:  # Switch→Allo
                if current_mode != 0:
                    # INVALID: Can only switch to Allo from Auto mode
                    # Convert to continuation in current mode
                    corrected.append(current_mode)
                    tokens_since_last_switch += 1
                else:
                    # VALID: Switching from Auto to Allo
                    corrected.append(3)
                    current_mode = 1  # Now in Allo mode
                    tokens_since_last_switch = 0  # Reset counter
        else:
            # Non-switch token (0 or 1)
            # Update current mode based on prediction
            if pred == 0:
                current_mode = 0  # Auto mode
            elif pred == 1:
                current_mode = 1  # Allo mode

            corrected.append(pred)
            tokens_since_last_switch += 1

    # Return in same format as input
    return np.array(corrected) if is_numpy else corrected


def verify_post_processing_rules(predictions: Union[np.ndarray, List[int]]) -> dict:
    """
    Verify that predictions follow all post-processing rules.
    Useful for debugging and validation.

    Args:
        predictions: Array or list of predicted labels

    Returns:
        Dictionary with violation counts and details
    """
    preds = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

    violations = {
        'adjacent_switches': [],
        'invalid_mode_transitions': [],
        'insufficient_spacing': [],
        'total_switches': 0
    }

    current_mode = 0
    tokens_since_last_switch = 0
    min_spacing = 2

    for i, pred in enumerate(preds):
        if pred == -100:
            continue

        is_switch = pred in [2, 3]

        if is_switch:
            violations['total_switches'] += 1

            # Check adjacent switches
            if tokens_since_last_switch == 0:
                violations['adjacent_switches'].append(i)

            # Check spacing
            if tokens_since_last_switch < min_spacing:
                violations['insufficient_spacing'].append({
                    'position': i,
                    'tokens_since_last': tokens_since_last_switch
                })

            # Check mode consistency
            if pred == 2 and current_mode != 1:
                violations['invalid_mode_transitions'].append({
                    'position': i,
                    'switch_type': '→Auto',
                    'current_mode': 'Auto' if current_mode == 0 else 'Allo'
                })
            elif pred == 3 and current_mode != 0:
                violations['invalid_mode_transitions'].append({
                    'position': i,
                    'switch_type': '→Allo',
                    'current_mode': 'Auto' if current_mode == 0 else 'Allo'
                })

            # Update state
            if pred == 2:
                current_mode = 0
            elif pred == 3:
                current_mode = 1
            tokens_since_last_switch = 0
        else:
            # Update mode
            if pred == 0:
                current_mode = 0
            elif pred == 1:
                current_mode = 1
            tokens_since_last_switch += 1

    return violations


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def process_model_predictions_with_post_processing(tokens: List[str],
                                                   tokenizer,
                                                   model,
                                                   device,
                                                   min_tokens_between_switches: int = 2) -> List[int]:
    """
    Complete pipeline: Get model predictions and apply post-processing.
    Works with any fine-tuned model (ALTO, mBERT, XLM-R, etc.)

    Args:
        tokens: List of input tokens
        tokenizer: Model tokenizer
        model: Fine-tuned model
        device: torch device
        min_tokens_between_switches: Minimum spacing between switches

    Returns:
        Post-processed predictions
    """
    import torch

    # Tokenize
    tokenizer_output = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in tokenizer_output.items()}

    # Get raw predictions
    with torch.no_grad():
        outputs = model(**inputs)
        raw_predictions = torch.argmax(outputs.logits, dim=2)

    # Align to original tokens
    word_ids = tokenizer_output.word_ids()
    aligned_preds = []
    previous_word_idx = None

    for j, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_preds.append(raw_predictions[0][j].item())
        previous_word_idx = word_idx

    # Apply post-processing
    corrected_preds = apply_post_processing_rules(
        aligned_preds,
        min_tokens_between_switches=min_tokens_between_switches
    )

    return corrected_preds

