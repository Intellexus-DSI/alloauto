import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
import logging
# from skwarelog import sl_init
from tqdm import tqdm

# logger = sl_init()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ignore_chars: List[str] = ["/", "@", "#", ";", ":", "(", ")"]


def clean_line(line: str) -> str:
    """
    Clean a line by removing ignored characters.
    """
    for char in ignore_chars:
        line = line.replace(char, "")
    # replace all sequences of more than one space with a space
    line = re.sub(r"\s+", " ", line).strip()
    return line


def get_syllables(line: str) -> Set[str]:
    """
    Extract syllables from a cleaned line.
    """
    cleaned = clean_line(line)
    if not cleaned:
        return set()
    return set(cleaned.split())


def calculate_line_similarity(sungbum_line: str, derge_line: str) -> float:
    """
    Calculate similarity between two lines based on shared syllables.
    Returns average of:
    (a) sungbum syllables found in derge / total sungbum syllables
    (b) Derge syllables found in sungbum / total derge syllables
    """
    sungbum_syllables = get_syllables(sungbum_line)
    derge_syllables = get_syllables(derge_line)

    if not sungbum_syllables and not derge_syllables:
        return 1.0  # Both empty
    if not sungbum_syllables or not derge_syllables:
        return 0.0  # one empty, one not

    # Calculate bidirectional overlap
    sungbum_in_derge = len(sungbum_syllables & derge_syllables) / len(sungbum_syllables)
    derge_in_sungbum = len(derge_syllables & sungbum_syllables) / len(derge_syllables)

    return (sungbum_in_derge + derge_in_sungbum) / 2.0


def find_verse_quotation_probability(
    sungbum_verse: List[str],
    derge_lines: List[str],
    exact_matches: Dict[int, List[int]],
) -> float:
    """
    Calculate probability that the sungbum verse is a quotation from Derge.
    """
    if not exact_matches:
        return 0.0

    verse_length = len(sungbum_verse)
    total_similarity = 0.0
    similarity_count = 0

    # For each exact match, try to find the best alignment
    best_alignment_score = 0.0

    for sungbum_idx, derge_indices in exact_matches.items():
        for derge_idx in derge_indices:
            # Try alignment starting from this match point
            alignment_score = 0.0
            alignment_count = 0

            # Check lines before and after the match
            for offset in range(
                -min(sungbum_idx, derge_idx),
                min(verse_length - sungbum_idx, len(derge_lines) - derge_idx),
            ):
                sungbum_line_idx = sungbum_idx + offset
                derge_line_idx = derge_idx + offset

                if 0 <= sungbum_line_idx < verse_length and 0 <= derge_line_idx < len(
                    derge_lines
                ):
                    similarity = calculate_line_similarity(
                        sungbum_verse[sungbum_line_idx], derge_lines[derge_line_idx]
                    )
                    alignment_score += similarity
                    alignment_count += 1

            if alignment_count > 0:
                avg_alignment_score = alignment_score / alignment_count
                best_alignment_score = max(best_alignment_score, avg_alignment_score)

    return best_alignment_score


def analyze_verse_matches(input_dir: str, output_dir: str, verbose: bool = False):
    """
    Analyze verse analysis files and add Derge matching information.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the Derge corpus lines with their indices
    logger.info("Loading Derge corpus...")
    derge_lines = []
    derge_line_to_indices = defaultdict(list)  # Maps cleaned line to list of indices

    derge_corpus_path = "./derge_tshad_split_ewts"  # Change to your Derge corpus path

    for root, _, files in os.walk(derge_corpus_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        cleaned_line = clean_line(line.strip())
                        if cleaned_line:
                            derge_line_to_indices[cleaned_line].append(len(derge_lines))
                            derge_lines.append(cleaned_line)

    logger.info(f"Loaded {len(derge_lines)} Derge lines")

    # Process each verse analysis file
    processed_files = 0
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing files", unit="file"):
            if file.endswith("_verse_analysis.json"): # and file.startswith("NG"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)

                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Process each verse entry
                for entry in data:
                    sungbum_lines = entry["lines"]
                    exact_matches = {}
                    total_exact_matches = 0

                    # Find exact matches for each line
                    for i, line in enumerate(sungbum_lines):
                        cleaned_line = clean_line(line)
                        if cleaned_line in derge_line_to_indices:
                            exact_matches[i] = derge_line_to_indices[cleaned_line]
                            total_exact_matches += len(
                                derge_line_to_indices[cleaned_line]
                            )

                    # Calculate quotation probability
                    quotation_prob = find_verse_quotation_probability(
                        sungbum_lines, derge_lines, exact_matches
                    )

                    # Add new fields to the entry
                    entry["num_exact_derge_matches"] = total_exact_matches
                    entry["probability_of_derge_quotation"] = round(quotation_prob, 4)

                    if verbose and total_exact_matches > 0:
                        logger.debug(
                            f"Verse in {file}: {total_exact_matches} matches, "
                            f"quotation prob: {quotation_prob:.4f}"
                        )

                # Save the enhanced data
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                processed_files += 1

                if verbose:
                    logger.debug(f"Processed {input_path} -> {output_path}")

    logger.info(f"Finished processing {processed_files} files")


def print_statistics(output_dir: str):
    """
    Print statistics about the matching results.
    """
    total_verses = 0
    verses_with_matches = 0
    high_quotation_prob = 0
    match_counts = []
    quotation_probs = []

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_verse_analysis.json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    data = json.load(f)

                for entry in data:
                    total_verses += 1
                    matches = entry["num_exact_derge_matches"]
                    prob = entry["probability_of_derge_quotation"]

                    if matches > 0:
                        verses_with_matches += 1
                        match_counts.append(matches)

                    quotation_probs.append(prob)

                    if prob > 0.8:
                        high_quotation_prob += 1

    logger.info("Statistics:")
    logger.info(f"Total verses analyzed: {total_verses}")
    logger.info(
        f"Verses with exact matches: {verses_with_matches} ({verses_with_matches / total_verses * 100:.1f}%)"
    )
    logger.info(
        f"Verses with high quotation probability (>0.8): {high_quotation_prob} ({high_quotation_prob / total_verses * 100:.1f}%)"
    )

    if match_counts:
        logger.info(
            f"Average exact matches per verse (for verses with matches): {sum(match_counts) / len(match_counts):.2f}"
        )

    if quotation_probs:
        logger.info(
            f"Average quotation probability: {sum(quotation_probs) / len(quotation_probs):.4f}"
        )


if __name__ == "__main__":
    input_directory = "./per_line_step_1_output"  # Change to your input directory
    output_directory = "./per_line_step_3_output"  # Change to your output directory

    analyze_verse_matches(input_directory, output_directory, verbose=True)
    print_statistics(output_directory)
    logger.info("Finished analyzing verse matches with Derge.")
