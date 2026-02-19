# hackathon-alloauto/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import re


from classify_allo_auto.inference_fine_tuned_CS_v2 import (
    CodeSwitchingInference4Class
)

# ---------- Setup FastAPI app and CORS ----------

app = FastAPI(title="Code Switching API")

# CORS middleware to allow requests from the frontend running on a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],  
    allow_credentials= False,
)

inferencer = CodeSwitchingInference4Class('levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_all_data_no_labels_no_partial')

# ---------- Schemas ----------

class AnalyzeTaggedRequest(BaseModel):
    text: str
    confidence_threshold: float = 0.7
    use_filtering: bool = True

class ClassifyChunksRequest(BaseModel):
    text: str  

# --- NEW: run prediction over long text by 512-token windows and stitch back ---
def predict_full_tokens(tokens: List[str], max_len: int = 512):
    """
    Runs inferencer.predict() over the token list in non-overlapping windows
    of up to `max_len`, and concatenates predictions and probs back.
    Returns (all_predictions, all_probs) aligned 1:1 to `tokens`.
    """
    all_preds, all_probs = [], []
    pos = 0
    while pos < len(tokens):
        window = tokens[pos:pos + max_len]
        preds, probs = inferencer.predict(window)   # both length == len(window)
        all_preds.extend(preds)
        all_probs.extend(probs)
        pos += max_len
    return all_preds, all_probs

def map_probs4_to_allo_auto(prob_vec_4: List[float]) -> Dict[str, float]:
    """
    4-class -> 2-class mapping:
      allo = non_switch_allo (1) + switch_to_allo (3)
      auto = non_switch_auto (0) + switch_to_auto (2)
    """
    allo = float(prob_vec_4[1] + prob_vec_4[3])
    auto = float(prob_vec_4[0] + prob_vec_4[2])
    return {"allo": allo, "auto": auto}

def merge_adjacent_same_label(chunks: List[Dict[str, Any]], sep: str = " / ") -> List[Dict[str, Any]]:
    """
    UI-friendly merge on the backend:
    מאחד רשומות עוקבות עם אותו label.
    - הטקסט מחובר עם מפריד (sep).
    - ההסתברויות allo/auto מחושבות כממוצע פשוט משוקלל לפי גודל כל תת־חטיבה.
      (כאן כל תת־חטיבה נספרת כ-1, כפי שהחזרת מהחיזוי; אם תרצה משקל אחר – אפשר להתאים.)
    """
    if not chunks:
        return []
    out = []
    cur = dict(chunks[0])
    cur["originalCount"] = 1

    for r in chunks[1:]:
        if r.get("label") == cur.get("label"):
            cur["chunk"] = f'{cur["chunk"]}{sep}{r["chunk"]}'
            cur["allo"] = (cur["allo"] * cur["originalCount"] + r["allo"]) / (cur["originalCount"] + 1)
            cur["auto"] = (cur["auto"] * cur["originalCount"] + r["auto"]) / (cur["originalCount"] + 1)
            cur["originalCount"] += 1
        else:
            out.append(cur)
            cur = dict(r)
            cur["originalCount"] = 1
    out.append(cur)
    return out


# ----------- Routes -----------

@app.get("/health")
def health():
    return {"ok": True}



@app.post("/classify-chunks")
def classify_chunks(req: ClassifyChunksRequest):
    """
    Code-switch–based chunking (no '/' splitting):
    - Processes the entire input by running the model in consecutive windows,
      each limited to ≤512 model tokens (WordPiece).
    - Concatenates token-level predictions (4 classes), maps to allo/auto,
      and splits chunks at code-switch points (labels 2 or 3) across the full text.
    - No sliding overlap; windows are back-to-back for full coverage.
    """
    # 1) Split into words; tokenizer will handle subwords internally
    all_words = req.text.split()
    if not all_words:
        return {
            "results": [],
            "meta": {
                "windows": 0,
                "total_words": 0,
                "total_model_tokens": 0,
                "max_model_tokens_per_window": 512,
            },
        }

    # Helper: build back-to-back word windows that each fit into ≤512 model tokens
    def make_windows_by_model_tokens(words, max_model_tokens=512):
        windows = []
        pos = 0
        n = len(words)
        while pos < n:
            subset = words[pos:]
            enc = inferencer.tokenizer(
                subset,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_model_tokens,
                padding=False,
            )
            word_ids = enc.word_ids()
            valid = [i for i in word_ids if i is not None]
            if not valid:
                # Fallback: force-progress by taking at least one word
                end = min(pos + 1, n)
                enc_single = inferencer.tokenizer(
                    words[pos:end],
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_model_tokens,
                    padding=False,
                )
                kept_model_tokens = int(enc_single["input_ids"].shape[-1])
            else:
                last_word_index = max(valid) + 1           # count within `subset`
                end = pos + last_word_index                # absolute end (exclusive)
                kept_model_tokens = int(enc["input_ids"].shape[-1])

            windows.append((pos, end, kept_model_tokens))
            if end <= pos:  # safety to avoid infinite loop
                end = pos + 1
            pos = end
        return windows

    # 2) Build windows and run the model on each one, then concatenate predictions
    windows = make_windows_by_model_tokens(all_words, max_model_tokens=512)

    all_preds = []
    all_probs = []
    total_model_tokens = 0

    for start, end, kept_model_tokens in windows:
        window_words = all_words[start:end]
        preds, probs = inferencer.predict(window_words)  # len == len(window_words)
        all_preds.extend(preds)
        all_probs.extend(probs)
        total_model_tokens += kept_model_tokens

    # Sanity: everything must align to the full word sequence
    assert len(all_preds) == len(all_words)
    assert len(all_probs) == len(all_words)

    # 3) Collect global switch positions (labels 2=to_auto, 3=to_allo)
    switch_positions = [i for i, p in enumerate(all_preds) if p in (2, 3)]

    # 4) Build chunks across the entire sequence
    results = []
    cur_start = 0

    def map_probs4_to_allo_auto(prob_vec_4):
        # 4-class -> 2-class mapping
        # allo = non_switch_allo(1) + switch_to_allo(3)
        # auto = non_switch_auto(0) + switch_to_auto(2)
        allo = float(prob_vec_4[1] + prob_vec_4[3])
        auto = float(prob_vec_4[0] + prob_vec_4[2])
        return {"allo": allo, "auto": auto}

    def flush_chunk(start: int, end: int):
        """Add one chunk [start:end) with averaged allo/auto probabilities."""
        if end <= start:
            return
        chunk_probs = all_probs[start:end]
        allo_sum, auto_sum = 0.0, 0.0
        for prob_vec in chunk_probs:
            mapped = map_probs4_to_allo_auto(prob_vec)
            allo_sum += mapped["allo"]
            auto_sum += mapped["auto"]
        token_count = end - start
        allo_mean = float(allo_sum / token_count)
        auto_mean = float(auto_sum / token_count)
        label = "allo" if allo_mean >= auto_mean else "auto"
        chunk_text = " ".join(all_words[start:end])
        results.append({
            "chunk": chunk_text,
            "allo": round(allo_mean, 6),
            "auto": round(auto_mean, 6),
            "allo_sum": round(allo_sum, 6),
            "auto_sum": round(auto_sum, 6),
            "token_count": token_count,
            "label": label,

        })

    for s in switch_positions:
        if s > cur_start:
            flush_chunk(cur_start, s)    # up to (but excluding) the switch token
        cur_start = s                    # next chunk starts at the switch token

    # Tail
    n = len(all_words)
    if cur_start < n:
        flush_chunk(cur_start, n)

    return {
        "results": results,
        "meta": {
            "windows": len(windows),
            "total_words": len(all_words),
            "total_model_tokens": total_model_tokens,
            "max_model_tokens_per_window": 512,
        },
    }
