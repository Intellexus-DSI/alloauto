# ALLO/AUTO: Segmenting Allochthonous and Autochthonous Regions in Classical Tibetan Texts

This repository contains the code, data, and models accompanying the paper:

> **Segmenting Allochthonous and Autochthonous Regions in Classical Tibetan Texts**
>
> Accepted at **LREC 2026**

## Abstract

We introduce a new computational framework for segmenting Classical Tibetan texts into *autochthonous* and *allochthonous* regions, distinguishing between indigenous Tibetan compositions and translated materials, primarily from Sanskrit sources. To support this task, we release the first annotated Tibetan corpus for ALLO/AUTO segmentation and evaluate several multilingual encoders, including mBERT and XLM-R, fine-tuned for sequence labeling. Our best model achieves strong alignment with expert annotations, showing that multilingual representations can effectively capture philological boundaries in low-resource settings. This work contributes new resources and methods for computational philology and sheds light on the linguistic markers that trace the intercultural transmission of Buddhist thought in Tibet.

## Repository Structure

```
.
├── dataset/                            # Annotated corpus and preprocessed splits
│   ├── annotated-data/                 #   Final annotated train/val/test CSVs
│   ├── annotated-data-raw/             #   Raw expert annotations
│   └── preprocessed_augmented/         #   Augmented training data with allo/auto samples
│
├── alloauto-segmentation-training/     # Main training and inference pipeline
│   ├── data/                           #   Raw Tibetan text data (.docx/.txt)
│   ├── data_preprocess/                #   Data preprocessing scripts
│   ├── fine_tune_ALTO_scripts/         #   ALTO architecture fine-tuning scripts
│   ├── fine_tune_benchmark_scripts/    #   Benchmark model fine-tuning (mBERT, XLM-R, CINO, etc.)
│   ├── evaluation_scripts/             #   Model evaluation scripts
│   └── inference/                      #   Inference engine and post-processing
│       └── post_inference/             #     Prediction smoothing and segmentation
│
├── alloauto-presentation/              # Web demo and API
│   ├── api/                            #   FastAPI backend for model serving
│   └── web/                            #   Interactive web interface
│
├── closed-models/                      # Evaluation against closed-source LLMs
│
├── annotation tool/                    # Auxiliary tools for corpus annotation
│   └── (verse detection, text processing utilities)
│
└── quote highlighter/                  # Quote detection utility for Tibetan texts
```

## Pipeline Overview

The segmentation pipeline consists of the following stages:

### 1. Data Preprocessing

Converts raw annotated Tibetan texts into labeled sequences for training.

```bash
cd alloauto-segmentation-training

# Step 1: Create initial train/val/test splits
python data_preprocess/data_preprocess_for_fine_tune.py

# Step 2: Augment with balanced allo/auto samples
python data_preprocess/data_preprocess_for_fine_tune_with_allo_auto.py
```

### 2. Fine-Tuning

**ALTO Architecture** (our proposed model with segmentation-aware loss):
```bash
python fine_tune_ALTO_scripts/fine_tune_all_models_on_ALTO_arch_aug_data.py
```

**Benchmark Models** (mBERT, XLM-R, CINO, Tibetan-RoBERTa):
```bash
python fine_tune_benchmark_scripts/fine_tune_all_benchmarks_standard_same_params_ALTO.py
```

Fine-tuning requires a GPU. The target GPU can be configured via the environment variable at the top of each script.

### 3. Evaluation

Evaluates all fine-tuned models on the test set, reporting precision, recall, F1, and F-beta scores:

```bash
python evaluation_scripts/evaluate_all_models.py            # 4-class NER
python evaluation_scripts/evaluate_all_models_3_class_focus_copy.py  # 3-class NER
```

### 4. Inference

Run inference on new Tibetan texts (.docx or .txt):

```bash
python inference/inference_ALTO.py
```

The inference engine uses a sliding-window approach (512-token windows with configurable stride) and averages predictions across overlapping windows.

### 5. Post-Processing

Smooth word-level predictions into coherent segments:

```bash
python inference/post_inference/smooth_predictions.py
```

Short segments below a configurable threshold are merged into adjacent segments for cleaner output.

## Label Schema

| Label | Description |
|-------|-------------|
| 0 | Non-switch Autochthonous (indigenous Tibetan composition) |
| 1 | Non-switch Allochthonous (translated material) |
| 2 | Switch to Autochthonous |
| 3 | Switch to Allochthonous |

## Models

Fine-tuned models are available on HuggingFace. The main model used in the paper:
- [`levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_MUL_SEG_RUNI`](https://huggingface.co/levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_MUL_SEG_RUNI)

## Web Demo

An interactive web demo is available at: https://mcskware.github.io/alloauto/web/

## Requirements

### Training and Inference

```bash
pip install -r alloauto-segmentation-training/requirements.txt
```

Key dependencies: PyTorch, Transformers, scikit-learn, pytorch-crf, ONNX Runtime

### Closed-Model Evaluation

```bash
pip install -r closed-models/requirements.txt
```

Requires API keys for the respective providers (set via environment variables).

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{alloauto2026,
  title={Segmenting Allochthonous and Autochthonous Regions in Classical Tibetan Texts},
  booktitle={Proceedings of the Language Resources and Evaluation Conference (LREC)},
  year={2026}
}
```

## License

Please see the repository for license details.
