# SmolDocling-F26-04

### Reproducing an Ultra-Compact Vision-Language Model for End-to-End Multi-Modal Document Conversion

**Track C — Research Paper Implementation**  
**Group ID:** F26-04  
**Course:** Artificial Intelligence — Spring 2026

---

## About This Project

This project is a reproduction study of the [SmolDocling paper](https://arxiv.org/abs/2503.11576) (Nassar et al., 2025). SmolDocling is a 256M-parameter vision-language model that converts document images into structured text using a unified markup format called DocTags. Despite being smaller than competing models, the paper claims SmolDocling matches or outperforms larger alternatives on equation recognition, full-page OCR, and table structure extraction.

We reproduce the paper's evaluation claims from Tables 2 and 4 using pretrained HuggingFace weights — no training or fine-tuning is performed. We compare SmolDocling against two baseline models (Nougat and TATR) across three benchmark datasets.

---

## Models

| Model | Parameters | Role | Link |
|-------|-----------|------|------|
| **SmolDocling-256M-preview** | 256M | Primary — unified document converter | [HuggingFace](https://huggingface.co/ds4sd/SmolDocling-256M-preview) |
| **Nougat-base** | 350M | Baseline — document-to-Markdown (equations + full-page OCR) | [HuggingFace](https://huggingface.co/facebook/nougat-base) |
| **TATR (Table Transformer)** | ~20M | Baseline — table structure recognition only | [HuggingFace](https://huggingface.co/microsoft/table-transformer-structure-recognition) |

**Why these baselines?**
- **Nougat** is the closest size competitor (350M vs 256M) and appears in the paper's Table 2. It outputs Mathpix Markdown rather than DocTags, making normalization necessary for fair comparison. Nougat is only evaluated on Im2LaTeX and DocLayNet — it was not designed for table structure and is not in the paper's Table 4.
- **TATR** is a dedicated table structure model. It competes with SmolDocling on PubTables-1M (TEDS metric) but cannot do OCR or equation recognition — highlighting the generalist vs. specialist trade-off.

---

## Datasets

| Dataset | Samples | Task | Metrics | Source |
|---------|---------|------|---------|--------|
| **Im2LaTeX-230K** | 238,329 | Equation Recognition | Edit Distance, F1, Precision, Recall, BLEU, METEOR | [Kaggle](https://www.kaggle.com/datasets/gregoryeritsyan/im2latex-230k) |
| **PubTables-1M** | 93,834 | Table Structure Recognition | TEDS (structure-only) | [HuggingFace](https://huggingface.co/datasets/bsmock/pubtables-1m) |
| **DocLayNet v1.2** | 4,999 (test) | Full-Page OCR & Layout | Edit Distance, F1, Precision, Recall, BLEU, METEOR | [HuggingFace](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2) |

Datasets are **not** included in this repo (they're too large). Each notebook downloads its dataset automatically during execution — see [Usage](#usage) below.

---

## Repository Structure

```
SmolDocling-F26-04/
│
├── README.md                          ← You are here
│
├── Phase 2 — Data Preprocessing
│   ├── im2latex_230k.ipynb            Downloads Im2LaTeX from Kaggle, normalizes LaTeX,
│   │                                  converts to DocTags CSV (238K rows)
│   ├── PubTables-1M.ipynb             Downloads PubTables XML annotations from HuggingFace,
│   │                                  converts to OTSL .txt files (93K tables)
│   ├── doclaynetpp.ipynb              Streams DocLayNet v1.2 test split, converts COCO
│   │                                  bounding boxes to DocTags JSONL + page PNGs
│   ├── preprocess.py                  Standalone PubTables OTSL converter
│   └── doclaynetpp.py                 Standalone DocLayNet preprocessor
│
├── Phase 3 — Model Inference
│   └── F26_04.ipynb                   Loads all 3 models, runs inference on all datasets,
│                                      saves 6 result files (3 CSVs + 3 JSONLs)
│
├── Phase 4 — Evaluation & Analysis
│   └── F26-04.ipynb                   Computes all 7 metrics, runs error analysis,
│                                      bias checks, and generates comparison tables
│
├── Reports
│   ├── F26-04_Phase2.pdf
│   ├── F26-04_Phase3.pdf
│   ├── F26-04_Phase4.pdf
│   ├── F26-04_Phase5.pdf
│   └── F26-04_Phase6_Report.pdf       Final technical report
│
└── results/                           (Generated at runtime — not committed)
    ├── smoldocling_latex_results.csv
    ├── smoldocling_doclaynet_results.jsonl
    ├── smoldocling_pubtables_results.csv
    ├── nougat_latex_results.csv
    ├── nougat_doclaynet_results.jsonl
    └── tatr_pubtables_results.csv
```

---

## Setup

### Prerequisites

- A **Google Colab** account (free tier works; T4 GPU runtime recommended)
- A **HuggingFace** account (for streaming datasets and downloading model weights)
- A **Kaggle** account + API token (`kaggle.json`) for the Im2LaTeX-230K dataset only
- Python 3.10+

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 (16 GB VRAM) | T4 or better |
| System RAM | 12 GB | 16 GB |
| Disk | 5 GB | 10 GB |
| Platform | Google Colab (free) | Colab Pro |

SmolDocling uses ~1 GB VRAM at bfloat16 precision. Nougat uses ~1.4 GB at float32. Both fit comfortably on a T4 with room to spare.

### Install Dependencies

Run this cell at the top of any notebook:

```python
# Core inference
!pip install -q transformers torch datasets Pillow tqdm pandas

# Metrics
!pip install -q python-Levenshtein nltk apted lxml

# OCR (for DocLayNet ground truth extraction)
!pip install -q pytesseract
!apt-get install -y tesseract-ocr

# SmolDocling output parsing
!pip install -q docling-core
```

### Mount Google Drive (recommended)

Colab runtimes are ephemeral — everything is lost when the session disconnects (~90 min idle or ~12 hours active). Always persist results to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/F26-04', exist_ok=True)
```

### Kaggle API Token (Im2LaTeX only)

1. Go to [kaggle.com](https://www.kaggle.com) → Profile → Settings → API → **Create New Token**
2. A `kaggle.json` file downloads to your machine
3. When the Im2LaTeX notebook prompts "Upload your kaggle.json", select that file

The other two datasets (PubTables-1M, DocLayNet) stream directly from HuggingFace and need no API key.

---

## Usage

The project runs in three sequential stages. Each stage produces output files consumed by the next.

### Stage 1: Preprocess Datasets (Phase 2)

Open each preprocessing notebook in Colab and run all cells:

| Notebook | Downloads From | Produces | Approx. Time |
|----------|---------------|----------|-------------|
| `im2latex_230k.ipynb` | Kaggle (via API) | `im2latex_ready.csv` — 238K equation pairs in DocTags format | ~15 min |
| `PubTables-1M.ipynb` | HuggingFace (wget) | `ground_truth_otsl/` — 93K OTSL structure-only .txt files | ~10 min |
| `doclaynetpp.ipynb` | HuggingFace (streaming) | `doclaynet_ready.jsonl` + page PNGs | ~20 min |

**Save outputs to Drive after each notebook:**
```python
import shutil
shutil.copytree('/content/output/', '/content/drive/MyDrive/F26-04/phase2_data/')
```

### Stage 2: Run Model Inference (Phase 3)

Open `F26_04.ipynb` in Colab with **T4 GPU runtime** (Runtime → Change runtime type → T4 GPU). The notebook:

1. Mounts Drive and loads preprocessed data from Stage 1
2. Downloads and loads SmolDocling-256M-preview from HuggingFace
3. Runs inference on all 3 datasets using paper-specified prompts:
   - `"Convert formula to LaTeX"` — Im2LaTeX (max_new_tokens=512)
   - `"Convert this page to docling"` — DocLayNet (max_new_tokens=2048)
   - `"Convert table to OTSL"` — PubTables (max_new_tokens=1024)
4. Clears GPU memory, loads Nougat-base, runs on Im2LaTeX + DocLayNet
5. Clears GPU memory, loads TATR, runs on PubTables-1M
6. Saves 6 result files to Drive

**Inference settings:**

| Setting | SmolDocling | Nougat | TATR |
|---------|------------|--------|------|
| Precision | bfloat16 | float32 | default |
| Decoding | greedy (`do_sample=False`) | greedy | — |
| `repetition_penalty` | 1.2 | 1.2 | — |
| `no_repeat_ngram_size` | — | 3 | — |
| VRAM usage | ~1 GB | ~1.4 GB | ~0.5 GB |

**Expected runtime:** ~90 minutes total on T4 GPU.

### Stage 3: Compute Metrics & Analysis (Phase 4)

Open `F26-04.ipynb` in Colab. The notebook:

1. Loads the 6 result files from Stage 2
2. Strips markup from outputs:
   - SmolDocling: removes `<tag>` wrappers and `<loc_N>` coordinates
   - Nougat: removes `\(...\)`, `\[...\]` math delimiters, markdown headers, bold/italic, table pipes
3. Computes metrics per-sample, then averages:
   - **Edit Distance** — Levenshtein character-level, normalized by `max(len(pred), len(gt))`
   - **F1 / Precision / Recall** — bag-of-words token overlap
   - **BLEU** — sentence-level with method-1 smoothing
   - **METEOR** — WordNet synonym-aware
   - **TEDS** — OTSL → tag tree, APTED edit distance (tables only)
4. Runs error analysis: length-based breakdown, numeric content effects, category-level bias
5. Outputs comparison tables matching paper Tables 2 and 4

---

## Evaluation Metrics

| Metric | Used For | Direction | Definition |
|--------|----------|-----------|------------|
| Edit Distance | Equations, Full-page | ↓ Lower is better | Levenshtein char distance / max(len(pred), len(gt)) |
| F1-score | Equations, Full-page | ↑ Higher is better | Harmonic mean of precision and recall (token-level) |
| Precision | Equations, Full-page | ↑ Higher is better | Fraction of predicted tokens that are correct |
| Recall | Equations, Full-page | ↑ Higher is better | Fraction of ground-truth tokens that were predicted |
| BLEU | Equations, Full-page | ↑ Higher is better | N-gram overlap with smoothing |
| METEOR | Equations, Full-page | ↑ Higher is better | Like BLEU but synonym-aware via WordNet |
| TEDS | Tables | ↑ Higher is better | Tree-edit-distance similarity on parsed OTSL |

---

## Paper Results We Reproduce

### Table 2 — Equations (Im2LaTeX-230K)

| Model | Size | ED ↓ | F1 ↑ | Precision ↑ | Recall ↑ | BLEU ↑ | METEOR ↑ |
|-------|------|------|------|------------|----------|--------|----------|
| SmolDocling | 256M | 0.11 | 0.95 | 0.96 | 0.95 | 0.83 | 0.89 |
| Nougat (base) | 350M | 0.62 | 0.60 | 0.60 | 0.53 | 0.33 | 0.41 |

### Table 2 — Full-Page OCR (DocLayNet v1.2)

| Model | Size | ED ↓ | F1 ↑ | Precision ↑ | Recall ↑ | BLEU ↑ | METEOR ↑ |
|-------|------|------|------|------------|----------|--------|----------|
| SmolDocling | 256M | 0.48 | 0.80 | 0.89 | 0.79 | 0.58 | 0.67 |
| Nougat (base) | 350M | 0.62 | 0.66 | 0.72 | 0.67 | 0.44 | 0.54 |

### Table 4 — Table Structure (PubTables-1M, structure-only TEDS)

| Model | TEDS ↑ |
|-------|--------|
| SmolDocling | 0.88 |

---

## Known Issues & Reproducibility Notes

**Edit Distance normalization matters.** The paper does not explicitly state its normalization formula. Using `ED / len(gt)` produces unbounded scores > 1.0 on DocLayNet. We use `ED / max(len(pred), len(gt))` to cap at 1.0.

**TEDS dialect sensitivity.** OTSL has multiple tokenization conventions. Our initial TEDS scored 0.27 before aligning to the paper's token vocabulary — after correction it matched at ~0.87. Always verify the OTSL token set (`<fcel>`, `<ecel>`, `<ched>`, `<rhed>`, `<srow>`, `<nl>`) matches between prediction and ground truth.

**DocLayNet sample composition bias.** The first N samples of the test split are disproportionately financial reports. The paper evaluates on the full 4,999-page test set. Evaluating on a small subset may produce lower F1 due to category imbalance.

**Nougat on isolated equations.** Nougat is trained on full academic pages. Feeding it cropped equation images produces near-zero F1 — this is expected behavior (domain mismatch), not a bug.

**Colab session volatility.** Sessions disconnect after ~90 min idle or ~12 hours active. Always save intermediate results to Google Drive between stages. The notebooks include Drive-mounting and save cells for this reason.

**SmolDocling repetition loops.** The model occasionally enters endless-repetition mode (documented in paper Section 5.3). The `repetition_penalty=1.2` setting mitigates this. Nougat additionally uses `no_repeat_ngram_size=3` as a hard guard.

---

## Project Timeline

| Phase | Focus | Deliverables |
|-------|-------|-------------|
| Phase 1 | Literature review & paper selection | Proposal document |
| Phase 2 | Data collection & preprocessing | 3 preprocessing notebooks, Phase 2 report |
| Phase 3 | Model implementation & inference | Inference notebook (`F26_04.ipynb`), Phase 3 report |
| Phase 4 | Evaluation & error analysis | Evaluation notebook (`F26-04.ipynb`), Phase 4 report |
| Phase 5 | Consolidated analysis & insights | Phase 5 final report |
| Phase 6 | Final technical report & documentation | Phase 6 report, updated README |

---

## References

1. Nassar, A. et al. (2025). *SmolDocling: An Ultra-Compact Vision-Language Model for End-to-End Multi-Modal Document Conversion.* [arXiv:2503.11576](https://arxiv.org/abs/2503.11576)
2. Blecher, L. et al. (2023). *Nougat: Neural Optical Understanding for Academic Documents.* [arXiv:2308.13418](https://arxiv.org/abs/2308.13418)
3. Smock, B. et al. (2022). *PubTables-1M: Towards Comprehensive Table Extraction from Unstructured Documents.* CVPR 2022.
4. Pfitzmann, B. et al. (2022). *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation.* KDD 2022.
5. Zhang, Z. et al. (2017). *Image-to-Markup Generation with Coarse-to-Fine Attention.* ICML 2017.

---

## License

This project is for academic purposes only. All models and datasets are used under their respective licenses. SmolDocling is released under Apache 2.0. Nougat is released under CC-BY-NC. TATR is released under MIT.
