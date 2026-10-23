# IQLoc Replication Package

## [Improving IR-based bug localization with semantics-driven query reduction](https://www.sciencedirect.com/science/article/pii/S0164121226000932)

Authors: **Asif Mohammed Samir** and **Mohammad Masudur Rahman**  
Published in **Journal of Systems and Software** (Volume 237, 2026) 

Replication package for end-to-end bug localization using a fixed-path A→E pipeline in `src/IQLoc`.

## Abstract

Despite decades of research, software bug localization remains challenging due to heterogeneous content and inherent ambiguities in bug reports. Existing methods, such as Information Retrieval (IR)-based approaches, often attempt to match source documents to bug reports, overlooking the context and semantics of the source code. On the other hand, Large Language Models (LLMs) (e.g., Transformer models) show promising results in understanding both texts and code. However, they have not yet been adapted well to localize software bugs using bug reports. They could also be data or resource-intensive. To bridge this gap, we propose, IQLoc, a novel approach that capitalizes on the strengths of both IR and LLMs for bug localization. In particular, we leverage the transformer-based model’s understanding of code semantics to reason about its suspiciousness and to reformulate search queries and thus enhance bug localization using Information Retrieval. To evaluate IQLoc, we refine the Bench4BL benchmark dataset and extend it by incorporating  ≈ 30% more recent bug reports, resulting in a benchmark containing  ≈ 7.5K bug reports. We evaluated IQLoc using three performance metrics and compare it against eight baseline techniques. Experimental results demonstrate its superiority, achieving up to 100.40% and 78.08% in MAP, 61.49% and 64.58% in MRR, and 76.98% and 100.90% in HIT@K for the test bug reports with random and time-wise splits, respectively. Moreover, IQLoc improves MAP by 118.70% for bug reports with stack traces, 111.87% for those that include code elements, and 127.45% for those containing only descriptions in natural language. By integrating program semantic understanding into Information Retrieval, IQLoc mitigates several longstanding challenges of traditional IR-based approaches in bug localization.


## Overview

The pipeline runs in five stages:

1. **A** (`a_Cache_ES_Results.py`): Retrieves top-K files from Elasticsearch and parses Java methods.
2. **B** (`b_Generate_BR_Keywords.py`): Generates bug-report keywords.
3. **C** (`c_Generate_CE_Score.py`): Scores methods with a cross-encoder.
4. **D** (`d_Generate_Source_Keywords.py`): Extracts source keywords from high-confidence methods.
5. **E** (`e_Localize_Bug.py`): Re-retrieves and evaluates localization performance.

All scripts are configured with fixed internal paths (no CLI arguments required).


## Repository Layout

- `Dataset/`: Input bug report datasets.
- `Output/Caching/`: Cached retrieval and CE scoring artifacts.
- `Output/Generation/`: Generated BR/source keyword artifacts.
- `Output/Localization/`: Final localization records and metrics.
- `src/IQLoc/`: Main A→E replication scripts.
- `src/IR/`: Elasticsearch indexing/retrieval utilities.
- `src/IR_Reretrieval/`: Re-retrieval utilities used in stage E.
- `src/Utils/`: JSON IO, parser, and evaluation helpers.

## Pre-requisites

- Python `3.10+`
- Elasticsearch (local instance, typically `http://localhost:9200`)
- Indexed source-code documents for your target projects/versions
- Optional NVIDIA CUDA-enabled GPU for faster model inference

## Installing Required Packages

### 1) Create and activate a virtual environment

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

### 2) Install PyTorch

Choose one option depending on your environment.

#### Option A: CPU-only (safe default)

```powershell
pip install torch torchvision torchaudio
```

#### Option B: CUDA-enabled (example)

Use a command from the official PyTorch selector for your CUDA/runtime combination: https://pytorch.org/get-started/locally/

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 3) Install Elasticsearch

#### Windows

1. Download Elasticsearch ZIP from: https://www.elastic.co/downloads/elasticsearch
2. Extract the archive.
3. Start Elasticsearch:

```powershell
cd <path-to-elasticsearch>
.\bin\elasticsearch.bat
```

Verify in browser or terminal:

```powershell
Invoke-WebRequest http://localhost:9200 | Select-Object -ExpandProperty Content
```

#### Linux (Ubuntu/Debian)

```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-<version>-amd64.deb
sudo dpkg -i elasticsearch-<version>-amd64.deb
sudo systemctl start elasticsearch
sudo systemctl enable elasticsearch
```

### 4) Install all Python requirements

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Setup

Current stage-A default input path is:

- `Dataset/Bench4BLExtended.json`

Place your input dataset at that exact location before running stage A.

## Paths to Update (if needed)

By default, the pipeline uses fixed internal paths and does **not** require per-step edits. If your local layout differs, update these constants in each script:

- **Step A** (`src/IQLoc/a_Cache_ES_Results.py`)
	- `DEFAULT_INPUT_FILE` → default: `Dataset/Bench4BLExtended.json`
	- `DEFAULT_OUTPUT_DIR` → default: `Output/Caching`
- **Step B** (`src/IQLoc/b_Generate_BR_Keywords.py`)
	- `DEFAULT_INPUT_FILE` → default: `Output/Caching/Cached_ES_Results.json`
	- `DEFAULT_OUTPUT_DIR` → default: `Output/Generation`
- **Step C** (`src/IQLoc/c_Generate_CE_Score.py`)
	- `DEFAULT_INPUT_FILE` → default: `Output/Generation/Cached_BR_Keywords.json`
	- `DEFAULT_OUTPUT_DIR` → default: `Output/Caching`
- **Step D** (`src/IQLoc/d_Generate_Source_Keywords.py`)
	- `DEFAULT_INPUT_FILE` → default: `Output/Caching/Cached_CE_Scores.json`
	- `DEFAULT_OUTPUT_DIR` → default: `Output/Generation`
- **Step E** (`src/IQLoc/e_Localize_Bug.py`)
	- `DEFAULT_INPUT_FILE` → default: `Output/Generation/Cached_Source_Keywords.json`
	- `DEFAULT_OUTPUT_DIR` → default: `Output/Localization`

Also verify your Elasticsearch/index configuration under `src/IR/config` and `src/IR_Reretrieval/config` if you use non-default index names or hosts.

## Create Required Folders

```powershell
New-Item -ItemType Directory -Force -Path "Dataset","Output\Caching","Output\Generation","Output\Localization"
```

## Elasticsearch Preparation

Before running A→E, make sure Elasticsearch is running and source files are indexed.

- Use the utilities in `src/IR/Indexer/` to create indices and populate documents.
- Ensure indexed documents contain fields used by retrieval (`file_url`, `source_code`).

## Run Replication (A→E)

Run from repository root:

### A) Cache retrieval + parse methods

```powershell
python src/IQLoc/a_Cache_ES_Results.py
```

Output:
- `Output/Caching/Cached_ES_Results.json`

### B) Generate bug-report keywords

```powershell
python src/IQLoc/b_Generate_BR_Keywords.py
```

Output:
- `Output/Generation/Cached_BR_Keywords.json`

### C) Cross-encoder method scoring

```powershell
python src/IQLoc/c_Generate_CE_Score.py
```

Output:
- `Output/Caching/Cached_CE_Scores.json`

### D) Generate source keywords

```powershell
python src/IQLoc/d_Generate_Source_Keywords.py
```

Output:
- `Output/Generation/Cached_Source_Keywords.json`

### E) Localize + evaluate

```powershell
python src/IQLoc/e_Localize_Bug.py
```

Outputs:
- `Output/Localization/Localized_Bugs.json`
- `Output/Localization/Localization_Performance.json`

## Data Flow Contract

- **A** populates `Results[]` with retrieval fields and `methods[]` (each method contains `method_name`, `method_body`).
- **B** adds bug-report-level `keywords`.
- **C** adds per-method `ce_score`.
- **D** adds per-method `src_kw_15` and `src_kw_sim_15` (for methods above CE threshold).
- **E** uses these fields for query expansion, re-retrieval, and evaluation metrics.

## Notes on Models

- BR/source keyword stages use `KeyBERT` + `flair` (`Salesforce/codet5-small` embedding).
- CE stage uses `CrossEncoder` via `sentence-transformers`.
- On first run, models may download from Hugging Face if not cached locally.

## Quick Validation

After running all stages, verify expected files exist:

```powershell
Get-ChildItem Output\Caching
Get-ChildItem Output\Generation
Get-ChildItem Output\Localization
```

## Troubleshooting

- **Elasticsearch connection fails**: confirm ES is running and index settings match search scripts.
- **No results in stage A**: check index content and project/sub-project/version fields.
- **Model loading errors**: ensure internet access or pre-downloaded model cache.
- **Empty evaluation in stage E**: verify `fixed_files` exists in dataset and prior outputs are populated.

## Citation

If this package supports your research, please cite:

- **Paper**: [*Improving IR-based bug localization with semantics-driven query reduction*](https://www.sciencedirect.com/science/article/pii/S0164121226000932)

```bibtex
@article{samir_IQLoc,
title = {Improving IR-based bug localization with semantics-driven query reduction},
journal = {Journal of Systems and Software},
volume = {237},
pages = {112860},
year = {2026},
issn = {0164-1212},
doi = {https://doi.org/10.1016/j.jss.2026.112860},
url = {https://www.sciencedirect.com/science/article/pii/S0164121226000932},
author = {Asif {Mohammed Samir} and Mohammad {Masudur Rahman}},
keywords = {Bug localization, Hybrid, Information retrieval, LLM, Program semantics}
}
```

