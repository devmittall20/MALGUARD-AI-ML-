# MalGuard-XAI: Explainable AI-Powered Malware Detection

MalGuard-XAI is a modular, production-ready, and explainable malware detection system that uses Machine Learning and Explainable AI (XAI) to classify Windows Portable Executable (PE) files. 

This repository contains the source code for static feature extraction, dataset generation, and model training.

## Project Structure

```
malguard_xai/
├── data/
│   ├── raw/                  # Raw PE binaries to analyze
│   │   ├── benign/           # Put benign executables here
│   │   └── malicious/        # Put malicious executables here
│   └── processed/            # Output feature files (CSV and Parquet)
├── models/                   # Serialized ML models (scalers, model binaries)
├── reports/                  # Generated performance metrics and charts
├── src/
│   ├── extraction/
│   │   ├── extractor.py      # Feature extraction (pefile, capstone, YARA)
│   │   └── rules.yar         # Custom YARA signature rules
│   ├── data/
│   │   ├── dataset.py        # DatasetManager (scanning, synthetic generation, splits)
│   │   └── pipeline.py       # CLI script for Stage 1 data pipeline
│   └── utils/
│       └── helpers.py        # Logging & parsing helpers
└── tests/                    # Unit tests
```

## Features Extracted

The `PEFeatureExtractor` processes PE files and generates over 80 features:
1. **PE Headers:** Machine, section counts, compiler timestamps, opt header characteristics, entry points, alignment, linker versions, subsystem, stacks, and sizes.
2. **Sections Details & Entropy:** Entropy, sizes, characteristics (executable/writable flag checks), and statistical entropy properties (min/max/mean/variance) over all sections.
3. **Imports & Exports:** DLL counts, function counts, and flags for 16 specific high-risk Windows APIs (e.g., `VirtualAlloc`, `WriteProcessMemory`, `CreateRemoteThread`).
4. **Opcodes (Disassembly):** Instruction counts, frequencies of 14 common mnemonics, and occurrences of 12 distinct 2-gram and 7 distinct 3-gram opcode patterns disassembled via **Capstone**.
5. **YARA Signatures:** Detections for suspicious APIs, common packing (e.g., UPX), anti-debugging patterns, and suspicious network indicators (URLs, IPs, User-Agents).

## Getting Started

### 1. Installation
Ensure Python 3.13+ is installed, then create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Data Preparation Pipeline
To generate a synthetic dataset of 5,000 samples and partition it (70/15/15 split):
```bash
python -m src.data.pipeline --action run-all --synthetic-samples 5000
```
This will save the balanced splits to `data/processed/` as both `.csv` and `.parquet` files:
- `train.csv` / `train.parquet`
- `val.csv` / `val.parquet`
- `test.csv` / `test.parquet`

### 3. Testing Feature Extraction on Local Files
Put some real benign executable files (e.g. `cmd.exe` or `notepad.exe` copied from `C:\Windows\System32`) into `data/raw/benign/` and run:
```bash
python -m src.data.pipeline --action extract
```
This extracts features from your local files and generates matching structured tables.

## Running Tests
Run all unit tests using pytest:
```bash
pytest tests/
```
