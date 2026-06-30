<div align="center">

# 🧠 EEG to Text
## Imagined-Speech EEG Decoder

Imagined speech decoding with **Riemannian geometry** + a **multi-scale Transformer**

```
╔═══════════════════════════════════════════════════════════════╗
║  EEG signal classification into 8 meta-classes                ║
║  Riemannian SPD tokens + MultiScale Attention Pooling         ║
╚═══════════════════════════════════════════════════════════════╝
```

---

<a href="#-quick-start"><img src="https://img.shields.io/badge/Quickstart-ready-00b894?style=for-the-badge" alt="Quickstart"></a>
<a href="#-requirements"><img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge" alt="Python"></a>
<a href="#-requirements"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?style=for-the-badge" alt="PyTorch"></a>
<a href="#-data"><img src="https://img.shields.io/badge/Data-Chisco-informational?style=for-the-badge" alt="Data"></a>
<a href="#-benchmarks"><img src="https://img.shields.io/badge/Status-Benchmark-6f42c1?style=for-the-badge" alt="Status"></a>
<a href="#-license"><img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" alt="License"></a>

<sub>📧 **Author:** Eduard Igorevich Verbetskii · Moscow Aviation Institute (National Research University) · [verbasik@gmail.com](mailto:verbasik@gmail.com)</sub>

</div>

---

## 📑 Navigation

| 🚀 Quick Start                       | 📚 Full Documentation              | 🔧 Configuration                 |
|:------------------------------------ |:---------------------------------- |:-------------------------------- |
| [Installation](#-quick-start)        | [Usage](#-usage)                   | [Parameters](#-configuration)    |
| [First Run](#-first-run-test-dryrun) | [Architecture](#-architecture)     | [Structure](#-project-structure) |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎯 Accuracy & Robustness
- ✅ **SI F1-macro: 0.253 ± 0.002**
- ✅ **SD F1-macro: 0.266 ± 0.010**
- ✅ **SD Balanced Accuracy: 0.285 ± 0.013**
- ✅ **2.3x above random chance (~0.125)**

</td>
<td width="50%">

### 🧮 Technology Stack
- 🔷 **Riemannian geometry** (SPD matrices)
- 🔶 **Multi-scale tokens** (128/96 + 256/128 samples)
- 🟦 **Transformer architecture** (2 layers, 4 heads)
- 🟩 **Hybrid normalization** + subject embeddings

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 📋 Requirements

```
Python 3.12+
PyTorch 2.2+
CUDA 11.8+ (optional, for acceleration)
```

### 💻 Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **💡 Tip:** For GPU acceleration, install `torch` with CUDA support:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### 📁 Data

```
Preprocessed EEG signals:
├── $EEG_PREPROCESSED_DIR
├── /mnt/data/data/derivatives/preprocessed_pkl/
├── /mnt/data/derivatives/preprocessed_pkl/
├── /mnt/data/EEG/preprocessed_pkl/
├── ./derivatives/preprocessed_pkl/
│   └── <subject>/eeg/*task-imagine*.pkl
│
Dictionaries and mappings:
├── json/
│   ├── classnumber.json      (39 source classes)
│   ├── textmaps.json         (text descriptions)
│   └── metaclasses.json      (39→8 mapping)
```

### ⚡ First Run (Test DryRun)

```bash
# 1-epoch check (fast!)
python3 Pipeline/test_dryrun.py
```

✅ If it completes successfully, the environment is ready.

### 🔢 Actual Input Shape

After excluding one channel, the current loader returns batches with this shape:

```text
batch: [B, 124, 1651]
sample: [124, 1651]
```

At 500 Hz, `T=1651` corresponds to roughly 3.3 seconds of imagined speech.

---

## 📖 Usage

### 🎓 Training the Deep Learning Model

```bash
python3 Pipeline/train.py
```

**Output files:**
```
Train/
├── checkpoints/<exp>/
│   └── best_model.pt                    # Best checkpoint
└── results/<exp>/
    ├── metrics.json                     # Final metrics
    ├── history.json                     # Training history
    ├── config_run.json                  # Run configuration
    └── val_preds.npz                    # Validation predictions
```

### 📊 Full SI/SD Evaluation

```bash
# SI pooled personalized + SD per-subject evaluation
python3 Pipeline/run_full_evaluation.py --pipeline both
```

Produces:
- Metrics for each pipeline, fold, and subject
- Bootstrap 95% confidence intervals
- Tables and plots in `Train/results/full_evaluation/`

### 📈 Saving Attention Maps

```bash
# Visualize Transformer attention
python3 Pipeline/train.py --save-attn
```

---

## ⚙️ Configuration

**Main file:** `Pipeline/config.py` → `default_config()`

<details>
<summary><b>📂 Data Parameters</b></summary>

| Parameter              | Value                                                                          | Description                            |
|:-----------------------|:-------------------------------------------------------------------------------|:---------------------------------------|
| `data_dir`             | `$EEG_PREPROCESSED_DIR` or `/mnt/data/data/derivatives/preprocessed_pkl`       | Path to preprocessed data              |
| `subject_ids`          | `["sub-01", ..., "sub-05"]`                                                    | Subject IDs used for training          |
| `task`                 | `'imagine'`                                                                    | Task type (imagined speech)            |
| `normalize`            | `'zscore_hybrid'`                                                              | Subject centering + global std         |
| `exclude_channels`     | `[124]`                                                                        | Exclude artifact channels              |

</details>

<details>
<summary><b>🧠 Model Parameters</b></summary>

| Parameter              | Value                | Description                                                         |
|:-----------------------|:---------------------|:--------------------------------------------------------------------|
| n_classes              | 8                    | Number of meta-classes                                              |
| proj_channels          | 24                   | Number of output channels after projection                          |
| **Small window**       | 128 / 96             | Window size / stride in samples                                     |
| **Large window**       | 256 / 128            | Window size / stride in samples                                     |
| spd_vec_dim            | 300                  | `24 * 25 / 2`, upper triangle of the SPD matrix                     |
| feature_proj           | `Linear(300, 128)`   | SPD-vector projection into token space                              |
| d_model                | 128                  | Embedding size                                                      |
| n_layers               | 2                    | Number of Transformer layers                                        |
| n_heads                | 4                    | Number of attention heads                                           |
| attn_heads             | 1                    | Number of attention pooling heads                                   |
| cov_type               | `'corr'`             | Covariance matrix type                                              |
| use_subject_embed      | `True`               | Use subject embeddings                                              |
| subject_embed_dim      | 16                   | Subject embedding size                                              |
| subject_embed_dropout  | 0.2                  | Dropout for subject embedding                                       |
| SI classifier input    | 272                  | `CLS(128) + attention pooled(128) + subject embedding(16)`          |
| SD classifier input    | 256                  | `CLS(128) + attention pooled(128)`                                  |

</details>

<details>
<summary><b>🧪 Evaluation Parameters</b></summary>

| Parameter                          | Value                | Description                                                             |
|:-----------------------------------|:---------------------|:------------------------------------------------------------------------|
| `evaluation.pipeline`              | `'both'`             | Run SI, then SD                                                         |
| `evaluation.si_use_subject_embed`  | `True`               | SI uses subject embeddings                                              |
| `evaluation.sd_use_subject_embed`  | `False`              | SD trains a separate model per subject                                  |
| `cv.protocol`                      | `'within_subject'`   | Each subject appears in both train and validation                       |
| `cv.mode`                          | `'within_subject'`   | Folds are built within each subject                                     |
| `model.unknown_subject_policy`     | `'auto'`             | `error` for within-subject, `zero` for subject-held-out                 |

The current SI baseline evaluates known-subject generalization. It is not
a strict test of transfer to a completely new subject; that requires a
separate `subject_heldout`/LOSO run.

</details>

<details>
<summary><b>🎯 Training Parameters</b></summary>

| Parameter                    | GPU  | CPU  | Description                                       |
|:-----------------------------|:-----|:-----|:--------------------------------------------------|
| `batch_size`                 | 16   | 8    | Batch size                                        |
| `lr`                         | 3e-4 | 3e-4 | Learning rate (AdamW)                             |
| `weight_decay`               | 1e-4 | 1e-4 | L2 regularization                                 |
| `early_stopping_patience`    | 8    | 8    | Epochs without improvement                        |
| `use_amp`                    | ✅   | ❌   | Automatic Mixed Precision                         |
| `grad_clip`                  | 1.0  | 1.0  | Gradient clipping                                 |
| `num_workers`                | 0    | 0    | DataLoader multiprocessing is disabled by default |

</details>

<details>
<summary><b>🔥 Loss & Optimizer Parameters</b></summary>

| Parameter     | Value                      | Description                               |
|---------------|----------------------------|-------------------------------------------|
| **Loss**      | CB-Focal (β=0.999, γ=1.75) | Class-Balanced Focal Loss                 |
| **Optimizer** | AdamW                      | With separate weight decay for embeddings |
| **Scheduler** | Cosine + Warmup            | Linear warmup → cosine decay              |

</details>

---

## 🏗️ Architecture

![Architecture](/EEG_TO_TEXT/assets/main.png)

### 🔑 Key Components

| Component          | Function              | Feature                           |
|--------------------|-----------------------|-----------------------------------|
| **SPD Tokens**     | Signal representation | Riemannian geometry, robustness   |
| **Multi-Scale**    | Multi-level analysis  | 128/96 + 256/128 samples          |
| **Transformer**    | Context learning      | 2 layers × 4 heads                |
| **Attention Pool** | Token aggregation     | Adaptive weights                  |
| **Subject Embed**  | Personalization       | Separate weight decay             |

---

## 📊 Benchmarks

### Current Full Evaluation Results

```
┌────────────────────────────────────────────────────────┐
│         Full run: SI + SD pipelines                    │
│                (30 successful runs)                    │
├────────────────────────────────────────────────────────┤
│ Pipeline / metric   │ Value         │ Confidence (95%) │
├─────────────────────┼───────────────┼──────────────────┤
│ SI F1-macro         │ 0.253 ± 0.002 │ [0.251; 0.255]   │
│ SI Accuracy         │ 0.283 ± 0.003 │ [0.281; 0.285]   │
│ SI Balanced Accuracy│ 0.266 ± 0.003 │ [0.264; 0.269]   │
│ SD F1-macro         │ 0.266 ± 0.010 │ [0.262; 0.271]   │
│ SD Accuracy         │ 0.285 ± 0.013 │ [0.280; 0.290]   │
│ SD Balanced Accuracy│ 0.285 ± 0.013 │ [0.281; 0.290]   │
└─────────────────────┴───────────────┴──────────────────┘
```

> Do not use the mixed overall mean as the main headline metric: the overall
> report mixes SI and SD experiments.

### Comparison with Random Chance

| Model                | Accuracy    | Note                       |
|----------------------|-------------|----------------------------|
| 🎲 **Random Chance** | **0.125**   | 8 classes (1/8)            |
| 🎯 **SI baseline**   | **0.283**   | ↑ 2.26x above random       |
| 🎯 **SD baseline**   | **0.285**   | ↑ 2.28x above random       |

### Per-Subject SD F1-Macro

| Subject | F1-macro      | 95% CI         |
|---------|---------------|----------------|
| sub-01  | 0.257 ± 0.009 | [0.249; 0.264] |
| sub-02  | 0.270 ± 0.011 | [0.261; 0.280] |
| sub-03  | 0.270 ± 0.012 | [0.260; 0.280] |
| sub-04  | 0.268 ± 0.006 | [0.262; 0.273] |
| sub-05  | 0.267 ± 0.005 | [0.262; 0.271] |

---

## 📂 Project Structure

```
EEG_to_Text/
│
├── 📜 README.md                          # This file
├── 📄 LICENSE                            # MIT License
├── 📋 requirements.txt                   # Dependencies
│
├── Pipeline/                             # ⭐ MAIN CODE
│   ├── config.py                         # Configuration
│   ├── data_loader.py                    # Data loader
│   ├── model.py                          # RTTMultiScale model
│   ├── riemannian_utils.py               # SPD operations
│   ├── trainer.py                        # Training logic
│   ├── train.py                          # Main training script
│   ├── run_full_evaluation.py            # SI/SD full evaluation
│   ├── feature_engineering.py            # Feature engineering
│   └── test_dryrun.py                    # Fast check
│
├── json/                                 # 📖 DICTIONARIES
│   ├── classnumber.json                  # 39 classes
│   ├── textmaps.json                     # Descriptions
│   └── metaclasses.json                  # 39→8 mapping
│
├── Train/                                # 📊 RESULTS
│   ├── checkpoints/
│   │   └── <exp_id>/
│   │       └── best_model.pt
│   └── results/
│       └── <exp_id>/
│           ├── metrics.json
│           ├── history.json
│           ├── config_run.json
│           └── val_preds.npz
│
└── derivatives/                          # 💾 LOCAL DATA
    └── preprocessed_pkl/                 # (optional)
        └── <subject>/eeg/
```

---

## 🤝 Contributing

Contributions are welcome.

### How to help:
1. **Bug fixes:** Open an issue with a description
2. **Improvements:** Fork → Branch → Commit → PR
3. **Documentation:** Clarifications and examples are always welcome
4. **Ablations:** Results from new configurations are useful

### PR requirements:
- ✅ Follow the project style (PEP 8)
- ✅ Add minimal tests
- ✅ Update documentation
- ✅ Describe the changes in the PR description

---

## 📜 License

This project is distributed under the **MIT** license. See [`LICENSE`](./LICENSE) for the full terms.

---

## 📚 Citation

If you use this project in research, please cite:

```bibtex
@thesis{verbetskii2026rttmultiscale,
title       = {Riemannian geometric features and transformer for decoding imagined speech from EEG},
author      = {Verbetskii, Eduard Igorevich},
institution = {Moscow Aviation Institute (National Research University)},
location    = {Moscow, Russia},
year        = {2026},
type        = {Master of Science},
note        = {Institute No. 8 `Computer Science and Applied Mathematics''; educational program `Machine Learning and Data Analysis''},
langid      = {russian}
}
```

---

## 📞 Contacts

- 👨‍💼 **Author:** Eduard Verbetskii
- 📧 **Email:** [verbasik@gmail.com](mailto:verbasik@gmail.com)
- 🏛️ **Institution:** Moscow Aviation Institute (National Research University)
- 📍 **Project:** Imagined Speech Decoding

---

<div align="center">

### ⭐ If you like this project, give it a star on GitHub.

**Made in Russia with love ❤️**

*Last updated: June 2026*

</div>
