# Transformer-Based Forecasting of Renewable Energy & Electricity Prices

**Authors:** Ayush Vispute · Mukul Gupta · Nishant Lalge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## 📌 Project Overview

Forecasting renewable energy generation and day-ahead electricity prices is a critical challenge for modern power grids due to extreme weather-driven volatility and the **merit-order effect**.

In this project, we evaluate standard recurrent architectures (BiLSTM) against modern Transformers. Recognizing the limitations of standard attention mechanisms (e.g., *Attention Collapse*) and the difficulty of beating seasonal persistence, we propose a novel architecture:

> **GRQ-PatchTST** *(Gated Residual Quantile PatchTST)*

Our custom architecture:
- Reduces Solar Generation MAE by **47%** compared to base Transformers
- Outputs **10th and 90th percentile risk bounds** to quantify downside risk of day-ahead market price crashes

---

## 🚀 Novel Architecture: GRQ-PatchTST

We adapted the standard PatchTST architecture specifically for the physical constraints of energy markets, introducing **three novel components**:

| Component | Description |
|---|---|
| **Explicit Persistence Routing** | Automatically extracts a T−7 day baseline, forcing the Transformer encoder to learn the *residual* (delta) rather than memorize weekly planetary cycles |
| **Gated Temperature Cross-Fusion** | Applies a temperature scalar to the query matrix and routes output through a Gated Linear Unit (GLU), solving the 0.33/0.33/0.33 attention collapse seen in standard multi-head attention |
| **Quantile Output Heads & Huber Loss** | Replaces standard linear outputs with quantile heads (Q10, Q50, Q90) trained via a custom **Quantile Huber Loss (Pinball Huber)** to handle extreme price crashes without exploding gradients |

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Open Power System Data (OPSD)](https://open-power-system-data.org/) |
| **Region** | Germany (DE) |
| **Timeframe** | 2015 – 2020 (Hourly resolution) |
| **Input Features** | 15 features including load, solar generation, and wind generation |
| **Target Variables** | Solar Generation (MW), Wind Onshore (MW), Day-Ahead Price (€/MWh) |

> **Note:** The DE_AT_LU and DE_LU bidding zones were merged to ensure a continuous price feature following the October 2018 market split.

---

## 🏆 Key Results (Test Set 2019–2020)

| Model | Solar MAE (MW) | Wind MAE (MW) | Price MAE (€/MWh) |
|---|:---:|:---:|:---:|
| Seasonal Persistence | 1,860 | 8,350 | 10.0 |
| BiLSTM | 2,450 | 7,900 | 12.4 |
| PatchTST+ (Base) | 3,745 | 5,510 | 8.2 |
| **GRQ-PatchTST (Ours)** | **1,610** | **5,250** | **8.7** |

> **Note:** The slight regression in median Price MAE is an accepted trade-off to unlock Q10/Q90 uncertainty quantification bounds for severe market crash scenarios.

<img width="1668" height="1299" alt="image" src="https://github.com/user-attachments/assets/68eb1342-85ef-470d-b8fa-9dca8d7fa5d2" />


---

## 📂 Repository Structure

```
dl-energy-forecasting/
│
├── data/
│   ├── time_series_60min_singleindex.csv   # Raw OPSD dataset (not included — too large)
│   └── processed.pkl                       # Scaled/merged data ready for PyTorch loaders
│
├── models/
│   ├── grq_patchtst.py                     # Core GRQ-PatchTST architecture definition
│   └── loss.py                             # Custom Quantile Huber Loss function
│
├── baseline_models.ipynb                   # Step 1: Data prep & Baseline Training (BiLSTM, Base PatchTST)
├── grq_training.ipynb                      # Step 2: GRQ-PatchTST isolated training loop
├── final_demo.ipynb                        # Step 3: Complete narrative, evaluation, and visualizations
│
├── DL_FinalProjectReport.docx              # Final Academic Paper
├── DL_Final_Clean.pptx                     # Presentation Slide Deck
└── README.md
```

---

## ⚙️ How to Run

We recommend running the notebooks via **Google Colab (T4 GPU)**.

### Prerequisites

1. Clone this repository to your Google Drive.
2. Ensure the `data/processed.pkl` file is in the correct directory.

### Notebook Execution Order

| Step | Notebook | Purpose |
|:---:|---|---|
| 1 | `baseline_models.ipynb` | Data prep & baseline training (BiLSTM, Base PatchTST) |
| 2 | `grq_training.ipynb` | Train the GRQ-PatchTST novel architecture |
| 3 | `final_demo.ipynb` | Full narrative, evaluation, attention heatmaps, uncertainty ribbons |

> **Tip:** For the full narrative and evaluation only, start directly at **Step 3** (`final_demo.ipynb`). It loads saved model weights and generates all final physical-unit evaluation tables and visualizations.

---
