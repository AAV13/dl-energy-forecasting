# Transformer-Based Forecasting of Renewable Energy & Electricity Prices

**University of Maryland, College Park | DATA612 Deep Learning | Spring 2025** **Authors:** Ayush Vispute, Mukul Gupta, Nishant Lalge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Forecasting renewable energy generation and day-ahead electricity prices is a critical challenge for modern power grids due to extreme weather-driven volatility and the merit-order effect. 

In this project, we evaluate standard recurrent architectures (BiLSTM) against modern Transformers. Recognizing the limitations of standard attention mechanisms (e.g., Attention Collapse) and the difficulty of beating seasonal persistence, we propose a novel architecture: **GRQ-PatchTST (Gated Residual Quantile PatchTST)**. 

Our custom architecture reduces Solar Generation Mean Absolute Error (MAE) by 47% compared to base Transformers and successfully outputs 10th and 90th percentile risk bounds to quantify the downside risk of day-ahead market price crashes.

---

## 🚀 Novel Architecture: GRQ-PatchTST
We adapted the standard PatchTST architecture specifically for the physical constraints of energy markets, introducing three novel components:

1. **Explicit Persistence Routing:** The model automatically extracts a $T-7$ day baseline, forcing the Transformer encoder to learn the *residual* (delta) rather than forcing it to memorize weekly planetary cycles.
2. **Gated Temperature Cross-Fusion:** To solve the 0.33/0.33/0.33 attention collapse observed in standard multi-head attention, we apply a temperature scalar to the query matrix and route the output through a Gated Linear Unit (GLU).
3. **Quantile Output Heads & Huber Loss:** Replaced standard linear outputs with quantile heads (Q10, Q50, Q90) trained via a custom **Quantile Huber Loss (Pinball Huber)** to handle extreme price crashes without exploding gradients.

---

## 📊 Dataset
* **Source:** Open Power System Data (OPSD)
* **Region:** Germany (DE)
* **Timeframe:** 2015 – 2020 (Hourly resolution)
* **Features:** 15 input features including load, solar generation, and wind generation.
* **Target Variables:** Solar Generation (MW), Wind Onshore (MW), Day-Ahead Price (€/MWh).
* *Note: The DE_AT_LU and DE_LU bidding zones were merged to ensure a continuous price feature following the October 2018 market split.*

---

## 📂 Repository Structure

```text
dl-energy-forecasting/
│
├── data/
│   ├── time_series_60min_singleindex.csv   # Raw OPSD data (Not included due to size)
│   └── processed.pkl                       # Scaled and merged data ready for training
│
├── models/
│   ├── grq_patchtst.py                     # Core architecture definition
│   └── loss.py                             # Custom Quantile Huber Loss function
│
├── dl_energy_project_v3.ipynb              # Baseline Training (BiLSTM, Base PatchTST)
├── grq_training (1).ipynb                  # GRQ-PatchTST Training & Visualizations
│
├── DL_FinalProjectReport.pdf               # Final Academic Paper
├── DL_Final_Clean.pptx                     # Presentation Slide Deck
└── README.md
