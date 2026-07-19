# AI-Driven Continuous Network Slicing for Smart Healthcare Systems over 5G (LSTM + DDPG)

## Overview
In smart healthcare environments, IoT/WBAN wearable sensors stream continuous patient vitals over 5G networks. To handle shifting patient populations and sudden clinical emergencies dynamically, this project replaces static network slicing with an AI-driven resource allocation framework. By predicting near-future per-patient traffic demand using an LSTM network, a Deep Deterministic Policy Gradient (DDPG) reinforcement learning agent dynamically allocates the shared 5G bandwidth pool to protect critical patients in real-time.

## Architecture
```
Wearable Sensors (HeartRate, BP, SpO2)
        │
        ▼
Healthcare Data (per patient, per time step) ── dataset.csv
        │
        ▼
Emergency flag + Priority computation (vitals-threshold based)
        │
        ▼
Traffic Prediction (LSTM)  ──uses historical Traffic_t-3..t-1──▶  Predicted Traffic(t+1)
        │
        ▼
DDPG Agent observes: [Predicted Traffic, Priority, Emergency, SINR, Queue,
                       PrevBandwidth, PrevLatency, DataSize, Lambda, Mu]
        │
        ▼
Action: Continuous Bandwidth Allocation B_i(t)  (jointly, across ALL patients
        active at time t, respecting a shared TOTAL_BW pool)
        │
        ▼
Environment (live step function) computes: Latency → Throughput → Reliability
        → Power → REU → Utility (= reward)
        │
        ▼
Reward fed back to DDPG → policy updated → repeat
```

## Repo Structure

| File / Directory | Description |
| :--- | :--- |
| `01_data_exploration.ipynb` | Sanity checks: ACF/PACF of Traffic, distribution plots, and SliceID/Emergency cross-checks. |
| `02_train_lstm.ipynb` | LSTM traffic-prediction training and evaluation (MAE, RMSE, MAPE). |
| `03_train_ddpg.ipynb` | DDPG reinforcement learning agent training loop using the simulation environment. |
| `04_evaluate.ipynb` | Compares the trained DDPG agent against static equal sharing and priority-heuristic baselines. |
| `config.py` | Configuration constants including weights, slice latency requirements, and power model coefficients. |
| `environment.py` | Live simulation environment implementing state transitions, QoS metrics, and reward functions. |
| `Paper/` | Contains the LaTeX draft of the research paper (`paper.tex`), references, and supporting system figures. |

## Setup & How to Run

### Dependencies
The following Python packages must be installed:
*   `numpy`
*   `pandas`
*   `torch` (PyTorch)
*   `matplotlib`
*   `scipy` (optional, for diagnostic scripts)

### Execution Order
Run the Jupyter notebooks in the following sequence:
1. `01_data_exploration.ipynb`
2. `02_train_lstm.ipynb`
3. `03_train_ddpg.ipynb`
4. `04_evaluate.ipynb`

### Dataset Note
The dataset files (`dataset.csv` and `dataset_backup.csv`) are gitignored due to GitHub size limits. The notebooks assume the presence of `dataset.csv` in the root directory. To run this project fresh, you must manually obtain the pre-generated `dataset.csv` and place it in the project root directory.

## Current Status & Known Limitations
*   **Evaluation Pipeline Mismatch:** The evaluation notebook currently contains a known architecture mismatch bug (detailed as Item 0 in `PROJECT_WORKFLOW.md`, Part B) where the evaluation actor network structure doesn't match the trained actor network structure.
*   **Performance Metrics:** Because of this mismatch, current evaluation numbers (e.g. cumulative utility, SLA violation rates) for DDPG do not reflect the optimized policy. Final evaluation results are pending the fixes outlined in `PROJECT_WORKFLOW.md`, Part B.

## Paper
The academic draft of this research is available under the `Paper/` directory:
*   LaTeX source: `Paper/paper.tex`
*   References: `Paper/bibl.bib`
*   System diagrams and figures: `Paper/images/`
