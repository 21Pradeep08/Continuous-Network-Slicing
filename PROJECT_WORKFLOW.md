# AI-Driven Continuous Network Slicing and Resource Allocation for Smart Healthcare Systems over 5G (LSTM + DDPG)

This document describes the complete workflow for implementing this project **from scratch**. It covers the problem, the dataset (`dataset.csv`), every equation used, and the exact implementation plan — using Jupyter notebooks (`.ipynb`) for all training and testing/evaluation work.

---

## 1. Problem Statement

Patients wear IoT/WBAN sensors (Heart Rate, Blood Pressure, SpO₂) that continuously stream health data over a 5G network. Not all patients are equally urgent — a stable patient can tolerate delay, but a patient in cardiac distress or in ICU needs near-instant data transmission. Static network slicing (fixed bandwidth per slice, unchanging over time) cannot adapt as real-world patient population and urgency shift hour to hour (e.g., 20 patients at 8 AM vs. 150 at 2 PM, sudden ambulance surges).

**Goal:** Build a system that continuously predicts network traffic and dynamically reallocates a shared, limited pool of 5G bandwidth across many patients in real time — protecting the most critical patients even under contention — using:
- **LSTM** to predict near-future per-patient traffic demand from historical traffic.
- **DDPG** (Deep Deterministic Policy Gradient) to continuously decide bandwidth allocation (a continuous action space) across all contending patients, trained against a reward built from QoS + energy-efficiency metrics.

---

## 2. Overall Architecture

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

---

## 3. Dataset — `dataset.csv`

**Rows:** one row per patient, per hourly time step.
**Important design decision:** `dataset.csv` includes a **baseline** allocation and its resulting QoS metrics (Bandwidth, Latency, Throughput, Reliability, Power, REU, Utility), computed with a fixed, static proportional-priority formula. This baseline exists for two reasons:
1. So the dataset is self-contained, inspectable, and can be validated/visualized on its own.
2. So it can serve as a **comparison baseline** when evaluating the trained DDPG agent ("did the learned policy actually beat this static formula?").

**Critically: during actual DDPG training, do NOT read `Bandwidth`/`Latency`/etc. from this file as fixed ground truth.** The DDPG agent chooses its own bandwidth action, which will differ from the baseline — so `Latency`, `Throughput`, `Reliability`, `Power`, `REU`, `Utility` must be **recomputed live** by an environment function every training step, using whatever bandwidth the agent actually chose. The equations for doing this are identical to the ones used to generate the baseline (Section 4) — just re-run against a different action.

### 3.1 Columns

| Column | Meaning | Notes |
|---|---|---|
| `Time` | Hour index (0–99 or however many time steps generated) | |
| `PatientID` | Patient identifier | |
| `SliceID` | Network slice, 1–5 | See slice→latency mapping below |
| `HeartRate` | bpm | ~N(70,10), clipped [50,120] |
| `BP` | Systolic blood pressure, mmHg | ~N(110,15), clipped [90,150] |
| `SpO2` | Oxygen saturation, % | ~N(98,2), clipped [85,100] |
| `Emergency` | Binary flag | 1 if `HeartRate>100` OR `BP>140` OR `SpO2<92`. **`SliceID==2` (ICU) is always forced to `Emergency=1`, regardless of vitals** — ICU patients are always treated as a critical/emergency condition by definition. |
| `LatencyReq` | Slice's required max latency (ms) | Fixed per `SliceID` — see mapping below |
| `Priority` | Derived priority score, [0,1] | See Eq. 4.1 |
| `Traffic` | Current traffic demand | AR(2) + seasonal + noise |
| `Traffic_t-3, Traffic_t-2, Traffic_t-1` | Lagged traffic (previous 3 hours) | LSTM input sequence |
| `DataSize` | Payload size to transmit this hour | Larger when `Emergency=1` |
| `SINR` | Signal-to-Interference-plus-Noise Ratio (dB) | ~N(15,5), clipped [0,30] |
| `Queue` | Queue length (packets waiting) | Poisson(`Lambda`/`Mu`) |
| `Lambda` | Packet arrival rate | Correlated with `Traffic`/`DataSize` |
| `Mu` | Packet service rate | Always `> Lambda` (enforced) |
| `Bandwidth` | **Baseline** allocated bandwidth (see Eq. 4.2) | NOT the DDPG action — a static reference value |
| `Latency` | **Baseline** resulting latency (Eq. 4.3) | Recompute live for DDPG training |
| `Throughput` | **Baseline** resulting throughput (Eq. 4.4) | Recompute live for DDPG training |
| `Reliability` | **Baseline** resulting reliability (Eq. 4.5) | Recompute live for DDPG training |
| `Power` | **Baseline** resulting power draw (Eq. 4.6) | Recompute live for DDPG training |
| `REU` | **Baseline** resource energy utilization (Eq. 4.7) | Recompute live for DDPG training |
| `Utility` | **Baseline** resulting utility/reward (Eq. 4.8) | Recompute live for DDPG training |

### 3.2 SliceID → Latency Requirement Mapping

| SliceID | Interpretation | `LatencyReq` (ms) |
|---|---|---|
| 1 | Emergency / URLLC | 5 |
| 2 | ICU (always Emergency=1) | 15 |
| 3 | ECG / continuous monitoring | 30 |
| 4 | Telemedicine | 60 |
| 5 | Medical Video / routine | 100 |

---

## 4. Mathematical Model

These are the exact equations used both to generate the baseline columns in `dataset.csv`, AND to be re-implemented as a live `step(state, action)` environment function for DDPG training.

### 4.1 Priority Score

```
Priority_i(t) = clip( w1·Emergency_i + w2·(LatencyReq_min / LatencyReq_i) + w3·min(Traffic_i(t)/Traffic_max, 1),  0, 1 )
```
- `w1=0.5, w2=0.3, w3=0.2`
- `LatencyReq_min = 5` (the tightest slice requirement, i.e. slice 1)
- `Traffic_max = 150` (normalization constant)

### 4.2 Bandwidth Allocation — weighted by Priority **and** DataSize

This is the corrected version (see Section 7 for why the naive version fails):

```
weight_i(t) = Priority_i(t) × DataSize_i(t)

Bandwidth_i(t) = ( weight_i(t) / Σ_j weight_j(t) ) × TOTAL_BW
```
subject to: `Σ_i Bandwidth_i(t) = TOTAL_BW` for every time step `t` (hard constraint — bandwidth must be normalized **jointly across every patient active at that same hour**, never computed for one patient in isolation).

- `TOTAL_BW = 30000` (tuned so the average per-patient share lands in a sensible few-unit range given ~10,000 concurrent patients — rescale proportionally if patient count changes)
- **Why multiply by `DataSize`:** if bandwidth were shared by `Priority` alone, a patient generating a much larger payload (as Emergency patients do) would still suffer higher transmission delay despite receiving more bandwidth, because their data volume grew faster than their bandwidth share. Weighting by `Priority × DataSize` keeps `DataSize/Bandwidth` (the transmission-delay ratio) stable regardless of payload size, so the `Priority` term is what actually determines who ends up faster — not accidentally cancelled out by their own data volume.

### 4.3 Latency

```
Latency_i(t) = DataSize_i(t) / ( Bandwidth_i(t) · log2(1 + SINR_i(t)) )   +   1 / (Mu_i(t) − Lambda_i(t))
```
First term = Shannon-capacity transmission delay. Second term = M/M/1 queueing delay. Constraint: `Lambda_i(t) < Mu_i(t)` always (enforced at generation time).

### 4.4 Throughput

```
Throughput_i(t) = Bandwidth_i(t) · log2(1 + SINR_i(t))
```

### 4.5 Reliability

```
Reliability_i(t) = 1                                                    if Latency_i(t) ≤ LatencyReq_i
Reliability_i(t) = exp( -(Latency_i(t) − LatencyReq_i) / LatencyReq_i )  otherwise
```
(Smooth, differentiable approximation of `Pr(Latency ≤ LatencyReq)` — chosen over a strict binary indicator so it's usable as a continuous training signal for DDPG.)

### 4.6 Power (4-component model)

```
Power_i(t) = P_device + P_BS + P_edge + P_AI

P_device ~ Uniform(0.5, 2.0)                     # wearable/IoT sensor transmit power
P_BS      = 5.0 + 1.0 · Bandwidth_i(t)           # base-station power scales with bandwidth
P_edge    = 10.0 + 0.05 · DataSize_i(t)          # edge-server processing power scales with payload
P_AI     ~ Uniform(50.0, 150.0)                  # shared LSTM/DDPG inference power
```

### 4.7 Resource Energy Utilization

```
REU_i(t) = Throughput_i(t) / Power_i(t)
```

### 4.8 Utility (Reward Function)

```
Utility_i(t) = w1·Throughput_i(t) + w2·Reliability_i(t) − w3·Latency_i(t) − w4·Power_i(t) + w5·REU_i(t)
```
- `w1=0.3, w2=0.3, w3=0.2, w4=0.1, w5=0.1` (sum to 1)
- This is the **reward** the DDPG agent is trained to maximize.

---

## 5. Train/Test Split Strategy

Use **both** split types, for different reasons:
1. **Time-based split**: reserve the last ~20% of time steps per patient as test, train on the earlier ~80%. Respects temporal order — the LSTM should only ever be evaluated on genuinely future traffic it hasn't seen.
2. **Patient-based split**: hold out a subset of `PatientID`s entirely from training, evaluate on them separately. Tests whether the learned policy generalizes to new patients rather than memorizing specific individuals.

Do not shuffle rows randomly across time when forming LSTM sequences — always slice by contiguous time windows per patient.

---

## 6. Implementation Plan

### 6.1 File Structure

```
project/
├── dataset.csv                     # generated dataset (Section 3)
├── PROJECT_WORKFLOW.md              # this document
├── config.py                        # constants: weights, LatencyReq map, TOTAL_BW, power coefficients
├── environment.py                   # live step(state, action) -> (next_state, reward, info), implementing Section 4 equations
├── 01_data_exploration.ipynb        # sanity checks: ACF/PACF of Traffic, distribution plots, SliceID/Emergency cross-checks
├── 02_train_lstm.ipynb              # LSTM traffic-prediction training + evaluation (MAE/RMSE/MAPE)
├── 03_train_ddpg.ipynb              # DDPG training loop using environment.py
└── 04_evaluate.ipynb                # Compare trained DDPG vs. dataset.csv's static baseline + a simple priority-heuristic baseline
```

**Note:** per instruction, all training and testing/evaluation work should live in Jupyter notebooks (`.ipynb`), not standalone `.py` scripts — this makes it easy to inspect intermediate outputs, plots, and metrics step by step. Only the reusable, non-interactive pieces (`config.py`, `environment.py`) should remain plain Python modules, since the notebooks need to `import` them.

### 6.2 `01_data_exploration.ipynb`

- Load `dataset.csv`.
- Plot `Traffic` time series and ACF/PACF for a handful of sample patients — confirm genuine autocorrelation (lag-1 to lag-3 decaying, NOT flat/random).
- Confirm `Σ Bandwidth` per `Time` step equals `TOTAL_BW` exactly (sanity check the shared-pool constraint holds).
- Confirm `SliceID==2` rows all have `Emergency==1`.
- Confirm, within each slice, `Emergency==1` patients have lower `Latency`/higher `Reliability` than `Emergency==0` patients (validates Eq. 4.2's DataSize-weighting fix).
- Histograms of `HeartRate`, `BP`, `SpO2`, `Traffic`, `Latency` per slice.

### 6.3 `02_train_lstm.ipynb`

- **Input:** sliding window of `[Traffic_t-3, Traffic_t-2, Traffic_t-1, Traffic]` per `PatientID`, ordered by `Time`.
- **Output:** predicted `Traffic(t+1)`.
- Split by time (Section 5) — train on early time steps, validate/test on later ones.
- Train with MSE loss.
- Evaluate with MAE, RMSE, MAPE; plot predicted vs. actual traffic for sample patients.
- Save the trained model (e.g. `lstm_traffic_predictor.pth`) at the end of the notebook for use in `03_train_ddpg.ipynb`.

### 6.4 `environment.py` — the live reward engine

Implements a `step(state, action)` function:
1. Receive `action = Bandwidth_i(t)` for every active patient at that time step (a vector, jointly normalized to respect `TOTAL_BW` — e.g. via softmax over the actor's raw outputs, then scaled).
2. Compute `Latency` (Eq. 4.3) → `Throughput` (Eq. 4.4) → `Reliability` (Eq. 4.5) → `Power` (Eq. 4.6) → `REU` (Eq. 4.7) → `Utility` (Eq. 4.8), all using the equations in Section 4 — same formulas as the baseline, but with the RL agent's own chosen bandwidth instead of the static Priority×DataSize baseline.
3. Return `reward = Utility`, plus `next_state` (advance to the next hour's row from `dataset.csv`, updating lagged fields).
4. Also add an explicit **SLA-violation penalty** on top of the base `Utility`, scaled by slice criticality, so violations on the tightest-SLA slices are penalized much more heavily than the smooth `Utility` formula alone provides:
   ```
   if Latency_i(t) > LatencyReq_i:
       penalty = { SliceID 1: 350, SliceID 2 (ICU): 300, SliceID 3: 20, SliceID 4: 10, SliceID 5: 5 }
       reward -= penalty
   ```

### 6.5 `03_train_ddpg.ipynb`

- **State vector:** `[Predicted_Traffic, Priority, Emergency, SINR, Queue, PrevBandwidth, PrevLatency, DataSize, Lambda, Mu]`
- **Action:** continuous bandwidth allocation, jointly across all patients active at that time step (batch-based training — sample a batch of concurrently-active patients each step, not one patient in isolation, so the agent actually experiences bandwidth contention).
- **Training loop:**
  ```
  for episode in episodes:
      for t in time_steps:
          batch_state = get_states_for_all_active_patients(t)     # from dataset.csv + LSTM prediction
          raw_actions = actor(batch_state) + exploration_noise
          actions = softmax_normalize(raw_actions) * TOTAL_BW      # enforce shared pool constraint
          next_state, reward, info = environment.step(batch_state, actions)
          replay_buffer.store(batch_state, actions, reward, next_state)
          train_actor_critic(replay_buffer.sample())
  ```
- Decay exploration noise once per **episode**, not per step (decaying it every single hourly step collapses exploration far too fast).
- Save trained weights (`ddpg_actor.pth`, `ddpg_critic.pth`) at the end.

### 6.6 `04_evaluate.ipynb`

- Run 3 policies on held-out test patients and compare:
  1. **Static/Equal Sharing** (naive baseline)
  2. **Priority×DataSize Heuristic** (the baseline already stored in `dataset.csv`)
  3. **Trained DDPG**
- Metrics: cumulative Utility, per-slice SLA violation rate (this is the most important one — the trained DDPG should show a **lower** violation rate than the static heuristic, especially for SliceID 1 and 2), average Reliability, average REU, Jain's fairness index across patients.
- Plot per-slice latency distributions (histogram, one subplot per `SliceID`) with the `LatencyReq` marked as a vertical line, for all 3 policies overlaid.

---

## 7. Open Design Choices / Assumptions Made (flag for review during implementation)

1. **`TOTAL_BW = 30000`** is tuned for ~10,000 concurrent patients giving ~3 units/patient average — rescale proportionally if patient count changes.
2. **Utility weights** (`w1..w5`) and **SLA violation penalties** are hand-picked defaults, not literature-derived — tune based on whether the deployment prioritizes ultra-low-latency vs. energy efficiency.
3. **`Priority × DataSize` bandwidth weighting** is a deliberately-designed fix to ensure critical patients reliably get faster service even accounting for their larger payloads — but it's still a *static* formula. The DDPG agent's job is to do even better than this, particularly under sudden population surges or simultaneous multi-patient emergencies where even a "protective" formula can't fully compensate. Don't be surprised (or alarmed) if the static baseline already performs reasonably well — the interesting evaluation question is whether DDPG beats it under contention, not whether the baseline is broken.
4. **`SliceID==2` (ICU) forced to `Emergency=1`** is a deliberate design choice — ICU is always treated as a critical condition regardless of momentary vitals.
