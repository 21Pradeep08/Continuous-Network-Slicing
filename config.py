"""
Configuration Module for 5G Network Slicing and Dynamic Resource Allocation.

This module serves as the centralized parameter repository for the entire research pipeline,
spanning data generation/exploration (01), LSTM traffic prediction (02), DDPG RL agent training (03),
and system evaluation (04). It establishes system-wide constraints including total wireless spectrum
bandwidth, per-slice latency Service Level Agreements (SLAs), priority weighting vectors, multi-objective
utility weights, SLA penalty schedules, patient cohort and temporal train/test split boundaries, and
system verification target thresholds.

Inputs: None (defines fundamental system constants).
Outputs: System parameters and hyperparameter constants consumed by environment.py and all notebook scripts.
"""

# ==============================================================================
# Spectrum & Bandwidth Allocation Parameters
# ==============================================================================
# Total available wireless spectrum bandwidth across all active network slices (in MHz / resource units).
# Scaled to 30000.0 MHz for the 10,000-patient full population scale (average budget: 3.0 MHz/patient).
TOTAL_BW = 30000.0

# ==============================================================================
# Per-Slice SLA Latency Requirements (ms)
# ==============================================================================
# Maps SliceID to maximum acceptable end-to-end delay threshold in milliseconds (ms).
# Slice 1: Emergency vital monitoring (SLA target: 5.0 ms)
# Slice 2: Intensive Care Unit / ICU continuous telemetry (SLA target: 15.0 ms)
# Slice 3: Ambulatory ECG streaming (SLA target: 30.0 ms)
# Slice 4: Telemedicine video consultation (SLA target: 60.0 ms)
# Slice 5: General ward health monitoring / bulk data (SLA target: 100.0 ms)
# NOTE: Paper draft mentions strict ultra-reliable low-latency bounds (URLLC < 10ms); here S2-S5 extend up to 100ms.
LATENCY_REQ_MAP = {
    1: 5.0,
    2: 15.0,
    3: 30.0,
    4: 60.0,
    5: 100.0
}

# ==============================================================================
# Composite Priority Score Weights & Normalizers (Equation 15 / Section 3.4)
# ==============================================================================
# Weight assigned to Emergency vital status flag in composite priority score (dimensionless, w1 = 0.5)
W_PRIO_EMERGENCY = 0.5
# Weight assigned to normalized inverse latency requirement in composite priority score (dimensionless, w2 = 0.3)
W_PRIO_LATENCY = 0.3
# Weight assigned to normalized traffic demand in composite priority score (dimensionless, w3 = 0.2)
W_PRIO_TRAFFIC = 0.2

# Minimum latency requirement anchor used for relative delay normalization (in ms, min across slices = 5.0 ms)
LATENCY_REQ_MIN = 5.0
# Maximum expected traffic demand ceiling used for traffic normalization in priority score (in traffic volume units)
# NOTE: In environment.py state normalization, traffic is divided by 100.0 instead of TRAFFIC_MAX (150.0).
TRAFFIC_MAX = 150.0

# ==============================================================================
# Multi-Objective Scalar Utility Function Weights (Equation 4.8 / Section 3.4)
# ==============================================================================
# Weights balance trade-offs in system performance; must sum to 1.0 (w_Th + w_Rel + w_Lat + w_Pwr + w_REU = 1.0).
# Weight for normalized raw aggregate data throughput (dimensionless, rewards high data rate)
W_UTIL_THROUGHPUT = 0.3
# Weight for SLA reliability exponential satisfaction metric (dimensionless, rewards meeting latency SLA)
W_UTIL_RELIABILITY = 0.3
# Weight for transmission + queueing latency penalty (dimensionless, penalizes delay)
W_UTIL_LATENCY = 0.2
# Weight for energy power consumption penalty (dimensionless, penalizes high power draw)
W_UTIL_POWER = 0.1
# Weight for Resource Energy Efficiency / REU ratio (dimensionless, rewards throughput per watt)
W_UTIL_REU = 0.1

# ==============================================================================
# SLA Violation Penalty Schedule per Slice (Section 6.4)
# ==============================================================================
# Additive penalty values subtracted from utility when a patient exceeds their slice latency threshold (in utility units).
# Higher penalties enforce strict priority compliance for critical medical slices (Emergency and ICU).
# NOTE: Earlier draft iterations specified higher penalties for S3-S5 {3: 150.0, 4: 100.0, 5: 50.0};
# current calibrated implementation uses {3: 20.0, 4: 10.0, 5: 5.0} matching PROJECT_WORKFLOW.md spec.
SLA_PENALTIES = {
    1: 350.0,
    2: 300.0,
    3: 20.0,
    4: 10.0,
    5: 5.0
}

# ==============================================================================
# Patient Cohort & Temporal Train/Test Split Boundaries
# ==============================================================================
# Disjoint patient cohort split: PatientIDs 1 to 8000 (80%) for training, 8001 to 10000 (20%) for testing
TRAIN_PATIENT_LIMIT = 8000  # Patients 1-8000 for training, 8001-10000 for test
# Temporal split boundary: Time steps 0-79 (80 steps) for training, 80-99 (20 steps) for testing/evaluation
TRAIN_TIME_LIMIT = 80       # Time steps 0-79 for training, 80-99 for test

# ==============================================================================
# System Performance Verification Target Thresholds
# ==============================================================================
# Minimum reliability target per patient-step (allowing ~10.53% latency overshoot; violated by 25.0226% of baseline rows)
R_MIN = 0.90     # Minimum reliability target per patient-step
# Minimum aggregate throughput per time step in Gbps at the 2,000-patient test cohort scale (violated by 8.00% of baseline time steps)
TH_MIN = 23.61   # Minimum aggregate throughput per time step in Gbps
# Minimum REU target per patient-step (corresponds to exactly the 11.8766% percentile of baseline rows)
REU_MIN = 0.02   # Minimum REU target per patient-step


