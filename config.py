# Configuration module for 5G Network Slicing and Resource Allocation

# Bandwidth configuration
TOTAL_BW = 30000.0

# Slice to Latency Requirement mapping (in ms)
LATENCY_REQ_MAP = {
    1: 5.0,
    2: 15.0,
    3: 30.0,
    4: 60.0,
    5: 100.0
}

# Priority Score Weights
W_PRIO_EMERGENCY = 0.5
W_PRIO_LATENCY = 0.3
W_PRIO_TRAFFIC = 0.2
LATENCY_REQ_MIN = 5.0
TRAFFIC_MAX = 150.0

# Utility Weights (must sum to 1.0)
W_UTIL_THROUGHPUT = 0.3
W_UTIL_RELIABILITY = 0.3
W_UTIL_LATENCY = 0.2
W_UTIL_POWER = 0.1
W_UTIL_REU = 0.1

# SLA Violation Penalties
SLA_PENALTIES = {
    1: 350.0,
    2: 300.0,
    3: 20.0,
    4: 10.0,
    5: 5.0
}

# Train/Test Split Constants
TRAIN_PATIENT_LIMIT = 8000  # Patients 1-8000 for training, 8001-10000 for test
TRAIN_TIME_LIMIT = 80       # Time steps 0-79 for training, 80-99 for test

# Verification Thresholds (Item 6)
R_MIN = 0.90     # Minimum reliability target per patient-step (allowing ~10.53% latency overshoot; violated by 25.0226% of baseline rows)
TH_MIN = 23.61   # Minimum aggregate throughput per time step in Gbps at the 2,000-patient test cohort scale (violated by 8.00% of baseline time steps)
REU_MIN = 0.02   # Minimum REU target per patient-step (corresponds to exactly the 11.8766% percentile of baseline rows)

