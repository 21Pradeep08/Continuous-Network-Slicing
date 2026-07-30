"""
5G Network Slicing Custom Gymnasium Simulation Environment.

This module defines `NetworkSlicingEnv`, an object-oriented simulation environment that models
joint wireless channel resource contention, transmission delay, M/M/1 queueing delay, energy
consumption, multi-objective utility, and Service Level Agreement (SLA) penalty dynamics across
5G network slices for connected patient cohorts.

Position in Pipeline:
    Core environment engine utilized during DDPG agent training (`03_train_ddpg.ipynb`) and
    policy evaluation (`04_evaluate.ipynb`).

Inputs:
    - `df`: Pandas DataFrame loaded from `dataset.csv` containing patient vital signs, slice assignments,
      channel SINR, queue states, arrival/service rates, and predicted traffic demands.
    - Configuration constants from `config.py` (`TOTAL_BW`, `W_UTIL_*`, `SLA_PENALTIES`, `LATENCY_REQ_MAP`).

Outputs:
    - State matrix `S` of shape `(num_patients, 10)` with normalized physical features.
    - Step reward array `R` of shape `(num_patients,)` incorporating multi-objective utility and SLA penalties.
    - Episode completion flag `done`.
    - Physical metric telemetry dictionary `info` (latency, throughput, reliability, power, REU, utility, reward).
"""

import numpy as np
import pandas as pd
import config

class NetworkSlicingEnv:
    def __init__(self, df):
        """
        Initializes the 5G network slicing environment with pre-loaded dataset records.
        
        df: The pandas DataFrame loaded from dataset.csv containing per-patient state features over time.
        """
        self.df = df
        self.rng = np.random.default_rng(42)
        self.reset_env()
        
    def reset(self, patient_ids, start_time=0):
        """
        Resets the environment state for a specific subset cohort of active patient IDs and initial time step.
        Pre-caches array attributes by time step to optimize state retrieval during execution.
        """
        self.patient_ids = list(patient_ids)
        self.num_patients = len(self.patient_ids)
        self.current_time = start_time
        
        # Pre-cache patient data for active patients to avoid slow dataframe filtering during steps
        active_df = self.df[self.df['PatientID'].isin(self.patient_ids)].copy()
        active_df['PatientID'] = pd.Categorical(active_df['PatientID'], categories=self.patient_ids, ordered=True)
        active_df = active_df.sort_values(['Time', 'PatientID'])
        
        # Cache array data by time step into contiguous float32/int32 numpy arrays
        self.time_data = {}
        for t in sorted(active_df['Time'].unique()):
            subset = active_df[active_df['Time'] == t]
            self.time_data[t] = {
                'DataSize': subset['DataSize'].values.astype(np.float32),
                'SINR': subset['SINR'].values.astype(np.float32),
                'Mu': subset['Mu'].values.astype(np.float32),
                'Lambda': subset['Lambda'].values.astype(np.float32),
                'SliceID': subset['SliceID'].values.astype(np.int32),
                'LatencyReq': subset['LatencyReq'].values.astype(np.float32),
                'Priority': subset['Priority'].values.astype(np.float32),
                'Emergency': subset['Emergency'].values.astype(np.float32),
                'Queue': subset['Queue'].values.astype(np.float32),
                'Traffic': subset['Traffic'].values.astype(np.float32),
                'Predicted_Traffic': subset['Predicted_Traffic'].values.astype(np.float32),
                'Bandwidth': subset['Bandwidth'].values.astype(np.float32),
                'Latency': subset['Latency'].values.astype(np.float32)
            }
        
        # Initialize prev_bandwidth and prev_latency from baseline dataset values
        t_data = self.time_data[start_time]
        self.prev_bandwidth = t_data['Bandwidth'].copy()
        self.prev_latency = t_data['Latency'].copy()
        self.queues = t_data['Queue'].copy()
        self.lambdas = t_data['Lambda'].copy()
            
        return self._get_state()
        
    def _get_state(self):
        """
        Algorithm Step: DDPG Environment State Vector Construction.
        
        Constructs and returns the normalized state matrix of shape (num_patients, 10) for active patients.
        All 10 features are scaled by specific normalizer bounds to promote stable gradient propagation:
          1. norm_traffic   : Predicted traffic demand normalized by 100.0 (units of traffic demand).
                              # NOTE: Exceeds 1.0 for ~6.6% of rows (max ~100.6); does not use config.TRAFFIC_MAX (150).
          2. norm_priority  : Composite priority score in range [0, 1] (Equation 15).
          3. norm_emergency : Medical emergency vital status flag {0.0, 1.0} (ICU/Emergency = 1.0).
          4. norm_sinr      : Channel Signal-to-Interference-plus-Noise Ratio normalized by 30.0 dB.
          5. norm_queue     : Network buffer queue length normalized by baseline cap 15.0 packets.
                              # NOTE: Dynamic queues can reach up to 50.0, leading to normalized values > 1.0.
          6. norm_prev_bw   : Allocated bandwidth in previous timestep normalized by average per-slice budget
                              (TOTAL_BW / 10000 = 3.0 MHz).
                              # NOTE: Baseline bandwidth ranges up to 63.5 MHz, leading to values up to ~21.17.
          7. norm_prev_lat  : Experienced latency in previous timestep normalized by maximum delay scale 100.0 ms.
          8. norm_datasize  : Medical packet payload size normalized by maximum payload scale 500.0 KB.
          9. norm_lambda    : Queue packet arrival rate lambda normalized by maximum arrival scale 10.0 pkts/ms.
         10. norm_mu        : Queue channel service capacity rate mu normalized by maximum service scale 20.0 pkts/ms.
        """
        t_data = self.time_data[self.current_time]
        
        norm_traffic = t_data['Predicted_Traffic'] / 100.0
        norm_priority = t_data['Priority']
        norm_emergency = t_data['Emergency']
        norm_sinr = t_data['SINR'] / 30.0
        norm_queue = self.queues / 15.0
        norm_prev_bw = self.prev_bandwidth / (config.TOTAL_BW / 10000.0)
        norm_prev_lat = self.prev_latency / 100.0
        norm_datasize = t_data['DataSize'] / 500.0
        norm_lambda = self.lambdas / 10.0
        norm_mu = t_data['Mu'] / 20.0
        
        # Construct the 10-dimensional state matrix in a vectorized row-by-row layout
        state_matrix = np.column_stack([
            norm_traffic,
            norm_priority,
            norm_emergency,
            norm_sinr,
            norm_queue,
            norm_prev_bw,
            norm_prev_lat,
            norm_datasize,
            norm_lambda,
            norm_mu
        ]).astype(np.float32)
        
        return state_matrix

    def step(self, actions):
        """
        Algorithm Step: Environment Transition & Multi-Objective Reward Computation.
        
        Executes one physical environment step given an allocated bandwidth vector `actions` of shape (num_patients,).
        Computes 5 core physical metrics (Throughput, Latency, Reliability, Power, REU), aggregates scalar utility,
        and applies dynamic SLA penalties to yield step rewards.
        """
        actions = np.array(actions, dtype=np.float32).flatten()
        # Enforce strict minimum bandwidth floor of 0.05 MHz per patient to avoid division-by-zero latency explosion
        actions = np.maximum(actions, 0.05)
        
        t_data = self.time_data[self.current_time]
        datasize = t_data['DataSize']
        sinr = t_data['SINR']
        mu = t_data['Mu']
        lambda_val = self.lambdas.copy()
        slice_id = t_data['SliceID']
        latency_req = t_data['LatencyReq']
        
        # 1. Throughput Calculation (Equation 4.4): R_i = B_i * log2(1 + SINR_i)
        throughputs = actions * np.log2(1.0 + sinr)
        
        # 2. Latency Calculation (Equation 4.3): Transmission delay + M/M/1 queueing delay W_q = 1 / (mu - lambda)
        queueing_delay = 1.0 / np.maximum(mu - lambda_val, 1e-3)
        latencies = (datasize / (actions * np.log2(1.0 + sinr))) + queueing_delay
        latencies = np.minimum(latencies, 200.0)  # Hard-clip latency at a realistic ceiling of 200.0 ms
        
        # 3. Reliability Metric (Equation 4.5): Exponential decay metric penalty for SLA latency overshoot
        reliabilities = np.ones_like(latencies)
        violating = latencies > latency_req
        if np.any(violating):
            reliabilities[violating] = np.exp(-(latencies[violating] - latency_req[violating]) / latency_req[violating])
            
        # 4. Power Consumption Model (Equation 4.6): P_total = P_device + P_BS(B_i) + P_edge(DataSize) + P_AIS
        p_devices = self.rng.uniform(0.5, 2.0, size=self.num_patients)
        p_ais = self.rng.uniform(50.0, 150.0, size=self.num_patients)
        
        p_bs = 5.0 + 1.0 * actions
        p_edge = 10.0 + 0.05 * datasize
        powers = p_devices + p_bs + p_edge + p_ais
        
        # 5. Resource Energy Efficiency / REU (Equation 4.7): REU_i = Throughput_i / Power_i
        reus = throughputs / powers
        
        # 6. Scalar Multi-Objective Utility (Equation 4.8):
        # U = w_th * Throughput + w_rel * Reliability - w_lat * Latency - w_pwr * Power_norm + w_reu * REU
        powers_norm = powers - 150.0  # Normalize power consumption by offset of 150.0 W
        utilities = (config.W_UTIL_THROUGHPUT * throughputs +
                     config.W_UTIL_RELIABILITY * reliabilities -
                     config.W_UTIL_LATENCY * latencies -
                     config.W_UTIL_POWER * powers_norm +
                     config.W_UTIL_REU * reus)
        
        # 7. SLA Violation Penalty & Step Reward (Section 6.4):
        # Subtract penalty proportional to latency overshoot ratio: R_i = U_i - penalty_i * (1 + (L_i - L_req_i) / L_req_i)
        rewards = utilities.copy()
        penalties = np.array([config.SLA_PENALTIES.get(s_id, 0.0) for s_id in slice_id])
        violation_ratio = (latencies - latency_req) / latency_req
        rewards[violating] = rewards[violating] - penalties[violating] * (1.0 + violation_ratio[violating])
        rewards = np.clip(rewards, -500.0, 500.0)  # Clip patient rewards to [-500, 500] range for training stability
        
        # Update state history buffer attributes for next environment timestep
        self.prev_bandwidth = actions.copy()
        self.prev_latency = latencies.copy()
        
        self.current_time += 1
        
        max_time = self.df['Time'].max()
        done = self.current_time > max_time
        
        if not done:
            # Read Queue and Lambda state arrays directly from pre-cached time_data for the new timestep
            self.queues = self.time_data[self.current_time]['Queue'].copy()
            self.lambdas = self.time_data[self.current_time]['Lambda'].copy()
            
        next_state = None if done else self._get_state()
        
        # Telemetry info container returned to caller
        info = {
            'latency': latencies.astype(np.float32),
            'throughput': throughputs.astype(np.float32),
            'reliability': reliabilities.astype(np.float32),
            'power': powers.astype(np.float32),
            'reu': reus.astype(np.float32),
            'utility': utilities.astype(np.float32),
            'reward': rewards.astype(np.float32)
        }
        
        return next_state, rewards.astype(np.float32), done, info
        
    def reset_env(self):
        """
        Resets environment attributes to default empty values.
        """
        self.patient_ids = []
        self.num_patients = 0
        self.current_time = 0
        self.prev_bandwidth = np.array([], dtype=np.float32)
        self.prev_latency = np.array([], dtype=np.float32)
        self.queues = np.array([], dtype=np.float32)
        self.lambdas = np.array([], dtype=np.float32)
        self.time_data = {}

