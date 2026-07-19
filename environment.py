import numpy as np
import pandas as pd
import config

class NetworkSlicingEnv:
    def __init__(self, df):
        """
        df: The pandas DataFrame loaded from dataset.csv
        """
        self.df = df
        self.rng = np.random.default_rng(42)
        self.reset_env()
        
    def reset(self, patient_ids, start_time=0):
        """
        Resets the environment for a specific set of patient IDs and a starting time.
        """
        self.patient_ids = list(patient_ids)
        self.num_patients = len(self.patient_ids)
        self.current_time = start_time
        
        # Pre-cache patient data for active patients to avoid slow dataframe filtering during steps
        active_df = self.df[self.df['PatientID'].isin(self.patient_ids)].copy()
        active_df['PatientID'] = pd.Categorical(active_df['PatientID'], categories=self.patient_ids, ordered=True)
        active_df = active_df.sort_values(['Time', 'PatientID'])
        
        # Cache array data by time step
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
        
        # Initialize prev_bandwidth and prev_latency from baseline
        t_data = self.time_data[start_time]
        self.prev_bandwidth = t_data['Bandwidth'].copy()
        self.prev_latency = t_data['Latency'].copy()
        self.queues = t_data['Queue'].copy()
        self.lambdas = t_data['Lambda'].copy()
            
        return self._get_state()
        
    def _get_state(self):
        """
        Constructs and returns the state matrix of shape (num_patients, 10) for active patients.
        The state features are normalized to [0, 1] range for training stability.
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
        
        # Construct the state matrix in a vectorized way
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
        actions: list or numpy array of shape (num_patients,) containing allocated bandwidths
        """
        actions = np.array(actions, dtype=np.float32).flatten()
        actions = np.maximum(actions, 0.05)  # Enforce minimum bandwidth of 0.05 to prevent latency explosion
        
        t_data = self.time_data[self.current_time]
        datasize = t_data['DataSize']
        sinr = t_data['SINR']
        mu = t_data['Mu']
        lambda_val = self.lambdas.copy()
        slice_id = t_data['SliceID']
        latency_req = t_data['LatencyReq']
        
        # 1. Throughput (Eq. 4.4)
        throughputs = actions * np.log2(1.0 + sinr)
        
        # 2. Latency (Eq. 4.3)
        queueing_delay = 1.0 / np.maximum(mu - lambda_val, 1e-3)
        latencies = (datasize / (actions * np.log2(1.0 + sinr))) + queueing_delay
        latencies = np.minimum(latencies, 200.0)  # Hard-clip latency at a realistic maximum of 200ms
        
        # 3. Reliability (Eq. 4.5)
        reliabilities = np.ones_like(latencies)
        violating = latencies > latency_req
        if np.any(violating):
            reliabilities[violating] = np.exp(-(latencies[violating] - latency_req[violating]) / latency_req[violating])
            
        # 4. Power (Eq. 4.6)
        p_devices = self.rng.uniform(0.5, 2.0, size=self.num_patients)
        p_ais = self.rng.uniform(50.0, 150.0, size=self.num_patients)
        
        p_bs = 5.0 + 1.0 * actions
        p_edge = 10.0 + 0.05 * datasize
        powers = p_devices + p_bs + p_edge + p_ais
        
        # 5. REU (Eq. 4.7)
        reus = throughputs / powers
        
        # 6. Utility (Eq. 4.8) - with power normalized by offset of 150.0
        powers_norm = powers - 150.0
        utilities = (config.W_UTIL_THROUGHPUT * throughputs +
                     config.W_UTIL_RELIABILITY * reliabilities -
                     config.W_UTIL_LATENCY * latencies -
                     config.W_UTIL_POWER * powers_norm +
                     config.W_UTIL_REU * reus)
        
        # 7. SLA Violation Penalty (Section 6.4)
        rewards = utilities.copy()
        penalties = np.array([config.SLA_PENALTIES.get(s_id, 0.0) for s_id in slice_id])
        violation_ratio = (latencies - latency_req) / latency_req
        rewards[violating] = rewards[violating] - penalties[violating] * (1.0 + violation_ratio[violating])
        rewards = np.clip(rewards, -500.0, 500.0)  # Clip patient rewards to [-500, 500] range
        
        # Update history for next step
        self.prev_bandwidth = actions.copy()
        self.prev_latency = latencies.copy()
        
        self.current_time += 1
        
        max_time = self.df['Time'].max()
        done = self.current_time > max_time
        
        if not done:
            # Read Queue and Lambda directly from dataset.csv per timestep
            self.queues = self.time_data[self.current_time]['Queue'].copy()
            self.lambdas = self.time_data[self.current_time]['Lambda'].copy()
            
        next_state = None if done else self._get_state()
        
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
        self.patient_ids = []
        self.num_patients = 0
        self.current_time = 0
        self.prev_bandwidth = np.array([], dtype=np.float32)
        self.prev_latency = np.array([], dtype=np.float32)
        self.queues = np.array([], dtype=np.float32)
        self.lambdas = np.array([], dtype=np.float32)
        self.time_data = {}
