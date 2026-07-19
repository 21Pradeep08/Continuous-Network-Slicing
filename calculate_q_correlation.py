# Written against DDPG network architecture:
# Actor (10->64->32->1)
# Critic12 (12->64->32->1) from pre-Slice-ID baseline
# Critic17 (17->64->32->1) from PER + Slice-ID instrumentation

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
import sys
import os

# Add the project directory to sys.path to import config and environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from environment import NetworkSlicingEnv

# Define Actor
class Actor(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * 15.0

# Define 12-input Critic (pre-Slice-ID)
class Critic12(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super(Critic12, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action, contention_pressure):
        if contention_pressure.dim() == 1:
            contention_pressure = contention_pressure.unsqueeze(-1)
        if contention_pressure.size(0) != state.size(0):
            contention_pressure = contention_pressure.expand(state.size(0), -1)
        x = torch.cat([state, action, contention_pressure], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define 17-input Critic (with Slice-ID)
class Critic17(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super(Critic17, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + 1 + 5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action, contention_pressure, slice_onehot):
        if contention_pressure.dim() == 1:
            contention_pressure = contention_pressure.unsqueeze(-1)
        if contention_pressure.size(0) != state.size(0):
            contention_pressure = contention_pressure.expand(state.size(0), -1)
        x = torch.cat([state, action, contention_pressure, slice_onehot], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def run_correlation_check(critic_path, actor_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('dataset.csv')
    
    # Check Critic weight shape to dynamically choose architecture
    checkpoint = torch.load(critic_path, map_location='cpu')
    in_features = checkpoint['fc1.weight'].shape[1]
    
    # Load Actor
    actor = Actor(state_dim=10, action_dim=1).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    
    # Load correct Critic
    if in_features == 12:
        critic = Critic12(state_dim=10, action_dim=1).to(device)
    elif in_features == 17:
        critic = Critic17(state_dim=10, action_dim=1).to(device)
    else:
        raise ValueError(f"Unknown Critic input dimension: {in_features}")
        
    critic.load_state_dict(checkpoint)
    critic.eval()
    
    # Define test patients
    test_patient_ids = df['PatientID'].unique()[8000:10000]
    num_test_patients = len(test_patient_ids)
    TOTAL_BW_test = config.TOTAL_BW * (num_test_patients / 10000.0) # 6000.0
    
    env = NetworkSlicingEnv(df)
    state = env.reset(test_patient_ids, start_time=0)
    
    states_list = []
    actions_list = []
    rewards_list = []
    q_preds_list = []
    
    done = False
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            raw_actions = actor(state_t)
            
        C_t = torch.sum(raw_actions) / TOTAL_BW_test
        C_t_val = C_t.item()
        
        actions = (raw_actions * (TOTAL_BW_test / raw_actions.sum())).cpu().numpy().flatten()
        actions_t = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1).to(device)
        
        with torch.no_grad():
            if in_features == 17:
                # FIX: Extract dynamic slice ID before stepping
                step_slice_ids = env.time_data[env.current_time]['SliceID']
                step_slice_onehot = np.eye(5)[step_slice_ids - 1]
                step_slice_onehot_t = torch.tensor(step_slice_onehot, dtype=torch.float32).to(device)
                q_pred = critic(state_t, actions_t, torch.tensor([[C_t_val]], dtype=torch.float32).to(device), step_slice_onehot_t).cpu().numpy().flatten()
            else:
                q_pred = critic(state_t, actions_t, torch.tensor([[C_t_val]], dtype=torch.float32).to(device)).cpu().numpy().flatten()
            
        states_list.append(state.copy())
        actions_list.append(actions.copy())
        q_preds_list.append(q_pred)
        
        next_state, reward, done, info = env.step(actions)
        rewards_list.append(reward.copy())
        state = next_state
        
    # Compute empirical returns G_t for each patient (scaled by 0.01 to match DDPG critic target scale)
    rewards_arr = np.array(rewards_list) * 0.01
    q_preds_arr = np.array(q_preds_list)
    
    T, N = rewards_arr.shape
    returns_arr = np.zeros_like(rewards_arr)
    
    gamma = 0.99
    G = np.zeros(N)
    for t in reversed(range(T)):
        G = rewards_arr[t] + gamma * G
        returns_arr[t] = G
        
    flat_q_preds = q_preds_arr.flatten()
    flat_returns = returns_arr.flatten()
    
    pearson = np.corrcoef(flat_q_preds, flat_returns)[0, 1]
    spearman, _ = spearmanr(flat_q_preds, flat_returns)
    
    print(f"Critic: {critic_path} (evaluated with {actor_path})")
    print(f"Pearson Correlation: {pearson:.4f}")
    print(f"Spearman Rank Correlation: {spearman:.4f}")
    print(f"Q_pred std: {flat_q_preds.std():.4f}, range: [{flat_q_preds.min():.4f}, {flat_q_preds.max():.4f}]")
    print(f"Return std: {flat_returns.std():.4f}, range: [{flat_returns.min():.4f}, {flat_returns.max():.4f}]")
    print("-" * 50)
    
    # Save the output to a text file for reporting
    out_file = 'q_correlation_output.txt'
    with open(out_file, 'w', encoding='utf-8') as out_f:
        out_f.write(f"Critic: {critic_path}\n")
        out_f.write(f"Pearson Correlation: {pearson:.4f}\n")
        out_f.write(f"Spearman Rank Correlation: {spearman:.4f}\n")
        out_f.write(f"Q_pred std: {flat_q_preds.std():.4f}, range: [{flat_q_preds.min():.4f}, {flat_q_preds.max():.4f}]\n")
        out_f.write(f"Return std: {flat_returns.std():.4f}, range: [{flat_returns.min():.4f}, {flat_returns.max():.4f}]\n")

if __name__ == '__main__':
    print("=== Q-CALIBRATION CORRELATION RUN ===")
    run_correlation_check('ddpg_critic_balanced.pth', 'ddpg_actor_balanced.pth')
    run_correlation_check('ddpg_critic_FINAL.pth', 'ddpg_actor_FINAL.pth')
