# Written against DDPG network architecture:
# Actor (10->64->32->1)

import pandas as pd
import numpy as np
import torch
import sys
import os

# Add the project directory to sys.path to import config and environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from environment import NetworkSlicingEnv

# Define Actor
class Actor(torch.nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * 15.0

def run_diversity_check(actor_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = Actor(state_dim=10, action_dim=1).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    
    df = pd.read_csv('dataset.csv')
    test_patient_ids = list(range(config.TRAIN_PATIENT_LIMIT + 1, 10001))
    num_test_patients = len(test_patient_ids)
    TOTAL_BW_test = config.TOTAL_BW * (num_test_patients / 10000.0) # 6000.0
    
    env = NetworkSlicingEnv(df)
    state = env.reset(test_patient_ids, start_time=0)
    done = False
    raw_acts_list = []
    scaled_acts_list = []
    
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            raw_actions = actor(state_t).cpu().numpy().flatten()
        scaled_actions = raw_actions * (TOTAL_BW_test / np.sum(raw_actions))
        raw_acts_list.append(raw_actions)
        scaled_acts_list.append(scaled_actions)
        next_state, reward, done, info = env.step(scaled_actions)
        state = next_state
        
    raw_acts = np.concatenate(raw_acts_list)
    scaled_acts = np.concatenate(scaled_acts_list)
    
    print(f"Actor Checkpoint: {actor_path}")
    print(f"Raw Actions - Mean: {np.mean(raw_acts):.4f}, Std: {np.std(raw_acts):.4f}, Range: [{np.min(raw_acts):.4f}, {np.max(raw_acts):.4f}]")
    print(f"Scaled Actions - Mean: {np.mean(scaled_acts):.4f}, Std: {np.std(scaled_acts):.4f}, Range: [{np.min(scaled_acts):.4f}, {np.max(scaled_acts):.4f}]")
    print("-" * 50)
    
    # Save the output to a text file for reporting
    out_file = 'action_diversity_output.txt'
    with open(out_file, 'w', encoding='utf-8') as out_f:
        out_f.write(f"Actor Checkpoint: {actor_path}\n")
        out_f.write(f"Raw Actions - Mean: {np.mean(raw_acts):.4f}, Std: {np.std(raw_acts):.4f}, Range: [{np.min(raw_acts):.4f}, {np.max(raw_acts):.4f}]\n")
        out_f.write(f"Scaled Actions - Mean: {np.mean(scaled_acts):.4f}, Std: {np.std(scaled_acts):.4f}, Range: [{np.min(scaled_acts):.4f}, {np.max(scaled_acts):.4f}]\n")

if __name__ == '__main__':
    print("=== ACTION DIVERSITY RUN ===")
    run_diversity_check('ddpg_actor_balanced.pth')
    run_diversity_check('ddpg_actor_FINAL.pth')
