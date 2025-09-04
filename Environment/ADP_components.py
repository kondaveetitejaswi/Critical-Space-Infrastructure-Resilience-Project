import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from transition_model import TransitionModel 

class ConstellationState:
    def __init__(self, state_dict: Dict):
        self.system_health = state_dict.get('system_health', 0.0)
        self.coverage_quality = state_dict.get('coverage_quality', 0.0)
        self.operational_count = state_dict.get('operational_count', 0)
        self.healthy_spares = state_dict.get('healthy_spares', 0)
        self.time_step = state_dict.get('time_step', 0)
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for NN input"""
        return np.array([
            self.system_health,
            self.coverage_quality,
            self.operational_count,
            self.healthy_spares
        ])

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class ModelBasedADPLearner:
    def __init__(self, transition_model: TransitionModel, 
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 planning_horizon: int = 5, num_rollouts: int = 10):
        self.value_function = ValueFunction()
        self.optimizer = torch.optim.Adam(self.value_function.parameters(), lr=learning_rate)
        self.transition_model = transition_model
        self.gamma = gamma
        self.planning_horizon = planning_horizon
        self.num_rollouts = num_rollouts
    
    def select_action(self, state: ConstellationState, epsilon: float = 0.1) -> str:
        """Select action using ε-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.choice(['REPLACE_SATELLITE', 'ACTIVATE_BACKUP', 
                                   'INCREASE_SIGNAL_POWER'])
        
        # Get value for each action directly without rollouts
        action_values = []
        state_tensor = torch.FloatTensor(state.to_vector())
        
        for action in ['REPLACE_SATELLITE', 'ACTIVATE_BACKUP', 'INCREASE_SIGNAL_POWER']:
            # Predict next state
            next_state_dict, reward = self.transition_model.predict_next_state(
                vars(state), action)
            next_state = ConstellationState(next_state_dict)
            next_state_tensor = torch.FloatTensor(next_state.to_vector())
            
            # Calculate action value
            with torch.no_grad():
                current_value = self.value_function(state_tensor)
                next_value = self.value_function(next_state_tensor)
                action_value = reward + self.gamma * next_value
                action_values.append((action, action_value.item()))
        
        return max(action_values, key=lambda x: x[1])[0]
    
    def update(self, state: ConstellationState, action: str,
               reward: float, next_state: ConstellationState):
        """Update value function using TD learning"""
        state_tensor = torch.FloatTensor(state.to_vector())
        next_state_tensor = torch.FloatTensor(next_state.to_vector())
        
        current_value = self.value_function(state_tensor)
        next_value = self.value_function(next_state_tensor).detach()
        expected_value = reward + self.gamma * next_value
        
        loss = nn.MSELoss()(current_value, expected_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_from_model(self):
        """Perform one step of model-based planning"""
        state_dict = {
            'system_health': np.random.uniform(0.5, 1.0),
            'coverage_quality': np.random.uniform(0.5, 1.0),
            'operational_count': np.random.randint(3, 7),
            'healthy_spares': np.random.randint(0, 3),
            'time_step': 0
        }
        state = ConstellationState(state_dict)
        action = self.select_action(state)
        next_state_dict, reward = self.transition_model.predict_next_state(
            vars(state), action)
        next_state = ConstellationState(next_state_dict)
        self.update(state, action, reward, next_state)

