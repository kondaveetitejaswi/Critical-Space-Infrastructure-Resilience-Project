import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TransitionModel:
    """Model that predicts next state given current state and action"""
    def __init__(self, num_satellites: int, num_planes: int,
                 w1: float = 10.0, w2: float = 5.0,
                 w3: float = 2.0, w4: float = 5.0, w5: float = 1.0):
        # Basic parameters
        self.num_satellites = num_satellites
        self.num_planes = num_planes
        
        # Degradation rates
        self.health_decay_rate = 0.999  # Slower degradation
        self.signal_decay_rate = 0.999
        self.attack_success_prob = 0.8
        
        # Attack impact parameters
        self.jamming_impacts = {
            'health': 0.05,
            'signal': 0.30
        }
        self.spoofing_impacts = {
            'health': 0.30,
            'signal': 0.05
        }
        
        # Reward weights
        self.w1 = w1  # Coverage quality weight
        self.w2 = w2  # System health weight
        self.w3 = w3  # Action cost weight
        self.w4 = w4  # Outage penalty weight
        self.w5 = w5  # Recovery time penalty weight
    
    def predict_next_state(self, current_state: Dict, action: str,
                          attack_type: str = None) -> Tuple[Dict, float]:
        """Predict next state and reward given current state and action"""
        next_state = current_state.copy()
        
        # Natural degradation
        next_state['system_health'] *= self.health_decay_rate
        next_state['coverage_quality'] *= self.signal_decay_rate
        
        # Apply action effects
        if action == 'REPLACE_SATELLITE':
            if next_state['healthy_spares'] > 0:
                next_state['healthy_spares'] -= 1
                next_state['system_health'] *= 1.1
                next_state['coverage_quality'] *= 1.1
                next_state['operational_count'] = min(
                    next_state['operational_count'] + 1,
                    self.num_satellites
                )
        
        elif action == 'INCREASE_SIGNAL_POWER':
            next_state['coverage_quality'] = min(1.0, 
                next_state['coverage_quality'] * 1.2)
        
        # Apply attack effects if successful
        if attack_type and np.random.random() < self.attack_success_prob:
            impacts = (self.jamming_impacts if attack_type == 'jamming' 
                      else self.spoofing_impacts)
            next_state['system_health'] *= impacts['health']
            next_state['coverage_quality'] *= impacts['signal']
        
        # Calculate reward
        reward = self.calculate_reward(current_state, next_state, action)
        
        return next_state, reward
    
    def calculate_reward(self, current_state: Dict, next_state: Dict, 
                        action: str) -> float:
        """Calculate reward using weighted components"""
        reward = 0.0
        
        # Coverage and health rewards
        reward += self.w1 * next_state['coverage_quality']
        reward += self.w2 * next_state['system_health']
        
        # Action costs
        if action == 'REPLACE_SATELLITE':
            reward -= self.w3
        elif action == 'INCREASE_SIGNAL_POWER':
            reward -= self.w3 * 0.5
        
        # Outage penalty
        if next_state['operational_count'] < current_state['operational_count']:
            reward -= self.w4 * (current_state['operational_count'] - 
                                next_state['operational_count'])
        
        # Recovery time penalty
        if next_state['system_health'] < 0.5:
            reward -= self.w5
        
        return reward
    
    def simulate_trajectory(self, initial_state: Dict, policy,
                          horizon: int = 10, num_rollouts: int = 5) -> List[float]:
        """Simulate multiple trajectories using the model"""
        rewards = []
        
        for _ in range(num_rollouts):
            state = initial_state.copy()
            trajectory_reward = 0.0
            discount = 1.0
            
            for t in range(horizon):
                # Get action from policy
                action = policy.select_action(state)
                
                # Simulate random attack
                attack = np.random.choice(['jamming', 'spoofing', None], 
                                        p=[0.2, 0.2, 0.6])
                
                # Get next state and reward
                next_state, reward = self.predict_next_state(state, action, attack)
                trajectory_reward += discount * reward
                discount *= 0.99
                state = next_state
            
            rewards.append(trajectory_reward)
        
        return rewards

# if __name__ == "__main__":
#     # Test the transition model
#     model = TransitionModel(num_satellites=8, num_planes=3)
    
#     initial_state = {
#         'system_health': 1.0,
#         'coverage_quality': 1.0,
#         'operational_count': 6,
#         'healthy_spares': 2
#     }
    
#     # Test state prediction
#     next_state, reward = model.predict_next_state(
#         initial_state, 
#         'REPLACE_SATELLITE',
#         'jamming'
#     )
    
#     print("Initial State:", initial_state)
#     print("Next State:", next_state)
#     print("Reward:", reward)
