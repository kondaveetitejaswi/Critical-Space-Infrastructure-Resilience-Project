import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from dataclasses import dataclass
from typing import List, Dict
import random
from ADP_components import ConstellationState
from ADP_components import TransitionModel, ModelBasedADPLearner

@dataclass
class Satellite:
    """Basic satellite representation"""
    id: str
    status: str  # operational/spare/decommissioned
    health: float = 1.0
    signal_quality: float = 1.0
    plane_id: int = 0

class GalileoConstellation:
    def __init__(self, operational_count: int = 6, spare_count: int = 2):
        self.operational_count = operational_count
        self.spare_count = spare_count
        
        # Orbital planes configuration
        self.num_planes = 3
        self.plane_spacing = 120  # degrees
        self.inclination = 56  # degrees
        
        # Initialize satellites
        self.satellites: List[Satellite] = []
        self._initialize_constellation()
        
        # Constellation metrics
        self.system_health = 1.0
        self.coverage_quality = 1.0
        self.time_step = 0

        self.episode_rewards = []

        # Initialize model-based components
        self.transition_model = TransitionModel(
            num_satellites=operational_count + spare_count,
            num_planes=self.num_planes
        )
        self.adp_learner = ModelBasedADPLearner(self.transition_model)
        
    def _initialize_constellation(self):
        """Distribute satellites across orbital planes"""
        for i in range(self.operational_count):
            plane_id = i % self.num_planes
            self.satellites.append(
                Satellite(id=f"GSAT-{i+1:02d}", status="operational", plane_id=plane_id)
            )
        
        for i in range(self.spare_count):
            plane_id = i % self.num_planes
            self.satellites.append(
                Satellite(id=f"GSAT-S{i+1:02d}", status="spare", plane_id=plane_id)
            )

    def get_state(self) -> Dict:
        """Get constellation state"""
        return {
            "system_health": self.system_health,
            "coverage_quality": self.coverage_quality,
            "operational_count": sum(1 for sat in self.satellites if sat.status == "operational" and sat.health > 0.5),
            "healthy_spares": sum(1 for sat in self.satellites if sat.status == "spare" and sat.health > 0.8),
            "time_step": self.time_step
        }

    def apply_attack(self, attack_type: str):
        """Apply attack on a random operational satellite"""
        operational_sats = [sat for sat in self.satellites if sat.status == "operational"]
        
        if not operational_sats:
            return  # No operational satellite to attack
        
        target_sat = random.choice(operational_sats)
        
        if attack_type == "jamming":
            # Jamming impacts signal quality heavily, health slightly
            target_sat.signal_quality *= 0.7
            target_sat.health *= 0.95
        elif attack_type == "spoofing":
            # Spoofing impacts health heavily, signal quality slightly
            target_sat.health *= 0.7
            target_sat.signal_quality *= 0.95
        
        # If health drops too low, satellite is decommissioned
        if target_sat.health < 0.3:
            target_sat.status = "decommissioned"
            self._replace_satellite(target_sat)
        
        self._update_constellation_metrics()

    def _replace_satellite(self, damaged_sat: Satellite):
        """Replace damaged satellite with spare"""
        spare_available = next(
            (sat for sat in self.satellites if sat.status == "spare" and sat.health > 0.8 and sat.plane_id == damaged_sat.plane_id), 
            None
        )
        
        if spare_available:
            damaged_sat.status = "decommissioned"
            spare_available.status = "operational"

    def _update_constellation_metrics(self):
        operational_health = [sat.health for sat in self.satellites if sat.status == "operational"]
        
        self.system_health = np.mean(operational_health) if operational_health else 0
        self.coverage_quality = len([h for h in operational_health if h > 0.5]) / self.operational_count if self.operational_count else 0

    def calculate_reward(self, old_state: Dict, new_state: Dict, action: str) -> float:
        """Calculate reward based on state transition and action taken"""
        reward = 0
        
        # Coverage maintenance reward
        reward += new_state['coverage_quality'] * 10
        
        # System health reward
        reward += new_state['system_health'] * 5
        
        # Penalize satellite replacements
        if action == 'REPLACE_SATELLITE':
            reward -= 2
        
        # Penalize loss of operational satellites
        if new_state['operational_count'] < old_state['operational_count']:
            reward -= 5
        
        return reward

    def step(self, attack_prob: float = 0.3):
        """Modified step function with model-based learning"""
        old_state = self.get_state()
        
        # Get action using model-based planning
        constellation_state = ConstellationState(old_state)
        action = self.adp_learner.select_action(constellation_state)
        
        # Execute action in real environment
        self._execute_action(action)
        
        # Progress time and apply attacks
        self.time_step += 1
        for sat in self.satellites:
            if sat.status != "decommissioned":
                sat.health *= self.transition_model.health_decay_rate
                sat.signal_quality *= self.transition_model.signal_decay_rate
        
        # Apply attack based on model probabilities
        if random.random() < attack_prob:
            attack_type = random.choice(["jamming", "spoofing"])
            self.apply_attack(attack_type)
        
        self._update_constellation_metrics()
        new_state = self.get_state()
        
        # Update learner with real experience
        reward = self.calculate_reward(old_state, new_state, action)
        self.adp_learner.update(
            ConstellationState(old_state),
            action,
            reward,
            ConstellationState(new_state)
        )
        
        self.episode_rewards.append(reward)
        return new_state
    
    def _execute_action(self, action: str):
        """Execute the selected action"""
        if action == 'REPLACE_SATELLITE':
            operational_sats = [sat for sat in self.satellites if sat.status == "operational"]
            if operational_sats:
                damaged_sat = min(operational_sats, key=lambda x: x.health)
                self._replace_satellite(damaged_sat)
            
        elif action == 'ACTIVATE_BACKUP':
            spare_sats = [sat for sat in self.satellites if sat.status == "spare"]
            if spare_sats:  # Only proceed if there are spare satellites
                best_spare = max(spare_sats, key=lambda x: x.health)
                best_spare.status = "operational"
        
        elif action == 'REPOSITION_SATELLITE':
            # Add logic for repositioning
            pass
        
        elif action == 'INCREASE_SIGNAL_POWER':
            operational_sats = [sat for sat in self.satellites if sat.status == "operational"]
            for sat in operational_sats:
                sat.signal_quality = min(1.0, sat.signal_quality * 1.1)

    def visualize(self):
        """Visualize current constellation state"""
        plt.figure(figsize=(10, 8))
        colors = {'operational': 'g', 'spare': 'b', 'decommissioned': 'r'}
        
        for sat in self.satellites:
            plt.scatter(sat.plane_id * self.plane_spacing, sat.health * 100,
                       c=colors[sat.status], label=sat.status, alpha=0.7)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title(f"Galileo Constellation Status (t={self.time_step})")
        plt.xlabel("Plane Position (degrees)")
        plt.ylabel("Satellite Health (%)")
        plt.grid(True)
        plt.show()

    def visualize_2d_polar(self):
        """Visualize constellation in 2D polar plot"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        colors = {'operational': 'g', 'spare': 'b', 'decommissioned': 'r'}
        markers = {'operational': 'o', 'spare': 's', 'decommissioned': 'x'}
        
        for sat in self.satellites:
            # Calculate position
            theta = np.radians(sat.plane_id * self.plane_spacing + random.uniform(0, 120 / len(self.satellites)))

            r = 1.0  # Normalized radius
            
            # Adjust radius based on health
            r *= sat.health
            
            ax.scatter(theta, r, c=colors[sat.status], marker=markers[sat.status], 
                      s=100, alpha=0.7, label=sat.status)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title(f"Galileo Constellation Status (t={self.time_step})")
        plt.grid(True)
        plt.show()

    def visualize_3d(self):
        """Visualize constellation in 3D with complete orbital paths"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {'operational': 'g', 'spare': 'b', 'decommissioned': 'r'}
        markers = {'operational': 'o', 'spare': 's', 'decommissioned': 'x'}
        
        # Earth radius (scaled)
        earth_radius = 1.0
        orbit_radius = 1.3  # Scaled orbit radius
        
        # Draw Earth
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.1)
        
        # Draw orbital planes and satellites
        for plane in range(self.num_planes):
            # Draw complete orbital path
            theta = np.linspace(0, 2*np.pi, 100)
            phi = np.radians(self.inclination)
            
            # Calculate orbital path
            x_orbit = orbit_radius * np.cos(theta) * np.cos(phi)
            y_orbit = orbit_radius * np.sin(theta) * np.cos(phi)
            z_orbit = orbit_radius * np.sin(phi) * np.ones_like(theta)
            
            # Rotate orbital plane based on plane spacing
            rotation = plane * self.plane_spacing
            rotation_rad = np.radians(rotation)
            x_rotated = x_orbit * np.cos(rotation_rad) - y_orbit * np.sin(rotation_rad)
            y_rotated = x_orbit * np.sin(rotation_rad) + y_orbit * np.cos(rotation_rad)
            
            # Plot orbital path
            ax.plot(x_rotated, y_rotated, z_orbit, 'gray', alpha=0.2)
            
            # Plot satellites in this plane
            plane_sats = [sat for sat in self.satellites if sat.plane_id == plane]
            for idx, sat in enumerate(plane_sats):
                # Satellite position in base orbital plane
                theta = np.radians((360 / len(plane_sats)) * idx)
                x = orbit_radius * np.cos(theta)
                y = orbit_radius * np.sin(theta)
                z = 0

                # Apply inclination rotation (around x-axis)
                incl = np.radians(self.inclination)
                y_incl = y * np.cos(incl) - z * np.sin(incl)
                z_incl = y * np.sin(incl) + z * np.cos(incl)

                # Apply plane rotation (around z-axis)
                rot = np.radians(plane * self.plane_spacing)
                x_rot = x * np.cos(rot) - y_incl * np.sin(rot)
                y_rot = x * np.sin(rot) + y_incl * np.cos(rot)
                z_rot = z_incl

                # Final position
                x_sat_rotated, y_sat_rotated, z_sat_rotated = x_rot, y_rot, z_rot

                
                # Plot satellite
                ax.scatter(x_sat_rotated, y_sat_rotated, z_sat_rotated, 
                          c=colors[sat.status], 
                          marker=markers[sat.status],
                          s=100, alpha=0.7, 
                          label=sat.status)
        
        # Clean up legend (remove duplicates)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        ax.set_title(f"Galileo Constellation 3D View (t={self.time_step})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Add viewing angle
        ax.view_init(elev=30, azim=60)

        plt.show()

# Example usage
if __name__ == "__main__":
    # Create constellation
    constellation = GalileoConstellation(operational_count=6, spare_count=2)
    
    # Training episodes
    num_episodes = 100
    steps_per_episode = 50
    planning_steps = 10  # Number of model-based planning steps between real steps
    
    all_episode_rewards = []
    
    for episode in range(num_episodes):
        constellation = GalileoConstellation(operational_count=6, spare_count=2)
        episode_rewards = []
        
        for step in range(steps_per_episode):
            # Real environment step
            state = constellation.step(attack_prob=0.4)
            
            # Model-based planning steps
            for _ in range(planning_steps):
                constellation.adp_learner.update_from_model()
            
            if state['operational_count'] == 0:
                print(f"\nEpisode {episode+1} terminated early - No operational satellites")
                break
            
            if step % 10 == 0:
                print(f"\nEpisode {episode+1}, Step {step}")
                print(f"System Health: {state['system_health']:.2f}")
                print(f"Coverage Quality: {state['coverage_quality']:.2f}")
                print(f"Operational Count: {state['operational_count']}")
                print(f"Healthy Spares: {state['healthy_spares']}")
        
        avg_reward = np.mean(constellation.episode_rewards)
        all_episode_rewards.append(avg_reward)
        print(f"\nEpisode {episode+1} complete. Average reward: {avg_reward:.2f}")
