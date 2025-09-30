import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from dataclasses import dataclass
from typing import List, Dict
import random
from common import ConstellationState
from transition_model import ProperModelBasedADPLearner

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

        # Initialize model-based components - using the proper implementation
        self.adp_learner = ProperModelBasedADPLearner(
            state_dim=4,  # [system_health, coverage_quality, operational_count, healthy_spares]
            action_dim=3,  # [REPLACE_SATELLITE, ACTIVATE_BACKUP, INCREASE_SIGNAL_POWER]
            learning_rate=0.001,
            gamma=0.99,
            planning_horizon=5,
            num_rollouts=10,
            planning_steps=10,
            batch_size=32
        )
        
        # Natural degradation rates (moved from transition model to constellation)
        self.health_decay_rate = 0.999
        self.signal_decay_rate = 0.999
        
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
        """Update system-level health and coverage metrics"""
        operational_health = [sat.health for sat in self.satellites if sat.status == "operational"]
        operational_signals = [sat.signal_quality for sat in self.satellites if sat.status == "operational"]
        
        # System health is average of operational satellite health
        self.system_health = np.mean(operational_health) if operational_health else 0
        
        # Coverage quality considers both health and signal quality
        if operational_health:
            # Satellites with health > 0.5 and signal > 0.5 contribute to coverage
            effective_sats = sum(1 for i in range(len(operational_health)) 
                               if operational_health[i] > 0.5 and operational_signals[i] > 0.5)
            self.coverage_quality = effective_sats / self.operational_count
        else:
            self.coverage_quality = 0

    def calculate_reward(self, old_state: Dict, new_state: Dict, action: str) -> float:
        """Calculate reward based on state transition and action taken"""
        reward = 0
        
        # Coverage maintenance reward (most important)
        reward += new_state['coverage_quality'] * 15
        
        # System health reward
        reward += new_state['system_health'] * 10
        
        # Action costs
        action_costs = {
            'REPLACE_SATELLITE': 3,
            'ACTIVATE_BACKUP': 1,
            'INCREASE_SIGNAL_POWER': 2
        }
        reward -= action_costs.get(action, 0)
        
        # Strong penalty for losing operational satellites
        if new_state['operational_count'] < old_state['operational_count']:
            reward -= 10 * (old_state['operational_count'] - new_state['operational_count'])
        
        # Bonus for maintaining high coverage
        if new_state['coverage_quality'] > 0.8:
            reward += 5
        
        # Severe penalty for constellation failure
        if new_state['operational_count'] == 0:
            reward -= 100
        
        return reward

    def step(self, attack_prob: float = 0.3):
        """Execute one simulation step with model-based learning"""
        old_state = self.get_state()
        
        # Get action using model-based planning
        constellation_state = ConstellationState(old_state)
        action = self.adp_learner.select_action(constellation_state, epsilon=0.1)
        
        # Execute action in real environment
        self._execute_action(action)
        
        # Progress time and apply natural degradation
        self.time_step += 1
        for sat in self.satellites:
            if sat.status != "decommissioned":
                sat.health *= self.health_decay_rate
                sat.signal_quality *= self.signal_decay_rate
        
        # Apply attack based on probability
        if random.random() < attack_prob:
            attack_type = random.choice(["jamming", "spoofing"])
            self.apply_attack(attack_type)
        
        # Update constellation metrics
        self._update_constellation_metrics()
        new_state = self.get_state()
        
        # Calculate reward and update learner
        reward = self.calculate_reward(old_state, new_state, action)
        
        # Check if episode should terminate
        done = new_state['operational_count'] == 0
        
        # Update learner with real experience
        self.adp_learner.update(
            ConstellationState(old_state),
            action,
            reward,
            ConstellationState(new_state),
            done
        )
        
        self.episode_rewards.append(reward)
        return new_state, done
    
    def _execute_action(self, action: str):
        """Execute the selected action"""
        if action == 'REPLACE_SATELLITE':
            # Find the least healthy operational satellite and replace it
            operational_sats = [sat for sat in self.satellites if sat.status == "operational"]
            if operational_sats:
                damaged_sat = min(operational_sats, key=lambda x: x.health)
                if damaged_sat.health < 0.8:  # Only replace if significantly damaged
                    self._replace_satellite(damaged_sat)
            
        elif action == 'ACTIVATE_BACKUP':
            # Activate the healthiest spare satellite
            spare_sats = [sat for sat in self.satellites if sat.status == "spare"]
            if spare_sats and len([s for s in self.satellites if s.status == "operational"]) < self.operational_count:
                best_spare = max(spare_sats, key=lambda x: x.health)
                best_spare.status = "operational"
        
        elif action == 'INCREASE_SIGNAL_POWER':
            # Boost signal quality of all operational satellites
            operational_sats = [sat for sat in self.satellites if sat.status == "operational"]
            for sat in operational_sats:
                sat.signal_quality = min(1.0, sat.signal_quality * 1.15)
                # Signal power increase has small health cost
                sat.health *= 0.98

    def get_training_statistics(self) -> Dict:
        """Get comprehensive training statistics"""
        stats = self.adp_learner.get_training_stats()
        
        # Add constellation-specific stats
        operational_count = sum(1 for sat in self.satellites if sat.status == "operational")
        spare_count = sum(1 for sat in self.satellites if sat.status == "spare")
        decommissioned_count = sum(1 for sat in self.satellites if sat.status == "decommissioned")
        
        stats.update({
            'constellation_health': self.system_health,
            'coverage_quality': self.coverage_quality,
            'operational_satellites': operational_count,
            'spare_satellites': spare_count,
            'decommissioned_satellites': decommissioned_count,
            'average_episode_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'time_step': self.time_step
        })
        
        return stats

    def reset(self):
        """Reset constellation to initial state"""
        self.satellites.clear()
        self._initialize_constellation()
        self.system_health = 1.0
        self.coverage_quality = 1.0
        self.time_step = 0
        self.episode_rewards.clear()

    def visualize(self):
        """Visualize current constellation state"""
        plt.figure(figsize=(12, 8))
        colors = {'operational': 'green', 'spare': 'blue', 'decommissioned': 'red'}
        
        for sat in self.satellites:
            plt.scatter(sat.plane_id * self.plane_spacing, sat.health * 100,
                       c=colors[sat.status], label=sat.status, alpha=0.7, s=100)
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title(f"Galileo Constellation Status (t={self.time_step})")
        plt.xlabel("Plane Position (degrees)")
        plt.ylabel("Satellite Health (%)")
        plt.grid(True, alpha=0.3)
        
        # Add text with current metrics
        plt.text(0.02, 0.98, f"System Health: {self.system_health:.2f}\nCoverage Quality: {self.coverage_quality:.2f}", 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def visualize_2d_polar(self):
        """Visualize constellation in 2D polar plot"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        colors = {'operational': 'green', 'spare': 'blue', 'decommissioned': 'red'}
        markers = {'operational': 'o', 'spare': 's', 'decommissioned': 'x'}
        sizes = {'operational': 120, 'spare': 100, 'decommissioned': 80}
        
        for sat in self.satellites:
            # Calculate position with some spread within plane
            base_angle = sat.plane_id * self.plane_spacing
            spread = 30  # degrees of spread within plane
            angle_offset = (hash(sat.id) % 100 - 50) * spread / 100
            theta = np.radians(base_angle + angle_offset)
            
            # Radius based on health (closer to center = healthier)
            r = 1.0 - (sat.health * 0.3)  # Invert so healthy sats are closer to edge
            
            ax.scatter(theta, r, c=colors[sat.status], marker=markers[sat.status], 
                      s=sizes[sat.status], alpha=0.8, label=sat.status)
        
        # Clean up legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        ax.set_title(f"Galileo Constellation Polar View (t={self.time_step})")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def visualize_3d(self):
        """Visualize constellation in 3D with complete orbital paths"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {'operational': 'green', 'spare': 'blue', 'decommissioned': 'red'}
        markers = {'operational': 'o', 'spare': 's', 'decommissioned': 'x'}
        sizes = {'operational': 100, 'spare': 80, 'decommissioned': 60}
        
        # Earth radius (scaled)
        earth_radius = 1.0
        orbit_radius = 2.0  # Scaled orbit radius
        
        # Draw Earth
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.2)
        
        # Draw orbital planes and satellites
        for plane in range(self.num_planes):
            # Draw complete orbital path
            theta = np.linspace(0, 2*np.pi, 100)
            phi = np.radians(self.inclination)
            
            # Calculate orbital path in inclined plane
            x_orbit = orbit_radius * np.cos(theta)
            y_orbit = orbit_radius * np.sin(theta) * np.cos(phi)
            z_orbit = orbit_radius * np.sin(theta) * np.sin(phi)
            
            # Rotate orbital plane based on plane spacing
            rotation_rad = np.radians(plane * self.plane_spacing)
            x_rotated = x_orbit * np.cos(rotation_rad) - y_orbit * np.sin(rotation_rad)
            y_rotated = x_orbit * np.sin(rotation_rad) + y_orbit * np.cos(rotation_rad)
            
            # Plot orbital path
            ax.plot(x_rotated, y_rotated, z_orbit, 'gray', alpha=0.3, linewidth=1)
            
            # Plot satellites in this plane
            plane_sats = [sat for sat in self.satellites if sat.plane_id == plane]
            for idx, sat in enumerate(plane_sats):
                # Distribute satellites evenly in orbital plane
                sat_theta = 2 * np.pi * idx / max(len(plane_sats), 1)
                
                # Position in inclined plane
                x = orbit_radius * np.cos(sat_theta)
                y = orbit_radius * np.sin(sat_theta) * np.cos(phi)
                z = orbit_radius * np.sin(sat_theta) * np.sin(phi)
                
                # Rotate for plane spacing
                x_final = x * np.cos(rotation_rad) - y * np.sin(rotation_rad)
                y_final = x * np.sin(rotation_rad) + y * np.cos(rotation_rad)
                z_final = z
                
                # Plot satellite
                ax.scatter(x_final, y_final, z_final, 
                          c=colors[sat.status], 
                          marker=markers[sat.status],
                          s=sizes[sat.status], 
                          alpha=0.8, 
                          label=sat.status)
        
        # Clean up legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        ax.set_title(f"Galileo Constellation 3D View (t={self.time_step})")
        ax.set_xlabel('X (Earth Radii)')
        ax.set_ylabel('Y (Earth Radii)')
        ax.set_zlabel('Z (Earth Radii)')
        
        # Set equal aspect ratio
        max_range = orbit_radius * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()


# Example usage and training
if __name__ == "__main__":
    # Create constellation
    constellation = GalileoConstellation(operational_count=6, spare_count=2)
    
    # Training parameters
    num_episodes = 20
    steps_per_episode = 50
    
    all_episode_rewards = []
    episode_lengths = []
    
    print("Starting Galileo Constellation Training...")
    print("=" * 50)
    
    for episode in range(num_episodes):
        # Reset constellation for new episode
        constellation.reset()
        episode_rewards = []
        
        for step in range(steps_per_episode):
            # Execute one step
            state, done = constellation.step(attack_prob=0.3)
            
            # Check for early termination
            if done:
                print(f"Episode {episode+1} terminated early at step {step} - Constellation failed")
                break
            
            # Print progress every 10 steps
            if step % 10 == 0 and step > 0:
                stats = constellation.get_training_statistics()
                print(f"Ep {episode+1:2d}, Step {step:2d} | "
                      f"Health: {stats['constellation_health']:.2f} | "
                      f"Coverage: {stats['coverage_quality']:.2f} | "
                      f"Op Sats: {stats['operational_satellites']} | "
                      f"Avg Reward: {stats['average_episode_reward']:.2f}")
        
        # Episode summary
        avg_reward = np.mean(constellation.episode_rewards) if constellation.episode_rewards else 0
        all_episode_rewards.append(avg_reward)
        episode_lengths.append(len(constellation.episode_rewards))
        
        final_stats = constellation.get_training_statistics()
        print(f"\nEpisode {episode+1} Summary:")
        print(f"  Length: {len(constellation.episode_rewards)} steps")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Final Health: {final_stats['constellation_health']:.2f}")
        print(f"  Final Coverage: {final_stats['coverage_quality']:.2f}")
        print(f"  Model Loss: {final_stats['avg_model_loss']:.4f}")
        print(f"  Value Loss: {final_stats['avg_value_loss']:.4f}")
        print("-" * 40)
        
        # Visualize every 10 episodes
        if (episode + 1) % 10 == 0:
            constellation.visualize()
    
    print(f"\nTraining Complete!")
    print(f"Average Episode Reward: {np.mean(all_episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    
    # Final visualization
    constellation.visualize_3d()

        # --------------------------
    # Learning Performance Plots
    # --------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards
    axes[0, 0].plot(all_episode_rewards, label="Avg Episode Reward")
    axes[0, 0].set_title("Episode Rewards Over Time")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(episode_lengths, label="Episode Length")
    axes[0, 1].set_title("Episode Lengths Over Time")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps Survived")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Model loss history
    axes[1, 0].plot(constellation.adp_learner.model_loss_history, label="Model Loss", alpha=0.7)
    axes[1, 0].set_title("Transition Model Loss")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Value function loss history
    axes[1, 1].plot(constellation.adp_learner.value_loss_history, label="Value Loss", alpha=0.7, color="orange")
    axes[1, 1].set_title("Value Function Loss")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle("Learning Performance of Model-Based ADP Learner", fontsize=16)
    plt.tight_layout()
    plt.show()
