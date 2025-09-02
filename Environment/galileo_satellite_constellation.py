import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from dataclasses import dataclass
from typing import List, Dict
import random

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

    def step(self, attack_prob: float = 0.3):
        """Progress constellation state by one time step"""
        self.time_step += 1
        
        # Natural degradation
        for sat in self.satellites:
            if sat.status != "decommissioned":
                sat.health *= 0.999
                sat.signal_quality *= 0.999
        
        # Random attack
        if random.random() < attack_prob:
            attack_type = random.choice(["jamming", "spoofing"])
            print(f"Attack at step {self.time_step}: {attack_type}")
            self.apply_attack(attack_type)
        
        self._update_constellation_metrics()
        return self.get_state()

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
    
    # Show initial constellation state (before any attacks)
    print("\nInitial Constellation State:")
    state = constellation.get_state()
    print(f"Operational satellites: {state['operational_count']}")
    print(f"Healthy spares: {state['healthy_spares']}")
    
    
    # Then run simulation if desired
    run_simulation = True  # Set to False to only see initial state
    if run_simulation:
        print("\nRunning simulation with attacks...")
        for _ in range(10):
            state = constellation.step(attack_prob=0.4)
            print(f"\nStep {constellation.time_step}")
            print(f"System Health: {state['system_health']:.2f}")
            print(f"Coverage Quality: {state['coverage_quality']:.2f}")
        
        # Show final state
        constellation.visualize_2d_polar()
        #constellation.visualize_3d()
