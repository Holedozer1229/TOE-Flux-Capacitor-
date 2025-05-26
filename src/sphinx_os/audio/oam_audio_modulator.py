import numpy as np
import logging
import pennylane as qml
from ...utils.math_utils import compute_j6_potential
from ...constants.config import Constants
from ...harmonic_generator import HarmonicGenerator

logger = logging.getLogger(__name__)

class OAMAudioModulator:
    """Modulates audio output with orbital angular momentum from J^6 coupling."""
    
    def __init__(self, sample_rate=Constants.SAMPLE_RATE, kappa_j6=Constants.KAPPA_J6):
        """Initialize the OAM audio modulator."""
        self.sample_rate = sample_rate
        self.kappa_j6 = kappa_j6
        self.harmonic_gen = HarmonicGenerator(sample_rate, kappa_j6)
        self.n_qubits = 4  # Reduced for simulation
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.params = np.random.uniform(0, 2 * np.pi, (self.n_qubits,))  # Phase modulation params
        logger.info("Initialized OAMAudioModulator with sample_rate=%d", sample_rate)
    
    def compute_oam(self, positions, velocities, masses):
        """Compute orbital angular momentum for three-body system."""
        oam = np.zeros(3)
        for pos, vel, mass in zip(positions, velocities, masses):
            oam += mass * np.cross(pos, vel)
        oam_magnitude = np.linalg.norm(oam)
        logger.debug("Computed OAM magnitude: %.4f", oam_magnitude)
        return oam_magnitude
    
    def j6_oam_modulation(self, phi, j4, psi, ricci_scalar, graviton_field, oam, body_positions=None, body_masses=None):
        """Modulate J^6 potential with OAM."""
        V_j6, _ = compute_j6_potential(
            np.array([phi]), np.array([j4]), psi, ricci_scalar, graviton_field,
            body_positions=body_positions, body_masses=body_masses
        )
        # OAM modulation (phase shift)
        modulation = np.sin(V_j6 + 0.01 * oam)  # OAM scales phase
        return modulation
    
    @qml.qnode(device=None)
    def phase_circuit(self, inputs, weights):
        """Quantum circuit to optimize phase modulation."""
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(inputs[i % len(inputs)], wires=i)
            qml.RX(weights[i], wires=i)
        return qml.expval(qml.PauliZ(0))
    
    def optimize_phase(self, trajectory_data, body_masses, epochs=10, learning_rate=0.1):
        """Optimize phase modulation using PennyLane."""
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        weights = self.params.copy()
        
        for epoch in range(epochs):
            # Extract positions and velocities
            positions = trajectory_data[epoch % len(trajectory_data)]
            velocities = np.diff(positions, axis=0, prepend=positions[0:1])
            
            # Compute OAM
            oam = self.compute_oam(positions, velocities, body_masses)
            
            # Mock inputs for J^6
            inputs = positions[:, :2].flatten()  # 2D projection
            if len(inputs) < self.n_qubits:
                inputs = np.pad(inputs, (0, self.n_qubits - len(inputs)))
            
            # Optimize phase
            weights, cost = opt.step_and_cost(lambda w: self.phase_circuit(inputs, w), weights)
            
            if epoch % 2 == 0:
                logger.info("Epoch %d: Phase Cost = %.4f", epoch, cost)
        
        self.params = weights
    
    def generate_oam_audio(self, trajectory_data, body_masses, phi, j4, psi, ricci_scalar, graviton_field, duration=10.0):
        """Generate audio with OAM-modulated J^6 coupling."""
        num_samples = int(self.sample_rate * duration)
        audio_output = np.zeros(num_samples)
        t = np.linspace(0, duration, num_samples)
        
        # Extract positions and velocities
        positions = trajectory_data[:len(t)]
        velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        
        for i in range(num_samples):
            # Compute OAM
            oam = self.compute_oam(positions[i % len(positions)], velocities[i % len(velocities)], body_masses)
            
            # Generate base harmonics
            harmonics = self.harmonic_gen.generate_harmonics(phi, j4, psi, ricci_scalar, graviton_field)
            
            # OAM modulation
            modulation = self.j6_oam_modulation(phi, j4, psi, ricci_scalar, graviton_field, oam)
            
            # Apply phase modulation
            audio_output[i] = harmonics * np.cos(2 * np.pi * modulation * t[i])
        
        # Normalize
        audio_output = np.clip(audio_output / (np.max(np.abs(audio_output)) + 1e-10), -1.0, 1.0)
        
        # Save output
        import os
        timestamp = np.datetime64('now').astype(str).replace(':', '')
        output_path = os.path.join("results", f"audio_output_oam_{timestamp}.npy")
        np.save(output_path, audio_output)
        logger.info("OAM-modulated audio saved to %s", output_path)
        
        return audio_output

def load_trajectory_data():
    """Load three-body trajectory data."""
    import glob
    traj_files = glob.glob("results/trajectories_*.npy")
    if not traj_files:
        raise FileNotFoundError("No trajectory files found")
    traj_file = max(traj_files, key=lambda x: x)
    return np.load(traj_file)

if __name__ == "__main__":
    from main import FluxCapacitor
    flux = FluxCapacitor()
    trajectory_data = load_trajectory_data()
    body_masses = Constants.THREE_BODY_MASSES
    modulator = OAMAudioModulator()
    modulator.optimize_phase(trajectory_data, body_masses, epochs=5)
    audio_output = modulator.generate_oam_audio(
        trajectory_data, body_masses, phi=1.0, j4=0.0, psi=np.ones(6)/np.sqrt(6),
        ricci_scalar=np.ones(6), graviton_field=np.zeros((6, 6, 6)), duration=2.0
    )
    print("OAM-modulated audio generated")
