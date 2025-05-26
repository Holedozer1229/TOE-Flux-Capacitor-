import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import serial
import time
from scipy.integrate import odeint
from scipy.linalg import expm
from scipy.io import wavfile
from ...utils.math_utils import compute_j6_potential
from ...constants.config import Constants
from ...audio.oam_audio_modulator import OAMAudioModulator
from ...physics.qubits.qubit_fabric import SpinNetwork

logger = logging.getLogger(__name__)

class CTCLatticeTOEModel:
    """CTC lattice model integrating TOESimulator's spacetime fabric and J^4 coupling."""
    
    def __init__(self, n_qubits=6, n_layers=4, dev_type="default.qubit", n_points=100, 
                 resolution=20, lambda_=2.72, kappa=0.1, charge_density=1e-12, serial_port=None):
        """Initialize the CTC lattice model with TOESimulator properties."""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev_type, wires=n_qubits)
        self.params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        self.oam_modulator = OAMAudioModulator()
        self.spin_network = SpinNetwork(n_qubits)
        self.n_points = n_points
        self.resolution = resolution
        self.lambda_ = lambda_
        self.kappa = kappa
        self.charge_density = charge_density
        self.dx = 4.0 / (resolution - 1)
        self.dt = self.dx / (2 * 3e8)
        self.time = 0.0
        self.c = 3e8
        self.eps_0 = 8.854187817e-12
        self.hbar = 1.0545718e-34
        self.nodes = 16
        self.arduino = None
        if serial_port:
            try:
                self.arduino = serial.Serial(serial_port, 115200, timeout=1)
                time.sleep(2)
                logger.info(f"Connected to Arduino on {serial_port}")
            except serial.SerialException as e:
                logger.error(f"Error connecting to Arduino: {e}")
        self.tetrahedral_coordinates = self._generate_tetrahedral_coordinates()
        self.H_tetrahedral = self._build_tetrahedral_hamiltonian()
        self.quantum_state = np.ones(resolution, dtype=complex) / np.sqrt(resolution)
        self.em = {'J4': np.zeros(resolution, dtype=np.float32)}
        self.all_flux_signals = []
        logger.info("Initialized CTCLatticeTOEModel with %d qubits, %d layers", n_qubits, n_layers)
    
    def _generate_tetrahedral_coordinates(self):
        """Generate tetrahedral coordinates for the lattice."""
        coords = np.zeros((self.resolution, 4))
        t = np.linspace(0, 2 * np.pi, self.resolution)
        coords[:, 0] = np.cos(t) * np.sin(t)
        coords[:, 1] = np.sin(t) * np.sin(t)
        coords[:, 2] = np.cos(t)
        coords[:, 3] = t / (2 * np.pi)
        return coords
    
    def _build_tetrahedral_hamiltonian(self):
        """Build tetrahedral Hamiltonian."""
        H = np.zeros((self.resolution, self.resolution), dtype=complex)
        for i in range(self.resolution):
            for j in range(i + 1, self.resolution):
                Δx = self.tetrahedral_coordinates[j] - self.tetrahedral_coordinates[i]
                distance = np.linalg.norm(Δx)
                if distance > 0:
                    H[i, j] = H[j, i] = 1j * 10 / (distance + 1e-10)
        np.fill_diagonal(H, -1j * np.linalg.norm(self.tetrahedral_coordinates[:, :3], axis=1))
        return H
    
    def ctc_m_shift(self, u, v, phi, t, oam, j4):
        """CTC-modulated m_shift with J^4 coupling."""
        ctc_term = Constants.CTC_PARAMS['kappa_ctc'] * np.sin(phi * t / Constants.CTC_PARAMS['tau'])
        j4_term = 1e-16 * j4  # TOESimulator j4_coupling_factor
        m_shift = 2.72 * (1 + 0.01 * ctc_term * (1 + 0.002 * oam + 0.001 * j4_term) * (1 + 0.001 * ctc_term**2))
        return m_shift
    
    def j6_cost(self, phi, j4, psi, ricci_scalar, graviton_field, oam, body_positions=None, body_masses=None):
        """Compute J^6 potential cost with OAM and J^4 modulation."""
        V_j6, _ = compute_j6_potential(
            np.array([phi]), np.array([j4]), psi, ricci_scalar, graviton_field,
            body_positions=body_positions, body_masses=body_masses
        )
        cost = np.mean(V_j6) * (1 + 0.002 * oam + 1e-16 * np.mean(self.em['J4']))
        return cost
    
    def ctc_non_local_gate(self, wires, phi, t, oam, j4):
        """Higher-order CTC gates with J^4 coupling."""
        ctc_term = Constants.CTC_PARAMS['kappa_ctc'] * np.sin(phi * t / Constants.CTC_PARAMS['tau'])
        j4_term = 1e-16 * j4
        for w in wires:
            qml.RZ(ctc_term * (1 + 0.002 * oam + 0.001 * j4_term) * (1 + 0.001 * ctc_term**2 + 0.0001 * ctc_term**4), wires=w)
        if len(wires) > 1:
            for i in range(len(wires) - 1):
                qml.ControlledPhaseShift(ctc_term * 0.2 * (1 + 0.002 * oam + 0.001 * j4_term), wires=[wires[i], wires[i + 1]])
        if len(wires) >= 3:
            qml.MultiControlledX(wires=wires[:3], control_values="111")
            qml.ctrl(qml.RZ(ctc_term * 0.1 * (1 + 0.002 * oam**2)), control=wires[:2])(wires[2])
            if len(wires) >= 4:
                qml.ctrl(qml.RY(ctc_term * 0.05 * (1 + 0.002 * oam + 0.001 * j4_term)), control=wires[:3])(wires[3])
    
    @qml.qnode(device=None)
    def circuit(self, inputs, weights, t=0.0, oam=0.0, j4=0.0):
        """Quantum circuit for lattice entanglement."""
        for i in range(self.n_qubits):
            qml.AngleEmbedding(inputs, wires=i, rotation="X")
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            self.ctc_non_local_gate(wires=list(range(self.n_qubits)), phi=inputs[0], t=t, oam=oam, j4=j4)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def mobius_spiral_trajectory(self, t, r, n, phi, time, oam, j4):
        """Möbius spiral with CTC and J^4 modulation."""
        theta = t * np.pi
        phi_spiral = n * theta / 2
        m_shift_val = self.ctc_m_shift(t, phi_spiral, phi, time, oam, j4)
        x = r * (np.cos(theta) * np.sin(phi_spiral)) * m_shift_val
        y = r * (np.sin(theta) * np.sin(phi_spiral)) * m_shift_val
        z = r * np.cos(phi_spiral) * m_shift_val
        return x, y, z
    
    def visualize_tetrahedron(self, a, b, c, phi, t, oam, j4):
        """Visualize tetrahedral lattice with CTC and J^4 effects."""
        u = np.linspace(-np.pi, np.pi, self.n_points)
        v = np.linspace(-np.pi / 2, np.pi / 2, self.n_points)
        u, v = np.meshgrid(u, v)
        
        def compute_face(x_sign, y_sign, z_sign):
            m_shift_val = np.array([[self.ctc_m_shift(ui, vi, phi, t, oam, j4) for vi in v[0]] for ui in u[:, 0]])
            x = x_sign * a * np.cosh(u) * np.cos(v) * m_shift_val
            y = y_sign * b * np.cosh(u) * np.sin(v) * m_shift_val
            z = z_sign * c * np.sinh(u) * m_shift_val
            return x, y, z
        
        faces = [
            compute_face(1, 1, 1),
            compute_face(-1, -1, 1),
            compute_face(1, -1, -1),
            compute_face(-1, 1, -1)
        ]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['b', 'g', 'r', 'y']
        for (x, y, z), color in zip(faces, colors):
            ax.plot_surface(x, y, z, color=color, alpha=0.5)
        
        t_spiral = np.linspace(0, 2 * np.pi, self.n_points)
        x, y, z = self.mobius_spiral_trajectory(t_spiral, 3, 2, phi, t, oam, j4)
        ax.plot(x, y, z, color='k', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        timestamp = np.datetime64('now').astype(str).replace(':', '')
        output_path = f"results/tetrahedron_ctc_toe_{timestamp}.png"
        plt.savefig(output_path)
        plt.close()
        logger.info("CTC lattice visualization saved to %s", output_path)
    
    def generate_flux_signal(self, duration=1.0, sample_rate=22050):
        """Generate flux signal with CTC and J^4 effects."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        flux_signal = np.zeros_like(t, dtype=np.float32)
        base_signal = np.sin(2 * np.pi * 0.00083 * t) * 0.5
        total_amplitude_mod = 0.0
        total_freq_mod = 0.0
        for i in range(self.resolution):
            j4_effect = 1e-16 * self.em['J4'][i]
            total_amplitude_mod += j4_effect * 0.01
            total_freq_mod += j4_effect * 10
        amplitude_mod = 1 + total_amplitude_mod / self.resolution
        freq_mod = 0.00083 + total_freq_mod / self.resolution
        flux_signal = amplitude_mod * np.sin(2 * np.pi * freq_mod * t) + base_signal
        max_abs = np.max(np.abs(flux_signal))
        if max_abs > 0:
            flux_signal /= max_abs
        self.all_flux_signals.append(flux_signal)
        return flux_signal
    
    def activate_flux_capacitor(self, signal, sample_rate=22050):
        """Activate flux capacitor with Arduino integration."""
        if self.arduino:
            try:
                scaled_signal = ((signal + 1) * 127.5).astype(np.uint8)
                for value in scaled_signal:
                    self.arduino.write(bytes([value]))
                    time.sleep(1 / sample_rate)
            except serial.SerialException as e:
                logger.error(f"Error communicating with Arduino: {e}")
    
    def save_combined_wav(self, sample_rate=22050):
        """Save combined flux signal to WAV file."""
        if not self.all_flux_signals:
            logger.warning("No flux signals to save.")
            return
        combined_signal = np.concatenate(self.all_flux_signals)
        signal_int16 = np.int16(combined_signal * 32767)
        timestamp = np.datetime64('now').astype(str).replace(':', '')
        output_path = f'results/flux_signal_ctc_toe_{timestamp}.wav'
        wavfile.write(output_path, sample_rate, signal_int16)
        logger.info(f"Saved flux signal to {output_path}")
        self.all_flux_signals = []
    
    def train(self, trajectory_data, body_masses, epochs=100, learning_rate=0.1):
        """Train the quantum circuit for lattice entanglement."""
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        weights = self.params.copy()
        costs = []
        
        for epoch in range(epochs):
            idx = epoch % len(trajectory_data)
            positions = trajectory_data[idx]
            velocities = np.diff(positions, axis=0, prepend=positions[0:1])
            
            # Compute OAM
            oam = self.oam_modulator.compute_oam(positions, velocities, body_masses)
            
            # Mock J^4 effect
            self.em['J4'] = np.random.normal(0, 1e-5, self.resolution)
            
            # Prepare inputs
            inputs = positions[:, :2].flatten()
            if len(inputs) < self.n_qubits:
                inputs = np.pad(inputs, (0, self.n_qubits - len(inputs)))
            
            # Compute J^6 cost
            phi = np.mean(inputs)
            j4 = np.mean(self.em['J4'])
            psi = np.ones(self.n_qubits) / np.sqrt(self.n_qubits)
            ricci_scalar = np.ones(self.n_qubits)
            graviton_field = np.zeros((self.n_qubits, 6, 6))
            j6_cost = self.j6_cost(phi, j4, psi, ricci_scalar, graviton_field, oam)
            
            # Optimize entanglement
            def cost_fn(w):
                expectations = self.circuit(inputs, w, t=epoch * 0.01, oam=oam, j4=j4)
                return -np.abs(np.mean(expectations))
            
            weights, circuit_cost = opt.step_and_cost(cost_fn, weights)
            costs.append(j6_cost + circuit_cost)
            
            # Visualize and generate audio every 10 epochs
            if epoch % 10 == 0:
                self.visualize_tetrahedron(1, 2, 3, phi, epoch * 0.01, oam, j4)
                flux_signal = self.generate_flux_signal()
                self.activate_flux_capacitor(flux_signal)
                logger.info("Epoch %d: Total Cost = %.4f", epoch, costs[-1])
        
        self.params = weights
        self.save_combined_wav()
        return costs
    
    def predict(self, inputs, t=0.0, oam=0.0, j4=0.0):
        """Predict entanglement measure."""
        expectations = self.circuit(inputs, self.params, t, oam, j4)
        entanglement_measure = np.abs(np.mean(expectations))
        logger.debug("Predicted entanglement measure: %.4f", entanglement_measure)
        return entanglement_measure
    
    def close(self):
        """Close Arduino connection."""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            logger.info("Serial connection closed.")

def load_trajectory_data():
    """Load three-body trajectory data."""
    import glob
    traj_files = glob.glob("results/trajectories_*.npy")
    if not traj_files:
        raise FileNotFoundError("No trajectory files found")
    traj_file = max(traj_files, key=lambda x: x)
    return np.load(traj_file)

if __name__ == "__main__":
    model = CTCLatticeTOEModel(n_qubits=6, n_layers=4, serial_port='COM3')  # Replace with your Arduino port
    try:
        trajectory_data = load_trajectory_data()
        body_masses = Constants.THREE_BODY_MASSES
        costs = model.train(trajectory_data, body_masses, epochs=50)
        inputs = trajectory_data[0][:, :2].flatten()
        entanglement = model.predict(inputs)
        print(f"Predicted entanglement measure: {entanglement}")
    finally:
        model.close()
