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
import sounddevice as sd
from ...utils.math_utils import compute_j6_potential
from ...constants.config import Constants
from ...audio.oam_audio_modulator import OAMAudioModulator
from ...physics.qubits.qubit_fabric import SpinNetwork

logger = logging.getLogger(__name__)

# CTC Constants
RS = 2.0
CONFIG = {
    "swarm_size": 5,
    "max_iterations": 200,
    "resolution": 20,
    "time_delay_steps": 3,
    "ctc_feedback_factor": 5.0,
    "entanglement_factor": 0.2,
    "charge": 1.0,
    "em_strength": 3.0,
    "nodes": 16
}

TARGET_PHYSICAL_STATE = int(time.time() * 1000)
START_TIME = time.perf_counter_ns() / 1e9
KNOWN_STATE = int(START_TIME * 1000) % 2**32

def repeating_curve(index):
    return 1 if index % 2 == 0 else 0

class CTCLatticeUnifiedModel:
    """CTC lattice model integrating UnifiedSpacetimeSimulator with higher-order CTC gates."""
    
    def __init__(self, n_qubits=6, n_layers=4, dev_type="default.qubit", n_points=100, 
                 resolution=CONFIG["resolution"], lambda_=1.0, kappa=0.1, charge_density=1e-12, serial_port=None):
        """Initialize the CTC lattice model with UnifiedSpacetimeSimulator properties."""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev_type, wires=n_qubits)
        self.params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        self.oam_modulator = OAMAudioModulator()
        self.spin_network = SpinNetwork(nodes=CONFIG["nodes"])
        self.n_points = n_points
        self.resolution = resolution
        self.lambda_ = lambda_
        self.kappa = kappa
        self.charge_density = charge_density
        self.dx = 4.0 / (resolution - 1)
        self.dt = 1e-3
        self.time = 0.0
        self.c = 3e8
        self.eps_0 = 8.854187817e-12
        self.hbar = 1.0545718e-34
        self.G = 6.67430e-11
        self.schumann_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]
        self.flux_freq = 0.00083
        self.schumann_amplitudes = [1.0, 0.5, 0.33, 0.25, 0.2]
        self.pythagorean_ratios = [1.0, 2.0, 3/2, 4/3]
        self.arduino = None
        if serial_port:
            try:
                self.arduino = serial.Serial(serial_port, 9600, timeout=1)
                time.sleep(2)
                logger.info(f"Connected to Arduino on {serial_port}")
            except serial.SerialException as e:
                logger.error(f"Error connecting to Arduino: {e}")
        self.tetrahedral_coordinates = self._generate_tetrahedral_coordinates()
        self.H_tetrahedral = self._build_tetrahedral_hamiltonian()
        self.quantum_state = np.ones(resolution, dtype=complex) / np.sqrt(resolution)
        self.em = {'J4': np.zeros(resolution, dtype=np.float32), 'A_mu': np.zeros((resolution, 4), dtype=np.float32), 
                   'F_munu': np.zeros((resolution, 4, 4), dtype=np.float32), 'J': np.zeros((resolution, 4), dtype=np.float32), 
                   'charge': CONFIG["charge"]}
        self.em['J'][:, 0] = self.charge_density * self.c
        self.em['J4'] = np.power(np.linalg.norm(self.em['J'], axis=1), 4)
        self.all_flux_signals = []
        self.iteration = 0
        self.ctc_influence = 0.0
        self.history = []
        self.fabric, self.edges = self.generate_spacetime_fabric()
        logger.info("Initialized CTCLatticeUnifiedModel with %d qubits, %d layers", n_qubits, n_layers)
    
    def generate_spacetime_fabric(self):
        scale = 2 / (3 * np.sqrt(2))
        vertices = [
            np.array([3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32)
        ]
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7), (7, 0)]
        spins = np.random.choice([0.5, 1.0], len(edges))
        fabric = np.array(vertices[:self.resolution], dtype=np.float32)
        return fabric, list(zip(edges[:self.resolution], spins))
    
    def _generate_tetrahedral_coordinates(self):
        coords = np.zeros((self.resolution, 4))
        t = np.linspace(0, 2 * np.pi, self.resolution)
        coords[:, 0] = np.cos(t) * np.sin(t)
        coords[:, 1] = np.sin(t) * np.sin(t)
        coords[:, 2] = np.cos(t)
        coords[:, 3] = t / (2 * np.pi)
        return coords
    
    def _build_tetrahedral_hamiltonian(self):
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
        ctc_term = Constants.CTC_PARAMS['kappa_ctc'] * np.sin(phi * t / Constants.CTC_PARAMS['tau'])
        j4_term = 1e-20 * j4  # UnifiedSpacetimeSimulator J^4 effect
        m_shift = 2.72 * (1 + 0.01 * ctc_term * (1 + 0.002 * oam + 0.001 * j4_term) * (1 + 0.001 * ctc_term**2 + 0.0001 * ctc_term**4))
        return m_shift
    
    def j6_cost(self, phi, j4, psi, ricci_scalar, graviton_field, oam, body_positions=None, body_masses=None):
        V_j6, _ = compute_j6_potential(
            np.array([phi]), np.array([j4]), psi, ricci_scalar, graviton_field,
            body_positions=body_positions, body_masses=body_masses
        )
        cost = np.mean(V_j6) * (1 + 0.002 * oam + 1e-20 * np.mean(self.em['J4']))
        return cost
    
    def ctc_non_local_gate(self, wires, phi, t, oam, j4):
        ctc_term = Constants.CTC_PARAMS['kappa_ctc'] * np.sin(phi * t / Constants.CTC_PARAMS['tau'])
        j4_term = 1e-20 * j4
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
        theta = t * np.pi
        phi_spiral = n * theta / 2
        m_shift_val = self.ctc_m_shift(t, phi_spiral, phi, time, oam, j4)
        x = r * (np.cos(theta) * np.sin(phi_spiral)) * m_shift_val
        y = r * (np.sin(theta) * np.sin(phi_spiral)) * m_shift_val
        z = r * np.cos(phi_spiral) * m_shift_val
        return x, y, z
    
    def visualize_tetrahedron(self, a, b, c, phi, t, oam, j4):
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
        output_path = f"results/tetrahedron_ctc_unified_{timestamp}.png"
        plt.savefig(output_path)
        plt.close()
        logger.info("CTC lattice visualization saved to %s", output_path)
    
    def generate_flux_signal(self, duration=10.0, sample_rate=44100, num_fourier_terms=10):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        flux_signal = np.zeros_like(t, dtype=np.float32)
        fourier_borel = np.zeros_like(t, dtype=np.float32)
        for n in range(1, num_fourier_terms + 1):
            k = 2 * n - 1
            fourier_borel += (1 / k) * np.sin(k * 2 * np.pi * self.flux_freq * t)
        fourier_borel *= (4 / np.pi)
        
        for i in range(self.resolution):
            A_mu_norm = np.linalg.norm(self.em['A_mu'][i])
            gw_plus = self.gw['plus'][i]
            gw_cross = self.gw['cross'][i]
            base_flux = A_mu_norm + gw_plus + gw_cross
            schumann_mod = sum(A * np.cos(2 * np.pi * f * t) for f, A in zip(self.schumann_freqs, self.schumann_amplitudes))
            flux_mod = np.cos(2 * np.pi * self.flux_freq * t)
            modulated_flux = base_flux * schumann_mod * flux_mod
            j4_effect = self.em['J4'][i] * 1e-20
            modulated_flux += j4_effect * np.sin(2 * np.pi * self.flux_freq * t)
            harmonic_scale = self.pythagorean_ratios[i % len(self.pythagorean_ratios)]
            scaled_flux = modulated_flux * harmonic_scale + fourier_borel
            if self.ctc_influence > 0:
                delay_samples = int(self.ctc_influence * sample_rate)
                if len(flux_signal) > delay_samples:
                    scaled_flux += 0.1 * flux_signal[:-delay_samples]
            flux_signal += np.power(np.abs(scaled_flux), 4) * np.sign(scaled_flux)
        
        flux_signal = np.clip(flux_signal, -1.0, 1.0)
        self.all_flux_signals.append(flux_signal)
        return flux_signal
    
    def activate_flux_capacitor(self, signal, sample_rate=44100):
        sd.play(signal, sample_rate)
        if self.arduino:
            try:
                scaled_signal = ((signal + 1) * 127.5).astype(np.uint8)
                start_time = time.perf_counter_ns() / 1e9
                for value in scaled_signal:
                    self.arduino.write(bytes([value]))
                    current_time = time.perf_counter_ns() / 1e9
                    if self.arduino.in_waiting > 0:
                        feedback = self.arduino.readline().decode().strip()
                        try:
                            hall_value = float(feedback)
                            self.em['J'][:, 0] += hall_value * 1e-6
                            self.em['J4'] = np.power(np.linalg.norm(self.em['J'], axis=1), 4)
                        except ValueError:
                            pass
                    time.sleep(1 / sample_rate)
                state = int(np.sum(self.bit_states * (2 ** np.arange(self.resolution))))
                fitness, delta_t, ctc_influence = self.compute_fitness(state, start_time)
                logger.info(f"Iteration {self.iteration}, Time {int(current_time * 1e9)}: "
                            f"Bit States = {self.bit_states.tolist()}, Entanglement = {self.temporal_entanglement[0]:.4f}, "
                            f"State = {state}, Fitness = {fitness:.2f}, DeltaT = {delta_t:.6f}, CTC Influence = {ctc_influence:.4f}")
            except serial.SerialException as e:
                logger.error(f"Error communicating with Arduino: {e}")
        sd.wait()
    
    def save_combined_wav(self, sample_rate=44100):
        if not self.all_flux_signals:
            logger.warning("No flux signals to save.")
            return
        combined_signal = np.concatenate(self.all_flux_signals)
        signal_int16 = np.int16(combined_signal * 32767)
        timestamp = np.datetime64('now').astype(str).replace(':', '')
        output_path = f'results/flux_signal_ctc_unified_{timestamp}.wav'
        wavfile.write(output_path, sample_rate, signal_int16)
        logger.info(f"Saved flux signal to {output_path}")
        self.all_flux_signals = []
    
    def compute_fitness(self, state, temporal_pos):
        current_time = time.perf_counter_ns() / 1e9
        delta_time = current_time - temporal_pos
        base_fitness = abs(state - KNOWN_STATE)
        ctc_influence = 0
        if self.iteration >= CONFIG["time_delay_steps"] and len(self.history) >= CONFIG["time_delay_steps"]:
            past_states = [h[1] for h in self.history[-CONFIG["time_delay_steps"]:]]
            ctc_influence = np.mean([s[0] for s in past_states]) * CONFIG["ctc_feedback_factor"]
            self.ctc_influence = 1.6667 if self.iteration % 2 == 0 else 3.3333
        fitness = base_fitness + ctc_influence
        return fitness, delta_time, ctc_influence
    
    def quantum_walk(self, iteration, current_time):
        A_mu = self.compute_vector_potential(iteration)
        self.em['A_mu'] = A_mu
        prob = np.abs(self.quantum_state)**2
        adj_matrix = self.spin_network.get_adjacency_matrix()
        self.spin_network.evolve(adj_matrix, 2 * np.pi / self.resolution)
        for idx in range(self.resolution):
            expected_state = repeating_curve(idx + iteration)
            self.bit_states[idx] = expected_state
            window = prob[max(0, idx - CONFIG["time_delay_steps"]):idx + 1]
            self.temporal_entanglement[idx] = CONFIG["entanglement_factor"] * np.mean(window) if window.size > 0 else 0
            em_perturbation = A_mu[idx, 0] * CONFIG["em_strength"]
            if np.random.random() < abs(em_perturbation) * self.temporal_entanglement[idx]:
                self.bit_states[idx] = 1 - self.bit_states[idx]
        self.quantum_state = expm(-1j * self.H_tetrahedral * (2 * np.pi / self.resolution)) @ self.quantum_state
        self.history.append((int(current_time * 1e9), self.bit_states.copy()))
    
    def compute_vector_potential(self, iteration):
        A = np.zeros((self.resolution, 4))
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        theta = np.arctan2(self.fabric[:, 1], self.fabric[:, 0])
        load_factor = (time.perf_counter_ns() / 1e9 - START_TIME) / 5
        A[:, 0] = CONFIG["charge"] / (4 * np.pi * (r + 1e-8)) * (1 + np.sin(iteration * 0.2) * load_factor)
        A[:, 3] = CONFIG["em_strength"] * r * np.sin(theta) * (1 + load_factor)
        return A
    
    def train(self, trajectory_data, body_masses, epochs=100, learning_rate=0.1):
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        weights = self.params.copy()
        costs = []
        
        for epoch in range(epochs):
            idx = epoch % len(trajectory_data)
            positions = trajectory_data[idx]
            velocities = np.diff(positions, axis=0, prepend=positions[0:1])
            
            # Compute OAM
            oam = self.oam_modulator.compute_oam(positions, velocities, body_masses)
            
            # Update J^4 effect
            self.em['J4'] = np.power(np.linalg.norm(self.em['J'], axis=1), 4)
            
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
            
            # Evolve system and generate audio
            self.quantum_walk(epoch, time.perf_counter_ns() / 1e9)
            if epoch % 10 == 0:
                self.visualize_tetrahedron(1, 2, 3, phi, epoch * 0.01, oam, j4)
                flux_signal = self.generate_flux_signal()
                self.activate_flux_capacitor(flux_signal)
                logger.info("Epoch %d: Total Cost = %.4f", epoch, costs[-1])
        
        self.params = weights
        self.save_combined_wav()
        return costs
    
    def predict(self, inputs, t=0.0, oam=0.0, j4=0.0):
        expectations = self.circuit(inputs, self.params, t, oam, j4)
        entanglement_measure = np.abs(np.mean(expectations))
        logger.debug("Predicted entanglement measure: %.4f", entanglement_measure)
        return entanglement_measure
    
    def close(self):
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            logger.info("Serial connection closed.")

def load_trajectory_data():
    import glob
    traj_files = glob.glob("results/trajectories_*.npy")
    if not traj_files:
        raise FileNotFoundError("No trajectory files found")
    traj_file = max(traj_files, key=lambda x: x)
    return np.load(traj_file)

if __name__ == "__main__":
    model = CTCLatticeUnifiedModel(n_qubits=6, n_layers=4, serial_port='COM3')
    try:
        trajectory_data = load_trajectory_data()
        body_masses = Constants.THREE_BODY_MASSES
        costs = model.train(trajectory_data, body_masses, epochs=50)
        inputs = trajectory_data[0][:, :2].flatten()
        entanglement = model.predict(inputs)
        print(f"Predicted entanglement measure: {entanglement}")
    finally:
        model.close()
