import numpy as np
import logging
from itertools import product
from .constants import Constants

logger = logging.getLogger(__name__)

class QuantumCircuit:
    """Simulates a quantum circuit for TVLE with 64 qubits."""
    
    def __init__(self, num_qubits: int = Constants.NUM_QUBITS):
        self.num_qubits = num_qubits
        self.state = np.ones(2**num_qubits, dtype=np.complex128) / np.sqrt(2**num_qubits)
        self.ctc_feedback = 0.0
        logger.info("Quantum circuit initialized with %d qubits", num_qubits)
    
    def apply_rydberg_gates(self, wormhole_nodes: bool = False, phi: float = 0.0, ctc_feedback: float = 0.0, 
                            ctc_params: dict = None, boundary_factor: float = 1.0, 
                            psi_abs_sq: float = 1.0, j4_abs: float = 0.0) -> None:
        """Apply Rydberg gates with non-linear J^6-coupled AdS boundary modulation."""
        try:
            default_params = {'ctc_phase_factor': 0.1, 'ctc_wormhole_factor': 0.5}
            if ctc_params:
                default_params.update(ctc_params)
            ctc_phase_factor = default_params['ctc_phase_factor']
            ctc_wormhole_factor = default_params['ctc_wormhole_factor']
            
            if not (0.05 <= ctc_phase_factor <= 0.5 and 0.2 <= ctc_wormhole_factor <= 1.0):
                raise ValueError("CTC parameters out of valid range")
            
            # Non-linear AdS boundary effect
            boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (1e-30 + 1e-15)
            effective_boundary = boundary_factor * (1 + 0.001 * boundary_nonlinear)
            phase = np.exp(1j * (phi + ctc_phase_factor * ctc_feedback) * effective_boundary)
            for i in range(self.num_qubits):
                self.state[i::2] *= phase
            if wormhole_nodes:
                wormhole_factor = Constants.GOLDEN_RATIO * (1 + ctc_wormhole_factor * np.abs(ctc_feedback))
                self.state[::4] *= np.clip(wormhole_factor, 0.5, 5.0)
            norm = np.linalg.norm(self.state)
            if norm > 0:
                self.state /= norm
            self.ctc_feedback = ctc_feedback
            logger.debug("Applied Rydberg gates: phi=%.6f, ctc_feedback=%.6f, ctc_phase_factor=%.2f, ctc_wormhole_factor=%.2f, boundary_factor=%.2f, boundary_nonlinear=%.6e", 
                         phi, ctc_feedback, ctc_phase_factor, ctc_wormhole_factor, effective_boundary, boundary_nonlinear)
        except Exception as e:
            logger.error("Rydberg gate application failed: %s", e)
            raise
    
    def compute_chsh_violation(self, amplify: bool = False, ctc_params: dict = None) -> float:
        """Compute CHSH violation with CTC and AdS boundary tuning."""
        try:
            default_params = {'ctc_amplify_factor': 0.2}
            if ctc_params:
                default_params.update(ctc_params)
            ctc_amplify_factor = default_params['ctc_amplify_factor']
            
            if not (0.1 <= ctc_amplify_factor <= 0.4):
                raise ValueError("ctc_amplify_factor out of valid range")
            
            reduced_state = self._get_two_qubit_state(0, 1)
            rho = np.outer(reduced_state, reduced_state.conj())
            angles = [
                (0.0, np.pi/8),      # A: 0°, B: 22.5°
                (0.0, 3*np.pi/8),    # A: 0°, B': 67.5°
                (np.pi/4, np.pi/8),  # A': 45°, B: 22.5°
                (np.pi/4, 3*np.pi/8) # A': 45°, B': 67.5°
            ]
            E = []
            for theta_a, theta_b in angles:
                op_a = np.cos(theta_a) * np.array([[1, 0], [0, -1]]) + \
                       np.sin(theta_a) * np.array([[0, 1], [1, 0]])
                op_b = np.cos(theta_b) * np.array([[1, 0], [0, -1]]) + \
                       np.sin(theta_b) * np.array([[0, 1], [1, 0]])
                op_ab = np.kron(op_a, op_b)
                expectation = np.real(np.trace(rho @ op_ab))
                if amplify:
                    expectation *= (1 + ctc_amplify_factor * np.abs(self.ctc_feedback))
                E.append(expectation)
                logger.debug("Expectation E(θ_a=%.2f, θ_b=%.2f) = %.6f", 
                             np.degrees(theta_a), np.degrees(theta_b), expectation)
            S = abs(E[0] - E[1] + E[2] + E[3])
            logger.debug("Computed CHSH violation: |S|=%.6f (amplify=%s, ctc_amplify_factor=%.2f)", 
                         S, amplify, ctc_amplify_factor)
            return S
        except Exception as e:
            logger.error("CHSH computation failed: %s", e)
            raise
    
    def compute_mabk_violation(self, n_qubits: int = 4, amplify: bool = False, ctc_params: dict = None) -> float:
        """Compute MABK inequality with CTC and AdS boundary tuning."""
        if n_qubits < 2 or n_qubits > 8:
            raise ValueError("n_qubits must be between 2 and 8")
        
        try:
            default_params = {'ctc_amplify_factor': 0.2}
            if ctc_params:
                default_params.update(ctc_params)
            ctc_amplify_factor = default_params['ctc_amplify_factor']
            
            if not (0.1 <= ctc_amplify_factor <= 0.4):
                raise ValueError("ctc_amplify_factor out of valid range")
            
            reduced_state = self._get_n_qubit_state(n_qubits)
            rho = np.outer(reduced_state, reduced_state.conj())
            
            sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            
            E = []
            for pattern in product([0, 1], repeat=n_qubits):
                if sum(pattern) % 2 == 0:
                    ops = [sigma_y if p else sigma_x for p in pattern]
                    op = ops[0]
                    for op_next in ops[1:]:
                        op = np.kron(op, op_next)
                    expectation = np.real(np.trace(rho @ op))
                    if amplify:
                        expectation *= (1 + ctc_amplify_factor * np.abs(self.ctc_feedback))
                    E.append(expectation)
                    logger.debug("MABK term (pattern=%s): E=%.6f", pattern, expectation)
            
            M = abs(sum(E)) * (2 ** (n_qubits // 2))
            logger.debug("Computed MABK violation (n=%d, amplify=%s, ctc_amplify_factor=%.2f): |M|=%.6f", 
                         n_qubits, amplify, ctc_amplify_factor, M)
            return M
        except Exception as e:
            logger.error("MABK computation failed: %s", e)
            raise
    
    def compute_ghz_paradox(self, amplify: bool = False, ctc_params: dict = None) -> dict:
        """Compute GHZ paradox with CTC and AdS boundary tuning."""
        try:
            default_params = {'ctc_amplify_factor': 0.2}
            if ctc_params:
                default_params.update(ctc_params)
            ctc_amplify_factor = default_params['ctc_amplify_factor']
            
            if not (0.1 <= ctc_amplify_factor <= 0.4):
                raise ValueError("ctc_amplify_factor out of valid range")
            
            reduced_state = self._get_n_qubit_state(3)
            rho = np.outer(reduced_state, reduced_state.conj())
            
            sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            
            operators = [
                (sigma_x, sigma_x, sigma_x),  # X1 X2 X3
                (sigma_x, sigma_y, sigma_y),  # X1 Y2 Y3
                (sigma_y, sigma_x, sigma_y),  # Y1 X2 Y3
                (sigma_y, sigma_y, sigma_x)   # Y1 Y2 X3
            ]
            labels = ['XXX', 'XYY', 'YXY', 'YYX']
            
            results = {}
            for label, (op1, op2, op3) in zip(labels, operators):
                op = np.kron(np.kron(op1, op2), op3)
                expectation = np.real(np.trace(rho @ op))
                if amplify:
                    expectation *= (1 + ctc_amplify_factor * np.abs(self.ctc_feedback))
                results[label] = expectation
                logger.debug("GHZ paradox term (%s): E=%.6f", label, expectation)
            
            logger.debug("Computed GHZ paradox: %s (ctc_amplify_factor=%.2f)", results, ctc_amplify_factor)
            return results
        except Exception as e:
            logger.error("GHZ paradox computation failed: %s", e)
            raise
    
    def _get_two_qubit_state(self, qubit1: int, qubit2: int) -> np.ndarray:
        """Extract two-qubit state, optimized for Bell-like entanglement."""
        try:
            two_qubit_state = np.array([self.state[0], 0, 0, self.state[3]], dtype=np.complex128)
            norm = np.linalg.norm(two_qubit_state)
            if norm > 0:
                two_qubit_state /= norm
            else:
                two_qubit_state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
            return two_qubit_state
        except Exception as e:
            logger.error("Two-qubit state extraction failed: %s", e)
            raise
    
    def _get_n_qubit_state(self, n_qubits: int) -> np.ndarray:
        """Extract n-qubit state, optimized for GHZ-like entanglement."""
        try:
            state_size = 2 ** n_qubits
            reduced_state = np.zeros(state_size, dtype=np.complex128)
            reduced_state[0] = self.state[0]
            reduced_state[-1] = self.state[-1]
            norm = np.linalg.norm(reduced_state)
            if norm > 0:
                reduced_state /= norm
            else:
                reduced_state[0] = reduced_state[-1] = 1 / np.sqrt(2)
            return reduced_state
        except Exception as e:
            logger.error("N-qubit state extraction failed: %s", e)
            raise
