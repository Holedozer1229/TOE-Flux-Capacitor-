import numpy as np
import logging
from typing import Tuple
from .constants import Constants
from .core.sphinx_os import SphinxOS
from .physics.lattice import TetrahedralLattice

logger = logging.getLogger(__name__)

class UnifiedSpacetimeSimulator:
    """Simulates 6D spacetime with non-linear J^6-coupled graviton and AdS boundary effects."""
    
    def __init__(self, grid_size: Tuple[int, ...], lambda_eigen: float):
        try:
            self.grid_size = grid_size
            self.lambda_eigen = lambda_eigen
            self.grid = np.zeros(grid_size, dtype=np.complex128)
            self.psi = np.exp(2j * np.pi * np.random.rand(*grid_size)) / np.sqrt(np.prod(grid_size))
            self.phi_golden = Constants.GOLDEN_RATIO
            self.sphinx_os = SphinxOS()
            self.lattice = TetrahedralLattice(grid_size)
            self.ricci_scalar = np.ones(grid_size, dtype=np.float64)
            logger.info("Spacetime simulator initialized with grid size %s, lambda_eigen=%.2f",
                        grid_size, lambda_eigen)
        except Exception as e:
            logger.error("Simulator initialization failed: %s", e)
            raise

    def initialize_tetrahedral_lattice(self) -> None:
        try:
            grid_shape = self.grid_size
            for idx in np.ndindex(grid_shape):
                weight = self.lambda_eigen * (1 + 0.1 * np.sum(np.array(idx) / np.array(grid_shape)))
                self.grid[idx] = weight
            self.psi = self.psi * np.exp(1j * self.grid)
            self.sphinx_os.quantum_state = self.psi
            logger.info("Tetrahedral lattice initialized with eigenvalue %.2f, grid max weight %.6f",
                        self.lambda_eigen, np.max(np.abs(self.grid)))
        except Exception as e:
            logger.error("Tetrahedral lattice initialization failed: %s", e)
            raise

    def mobius_spiral_trajectory(self, t: float, r: float, n: float, m_shift_amplitude: float, 
                                 ricci_scalar: np.ndarray, graviton_field: np.ndarray, 
                                 psi: np.ndarray, j4_field: np.ndarray) -> Tuple[float, float, float]:
        """Compute Möbius spiral with non-linear J^6-coupled graviton and AdS boundary."""
        try:
            theta = t * np.pi
            phi = n * theta / 2
            ricci_mean = np.mean(np.abs(ricci_scalar)) if ricci_scalar.size > 0 else 1.0
            graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1)) if graviton_field.size > 0 else 0.0
            psi_abs_sq = np.mean(np.abs(psi)**2) if psi.size > 0 else 1.0
            j4_abs = np.mean(np.abs(j4_field)) if j4_field.size > 0 else 0.0
            graviton_nonlinear = np.abs(graviton_trace)**6 / (1e-30 + 1e-15)
            napoleon_factor = 1 + 0.05 * np.cos(3 * theta)
            ads_boundary_factor = np.exp(-0.1 * np.abs(t))
            boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (1e-30 + 1e-15)
            m_shift = m_shift_amplitude * (1 + 0.1 * np.sin(theta) * (1 + 0.01 * ricci_mean + 0.01 * graviton_trace + 0.001 * graviton_nonlinear + 0.001 * boundary_nonlinear) * napoleon_factor * ads_boundary_factor)
            x = r * np.cos(theta) * np.sin(phi) * m_shift
            y = r * np.sin(theta) * np.sin(phi) * m_shift
            z = r * np.cos(phi) * m_shift
            logger.debug("Möbius spiral: x=%.6f, y=%.6f, z=%.6f, m_shift=%.6f, rio_mean=%.6f, graviton_trace=%.6f, graviton_nonlinear=%.6e, boundary_nonlinear=%.6e", 
                         x, y, z, m_shift, ricci_mean, graviton_trace, graviton_nonlinear, boundary_nonlinear)
            return x, y, z
        except Exception as e:
            logger.error("Möbius spiral computation failed: %s", e)
            raise
    
    def tetrahedral_weights(self, idx: tuple, grid_shape: tuple, m_shift_amplitude: float, 
                            ricci_scalar: np.ndarray, graviton_field: np.ndarray, 
                            psi: np.ndarray, j4_field: np.ndarray) -> float:
        """Compute tetrahedral weights with non-linear J^6-coupled graviton and AdS boundary."""
        try:
            u = 2 * np.pi * (np.sum(idx) / np.sum(grid_shape)) - np.pi
            v = np.pi * (idx[0] / grid_shape[0]) - np.pi / 2
            ricci_mean = np.mean(np.abs(ricci_scalar)) if ricci_scalar.size > 0 else 1.0
            graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1)) if graviton_field.size > 0 else 0.0
            psi_abs_sq = np.mean(np.abs(psi)**2) if psi.size > 0 else 1.0
            j4_abs = np.mean(np.abs(j4_field)) if j4_field.size > 0 else 0.0
            graviton_nonlinear = np.abs(graviton_trace)**6 / (1e-30 + 1e-15)
            boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (1e-30 + 1e-15)
            bary_weights = self.lattice.get_barycentric_weights(idx)
            napoleon_factor = 1 + 0.05 * np.cos(3 * (u + v))
            ads_boundary_factor = np.exp(-0.1 * np.sum(np.abs(np.array(idx) - np.array(grid_shape)/2)))
            m_shift = m_shift_amplitude * (1 + 0.1 * np.cos(u + v) * (1 + 0.01 * ricci_mean + 0.01 * graviton_trace + 0.001 * graviton_nonlinear + 0.001 * boundary_nonlinear) * napoleon_factor * ads_boundary_factor)
            interpolated_m_shift = np.sum(bary_weights * m_shift)
            logger.debug("Tetrahedral weight: idx=%s, m_shift=%.6f, interpolated_m_shift=%.6f, rio_mean=%.6f, graviton_trace=%.6f, graviton_nonlinear=%.6e, boundary_nonlinear=%.6e", 
                         idx, m_shift, interpolated_m_shift, ricci_mean, graviton_trace, graviton_nonlinear, boundary_nonlinear)
            return interpolated_m_shift
        except Exception as e:
            logger.error("Tetrahedral weights computation failed: %s", e)
            raise
    
    def compute_ctc_term(self, tau: float, phi: float, j6_modulation: float, ctc_params: dict = None) -> float:
        """Compute CTC term with non-linear J^6-coupled modulation."""
        try:
            default_params = {
                'kappa_ctc': Constants.KAPPA_CTC,
                'r': 3.0,
                'n': 2.0,
                'm_shift_amplitude': 2.72
            }
            if ctc_params:
                default_params.update(ctc_params)
            kappa_ctc = default_params['kappa_ctc']
            r = default_params['r']
            n = default_params['n']
            m_shift_amplitude = default_params['m_shift_amplitude']
            
            if not (0.1 <= kappa_ctc <= 1.0 and 1.0 <= r <= 5.0 and 0.5 <= n <= 3.0 and 1.5 <= m_shift_amplitude <= 4.0):
                raise ValueError("CTC parameters out of valid range")
            
            t = tau / 2.0
            graviton_field = self.sphinx_os.graviton_field if hasattr(self.sphinx_os, 'graviton_field') else np.zeros(self.grid_size + (6, 6))
            x, y, z = self.mobius_spiral_trajectory(t, r, n, m_shift_amplitude, self.ricci_scalar, graviton_field, self.psi, self.grid)
            spiral_factor = np.sqrt(x**2 + y**2 + z**2)
            spiral_factor = np.clip(spiral_factor, 0.1, 10.0)
            
            idx = tuple(np.array(self.grid_size) // 2)
            tetra_weight = self.tetrahedral_weights(idx, self.grid_size, m_shift_amplitude, self.ricci_scalar, graviton_field, self.psi, self.grid)
            
            self.sphinx_os.quantum_walk()
            self.psi = self.sphinx_os.quantum_state
            past_psi = self.psi * np.exp(-1j * tau * self.lambda_eigen)
            arg_diff = np.angle(self.psi) - np.angle(past_psi)
            ctc_term = kappa_ctc * np.exp(1j * phi * np.tanh(arg_diff)) * np.abs(self.psi) * spiral_factor * tetra_weight
            ctc_value = np.clip(np.mean(ctc_term).real, -1.0, 1.0)
            logger.debug("CTC term computed: value=%.6f, spiral_factor=%.6f, tetra_weight=%.6f, kappa_ctc=%.2f, r=%.2f, n=%.2f, m_shift_amplitude=%.2f, j6_modulation=%.6f", 
                         ctc_value, spiral_factor, tetra_weight, kappa_ctc, r, n, m_shift_amplitude, j6_modulation)
            return ctc_value
        except Exception as e:
            logger.error("CTC term computation failed: %s", e)
            raise

    def compute_entanglement_entropy(self) -> float:
        """Compute entanglement entropy of the quantum state."""
        try:
            probabilities = np.abs(self.psi)**2
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            logger.debug("Computed entanglement entropy: %.6f", entropy)
            return entropy
        except Exception as e:
            logger.error("Entanglement entropy computation failed: %s", e)
            raise

    def compute_scalar_field(self, r: np.ndarray, t: float) -> float:
        """Compute scalar field value at position r and time t."""
        weights = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        center = np.array(self.grid_size) / 2
        r_6d = np.sqrt(np.sum(weights * (r - center)**2))
        k = Constants.K / Constants.DELTA_X
        omega = 2 * np.pi / (100 * Constants.DELTA_T)
        term1 = -r_6d**2 * np.cos(k * r_6d - omega * t)
        term2 = 2 * r_6d * np.sin(k * r_6d - omega * t)
        term3 = 2 * np.cos(k * r_6d - omega * t)
        term4 = 0.1 * np.sin(1e-3 * r_6d)
        phi = -(term1 + term2 + term3 + term4)
        return phi
