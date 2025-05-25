import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_stress_energy(em_fields: dict, quantum_state: np.ndarray, nugget_field: np.ndarray,
                          metric: np.ndarray, inverse_metric: np.ndarray, I_mu_nu: np.ndarray,
                          grid_size: tuple, deltas: list) -> np.ndarray:
    """Compute stress-energy tensor with Rio curvature."""
    try:
        stress_energy = np.zeros(grid_size + (6, 6), dtype=np.float64)
        for idx in np.ndindex(grid_size):
            stress_energy[idx] = I_mu_nu[idx] * np.abs(quantum_state[idx])**2
        logger.debug("Stress-energy tensor computed: mean=%.6f", np.mean(stress_energy))
        return stress_energy
    except Exception as e:
        logger.error("Stress-energy computation failed: %s", e)
        raise

def compute_einstein_tensor(ricci_tensor: np.ndarray, ricci_scalar: np.ndarray, metric: np.ndarray,
                            grid_size: tuple) -> np.ndarray:
    """Compute Einstein tensor."""
    try:
        einstein_tensor = np.zeros(grid_size + (6, 6), dtype=np.float64)
        for idx in np.ndindex(grid_size):
            einstein_tensor[idx] = ricci_tensor[idx] - 0.5 * ricci_scalar[idx] * metric[idx]
        logger.debug("Einstein tensor computed: mean=%.6f", np.mean(einstein_tensor))
        return einstein_tensor
    except Exception as e:
        logger.error("Einstein tensor computation failed: %s", e)
        raise

def compute_information_tensor(electron_field: np.ndarray, grid_size: tuple, metric: np.ndarray,
                               einstein_tensor: np.ndarray) -> tuple:
    """Compute information tensor and relative entropy."""
    try:
        I_mu_nu = np.zeros(grid_size + (6, 6), dtype=np.float64)
        relative_entropy = np.zeros(grid_size, dtype=np.float64)
        for idx in np.ndindex(grid_size):
            I_mu_nu[idx] = np.eye(6) * np.sum(np.abs(electron_field[idx])**2)
            relative_entropy[idx] = np.sum(np.abs(electron_field[idx])**2 * np.log(np.abs(electron_field[idx])**2 + 1e-10))
        logger.debug("Information tensor computed: mean=%.6f, relative_entropy_mean=%.6f",
                     np.mean(I_mu_nu), np.mean(relative_entropy))
        return I_mu_nu, relative_entropy
    except Exception as e:
        logger.error("Information tensor computation failed: %s", e)
        raise
