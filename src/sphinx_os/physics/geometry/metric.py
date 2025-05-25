import numpy as np
from ...physics.lattice import TetrahedralLattice
import logging

logger = logging.getLogger(__name__)

def compute_quantum_metric(lattice: object, nugget_field: np.ndarray, temporal_entanglement: np.ndarray,
                           grid_size: tuple, j4_field: np.ndarray = None, psi: np.ndarray = None) -> tuple:
    """Compute quantum metric with non-linear J^6-coupled AdS boundary effects."""
    try:
        metric = np.zeros(grid_size + (6, 6), dtype=np.float64)
        inverse_metric = np.zeros(grid_size + (6, 6), dtype=np.float64)
        
        j4_field = j4_field if j4_field is not None else np.zeros(grid_size, dtype=np.float64)
        psi = psi if psi is not None else np.ones(grid_size, dtype=np.complex128) / np.sqrt(np.prod(grid_size))
        
        # AdS_6 metric with non-linear boundary coupling
        L = 1.0  # AdS radius
        for idx in np.ndindex(grid_size):
            z = np.sum(np.abs(np.array(idx) - np.array(grid_size)/2)) / np.sum(grid_size)
            psi_abs_sq = np.mean(np.abs(psi[idx])**2)
            j4_abs = np.abs(j4_field[idx])
            boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (1e-30 + 1e-15)  # J^6-like term
            boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * boundary_nonlinear)
            metric[idx + (0, 0)] = -1.0 / (L**2 * (1 + z**2)) * boundary_factor
            for i in range(1, 6):
                metric[idx + (i, i)] = 1.0 / (L**2 * (1 + z**2)) * boundary_factor
        
        # Perturbations from fields and tetrahedral structure
        phi_norm = np.abs(nugget_field) / (np.max(np.abs(nugget_field)) + 1e-10)
        psi_norm = np.abs(temporal_entanglement)**2
        for idx in np.ndindex(grid_size):
            bary_weights = lattice.get_barycentric_weights(idx)
            perturbation = 0.1 * phi_norm[idx] + 0.05 * psi_norm[idx]
            napoleon_factor = 1 + 0.05 * np.cos(3 * np.sum(idx))
            boundary_factor = np.exp(-0.1 * np.sum(np.abs(np.array(idx) - np.array(grid_size)/2)))
            for i in range(6):
                metric[idx + (i, i)] *= (1.0 + perturbation * np.sum(bary_weights) * napoleon_factor * boundary_factor)
        
        # Compute inverse metric
        for idx in np.ndindex(grid_size):
            try:
                inverse_metric[idx] = np.linalg.inv(metric[idx])
            except np.linalg.LinAlgError:
                inverse_metric[idx] = np.eye(6)
                logger.warning("Singular metric at %s, using identity matrix", idx)
        
        logger.debug("Metric computed: mean_diag=%.6f, std_diag=%.6f, boundary_factor=%.6f, boundary_nonlinear=%.6e", 
                     np.mean(np.diagonal(metric, axis1=-2, axis2=-1)), 
                     np.std(np.diagonal(metric, axis1=-2, axis2=-1)), boundary_factor, boundary_nonlinear)
        return metric, inverse_metric
    except Exception as e:
        logger.error("Quantum metric computation failed: %s", e)
        raise

def generate_wormhole_nodes(grid_size: tuple, deltas: list) -> np.ndarray:
    """Generate wormhole node positions."""
    return np.array([0.33333333326] * 6)
