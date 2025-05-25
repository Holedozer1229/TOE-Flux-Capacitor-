import numpy as np
from itertools import product
import logging

logger = logging.getLogger(__name__)

def partial_derivative(array: np.ndarray, idx: tuple, axis: int, delta: float, grid_size: tuple) -> float:
    """Compute partial derivative using central difference."""
    try:
        idx_plus = list(idx)
        idx_minus = list(idx)
        idx_plus[axis] = min(idx[axis] + 1, grid_size[axis] - 1)
        idx_minus[axis] = max(idx[axis] - 1, 0)
        return (array[tuple(idx_plus)] - array[tuple(idx_minus)]) / (2 * delta)
    except Exception as e:
        logger.error("Partial derivative failed: %s", e)
        raise

def compute_affine_connection(metric: np.ndarray, inverse_metric: np.ndarray, deltas: list, 
                              grid_size: tuple, psi: np.ndarray = None, j4_field: np.ndarray = None) -> np.ndarray:
    """Compute Christoffel symbols with non-linear J^6-coupled AdS boundary adjustments."""
    try:
        connection = np.zeros(grid_size + (6, 6, 6), dtype=np.float64)
        psi = psi if psi is not None else np.ones(grid_size, dtype=np.complex128) / np.sqrt(np.prod(grid_size))
        j4_field = j4_field if j4_field is not None else np.zeros(grid_size, dtype=np.float64)
        
        for idx in np.ndindex(grid_size):
            z = np.sum(np.abs(np.array(idx) - np.array(grid_size)/2)) / np.sum(grid_size)
            psi_abs_sq = np.mean(np.abs(psi[idx])**2)
            j4_abs = np.abs(j4_field[idx])
            boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (1e-30 + 1e-15)
            boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * boundary_nonlinear)
            for lambda_idx, mu, nu in product(range(6), repeat=3):
                for rho in range(6):
                    term1 = partial_derivative(metric, idx, mu, deltas[mu], grid_size)[(rho, nu)]
                    term2 = partial_derivative(metric, idx, nu, deltas[nu], grid_size)[(rho, mu)]
                    term3 = partial_derivative(metric, idx, rho, deltas[rho], grid_size)[(mu, nu)]
                    connection[idx + (lambda_idx, mu, nu)] += 0.5 * inverse_metric[idx + (lambda_idx, rho)] * (term1 + term2 - term3) * boundary_factor
                connection[idx + (lambda_idx, mu, nu)] = np.clip(connection[idx + (lambda_idx, mu, nu)], -1e3, 1e3)
        logger.debug("Affine connection computed: mean=%.6f, std=%.6f, boundary_nonlinear=%.6e", 
                     np.mean(connection), np.std(connection), boundary_nonlinear)
        return connection
    except Exception as e:
        logger.error("Affine connection computation failed: %s", e)
        raise

def compute_riemann_tensor(metric: np.ndarray, inverse_metric: np.ndarray, deltas: list, 
                           grid_size: tuple, graviton_field: np.ndarray = None, 
                           psi: np.ndarray = None, j4_field: np.ndarray = None) -> np.ndarray:
    """Compute Riemann curvature tensor with non-linear J^6-coupled AdS boundary and graviton effects."""
    try:
        connection = compute_affine_connection(metric, inverse_metric, deltas, grid_size, psi, j4_field)
        riemann = np.zeros(grid_size + (6, 6, 6, 6), dtype=np.float64)
        graviton_field = graviton_field if graviton_field is not None else np.zeros(grid_size + (6, 6))
        for idx in np.ndindex(grid_size):
            z = np.sum(np.abs(np.array(idx) - np.array(grid_size)/2)) / np.sum(grid_size)
            psi_abs_sq = np.mean(np.abs(psi[idx])**2) if psi is not None else 1.0
            j4_abs = np.abs(j4_field[idx]) if j4_field is not None else 0.0
            boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (1e-30 + 1e-15)
            boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * boundary_nonlinear)
            graviton_perturbation = 0.01 * np.mean(np.trace(graviton_field[idx]))
            for rho, sigma, mu, nu in product(range(6), repeat=4):
                term1 = partial_derivative(connection, idx, mu, deltas[mu], grid_size)[(rho, nu, sigma)]
                term2 = partial_derivative(connection, idx, nu, deltas[nu], grid_size)[(rho, mu, sigma)]
                term3 = 0.0
                for lambda_idx in range(6):
                    term3 += connection[idx + (rho, mu, lambda_idx)] * connection[idx + (lambda_idx, nu, sigma)]
                    term3 -= connection[idx + (rho, nu, lambda_idx)] * connection[idx + (lambda_idx, mu, sigma)]
                bary_weights = np.array([0.25, 0.25, 0.25, 0.25])
                riemann[idx + (rho, sigma, mu, nu)] = np.sum(bary_weights * (term1 - term2 + term3 + graviton_perturbation))
                napoleon_factor = 1 + 0.05 * np.cos(3 * np.sum(idx))
                riemann[idx + (rho, sigma, mu, nu)] *= napoleon_factor * boundary_factor
                riemann[idx + (rho, sigma, mu, nu)] = np.clip(riemann[idx + (rho, sigma, mu, nu)], -1e5, 1e5)
        logger.debug("Riemann tensor computed: mean=%.6f, std=%.6f, graviton_perturbation=%.6f, boundary_nonlinear=%.6e", 
                     np.mean(riemann), np.std(riemann), graviton_perturbation, boundary_nonlinear)
        return riemann
    except Exception as e:
        logger.error("Riemann tensor computation failed: %s", e)
        raise

def compute_curvature(riemann_tensor: np.ndarray, inverse_metric: np.ndarray, grid_size: tuple) -> tuple:
    """Compute Ricci tensor and Rio Ricci scalar with AdS boundary conditions."""
    try:
        ricci_tensor = np.zeros(grid_size + (6, 6), dtype=np.float64)
        ricci_scalar = np.zeros(grid_size, dtype=np.float64)
        for idx in np.ndindex(grid_size):
            z = np.sum(np.abs(np.array(idx) - np.array(grid_size)/2)) / np.sum(grid_size)
            boundary_factor = np.exp(-0.1 * z)
            for mu, nu in product(range(6), repeat=2):
                for lambda_idx in range(6):
                    ricci_tensor[idx + (mu, nu)] += riemann_tensor[idx + (lambda_idx, mu, lambda_idx, nu)]
            for mu, nu in product(range(6), repeat=2):
                ricci_scalar[idx] += inverse_metric[idx + (mu, nu)] * ricci_tensor[idx + (mu, nu)] * boundary_factor
            ricci_scalar[idx] = np.clip(ricci_scalar[idx], -1e5, 1e5)
        logger.debug("Rio Ricci scalar computed: mean=%.6f, std=%.6f", 
                     np.mean(ricci_scalar), np.std(ricci_scalar))
        return ricci_tensor, ricci_scalar
    except Exception as e:
        logger.error("Curvature computation failed: %s", e)
        raise
