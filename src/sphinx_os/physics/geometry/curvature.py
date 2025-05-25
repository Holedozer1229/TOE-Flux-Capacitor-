import numpy as np
import logging
from ...constants import Constants

logger = logging.getLogger(__name__)

def compute_riemann_tensor(metric: np.ndarray, grid_size: tuple, dx: float, 
                           body_positions: list = None) -> np.ndarray:
    """Compute the Riemann curvature tensor R^{\rho}_{\sigma\mu\nu}."""
    try:
        riemann = np.zeros(grid_size + (6, 6, 6, 6), dtype=np.float64)
        christoffel = np.zeros(grid_size + (6, 6, 6), dtype=np.float64)
        
        # Compute Christoffel symbols (first kind)
        for idx in np.ndindex(grid_size):
            for rho, mu, nu in np.ndindex((6, 6, 6)):
                for sigma in range(6):
                    christoffel[idx + (rho, mu, nu)] += 0.5 * (
                        np.roll(metric[idx + (mu, sigma)], -1, axis=mu) / dx -
                        np.roll(metric[idx + (nu, sigma)], -1, axis=nu) / dx +
                        np.roll(metric[idx + (mu, nu)], -1, axis=sigma) / dx
                    )
        
        # Compute Riemann tensor
        for idx in np.ndindex(grid_size):
            for rho, sigma, mu, nu in np.ndindex((6, 6, 6, 6)):
                riemann[idx + (rho, sigma, mu, nu)] = (
                    np.roll(christoffel[idx + (rho, nu, sigma)], -1, axis=mu) / dx -
                    np.roll(christoffel[idx + (rho, mu, sigma)], -1, axis=nu) / dx
                )
                for lambda_idx in range(6):
                    riemann[idx + (rho, sigma, mu, nu)] += (
                        christoffel[idx + (rho, mu, lambda_idx)] * 
                        christoffel[idx + (lambda_idx, nu, sigma)] -
                        christoffel[idx + (rho, nu, lambda_idx)] * 
                        christoffel[idx + (lambda_idx, mu, sigma)]
                    )
        
        logger.debug("Riemann tensor computed: mean=%.6e, std=%.6e", 
                     np.mean(riemann), np.std(riemann))
        return riemann
    except Exception as e:
        logger.error("Riemann tensor computation failed: %s", e)
        raise

def compute_curvature(metric: np.ndarray, inverse_metric: np.ndarray, grid_size: tuple, 
                      dx: float, scalar_field: np.ndarray = None, 
                      body_positions: list = None) -> tuple:
    """Compute Ricci tensor and Rio Ricci scalar with unsimplified AdS boundary effects."""
    try:
        riemann = compute_riemann_tensor(metric, grid_size, dx, body_positions)
        ricci_tensor = np.zeros(grid_size + (6, 6), dtype=np.float64)
        ricci_scalar = np.zeros(grid_size, dtype=np.float64)
        
        # Compute Ricci tensor
        for idx in np.ndindex(grid_size):
            for mu, nu in np.ndindex((6, 6)):
                for lambda_idx in range(6):
                    ricci_tensor[idx + (mu, nu)] += riemann[idx + (lambda_idx, mu, lambda_idx, nu)]
        
        # Compute Rio Ricci scalar with AdS boundary and three-body effects
        j6_scale = 1e-30
        epsilon = 1e-15
        z = np.sum(np.abs(np.array(grid_size) / 2)) / np.sum(grid_size)
        for idx in np.ndindex(grid_size):
            for mu, nu in np.ndindex((6, 6)):
                ricci_scalar[idx] += inverse_metric[idx + (mu, nu)] * ricci_tensor[idx + (mu, nu)]
            
            # Apply unsimplified AdS boundary factor
            phi_norm = (np.abs(scalar_field[idx]) / (np.max(np.abs(scalar_field)) + 1e-10) 
                        if scalar_field is not None else 1.0)
            boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * (phi_norm**6 / (j6_scale + epsilon)))
            ricci_scalar[idx] *= boundary_factor
            
            # Three-body curvature perturbation
            if body_positions:
                G = 6.67430e-11
                for pos in body_positions:
                    dist = np.sqrt(sum((np.array(idx[:3]) - pos)**2) + 1e-15)
                    ricci_scalar[idx] += G / dist * 0.01  # Small curvature perturbation
        
        # Clip for stability
        ricci_scalar = np.clip(ricci_scalar, -1e5, 1e5)
        
        # Log metrics
        ricci_mean = np.mean(np.abs(ricci_scalar))
        ricci_std = np.std(ricci_scalar)
        dist_sum = (sum(np.sqrt(sum((p1 - p2)**2)) for i, p1 in enumerate(body_positions) 
                        for p2 in body_positions[i+1:]) if body_positions else 0.0)
        logger.debug("Curvature computed: ricci_mean=%.6f, ricci_std=%.6f, boundary_factor=%.6f, body_dist_sum=%.6f", 
                     ricci_mean, ricci_std, boundary_factor, dist_sum)
        return ricci_tensor, ricci_scalar
    except Exception as e:
        logger.error("Curvature computation failed: %s", e)
        raise
