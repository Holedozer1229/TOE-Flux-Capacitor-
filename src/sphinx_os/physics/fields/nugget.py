import numpy as np
import logging
from ...utils.math_utils import compute_j6_potential

logger = logging.getLogger(__name__)

def initialize_nugget_field(grid_size: tuple, deltas: list) -> np.ndarray:
    """Initialize the Nugget scalar field."""
    try:
        nugget_field = np.zeros(grid_size, dtype=np.complex128)
        for idx in np.ndindex(grid_size):
            nugget_field[idx] = np.random.normal(0, 1e-5) + 1j * np.random.normal(0, 1e-5)
        logger.info("Nugget field initialized with grid size %s", grid_size)
        return nugget_field
    except Exception as e:
        logger.error("Nugget field initialization failed: %s", e)
        raise

def evolve_nugget_field(nugget_field: np.ndarray, grid_size: tuple, deltas: list, dt: float,
                        graviton_field: np.ndarray, ricci_scalar: np.ndarray, psi: np.ndarray,
                        j4_field: np.ndarray, body_positions: list = None, 
                        body_masses: list = None) -> tuple:
    """Evolve the Nugget scalar field with J^6 potential and three-body effects."""
    try:
        dx = deltas[1]
        steps = []
        phi = nugget_field
        laplacian = np.zeros_like(phi)
        
        # Compute Laplacian
        for dim in range(6):
            laplacian += np.roll(phi, 1, axis=dim) + np.roll(phi, -1, axis=dim) - 2 * phi
        laplacian /= dx**2
        
        # Compute J^6 potential and its derivative
        V_j6, dV_j6_dphi = compute_j6_potential(
            phi, j4_field, psi, ricci_scalar, graviton_field,
            body_positions=body_positions, body_masses=body_masses
        )
        
        # Graviton coupling
        graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1)) if graviton_field is not None else 0.0
        graviton_nonlinear = np.abs(graviton_trace)**6 / (1e-30 + 1e-15)  # Unsimplified J^6-like term
        graviton_coupling = 0.01 * graviton_trace + 0.001 * graviton_nonlinear
        
        # Three-body gravitational potential
        three_body_potential = np.zeros_like(phi)
        if body_positions and body_masses:
            G = 6.67430e-11  # Gravitational constant
            for idx in np.ndindex(grid_size):
                for pos, mass in zip(body_positions, body_masses):
                    dist = np.sqrt(sum((np.array(idx[:3]) - pos)**2) + 1e-15)
                    three_body_potential[idx] += G * mass / dist
        
        # Evolve field
        d2phi_dt2 = laplacian - dV_j6_dphi - graviton_coupling * phi - three_body_potential
        new_phi = phi + dt * (phi - np.roll(phi, 1, axis=0)) / dt + 0.5 * dt**2 * d2phi_dt2
        
        # Clip for numerical stability
        new_phi = np.clip(new_phi, -1e3, 1e3)
        
        # Log metrics
        phi_mean = np.mean(np.abs(new_phi))
        dist_sum = (sum(np.sqrt(sum((p1 - p2)**2)) for i, p1 in enumerate(body_positions) 
                        for p2 in body_positions[i+1:]) if body_positions else 0.0)
        steps.append({
            "phi_mean": phi_mean,
            "phi_norm": np.linalg.norm(new_phi),
            "graviton_coupling": graviton_coupling,
            "body_dist_sum": dist_sum
        })
        
        logger.debug("Nugget field evolved: phi_mean=%.6f, phi_norm=%.6f, graviton_coupling=%.6e, body_dist_sum=%.6f", 
                     steps[-1]["phi_mean"], steps[-1]["phi_norm"], steps[-1]["graviton_coupling"], dist_sum)
        return new_phi, steps
    except Exception as e:
        logger.error("Nugget field evolution failed: %s", e)
        raise
