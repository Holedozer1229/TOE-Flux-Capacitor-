import numpy as np
import logging

logger = logging.getLogger(__name__)

def initialize_graviton_field(grid_size: tuple, deltas: list) -> np.ndarray:
    """Initialize the spin-2 graviton field h_{\mu\nu}."""
    try:
        graviton_field = np.zeros(grid_size + (6, 6), dtype=np.float64)
        for idx in np.ndindex(grid_size):
            graviton_field[idx] = np.random.normal(0, 1e-5, (6, 6))
            graviton_field[idx] = (graviton_field[idx] + graviton_field[idx].T) / 2
        logger.info("Graviton field initialized with grid size %s", grid_size)
        return graviton_field
    except Exception as e:
        logger.error("Graviton field initialization failed: %s", e)
        raise

def evolve_graviton_field(graviton_field: np.ndarray, grid_size: tuple, deltas: list, dt: float,
                          scalar_field: np.ndarray, ricci_scalar: np.ndarray, 
                          psi: np.ndarray, j4_field: np.ndarray) -> tuple:
    """Evolve the graviton field with non-linear J^6 coupling."""
    try:
        dx = deltas[1]
        steps = []
        h = graviton_field
        laplacian = np.zeros_like(h)
        
        for dim in range(6):
            laplacian += np.roll(h, 1, axis=dim) + np.roll(h, -1, axis=dim) - 2 * h
        laplacian /= dx**2
        
        # Non-linear J^6 source term
        scalar_mean = np.mean(np.abs(scalar_field)) if scalar_field.size > 0 else 0.0
        rio_mean = np.mean(ricci_scalar) if ricci_scalar.size > 0 else 1.0
        psi_abs_sq = np.mean(np.abs(psi)**2) if psi.size > 0 else 1.0
        j4_abs = np.mean(np.abs(j4_field)) if j4_field.size > 0 else 0.0
        j6_source = (scalar_mean**6 * rio_mean * psi_abs_sq * j4_abs**3) / (1e-30 + 1e-15)
        source = 0.01 * scalar_mean * rio_mean * np.eye(6) + 0.001 * j6_source * np.eye(6)
        
        d2h_dt2 = laplacian + source
        new_h = h + dt * (h - np.roll(h, 1, axis=0)) / dt + 0.5 * dt**2 * d2h_dt2
        
        for idx in np.ndindex(grid_size):
            new_h[idx] = (new_h[idx] + new_h[idx].T) / 2
        
        new_h = np.clip(new_h, -1e3, 1e3)
        
        graviton_trace = np.mean(np.trace(new_h, axis1=-2, axis2=-1))
        steps.append({
            "graviton_trace": graviton_trace,
            "graviton_norm": np.linalg.norm(new_h),
            "j6_source": j6_source
        })
        
        logger.debug("Graviton field evolved: trace=%.6f, norm=%.6f, j6_source=%.6e", 
                     steps[-1]["graviton_trace"], steps[-1]["graviton_norm"], steps[-1]["j6_source"])
        return new_h, steps
    except Exception as e:
        logger.error("Graviton field evolution failed: %s", e)
        raise
