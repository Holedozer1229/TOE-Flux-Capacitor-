import numpy as np
from ....utils.math_utils import compute_j6_potential
from ....constants import CONFIG
import logging

logger = logging.getLogger(__name__)

def evolve_nugget_field(nugget_field: np.ndarray, grid_size: tuple, deltas: list, dt: float,
                        j4_field: np.ndarray = None, psi: np.ndarray = None, ricci_scalar: np.ndarray = None,
                        graviton_field: np.ndarray = None) -> tuple:
    """Evolve the Nugget scalar field with non-linear J^6 coupling and graviton interactions."""
    try:
        alpha_phi = CONFIG["alpha_phi"]
        m_shift = CONFIG["m_shift"]
        dx = deltas[1]
        steps = []
        
        j4_field = j4_field if j4_field is not None else np.zeros(grid_size, dtype=np.float64)
        psi = psi if psi is not None else np.ones(grid_size, dtype=np.complex128) / np.sqrt(np.prod(grid_size))
        ricci_scalar = ricci_scalar if ricci_scalar is not None else np.ones(grid_size, dtype=np.float64)
        graviton_field = graviton_field if graviton_field is not None else np.zeros(grid_size + (6, 6), dtype=np.float64)
        
        if not np.all(np.isfinite(ricci_scalar)):
            logger.warning("Non-finite Rio Ricci scalar detected, clamping values")
            ricci_scalar = np.clip(ricci_scalar, -1e5, 1e5)
        
        phi = nugget_field
        dV_dphi = 2 * alpha_phi * phi + 4 * m_shift * phi**3
        
        V_j6, dV_j6_dphi = compute_j6_potential(
            phi, j4_field, psi, ricci_scalar, graviton_field=graviton_field,
            kappa_j6=CONFIG["kappa_j6"],
            kappa_j6_eff=CONFIG["kappa_j6_eff"],
            j6_scaling_factor=CONFIG["j6_scaling_factor"],
            epsilon=CONFIG["epsilon"],
            omega_res=CONFIG["resonance_frequency"] * 2 * np.pi
        )
        
        laplacian = np.zeros_like(phi)
        for dim in range(6):
            laplacian += np.roll(phi, 1, axis=dim) + np.roll(phi, -1, axis=dim) - 2 * phi
        laplacian /= dx**2
        
        # Non-linear graviton coupling
        graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1))
        graviton_nonlinear = np.abs(graviton_trace)**6 / (1e-30 + 1e-15)  # J^6-like term
        graviton_coupling = 0.01 * graviton_trace * phi + 0.001 * graviton_nonlinear * phi
        
        d2phi_dt2 = laplacian - dV_dphi - dV_j6_dphi - graviton_coupling
        new_phi = phi + dt * (phi - np.roll(phi, 1, axis=0)) / dt + 0.5 * dt**2 * d2phi_dt2
        
        new_phi = np.clip(new_phi, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        
        steps.append({
            "phi_mean": np.mean(np.abs(new_phi)),
            "V_j6_mean": np.mean(V_j6),
            "dV_j6_dphi_mean": np.mean(np.abs(dV_j6_dphi)),
            "rio_mean": np.mean(ricci_scalar),
            "graviton_trace": graviton_trace,
            "graviton_nonlinear": graviton_nonlinear
        })
        
        logger.debug("Nugget field evolved: phi_mean=%.6f, V_j6_mean=%.6e, rio_mean=%.6f, graviton_trace=%.6f, graviton_nonlinear=%.6e", 
                     steps[-1]["phi_mean"], steps[-1]["V_j6_mean"], steps[-1]["rio_mean"], 
                     steps[-1]["graviton_trace"], steps[-1]["graviton_nonlinear"])
        return new_phi, steps
    except Exception as e:
        logger.error("Nugget field evolution failed: %s", e)
        raise
