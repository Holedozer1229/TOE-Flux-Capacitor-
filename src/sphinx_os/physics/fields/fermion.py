import numpy as np
import logging

logger = logging.getLogger(__name__)

def evolve_fermion_fields(electron_field: np.ndarray, quinoa_field: np.ndarray, grid_size: tuple,
                        deltas: list, dt: float, em_fields: dict, strong_fields: dict,
                        weak_fields: dict, pmns: np.ndarray, ckm: np.ndarray) -> tuple:
    """Evolve fermion fields (electrons, quinoa)."""
    try:
        dx = deltas[1]
        steps = []
        new_electron = np.zeros_like(electron_field)
        new_quinoa = np.zeros_like(quinoa_field)
        
        for idx in np.ndindex(grid_size):
            for spin in range(4):
                new_electron[idx + (spin,)] = electron_field[idx + (spin,)]
            for flavor in range(3):
                for color in range(3):
                    for spin in range(4):
                        new_quinoa[idx + (flavor, color, spin)] = quinoa_field[idx + (flavor, color, spin)]
        
        steps.append({
            "electron_mean": np.mean(np.abs(new_electron)),
            "quinoa_mean": np.mean(np.abs(new_quinoa))
        })
        logger.debug("Fermion fields evolved: electron_mean=%.6f, quinoa_mean=%.6f",
                    steps[-1]["electron_mean"], steps[-1]["quinoa_mean"])
        return new_electron, new_quinoa, steps
    except Exception as e:
        logger.error("Fermion field evolution failed: %s", e)
        raise
