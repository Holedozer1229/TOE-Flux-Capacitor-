import numpy as np
import logging

logger = logging.getLogger(__name__)

def initialize_em_fields(grid_size: tuple, wormhole_nodes: np.ndarray, time: float, j4_field: np.ndarray) -> dict:
    """Initialize electromagnetic fields."""
    try:
        em_fields = {
            "J4": j4_field,
            "metric": np.zeros(grid_size + (6, 6), dtype=np.float64)
        }
        for idx in np.ndindex(grid_size):
            em_fields["metric"][idx] = np.eye(6)
        logger.info("Electromagnetic fields initialized")
        return em_fields
    except Exception as e:
        logger.error("EM fields initialization failed: %s", e)
        raise

def initialize_weak_fields(grid_size: tuple, dx: float) -> dict:
    """Initialize weak fields."""
    try:
        weak_fields = {"W": np.zeros(grid_size + (3,), dtype=np.complex128)}
        logger.info("Weak fields initialized")
        return weak_fields
    except Exception as e:
        logger.error("Weak fields initialization failed: %s", e)
        raise

def initialize_strong_fields(grid_size: tuple, dx: float) -> dict:
    """Initialize strong fields."""
    try:
        strong_fields = {"G": np.zeros(grid_size + (8,), dtype=np.complex128)}
        logger.info("Strong fields initialized")
        return strong_fields
    except Exception as e:
        logger.error("Strong fields initialization failed: %s", e)
        raise

def evolve_gauge_fields(strong_fields: dict, weak_fields: dict, grid_size: tuple, deltas: list) -> tuple:
    """Evolve gauge fields."""
    try:
        new_strong = strong_fields.copy()
        new_weak = weak_fields.copy()
        logger.debug("Gauge fields evolved")
        return new_strong, new_weak
    except Exception as e:
        logger.error("Gauge fields evolution failed: %s", e)
        raise
