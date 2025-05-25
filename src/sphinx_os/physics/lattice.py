import numpy as np
import logging

logger = logging.getLogger(__name__)

class TetrahedralLattice:
    """6D tetrahedral lattice for spacetime simulation."""
    
    def __init__(self, grid_size: tuple):
        self.grid_size = grid_size
        self.barycentric_coords = {}
        logger.info("Tetrahedral lattice initialized with grid size %s", grid_size)
    
    def get_barycentric_weights(self, idx: tuple) -> np.ndarray:
        """Compute barycentric interpolation weights with Napoleon’s theorem modulation."""
        try:
            if idx not in self.barycentric_coords:
                weights = np.array([0.25, 0.25, 0.25, 0.25])  # Simplified for center
                centroid_factor = 1 + 0.05 * np.cos(3 * np.sum(idx))  # Napoleon’s theorem
                weights *= centroid_factor
                weights /= np.sum(weights)  # Normalize
                self.barycentric_coords[idx] = weights
            logger.debug("Barycentric weights for idx=%s: %s", idx, self.barycentric_coords[idx])
            return self.barycentric_coords[idx]
        except Exception as e:
            logger.error("Barycentric weights computation failed: %s", e)
            raise
    
    @property
    def coordinates(self) -> np.ndarray:
        """Generate lattice coordinates."""
        try:
            coords = np.array(np.meshgrid(*[np.arange(s) for s in self.grid_size])).T.reshape(-1, len(self.grid_size))
            logger.debug("Generated lattice coordinates shape: %s", coords.shape)
            return coords
        except Exception as e:
            logger.error("Lattice coordinates generation failed: %s", e)
            raise
