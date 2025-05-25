import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_entanglement_entropy(state: np.ndarray) -> float:
    """Compute entanglement entropy of a quantum state."""
    try:
        probabilities = np.abs(state)**2
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        logger.debug("Computed entanglement entropy: %.6f", entropy)
        return entropy
    except Exception as e:
        logger.error("Entanglement entropy computation failed: %s", e)
        raise
