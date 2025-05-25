import numpy as np
import logging

logger = logging.getLogger(__name__)

class EntanglementCache:
    """Caches entanglement entropy history."""
    
    def __init__(self):
        self.entropy_history = []
        logger.info("Entanglement cache initialized")
    
    def cache_entropy(self, entropy: float) -> None:
        """Cache an entropy value."""
        try:
            self.entropy_history.append(entropy)
            logger.debug("Cached entropy: %.6f", entropy)
        except Exception as e:
            logger.error("Failed to cache entropy: %s", e)
            raise
    
    def get_history(self) -> list:
        """Return the entropy history."""
        return self.entropy_history
