import numpy as np
import logging

logger = logging.getLogger(__name__)

class Constants:
    """Configuration parameters for the SphinxOs package."""
    
    # Lattice and simulation grid parameters
    GRID_SIZE = (5, 5, 5, 5, 3, 3)  # 6D grid size
    DELTA_X = 1.0 / max(GRID_SIZE)  # Spatial step
    DELTA_T = 1e-3  # Temporal step
    LAMBDA_EIGEN = 2.72  # Eigenvalue for lattice stability
    
    # Physical constants
    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    C = 2.99792458e8  # Speed of light (m/s)
    ADS_RADIUS = 1.0  # AdS radius (m)
    
    # J^6 potential parameters
    KAPPA_J6 = 1.0  # J^6 coupling constant
    KAPPA_J6_EFF = 1e-33  # Effective J^6 coupling
    J6_SCALE = 1e-30  # J^6 scaling factor for boundary terms
    EPSILON = 1e-15  # Small constant for numerical stability
    RESONANCE_FREQUENCY = 1e6  # Resonance frequency (Hz)
    J6_PARAM_RANGES = {
        'kappa_j6': [0.5, 1.0, 2.0],
        'kappa_j6_eff': [1e-34, 1e-33, 1e-32],
        'j6_scaling_factor': [1e-31, 1e-30, 1e-29],
        'epsilon': [1e-16, 1e-15, 1e-14],
        'resonance_frequency': [0.5e6, 1e6, 2e6]
    }
    
    # CTC parameters
    BETA = 0.1  # CTC modulation factor
    K = 0.1  # Wave number for scalar field
    WORMHOLE_NODES = [0.33333333326] * 6  # Default wormhole node positions
    
    # Quantum circuit parameters
    NUM_QUBITS = 6  # Number of qubits
    SAMPLE_RATE = 44100  # Audio sample rate (Hz)
    
    # Three-body simulation parameters
    THREE_BODY_MASSES = [1e30, 1e30, 1e30]  # Default masses (kg)
    THREE_BODY_POSITIONS = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0])
    ]  # Default positions (m)
    THREE_BODY_VELOCITIES = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1e-3, 0.0]),
        np.array([0.0, 0.0, 1e-3])
    ]  # Default velocities (m/s)
    
    # AdS boundary parameters
    BOUNDARY_FACTOR_RANGE = [0.8, 0.9, 1.0]  # Range for boundary factor sweeps
    
    def __init__(self):
        """Initialize constants and log configuration."""
        try:
            logger.info("Constants initialized: GRID_SIZE=%s, J6_SCALE=%.2e, EPSILON=%.2e, SAMPLE_RATE=%d", 
                        self.GRID_SIZE, self.J6_SCALE, self.EPSILON, self.SAMPLE_RATE)
            logger.debug("Three-body defaults: masses=%s, positions=%s, velocities=%s", 
                         self.THREE_BODY_MASSES, self.THREE_BODY_POSITIONS, self.THREE_BODY_VELOCITIES)
        except Exception as e:
            logger.error("Constants initialization failed: %s", e)
            raise
