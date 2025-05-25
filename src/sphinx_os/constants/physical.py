import numpy as np

class Constants:
    """Physical and simulation constants."""
    PLANCK_CONSTANT = 6.62607015e-34  # J·s
    SPEED_OF_LIGHT = 2.99792458e8  # m/s
    GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
    PLANCK_LENGTH = 1.616255e-35  # m
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    KAPPA_CTC = 0.5
    NUM_QUBITS = 64
    GRID_SIZE = (5, 5, 5, 5, 3, 3)
    LAMBDA_EIGEN = 2.72
    SAMPLE_RATE = 44100
    WORMHOLE_NODES = [(2, 2, 2, 2, 1, 1)]
    K = 1e-15  # Wave number scaling
    DELTA_X = 1e-15  # Spatial step
    DELTA_T = 1e-12  # Time step
    BETA = 0.1  # Scalar field parameter
