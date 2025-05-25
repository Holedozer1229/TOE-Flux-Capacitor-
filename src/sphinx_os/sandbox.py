import numpy as np
import logging
from .anubis_core import UnifiedSpacetimeSimulator
from .harmonic_generator import HarmonicGenerator
from .constants import Constants

logger = logging.getLogger(__name__)

def run_sandbox_simulation(grid_size: tuple = (3, 3, 3, 3, 2, 2), duration: float = 1.0):
    """Run an experimental simulation with non-linear J^6 coupling."""
    try:
        simulator = UnifiedSpacetimeSimulator(grid_size, Constants.LAMBDA_EIGEN)
        harmonic_gen = HarmonicGenerator(Constants.SAMPLE_RATE, Constants.KAPPA_J6)
        simulator.initialize_tetrahedral_lattice()
        
        num_samples = int(Constants.SAMPLE_RATE * duration)
        audio_output = np.zeros(num_samples)
        grid_center = np.array(grid_size) / 2
        
        for i, t in enumerate(np.linspace(0, duration, num_samples)):
            phi = simulator.compute_scalar_field(grid_center, t)
            j4 = np.mean(simulator.grid)
            psi = simulator.psi
            ricci_scalar = simulator.ricci_scalar
            graviton_field = simulator.sphinx_os.graviton_field if hasattr(simulator.sphinx_os, 'graviton_field') else np.zeros(grid_size + (6, 6))
            audio_output[i] = harmonic_gen.generate_harmonics(phi, j4, psi, ricci_scalar, graviton_field)
        
        peaks = harmonic_gen.analyze_harmonics(audio_output, "results/sandbox_harmonics.png")
        entropy = simulator.compute_entanglement_entropy()
        j6_stats = harmonic_gen.analyze_j6_potential(phi, j4, psi, ricci_scalar, graviton_field, boundary_factor=1.0)
        
        logger.info("Sandbox simulation completed: peaks=%s, entropy=%.6f, rio_mean=%.6f, graviton_nonlinear=%.6e", 
                    peaks, entropy, j6_stats['rio_mean'], j6_stats['graviton_nonlinear'])
        return audio_output
    except Exception as e:
        logger.error("Sandbox simulation failed: %s", e)
        raise
