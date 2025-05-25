import numpy as np
from .anubis_core import UnifiedSpacetimeSimulator
from .qubit_fabric import QuantumCircuit
from .harmonic_generator import HarmonicGenerator

class UnifiedTOE:
    """Orchestrates TOE simulation components with graviton and AdS boundary effects."""
    
    def __init__(self, simulator: UnifiedSpacetimeSimulator, circuit: QuantumCircuit, harmonic_gen: HarmonicGenerator):
        self.simulator = simulator
        self.circuit = circuit
        self.harmonic_gen = harmonic_gen
    
    def integrate_interactions(self, audio_input: float, t: float, tau: float, grid_center: np.ndarray,
                              boundary_factor: float = 1.0, psi_abs_sq: float = 1.0, j4_abs: float = 0.0) -> tuple:
        """Integrate scalar field, quantum, graviton, and harmonic interactions with non-linear J^6 and AdS boundary."""
        phi = self.simulator.compute_scalar_field(grid_center, t)
        graviton_field = self.simulator.sphinx_os.graviton_field if hasattr(self.simulator.sphinx_os, 'graviton_field') else np.zeros(self.simulator.grid_size + (6, 6))
        j6_modulation = self.harmonic_gen.generate_harmonics(
            phi, np.mean(self.simulator.grid), self.simulator.psi, self.simulator.ricci_scalar, graviton_field, boundary_factor
        )
        ctc_effect = self.simulator.compute_ctc_term(t, phi, j6_modulation)
        self.circuit.apply_rydberg_gates(wormhole_nodes=True, phi=phi, ctc_feedback=ctc_effect, boundary_factor=boundary_factor, psi_abs_sq=psi_abs_sq, j4_abs=j4_abs)
        harmonics = audio_input * j6_modulation
        logger.debug("Integrated interactions: harmonics=%.6f, ctc_effect=%.6f, boundary_factor=%.2f, psi_abs_sq=%.6f, j4_abs=%.6f", 
                     np.mean(harmonics), ctc_effect, boundary_factor, psi_abs_sq, j4_abs)
        return harmonics, ctc_effect
