import pytest
import numpy as np
from src.sphinx_os.anubis_core import UnifiedSpacetimeSimulator

def test_entanglement_entropy():
    simulator = UnifiedSpacetimeSimulator((3, 3, 3, 3, 2, 2), 2.72)
    simulator.initialize_tetrahedral_lattice()
    entropy = simulator.compute_entanglement_entropy()
    assert entropy >= 0, f"Entanglement entropy is negative: {entropy}"
    assert np.isfinite(entropy), f"Entanglement entropy is non-finite: {entropy}"
    print(f"Entanglement entropy: {entropy:.6f}")
