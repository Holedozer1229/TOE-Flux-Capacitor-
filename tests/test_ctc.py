import pytest
import numpy as np
from src.sphinx_os.anubis_core import UnifiedSpacetimeSimulator

def test_ctc_term():
    simulator = UnifiedSpacetimeSimulator((3, 3, 3, 3, 2, 2), 2.72)
    simulator.initialize_tetrahedral_lattice()
    tau = 1.0
    phi = 1.0
    j6_modulation = 0.5
    ctc_term = simulator.compute_ctc_term()
    assert np.isfinite(ctc_term), f"CTC term is non-finite: {ctc_term}"
    assert -1.0 <= ctc_term <= 1.0, f"CTC term out of bounds: {ctc_term}"
    print(f"CTC term: {ctc_term:.6f}")
