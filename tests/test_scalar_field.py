import pytest
import numpy as np
from src.sphinx_os.anubis_core import UnifiedSpacetimeSimulator
from src.sphinx_os.constants import Constants

def test_scalar_field():
    simulator = UnifiedSpacetimeSimulator((3, 3, 3, 3, 2, 2), 2.72)
    r = np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0])
    t = 0.0
    phi = simulator.compute_scalar_field(r, t)
    assert np.isfinite(phi), f"Nugget scalar field value is non-finite: {phi}"
    assert abs(phi) < 1e5, f"Nugget scalar field value too large: {phi}"
    print(f"Nugget scalar field at r={r}, t={t}: {phi:.6f}")
