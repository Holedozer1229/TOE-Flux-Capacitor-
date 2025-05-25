import pytest
import numpy as np
from src.sphinx_os.harmonic_generator import HarmonicGenerator

def test_harmonics():
    harmonic_gen = HarmonicGenerator(44100, 1.0)
    phi = np.array([1.0])
    j4 = np.array([0.0])
    psi = np.array([1.0])
    ricci_scalar = np.array([1.0])
    harmonics = harmonic_gen.generate_harmonics(phi, j4, psi, ricci_scalar)
    assert np.all(np.isfinite(harmonics)), "Non-finite harmonics detected"
    assert np.all(np.abs(harmonics) <= 1.0), "Harmonics exceed amplitude bounds"
    print(f"Harmonics mean: {np.mean(harmonics):.6f}, std: {np.std(harmonics):.6f}")
