import pytest
import numpy as np
from src.sphinx_os.anubis_core import UnifiedSpacetimeSimulator
from src.sphinx_os.physics.geometry.metric import compute_quantum_metric
from src.sphinx_os.physics.geometry.curvature import compute_riemann_tensor, compute_curvature
from src.sphinx_os.physics.fields.graviton import initialize_graviton_field
from src.sphinx_os.constants import Constants

def test_geometry():
    grid_size = (3, 3, 3, 3, 2, 2)
    simulator = UnifiedSpacetimeSimulator(grid_size, 2.72)
    nugget_field = np.ones(grid_size, dtype=np.complex128)
    temporal_entanglement = np.ones(grid_size, dtype=np.complex128) / np.sqrt(np.prod(grid_size))
    graviton_field = initialize_graviton_field(grid_size, [1e-12, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15])
    deltas = [1e-12, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15]
    
    # Compute metric with AdS boundary conditions
    metric, inverse_metric = compute_quantum_metric(simulator.lattice, nugget_field, temporal_entanglement, grid_size)
    assert np.all(np.isfinite(metric)), "Non-finite metric values"
    assert np.all(np.isfinite(inverse_metric)), "Non-finite inverse metric values"
    
    # Compute Riemann tensor with graviton effects
    riemann_tensor = compute_riemann_tensor(metric, inverse_metric, deltas, grid_size, graviton_field)
    assert np.all(np.isfinite(riemann_tensor)), "Non-finite Riemann tensor values"
    
    # Compute Ricci tensor and Rio Ricci scalar
    ricci_tensor, ricci_scalar = compute_curvature(riemann_tensor, inverse_metric, grid_size)
    assert np.all(np.isfinite(ricci_tensor)), "Non-finite Ricci tensor values"
    assert np.all(np.isfinite(ricci_scalar)), "Non-finite Rio Ricci scalar values"
    
    # Verify Rio Ricci scalar properties
    rio_mean = np.mean(ricci_scalar)
    rio_std = np.std(ricci_scalar)
    assert -1e5 <= rio_mean <= 1e5, f"Rio Ricci scalar mean out of bounds: {rio_mean}"
    assert rio_std >= 0, f"Negative Rio Ricci scalar std: {rio_std}"
    
    # Validate graviton field properties
    graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1))
    assert -1e3 <= graviton_trace <= 1e3, f"Graviton trace out of bounds: {graviton_trace}"
    
    # Validate barycentric interpolation and AdS boundary effects
    idx = (1, 1, 1, 1, 1, 1)
    bary_weights = simulator.lattice.get_barycentric_weights(idx)
    assert np.all(bary_weights >= 0), "Negative barycentric weights"
    assert abs(np.sum(bary_weights) - 1.0) < 1e-10, "Barycentric weights not normalized"
    
    # Check AdS boundary factor influence
    boundary_factor = np.exp(-0.1 * np.sum(np.abs(np.array(idx) - np.array(grid_size)/2)))
    assert 0.8 <= boundary_factor <= 1.0, f"Boundary factor out of range: {boundary_factor}"
    
    print(f"Rio Ricci Scalar: mean={rio_mean:.6f}, std={rio_std:.6f}")
    print(f"Graviton Trace: {graviton_trace:.6f}")
    print(f"Barycentric Weights at {idx}: {bary_weights}")
    print(f"AdS Boundary Factor at {idx}: {boundary_factor:.6f}")
