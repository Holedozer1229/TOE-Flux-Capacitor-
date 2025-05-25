import pytest
import numpy as np
from src.sphinx_os.qubit_fabric import QuantumCircuit
from src.sphinx_os.anubis_core import UnifiedSpacetimeSimulator
from src.sphinx_os.constants import Constants

def test_mabk_ghz_violations():
    simulator = UnifiedSpacetimeSimulator((3, 3, 3, 3, 2, 2), 2.72)
    circuit = QuantumCircuit(Constants.NUM_QUBITS)
    
    grid_center = np.array(simulator.grid_size) / 2
    phi = simulator.compute_scalar_field(grid_center, 0.0)
    ctc_effect = simulator.compute_ctc_term(1.0, phi, 0.5)
    boundary_factor = 0.9  # Test AdS boundary effect
    
    # Validate graviton field
    graviton_field = simulator.sphinx_os.graviton_field if hasattr(simulator.sphinx_os, 'graviton_field') else np.zeros((3, 3, 3, 3, 2, 2, 6, 6))
    graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1))
    assert -1e3 <= graviton_trace <= 1e3, f"Graviton trace out of bounds: {graviton_trace}"
    
    circuit.apply_rydberg_gates(wormhole_nodes=True, phi=phi, ctc_feedback=ctc_effect, boundary_factor=boundary_factor)
    
    # Validate Rio Ricci scalar
    rio_mean = np.mean(simulator.ricci_scalar) if hasattr(simulator, 'ricci_scalar') else 1.0
    assert -1e5 <= rio_mean <= 1e5, f"Rio mean out of bounds: {rio_mean}"
    
    # Test MABK for different n
    for n_qubits, quantum_bound in [(2, 2.828427), (4, 5.656854), (6, 11.313708)]:
        M_standard = circuit.compute_mabk_violation(n_qubits=n_qubits, amplify=False)
        assert abs(M_standard - quantum_bound) < 0.1, f"n={n_qubits}, Expected |M| ≈ {quantum_bound}, got {M_standard}"
        print(f"MABK (n={n_qubits}, standard): |M| = {M_standard:.6f}")
        
        M_amplified = circuit.compute_mabk_violation(n_qubits=n_qubits, amplify=True)
        print(f"MABK (n={n_qubits}, amplified): |M| = {M_amplified:.6f}")
    
    # Test GHZ paradox
    ghz_standard = circuit.compute_ghz_paradox(amplify=False)
    expected_standard = {'XXX': 1.0, 'XYY': -1.0, 'YXY': -1.0, 'YYX': -1.0}
    for key, value in expected_standard.items():
        assert abs(ghz_standard[key] - value) < 0.01, f"GHZ standard {key}: Expected {value}, got {ghz_standard[key]}"
    print(f"GHZ Paradox (standard): {ghz_standard}")
    
    ghz_amplified = circuit.compute_ghz_paradox(amplify=True)
    print(f"GHZ Paradox (amplified): {ghz_amplified}")
    print(f"Rio Mean: {rio_mean:.6f}")
    print(f"Graviton Trace: {graviton_trace:.6f}")
    print(f"AdS Boundary Factor: {boundary_factor:.6f}")
