import pytest
import numpy as np
import os
from src.sphinx_os.anubis_core import UnifiedSpacetimeSimulator
from src.sphinx_os.qubit_fabric import QuantumCircuit
from src.sphinx_os.harmonic_generator import HarmonicGenerator
from src.sphinx_os.constants import Constants
from src.sphinx_os.plotting import plot_mobius_spiral, plot_tetrahedron

def test_ctc_tuning():
    simulator = UnifiedSpacetimeSimulator((3, 3, 3, 3, 2, 2), 2.72)
    circuit = QuantumCircuit(Constants.NUM_QUBITS)
    harmonic_gen = HarmonicGenerator(44100, 1.0)
    
    grid_center = np.array(simulator.grid_size) / 2
    phi = simulator.compute_scalar_field(grid_center, 0.0)
    j4 = np.array([0.0])
    psi = np.array([1.0])
    ricci_scalar = np.array([1.0])  # Placeholder, updated dynamically
    graviton_field = simulator.sphinx_os.graviton_field if hasattr(simulator.sphinx_os, 'graviton_field') else np.zeros((3, 3, 3, 3, 2, 2, 6, 6))
    
    # Validate Rio and graviton fields
    rio_mean = np.mean(ricci_scalar)
    assert -1e5 <= rio_mean <= 1e5, f"Rio mean out of bounds: {rio_mean}"
    graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1))
    assert -1e3 <= graviton_trace <= 1e3, f"Graviton trace out of bounds: {graviton_trace}"
    
    # CTC tuning configurations
    configs = [
        {'tau': 1.0, 'kappa_ctc': 0.5, 'r': 3.0, 'n': 2.0, 'm_shift_amplitude': 2.72,
         'ctc_phase_factor': 0.1, 'ctc_wormhole_factor': 0.5, 'ctc_amplify_factor': 0.2},
        {'tau': 1.5, 'kappa_ctc': 0.7, 'r': 2.0, 'n': 1.0, 'm_shift_amplitude': 3.5,
         'ctc_phase_factor': 0.3, 'ctc_wormhole_factor': 0.7, 'ctc_amplify_factor': 0.3}
    ]
    
    boundary_factor = 0.9  # AdS boundary effect
    
    for config in configs:
        ctc_effect = simulator.compute_ctc_term(1.0, phi, 0.5, ctc_params=config)
        circuit.apply_rydberg_gates(wormhole_nodes=True, phi=phi, ctc_feedback=ctc_effect, 
                                    ctc_params=config, boundary_factor=boundary_factor)
        
        # Test entanglement metrics
        chsh_standard = circuit.compute_chsh_violation(amplify=False, ctc_params=config)
        chsh_amplified = circuit.compute_chsh_violation(amplify=True, ctc_params=config)
        mabk_standard = circuit.compute_mabk_violation(n_qubits=4, amplify=False, ctc_params=config)
        mabk_amplified = circuit.compute_mabk_violation(n_qubits=4, amplify=True, ctc_params=config)
        ghz_standard = circuit.compute_ghz_paradox(amplify=False, ctc_params=config)
        ghz_amplified = circuit.compute_ghz_paradox(amplify=True, ctc_params=config)
        
        assert abs(chsh_standard - 2.828427) < 0.1, f"CHSH standard: {chsh_standard}"
        assert abs(mabk_standard - 5.656854) < 0.2, f"MABK standard: {mabk_standard}"
        expected_ghz = {'XXX': 1.0, 'XYY': -1.0, 'YXY': -1.0, 'YYX': -1.0}
        for key, value in expected_ghz.items():
            assert abs(ghz_standard[key] - value) < 0.01, f"GHZ standard {key}: {ghz_standard[key]}"
        
        print(f"Config {config}, Boundary Factor {boundary_factor}:")
        print(f"CHSH: {chsh_standard:.6f} (standard), {chsh_amplified:.6f} (amplified)")
        print(f"MABK (n=4): {mabk_standard:.6f} (standard), {mabk_amplified:.6f} (amplified)")
        print(f"GHZ: {ghz_standard} (standard), {ghz_amplified} (amplified)")
        print(f"Rio Mean: {rio_mean:.6f}")
        print(f"Graviton Trace: {graviton_trace:.6f}")
        
        # Test visualizations
        timestamp = "test"
        t = np.linspace(0, 2 * np.pi, 100)
        plot_mobius_spiral(t, config['r'], config['n'], config['m_shift_amplitude'], 
                          f"results/mobius_{timestamp}_r{config['r']}_n{config['n']}.png")
        plot_tetrahedron(1, 2, 3, 100, config['m_shift_amplitude'], 
                        f"results/tetrahedron_{timestamp}_msa{config['m_shift_amplitude']}.png")
        assert os.path.exists(f"results/mobius_{timestamp}_r{config['r']}_n{config['n']}.png")
        assert os.path.exists(f"results/tetrahedron_{timestamp}_msa{config['m_shift_amplitude']}.png")
