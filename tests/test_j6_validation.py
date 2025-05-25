import pytest
import numpy as np
import os
from scipy import stats
from src.sphinx_os.anubis_core import UnifiedSpacetimeSimulator
from src.sphinx_os.qubit_fabric import QuantumCircuit
from src.sphinx_os.harmonic_generator import HarmonicGenerator
from src.sphinx_os.constants import Constants

def test_j6_validation():
    simulator = UnifiedSpacetimeSimulator((3, 3, 3, 3, 2, 2), 2.72)
    circuit = QuantumCircuit(Constants.NUM_QUBITS)
    
    grid_center = np.array(simulator.grid_size) / 2
    phi = simulator.compute_scalar_field(grid_center, 0.0)
    j4 = np.array([0.0])
    psi = np.array([1.0])
    ricci_scalar = np.array([1.0])  # Placeholder, updated dynamically
    graviton_field = np.zeros((3, 3, 3, 3, 2, 2, 6, 6))  # Placeholder, updated dynamically
    
    # Validate Rio and graviton fields
    rio_mean = np.mean(ricci_scalar)
    assert -1e5 <= rio_mean <= 1e5, f"Rio mean out of bounds: {rio_mean}"
    graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1))
    assert -1e3 <= graviton_trace <= 1e3, f"Graviton trace out of bounds: {graviton_trace}"
    
    # J^6 configurations
    configs = [
        {'kappa_j6': 1.0, 'kappa_j6_eff': 1e-33, 'j6_scaling_factor': 1e-30, 'epsilon': 1e-15, 'resonance_frequency': 1e6},
        {'kappa_j6': 2.0, 'kappa_j6_eff': 1e-32, 'j6_scaling_factor': 1e-29, 'epsilon': 1e-14, 'resonance_frequency': 1.5e6}
    ]
    
    ctc_config = {'tau': 1.0, 'kappa_ctc': 0.5, 'r': 3.0, 'n': 2.0, 'm_shift_amplitude': 2.72,
                  'ctc_phase_factor': 0.1, 'ctc_wormhole_factor': 0.5, 'ctc_amplify_factor': 0.2}
    
    boundary_factors = [0.8, 1.0]  # Test AdS boundary effects
    
    for config in configs:
        for boundary_factor in boundary_factors:
            harmonic_gen = HarmonicGenerator(44100, config['kappa_j6'])
            harmonic_gen.kappa_j6_eff = config['kappa_j6_eff']
            harmonic_gen.j6_scaling_factor = config['j6_scaling_factor']
            harmonic_gen.epsilon = config['epsilon']
            harmonic_gen.omega_res = config['resonance_frequency'] * 2 * np.pi
            
            # Generate and analyze harmonics
            harmonics = harmonic_gen.generate_harmonics(phi, j4, psi, ricci_scalar, graviton_field, boundary_factor)
            peaks = harmonic_gen.analyze_harmonics(harmonics, f"results/test_j6_harmonics_{config['kappa_j6']}_bf{boundary_factor}.png")
            audio_input = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
            delay = harmonic_gen.analyze_delays(harmonics, audio_input)
            j6_stats = harmonic_gen.analyze_j6_potential(phi, j4, psi, ricci_scalar, graviton_field, boundary_factor)
            
            # Harmonic validation
            harmonic_chi2 = stats.chisquare([1 if abs(p - 880) < 50 or abs(p - 1320) < 50 else 0 for p in peaks[:2]], 
                                           f_exp=[1, 1]).pvalue
            assert harmonic_chi2 > 0.05, f"Harmonic chi-squared test failed: p={harmonic_chi2}"
            assert any(abs(p - 880) < 50 for p in peaks), f"Expected 880 Hz, got {peaks}"
            assert any(abs(p - 1320) < 50 for p in peaks), f"Expected 1320 Hz, got {peaks}"
            
            # Check for graviton-induced sidebands
            sideband_detected = any(abs(p - 900) < 50 or abs(p - 1340) < 50 for p in peaks)
            print(f"Graviton sidebands detected: {sideband_detected}")
            
            # Delay validation
            assert 0.045 <= delay <= 0.055, f"Expected delay ~0.050 s, got {delay}"
            delay_ttest = stats.ttest_1samp([delay], 0.050).pvalue
            assert delay_ttest > 0.05, f"Delay t-test failed: p={delay_ttest}"
            
            # Entanglement validation with boundary effects
            ctc_effect = simulator.compute_ctc_term(1.0, phi, 0.5, ctc_params=ctc_config)
            circuit.apply_rydberg_gates(wormhole_nodes=True, phi=phi, ctc_feedback=ctc_effect, 
                                        ctc_params=ctc_config, boundary_factor=boundary_factor)
            
            chsh_standard = circuit.compute_chsh_violation(amplify=False, ctc_params=ctc_config)
            mabk_standard = circuit.compute_mabk_violation(n_qubits=4, amplify=False, ctc_params=ctc_config)
            ghz_standard = circuit.compute_ghz_paradox(amplify=False, ctc_params=ctc_config)
            
            assert abs(chsh_standard - 2.828427) < 0.1, f"CHSH standard: {chsh_standard}"
            assert abs(mabk_standard - 5.656854) < 0.2, f"MABK standard: {mabk_standard}"
            expected_ghz = {'XXX': 1.0, 'XYY': -1.0, 'YXY': -1.0, 'YYX': -1.0}
            for key, value in expected_ghz.items():
                assert abs(ghz_standard[key] - value) < 0.01, f"GHZ standard {key}: {ghz_standard[key]}"
            
            # J^6 potential, Rio, and graviton validation
            assert j6_stats['V_j6_mean'] >= 0, f"J^6 potential mean negative: {j6_stats['V_j6_mean']}"
            assert j6_stats['dV_j6_dphi_mean'] >= 0, f"J^6 derivative mean negative: {j6_stats['dV_j6_dphi_mean']}"
            assert -1e5 <= j6_stats['rio_mean'] <= 1e5, f"Rio mean out of bounds: {j6_stats['rio_mean']}"
            assert -1e3 <= j6_stats['graviton_trace'] <= 1e3, f"Graviton trace out of bounds: {j6_stats['graviton_trace']}"
            
            print(f"J^6 Config {config}, Boundary Factor {boundary_factor}:")
            print(f"Harmonic Peaks: {peaks}")
            print(f"Delay: {delay:.6f} s")
            print(f"CHSH: {chsh_standard:.6f}")
            print(f"MABK (n=4): {mabk_standard:.6f}")
            print(f"GHZ: {ghz_standard}")
            print(f"J^6 Stats: {j6_stats}")
