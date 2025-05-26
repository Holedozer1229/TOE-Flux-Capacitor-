import numpy as np
import pytest
import glob
import os
from scipy.signal import find_peaks
from main import FluxCapacitor
from sphinx_os.constants.config import Constants
from sphinx_os.visualization.visualize import visualize_trajectories

# Test configuration
SAMPLE_RATE = Constants.SAMPLE_RATE  # 44100 Hz
DURATION = 1.0  # Short duration for testing
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_three_body_simulation():
    """Test three-body simulation outputs and metrics."""
    # Initialize FluxCapacitor
    flux = FluxCapacitor()
    
    # Run three-body simulation with default parameters
    audio_output = flux.run_three_body_simulation(
        body_positions=Constants.THREE_BODY_POSITIONS,
        body_masses=Constants.THREE_BODY_MASSES,
        velocities=Constants.THREE_BODY_VELOCITIES,
        duration=DURATION
    )
    
    # Check audio output
    assert len(audio_output) == int(SAMPLE_RATE * DURATION), "Audio output length mismatch"
    assert np.all(np.abs(audio_output) <= 1.0), "Audio output exceeds clipping range"
    
    # Find latest trajectory file
    traj_files = glob.glob(os.path.join(OUTPUT_DIR, "trajectories_*.npy"))
    assert traj_files, "No trajectory files generated"
    traj_file = max(traj_files, key=lambda x: x)
    
    # Load and verify trajectories
    trajectories = np.load(traj_file)
    assert trajectories.shape[0] == 3, "Expected 3 body trajectories"
    assert trajectories.shape[2] == 3, "Expected 3D positions"
    
    # Generate and verify trajectory plot
    plot_path = os.path.join(OUTPUT_DIR, "trajectories_test.png")
    visualize_trajectories(trajectories, plot_path)
    assert os.path.exists(plot_path), "Trajectory plot not generated"
    
    # Analyze harmonics
    freqs = np.fft.fftfreq(len(audio_output), 1 / SAMPLE_RATE)
    spectrum = np.abs(np.fft.fft(audio_output))
    peaks, _ = find_peaks(spectrum[:len(spectrum)//2], height=np.max(spectrum)/10)
    peak_freqs = freqs[peaks]
    peak_freqs = peak_freqs[peak_freqs > 0]
    
    # Check for expected harmonics (880 Hz, 1320 Hz, sidebands ~900 Hz, ~1340 Hz)
    main_harmonics = [880, 1320]
    sidebands = [900, 1340]
    found_harmonics = any(abs(f - h) < 50 for f in peak_freqs for h in main_harmonics)
    found_sidebands = any(abs(f - s) < 50 for f in peak_freqs for s in sidebands)
    assert found_harmonics, f"Expected harmonics {main_harmonics} Hz, found {peak_freqs}"
    assert found_sidebands, f"Expected sidebands {sidebands} Hz, found {peak_freqs}"
    
    # Check log file for delays
    log_files = glob.glob(os.path.join(OUTPUT_DIR, "flux_capacitor.log"))
    assert log_files, "No log file generated"
    log_file = max(log_files, key=lambda x: x)
    with open(log_file, "r") as f:
        log_content = f.read()
        assert "delay=0.04" in log_content or "delay=0.05" in log_content, "Expected delay ~50 ms not found"

def test_entanglement_metrics():
    """Test entanglement metrics (CHSH |S| ~3.2)."""
    flux = FluxCapacitor()
    flux.initialize_spin_network()
    chsh_amplified = flux.circuit.compute_chsh_violation(amplify=True)
    assert 3.0 <= chsh_amplified <= 3.4, f"Expected CHSH |S| ~3.2, got {chsh_amplified}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
