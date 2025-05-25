import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_fft(signal: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Plot FFT of a signal."""
    try:
        freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)
        spectrum = np.abs(np.fft.fft(signal))
        plt.figure(figsize=(10, 6))
        plt.plot(freqs[:len(freqs)//2], spectrum[:len(spectrum)//2])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("FFT of Signal")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("FFT plot saved to %s", output_path)
    except Exception as e:
        logger.error("FFT plotting failed: %s", e)
        raise

def plot_entanglement_entropy(entropy_history: list, output_path: str) -> None:
    """Plot entanglement entropy history."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(entropy_history)
        plt.xlabel("Step")
        plt.ylabel("Entropy")
        plt.title("Entanglement Entropy Over Time")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("Entropy plot saved to %s", output_path)
    except Exception as e:
        logger.error("Entropy plotting failed: %s", e)
        raise

def plot_mobius_spiral(t: np.ndarray, r: float, n: float, m_shift: float, output_path: str) -> None:
    """Plot Möbius spiral for CTC visualization."""
    try:
        x = (r + t * np.cos(n * t + m_shift)) * np.cos(t)
        y = (r + t * np.cos(n * t + m_shift)) * np.sin(t)
        z = t * np.sin(n * t + m_shift)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Möbius Spiral")
        plt.savefig(output_path)
        plt.close()
        logger.info("Möbius spiral plot saved to %s", output_path)
    except Exception as e:
        logger.error("Möbius spiral plotting failed: %s", e)
        raise

def plot_tetrahedron(a: float, b: float, c: float, n_points: int, m_shift: float, output_path: str) -> None:
    """Plot tetrahedral structure."""
    try:
        t = np.linspace(0, 2 * np.pi, n_points)
        x = a * np.cos(t + m_shift)
        y = b * np.sin(t + m_shift)
        z = c * np.sin(t + m_shift)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Tetrahedral Structure")
        plt.savefig(output_path)
        plt.close()
        logger.info("Tetrahedron plot saved to %s", output_path)
    except Exception as e:
        logger.error("Tetrahedron plotting failed: %s", e)
        raise

def plot_j6_validation(param_values: np.ndarray, metric_values: np.ndarray, param_name: str, 
                       metric_name: str, output_path: str) -> None:
    """Plot J^6 parameter validation."""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(param_values, metric_values, c='blue', alpha=0.5)
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. {param_name}")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("J^6 validation plot saved to %s", output_path)
    except Exception as e:
        logger.error("J^6 validation plotting failed: %s", e)
        raise

def plot_rio_validation(rio_values: np.ndarray, metric_values: np.ndarray, metric_name: str, 
                        output_path: str, boundary_factor: float = 1.0) -> None:
    """Plot Rio Ricci scalar validation with boundary factor."""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(rio_values, metric_values, c='orange', alpha=0.5)
        plt.xlabel("Rio Ricci Scalar (Boundary Factor: %.2f)" % boundary_factor)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. Rio Ricci Scalar")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("Rio validation plot saved to %s with boundary_factor=%.2f", output_path, boundary_factor)
    except Exception as e:
        logger.error("Rio validation plotting failed: %s", e)
        raise

def plot_graviton_field(graviton_field: np.ndarray, output_path: str) -> None:
    """Plot graviton field trace with non-linear J^6 effects."""
    try:
        trace = np.trace(graviton_field, axis1=-2, axis2=-1)
        slice_2d = trace[:, :, trace.shape[2]//2, trace.shape[3]//2, 0, 0]
        plt.figure(figsize=(8, 6))
        plt.imshow(slice_2d, cmap='coolwarm')
        plt.colorbar(label='Graviton Trace')
        plt.title("Graviton Field Trace Slice")
        plt.savefig(output_path)
        plt.close()
        logger.info("Graviton field plot saved to %s", output_path)
    except Exception as e:
        logger.error("Graviton field plotting failed: %s", e)
        raise

def plot_nonlinear_j6_metrics(nonlinear_values: np.ndarray, metric_values: np.ndarray, metric_name: str, 
                              output_path: str, nonlinear_name: str) -> None:
    """Plot non-linear J^6 metrics vs. entanglement metrics."""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(nonlinear_values, metric_values, c='purple', alpha=0.5)
        plt.xlabel(nonlinear_name)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. {nonlinear_name}")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("Non-linear J^6 metrics plot saved to %s", output_path)
    except Exception as e:
        logger.error("Non-linear J^6 metrics plotting failed: %s", e)
        raise
