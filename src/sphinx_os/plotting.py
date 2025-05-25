import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

logger = logging.getLogger(__name__)

def plot_fft(audio_output: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Plot FFT of audio output."""
    try:
        fft = np.fft.fft(audio_output)
        freqs = np.fft.fftfreq(len(fft), 1 / sample_rate)
        plt.figure(figsize=(10, 6))
        plt.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(freqs)//2])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("FFT of Audio Output")
        plt.savefig(output_path)
        plt.close()
        logger.info("FFT plot saved to %s", output_path)
    except Exception as e:
        logger.error("FFT plotting failed: %s", e)
        raise

def plot_entanglement_entropy(entropy: float, output_path: str) -> None:
    """Plot entanglement entropy."""
    try:
        plt.figure(figsize=(8, 6))
        plt.plot([entropy], 'o')
        plt.xlabel("Iteration")
        plt.ylabel("Entanglement Entropy")
        plt.title("Entanglement Entropy")
        plt.savefig(output_path)
        plt.close()
        logger.info("Entropy plot saved to %s", output_path)
    except Exception as e:
        logger.error("Entropy plotting failed: %s", e)
        raise

def plot_mobius_spiral(t: np.ndarray, r: float, n: float, m_shift_amplitude: float, output_path: str) -> None:
    """Plot Möbius spiral trajectory for CTC visualization."""
    try:
        def m_shift(t, phi):
            return m_shift_amplitude * (1 + 0.1 * np.sin(t))
        
        theta = t * np.pi
        phi = n * theta / 2
        x = r * np.cos(theta) * np.sin(phi) * m_shift(theta, phi)
        y = r * np.sin(theta) * np.sin(phi) * m_shift(theta, phi)
        z = r * np.cos(phi) * m_shift(theta, phi)
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, color='r')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Möbius Spiral Trajectory")
        plt.savefig(output_path)
        plt.close()
        logger.info("Möbius spiral plot saved to %s", output_path)
    except Exception as e:
        logger.error("Möbius spiral plotting failed: %s", e)
        raise

def plot_tetrahedron(a: float, b: float, c: float, n_points: int, m_shift_amplitude: float, output_path: str) -> None:
    """Plot tetrahedral geometry with m_shift for CTC visualization."""
    try:
        def m_shift(u, v):
            return m_shift_amplitude * (1 + 0.1 * np.cos(u + v))
        
        u = np.linspace(-np.pi, np.pi, n_points)
        v = np.linspace(-np.pi / 2, np.pi / 2, n_points)
        u, v = np.meshgrid(u, v)
        
        face1_x = a * np.cosh(u) * np.cos(v) * m_shift(u, v)
        face1_y = b * np.cosh(u) * np.sin(v) * m_shift(u, v)
        face1_z = c * np.sinh(u) * m_shift(u, v)
        
        face2_x = -a * np.cosh(u) * np.cos(v) * m_shift(u, v)
        face2_y = -b * np.cosh(u) * np.sin(v) * m_shift(u, v)
        face2_z = c * np.sinh(u) * m_shift(u, v)
        
        face3_x = a * np.cosh(u) * np.cos(v) * m_shift(u, v)
        face3_y = -b * np.cosh(u) * np.sin(v) * m_shift(u, v)
        face3_z = -c * np.sinh(u) * m_shift(u, v)
        
        face4_x = -a * np.cosh(u) * np.cos(v) * m_shift(u, v)
        face4_y = b * np.cosh(u) * np.sin(v) * m_shift(u, v)
        face4_z = -c * np.sinh(u) * m_shift(u, v)
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(face1_x, face1_y, face1_z, color='b', alpha=0.5)
        ax.plot_surface(face2_x, face2_y, face2_z, color='g', alpha=0.5)
        ax.plot_surface(face3_x, face3_y, face3_z, color='r', alpha=0.5)
        ax.plot_surface(face4_x, face4_y, face4_z, color='y', alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Tetrahedral Geometry")
        plt.savefig(output_path)
        plt.close()
        logger.info("Tetrahedron plot saved to %s", output_path)
    except Exception as e:
        logger.error("Tetrahedron plotting failed: %s", e)
        raise

def plot_j6_validation(param_values: np.ndarray, metrics: np.ndarray, param_name: str, metric_name: str, output_path: str) -> None:
    """Plot J^6 validation metrics vs. parameter values."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metrics, 'o-')
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

def plot_rio_validation(rio_values: np.ndarray, metrics: np.ndarray, metric_name: str, output_path: str, boundary_factor: float = 1.0) -> None:
    """Plot metrics vs. Rio Ricci scalar values with AdS boundary factor."""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(rio_values, metrics, c='blue', alpha=0.5)
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
    """Visualize a 2D slice of the graviton field trace."""
    try:
        graviton_trace = np.trace(graviton_field, axis1=-2, axis2=-1)
        slice_2d = graviton_trace[:, :, graviton_trace.shape[2]//2, graviton_trace.shape[3]//2, 0, 0]
        plt.figure(figsize=(8, 6))
        plt.imshow(slice_2d, cmap='magma')
        plt.colorbar(label='Graviton Field Trace')
        plt.title("Graviton Field Trace Slice (Non-Linear J^6)")
        plt.savefig(output_path)
        plt.close()
        logger.info("Graviton field visualization saved to %s", output_path)
    except Exception as e:
        logger.error("Graviton field visualization failed: %s", e)
        raise

def plot_nonlinear_j6_metrics(nonlinear_values: np.ndarray, metrics: np.ndarray, metric_name: str, output_path: str, label: str = "Non-Linear J^6 Term") -> None:
    """Plot metrics vs. non-linear J^6 graviton or boundary terms."""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(nonlinear_values, metrics, c='purple', alpha=0.5)
        plt.xlabel(label)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. {label}")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("Non-linear J^6 metrics plot saved to %s", output_path)
    except Exception as e:
        logger.error("Non-linear J^6 metrics plotting failed: %s", e)
        raise
