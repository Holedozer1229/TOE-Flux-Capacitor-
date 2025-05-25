import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.fft import fft, fftfreq
from ..utils.math_utils import compute_j6_potential
from ..constants import CONFIG

logger = logging.getLogger(__name__)

class HarmonicGenerator:
    """Generates audio harmonics with non-linear J^6-coupled graviton and AdS boundary effects."""
    
    def __init__(self, sample_rate: int, kappa_j6: float):
        self.sample_rate = sample_rate
        self.kappa_j6 = kappa_j6
        self.kappa_j6_eff = CONFIG["kappa_j6_eff"]
        self.j6_scaling_factor = CONFIG["j6_scaling_factor"]
        self.epsilon = CONFIG["epsilon"]
        self.omega_res = CONFIG["resonance_frequency"] * 2 * np.pi
    
    def generate_harmonics(self, phi: np.ndarray, j4: np.ndarray, psi: np.ndarray, 
                           ricci_scalar: np.ndarray, graviton_field: np.ndarray = None, 
                           boundary_factor: float = 1.0) -> np.ndarray:
        """Generate harmonics with non-linear J^6 coupling."""
        try:
            V_j6, _ = compute_j6_potential(
                phi, j4, psi, ricci_scalar, graviton_field=graviton_field,
                kappa_j6=self.kappa_j6,
                kappa_j6_eff=self.kappa_j6_eff,
                j6_scaling_factor=self.j6_scaling_factor,
                epsilon=self.epsilon,
                omega_res=self.omega_res,
                boundary_factor=boundary_factor
            )
            harmonics = np.sin(2 * np.pi * V_j6 / self.sample_rate)
            harmonics = np.clip(harmonics, -1.0, 1.0)
            logger.debug("Generated harmonics: mean=%.6f, std=%.6f", np.mean(harmonics), np.std(harmonics))
            return harmonics
        except Exception as e:
            logger.error("Harmonic generation failed: %s", e)
            raise
    
    def analyze_harmonics(self, audio_output: np.ndarray, output_path: str) -> list:
        """Analyze harmonic frequencies with high precision for graviton signatures."""
        try:
            N = len(audio_output)
            fft_result = fft(audio_output)
            freqs = fftfreq(N, 1 / self.sample_rate)
            positive_freqs = freqs[:N//2]
            positive_fft = np.abs(fft_result[:N//2])
            peak_indices = np.argsort(positive_fft)[-5:][::-1]
            peaks = positive_freqs[peak_indices]
            plt.figure(figsize=(10, 6))
            plt.plot(positive_freqs, positive_fft)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.title("FFT of Audio Output (Non-Linear J^6 Graviton-Modulated)")
            plt.savefig(output_path)
            plt.close()
            logger.info("Harmonic peaks detected: %s", peaks.tolist())
            return peaks.tolist()
        except Exception as e:
            logger.error("Harmonic analysis failed: %s", e)
            raise
    
    def analyze_delays(self, audio_output: np.ndarray, audio_input: np.ndarray) -> float:
        """Analyze temporal delays, targeting ~50 ms."""
        try:
            corr = np.correlate(audio_output, audio_input, mode='full')
            lag = np.argmax(corr) - len(audio_input) + 1
            delay = lag / self.sample_rate
            logger.debug("Computed delay: %.6f s", delay)
            return delay
        except Exception as e:
            logger.error("Delay analysis failed: %s", e)
            raise
    
    def analyze_j6_potential(self, phi: np.ndarray, j4: np.ndarray, psi: np.ndarray, 
                             ricci_scalar: np.ndarray, graviton_field: np.ndarray = None, 
                             boundary_factor: float = 1.0) -> dict:
        """Analyze J^6 potential statistics with non-linear graviton and boundary effects."""
        try:
            V_j6, dV_j6_dphi = compute_j6_potential(
                phi, j4, psi, ricci_scalar, graviton_field=graviton_field,
                kappa_j6=self.kappa_j6,
                kappa_j6_eff=self.kappa_j6_eff,
                j6_scaling_factor=self.j6_scaling_factor,
                epsilon=self.epsilon,
                omega_res=self.omega_res,
                boundary_factor=boundary_factor
            )
            graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1)) if graviton_field is not None else 0.0
            graviton_nonlinear = np.abs(graviton_trace)**6 / (1e-30 + 1e-15) if graviton_field is not None else 0.0
            stats = {
                'V_j6_mean': np.mean(V_j6),
                'V_j6_std': np.std(V_j6),
                'dV_j6_dphi_mean': np.mean(np.abs(dV_j6_dphi)),
                'dV_j6_dphi_std': np.std(np.abs(dV_j6_dphi)),
                'rio_mean': np.mean(ricci_scalar),
                'rio_std': np.std(ricci_scalar),
                'graviton_trace': graviton_trace,
                'graviton_nonlinear': graviton_nonlinear,
                'boundary_factor': boundary_factor
            }
            logger.debug("J^6 potential stats: %s", stats)
            return stats
        except Exception as e:
            logger.error("J^6 potential analysis failed: %s", e)
            raise
