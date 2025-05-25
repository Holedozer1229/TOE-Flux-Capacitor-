import tkinter as tk
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FluxGUI:
    """Graphical user interface for Flux Capacitor."""
    
    def __init__(self, flux_capacitor):
        self.flux_capacitor = flux_capacitor
        self.root = tk.Tk()
        self.root.title("TOE Flux Capacitor")
        self.label = tk.Label(self.root, text="Flux Capacitor Control Panel")
        self.label.pack()
        self.run_button = tk.Button(self.root, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack()
        self.metrics_label = tk.Label(self.root, text="Metrics: N/A")
        self.metrics_label.pack()
        logger.info("GUI initialized")
    
    def run_simulation(self):
        """Run the simulation and update metrics."""
        try:
            audio_input = np.sin(2 * np.pi * 440 * np.linspace(0, self.flux_capacitor.duration, 
                                                               int(self.flux_capacitor.sample_rate * self.flux_capacitor.duration)))
            audio_output = self.flux_capacitor.run(audio_input)
            metrics = self.flux_capacitor.entanglement_cache.entropy_history[-1] if self.flux_capacitor.entanglement_cache.entropy_history else 0.0
            chsh = self.flux_capacitor.circuit.compute_chsh_violation()
            rio_mean = np.mean(self.flux_capacitor.ricci_scalar) if hasattr(self.flux_capacitor, 'ricci_scalar') else 1.0
            graviton_trace = np.mean(np.trace(self.flux_capacitor.simulator.sphinx_os.graviton_field, axis1=-2, axis2=-1)) if hasattr(self.flux_capacitor.simulator.sphinx_os, 'graviton_field') else 0.0
            self.metrics_label.config(text=f"Entropy: {metrics:.6f}, CHSH: {chsh:.6f}, Rio Mean: {rio_mean:.6f}, Graviton Trace: {graviton_trace:.6f}")
            logger.info("Simulation run completed: entropy=%.6f, chsh=%.6f, rio_mean=%.6f, graviton_trace=%.6f", 
                        metrics, chsh, rio_mean, graviton_trace)
        except Exception as e:
            logger.error("Simulation run failed: %s", e)
            raise
    
    def start(self):
        """Start the GUI event loop."""
        try:
            self.root.mainloop()
            logger.info("GUI event loop started")
        except Exception as e:
            logger.error("GUI event loop failed: %s", e)
            raise
