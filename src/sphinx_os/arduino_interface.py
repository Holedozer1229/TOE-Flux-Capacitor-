import serial
import logging
import time

logger = logging.getLogger(__name__)

class ArduinoInterface:
    """Interface with Arduino for hardware control."""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 9600):
        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            logger.info("Arduino interface initialized on port %s", port)
        except Exception as e:
            logger.error("Arduino initialization failed: %s", e)
            raise
    
    def send_control_signal(self, signal: float) -> None:
        """Send control signal to Arduino."""
        try:
            scaled_signal = int(255 * abs(signal))  # Scale to 0-255 for PWM
            self.serial.write(f"{scaled_signal}\n".encode())
            logger.debug("Sent control signal: %.6f (scaled: %d)", signal, scaled_signal)
        except Exception as e:
            logger.error("Failed to send control signal: %s", e)
            raise
    
    def close(self) -> None:
        """Close the serial connection."""
        try:
            self.serial.close()
            logger.info("Arduino serial connection closed")
        except Exception as e:
            logger.error("Failed to close Arduino connection: %s", e)
            raise
