# Hardware Setup for TOE Flux Capacitor

## Overview
The TOE Flux Capacitor integrates a retro 8-track player with an Arduino Uno to produce audio outputs driven by quantum-gravitational simulations. The hardware setup modulates electromagnetic signals to generate harmonics (880 Hz, 1320 Hz) and delays (~50 ms), verifiable with a gaussmeter and audio analysis software (e.g., Audacity).

## Components
- **Arduino Uno**: Controls signal output to electromagnets.
  - Cost: ~$25
  - Specifications: ATmega328P, 14 digital I/O pins, 6 PWM outputs
- **8-Track Player**: Vintage audio device for harmonic playback.
  - Cost: ~$100 (refurbished)
  - Model: Compatible with 12V power, stereo output
- **Electromagnets**: Modulate magnetic fields based on CTC signals.
  - Cost: ~$10 each (x2)
  - Specifications: 12V, 1A, 10N force
- **Gaussmeter**: Measures magnetic field strength.
  - Cost: ~$150
  - Model: Handheld, 0–2000 mG range
- **Miscellaneous**: Breadboard, wires, 12V power supply (~$20 total)

## Setup Instructions
1. Connect the Arduino Uno to a computer via USB (port `/dev/ttyUSB0`).
2. Attach electromagnets to PWM pins (e.g., pin 9) and ground.
3. Power the 8-track player with a 12V supply, connecting audio outputs to speakers.
4. Upload `flux_capacitor.ino` to the Arduino using the Arduino IDE.
5. Run the simulation (`main.py`), which sends CTC signals to the Arduino.
6. Use Audacity to record audio outputs and a gaussmeter to measure field strength.

## Validation
- **Audio Outputs**: Verify harmonics at 880 Hz and 1320 Hz using Audacity’s FFT.
- **Magnetic Fields**: Confirm field strength correlates with CTC signals (scaled 0–255).
- **Delays**: Measure ~50 ms delays via cross-correlation in Audacity.
