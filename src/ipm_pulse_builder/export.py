#!/usr/bin/env python3
"""
Siglent SDG1062X export + minimal LAN control

- CSV export: one-column Y samples (0/1 mapped to Low/High volts), equally spaced.
  Load via SDG front-panel: Store/Recall -> File Type: Data -> pick CSV on USB.
  Then set ARB Frequency/Period on the SDG (or via SCPI below) to scale timing.

- Minimal LAN control: set ARB frequency/period and levels over VISA.
  (Uploading ARB data over LAN is possible but device/firmware specific;
   we start with USB CSV for reliability and control timing via SCPI.)

Requires:
- Your local pulse_schedule.py classes (PulseProgram etc.)
- Optional: pyvisa (pip install pyvisa pyvisa-py) for LAN control
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import csv

# ---------------------------
# Helpers to sample a program
# ---------------------------

def program_to_binary_series(program, points: int) -> Tuple[List[float], List[float]]:
    """
    Resample program→ fixed-length binary series using half-open windows.
    Returns (t_us, y_on) where y_on ∈ {0.0, 1.0}.
    """
    if points < 2:
        raise ValueError("points must be >= 2")
    windows = program.schedule()
    if not windows:
        return [0.0, 1.0], [0.0, 0.0]
    t_end = windows[-1][1]
    dt = t_end / (points - 1)
    t = [k * dt for k in range(points)]
    y_on = [0.0] * points

    wi = 0
    for k, tk in enumerate(t):
        while wi < len(windows) and tk > windows[wi][1]:
            wi += 1
        on = (wi < len(windows)) and (windows[wi][0] < tk < windows[wi][1])  # half-open
        y_on[k] = 1.0 if on else 0.0

    # Guarantee start/end low
    y_on[0] = 0.0
    y_on[-1] = 0.0
    return t, y_on


def export_sdg_csv(program,
                   csv_path: str,
                   points: int = 16000) -> dict:
    """
    Export ONE-COLUMN CSV for SDG ARB import, but write samples as −1/+1.
    Rationale: SDG maps −1 → Low, +1 → High. Using −1/+1 avoids the “0 maps to mid-rail” issue.
    """
    if points > 16000: points = 16000
    if points < 2: points = 2

    t_us, y_on = program_to_binary_series(program, points)

    # Map 0/1 to −1/+1 so SDG Low/High rails work as intended
    y_norm = [1.0 if v >= 0.5 else -1.0 for v in y_on]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for v in y_norm:
            w.writerow([f"{v:.6f}"])

    total_us = t_us[-1] if t_us else 0.0
    Ts_us = total_us / (points - 1) if points > 1 else 0.0
    return {
        "points": points,
        "total_duration_us": total_us,
        "sample_period_us": Ts_us,
        "suggested_frequency_hz": (1.0 / (total_us * 1e-6)) if total_us > 0 else 0.0
    }

# ---------------------------
# Minimal SDG LAN controller
# ---------------------------

@dataclass
class SDGLevels:
    high_v: float = 5.0
    low_v: float = 0.0

class SDG1000X:
    """
    Minimal SCPI wrapper to set ARB freq/period and IO levels.

    Notes:
    - Resource string (VXI-11 over LAN): "TCPIP0::<HOST>::inst0::INSTR"
    - You must manually load/select your ARB waveform on the front panel
      (or from USB) the first time. Then you can adjust timing/levels here.
    """
    def __init__(self, host: str, visa_backend: str = ""):
        self.host = host
        self.rm = None
        self.inst = None
        self.backend = visa_backend  # e.g., "@py" to force pyvisa-py

    def connect(self, timeout_ms: int = 3000):
        import pyvisa as visa
        self.rm = visa.ResourceManager(self.backend) if self.backend else visa.ResourceManager()
        self.inst = self.rm.open_resource(f"TCPIP0::{self.host}::inst0::INSTR",
                                          timeout=timeout_ms)
        # ID check
        try:
            idn = self.inst.query("*IDN?")
        except Exception:
            idn = "UNKNOWN"
        return idn.strip()

    def close(self):
        try:
            if self.inst:
                self.inst.close()
        finally:
            self.inst = None
            if self.rm:
                self.rm.close()
                self.rm = None

    # --- basic setters (channel 1 by default) ---
    def set_arb_selected(self, ch: int = 1):
        """
        Put the channel in ARB mode (does not upload data).
        Siglent SCPI (typical): Cn:BSWV WVTP,ARB
        """
        self.inst.write(f"C{ch}:BSWV WVTP,ARB")

    def set_frequency(self, hz: float, ch: int = 1):
        """Set ARB frequency (scales the whole waveform in time)."""
        self.inst.write(f"C{ch}:BSWV FRQ,{hz:.9g}")

    def set_period(self, seconds: float, ch: int = 1):
        """Alternative: set period directly."""
        if seconds <= 0:
            raise ValueError("period must be > 0")
        hz = 1.0 / seconds
        self.set_frequency(hz, ch=ch)

    def set_levels(self, levels: SDGLevels, ch: int = 1):
        """
        Set logic-like rails using High/Low level form.
        Typical Siglent syntax: Cn:BSWV HILV,<V>, LOWV,<V>
        """
        self.inst.write(f"C{ch}:BSWV HILV,{levels.high_v:.6f}")
        self.inst.write(f"C{ch}:BSWV LOWV,{levels.low_v:.6f}")

    def output_on(self, ch: int = 1, state: bool = True):
        self.inst.write(f"C{ch}:OUTP {'ON' if state else 'OFF'}")

    # --- burst/trigger (optional) ---
    def burst_once_ext_trig(self, ch: int = 1, ncycles: int = 1):
        """
        Configure a single-shot (N cycles) burst, waiting for external trigger.
        Typical Siglent burst syntax (model-dependent); adjust if needed:
          Cn:BTWV STATE,ON,MD,TRIG,NCYC,<N>,TRSR,EXT
        """
        self.inst.write(f"C{ch}:BTWV STATE,ON")
        self.inst.write(f"C{ch}:BTWV MD,TRIG")
        self.inst.write(f"C{ch}:BTWV NCYC,{max(1,int(ncycles))}")
        self.inst.write(f"C{ch}:BTWV TRSR,EXT")

