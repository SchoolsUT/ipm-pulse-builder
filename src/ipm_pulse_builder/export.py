#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple
import csv
from pathlib import Path

# program: must provide duration_us() and sample(npoints) -> (t_us, y_pm1)
def export_sdg_csv(program, path: str, npoints: int = 4096) -> float:
    """
    Write Siglent 'Data' CSV with columns: Time(s), Ampl(V).
    Uses ±1 amplitude; actual volts are set by High/Low on the SDG.
    Returns the suggested ARB frequency (Hz) = 1 / total_duration.
    """
    total_us = float(program.duration_us())
    if total_us <= 0.0:
        raise ValueError("Program duration is zero.")

    f_arb = 1.0 / (total_us * 1e-6)  # repeat once per program period
    t_us, y_pm1 = program.sample(npoints=npoints)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time(s)", "Ampl(V)"])
        for tu, a in zip(t_us, y_pm1):
            w.writerow([f"{tu*1e-6:.12g}", f"{a:.6g}"])
    return f_arb

def export_signal(program, path: str, npoints: int = 4096, *, sdg=None, ui=None):
    """
    Export either:
      - Siglent CSV (Time(s), Ampl(V)) if path ends with .csv
      - IPM project file (JSON) otherwise (recommended: .ipmproj.json)

    Returns:
      - float ARB frequency for CSV exports
      - None for project exports
    """
    p = Path(path)
    suffixes = "".join(p.suffixes).lower()

    # CSV export
    if p.suffix.lower() == ".csv":
        return export_sdg_csv(program, path, npoints=npoints)

    # Project export
    # (accept .ipmproj, .ipmproj.json, .json, etc.)
    from signal_io import save_project
    save_project(str(p), program, sdg=sdg or {}, ui=ui or {})
    return None
