#!/usr/bin/env python3
"""
Calibration + simple predictor sandbox

- Autoselect Tek CSV
- Extract time + current (CH2) and gate (CH1 if present)
- Baseline subtract current
- Find peak current
- Define rise window as low–high % of peak (by current amplitude)
- Define fall window as high–low % of peak (by current amplitude)
- Fit:
    Rise: I(t) ≈ m_rise * t + b_rise   => f_rise(I) = dI/dt ≈ m_rise
    Fall: I(t) ≈ I0 * exp(-(t - t0)/tau) => f_fall(I) = dI/dt ≈ -(1/τ) I

Then:
- Use CH1 from the CSV as the gate (if available), OR a synthetic 1.3 µs gate
- Simulate current vs time for that gate using the fitted slopes
- Plot measured vs simulated current and the gate
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv
import os
import math

# 1 mV = 1 A, CH2 in volts
CH2_TO_AMP = 1000.0


@dataclass
class Trace:
    t_s: List[float]
    I_A: List[float]


@dataclass
class GatePulse:
    t_s: List[float]
    gate_V: List[float]


@dataclass
class CalData:
    current: Trace
    gate: Optional[GatePulse]


# ---------- CSV loading ----------

def _find_header_and_columns(rows: List[List[str]]) -> Tuple[int, int, Optional[int], int]:
    """
    Return (header_idx, i_time, i_ch1, i_ch2)

    i_ch1 may be None if CH1-like column is not found.
    """
    header_idx = None
    header = None
    for i, row in enumerate(rows[:30]):
        lowered = [c.strip().lower() for c in row]
        if any("time" in c for c in lowered):
            header_idx = i
            header = row
            break

    if header_idx is None or header is None:
        raise RuntimeError("Could not find header row with 'time'")

    colnames = [c.strip().lower() for c in header]

    def find_col(*keys: str) -> int:
        for j, name in enumerate(colnames):
            for k in keys:
                if k in name:
                    return j
        raise RuntimeError(f"Could not find any of {keys} in header row: {header!r}")

    def try_find_col(*keys: str) -> Optional[int]:
        for j, name in enumerate(colnames):
            for k in keys:
                if k in name:
                    return j
        return None

    i_time = find_col("time")
    # Gate on CH1 if possible
    i_ch1 = try_find_col("ch1", "channel1", "gate")
    # Current on CH2 from Pearson
    i_ch2 = find_col("ch2", "channel2", "current")
    return header_idx, i_time, i_ch1, i_ch2


def load_time_current_and_gate(csv_path: str) -> CalData:
    """
    Load TIME, CH2 (current) and optionally CH1 (gate) from a Tek CSV.
    Returns CalData(current=Trace, gate=GatePulse or None).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("CSV is empty")

    header_idx, i_time, i_ch1, i_ch2 = _find_header_and_columns(rows)

    t_s: List[float] = []
    I_A_raw: List[float] = []
    gate_V_raw: List[float] = [] if i_ch1 is not None else None

    for row in rows[header_idx + 1:]:
        if len(row) <= max(i_time, i_ch2, (i_ch1 or 0)):
            continue
        try:
            t_val = float(row[i_time])
            ch2_v = float(row[i_ch2])
        except ValueError:
            continue

        t_s.append(t_val)
        I_A_raw.append(ch2_v * CH2_TO_AMP)

        if i_ch1 is not None:
            try:
                ch1_v = float(row[i_ch1])
            except ValueError:
                ch1_v = 0.0
            gate_V_raw.append(ch1_v)

    if len(t_s) < 4:
        raise RuntimeError("Not enough valid data rows after header")

    # Baseline subtract current using early portion
    n = len(I_A_raw)
    n_base = max(20, n // 10)
    n_base = min(n_base, n)
    baseline = sum(I_A_raw[:n_base]) / float(n_base)
    I_A = [I - baseline for I in I_A_raw]

    current_trace = Trace(t_s=t_s, I_A=I_A)

    if gate_V_raw is not None:
        # (Optional) baseline adjust CH1 a bit if needed
        # but for typical 0/5 V gate this is probably unnecessary.
        gate = GatePulse(t_s=list(t_s), gate_V=gate_V_raw)
    else:
        gate = None

    return CalData(current=current_trace, gate=gate)


# ---------- Window selection by amplitude ----------

def find_rise_fall_windows(trace: Trace,
                           frac_rise_lo: float = 0.02,
                           frac_rise_hi: float = 0.95,
                           frac_fall_lo: float = 0.02,
                           frac_fall_hi: float = 0.95) -> Tuple[Trace, Trace]:
    """
    Use ONLY parts of the trace where the current is between chosen
    fractions of the peak.
    """
    t = trace.t_s
    I = trace.I_A
    n = len(t)
    if n == 0:
        raise ValueError("Empty trace")

    # Peak index and value
    i_peak = max(range(n), key=lambda i: I[i])
    I_peak = I[i_peak]

    if I_peak <= 0:
        raise RuntimeError("Peak current is not positive; cannot define windows cleanly")

    # --- Rise window by amplitude ---
    rise_lo = frac_rise_lo * I_peak
    rise_hi = frac_rise_hi * I_peak

    i_rise_start = None
    i_rise_end   = None

    for i in range(i_peak + 1):
        if I[i] >= rise_lo:
            i_rise_start = i
            break

    if i_rise_start is not None:
        for j in range(i_rise_start, i_peak + 1):
            if I[j] >= rise_hi:
                i_rise_end = j
                break

    if i_rise_start is None:
        i_rise_start = 0
    if i_rise_end is None:
        i_rise_end = i_peak

    t_rise = t[i_rise_start: i_rise_end + 1]
    I_rise = I[i_rise_start: i_rise_end + 1]

    # --- Fall window by amplitude ---
    fall_hi = frac_fall_hi * I_peak
    fall_lo = frac_fall_lo * I_peak

    i_fall_start = None
    i_fall_end   = None

    for i in range(i_peak, n):
        if I[i] >= fall_hi:
            i_fall_start = i
            break
    if i_fall_start is None:
        i_fall_start = i_peak

    for j in range(i_fall_start, n):
        if I[j] <= fall_lo:
            i_fall_end = j
            break
    if i_fall_end is None:
        i_fall_end = n - 1

    t_fall = t[i_fall_start: i_fall_end + 1]
    I_fall = I[i_fall_start: i_fall_end + 1]

    rise_trace = Trace(t_s=t_rise, I_A=I_rise)
    fall_trace = Trace(t_s=t_fall, I_A=I_fall)
    return rise_trace, fall_trace


# ---------- Fits: I(t) ----------

def fit_rise_linear(rise: Trace) -> Tuple[float, float]:
    """
    Fit I(t) = m * t + b on the *already trimmed* rising portion.
    Returns (m, b).
    """
    t = rise.t_s
    I = rise.I_A
    n = len(t)
    if n < 2:
        raise ValueError("Not enough points in rising window to fit")

    xs: List[float] = []
    ys: List[float] = []
    for ti, Ii in zip(t, I):
        if math.isfinite(ti) and math.isfinite(Ii):
            xs.append(ti)
            ys.append(Ii)

    if len(xs) < 2:
        raise ValueError("Not enough finite points in rising window to fit")

    npts = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = npts * sum_xx - sum_x * sum_x
    if denom == 0.0:
        m = 0.0
        b = sum_y / npts
    else:
        m = (npts * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / npts

    return m, b


def fit_fall_exponential(fall: Trace,
                         min_I_A: float = 0.05,
                         use_abs: bool = True) -> Tuple[float, float]:
    """
    Fit I(t) ≈ I0 * exp(-(t - t0)/tau) on the *already trimmed* falling portion.

    Returns (tau, ln_I0), so:
        I(t) ≈ exp(ln_I0) * exp(-(t - t0)/tau)
        dI/dt = -(1/τ) * I
    """
    t = fall.t_s
    I = fall.I_A
    if len(t) < 2:
        raise ValueError("Not enough points in falling window to fit")

    xs: List[float] = []
    ys: List[float] = []

    t0 = t[0]
    for ti, Ii in zip(t, I):
        Ii_eff = abs(Ii) if use_abs else Ii
        if Ii_eff <= 0.0 or Ii_eff < min_I_A:
            continue
        if not (math.isfinite(ti) and math.isfinite(Ii_eff)):
            continue
        x = ti - t0
        y = math.log(Ii_eff)
        xs.append(x)
        ys.append(y)

    if len(xs) < 2:
        raise ValueError("Not enough valid points in falling window to fit exponential")

    npts = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = npts * sum_xx - sum_x * sum_x
    if denom == 0.0:
        raise ValueError("Degenerate fall fit")

    m = (npts * sum_xy - sum_x * sum_y) / denom
    c = (sum_y - m * sum_x) / npts

    if m == 0.0:
        raise ValueError("Zero slope in fall fit; cannot get tau")

    tau = -1.0 / m
    ln_I0 = c
    return tau, ln_I0


# ---------- Synthetic test gate (for when CH1 isn't used) ----------

def make_default_test_pulse(width_us: float = 1.3,
                            pre_us: float = 1.0,
                            post_us: float = 3.0,
                            dt_ns: float = 100.0) -> GatePulse:
    """
    Simple single-pulse gate:
      - 0 V for pre_us
      - 5 V for width_us
      - 0 V for post_us
    """
    dt_s = dt_ns * 1e-9
    total_us = pre_us + width_us + post_us
    total_s = total_us * 1e-6
    n_steps = int(total_s / dt_s) + 1

    t_s: List[float] = []
    gate_V: List[float] = []

    for i in range(n_steps):
        t = i * dt_s
        t_us = t * 1e6
        if pre_us <= t_us < (pre_us + width_us):
            v = 5.0
        else:
            v = 0.0
        t_s.append(t)
        gate_V.append(v)

    return GatePulse(t_s=t_s, gate_V=gate_V)


# ---------- Predictor: integrate dI/dt from gate + fitted slopes ----------

def simulate_current_from_gate(
    gate: GatePulse,
    m_rise: float,
    k_fall: float,
    I0: float = 0.0,
    gate_threshold: float = 2.5,
) -> Trace:
    """
    Integrate dI/dt given a gate waveform:

      if gate > gate_threshold:  dI/dt = m_rise
      else:                      dI/dt = -k_fall * I

    Simple forward-Euler integration.
    """
    t = gate.t_s
    g = gate.gate_V
    n = len(t)
    if n < 2:
        raise ValueError("Gate pulse needs at least 2 samples")

    I: List[float] = [0.0] * n
    I[0] = I0

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            I[i] = I[i - 1]
            continue

        if g[i - 1] > gate_threshold:
            dIdt = m_rise
        else:
            dIdt = -k_fall * I[i - 1]

        I[i] = I[i - 1] + dIdt * dt

    return Trace(t_s=list(t), I_A=I)


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    ap = argparse.ArgumentParser(
        description="Calibrate rise/fall slopes from Tek CSV and simulate current for a gate pulse."
    )
    ap.add_argument("csv", nargs="?", help="Tek CSV file (if omitted, file dialog opens).")
    ap.add_argument("--rise_lo", type=float, default=0.02,
                    help="Rise lower fraction of peak (default 0.02).")
    ap.add_argument("--rise_hi", type=float, default=0.95,
                    help="Rise upper fraction of peak (default 0.95).")
    ap.add_argument("--fall_lo", type=float, default=0.02,
                    help="Fall lower fraction of peak (default 0.02).")
    ap.add_argument("--fall_hi", type=float, default=0.95,
                    help="Fall upper fraction of peak (default 0.95).")
    ap.add_argument("--gate_width_us", type=float, default=1.3,
                    help="Synthetic gate high width in µs (default 1.3).")
    ap.add_argument("--gate_pre_us", type=float, default=1.0,
                    help="Synthetic gate pre-pulse low time in µs (default 1).")
    ap.add_argument("--gate_post_us", type=float, default=3.0,
                    help="Synthetic gate post-pulse low time in µs (default 3).")
    ap.add_argument("--gate_dt_ns", type=float, default=100.0,
                    help="Synthetic gate time step in ns (default 100).")
    ap.add_argument("--decay_tau_mult", type=float, default=5.0,
                    help="Extra simulation length after synthetic gate in units of tau_fall (default 5).")
    ap.add_argument("--no_csv_gate", action="store_true",
                    help="Ignore CH1 in the CSV and use a synthetic gate instead.")
    args = ap.parse_args()

    # --- CSV selection ---
    csv_path = args.csv
    if csv_path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            csv_path = filedialog.askopenfilename(
                title="Select Tek CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            root.update()
            root.destroy()
        except Exception as e:
            raise RuntimeError(f"Failed to open file dialog: {e}")

    if not csv_path:
        raise SystemExit("No CSV file selected")

    print(f"Using CSV: {csv_path}")

    # --- Load & window ---
    cal = load_time_current_and_gate(csv_path)
    trace = cal.current
    csv_gate = cal.gate

    rise, fall = find_rise_fall_windows(
        trace,
        frac_rise_lo=args.rise_lo,
        frac_rise_hi=args.rise_hi,
        frac_fall_lo=args.fall_lo,
        frac_fall_hi=args.fall_hi,
    )

    print(f"Total samples: {len(trace.t_s)}")
    print(f"Rise window samples: {len(rise.t_s)}")
    print(f"Fall window samples: {len(fall.t_s)}")

    # --- Fits (give analytic slope functions) ---
    m_rise, b_rise = fit_rise_linear(rise)
    print("\nRISE:")
    print(f"  I(t) ≈ {m_rise:.3e} * t + {b_rise:.3e}")
    print(f"  => f_rise(I): dI/dt ≈ {m_rise:.3e} A/s (independent of I)")

    try:
        tau_fall, ln_I0_fall = fit_fall_exponential(fall, min_I_A=0.05, use_abs=True)
        I0_fall = math.exp(ln_I0_fall)
        k_fall = 1.0 / tau_fall
        print("\nFALL:")
        print(f"  I(t) ≈ {I0_fall:.3e} * exp(-(t - t0)/{tau_fall:.3e})")
        print(f"  => f_fall(I): dI/dt ≈ -{k_fall:.3e} * I")
    except Exception as e:
        tau_fall = None
        k_fall = None
        print(f"\nFall fit failed: {e}")

    # --- Choose gate: CSV CH1 (preferred) or synthetic ---
    gate_used: Optional[GatePulse] = None

    if (not args.no_csv_gate) and csv_gate is not None:
        print("\nUsing CH1 from CSV as gate.")
        gate_used = csv_gate
    else:
        print("\nUsing synthetic gate.")
        gate_used = make_default_test_pulse(
            width_us=args.gate_width_us,
            pre_us=args.gate_pre_us,
            post_us=args.gate_post_us,
            dt_ns=args.gate_dt_ns,
        )
        # extend synthetic gate with zeros for ~decay_tau_mult * tau_fall
        if k_fall is not None and len(gate_used.t_s) >= 2:
            dt = gate_used.t_s[1] - gate_used.t_s[0]
            extra_T = args.decay_tau_mult * tau_fall
            extra_steps = int(extra_T / dt)
            if extra_steps > 0:
                t_last = gate_used.t_s[-1]
                for k in range(1, extra_steps + 1):
                    gate_used.t_s.append(t_last + k * dt)
                    gate_used.gate_V.append(0.0)

    # --- Simulate current from chosen gate ---
    if k_fall is not None and gate_used is not None:
        sim_trace = simulate_current_from_gate(gate_used, m_rise=m_rise, k_fall=k_fall, I0=0.0)
        print(f"\nGate samples: {len(gate_used.t_s)}")
        print(f"Simulated current peak ≈ {max(sim_trace.I_A):.3f} A")
    else:
        sim_trace = None
        print("\nSkipping simulation because fall fit failed or no gate available.")

    # --- Plot ---
    if plt is not None:
        import matplotlib.pyplot as plt  # ensure alias

        t0 = trace.t_s[0]
        t_meas_us = [(tt - t0) * 1e6 for tt in trace.t_s]

        plt.figure(figsize=(9, 8))

        # 1) Measured trace + windows + fits
        plt.subplot(2, 1, 1)
        plt.plot(t_meas_us, trace.I_A, label="Full current trace", alpha=0.3)

        tr_us = [(tt - t0) * 1e6 for tt in rise.t_s]
        tf_us = [(tt - t0) * 1e6 for tt in fall.t_s]
        plt.plot(tr_us, rise.I_A, "C1", label="Rise window")
        plt.plot(tf_us, fall.I_A, "C2", label="Fall window")

        Ir_fit = [m_rise * ti + b_rise for ti in rise.t_s]
        plt.plot(tr_us, Ir_fit, "C1--", label="Rise linear fit")

        if tau_fall is not None:
            t0_f = fall.t_s[0]
            If_fit = [math.exp(ln_I0_fall) * math.exp(-(ti - t0_f) / tau_fall) for ti in fall.t_s]
            plt.plot(tf_us, If_fit, "C2--", label="Fall exp fit")

        plt.xlabel("Time [µs]")
        plt.ylabel("Measured current [A]")
        plt.legend()

        # 2) Measured vs simulated current + gate
        plt.subplot(2, 1, 2)
        if gate_used is not None and sim_trace is not None:
            t_gate_us = [tt * 1e6 for tt in gate_used.t_s]
            I_sim = sim_trace.I_A

            # For comparison, interpolate measured current onto sim time grid (optional)
            # but simplest is just to overplot both vs their own time bases.

            plt.plot(t_meas_us, trace.I_A, label="Measured current", alpha=0.3)
            plt.plot(t_gate_us, I_sim, label="Simulated current")

            # scale gate onto same axis
            I_peak_plot = max(max(I_sim), max(trace.I_A)) if trace.I_A and I_sim else 1.0
            gate_scaled = [(v / 5.0) * I_peak_plot for v in gate_used.gate_V]
            plt.plot(t_gate_us, gate_scaled, label="Gate (scaled)", alpha=0.7)

            plt.xlabel("Time [µs]")
            plt.ylabel("Current / scaled gate")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No simulation (fall fit failed or no gate)",
                     ha="center", va="center", transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.show()
