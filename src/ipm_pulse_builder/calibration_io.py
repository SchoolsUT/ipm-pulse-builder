#!/usr/bin/env python3
from __future__ import annotations
import os, csv, json, math, glob, time
from typing import List, Tuple, Dict, Any, Optional

# ====== Configuration ======
# Root folder that holds calibration data:
# cal/<load_id>/<voltage>V/{ model.json , <voltage>V.csv }
CAL_ROOT = os.path.join(os.path.dirname(__file__), "cal")
SCHEMA_VERSION = 2

# CH2 conversion (already includes your 20 dB pad): 1 mV -> 1 A
# I[A] = 1000 * V_ch2[V]
CH2_TO_AMP = 1000.0

# Gate thresholding on CH1 (volts). We apply hysteresis around this.
GATE_THR_V = 2.5
GATE_HYS_V = 0.25  # +/- around GATE_THR_V

# Minimum ON/OFF segment duration we keep (seconds)
MIN_SEG_S = 50e-9  # 50 ns

# Smoothing window (odd number of points) for dI/dt
DERIV_WIN = 9

# Number of bins for I->dI/dt lookup tables
N_BINS = 64


# =========================
# Public API
# =========================
def list_load_ids() -> List[str]:
    """Return all load_ids under CAL_ROOT (directory names)."""
    if not os.path.isdir(CAL_ROOT):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(CAL_ROOT)):
        p = os.path.join(CAL_ROOT, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return out


def list_voltages_for_load(load_id: str) -> List[int]:
    """Return all available voltages (int) for a given load_id."""
    base = os.path.join(CAL_ROOT, load_id)
    if not os.path.isdir(base):
        return []
    volts: List[int] = []
    for name in sorted(os.listdir(base)):
        if not name.lower().endswith("v"):
            continue
        p = os.path.join(base, name)
        if not os.path.isdir(p):
            continue
        try:
            v = int(name[:-1])  # strip trailing 'V'
            volts.append(v)
        except Exception:
            pass
    return volts


def load_and_validate_model(load_id: str, voltage: int) -> Tuple[bool, Dict[str, Any] | None]:
    """Load model.json and validate schema/content. Returns (ok, dict_or_none)."""
    path = _model_path(load_id, voltage)
    if not os.path.isfile(path):
        return False, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False, None

    if not isinstance(data, dict):
        return False, None

    if int(data.get("schema", -1)) != SCHEMA_VERSION:
        # Schema mismatch -> don't try to interpret
        return False, None

    rise = data.get("rise", {})
    fall = data.get("fall", {})
    limits = data.get("limits", {})

    if not isinstance(rise, dict) or not isinstance(fall, dict) or not isinstance(limits, dict):
        return False, None

    for key in ("I_grid_A", "dIdt_grid_A_per_s"):
        if key not in rise or key not in fall:
            return False, None

    try:
        Imax = float(limits["I_max_A"])
    except Exception:
        return False, None

    if Imax <= 0:
        return False, None

    # Quick consistency check on grids
    try:
        rI = list(map(float, rise["I_grid_A"]))
        rD = list(map(float, rise["dIdt_grid_A_per_s"]))
        fI = list(map(float, fall["I_grid_A"]))
        fD = list(map(float, fall["dIdt_grid_A_per_s"]))
    except Exception:
        return False, None

    if len(rI) < 4 or len(rI) != len(rD):
        return False, None
    if len(fI) < 4 or len(fI) != len(fD):
        return False, None

    return True, data


def get_or_build_model(load_id: str, voltage: int) -> Tuple[bool, Dict[str, Any] | None]:
    """
    Public entry: first tries to load an existing model.json.
    If missing or invalid, will attempt to auto-build from a single CSV in
    cal/<load_id>/<voltage>V/ named '<voltage>V.csv'.

    Returns (ok, model_dict_or_none).
    """
    ok, m = load_and_validate_model(load_id, voltage)
    if ok:
        return ok, m

    # Try to auto-build from CSV named '<voltage>V.csv'
    vdir = _resolve_voltage_dir(load_id, voltage)
    if vdir is None:
        return False, None
    csv_path = _pick_csv_in_dir(vdir, f"{int(voltage)}V.csv")
    if csv_path is None:
        return False, None
    try:
        model = _autobuild_from_single_csv(load_id, voltage, csv_path)
    except Exception:
        return False, None

    # Validate what we just wrote; if it fails, we give up
    ok2, m2 = load_and_validate_model(load_id, voltage)
    if not ok2:
        return False, None
    return True, m2


# =========================
# Internal helpers
# =========================
def _resolve_voltage_dir(load_id: str, voltage: int) -> Optional[str]:
    base = os.path.join(CAL_ROOT, load_id)
    if not os.path.isdir(base):
        return None
    cand = os.path.join(base, f"{int(voltage)}V")
    if os.path.isdir(cand):
        return cand
    # Fallback: try case-insensitive
    for name in os.listdir(base):
        if name.lower() == f"{int(voltage)}v":
            p = os.path.join(base, name)
            if os.path.isdir(p):
                return p
    return None


def _model_path(load_id: str, voltage: int) -> str:
    vdir = _resolve_voltage_dir(load_id, voltage)
    if vdir is None:
        vdir = os.path.join(CAL_ROOT, load_id, f"{int(voltage)}V")
    return os.path.join(vdir, "model.json")


def _pick_csv_in_dir(vdir: str, prefer_name: str) -> Optional[str]:
    """Pick a CSV in vdir, preferring 'prefer_name' if present."""
    prefer_path = os.path.join(vdir, prefer_name)
    if os.path.isfile(prefer_path):
        return prefer_path

    # Otherwise, look for any *.csv
    cand = sorted(glob.glob(os.path.join(vdir, "*.csv")))
    if not cand:
        return None
    return cand[0]


def _baseline_zero(I_A: List[float], n_pre: int = 50) -> List[float]:
    """
    Baseline subtraction: use first n_pre samples to estimate DC offset.
    """
    if not I_A:
        return []
    n = min(len(I_A), max(1, n_pre))
    base = sum(I_A[:n]) / float(n)
    return [x - base for x in I_A]


def _derivative_smooth(t_s: List[float], I_A: List[float], win: int = 9) -> List[float]:
    """
    Compute a smoothed derivative dI/dt using a small moving window.
    Uses central difference inside the window then applies a boxcar average.

    Returns list of same length as inputs.
    """
    n = len(t_s)
    if n < 3:
        return [0.0] * n
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1
    half = win // 2

    # Raw derivative
    d_raw = [0.0] * n
    for i in range(1, n - 1):
        dt = t_s[i + 1] - t_s[i - 1]
        if dt <= 0:
            d_raw[i] = 0.0
        else:
            d_raw[i] = (I_A[i + 1] - I_A[i - 1]) / dt
    d_raw[0] = d_raw[1]
    d_raw[-1] = d_raw[-2]

    # Boxcar smooth
    d_smooth = [0.0] * n
    for i in range(n):
        j0 = max(0, i - half)
        j1 = min(n, i + half + 1)
        s = 0.0
        c = 0
        for j in range(j0, j1):
            v = d_raw[j]
            if math.isfinite(v):
                s += v
                c += 1
        d_smooth[i] = (s / c) if c else 0.0

    return d_smooth


def _gate_on_intervals_with_subsample_edges(
    t_s: List[float],
    v_s: List[float],
    thr: float,
    hys: float,
) -> List[Tuple[float, float]]:
    """
    Detect gate ON intervals from CH1 using hysteresis and sub-sample edge times.
    Returns list of (t_start, t_end) in seconds.

    We treat:
      - OFF -> ON when crossing thr + hys
      - ON  -> OFF when crossing thr - hys

    Edge times are linearly interpolated between the two samples around the crossing.
    """
    if not t_s or len(t_s) != len(v_s):
        return []

    on_intervals: List[Tuple[float, float]] = []
    state = False  # False = OFF, True = ON
    t_start: Optional[float] = None

    hi = thr + hys
    lo = thr - hys

    for i in range(1, len(t_s)):
        t0, t1 = t_s[i - 1], t_s[i]
        v0, v1 = v_s[i - 1], v_s[i]

        if not state:
            # OFF, watch for crossing upward through hi
            if (v0 < hi <= v1) or (v0 <= hi < v1):
                # interpolate
                if v1 != v0:
                    frac = (hi - v0) / (v1 - v0)
                else:
                    frac = 0.0
                t_edge = t0 + frac * (t1 - t0)
                t_start = t_edge
                state = True
        else:
            # ON, watch for crossing downward through lo
            if (v0 > lo >= v1) or (v0 >= lo > v1):
                if v1 != v0:
                    frac = (lo - v0) / (v1 - v0)
                else:
                    frac = 0.0
                t_edge = t0 + frac * (t1 - t0)
                if t_start is not None and t_edge > t_start:
                    on_intervals.append((t_start, t_edge))
                state = False
                t_start = None

    # If still ON at end, close interval at last sample
    if state and t_start is not None:
        on_intervals.append((t_start, t_s[-1]))

    return on_intervals


def _deglitch_and_merge(
    intervals: List[Tuple[float, float]],
    min_keep: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Remove very short intervals and merge two that are separated by tiny gaps.
    """
    if not intervals:
        return []

    # First, drop intervals shorter than min_keep
    kept = []
    for a, b in intervals:
        if b <= a:
            continue
        if (b - a) >= min_keep:
            kept.append((a, b))

    if not kept:
        return []

    # Then merge small gaps
    merged: List[Tuple[float, float]] = []
    cur_a, cur_b = kept[0]
    for a, b in kept[1:]:
        if a <= cur_b + min_keep:
            # extend current
            if b > cur_b:
                cur_b = b
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return merged


def _bin_average_rate(
    pts: List[Tuple[float, float]],
    I_max: float,
    nbins: int = N_BINS,
    enforce_nonneg: bool = False,
    enforce_nonpos: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Bin-average dI/dt vs I over [0, I_max].
    Returns (I_grid, dIdt_grid).
    """
    if nbins < 4:
        nbins = 4
    if I_max <= 0:
        raise ValueError("I_max must be > 0")

    bin_counts = [0] * nbins
    bin_sums = [0.0] * nbins

    for I, di in pts:
        if not math.isfinite(I) or not math.isfinite(di):
            continue
        if I < 0 or I > I_max:
            continue
        k = int((I / I_max) * (nbins - 1))
        if k < 0:
            k = 0
        elif k >= nbins:
            k = nbins - 1
        bin_counts[k] += 1
        bin_sums[k] += di

    I_grid: List[float] = []
    d_grid: List[float] = []
    for k in range(nbins):
        I = I_max * k / float(nbins - 1)
        I_grid.append(I)
        if bin_counts[k] > 0:
            v = bin_sums[k] / bin_counts[k]
        else:
            v = 0.0
        if enforce_nonneg and v < 0:
            v = 0.0
        if enforce_nonpos and v > 0:
            v = 0.0
        d_grid.append(v)

    return I_grid, d_grid


def _find_rise_fall_windows_amp(
    t_s: List[float],
    I_A: List[float],
    frac_rise_lo: float = 0.02,
    frac_rise_hi: float = 0.95,
    frac_fall_lo: float = 0.02,
    frac_fall_hi: float = 0.95,
) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
    """
    Simple amplitude-based rise/fall windowing (same logic as the standalone
    calibration sandbox):

    - Find global peak current I_peak (>0).
    - Rise window: indices where I goes from frac_rise_lo*I_peak up to
      frac_rise_hi*I_peak on the way up.
    - Fall window: indices where I goes from frac_fall_hi*I_peak down to
      frac_fall_lo*I_peak on the way down.

    Returns ((t_rise, I_rise), (t_fall, I_fall)).
    """
    if not t_s or not I_A or len(t_s) != len(I_A):
        raise ValueError("Empty or mismatched trace for rise/fall windowing")

    n = len(t_s)
    # Peak index and value
    i_peak = max(range(n), key=lambda i: I_A[i])
    I_peak = I_A[i_peak]
    if I_peak <= 0.0:
        raise RuntimeError("Peak current is not positive; cannot define windows cleanly")

    # ---- Rise window ----
    rise_lo = frac_rise_lo * I_peak
    rise_hi = frac_rise_hi * I_peak

    i_rise_start = None
    i_rise_end = None

    for i in range(i_peak + 1):
        if I_A[i] >= rise_lo:
            i_rise_start = i
            break
    if i_rise_start is not None:
        for j in range(i_rise_start, i_peak + 1):
            if I_A[j] >= rise_hi:
                i_rise_end = j
                break

    if i_rise_start is None:
        i_rise_start = 0
    if i_rise_end is None:
        i_rise_end = i_peak

    t_rise = t_s[i_rise_start : i_rise_end + 1]
    I_rise = I_A[i_rise_start : i_rise_end + 1]

    # ---- Fall window ----
    fall_hi = frac_fall_hi * I_peak
    fall_lo = frac_fall_lo * I_peak

    i_fall_start = None
    i_fall_end = None

    for i in range(i_peak, n):
        if I_A[i] >= fall_hi:
            i_fall_start = i
            break
    if i_fall_start is None:
        i_fall_start = i_peak

    for j in range(i_fall_start, n):
        if I_A[j] <= fall_lo:
            i_fall_end = j
            break
    if i_fall_end is None:
        i_fall_end = n - 1

    t_fall = t_s[i_fall_start : i_fall_end + 1]
    I_fall = I_A[i_fall_start : i_fall_end + 1]

    return (t_rise, I_rise), (t_fall, I_fall)


def _fit_rise_linear_It(t_rise: List[float], I_rise: List[float]) -> Tuple[float, float]:
    """
    Fit I(t) = m * t + b on the already trimmed rising portion.
    Returns (m, b).
    """
    if len(t_rise) < 2:
        raise ValueError("Not enough points in rising window to fit")

    xs: List[float] = []
    ys: List[float] = []
    for ti, Ii in zip(t_rise, I_rise):
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


def _fit_fall_exponential_It(
    t_fall: List[float],
    I_fall: List[float],
    min_I_A: float = 0.05,
    use_abs: bool = True,
) -> Tuple[float, float]:
    """
    Fit I(t) ≈ I0 * exp(-(t - t0)/tau) on the trimmed falling portion.

    Returns (tau_s, ln_I0).
    """
    if len(t_fall) < 2:
        raise ValueError("Not enough points in falling window to fit")

    xs: List[float] = []
    ys: List[float] = []

    t0 = t_fall[0]
    for ti, Ii in zip(t_fall, I_fall):
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
        raise ValueError("Degenerate fall fit (zero denominator)")

    m = (npts * sum_xy - sum_x * sum_y) / denom
    c = (sum_y - m * sum_x) / npts

    if m == 0.0:
        raise ValueError("Zero slope in fall fit; cannot get tau")

    tau = -1.0 / m
    ln_I0 = c
    return tau, ln_I0


# =========================
# Auto-build from CSV
# =========================
def _autobuild_from_single_csv(load_id: str, voltage: int, csv_path: str) -> Dict[str, Any]:
    """
    New auto-build path based on the standalone calibration sandbox:

    - Read TIME, CH1, CH2 from the Tek CSV.
    - Baseline-subtract the current trace.
    - Define rise and fall windows purely by current amplitude
      (fractions of the global peak).
    - Fit I(t) on those windows:
         rise: I(t) ≈ m_rise * t + b_rise      (=> dI/dt = m_rise)
         fall: I(t) ≈ I0 * exp(-(t-t0)/tau_f)  (=> dI/dt = -(1/tau_f) * I)
    - Convert these analytic forms into I -> dI/dt lookup tables
      so the predictor can keep using the same grid/interp machinery.
    - Save model.json with the same public shape as before:
         limits.I_max_A
         rise.{I_grid_A, dIdt_grid_A_per_s}
         fall.{I_grid_A, dIdt_grid_A_per_s}
    """
    t_s, ch1_V, I_A_raw = _read_time_ch1_ch2(csv_path)
    if len(t_s) < 8:
        raise RuntimeError("Calibration CSV too short")

    # Baseline-subtract current (reuse existing helper)
    I0 = _baseline_zero(I_A_raw, n_pre=min(50, len(I_A_raw) // 10 or 1))

    # Amplitude-based windows (2–95 % on each side by default)
    (t_rise, I_rise), (t_fall, I_fall) = _find_rise_fall_windows_amp(
        t_s,
        I0,
        frac_rise_lo=0.02,
        frac_rise_hi=0.95,
        frac_fall_lo=0.02,
        frac_fall_hi=0.95,
    )

    if len(t_rise) < 50 or len(t_fall) < 50:
        raise RuntimeError(
            "Insufficient usable rise/fall samples after windowing. "
            "Check calibration CSV."
        )

    # ---- Fits in I(t) space ----
    m_rise, b_rise = _fit_rise_linear_It(t_rise, I_rise)
    tau_fall, ln_I0_fall = _fit_fall_exponential_It(
        t_fall, I_fall, min_I_A=0.05, use_abs=True
    )
    k_fall = 1.0 / tau_fall  # so dI/dt = -k_fall * I in the predictor

    # Use the max over the windows as I_max
    I_max = max(max(I_rise), max(I_fall))
    if I_max <= 0.0:
        raise RuntimeError("Non-positive I_max from calibration trace")

    # ---- Build lookup tables from the analytic forms ----
    nb = max(8, N_BINS)
    dI = I_max / float(nb - 1)
    I_grid = [dI * i for i in range(nb)]

    # For rise, dI/dt is constant m_rise (independent of I)
    rI = list(I_grid)
    rD = [float(m_rise)] * nb

    # For fall, dI/dt = -(1/tau_fall) * I = -k_fall * I
    fI = list(I_grid)
    fD = [-float(k_fall) * I for I in I_grid]

    model = {
        "schema": SCHEMA_VERSION,
        "meta": {
            "load_id": load_id,
            "voltage_V": int(voltage),
            "time_units": "s",
            "created_utc": _utc_now_iso(),
            "source_file": os.path.basename(csv_path),
            "rise_fit": {
                "model": "I(t) = m*t + b",
                "m_rise_A_per_s": float(m_rise),
                "b_rise_A": float(b_rise),
            },
            "fall_fit": {
                "model": "I(t) = I0 * exp(-(t-t0)/tau)",
                "tau_s": float(tau_fall),
                "ln_I0": float(ln_I0_fall),
            },
            "n_bins": nb,
        },
        "limits": {
            "I_max_A": float(I_max),
        },
        "rise": {"I_grid_A": rI, "dIdt_grid_A_per_s": rD},
        "fall": {"I_grid_A": fI, "dIdt_grid_A_per_s": fD},
    }

    _atomic_write_json(_model_path(load_id, voltage), model)
    return model


# =========================
# CSV helpers
# =========================
def _read_time_ch1_ch2(path: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Robust Tek-style CSV reader: finds the real header row that contains both TIME and CH*
    Returns (t_s, ch1_V, I_A). Raises if missing columns.
    """
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.reader(f))

    # Find header row by scanning the first ~200 lines for TIME and CH2
    header_idx = None
    for i in range(min(200, len(rows))):
        low = [c.strip().lower() for c in rows[i]]
        if any("time" in c for c in low) and any("ch2" in c for c in low):
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError("Could not find header row with TIME and CH2")

    header = rows[header_idx]
    cols_low = [c.strip().lower() for c in header]

    def find_col(*keys: str) -> int:
        for j, name in enumerate(cols_low):
            for k in keys:
                if k in name:
                    return j
        raise RuntimeError(f"Could not find any of {keys} in header: {header!r}")

    i_time = find_col("time")
    i_ch2 = find_col("ch2", "channel2", "current")
    i_ch1 = None
    # CH1 is optional
    for cand in ("ch1", "channel1", "gate"):
        for j, name in enumerate(cols_low):
            if cand in name:
                i_ch1 = j
                break
        if i_ch1 is not None:
            break

    t_s: List[float] = []
    ch1_V: List[float] = []
    I_A: List[float] = []

    for row in rows[header_idx + 1 :]:
        if len(row) <= i_time or len(row) <= i_ch2:
            continue
        try:
            t_val = float(row[i_time])
            ch2_v = float(row[i_ch2])
        except Exception:
            continue
        t_s.append(t_val)
        I_A.append(ch2_v * CH2_TO_AMP)
        if i_ch1 is not None and len(row) > i_ch1:
            try:
                ch1_v = float(row[i_ch1])
            except Exception:
                ch1_v = 0.0
        else:
            ch1_v = 0.0
        ch1_V.append(ch1_v)

    if len(t_s) < 4:
        raise RuntimeError("CSV has too few valid samples")

    return t_s, ch1_V, I_A


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
