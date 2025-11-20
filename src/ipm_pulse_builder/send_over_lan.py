# send_over_lan.py
#!/usr/bin/env python3
from __future__ import annotations
import math, struct, time, re
from typing import Tuple, Sequence

import pyvisa

DEFAULT_IP = "192.168.3.100"

def _map_pm1_to_i16(samples_pm1: Sequence[float]) -> bytes:
    out = bytearray(2 * len(samples_pm1))
    j = 0
    for v in samples_pm1:
        x = 32767 if v > 0 else -32768  # hard rails (matches terminal tests)
        out[j:j+2] = struct.pack("<h", x)
        j += 2
    return bytes(out)

def _force_low_edges(y: list[float]) -> None:
    if not y:
        return
    y[0]  = -1.0
    y[-1] = -1.0

def _choose_points(duration_us: float, target_dt_us: float = 0.2, max_pts: int = 4096) -> int:
    if duration_us <= 0:
        return 256
    n = int(math.ceil(duration_us / target_dt_us))
    n = max(256, min(max_pts, n))
    for k in (4096, 2048, 1024, 512, 256):
        if n >= k:
            return k
    return n

def _query_levels(sdg, ch: str) -> tuple[str, float | None, float | None]:
    r = sdg.query(f"{ch}:BSWV?").strip()
    mH = re.search(r"HLEV,([-\d\.eE]+)V", r)
    mL = re.search(r"LLEV,([-\d\.eE]+)V", r)
    H = float(mH.group(1)) if mH else None
    L = float(mL.group(1)) if mL else None
    return r, H, L

# SDG effective ARB sample rate (empirical for SDG1062X)
SDG_ARB_FS = 30_000_000
_ALLOWED_ARBLENS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

def _pick_npoints_from_waveform(t_us, y, target_samples_on_min_high: int = 8, requested_n: int | None = None) -> int:
    """
    Choose npoints based on the *actual* waveform you just generated.
      - Fits the full record into the SDG's fixed ARB sample rate (N <= T * fs)
      - Ensures the *shortest high pulse* is captured with >= target_samples_on_min_high samples
      - Snaps to allowed lengths the SDG likes (powers of two)

    Returns a single integer npoints to use for the *final* sampling/upload.
    """
    if not t_us or not y or len(t_us) != len(y):
        return _ALLOWED_ARBLENS[0]

    T_us = float(t_us[-1] - t_us[0])
    if T_us <= 0:
        return _ALLOWED_ARBLENS[0]

    # Find shortest continuous HIGH segment in the provisional waveform
    min_high_us = None
    in_high = y[0] > 0
    start_t = t_us[0] if in_high else None
    for i in range(1, len(y)):
        v = y[i] > 0
        if not in_high and v:
            in_high = True
            start_t = t_us[i]
        elif in_high and not v:
            dur = t_us[i] - (start_t if start_t is not None else t_us[i])
            min_high_us = dur if (min_high_us is None or dur < min_high_us) else min_high_us
            in_high = False
            start_t = None
    if in_high and start_t is not None:
        dur = t_us[-1] - start_t
        min_high_us = dur if (min_high_us is None or dur < min_high_us) else min_high_us

    # Fit constraints
    n_fit_max = int((T_us * 1e-6) * SDG_ARB_FS)  # cannot exceed this or SDG will stretch/override
    if n_fit_max < _ALLOWED_ARBLENS[0]:
        return _ALLOWED_ARBLENS[0]

    # If user requested and it fits both duration and allowed set, honor it
    if requested_n and requested_n in _ALLOWED_ARBLENS and requested_n <= n_fit_max:
        return requested_n

    # Ensure enough samples on the *shortest* high pulse
    if min_high_us and min_high_us > 0:
        # Samples on min-high = N * (min_high_us / T_us)  ⇒  N >= target * (T_us / min_high_us)
        n_min_for_high = int((target_samples_on_min_high * T_us) / min_high_us + 0.9999)
        # Snap to largest allowed that satisfies both (<= n_fit_max and >= n_min_for_high)
        candidates = [a for a in _ALLOWED_ARBLENS if a <= n_fit_max and a >= n_min_for_high]
        if candidates:
            return candidates[-1]

    # Otherwise just take the largest allowed that fits in duration
    candidates = [a for a in _ALLOWED_ARBLENS if a <= n_fit_max]
    return candidates[-1]

def send_user_arb_over_lan(
    program,
    host: str = DEFAULT_IP,
    channel: str = "C1",
    name: str = "ipm_gui",
    npoints: int = 2048,
    high_v: float = 5.0,
    low_v: float = 0.0,
    load: str = "50",
) -> Tuple[float, int]:
    ch = channel.strip().upper()
    if ch not in ("C1", "C2"):
        raise ValueError("channel must be 'C1' or 'C2'")

    duration_us = max(program.duration_us(), 1e-9)

    # Probe once to measure true record and shortest HIGH segment
    t_probe, y_probe = program.sample(512)  # cheap, fast

    # Pick final npoints: honor an explicit npoints if it fits; otherwise downselect
    npoints_final = _pick_npoints_from_waveform(
        t_probe, y_probe,
        target_samples_on_min_high=8,
        requested_n=npoints  # <- keeps caller's choice when valid
    )

    # Build the payload with the chosen length
    t_us, y = program.sample(npoints_final)

    # Frequency must match intended duration (what preview uses)
    f_arb = 1.0 / (duration_us * 1e-6)

    y = list(y)
    _force_low_edges(y)

    payload = _map_pm1_to_i16(y)

    # Record repeat rate (Hz) from total record length
    T_us_true = program.duration_us()
    f_arb = 1.0 / (T_us_true * 1e-6)

    # ------- EXACT terminal-equivalent VISA session -------
    rm = pyvisa.ResourceManager()
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 12000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"

    try:
        sdg.write("*CLS")
        sdg.write(f"{ch}:OUTP OFF")
        sdg.write(f"{ch}:SRATE MODE,TARB")

        header = (
            f"{ch}:WVDT WVNM,{name},"
            f"TYPE,5,"                                   # int16
            f"LENGTH,{len(payload)}B,"                   # raw byte count
            f"FREQ,1000.000000,AMPL,2.000,OFST,0.000,"   # placeholders (ignored for ARB preview)
            f"PHASE,0.0,"
            f"WAVEDATA,"
        )

        # CRITICAL: one write_raw with HEADER+payload, NO trailing newline
        sdg.write_raw(header.encode("ascii") + payload)

        time.sleep(0.6)  # settle so SDG ingests the bytes

        # Select & configure ARB, frequency, load
        sdg.write(f"{ch}:ARWV NAME,{name}")
        sdg.write(f"{ch}:BSWV WVTP,ARB")
        # ---- Minimal load set (do not toggle outputs, do not reorder anything) ----
        load_norm = (load or "").strip().upper()
        if load_norm in ("50", "50OHM"):
            # Some firmware honors OUTP, some reflects in BSWV — do both, no state flips
            sdg.write(f"{ch}:OUTP LOAD,50")
            sdg.write(f"{ch}:BSWV LOAD,50")
        else:
            sdg.write(f"{ch}:OUTP LOAD,HZ")
            sdg.write(f"{ch}:BSWV LOAD,HiZ")
# ---------------------------------------------------------------------------

        sdg.write(f"{ch}:BSWV FRQ,{f_arb}")
        sdg.write(f"{ch}:BSWV LOAD,{load}")

        # Set levels: try HLEV/LLEV first, fallback to AMPL/OFST if edge cases
        sdg.write(f"{ch}:BSWV HLEV,{high_v}V")
        sdg.write(f"{ch}:BSWV LLEV,{low_v}V")
        time.sleep(0.1)
        r, H, L = _query_levels(sdg, ch)

        if H is None or L is None or abs(H - high_v) > 0.05 or abs(L - low_v) > 0.05:
            amp  = high_v - low_v
            ofst = 0.5 * (high_v + low_v)
            sdg.write(f"{ch}:BSWV AMPL,{amp}V")
            sdg.write(f"{ch}:BSWV OFST,{ofst}V")
            time.sleep(0.1)
            r, H, L = _query_levels(sdg, ch)
        
        # Burst: single cycle, manual/software trigger source ---
        sdg.write(f"{ch}:BTWV STATE,ON")
        sdg.write(f"{ch}:BTWV MODE,TRIG")   # triggered burst (one record per trigger)
        sdg.write(f"{ch}:BTWV NCYC,1")      # 1 cycle per trigger
        sdg.write(f"{ch}:BTWV TRSR,MAN")    # manual/software trigger source

        sdg.write(f"{ch}:OUTP ON")

        err = sdg.query("SYST:ERR?").strip()
        if not err.startswith("0"):
            raise RuntimeError(f"SDG error after upload/select: {err}")

        # Optional sanity check: confirm selected name
        sel = sdg.query(f"{ch}:ARWV?").strip()
        if name not in sel:
            # Not fatal—some firmwares echo index or .bin—just return the rate
            pass

        return f_arb, npoints_final

    finally:
        try:
            sdg.close()
        except Exception:
            pass
        try:
            rm.close()
        except Exception:
            pass
        
def setup_iota_over_lan(gas_us: float, delay_ms: float, host: str):
    """
    Minimal IOTA setup for CH2 over LAN.
    Leaves all other behavior untouched.
    """
    import pyvisa
    rm = pyvisa.ResourceManager()
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 5000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"
    try:
        sdg.write("*CLS")
        sdg.write("C2:OUTP OFF")
        sdg.write("C2:BSWV WVTP,PULS")
        sdg.write(f"C2:BSWV WIDTH,{gas_us}US")
        sdg.write(f"C2:DTIM {delay_ms}MS")
        sdg.write("C2:OUTP ON")
        # optional sanity check, harmless if you don't want it:
        # err = sdg.query("SYST:ERR?").strip()
        # if not err.startswith("0"):
        #     raise RuntimeError(f"IOTA error: {err}")
    finally:
        try: sdg.close()
        except Exception: pass
        try: rm.close()
        except Exception: pass

# --- Arm / Output and Trigger helpers (tiny, safe) ---
def set_output(host: str, channel: str = "C1", on: bool = True) -> None:
    import pyvisa
    rm = pyvisa.ResourceManager("@py")
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 3000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"
    try:
        sdg.write(f"{channel}:OUTP {'ON' if on else 'OFF'}")
    finally:
        try: sdg.close()
        except: pass
        try: rm.close()
        except: pass


def trigger_channel(host: str, channel: str = "C1") -> None:
    import pyvisa
    rm = pyvisa.ResourceManager("@py")
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 3000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"
    try:
        sdg.write(f"{channel}:TRIG")
    finally:
        try: sdg.close()
        except: pass
        try: rm.close()
        except: pass

def ensure_outputs_off(host: str, prefer_channel: str = "C1") -> None:
    """
    Turn both outputs OFF, addressing the non-preferred channel first and the
    preferred channel last so the SDG UI remains focused on the preferred one.
    """
    import pyvisa
    rm = pyvisa.ResourceManager("@py")
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 3000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"
    try:
        pref = prefer_channel.upper()
        other = "C2" if pref == "C1" else "C1"

        # Turn OFF the non-preferred channel first…
        try:
            sdg.write(f"{other}:OUTP OFF")
        except Exception:
            pass

        # …then the preferred channel last, so the UI remains on it.
        try:
            sdg.write(f"{pref}:OUTP OFF")
        except Exception:
            pass

    finally:
        try: sdg.close()
        except Exception: pass
        try: rm.close()
        except Exception: pass