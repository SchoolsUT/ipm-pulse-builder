#!/usr/bin/env python3
from __future__ import annotations
import math, struct, time, re
from typing import Tuple, Sequence

import pyvisa

DEFAULT_IP = "192.168.3.100"

def _map_pm1_to_i16(samples_pm1: Sequence[float]) -> bytes:
    # Map ±1.0 to int16 rails. (Your gate is already ±1, so this preserves shape.)
    out = bytearray(2 * len(samples_pm1))
    j = 0
    for v in samples_pm1:
        x = 32767 if v > 0 else -32768
        out[j:j+2] = struct.pack("<h", x)
        j += 2
    return bytes(out)

def _force_low_edges(y: list[float]) -> None:
    if not y:
        return
    y[0]  = -1.0
    y[-1] = -1.0

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
    if not t_us or not y or len(t_us) != len(y):
        return _ALLOWED_ARBLENS[0]
    T_us = float(t_us[-1] - t_us[0])
    if T_us <= 0:
        return _ALLOWED_ARBLENS[0]

    # Find shortest continuous HIGH segment
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

    # Fit to total duration at fixed sample rate
    n_fit_max = int((T_us * 1e-6) * SDG_ARB_FS)
    if n_fit_max < _ALLOWED_ARBLENS[0]:
        return _ALLOWED_ARBLENS[0]

    # Honor explicit request if valid
    if requested_n and requested_n in _ALLOWED_ARBLENS and requested_n <= n_fit_max:
        return requested_n

    # Ensure enough samples on shortest high
    if min_high_us and min_high_us > 0:
        n_min_for_high = int((target_samples_on_min_high * T_us) / min_high_us + 0.9999)
        candidates = [a for a in _ALLOWED_ARBLENS if a <= n_fit_max and a >= n_min_for_high]
        if candidates:
            return candidates[-1]

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
    load: str = "HiZ",
) -> Tuple[float, int]:
    ch = channel.strip().upper()
    if ch not in ("C1", "C2"):
        raise ValueError("channel must be 'C1' or 'C2'")

    # Probe once to measure true record and shortest HIGH segment
    t_probe, y_probe = program.sample(512)  # small, fast

    # Pick final npoints (caller’s npoints honored if valid)
    npoints_final = _pick_npoints_from_waveform(
        t_probe, y_probe,
        target_samples_on_min_high=8,
        requested_n=npoints
    )

    # Build the payload with the chosen length
    t_us, y = program.sample(npoints_final)
    duration_us = max(program.duration_us(), 1e-9)
    f_arb = 1.0 / (duration_us * 1e-6)  # frequency matches total record

    y = list(y)
    _force_low_edges(y)
    payload = _map_pm1_to_i16(y)

    # VISA session
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
            f"FREQ,1000.000000,AMPL,2.000,OFST,0.000,"
            f"PHASE,0.0,"
            f"WAVEDATA,"
        )

        # Single raw write: header + payload (no trailing newline)
        sdg.write_raw(header.encode("ascii") + payload)
        time.sleep(0.6)  # allow ingest

        # Select ARB and set params
        sdg.write(f"{ch}:ARWV NAME,{name}")
        sdg.write(f"{ch}:BSWV WVTP,ARB")
        sdg.write(f"{ch}:BSWV FRQ,{f_arb}")
        sdg.write(f"{ch}:BSWV LOAD,{load}")

        # Levels: set H/L explicitly; fallback to AMPL/OFST if needed
        sdg.write(f"{ch}:BSWV HLEV,{high_v}V")
        sdg.write(f"{ch}:BSWV LLEV,{low_v}V")
        time.sleep(0.1)
        _, H, L = _query_levels(sdg, ch)
        if H is None or L is None or abs(H - high_v) > 0.05 or abs(L - low_v) > 0.05:
            amp  = high_v - low_v
            ofst = 0.5 * (high_v + low_v)
            sdg.write(f"{ch}:BSWV AMPL,{amp}V")
            sdg.write(f"{ch}:BSWV OFST,{ofst}V")
            time.sleep(0.1)

        # Burst: one record per trigger; manual trigger source
        sdg.write(f"{ch}:BTWV STATE,ON")
        sdg.write(f"{ch}:BTWV MODE,TRIG")
        sdg.write(f"{ch}:BTWV NCYC,1")
        sdg.write(f"{ch}:BTWV TRSR,MAN")

        sdg.write(f"{ch}:OUTP ON")

        err = sdg.query("SYST:ERR?").strip()
        if not err.startswith("0"):
            raise RuntimeError(f"SDG error after upload/select: {err}")

        # Not fatal if the scope reports index instead of name
        _ = sdg.query(f"{ch}:ARWV?").strip()

        return f_arb, npoints_final

    finally:
        try: sdg.close()
        except Exception: pass
        try: rm.close()
        except Exception: pass

# --- Arm / Output and Trigger helpers ---
def set_output(host: str, channel: str = "C1", on: bool = True) -> None:
    rm = pyvisa.ResourceManager("@py")
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 3000
    sdg.write_termination = "\n"; sdg.read_termination  = "\n"
    try:
        sdg.write(f"{channel}:OUTP {'ON' if on else 'OFF'}")
    finally:
        try: sdg.close()
        except: pass
        try: rm.close()
        except: pass

def trigger_channel(host: str, channel: str = "C1") -> None:
    rm = pyvisa.ResourceManager("@py")
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 3000
    sdg.write_termination = "\n"; sdg.read_termination  = "\n"
    try:
        sdg.write(f"{channel}:TRIG")
    finally:
        try: sdg.close()
        except: pass
        try: rm.close()
        except: pass

def ensure_outputs_off(host: str, prefer_channel: str = "C1") -> None:
    rm = pyvisa.ResourceManager("@py")
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 3000
    sdg.write_termination = "\n"; sdg.read_termination  = "\n"
    try:
        pref = prefer_channel.upper()
        other = "C2" if pref == "C1" else "C1"
        try: sdg.write(f"{other}:OUTP OFF")
        except Exception: pass
        try: sdg.write(f"{pref}:OUTP OFF")
        except Exception: pass
    finally:
        try: sdg.close()
        except Exception: pass
        try: rm.close()
        except Exception: pass

# --- IOTA (CH2) helper: simple 5 V TTL pulse, no delay ----------------------
def setup_iota_over_lan(gas_us: float, host: str):
    """
    Configure CH2 as a single 5 V TTL pulse of width = gas_us (microseconds).
    Does NOT auto-trigger. Use the scope front-panel trigger or a separate
    software trigger if you want to fire it.
    """
    import pyvisa, time
    rm = pyvisa.ResourceManager()
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 5000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"
    try:
        sdg.write("*CLS")
        sdg.write("C2:OUTP OFF")
        sdg.write("C2:BSWV WVTP,PULS")
        sdg.write("C2:BSWV HLEV,5V")          # 5 V high
        sdg.write("C2:BSWV LLEV,0V")          # 0 V low
        sdg.write(f"C2:BSWV WIDTH,{float(gas_us)}US")
        # Burst = one pulse per trigger
        sdg.write("C2:BTWV STATE,ON")
        sdg.write("C2:BTWV MODE,TRIG")
        sdg.write("C2:BTWV NCYC,1")
        sdg.write("C2:BTWV TRSR,MAN")
        sdg.write("C2:OUTP ON")
        # Optional: sanity check
        # err = sdg.query("SYST:ERR?").strip()
        # if not err.startswith("0"):
        #     raise RuntimeError(f"IOTA config error: {err}")
    finally:
        try: sdg.close()
        except: pass
        try: rm.close()
        except: pass
