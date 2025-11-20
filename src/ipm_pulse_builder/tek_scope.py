# tek_scope.py
#!/usr/bin/env python3
from __future__ import annotations
import os, csv
from typing import Dict, List, Tuple, Iterable, Optional

import pyvisa

# ---------------------------
# Tiny VISA helpers
# ---------------------------
def _w(inst, cmd: str) -> None:
    inst.write(cmd)

def _q(inst, cmd: str) -> str:
    return inst.query(cmd).strip()

def _try_queries(inst, queries: Iterable[str]) -> str:
    last = None
    for q in queries:
        try:
            return inst.query(q).strip()
        except Exception as e:
            last = e
    if last:
        raise last
    raise RuntimeError("No matching query worked.")

def _has_cmd(inst, probe: str) -> bool:
    try:
        inst.query(probe)
        return True
    except Exception:
        return False

# ---------------------------
# Tek preamble + curve parsers
# ---------------------------
def _parse_curve_ascii(curve_reply: str) -> List[float]:
    # Tek returns "y,y,y,..." or sometimes "CURVE y,y,y,..."
    s = curve_reply.strip()
    if not s:
        return []
    if s[:5].upper() == "CURVE":
        s = s.split(None, 1)[-1]
    parts = s.replace(";", ",").split(",")
    out: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out

def _parse_wfmpre(pre: str) -> Dict[str, float]:
    """
    Extract XINCR,XZERO,YMULT,YOFF,YZERO from WFMO?/WFMOutpre? reply.
    We tolerate 'XINCR 1.0E-6' or 'XINCR,1.0E-6' tokenization.
    """
    toks = pre.replace(";", ",").replace("\n", ",").split(",")
    keyvals: Dict[str, str] = {}
    last_key = None
    KEYS = {"XINCR", "XZERO", "YMULT", "YOFF", "YZERO"}
    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue
        up = tok.upper()
        if up in KEYS:
            last_key = up
            continue
        if last_key:
            keyvals[last_key] = tok
            last_key = None

    def _g(k: str, default: float = 0.0) -> float:
        try:
            return float(keyvals.get(k, default))
        except Exception:
            return default

    return {
        "xincr": _g("XINCR", 1e-6),
        "xzero": _g("XZERO", 0.0),
        "ymult": _g("YMULT", 1.0),
        "yoff":  _g("YOFF",  0.0),
        "yzero": _g("YZERO", 0.0),
    }

# ---------------------------
# Channel ON/OFF detection
# ---------------------------
def _query_bool(inst, *cmds: str) -> bool:
    """Try a few boolean-ish queries; return True if any says channel is on."""
    for c in cmds:
        try:
            r = inst.query(c).strip().upper()
            if r in ("1", "ON", "TRUE"):
                return True
            if r in ("0", "OFF", "FALSE"):
                return False
            try:
                return float(r) != 0.0
            except Exception:
                pass
        except Exception:
            pass
    return False  # safest default

def _which_channels_on(inst, candidates=("CH1","CH2","CH3","CH4")) -> List[str]:
    active: List[str] = []
    for ch in candidates:
        chu = ch.upper()
        idx = "".join(c for c in chu if c.isdigit()) or chu[-1]
        is_on = _query_bool(
            inst,
            f"SELECT:{chu}?",   # common on many Tek models
            f"SEL:{chu}?",      # alias
            f"CH{idx}:DISP?"    # some models use channel display flag
        )
        if is_on:
            active.append(chu)
    return active

# ---------------------------
# Capture & save APIs
# ---------------------------
def scope_capture_and_fetch(
    host: str,
    sources: Optional[Iterable[str]] = None,  # None => auto-detect active channels
    *,
    single_sequence: bool = True,
    timeout_ms: int = 12000,
    record_length: Optional[int] = None,
    average_count: Optional[int] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Fetch calibrated waveforms from a Tek (e.g., MDO34) over LAN.

    Returns:
        {
          "CH1": {"t_s":[...], "y_V":[...]},
          "CH2": {...},
          ...
        }
    """
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(f"TCPIP0::{host}::INSTR")
    inst.timeout = timeout_ms
    inst.write_termination = "\n"
    inst.read_termination  = "\n"

    try:
        # Do NOT *CLS here; we don’t want to clear the user’s setup.
        _q(inst, "*IDN?")

        # Optional setup (non-destructive)
        if record_length:
            _w(inst, f"HOR:RECO {int(record_length)}")
        if average_count and average_count > 1:
            _w(inst, "ACQ:MODE AVE")
            _w(inst, f"ACQ:NUMAVG {int(average_count)}")
        else:
            _w(inst, "ACQ:MODE SAM")

        # If requested: arm single sequence and wait for completion
        if single_sequence:
            _w(inst, "ACQ:STOPA SEQ")
            _w(inst, "ACQ:STATE RUN")
            _q(inst, "*OPC?")  # block until acquisition is complete

        # Fetch setup
        _w(inst, "DATa:ENCdg ASCii")
        _w(inst, "DATa:WIDth 1")
        _w(inst, "DATa:STARt 1")
        _w(inst, "DATa:STOP 1e9")

        # Auto-pick active channels if none specified
        if sources is None:
            chans = _which_channels_on(inst)
            if not chans:
                # If nothing is ON, default to CH1 to avoid silent no-op.
                chans = ["CH1"]
        else:
            chans = [s.strip().upper() for s in sources if s and s.strip()]

        out: Dict[str, Dict[str, List[float]]] = {}
        for ch in chans:
            _w(inst, f"DATa:SOUrce {ch}")
            pre = _try_queries(inst, ("WFMOutpre?", "WFMO?"))
            scal = _parse_wfmpre(pre)

            raw = _q(inst, "CURVe?")
            y_raw = _parse_curve_ascii(raw)

            n = len(y_raw)
            t_s = [scal["xzero"] + i * scal["xincr"] for i in range(n)]
            y_V = [(v - scal["yoff"]) * scal["ymult"] + scal["yzero"] for v in y_raw]

            out[ch] = {"t_s": t_s, "y_V": y_V}

        return out

    finally:
        try: inst.close()
        except: pass
        try: rm.close()
        except: pass

def save_scope_csvs(
    capture: Dict[str, Dict[str, List[float]]],
    out_dir: str,
    prefix: str = "tek",
) -> List[str]:
    """
    Save each channel to its own CSV: Time_s, <CHx>_V
    Returns list of paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for ch, data in capture.items():
        t_s = data["t_s"]; y = data["y_V"]
        path = os.path.join(out_dir, f"{prefix}_{ch}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Time_s", f"{ch}_V"])
            for ti, yi in zip(t_s, y):
                w.writerow([f"{ti:.12g}", f"{yi:.12g}"])
        written.append(path)
    return written

def save_scope_csv_combined(
    capture: Dict[str, Dict[str, List[float]]],
    out_path: str,
    align: str = "truncate",  # "truncate" (safe) | "first"
) -> str:
    """
    Single CSV with Time_s + one column per active channel.
    Assumes channels share the same horizontal scale (Tek default).
    align="truncate": trims to min length across channels (safest).
    """
    if not capture:
        raise ValueError("No capture data to write.")

    # Establish a reference time vector (use the first channel)
    chans = sorted(capture.keys())
    ref = capture[chans[0]]
    t_ref = ref["t_s"]

    if align == "truncate":
        n = min(len(ch["t_s"]) for ch in capture.values())
        t_out = t_ref[:n]
        y_cols = {ch: capture[ch]["y_V"][:n] for ch in chans}
    else:
        # align="first": just emit lengths as-is, row-by-row from the first channel
        n = len(t_ref)
        t_out = t_ref
        y_cols = {}
        for ch in chans:
            y = capture[ch]["y_V"]
            y_cols[ch] = y[:n] if len(y) >= n else (y + ["" for _ in range(n - len(y))])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Time_s"] + [f"{ch}_V" for ch in chans]
        w.writerow(header)
        for i in range(len(t_out)):
            row = [f"{t_out[i]:.12g}"] + [
                (f"{y_cols[ch][i]:.12g}" if (i < len(y_cols[ch]) and y_cols[ch][i] != "") else "")
                for ch in chans
            ]
            w.writerow(row)
    return out_path
