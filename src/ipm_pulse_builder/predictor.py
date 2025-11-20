#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

def predict_current_from_gate(t_us: List[float],
                              gate_pm1: List[float],
                              model: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """
    Edge-aligned adaptive integration of dI/dt = f(I).
    - gate_pm1: values in [-1, +1]; treat >0 as ON.
    - model: {limits.I_max_A, rise.{I_grid_A,dIdt_grid_A_per_s}, fall.{...}}

    Returns (t_us_out, I_pred_A) with same timestamps as input (for plotting).
    """
    if not t_us or not gate_pm1: return t_us, [0.0]*len(t_us)
    if len(t_us) != len(gate_pm1): return t_us, [0.0]*len(t_us)

    Imax = float(model.get("limits", {}).get("I_max_A", 0.0))
    rI = model["rise"]["I_grid_A"]; rD = model["rise"]["dIdt_grid_A_per_s"]
    fI = model["fall"]["I_grid_A"]; fD = model["fall"]["dIdt_grid_A_per_s"]

    def interp(xx: List[float], yy: List[float], x: float) -> float:
        if x <= xx[0]: return yy[0]
        if x >= xx[-1]: return yy[-1]
        # binary search
        lo, hi = 0, len(xx)-1
        while hi - lo > 1:
            mid = (lo + hi)//2
            if xx[mid] <= x: lo = mid
            else: hi = mid
        x0, x1 = xx[lo], xx[hi]
        y0, y1 = yy[lo], yy[hi]
        w = (x - x0)/(x1 - x0) if x1 != x0 else 0.0
        return y0 + w*(y1 - y0)

    # Build exact edge times from gate_pm1 (ON if >0)
    edges: List[Tuple[float, bool]] = []  # (t_cross, state_on_after)
    state = gate_pm1[0] > 0.0
    t0 = t_us[0]
    for i in range(1, len(t_us)):
        a, b = gate_pm1[i-1] > 0.0, gate_pm1[i] > 0.0
        if a != b:
            # linear crossing time on gate_pm1 between -1 and +1
            # we interpolate on the raw numeric values
            v1, v2 = gate_pm1[i-1], gate_pm1[i]
            if v2 != v1:
                tc = t_us[i-1] + (0.0 - v1) * (t_us[i] - t_us[i-1]) / (v2 - v1)
            else:
                tc = t_us[i]
            state = b
            edges.append((tc, state))

    # Integrate piecewise between [t_start, t_end] segments (ON or OFF)
    I = 0.0
    out_I = [0.0]*len(t_us)
    seg_start = t_us[0]
    seg_state = gate_pm1[0] > 0.0
    edge_idx = 0

    # helper to step integrate over a small interval [ta, tb]
    def step_integrate(ta: float, tb: float, on: bool, I0: float) -> float:
        dt = (tb - ta) * 1e-6  # seconds
        if dt <= 0: return I0
        # adaptive sub-stepping: ~32 steps minimum on short segments
        target_steps = max(4, int(math.ceil((tb - ta) / max(0.05, (tb - ta)/32.0))))
        sub_dt = dt / target_steps
        Icur = I0
        for _ in range(target_steps):
            dIdt = interp(rI, rD, Icur) if on else interp(fI, fD, Icur)
            Icur += dIdt * sub_dt
            if Icur < 0.0: Icur = 0.0
            if Imax > 0.0 and Icur > Imax: Icur = Imax
        return Icur

    # Walk across each plotting interval [t_us[i-1], t_us[i]] and split by edges
    for i in range(1, len(t_us)):
        ta, tb = t_us[i-1], t_us[i]
        # consume any edges inside (ta, tb]
        t_prev = ta
        while edge_idx < len(edges) and edges[edge_idx][0] <= tb:
            t_edge, new_state = edges[edge_idx]
            t_edge = max(ta, min(tb, t_edge))
            # integrate up to edge with current state
            I = step_integrate(t_prev, t_edge, seg_state, I)
            t_prev = t_edge
            seg_state = new_state
            edge_idx += 1
        # integrate remainder to tb
        I = step_integrate(t_prev, tb, seg_state, I)
        out_I[i] = I

    return t_us, out_I
