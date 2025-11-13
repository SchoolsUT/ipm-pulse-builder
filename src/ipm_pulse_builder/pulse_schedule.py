#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union

# ---------- Public data types ----------

@dataclass(frozen=True)
class Gap:
    """Explicit LOW time between items (µs)."""
    gap_us: float
    def __post_init__(self):
        if self.gap_us < 0:
            raise ValueError("gap_us must be >= 0")

@dataclass(frozen=True)
class ChargePWM:
    """One charge pulse, then optional PWM train (trailing-edge anchored)."""
    charge_width_us: float
    pwm_width_us: float
    pwm_period_us: float
    pwm_count: int

    def __post_init__(self):
        if self.charge_width_us < 0:
            raise ValueError("charge_width_us must be >= 0")
        if self.pwm_count > 0:
            if self.pwm_period_us <= 0:
                raise ValueError("pwm_period_us must be > 0 when pwm_count > 0")
            if not (0.0 < self.pwm_width_us < self.pwm_period_us):
                raise ValueError("Require 0 < pwm_width_us < pwm_period_us when pwm_count > 0")

Item = Union[Gap, ChargePWM]

# ---------- Program + synthesis ----------

class Program:
    """Sequence of ChargePWM and Gap items, starting at t=0 µs."""
    def __init__(self) -> None:
        self.items: List[Item] = []

    def add(self, it: Item) -> "Program":
        # __post_init__ validation already ran
        self.items.append(it)
        return self

    # --- schedule ---
    def _windows_and_end(self) -> Tuple[List[Tuple[float, float]], float]:
        """Return (ON windows list, total_duration_us)."""
        t = 0.0
        wins: List[Tuple[float, float]] = []
        for it in self.items:
            if isinstance(it, Gap):
                t += it.gap_us
                continue

            # Charge
            ch = max(0.0, it.charge_width_us)
            if ch > 0.0:
                wins.append((t, t + ch))
            t_charge_end = t + ch

            # PWM, trailing-edge anchored
            if it.pwm_count > 0:
                # For each period k = 1..N:
                # period_end = t_charge_end + k * T
                # ON = [period_end - w, period_end]
                for k in range(1, it.pwm_count + 1):
                    period_end = t_charge_end + k * it.pwm_period_us
                    on_start = period_end - it.pwm_width_us
                    on_end = period_end
                    wins.append((on_start, on_end))
                t = t_charge_end + it.pwm_count * it.pwm_period_us
            else:
                t = t_charge_end

        # Merge any overlaps/contiguity (defensive)
        if not wins:
            return [], t
        wins.sort(key=lambda ab: ab[0])
        merged: List[Tuple[float, float]] = []
        s, e = wins[0]
        for s2, e2 in wins[1:]:
            if s2 <= e:  # overlap/contiguous
                e = max(e, e2)
            else:
                merged.append((s, e))
                s, e = s2, e2
        merged.append((s, e))
        return merged, max(t, 0.0)

    def duration_us(self) -> float:
        _, tend = self._windows_and_end()
        return tend

    # --- sampling for preview/export ---
    def sample(self, npoints: int = 4096) -> Tuple[List[float], List[float]]:
        """
        Return (t_us[], y_pm1[]) with y in {-1, +1}.
        Guarantees first and last samples are LOW (−1).
        """
        n = max(2, int(npoints))
        wins, T = self._windows_and_end()
        if T <= 0.0:
            return [0.0, 1.0], [-1.0, -1.0]

        dt = T / (n - 1)
        t_us = [i * dt for i in range(n)]
        y = [-1.0] * n

        wi = 0
        for i, tk in enumerate(t_us):
            # move to the current/next window
            while wi < len(wins) and tk > wins[wi][1]:
                wi += 1
            on = wi < len(wins) and (wins[wi][0] <= tk <= wins[wi][1])
            y[i] = 1.0 if on else -1.0

        # hard guarantees: start LOW, end LOW
        y[0] = -1.0
        y[-1] = -1.0
        return t_us, y
