#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional

Level = Literal[-1, 1]   # -1 = LOW, +1 = HIGH

# =========================
# Core timeline primitives
# =========================

@dataclass(frozen=True)
class ChargePWM:
    """One 'charge + PWM' gate block.
       - Charge pulse goes HIGH for charge_width_us.
       - PWM train: each pulse cycle starts LOW, then goes HIGH for pwm_width_us,
         so there is explicitly a LOW gap before the first short HIGH pulse.
       - If pwm_count == 0, this reduces to the single charge pulse.
    """
    charge_width_us: float
    pwm_width_us: float
    pwm_period_us: float
    pwm_count: int

    def validate(self) -> None:
        if self.charge_width_us <= 0:
            raise ValueError("charge_width_us must be > 0")
        if self.pwm_width_us < 0 or self.pwm_period_us <= 0:
            raise ValueError("pwm_* must be >= 0 and period > 0")
        if self.pwm_width_us > self.pwm_period_us:
            raise ValueError("pwm_width_us must be <= pwm_period_us")
        if self.pwm_count < 0:
            raise ValueError("pwm_count must be >= 0")

    def windows(self, t0_us: float) -> List[Tuple[float, float, Level]]:
        """Return [(t_start, t_end, level)], level=+1 for HIGH."""
        self.validate()
        t = float(t0_us)
        out: List[Tuple[float, float, Level]] = []
        # Charge HIGH
        out.append((t, t + self.charge_width_us, 1))
        t += self.charge_width_us
        # PWM train: each cycle begins with LOW then HIGH
        for _ in range(self.pwm_count):
            low_dur = self.pwm_period_us - self.pwm_width_us
            if low_dur > 0:
                # Explicit LOW segment (we don't append LOW to the list; we only record HIGH windows,
                # the sampler knows LOW is the default between HIGH windows).
                t += low_dur
            if self.pwm_width_us > 0:
                out.append((t, t + self.pwm_width_us, 1))
                t += self.pwm_width_us
        return out

    def duration_us(self) -> float:
        return self.charge_width_us + self.pwm_count * self.pwm_period_us


@dataclass(frozen=True)
class Gap:
    """Pure idle/LOW time (no HIGH windows)."""
    gap_us: float
    def validate(self) -> None:
        if self.gap_us < 0:
            raise ValueError("gap_us must be >= 0")
    def windows(self, t0_us: float) -> List[Tuple[float, float, Level]]:
        self.validate()
        # No HIGH windows during a gap.
        return []
    def duration_us(self) -> float:
        self.validate()
        return self.gap_us


Block = ChargePWM | Gap


class Program:
    """A linear sequence of Blocks starting at t=0 at LOW (-1)."""
    def __init__(self) -> None:
        self.blocks: List[Block] = []

    def add(self, block: Block) -> None:
        self.blocks.append(block)

    def clear(self) -> None:
        self.blocks.clear()

    # ---------- exact HIGH windows ----------
    def windows(self) -> List[Tuple[float, float, Level]]:
        t = 0.0
        out: List[Tuple[float, float, Level]] = []
        for b in self.blocks:
            out.extend(b.windows(t))
            t += b.duration_us()
        # merge adjacent/overlapping HIGH windows
        if not out:
            return []
        out.sort(key=lambda w: w[0])
        merged: List[Tuple[float, float, Level]] = []
        s0, e0, _ = out[0]
        for s, e, lv in out[1:]:
            if s <= e0:  # overlap/adjacent
                e0 = max(e0, e)
            else:
                merged.append((s0, e0, 1))
                s0, e0 = s, e
        merged.append((s0, e0, 1))
        return merged

    def duration_us(self) -> float:
        return sum(b.duration_us() for b in self.blocks)

    # ---------- sampler used by CSV/preview/LAN ----------
    def sample(self, npoints: int) -> Tuple[List[float], List[float]]:
        """Return equally spaced samples in [-1,+1], starting at t=0, ending at T_end.
           LOW = -1, HIGH = +1. npoints >= 2."""
        T = max(0.0, self.duration_us())
        if npoints < 2 or T <= 0.0:
            return [0.0, T], [-1.0, -1.0]

        dt = T / (npoints - 1)
        t = [k * dt for k in range(npoints)]
        y = [-1.0] * npoints

        wins = self.windows()
        wi = 0
        for k, tk in enumerate(t):
            while wi < len(wins) and tk > wins[wi][1]:
                wi += 1
            if wi < len(wins) and wins[wi][0] <= tk <= wins[wi][1]:
                y[k] = +1.0
        # Ensure last sample goes LOW
        y[-1] = -1.0
        return t, y
