#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union

# Gate level convention used everywhere else:
#   +1.0 => IPM gate HIGH (charging / PWM ON)
#   -1.0 => IPM gate LOW  (idle)
LOW  = -1.0
HIGH = +1.0

# ---------- Primitives ----------

class Block:
    def duration_us(self) -> float:
        raise NotImplementedError

    def sample(self, npoints: int) -> Tuple[List[float], List[float]]:
        """Return (t_us, y_pm1) for this block only, starting at t=0."""
        raise NotImplementedError


@dataclass(frozen=True)
class Gap(Block):
    """Pure low period."""
    gap_us: float

    def duration_us(self) -> float:
        return max(float(self.gap_us), 0.0)

    def sample(self, npoints: int) -> Tuple[List[float], List[float]]:
        T = self.duration_us()
        if npoints < 2:
            npoints = 2
        if T <= 0:
            return [0.0, 0.0], [LOW, LOW]
        dt = T / (npoints - 1)
        t = [i * dt for i in range(npoints)]
        y = [LOW] * npoints
        return t, y


@dataclass(frozen=True)
class ChargePWM(Block):
    """
    One "charge" high segment followed by a PWM train:
        - charge_width_us: initial high (single shot)
        - pwm_width_us:    width of each PWM high
        - pwm_period_us:   period of each PWM cycle (high + low)
        - pwm_count:       number of PWM cycles
    """
    charge_width_us: float
    pwm_width_us: float
    pwm_period_us: float
    pwm_count: int

    def duration_us(self) -> float:
        ch = max(float(self.charge_width_us), 0.0)
        pw = max(float(self.pwm_width_us), 0.0)
        pp = max(float(self.pwm_period_us), 0.0)
        n  = max(int(self.pwm_count), 0)
        # total time = charge + n * period
        return ch + n * pp

    def sample(self, npoints: int) -> Tuple[List[float], List[float]]:
        T = self.duration_us()
        if npoints < 2:
            npoints = 2
        if T <= 0:
            return [0.0, 0.0], [LOW, LOW]

        # Build piecewise schedule of [ (t_start_us, t_end_us, level) , ... ]
        segs: List[Tuple[float, float, float]] = []
        t0 = 0.0

        # Charge high segment
        ch = max(float(self.charge_width_us), 0.0)
        if ch > 0:
            segs.append((t0, t0 + ch, HIGH))
            t0 += ch

        # PWM cycles
        pw = max(float(self.pwm_width_us), 0.0)
        pp = max(float(self.pwm_period_us), 0.0)
        n  = max(int(self.pwm_count), 0)
        for _ in range(n):
            # high
            if pw > 0:
                t1 = min(t0 + pw, t0 + pp) if pp > 0 else t0 + pw
                segs.append((t0, t1, HIGH))
            else:
                t1 = t0
            # low (rest of the period)
            if pp > 0:
                t2 = t0 + pp
                if t2 > t1:
                    segs.append((t1, t2, LOW))
                t0 = t2
            else:
                t0 = t1

        # Now uniformly sample T with npoints and map to levels
        dt = T / (npoints - 1)
        t = [i * dt for i in range(npoints)]
        y = [LOW] * npoints

        # Walk segments once; fill y by range
        si = 0
        for i, ti in enumerate(t):
            # advance segment index until segs[si] covers ti
            while si < len(segs) and ti >= segs[si][1]:
                si += 1
            if si < len(segs):
                t_start, t_end, lvl = segs[si]
                if t_start <= ti <= t_end:
                    y[i] = lvl
                else:
                    y[i] = LOW
            else:
                y[i] = LOW

        # Ensure hard low at the very end (ARB edge robustness)
        if y:
            y[0] = LOW
            y[-1] = LOW
        return t, y


# ---------- Program (sequence + fixed pre-gap) ----------

class Program:
    """
    A sequence of blocks with an immutable pre-gap at the very beginning.
    pre_gap_us is a fixed low segment, not represented as a moveable item.
    """
    def __init__(self) -> None:
        self.items: List[Union[Gap, ChargePWM]] = []
        self.pre_gap_us: float = 0.0  # << fixed gap at t=0

    # API parity with your existing GUI
    def add(self, block: Union[Gap, ChargePWM]) -> None:
        self.items.append(block)

    def clear(self) -> None:
        self.items.clear()
        self.pre_gap_us = 0.0

    def duration_us(self) -> float:
        t = float(self.pre_gap_us)
        for it in self.items:
            t += it.duration_us()
        return max(t, 0.0)

    def sample(self, npoints: int) -> Tuple[List[float], List[float]]:
        """
        Uniformly sample entire program (including pre-gap) into npoints.
        Time starts at 0 and ends at total duration.
        """
        T = self.duration_us()
        if npoints < 2:
            npoints = 2
        if T <= 0:
            return [0.0, 0.0], [LOW, LOW]

        # Build a piecewise schedule for the whole program:
        # Start with the fixed pre-gap
        segs: List[Tuple[float, float, float]] = []
        t0 = 0.0
        pg = max(float(self.pre_gap_us), 0.0)
        if pg > 0:
            segs.append((t0, t0 + pg, LOW))
            t0 += pg

        # Append each block's segments, using a block-local sample count
        # that is proportional to its duration, so we do not lose PWM detail.
        for it in self.items:
            if T > 0:
                frac = it.duration_us() / T
            else:
                frac = 1.0
            # Give each block at least 16 samples so small pulses survive,
            # and scale with the global requested resolution.
            local_n = max(16, int(frac * npoints))
            bt, by = it.sample(local_n)

            # bt starts at 0; shift by t0
            for i in range(len(bt) - 1):
                a = t0 + bt[i]
                b = t0 + bt[i + 1]
                lvl = HIGH if by[i] > 0 else LOW
                segs.append((a, b, lvl))
            t0 += it.duration_us()

        # Collapse adjacent segments with same level to reduce edges
        segs2: List[Tuple[float, float, float]] = []
        for s in segs:
            if not segs2:
                segs2.append(s)
            else:
                a0, b0, l0 = segs2[-1]
                a1, b1, l1 = s
                if abs(b0 - a1) < 1e-12 and l0 == l1:
                    segs2[-1] = (a0, b1, l0)
                else:
                    segs2.append(s)
        segs = segs2

        # Uniform sample across total duration
        dt = T / (npoints - 1)
        t = [i * dt for i in range(npoints)]
        y = [LOW] * npoints

        si = 0
        for i, ti in enumerate(t):
            # advance to segment that might cover ti
            while si < len(segs) and ti >= segs[si][1]:
                si += 1
            if si < len(segs):
                a, b, lvl = segs[si]
                if a <= ti <= b:
                    y[i] = lvl
                else:
                    y[i] = LOW
            else:
                y[i] = LOW

        # Hard low at both ends for ARB robustness
        y[0] = LOW
        y[-1] = LOW
        return t, y
