#!/usr/bin/env python3
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv, tempfile, os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pulse_schedule import Program, ChargePWM, Gap
from export import export_sdg_csv
from send_over_lan import (
    setup_iota_over_lan,
    send_user_arb_over_lan,   # direct ARB over LAN
)

SDG_HOST = "192.168.3.100"
LAN_NPOINTS = 2048  # points used when uploading over LAN (can set to 4096 if you want)

def f3(x: float) -> str: return f"{x:.3f}"

# =========================
# Left: Pulse Editor
# =========================
class PulseEditor(ttk.LabelFrame):
    def __init__(self, master, on_add_pulse, on_add_gap, on_update_sel):
        super().__init__(master, text="Pulse Editor (Charging + PWM)")
        self._on_add_pulse = on_add_pulse
        self._on_add_gap = on_add_gap
        self._on_update_sel = on_update_sel

        self.v_charge = tk.StringVar(value="85.0")
        self.v_pwmw  = tk.StringVar(value="6.7")
        self.v_pwmp  = tk.StringVar(value="20.0")
        self.v_pwmn  = tk.StringVar(value="10")
        self.v_gap   = tk.StringVar(value="100.0")

        g = ttk.Frame(self); g.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        rows = [
            ("Charge width (µs)", self.v_charge),
            ("PWM width (µs)",   self.v_pwmw),
            ("PWM period (µs)",  self.v_pwmp),
            ("PWM count",        self.v_pwmn),
        ]
        for r, (label, var) in enumerate(rows):
            ttk.Label(g, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(g, textvariable=var, width=12).grid(row=r, column=1, sticky="ew")
        ttk.Button(g, text="Add to Sequence", command=self._add_pulse).grid(row=4, column=0, sticky="ew", pady=(8,0))
        ttk.Button(g, text="Update Selected", command=self._update_sel).grid(row=4, column=1, sticky="ew", pady=(8,0))

        # GAP
        ttk.Label(g, text="Insert GAP (µs)").grid(row=5, column=0, sticky="w", pady=(12,0))
        ttk.Entry(g, textvariable=self.v_gap, width=12).grid(row=5, column=1, sticky="ew", pady=(12,0))
        ttk.Button(g, text="Insert Gap", command=self._add_gap).grid(row=6, column=0, columnspan=2, sticky="ew")

        g.columnconfigure(1, weight=1)

    def _read_pulse(self) -> ChargePWM:
        return ChargePWM(
            charge_width_us=float(self.v_charge.get()),
            pwm_width_us=float(self.v_pwmw.get()),
            pwm_period_us=float(self.v_pwmp.get()),
            pwm_count=int(self.v_pwmn.get())
        )

    def set_fields_from_pulse(self, p: ChargePWM) -> None:
        self.v_charge.set(f3(p.charge_width_us))
        self.v_pwmw.set(f3(p.pwm_width_us))
        self.v_pwmp.set(f3(p.pwm_period_us))
        self.v_pwmn.set(str(p.pwm_count))

    def _add_pulse(self):
        try:
            self._on_add_pulse(self._read_pulse())
        except Exception as e:
            messagebox.showerror("Pulse", str(e))

    def _update_sel(self):
        try:
            self._on_update_sel(self._read_pulse())
        except Exception as e:
            messagebox.showerror("Update", str(e))

    def _add_gap(self):
        try:
            g = Gap(gap_us=float(self.v_gap.get()))
            self._on_add_gap(g)
        except Exception as e:
            messagebox.showerror("Gap", str(e))

# =========================
# Middle: Sequence list
# =========================
class SequenceList(ttk.LabelFrame):
    def __init__(self, master, on_select):
        super().__init__(master, text="Sequence")
        self.on_select = on_select
        self.items: list[ChargePWM | Gap] = []

        frm = ttk.Frame(self); frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self.lb = tk.Listbox(frm, width=44, height=16)
        self.lb.grid(row=0, column=0, rowspan=6, sticky="nsew")
        sb = ttk.Scrollbar(frm, orient="vertical", command=self.lb.yview)
        self.lb.config(yscrollcommand=sb.set); sb.grid(row=0, column=1, rowspan=6, sticky="ns")
        ttk.Button(frm, text="↑", command=self.up).grid(row=0, column=2, sticky="ew")
        ttk.Button(frm, text="↓", command=self.down).grid(row=1, column=2, sticky="ew")
        ttk.Button(frm, text="Remove", command=self.remove).grid(row=2, column=2, sticky="ew")
        ttk.Button(frm, text="Clear", command=self.clear).grid(row=3, column=2, sticky="ew")

        frm.rowconfigure(5, weight=1); frm.columnconfigure(0, weight=1)
        self.lb.bind("<<ListboxSelect>>", self._sel)

    def _label(self, it: ChargePWM | Gap) -> str:
        if isinstance(it, Gap):
            return f"GAP {f3(it.gap_us)} µs"
        return f"Pulse  Charge {f3(it.charge_width_us)} µs | PWM {f3(it.pwm_width_us)}/{f3(it.pwm_period_us)} µs ×{it.pwm_count}"

    def add(self, it: ChargePWM | Gap) -> None:
        self.items.append(it); self.lb.insert(tk.END, self._label(it))

    def update_selected(self, p: ChargePWM) -> None:
        sel = self.lb.curselection()
        if not sel: return
        i = sel[0]
        if isinstance(self.items[i], Gap):
            messagebox.showwarning("Update", "Selected item is a GAP. Remove and re-insert to change.")
            return
        self.items[i] = p
        self.lb.delete(i); self.lb.insert(i, self._label(p)); self.lb.select_set(i)

    def remove(self):
        sel = self.lb.curselection()
        if not sel: return
        i = sel[0]
        self.lb.delete(i); self.items.pop(i)

    def clear(self):
        self.lb.delete(0, tk.END); self.items.clear()

    def up(self):
        sel = self.lb.curselection()
        if not sel or sel[0] == 0: return
        i = sel[0]
        self.items[i-1], self.items[i] = self.items[i], self.items[i-1]
        txt = self.lb.get(i); self.lb.delete(i); self.lb.insert(i-1, txt); self.lb.select_set(i-1)

    def down(self):
        sel = self.lb.curselection()
        if not sel or sel[0] == len(self.items)-1: return
        i = sel[0]
        self.items[i+1], self.items[i] = self.items[i], self.items[i+1]
        txt = self.lb.get(i); self.lb.delete(i); self.lb.insert(i+1, txt); self.lb.select_set(i+1)

    def _sel(self, _):
        sel = self.lb.curselection()
        if not sel: return
        it = self.items[sel[0]]
        if isinstance(it, ChargePWM):
            self.on_select(it)

# =========================
# Right: Options / IOTA
# =========================
class OptionsPane(ttk.LabelFrame):
    def __init__(self, master, on_preview, on_export, on_send,
                 on_iota_send, on_iota_to_ipm):
        super().__init__(master, text="Options / Preview / Send (CH1 = IPM)")
        self.on_preview, self.on_export, self.on_send = on_preview, on_export, on_send
        self.on_iota_send, self.on_iota_to_ipm = on_iota_send, on_iota_to_ipm

        self.spacing = tk.StringVar(value="0.0")
        self.gas_us  = tk.StringVar(value="500.0")
        self.delay_ms= tk.StringVar(value="0.0")

        g = ttk.Frame(self); g.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        ttk.Label(g, text="Spacing between all blocks (µs)").grid(row=0, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.spacing, width=10).grid(row=0, column=1, sticky="ew")

        ttk.Button(g, text="Preview", command=self._preview).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8,0))
        ttk.Button(g, text="Export CSV (USB)", command=self._export).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(g, text="Send to SDG (LAN)", command=self._send).grid(row=3, column=0, columnspan=2, sticky="ew")

        sep = ttk.Separator(g, orient="horizontal"); sep.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10,6))
        ttk.Label(g, text="IOTA (CH2)").grid(row=5, column=0, columnspan=2)

        ttk.Label(g, text="Gas Injection Duration (µs)").grid(row=6, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.gas_us, width=10).grid(row=6, column=1, sticky="ew")
        ttk.Label(g, text="Delay to IPM (ms)").grid(row=7, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.delay_ms, width=10).grid(row=7, column=1, sticky="ew")

        ttk.Button(g, text="Send IOTA (CH2)", command=self._iota_send).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6,0))
        ttk.Button(g, text="IOTA → IPM (coarse via LAN)", command=self._iota_to_ipm).grid(row=9, column=0, columnspan=2, sticky="ew")

        g.columnconfigure(1, weight=1)

    def _spacing(self) -> float:
        try: return float(self.spacing.get() or 0.0)
        except: return 0.0

    def _preview(self): self.on_preview(self._spacing())
    def _export(self):  self.on_export(self._spacing())
    def _send(self):    self.on_send(self._spacing())

    def _iota_send(self):
        try:
            gas = float(self.gas_us.get())
            dms = float(self.delay_ms.get())
            self.on_iota_send(gas, dms)
        except Exception as e:
            messagebox.showerror("IOTA", str(e))

    def _iota_to_ipm(self):
        try:
            gas = float(self.gas_us.get())
            dms = float(self.delay_ms.get())
            self.on_iota_to_ipm(gas, dms)
        except Exception as e:
            messagebox.showerror("IOTA", str(e))

# =========================
# Preview panel
# =========================
class PlotPane(ttk.LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Preview")
        self.fig = Figure(figsize=(8,3.3), layout="constrained")
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_from_csv(self, program: Program, npoints: int = 4096):
        # generate the SAME data we export
        t_us, y = program.sample(npoints)
        self.ax1.clear(); self.ax2.clear()
        self.ax1.plot(t_us, y)
        self.ax1.set_ylabel("IPM Input (±1)")
        self.ax1.set_ylim(-1.1, 1.1)

        # simple rise/decay placeholder current
        import math
        T = max(program.duration_us(), 1e-9)
        dt = T / (len(t_us) - 1)
        I, cur = [], 0.0
        tau_on = 50.0; tau_off = 100.0; Imax = 85.0
        for v in y:
            if v > 0:
                cur += (Imax - cur) * (1 - math.exp(-dt/tau_on))
            else:
                cur -= cur * (1 - math.exp(-dt/tau_off))
            I.append(max(0.0, cur))
        self.ax2.plot(t_us, I)
        self.ax2.set_ylabel("Predicted Discharge Current (A)")
        self.ax2.set_xlabel("Time (µs)")
        self.canvas.draw_idle()

# =========================
# Main app wiring
# =========================
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IPM Pulse Builder"); self.geometry("1200x760")

        root = ttk.Frame(self); root.pack(fill=tk.BOTH, expand=True)
        self.editor = PulseEditor(root, self._add_pulse, self._add_gap, self._update_sel)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.seq = SequenceList(root, self._on_select)
        self.seq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.opts = OptionsPane(
            root, self.on_preview, self.on_export, self.on_send,
            self.on_iota_send, self.on_iota_to_ipm
        )
        self.opts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.plot = PlotPane(self); self.plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

    def _on_select(self, p: ChargePWM): self.editor.set_fields_from_pulse(p)

    def _add_pulse(self, p: ChargePWM): self.seq.add(p)
    def _add_gap(self, g: Gap): self.seq.add(g)

    def _update_sel(self, p: ChargePWM): self.seq.update_selected(p)

    # ---- program assembly ----
    def _build_program(self, spacing_us: float) -> Program:
        prog = Program()
        first = True
        for it in self.seq.items:
            if not first and spacing_us > 0:
                prog.add(Gap(spacing_us))
            prog.add(it)
            first = False
        return prog

    # ---- UI handlers ----
    def on_preview(self, spacing_us: float):
        prog = self._build_program(spacing_us)
        self.plot.show_from_csv(prog)

    def on_export(self, spacing_us: float):
        if not self.seq.items:
            messagebox.showwarning("Export", "Sequence is empty."); return
        path = filedialog.asksaveasfilename(title="Save Siglent CSV",
                                            defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")],
                                            initialfile="IPM_GATE.csv")
        if not path: return
        prog = self._build_program(spacing_us)
        arb_f = export_sdg_csv(prog, path, npoints=4096)
        messagebox.showinfo("Exported",
            f"Saved: {os.path.basename(path)}\n"
            f"Total duration: {prog.duration_us():.3f} µs\n"
            f"Suggested ARB frequency: {arb_f:.2f} Hz\n\n"
            "On the SDG:\n• Store/Recall → File Type: Data → select CSV on USB\n"
            "• Set ARB Frequency to the value above\n• Set High=5 V, Low=0 V, Load=Hi-Z")

    def on_send(self, spacing_us: float):
        if not self.seq.items:
            messagebox.showwarning("Send", "Sequence is empty."); return
        prog = self._build_program(spacing_us)
        try:
            f_arb, npts = send_user_arb_over_lan(
                program=prog,
                host=SDG_HOST,
                channel="C1",
                name="ipm_gui",
                npoints=LAN_NPOINTS,
                high_v=5.0, low_v=0.0, load="HiZ"
            )
        except Exception as e:
            messagebox.showerror("Send to SDG (LAN)", f"Failed: {e}")
            return
        messagebox.showinfo("Sent",
            f"Wave uploaded & selected on CH1\n"
            f"Points: {npts}\n"
            f"Total duration: {prog.duration_us():.3f} µs\n"
            f"ARB frequency set to {f_arb:.2f} Hz\n"
            "Levels: High 5.0 V / Low 0.0 V / Load Hi-Z")

    # IOTA (CH2)
    def on_iota_send(self, gas_us: float, delay_ms: float):
        try:
            setup_iota_over_lan(gas_us=gas_us, delay_ms=delay_ms, host=SDG_HOST)
        except Exception as e:
            messagebox.showerror("IOTA (CH2)", f"Failed: {e}")
            return
        messagebox.showinfo("IOTA", "CH2 configured. Press CH2 Trigger (front panel) to fire one shot.")

    def on_iota_to_ipm(self, gas_us: float, delay_ms: float):
        self.on_iota_send(gas_us, delay_ms)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
