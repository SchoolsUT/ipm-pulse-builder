#!/usr/bin/env python3
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from typing import Dict, Any

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pulse_schedule import Program, ChargePWM, Gap
from export import export_sdg_csv
from send_over_lan import (
    setup_iota_over_lan,
    send_user_arb_over_lan,
    set_output,
    trigger_channel,
    ensure_outputs_off,
)
from calibration_io import list_load_ids, list_voltages_for_load, get_or_build_model
from predictor import predict_current_from_gate

from tek_scope import scope_capture_and_fetch, save_scope_csvs, save_scope_csv_combined

SDG_HOST = "192.168.60.2"
TEK_HOST = "192.168.70.2"

def f3(x: float) -> str: return f"{x:.3f}"

# =========================
# Left: Pulse Editor
# =========================
class PulseEditor(ttk.LabelFrame):
    def __init__(self, master, on_add_pulse, on_add_gap, on_update_sel,
                 on_start_delay_changed=None):
        super().__init__(master, text="Pulse Editor (Charging + PWM)")
        self._on_add_pulse = on_add_pulse
        self._on_add_gap = on_add_gap
        self._on_update_sel = on_update_sel
        self._on_start_delay_changed = on_start_delay_changed

        self.start_delay_us = tk.StringVar(value="0.0")
        self.v_charge = tk.StringVar(value="85.0")
        self.v_pwmw  = tk.StringVar(value="6.7")
        self.v_pwmp  = tk.StringVar(value="20.0")
        self.v_pwmn  = tk.StringVar(value="10")
        self.v_gap   = tk.StringVar(value="100.0")

        # keep SequenceList's Start Delay row in sync
        def _start_delay_trace(*_):
            if self._on_start_delay_changed is None:
                return
            try:
                val = float(self.start_delay_us.get() or 0.0)
            except Exception:
                return
            self._on_start_delay_changed(val)

        self.start_delay_us.trace_add("write", _start_delay_trace)

        g = ttk.Frame(self); g.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Pulse parameters
        rows = [
            ("Charge width (µs)", self.v_charge),
            ("PWM width (µs)",   self.v_pwmw),
            ("PWM period (µs)",  self.v_pwmp),
            ("PWM count",        self.v_pwmn),
        ]
        for i, (label, var) in enumerate(rows, start=0):
            ttk.Label(g, text=label).grid(row=i, column=0, sticky="w")
            ttk.Entry(g, textvariable=var, width=12).grid(row=i, column=1, sticky="ew")

        # Add / Update buttons
        btn_row = len(rows)
        ttk.Button(g, text="Add to Sequence", command=self._add_pulse).grid(row=btn_row, column=0, sticky="ew", pady=(8,0))
        ttk.Button(g, text="Update Selected", command=self._update_sel).grid(row=btn_row, column=1, sticky="ew", pady=(8,0))

        # GAP controls
        gap_row = btn_row + 1
        ttk.Label(g, text="Insert GAP (µs)").grid(row=gap_row, column=0, sticky="w", pady=(12,0))
        ttk.Entry(g, textvariable=self.v_gap, width=12).grid(row=gap_row, column=1, sticky="ew", pady=(12,0))
        ttk.Button(g, text="Insert Gap", command=self._add_gap).grid(row=gap_row+1, column=0, columnspan=2, sticky="ew")

        # Start Delay section (own little area below GAP)
        sd_row = gap_row + 2
        ttk.Label(g, text="Start Delay (µs)").grid(row=sd_row, column=0, sticky="w", pady=(12,0))
        ttk.Entry(g, textvariable=self.start_delay_us, width=12).grid(row=sd_row, column=1, sticky="ew", pady=(12,0))

        g.columnconfigure(1, weight=1)

    def get_start_delay_us(self) -> float:
        try:
            return float(self.start_delay_us.get() or 0.0)
        except Exception:
            return 0.0

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
        self.start_delay_us: float = 0.0  # display-only, Program.pre_gap_us lives in MainApp

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

        # Fixed Start Delay row at index 0
        self.lb.insert(0, "Start Delay 0.000 µs")
        try:
            self.lb.itemconfig(0, fg="gray")
        except tk.TclError:
            pass

    def set_start_delay_us(self, val: float) -> None:
        """Update the Start Delay display row (index 0)."""
        try:
            v = float(val)
        except Exception:
            v = 0.0
        if v < 0:
            v = 0.0
        self.start_delay_us = v
        label = f"Start Delay {f3(self.start_delay_us)} µs"
        if self.lb.size() == 0:
            self.lb.insert(0, label)
        else:
            self.lb.delete(0)
            self.lb.insert(0, label)
        try:
            self.lb.itemconfig(0, fg="gray")
        except tk.TclError:
            pass

    def _label(self, it: ChargePWM | Gap) -> str:
        if isinstance(it, Gap):
            return f"GAP {f3(it.gap_us)} µs"
        return f"Pulse  Charge {f3(it.charge_width_us)} µs | PWM {f3(it.pwm_width_us)}/{f3(it.pwm_period_us)} µs ×{it.pwm_count}"

    def add(self, it: ChargePWM | Gap) -> None:
        self.items.append(it)
        self.lb.insert(tk.END, self._label(it))  # goes after Start Delay row

    def update_selected(self, p: ChargePWM) -> None:
        sel = self.lb.curselection()
        if not sel: return
        idx = sel[0]
        if idx == 0:
            messagebox.showwarning("Update", "Select a pulse to update, not the Start Delay row.")
            return
        i = idx - 1
        if isinstance(self.items[i], Gap):
            messagebox.showwarning("Update", "Selected item is a GAP. Remove and re-insert to change.")
            return
        self.items[i] = p
        self.lb.delete(idx); self.lb.insert(idx, self._label(p)); self.lb.select_set(idx)

    def remove(self):
        sel = self.lb.curselection()
        if not sel: return
        idx = sel[0]
        if idx == 0:
            messagebox.showwarning("Remove", "Start Delay is fixed. Set it in the editor.")
            return
        i = idx - 1
        self.lb.delete(idx); self.items.pop(i)

    def clear(self):
        # preserve Start Delay row at 0
        if self.lb.size() > 1:
            self.lb.delete(1, tk.END)
        self.items.clear()

    def up(self):
        sel = self.lb.curselection()
        if not sel: return
        idx = sel[0]
        # idx 0 is Start Delay; idx 1 is first movable item and can't move up
        if idx <= 1: return
        i = idx - 1
        self.items[i-1], self.items[i] = self.items[i], self.items[i-1]
        txt = self.lb.get(idx); self.lb.delete(idx); self.lb.insert(idx-1, txt); self.lb.select_set(idx-1)

    def down(self):
        sel = self.lb.curselection()
        if not sel: return
        idx = sel[0]
        # can't move Start Delay; last movable item is at idx == len(items)
        if idx == 0 or idx >= len(self.items): return
        i = idx - 1
        self.items[i+1], self.items[i] = self.items[i], self.items[i+1]
        txt = self.lb.get(idx); self.lb.delete(idx); self.lb.insert(idx+1, txt); self.lb.select_set(idx+1)

    def _sel(self, _):
        sel = self.lb.curselection()
        if not sel: return
        idx = sel[0]
        if idx == 0:
            # clicking Start Delay does nothing
            return
        it = self.items[idx-1]
        if isinstance(it, ChargePWM):
            self.on_select(it)

# =========================
# Right: Options / IOTA / Scope
# =========================
class OptionsPane(ttk.LabelFrame):
    def __init__(self, master, on_preview, on_export, on_send,
                 on_iota_send, on_iota_to_ipm,
                 on_arm_changed=None, on_trigger=None,
                 on_cal_refresh=None, on_cal_load_changed=None, on_cal_voltage_changed=None,
                 on_scope_grab=None):
        super().__init__(master, text="Options / Preview / Send (CH1 = IPM)")
        self.on_preview, self.on_export, self.on_send = on_preview, on_export, on_send
        self.on_iota_send, self.on_iota_to_ipm = on_iota_send, on_iota_to_ipm
        self.on_arm_changed = on_arm_changed
        self.on_trigger = on_trigger
        self.on_cal_refresh = on_cal_refresh
        self.on_cal_load_changed = on_cal_load_changed
        self.on_cal_voltage_changed = on_cal_voltage_changed
        self.on_scope_grab = on_scope_grab

        self.spacing   = tk.StringVar(value="0.0")
        self.gas_us    = tk.StringVar(value="500.0")

        cal = ttk.Frame(self); cal.pack(fill=tk.X, padx=10, pady=(8,2))
        ttk.Label(cal, text="Load").grid(row=0, column=0, sticky="w")
        self.cb_load = ttk.Combobox(cal, state="readonly", width=16, values=[])
        self.cb_load.grid(row=0, column=1, sticky="ew", padx=(4,8))
        self.cb_load.bind("<<ComboboxSelected>>", lambda e: self.on_cal_load_changed and self.on_cal_load_changed(self.cb_load.get()))
        ttk.Label(cal, text="V").grid(row=0, column=2, sticky="w")
        self.cb_volt = ttk.Combobox(cal, state="readonly", width=8, values=[])
        self.cb_volt.grid(row=0, column=3, sticky="ew", padx=(4,8))
        self.cb_volt.bind("<<ComboboxSelected>>", lambda e: self.on_cal_voltage_changed and self.on_cal_voltage_changed(self.cb_volt.get()))
        ttk.Button(cal, text="Refresh", command=lambda: self.on_cal_refresh and self.on_cal_refresh()).grid(row=0, column=4, sticky="ew")
        cal.columnconfigure(1, weight=1)

        self.warn = ttk.Label(self, text="", foreground="#7a4d00", background="#fff3cd")
        self.warn.pack(fill=tk.X, padx=10, pady=(0,6))
        self.warn.pack_forget()

        g = ttk.Frame(self); g.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        ttk.Label(g, text="Spacing between blocks (µs)").grid(row=0, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.spacing, width=10).grid(row=0, column=1, sticky="ew")

        ttk.Button(g, text="Preview", command=self._preview).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8,0))
        ttk.Button(g, text="Export CSV (USB)", command=self._export).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(g, text="Send to SDG (LAN)", command=self._send).grid(row=3, column=0, columnspan=2, sticky="ew")

        ttk.Button(g, text="Grab Scope CSVs", command=self._scope_grab).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6,0))

        # Arm/Trigger (optional)
        row = 5
        if self.on_arm_changed is not None:
            self._armed = tk.BooleanVar(value=False)
            ttk.Checkbutton(g, text="Arm (CH1 Output)", variable=self._armed, command=self._toggle_arm)\
                .grid(row=row, column=0, columnspan=2, sticky="w", pady=(6,0))
            row += 1
            if self.on_trigger is not None:
                self.btn_trig = ttk.Button(g, text="Trigger (CH1)", command=self._trigger)
                self.btn_trig.grid(row=row, column=0, columnspan=2, sticky="ew")
                self.btn_trig.grid_remove()
                row += 1

        sep = ttk.Separator(g, orient="horizontal"); sep.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10,6))
        row += 1
        ttk.Label(g, text="IOTA (CH2)").grid(row=row, column=0, columnspan=2); row += 1

        ttk.Label(g, text="Gas Injection Duration (µs)").grid(row=row, column=0, sticky="w")
        ttk.Entry(g, textvariable=self.gas_us, width=10).grid(row=row, column=1, sticky="ew"); row += 1

        ttk.Button(g, text="Send IOTA (CH2)", command=self._iota_send).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6,0)); row += 1
        ttk.Button(g, text="IOTA → IPM (coarse via LAN)", command=self._iota_to_ipm).grid(row=row, column=0, columnspan=2, sticky="ew")

        g.columnconfigure(1, weight=1)

    def set_cal_lists(self, loads: list[str], volts: list[int], cur_load: str | None, cur_volt: int | None):
        self.cb_load["values"] = loads
        if cur_load in loads:
            self.cb_load.set(cur_load)
        elif loads:
            self.cb_load.set(loads[0])
        else:
            self.cb_load.set("")
        self.cb_volt["values"] = [str(v) for v in volts]
        if cur_volt and str(cur_volt) in self.cb_volt["values"]:
            self.cb_volt.set(str(cur_volt))
        elif volts:
            self.cb_volt.set(str(volts[0]))
        else:
            self.cb_volt.set("")

    def set_model_warning(self, visible: bool, text: str = ""):
        if visible:
            self.warn.configure(text=text)
            self.warn.pack(fill=tk.X, padx=10, pady=(0,6))
        else:
            self.warn.pack_forget()

    def _spacing(self) -> float:
        try: return float(self.spacing.get() or 0.0)
        except: return 0.0

    def _preview(self): self.on_preview(self._spacing())
    def _export(self):  self.on_export(self._spacing())
    def _send(self):    self.on_send(self._spacing())

    def _toggle_arm(self):
        armed = self._armed.get()
        try:
            if self.on_arm_changed: self.on_arm_changed(armed)
        except Exception as e:
            self._armed.set(not armed)
            messagebox.showerror("Arm (CH1)", str(e)); return
        if hasattr(self, "btn_trig"):
            if armed: self.btn_trig.grid()
            else: self.btn_trig.grid_remove()

    def _trigger(self):
        try:
            if self.on_trigger: self.on_trigger()
        except Exception as e:
            messagebox.showerror("Trigger (CH1)", str(e))

    def _iota_send(self):
        try:
            gas = float(self.gas_us.get())
            self.on_iota_send(gas)
        except Exception as e:
            messagebox.showerror("IOTA", str(e))

    def _iota_to_ipm(self):
        try:
            gas = float(self.gas_us.get())
            self.on_iota_to_ipm(gas)
        except Exception as e:
            messagebox.showerror("IOTA", str(e))

    def _scope_grab(self):
        try:
            if self.on_scope_grab:
                self.on_scope_grab()
        except Exception as e:
            messagebox.showerror("Tek Scope", str(e))

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
        t_us, y = program.sample(npoints)
        self.ax1.clear(); self.ax2.clear()
        self.ax1.plot(t_us, y)
        self.ax1.set_ylabel("IPM Input (±1)")
        self.ax1.set_ylim(-1.1, 1.1)
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

        # sequence first so editor can push start-delay updates into it
        self.seq = SequenceList(root, self._on_select)
        self.editor = PulseEditor(root, self._add_pulse, self._add_gap, self._update_sel,
                                  on_start_delay_changed=self.seq.set_start_delay_us)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.seq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.model_ok = False
        self.model: Dict[str, Any] | None = None
        self._ack_no_model = False
        self.cal_load = None
        self.cal_voltage = None

        self.opts = OptionsPane(
            root,
            self.on_preview, self.on_export, self.on_send,
            self.on_iota_send, self.on_iota_to_ipm,
            self.on_arm_changed, self.on_trigger,
            on_cal_refresh=self._cal_refresh,
            on_cal_load_changed=self._cal_load_changed,
            on_cal_voltage_changed=self._cal_voltage_changed,
            on_scope_grab=self.on_scope_grab,
        )
        self.opts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._cal_refresh()

        self.plot = PlotPane(self); self.plot.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        # initialize sequence Start Delay row from editor value
        self.seq.set_start_delay_us(self.editor.get_start_delay_us())

        try:
            ensure_outputs_off(SDG_HOST, prefer_channel="C1")
        except Exception as e:
            print(f"[INIT] Non-fatal: ensure_outputs_off failed: {e}")

    def _on_select(self, p: ChargePWM): self.editor.set_fields_from_pulse(p)
    def _add_pulse(self, p: ChargePWM): self.seq.add(p)
    def _add_gap(self, g: Gap): self.seq.add(g)
    def _update_sel(self, p: ChargePWM): self.seq.update_selected(p)

    def _build_program(self, spacing_us: float) -> Program:
        prog = Program()
        prog.pre_gap_us = max(0.0, self.editor.get_start_delay_us())
        first = True
        for it in self.seq.items:
            if not first and spacing_us > 0:
                prog.add(Gap(spacing_us))
            prog.add(it)
            first = False
        return prog

    def _cal_refresh(self):
        loads = list_load_ids()
        cur_load = self.cal_load if (self.cal_load in loads) else (loads[0] if loads else None)
        volts = list_voltages_for_load(cur_load) if cur_load else []
        cur_volt = self.cal_voltage if (self.cal_voltage in volts) else (volts[0] if volts else None)
        self.cal_load, self.cal_voltage = cur_load, cur_volt
        self.opts.set_cal_lists(loads, volts, cur_load, cur_volt)
        self._load_model_and_set_banner()

    def _cal_load_changed(self, load_id: str):
        self.cal_load = load_id or None
        volts = list_voltages_for_load(self.cal_load) if self.cal_load else []
        self.cal_voltage = volts[0] if volts else None
        self.opts.set_cal_lists(list_load_ids(), volts, self.cal_load, self.cal_voltage)
        self._load_model_and_set_banner()

    def _cal_voltage_changed(self, volt_str: str):
        try:
            self.cal_voltage = int(volt_str) if volt_str else None
        except Exception:
            self.cal_voltage = None
        self._load_model_and_set_banner()

    def _load_model_and_set_banner(self):
        self._ack_no_model = False
        if not self.cal_load or self.cal_voltage is None:
            self.model_ok, self.model = False, None
            self.opts.set_model_warning(True, "No calibration model selected. Preview predictor disabled.")
            return
        ok, res = get_or_build_model(self.cal_load, self.cal_voltage)
        if ok:
            self.model_ok, self.model = True, res
            self.opts.set_model_warning(False, "")
        else:
            self.model_ok, self.model = False, None
            self.opts.set_model_warning(
                True,
                f"No calibration model for {self.cal_load} @ {self.cal_voltage} V. "
                "Place a CSV in cal/<load>/<V>V/ (e.g., 600V.csv) to auto-build."
            )

    def on_preview(self, spacing_us: float):
        prog = self._build_program(spacing_us)
        self.plot.show_from_csv(prog)
        if self.model_ok and self.model:
            t_us, gate_y = prog.sample(4096)
            t_pred, I_pred = predict_current_from_gate(list(t_us), list(gate_y), self.model)
            self.plot.ax2.clear()
            self.plot.ax2.plot(t_pred, I_pred)
            self.plot.ax2.set_ylabel("Predicted Discharge Current (A)")
            self.plot.ax2.set_xlabel("Time (µs)")
            self.plot.canvas.draw_idle()

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
            "• Set ARB Frequency to the value above\n• Set High=5 V, Low=0 V, Load 50 Ω")

    def on_send(self, spacing_us: float):
        if not self._ack_no_model and not self.model_ok:
            if not messagebox.askyesno(
                "No calibration model",
                "No calibration model is loaded for the selected load/voltage.\n"
                "IPM safety limits still apply, but current prediction is disabled.\n\n"
                "Proceed anyway?"
            ):
                return
            self._ack_no_model = True
        if not self.seq.items:
            messagebox.showwarning("Send", "Sequence is empty."); return
        prog = self._build_program(spacing_us)
        try:
            f_arb, npts = send_user_arb_over_lan(
                program=prog,
                host=SDG_HOST,
                channel="C1",
                name="ipm_gui",
                high_v=5.0, low_v=0.0, load="50"
            )
        except Exception as e:
            messagebox.showerror("Send to SDG (LAN)", f"Failed: {e}")
            return
        messagebox.showinfo("Sent",
            f"Wave uploaded & selected on CH1\n"
            f"Points: {npts}\n"
            f"Total duration: {prog.duration_us():.3f} µs\n"
            f"ARB frequency set to {f_arb:.2f} Hz\n"
            "Levels: High 5.0 V / Low 0.0 V / Load 50 Ω")

    # IOTA (CH2)
    def on_iota_send(self, gas_us: float):
        try:
            setup_iota_over_lan(gas_us=gas_us, host=SDG_HOST)
        except Exception as e:
            messagebox.showerror("IOTA (CH2)", f"Failed: {e}")
            return
        messagebox.showinfo("IOTA", "CH2 configured. Press CH2 Trigger to fire one pulse.")

    def on_iota_to_ipm(self, gas_us: float):
        # Kept for UI parity; same as on_iota_send without any delay plumbing
        self.on_iota_send(gas_us)

    def on_arm_changed(self, armed: bool):
        try:
            set_output(SDG_HOST, "C1", on=armed)
        except Exception as e:
            raise RuntimeError(f"Failed to {'arm' if armed else 'disarm'} CH1: {e}")

    def on_trigger(self):
        trigger_channel(SDG_HOST, "C1")

    def on_scope_grab(self):
        out_dir = filedialog.askdirectory(title="Choose folder for Tek CSVs")
        if not out_dir:
            return
        data = scope_capture_and_fetch(
            host=TEK_HOST,
            sources=("CH1","CH2","CH3","CH4"),
            single_sequence=False,
        )
        save_scope_csvs(data, out_dir=out_dir, prefix="tek")
        combined = os.path.join(out_dir, "tek_all.csv")
        save_scope_csv_combined(data, combined, align="truncate")
        messagebox.showinfo("Tek Scope", f"Saved CSVs to:\n{out_dir}\n\nCombined file:\n{combined}")

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
