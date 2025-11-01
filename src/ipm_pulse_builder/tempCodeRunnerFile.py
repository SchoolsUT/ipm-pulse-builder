"""
IPM Pulse Builder – GUI (Phase A)
---------------------------------
Clean, uncluttered Tkinter GUI for constructing IPM input pulses and arranging
them into a sequence. This GUI stays *model-only* (no hardware control here).

• Left:  Pulse Editor (one charging + PWM block → IPMInputPulse)
• Middle:Sequence List (ordered IPMInputPulse blocks with up/down/remove)
• Right: Sequence options (spacing µs), preview (gate + placeholder current), save/load preset

Dependencies: Tkinter, Matplotlib (for preview), and your local `pulse_schedule.py` module.

Run:
  python3 gui_app.py
"""
from __future__ import annotations
import json
from typing import List, Tuple
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Matplotlib embedding for previews
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import your model classes
try:
    from pulse_schedule import (
        IPMInputPulse,
        IPMPulseTrain,
        PulseProgram,
    )
except Exception as e:  # helpful error if module not found
    raise SystemExit("Could not import pulse_schedule.py. Ensure it is in the same folder.\n" + str(e))

# -----------------------------
# Small utilities
# -----------------------------
def f3(x: float) -> str:
    return f"{x:.3f}"

def parse_float(var: tk.StringVar, name: str) -> float:
    try:
        return float(var.get())
    except ValueError:
        raise ValueError(f"{name} must be a number")

def pulse_to_dict(p: IPMInputPulse) -> dict:
    return {
        "charge_width_us": p.charge_width_us,
        "pwm_width_us": p.pwm_width_us,
        "pwm_period_us": p.pwm_period_us,
        "pwm_count": p.pwm_count,
    }

def pulse_from_dict(d: dict) -> IPMInputPulse:
    return IPMInputPulse(
        charge_width_us=float(d.get("charge_width_us", 0.0)),
        pwm_width_us=float(d.get("pwm_width_us", 0.0)),
        pwm_period_us=float(d.get("pwm_period_us", 0.0)),
        pwm_count=int(d.get("pwm_count", 0)),
    )

# -----------------------------
# Pulse Editor (left panel)
# -----------------------------
class PulseEditor(ttk.LabelFrame):
    """Editor for a single IPMInputPulse (charge + PWM)."""
    def __init__(self, master, on_add, on_update):
        super().__init__(master, text="Pulse Editor (Charging + PWM)")
        self.on_add = on_add
        self.on_update = on_update

        self.charge_width_us = tk.StringVar(value="85.0")
        self.pwm_width_us    = tk.StringVar(value="6.7")
        self.pwm_period_us   = tk.StringVar(value="20.0")
        self.pwm_count       = tk.StringVar(value="10")

        grid = ttk.Frame(self)
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        r = 0
        ttk.Label(grid, text="Charge width (µs)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.charge_width_us, width=12).grid(row=r, column=1, sticky="ew")
        r += 1

        ttk.Label(grid, text="PWM width (µs)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.pwm_width_us, width=12).grid(row=r, column=1, sticky="ew")
        r += 1

        ttk.Label(grid, text="PWM period (µs)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.pwm_period_us, width=12).grid(row=r, column=1, sticky="ew")
        r += 1

        ttk.Label(grid, text="PWM count").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.pwm_count, width=12).grid(row=r, column=1, sticky="ew")
        r += 1

        btns = ttk.Frame(grid)
        btns.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8,0))
        ttk.Button(btns, text="Add to Sequence", command=self._handle_add).pack(side=tk.LEFT)
        ttk.Button(btns, text="Update Selected", command=self._handle_update).pack(side=tk.LEFT, padx=6)

        for c in (0,1):
            grid.columnconfigure(c, weight=1)

    def get_pulse(self) -> IPMInputPulse:
        cw = parse_float(self.charge_width_us, "Charge width")
        pw = parse_float(self.pwm_width_us, "PWM width")
        pp = parse_float(self.pwm_period_us, "PWM period")
        pc = int(parse_float(self.pwm_count, "PWM count"))
        if pw <= 0 or pp <= 0 or cw <= 0:
            raise ValueError("Widths and period must be > 0")
        if pw > pp:
            raise ValueError("PWM width must be ≤ period")
        if pc < 1:
            raise ValueError("PWM count must be ≥ 1")
        return IPMInputPulse(cw, pw, pp, pc)

    def set_pulse(self, p: IPMInputPulse):
        self.charge_width_us.set(f3(p.charge_width_us))
        self.pwm_width_us.set(f3(p.pwm_width_us))
        self.pwm_period_us.set(f3(p.pwm_period_us))
        self.pwm_count.set(str(p.pwm_count))

    def _handle_add(self):
        try:
            self.on_add(self.get_pulse())
        except Exception as e:
            messagebox.showerror("Add Pulse", str(e))

    def _handle_update(self):
        try:
            self.on_update(self.get_pulse())
        except Exception as e:
            messagebox.showerror("Update Pulse", str(e))

# -----------------------------
# Sequence List (middle panel)
# -----------------------------
class SequenceList(ttk.LabelFrame):
    """Shows an ordered list of IPMInputPulse blocks with simple controls."""
    def __init__(self, master, on_select_change):
        super().__init__(master, text="Sequence")
        self.on_select_change = on_select_change
        self.items: List[IPMInputPulse] = []

        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.listbox = tk.Listbox(frame, height=16)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        btns = ttk.Frame(frame)
        btns.pack(side=tk.LEFT, fill=tk.Y, padx=(8,0))
        ttk.Button(btns, text="↑", width=4, command=self.move_up).pack(pady=2)
        ttk.Button(btns, text="↓", width=4, command=self.move_down).pack(pady=2)
        ttk.Button(btns, text="Remove", command=self.remove).pack(pady=(8,2))
        ttk.Button(btns, text="Clear", command=self.clear).pack()

    def _on_select(self, _evt=None):
        idx = self.selected_index()
        if idx is not None:
            self.on_select_change(self.items[idx])

    def selected_index(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def append(self, p: IPMInputPulse):
        self.items.append(p)
        self.listbox.insert(tk.END, self._label(p))

    def update_selected(self, p: IPMInputPulse):
        idx = self.selected_index()
        if idx is None:
            raise ValueError("Select an item to update")
        self.items[idx] = p
        self.listbox.delete(idx)
        self.listbox.insert(idx, self._label(p))
        self.listbox.selection_set(idx)

    def move_up(self):
        idx = self.selected_index()
        if idx is None or idx == 0:
            return
        self.items[idx-1], self.items[idx] = self.items[idx], self.items[idx-1]
        txt = self.listbox.get(idx)
        self.listbox.delete(idx)
        self.listbox.insert(idx-1, txt)
        self.listbox.selection_set(idx-1)

    def move_down(self):
        idx = self.selected_index()
        if idx is None or idx == len(self.items)-1:
            return
        self.items[idx+1], self.items[idx] = self.items[idx], self.items[idx+1]
        txt = self.listbox.get(idx)
        self.listbox.delete(idx)
        self.listbox.insert(idx+1, txt)
        self.listbox.selection_set(idx+1)

    def remove(self):
        idx = self.selected_index()
        if idx is None:
            return
        self.items.pop(idx)
        self.listbox.delete(idx)

    def clear(self):
        self.items.clear()
        self.listbox.delete(0, tk.END)

    @staticmethod
    def _label(p: IPMInputPulse) -> str:
        return (
            f"Charge {f3(p.charge_width_us)} µs | "
            f"PWM {f3(p.pwm_width_us)}/{f3(p.pwm_period_us)} µs x{p.pwm_count}"
        )

# -----------------------------
# Preview Panel (bottom)
# -----------------------------
class PreviewPanel(ttk.LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Preview")
        self.fig = Figure(figsize=(8, 3.6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax1.set_ylabel("IPM Input")
        self.ax2.set_ylabel("Predicted Discharge Current (A)")
        self.ax2.set_xlabel("Time (µs)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plots(self, gate: Tuple[list, list], current: Tuple[list, list]):
        tg, yg = gate
        tc, ic = current
        self.ax1.clear(); self.ax2.clear()
        # guard equal-length plotting where appropriate
        try:
            self.ax1.step(tg, yg, where='post')
        except Exception:
            # fallback: plot points
            self.ax1.plot(tg, yg, drawstyle='steps-post')
        self.ax1.set_ylabel("IPM Input"); self.ax1.grid(True, alpha=0.3)
        # ensure tc/ic lengths match for plotting
        if len(tc) != len(ic):
            # simple resample/interpolate ic on tc if needed
            import numpy as _np
            if len(tc) > 1 and len(ic) > 0:
                xs = _np.linspace(tc[0], tc[-1], num=len(ic))
                ys = _np.interp(xs, tc, ic)
                self.ax2.plot(xs, ys)
            else:
                self.ax2.plot(tc, ic)
        else:
            self.ax2.plot(tc, ic)
        self.ax2.set_ylabel("Predicted Discharge Current (A)"); self.ax2.set_xlabel("Time (µs)"); self.ax2.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

# -----------------------------
# Sequence Options (right panel)
# -----------------------------
class SequenceOptions(ttk.LabelFrame):
    def __init__(self, master, get_items, on_preview):
        super().__init__(master, text="Sequence Options, Preview & Summary")
        self.get_items = get_items
        self.on_preview = on_preview

        self.spacing_us = tk.StringVar(value="0.0")
        self.summary = tk.StringVar(value="Total: 0 pulses, 0.00 µs")

        # Placeholder current model params
        self.model_enable = tk.BooleanVar(value=True)
        self.tau_on_us = tk.StringVar(value="50.0")
        self.tau_off_us = tk.StringVar(value="100.0")
        self.imax_a = tk.StringVar(value="100.0")

        # Simple source/load model for delivered gate voltage (e.g., 50 Ω source & 50 Ω IPM input)
        self.gen_volt = tk.StringVar(value="10.0")  # generator open-circuit (or Hi-Z) amplitude
        self.src_ohm = tk.StringVar(value="50.0")   # source impedance Ω
        self.load_ohm = tk.StringVar(value="50.0")  # load/input impedance Ω
        self.show_voltage = tk.BooleanVar(value=True)  # show IPM input as volts at load

        grid = ttk.Frame(self)
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        r = 0
        ttk.Label(grid, text="Spacing between blocks (µs)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.spacing_us, width=10).grid(row=r, column=1, sticky="ew")
        r += 1

        ttk.Separator(grid, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
        r += 1

        ttk.Checkbutton(grid, text="Predict current (placeholder)", variable=self.model_enable).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Label(grid, text="τ_on (µs)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.tau_on_us, width=10).grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(grid, text="τ_off (µs)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.tau_off_us, width=10).grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(grid, text="I_max (A)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.imax_a, width=10).grid(row=r, column=1, sticky="ew")
        r += 1

        ttk.Separator(grid, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
        r += 1

        ttk.Label(grid, text="Generator V (open-circuit)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.gen_volt, width=10).grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(grid, text="Source Z (Ω)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.src_ohm, width=10).grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Label(grid, text="Load Z (Ω)").grid(row=r, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.load_ohm, width=10).grid(row=r, column=1, sticky="ew")
        r += 1
        ttk.Checkbutton(grid, text="Show IPM input as volts at load", variable=self.show_voltage).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        ttk.Separator(grid, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
        r += 1

        ttk.Button(grid, text="Build Summary", command=self.recalc).grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8,0))
        r += 1
        ttk.Button(grid, text="Preview", command=self.preview).grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        ttk.Label(grid, textvariable=self.summary, foreground="#333", anchor="w").grid(row=r, column=0, columnspan=2, sticky="w", pady=(6,2))
        r += 1

        # Save/Load preset
        ttk.Button(grid, text="Save Preset", command=self.save_preset).grid(row=r, column=0, sticky="ew", pady=(12,0))
        ttk.Button(grid, text="Load Preset", command=self.load_preset).grid(row=r, column=1, sticky="ew", pady=(12,0))

        for c in (0,1):
            grid.columnconfigure(c, weight=1)

    def recalc(self):
        try:
            spacing = float(self.spacing_us.get())
        except ValueError:
            messagebox.showerror("Spacing", "Spacing must be a number")
            return
        items = self.get_items()
        train = IPMPulseTrain(spacing_us=spacing)
        for p in items:
            train.add(p)
        program: PulseProgram = train.to_program()
        stats = program.stats()
        self.summary.set(
            f"Blocks: {len(items)} | Total duration: {f3(stats['total_duration_us'])} µs | "
            f"Avg duty: {stats['avg_duty']:.3f}"
        )

    def preview(self):
        # Pass preview parameters to main app
        try:
            params = {
                'spacing_us': float(self.spacing_us.get() or 0.0),
                'model_enable': bool(self.model_enable.get()),
                'tau_on_us': float(self.tau_on_us.get() or 0.0),
                'tau_off_us': float(self.tau_off_us.get() or 0.0),
                'imax_a': float(self.imax_a.get() or 0.0),
                # voltage-delivery model
                'gen_volt': float(self.gen_volt.get() or 0.0),
                'src_ohm': float(self.src_ohm.get() or 0.0),
                'load_ohm': float(self.load_ohm.get() or 0.0),
                'show_voltage': bool(self.show_voltage.get()),
            }
        except ValueError:
            messagebox.showerror("Preview", "Preview parameters must be numbers")
            return
        self.on_preview(params)

    def save_preset(self):
        items = self.get_items()
        data = {
            "spacing_us": float(self.spacing_us.get() or 0.0),
            "blocks": [pulse_to_dict(p) for p in items],
        }
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save", f"Failed to save preset: {e}")
            return
        messagebox.showinfo("Save", f"Saved preset → {path}")

    def load_preset(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All","*.*")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load", f"Failed to read file: {e}")
            return
        try:
            self.spacing_us.set(str(float(data.get("spacing_us", 0.0))))
            blocks = [pulse_from_dict(d) for d in data.get("blocks", [])]
        except Exception as e:
            messagebox.showerror("Load", f"Invalid preset: {e}")
            return
        # Delegate update back to main app which owns the list widget
        self._loaded_blocks = blocks  # consumed by MainApp.on_preset_loaded
        # generate virtual event that the owner (MainApp) is expected to be bound to
        self.event_generate("<<PresetLoaded>>", when="tail")

# -----------------------------
# Main App
# -----------------------------
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IPM Pulse Builder")
        self.geometry("1200x720")

        # Top-level container
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        # Left: editor
        self.editor = PulseEditor(root, on_add=self.on_add_pulse, on_update=self.on_update_pulse)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Middle: sequence list
        self.sequence = SequenceList(root, on_select_change=self.editor.set_pulse)
        self.sequence.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Right: options + preview controls
        self.options = SequenceOptions(root, get_items=lambda: self.sequence.items, on_preview=self.on_preview)
        self.options.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.options.bind("<<PresetLoaded>>", self.on_preset_loaded)

        # Bottom: live preview (two stacked plots)
        self.preview = PreviewPanel(self)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        self._build_menu()

    def _build_menu(self):
        menubar = tk.Menu(self)
        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="About", command=lambda: messagebox.showinfo(
            "About",
            "IPM Pulse Builder – clean editor for charging+PWM blocks.\n"
            "Save/Load presets; sequence builder + preview only."
        ))
        menubar.add_cascade(label="Help", menu=helpm)
        self.config(menu=menubar)

    # --- callbacks ---
    def on_add_pulse(self, p: IPMInputPulse):
        self.sequence.append(p)
        self.options.recalc()

    def on_update_pulse(self, p: IPMInputPulse):
        self.sequence.update_selected(p)
        self.options.recalc()

    def on_preset_loaded(self, _evt=None):
        blocks = getattr(self.options, "_loaded_blocks", None)
        if blocks is None:
            return
        self.sequence.clear()
        for p in blocks:
            self.sequence.append(p)
        self.options.recalc()
        self.options._loaded_blocks = None

    # --- preview orchestration ---
    def on_preview(self, params: dict):
        """Build program from current list + spacing, then draw gate & current.
        Robust to empty sequences and shows a dialog on unexpected errors.
        """
        try:
            items = self.sequence.items
            if not items:
                messagebox.showinfo("Preview", "Add at least one IPM pulse to the sequence first.")
                return

            train = IPMPulseTrain(spacing_us=float(params.get('spacing_us', 0.0)))
            for p in items:
                train.add(p)
            program: PulseProgram = train.to_program()

            # Gate timeseries (0/1)
            dt = self._auto_dt(program)
            t_gate, y_gate = program.to_timeseries(dt_us=dt)

            # Convert gate to delivered V at load if requested
            if params.get('show_voltage', True):
                Vs = max(0.0, float(params.get('gen_volt', 0.0)))
                Rs = max(0.0, float(params.get('src_ohm', 0.0)))
                Rl = max(0.0, float(params.get('load_ohm', 0.0)))
                v_scale = (Rl / (Rs + Rl)) if (Rs + Rl) > 0 else 0.0
                y_gate = [v * Vs * v_scale for v in y_gate]  # 0/1 → volts at IPM input

            # Placeholder current model
            if params.get('model_enable', True):
                tau_on = float(params.get('tau_on_us', 50.0))
                tau_off = float(params.get('tau_off_us', 100.0))
                # guard against zero/negative taus
                tau_on = max(1e-6, tau_on)
                tau_off = max(1e-6, tau_off)
                t_cur, i_cur = self._predict_current_from_windows(
                    program.schedule(),
                    dt_us=dt,
                    tau_on_us=tau_on,
                    tau_off_us=tau_off,
                    imax=float(params.get('imax_a', 100.0))
                )
                # align current timebase with gate timebase if needed
                if len(t_cur) != len(t_gate):
                    # simple linear interpolation of i_cur to t_gate
                    import numpy as _np
                    i_cur = list(_np.interp(t_gate, t_cur, i_cur))
                    t_cur = t_gate
            else:
                t_cur, i_cur = t_gate, [0.0]*len(t_gate)

            self.preview.update_plots((t_gate, y_gate), (t_cur, i_cur))

            # Append delivered V example to summary
            if params.get('show_voltage', True):
                Rs = max(0.0, float(params.get('src_ohm', 0.0)))
                Rl = max(0.0, float(params.get('load_ohm', 0.0)))
                Vs = max(0.0, float(params.get('gen_volt', 0.0)))
                vload = Vs * (Rl / (Rs + Rl)) if (Rs + Rl) > 0 else 0.0
                self.options.summary.set(self.options.summary.get() + f" | V_load≈{vload:.2f} V when gate=1")

        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    @staticmethod
    def _auto_dt(program: PulseProgram) -> float:
        # choose a dt that resolves the shortest pulse with ~20 samples
        windows = program.schedule()
        if not windows:
            return 1.0
        min_width = min((e - s) for s, e in windows)
        # avoid extremely small dt and ensure positive
        dt = max(1e-3, min_width / 20.0)
        return dt

    @staticmethod
    def _predict_current_from_windows(windows: List[Tuple[float, float]], dt_us: float,
                                      tau_on_us: float, tau_off_us: float, imax: float) -> Tuple[list, list]:
        # Simple first-order rise/decay placeholder model
        if not windows:
            return [0.0], [0.0]
        T = windows[-1][1]
        n = int(max(1, round(T / dt_us)))
        t = [i * dt_us for i in range(n + 1)]
        i = [0.0]*(n+1)
        wi = 0
        for k, tk in enumerate(t[1:], start=1):
            # Advance window index
            while wi < len(windows) and tk > windows[wi][1]:
                wi += 1
            on = (wi < len(windows)) and (windows[wi][0] <= tk <= windows[wi][1])
            if on:
                di = (imax - i[k-1]) * (dt_us / tau_on_us)
            else:
                di = -i[k-1] * (dt_us / tau_off_us)
            i[k] = max(0.0, i[k-1] + di)
        return t, i

# -----------------------------
# Main
#