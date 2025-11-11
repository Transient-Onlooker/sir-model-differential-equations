import tkinter as tk
import os
import sys
import json
from scipy.integrate import solve_ivp
from tkinter import ttk, messagebox, Toplevel, font
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def seirs_differential(t, y, N, vaccination_start_day, variants, interventions):
    # --- Determine current parameters by layering variants up to time t ---
    effective_params = {}
    for variant in sorted(variants, key=lambda x: x['day']):
        if t >= variant['day']:
            effective_params.update(variant["params"])

    p = effective_params
    base_beta = p.get("beta", 0.2)
    gamma = p.get("gamma", 0.1)
    mu = p.get("mu", 0.01)
    incubation = p.get("incubation_period", 5)
    alpha = 1 / incubation if incubation > 0 else 0
    epsilon_v = p.get("epsilon_v", 0.95)
    epsilon_n = p.get("epsilon_n", 0.98)
    nat_dur = p.get("nat_dur", 180)
    omega_n = 1 / nat_dur if nat_dur > 0 else 0
    vax_dur = p.get("vax_dur", 365)
    omega_v = 1 / vax_dur if vax_dur > 0 else 0
    xi = p.get("xi", 0.01)
    xi_r_multiplier = p.get("xi_r_multiplier", 1.0)

    current_multiplier = 1.0
    for intervention in sorted(interventions, key=lambda x: x['day']):
        if t >= intervention['day']:
            current_multiplier = intervention['multiplier']
    beta = base_beta * current_multiplier

    # --- Original ODE Logic ---
    S, E, I, R, V, D = y
    current_xi_s = xi if t >= vaccination_start_day else 0
    current_xi_r = xi * xi_r_multiplier if t >= vaccination_start_day else 0

    infection_from_s = beta * S * I / N
    infection_from_v = beta * (1.0 - epsilon_v) * V * I / N
    infection_from_r = beta * (1.0 - epsilon_n) * R * I / N

    waning_n = omega_n * R
    waning_v = omega_v * V

    vax_from_s = current_xi_s * S
    vax_from_r = current_xi_r * R
    vax_from_e = current_xi_s * E

    dSdt = -infection_from_s - vax_from_s + waning_n + waning_v
    dEdt = infection_from_s + infection_from_v + infection_from_r - vax_from_e - alpha * E
    dIdt = alpha * E - (gamma + mu) * I
    dRdt = gamma * I - infection_from_r - vax_from_r - waning_n
    dVdt = vax_from_s + vax_from_r + vax_from_e - infection_from_v - waning_v
    dDdt = mu * I
    
    return [dSdt, dEdt, dIdt, dRdt, dVdt, dDdt]

class SeirsGuiApp(tk.Tk):
    def __init__(self, master):
        try:
            font_path = None
            for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
                if 'malgun' in font.lower(): # Windows Malgun Gothic
                    font_path = font
                    break
            if font_path:
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                plt.rcParams['font.family'] = font_name
            else:
                # Fallback for other systems (e.g., macOS, Linux with Nanum)
                for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
                    if 'nanumgothic' in font.lower():
                        font_path = font
                        break
                if font_path:
                    font_name = font_manager.FontProperties(fname=font_path).get_name()
                    plt.rcParams['font.family'] = font_name

            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"Font setting failed: {e}")

        self.master = master
        self.master.geometry("900x950")

        self.adv_settings_window = None
        self.variants_window = None
        self.interventions_window = None
        self.plot_window = None

        self.labels = {}
        self.virus_presets = {}
        self.load_labels()
        self.load_virus_presets()
        self.setup_vars()
        self.create_widgets()

    def load_labels(self, lang='ko'):
        filename = f"ui_labels_{lang}.json"
        path = resource_path(filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.labels = {}
        self.master.title(self.labels.get("window_title", "SEIRS Model Simulator"))

    def load_virus_presets(self):
        try:
            with open(resource_path('virus_presets.json'), 'r', encoding='utf-8') as f:
                self.virus_presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.virus_presets = {}

    def setup_vars(self):
        self.N_var = tk.StringVar(value="1000000")
        self.I0_var = tk.StringVar(value="10")
        self.sim_days_var = tk.StringVar(value="365")
        self.vax_start_day_var = tk.StringVar(value="30")

        self.use_variants_var = tk.BooleanVar(value=False)
        self.use_interventions_var = tk.BooleanVar(value=False)

        self.beta_var = tk.StringVar(value="20.0")
        self.gamma_var = tk.StringVar(value="10.0")
        self.mu_var = tk.StringVar(value="1.0")
        self.inc_var = tk.StringVar(value="5")
        self.nat_eff_var = tk.StringVar(value="90")
        self.nat_dur_var = tk.StringVar(value="180")
        self.vax_eff_var = tk.StringVar(value="95")
        self.vax_dur_var = tk.StringVar(value="365")
        self.xi_var = tk.StringVar(value="1.0")
        self.rec_vax_mult_var = tk.StringVar(value="1.5")

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Basic settings
        params_frame = ttk.LabelFrame(main_frame, text=self.labels.get("parameters_title", "Parameters"), padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        for i in [1, 3, 5, 7]:
            params_frame.grid_columnconfigure(i, weight=1)

        self.create_entry(params_frame, "population_label", self.N_var, 0, 0)
        self.create_entry(params_frame, "initial_infected_label", self.I0_var, 0, 2)
        self.create_entry(params_frame, "sim_duration_label", self.sim_days_var, 0, 4)
        self.create_entry(params_frame, "vax_start_day_label", self.vax_start_day_var, 0, 6)

        # Scenario settings
        scenario_frame = ttk.LabelFrame(main_frame, text="Scenarios", padding="10")
        scenario_frame.pack(fill=tk.X, pady=5)

        self.use_variants_check = ttk.Checkbutton(scenario_frame, text=self.labels.get("use_variants_label", "Use Variant Scenario"), variable=self.use_variants_var, command=self.toggle_scenario_buttons)
        self.use_variants_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.use_interventions_check = ttk.Checkbutton(scenario_frame, text=self.labels.get("use_interventions_label", "Use Intervention Scenario"), variable=self.use_interventions_var, command=self.toggle_scenario_buttons)
        self.use_interventions_check.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # Action buttons
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.X, pady=5)

        self.adv_settings_button = ttk.Button(action_frame, text=self.labels.get("adv_settings_button", "Advanced Settings"), command=self.open_settings_window)
        self.adv_settings_button.pack(side=tk.LEFT, padx=5)

        self.variants_button = ttk.Button(action_frame, text=self.labels.get("variants_button", "Variant Settings"), command=self.open_variants_window, state="disabled")
        self.variants_button.pack(side=tk.LEFT, padx=5)

        self.interventions_button = ttk.Button(action_frame, text=self.labels.get("interventions_button", "Intervention Settings"), command=self.open_interventions_window, state="disabled")
        self.interventions_button.pack(side=tk.LEFT, padx=5)

        self.run_button = ttk.Button(main_frame, text=self.labels.get("run_button", "Run Simulation"), command=self.run_simulation)
        self.run_button.pack(pady=10)

        # Plot frame
        self.plot_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_xlabel("Days")
        self.ax.set_ylabel("Population")
        self.ax.grid(True)
        self.fig.tight_layout()

    def create_entry(self, parent, label_key, var, row, col):
        ttk.Label(parent, text=self.labels.get(label_key, label_key)).grid(row=row, column=col, padx=5, pady=5, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=col + 1, padx=5, pady=5, sticky="ew")

    def toggle_scenario_buttons(self):
        is_variant_scenario = self.use_variants_var.get()
        self.adv_settings_button.config(state="disabled" if is_variant_scenario else "normal")
        self.variants_button.config(state="normal" if is_variant_scenario else "disabled")
        self.interventions_button.config(state="normal" if self.use_interventions_var.get() else "disabled")

    def open_settings_window(self):
        if self.adv_settings_window and self.adv_settings_window.winfo_exists():
            self.adv_settings_window.focus()
            return

        self.adv_settings_window = Toplevel(self.master)
        self.adv_settings_window.title(self.labels.get("adv_settings_title", "Advanced Settings"))
        
        settings_frame = ttk.Frame(self.adv_settings_window, padding="10")
        settings_frame.pack(fill=tk.BOTH, expand=True)

        # Virus Preset Selector
        preset_frame = ttk.LabelFrame(settings_frame, text=self.labels.get("virus_preset_label", "Virus Preset"), padding="10")
        preset_frame.pack(fill=tk.X, pady=5)
        
        preset_names = [self.labels.get("preset_default", "Default")] + list(self.virus_presets.keys())
        self.preset_var = tk.StringVar(value=preset_names[0])
        
        preset_menu = ttk.Combobox(preset_frame, textvariable=self.preset_var, values=preset_names, state="readonly")
        preset_menu.pack(fill=tk.X)

        # Parameter entries
        params_frame = ttk.LabelFrame(settings_frame, text=self.labels.get("parameters_title", "Parameters"), padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        params_frame.grid_columnconfigure(1, weight=1)
        params_frame.grid_columnconfigure(3, weight=1)

        entries = {
            "beta_label": self.beta_var, "gamma_label": self.gamma_var,
            "mu_label": self.mu_var, "inc_label": self.inc_var,
            "nat_eff_label": self.nat_eff_var, "nat_dur_label": self.nat_dur_var,
            "vax_eff_label": self.vax_eff_var, "vax_dur_label": self.vax_dur_var,
            "xi_label": self.xi_var, "rec_vax_mult_label": self.rec_vax_mult_var
        }
        
        row, col = 0, 0
        for label_key, var in entries.items():
            self.create_entry(params_frame, label_key, var, row, col)
            col += 2
            if col > 2:
                col = 0
                row += 1

        def on_preset_select(event):
            preset_name = self.preset_var.get()
            if preset_name != self.labels.get("preset_default", "Default"):
                preset_data = self.virus_presets.get(preset_name, {})
                self.beta_var.set(str(preset_data.get("beta", self.beta_var.get())))
                self.gamma_var.set(str(preset_data.get("gamma", self.gamma_var.get())))
                self.mu_var.set(str(preset_data.get("mu", self.mu_var.get())))
                self.inc_var.set(str(preset_data.get("incubation_period", self.inc_var.get())))
                self.nat_eff_var.set(str(preset_data.get("natural_immunity_effectiveness", self.nat_eff_var.get())))
                self.nat_dur_var.set(str(preset_data.get("natural_immunity_duration", self.nat_dur_var.get())))
                self.vax_eff_var.set(str(preset_data.get("vaccine_effectiveness", self.vax_eff_var.get())))
                self.vax_dur_var.set(str(preset_data.get("vaccine_duration", self.vax_dur_var.get())))
                self.xi_var.set(str(preset_data.get("cross_immunity", self.xi_var.get())))
                self.rec_vax_mult_var.set(str(preset_data.get("recovered_vaccine_multiplier", self.rec_vax_mult_var.get())))

        preset_menu.bind("<<ComboboxSelected>>", on_preset_select)

        ttk.Button(settings_frame, text=self.labels.get("save_button", "Save"), command=self.adv_settings_window.destroy).pack(pady=10)

    def open_variants_window(self):
        if self.variants_window and self.variants_window.winfo_exists():
            self.variants_window.focus()
            return
        self.variants_window = VariantEditor(self.master, self, self.labels, self.virus_presets)

    def open_interventions_window(self):
        if self.interventions_window and self.interventions_window.winfo_exists():
            self.interventions_window.focus()
            return
        self.interventions_window = InterventionEditor(self, self.labels)

    def bind_updates(self):
        self.pop_var.trace_add("write", self.update_rec_i0_label)

    def update_rec_i0_label(self, *args):
        try:
            N = int(self.pop_var.get())
            rec_i0 = max(1, int(N / 10000))
            self.rec_i0_var.set(f"~{rec_i0}")
        except (ValueError, tk.TclError):
            self.rec_i0_var.set("?")

    def open_interventions_window(self):
        if self.interventions_window and self.interventions_window.winfo_exists():
            self.interventions_window.focus()
            return
        # Assuming InterventionEditor exists and has a similar constructor
        self.interventions_window = InterventionEditor(self.master, self, self.labels)

    def run_simulation(self):
        try:
            # --- Always create base parameters from GUI ---
            custom_params = {
                "beta": float(self.beta_var.get()) / 100,
                "gamma": float(self.gamma_var.get()) / 100,
                "mu": float(self.mu_var.get()) / 100,
                "incubation_period": int(self.inc_var.get()),
                "epsilon_n": float(self.nat_eff_var.get()) / 100,
                "nat_dur": int(self.nat_dur_var.get()),
                "epsilon_v": float(self.vax_eff_var.get()) / 100,
                "vax_dur": int(self.vax_dur_var.get()),
                "xi": float(self.xi_var.get()) / 100,
                "xi_r_multiplier": float(self.rec_vax_mult_var.get())
            }
            
            # --- Build the final list of variants ---
            final_variants = [{"day": 0, "name": "Base Settings", "params": custom_params}]

            if self.use_variants_var.get():
                try:
                    with open(resource_path('variants.json'), 'r', encoding='utf-8') as f:
                        scenario_variants = json.load(f)
                    final_variants.extend(scenario_variants)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass

            # --- Load interventions ---
            interventions = []
            if self.use_interventions_var.get():
                try:
                    with open(resource_path('interventions.json'), 'r', encoding='utf-8') as f:
                        interventions = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass

            # --- Gather basic parameters from GUI ---
            N=int(self.N_var.get()); I0=int(self.I0_var.get()); days=int(self.sim_days_var.get()); vacc_start=int(self.vax_start_day_var.get())

            # --- Set initial conditions ---
            initial_variant_params = custom_params
            alpha = 1 / initial_variant_params.get("incubation_period", 5) if initial_variant_params.get("incubation_period", 5) > 0 else 0
            if alpha > 0:
                E0 = int((initial_variant_params.get("beta", 0.2) / alpha) * I0)
            else:
                E0 = 0

            S0 = N - I0 - E0
            if S0 < 0:
                messagebox.showerror(self.labels.get("error_input", "Input Error"), self.labels.get("error_population_small", "Initial infected/exposed count cannot exceed total population."))
                return
            
            # --- Run solver ---
            sol = solve_ivp(
                fun=seirs_differential, t_span=[0, days], y0=[S0, E0, I0, 0, 0, 0],
                args=(N, vacc_start, final_variants, interventions),
                t_eval=np.linspace(0, days, days + 1)
            )
            t, (S, E, I, R, V, D) = sol.t, sol.y

            # --- Plot results ---
            self.ax.clear()
            self.ax.plot(t, S, label=self.labels.get('susceptible_legend', 'Susceptible'))
            self.ax.plot(t, E, label=self.labels.get('exposed_legend', 'Exposed'), c='orange')
            self.ax.plot(t, I, label=self.labels.get('infected_legend', 'Infected'), c='r')
            self.ax.plot(t, R, label=self.labels.get('recovered_legend', 'Recovered'), c='g')
            self.ax.plot(t, V, label=self.labels.get('vaccinated_legend', 'Vaccinated'), c='purple')
            self.ax.plot(t, D, label=self.labels.get('deceased_legend', 'Deceased'), c='black')

            self.ax.set_title(self.labels.get("simulation_result_title", "SEIRS Model"))
            self.ax.set_xlabel(self.labels.get("sim_duration_label", "Days"))
            self.ax.set_ylabel(self.labels.get("population_label", "Population"))
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()

        except ValueError:
            messagebox.showerror(self.labels.get("error_input", "Input Error"), self.labels.get("error_valid_number", "Please enter valid numbers for all parameters."))
        except Exception as e:
            messagebox.showerror(self.labels.get("error_simulation", "Simulation Error"), f'{self.labels.get("error_simulation_text", "An error occurred during simulation.")}\n\n{e}')

class VariantEditor(Toplevel):
    def __init__(self, master, app, labels, presets):
        super().__init__(master)
        self.app = app
        self.parent = app  # For compatibility with existing code using self.parent
        self.labels = labels
        self.title(self.labels.get("variants_title", "Variant Scenario Editor"))
        self.geometry("800x600")

        self.variants = []
        self.editor_vars = {}
        self.editor_entries = {}
        self.current_edit_index = None
        self.virus_presets = presets

        self._setup_layout()
        self._setup_editor_widgets()
        self._load_initial_data()
        
        self.variant_listbox.bind('<<ListboxSelect>>', self._on_variant_select)
        if self.variant_listbox.size() > 0:
            self.variant_listbox.selection_set(0)
            self.variant_listbox.event_generate("<<ListboxSelect>>")

    def _setup_layout(self):
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame = ttk.Frame(main_pane); main_pane.add(left_frame, weight=2)
        right_frame = ttk.Frame(main_pane); main_pane.add(right_frame, weight=3)
        button_frame = ttk.Frame(self); button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        list_frame = ttk.LabelFrame(left_frame, text=self.labels.get("variants_list_title", "Variants"))
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.variant_listbox = tk.Listbox(list_frame, exportselection=False)
        self.variant_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.editor_frame = ttk.LabelFrame(right_frame, text=self.labels.get("variant_editor_pane_title", "Edit Variant"))
        self.editor_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(button_frame, text=self.labels.get("add_variant_button", "Add Variant"), command=self._add_variant).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=self.labels.get("remove_variant_button", "Remove Selected"), command=self._remove_variant).pack(side=tk.LEFT)
        ttk.Button(button_frame, text=self.labels.get("save_and_close_button", "Save & Close"), command=self._save_and_close).pack(side=tk.RIGHT)

    def _setup_editor_widgets(self):
        editor_content_frame = ttk.Frame(self.editor_frame)
        editor_content_frame.pack(fill=tk.X, padx=5, pady=5)

        preset_frame = ttk.Frame(editor_content_frame)
        preset_frame.pack(fill=tk.X, pady=(5, 10))
        ttk.Label(preset_frame, text="Virus Presets:").pack(side=tk.LEFT, padx=(0, 5))
        self.preset_combo = ttk.Combobox(preset_frame, values=list(self.virus_presets.keys()), state="readonly")
        self.preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_select)

        separator = ttk.Separator(editor_content_frame, orient='horizontal')
        separator.pack(fill='x', pady=10)

        params_frame = ttk.Frame(editor_content_frame)
        params_frame.pack(fill=tk.BOTH, expand=True)
        params_frame.grid_columnconfigure(1, weight=1)

        def create_entry(parent, text, var, row):
            ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=3)
            entry = ttk.Entry(parent, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
            return entry

        self.param_meta = [
            ('day', self.labels.get('variant_day_label', 'Day')),
            ('name', self.labels.get('variant_name_label', 'Variant Name')),
            ('beta', self.labels.get('infection_rate_label', 'Infection Rate (beta %)')),
            ('gamma', self.labels.get('recovery_rate_label', 'Recovery Rate (gamma %)')),
            ('mu', self.labels.get('mortality_rate_label', 'Mortality Rate (mu %)')),
            ('incubation_period', self.labels.get('incubation_period_label', 'Incubation (days)')),
            ('xi', self.labels.get('vax_rate_label', 'Daily Vax Rate (xi %)')),
            ('xi_r_multiplier', self.labels.get('rec_vax_multiplier_label', 'Recovered Vax Multiplier')),
            ('epsilon_v', self.labels.get('vax_efficacy_label', 'Vax Efficacy (%)')),
            ('vax_dur', self.labels.get('vax_immunity_dur_label', 'Vax Immunity (days)')),
            ('epsilon_n', self.labels.get('nat_immunity_eff_label', 'Natural Efficacy (%)')),
            ('nat_dur', self.labels.get('nat_immunity_dur_label', 'Natural Immunity (days)'))
        ]

        row = 0
        for key, text in self.param_meta:
            self.editor_vars[key] = tk.StringVar()
            entry = create_entry(params_frame, text, self.editor_vars[key], row)
            self.editor_entries[key] = entry
            self.editor_vars[key].trace_add("write", self._on_editor_update)
            row += 1

    def _on_preset_select(self, event):
        preset_name = self.preset_combo.get()
        if not preset_name or preset_name == "사용자 정의": return
        
        preset_data = self.virus_presets.get(preset_name, {})
        
        key_map = {
            'beta': 'beta', 'gamma': 'gamma', 'mu': 'mu',
            'incubation_period': 'incubation_period', 'xi': 'cross_immunity',
            'xi_r_multiplier': 'recovered_vaccine_multiplier',
            'epsilon_v': 'vaccine_effectiveness', 'vax_dur': 'vaccine_duration',
            'epsilon_n': 'natural_immunity_effectiveness', 'nat_dur': 'natural_immunity_duration'
        }

        # Temporarily remove traces to prevent feedback loop
        for var in self.editor_vars.values(): var.trace_remove('write', var.trace_info()[1])

        for key, _ in self.param_meta:
            if key in ['day', 'name']: continue
            
            preset_key = key_map.get(key)
            if not preset_key: continue

            value = preset_data.get(preset_key)
            if value is not None:
                if key in ['beta', 'gamma', 'mu', 'epsilon_v', 'epsilon_n', 'xi']:
                    self.editor_vars[key].set(f"{value * 100:.2f}")
                else:
                    self.editor_vars[key].set(value)
            else:
                # If a param is not in the preset, clear it
                self.editor_vars[key].set("")
        
        # Re-add traces
        for var in self.editor_vars.values(): var.trace_add("write", self._on_editor_update)
        
        # Trigger update to save the new values to the variant object
        self._on_editor_update()

    def _load_initial_data(self):
        try:
            base_params = {
                "beta": float(self.parent.beta_var.get()) / 100,
                "gamma": float(self.parent.gamma_var.get()) / 100,
                "mu": float(self.parent.mu_var.get()) / 100,
                "incubation_period": int(self.parent.inc_var.get()),
                "epsilon_n": float(self.parent.nat_eff_var.get()) / 100,
                "nat_dur": int(self.parent.nat_dur_var.get()),
                "epsilon_v": float(self.parent.vax_eff_var.get()) / 100,
                "vax_dur": int(self.parent.vax_dur_var.get()),
                "xi": float(self.parent.xi_var.get()) / 100,
                "xi_r_multiplier": float(self.parent.rec_vax_mult_var.get())
            }
            self.variants = [{'day': 0, 'name': 'Base Settings', 'params': base_params}]
        except (ValueError, tk.TclError):
            messagebox.showerror("Error", "The values in the main window's advanced settings are invalid.", parent=self)
            self.destroy()
            return

        try:
            with open(resource_path('variants.json'), 'r', encoding='utf-8') as f:
                self.variants.extend(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        self._update_listbox()

    def _update_listbox(self):
        selected_item = None
        if self.variant_listbox.curselection():
            selected_item = self.variants[self.variant_listbox.curselection()[0]]

        self.variants.sort(key=lambda v: v.get('day', 0))
        self.variant_listbox.delete(0, tk.END)
        
        new_selection_index = None
        for i, variant in enumerate(self.variants):
            self.variant_listbox.insert(tk.END, f"Day {variant.get('day', 0)}: {variant.get('name', 'Unnamed')}")
            if selected_item is variant:
                new_selection_index = i
        
        if new_selection_index is not None:
            self.variant_listbox.selection_set(new_selection_index)

    def _on_variant_select(self, event):
        selection_indices = self.variant_listbox.curselection()
        if not selection_indices:
            return
        
        self.current_edit_index = selection_indices[0]
        selected_variant = self.variants[self.current_edit_index]

        for var in self.editor_vars.values(): var.trace_remove('write', var.trace_info()[1])

        params = selected_variant.get('params', {})
        self.editor_vars['day'].set(selected_variant.get('day', ''))
        self.editor_vars['name'].set(selected_variant.get('name', ''))

        for key, _ in self.param_meta:
            if key in ['day', 'name']: continue
            value = params.get(key)
            if value is not None:
                if key in ['beta', 'gamma', 'mu', 'epsilon_v', 'epsilon_n', 'xi']:
                    self.editor_vars[key].set(f"{value * 100:.2f}")
                else:
                    self.editor_vars[key].set(value)
            else:
                self.editor_vars[key].set("")

        state = "disabled" if selected_variant.get('day') == 0 else "normal"
        self.editor_entries['day'].config(state=state)
        self.editor_entries['name'].config(state=state)
        self.preset_combo.set("사용자 정의")

        for var in self.editor_vars.values(): var.trace_add("write", self._on_editor_update)

    def _on_editor_update(self, *args):
        if self.current_edit_index is None: return

        variant = self.variants[self.current_edit_index]
        try:
            variant['day'] = int(self.editor_vars['day'].get())
            variant['name'] = self.editor_vars['name'].get()
        except (ValueError, TypeError): pass

        if 'params' not in variant: variant['params'] = {}

        for key, _ in self.param_meta:
            if key in ['day', 'name']: continue
            value_str = self.editor_vars[key].get()
            if value_str.strip() == "":
                if key in variant['params']: del variant['params'][key]
            else:
                try:
                    if key in ['beta', 'gamma', 'mu', 'epsilon_v', 'epsilon_n', 'xi']:
                        variant['params'][key] = float(value_str) / 100.0
                    elif key == 'xi_r_multiplier':
                        variant['params'][key] = float(value_str)
                    else:
                        variant['params'][key] = int(value_str)
                except ValueError: pass
        self._update_listbox()

    def _add_variant(self):
        new_day = 100
        if len(self.variants) > 0:
            new_day = self.variants[-1].get('day', 0) + 30
        
        new_variant = {'day': new_day, 'name': 'New Variant', 'params': {}}
        self.variants.append(new_variant)
        self._update_listbox()
        self.variant_listbox.selection_clear(0, tk.END)
        self.variant_listbox.selection_set(tk.END)
        self.variant_listbox.event_generate("<<ListboxSelect>>")

    def _remove_variant(self):
        if self.current_edit_index is None or self.current_edit_index == 0:
            messagebox.showerror("Error", "Cannot remove the Base Settings variant.", parent=self)
            return
        del self.variants[self.current_edit_index]
        self._update_listbox()
        new_selection = max(0, self.current_edit_index - 1)
        self.variant_listbox.selection_set(new_selection)
        self.variant_listbox.event_generate("<<ListboxSelect>>")

    def _save_and_close(self):
        base_params = self.variants[0]['params']
        for key, value in base_params.items():
            parent_var_name = {
                'beta': 'beta_var', 'gamma': 'gamma_var', 'mu': 'mu_var',
                'incubation_period': 'inc_var', 'xi': 'xi_var', 'xi_r_multiplier': 'rec_vax_mult_var',
                'epsilon_v': 'vax_eff_var', 'vax_dur': 'vax_dur_var', 'epsilon_n': 'nat_eff_var', 'nat_dur': 'nat_dur_var'
            }.get(key)

            if parent_var_name and hasattr(self.parent, parent_var_name):
                parent_var = getattr(self.parent, parent_var_name)
                if key in ['beta', 'gamma', 'mu', 'epsilon_v', 'epsilon_n', 'xi']:
                    parent_var.set(f"{value * 100:.2f}")
                else:
                    parent_var.set(value)

        scenario_variants = [v for v in self.variants if v.get('day', 0) > 0]

        try:
            path = resource_path('variants.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(scenario_variants, f, indent=4, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save variants.json:\n{e}", parent=self)
            return

        messagebox.showinfo(self.labels.get("save_success_title", "Success"), self.labels.get("save_variants_success_text", "Variant scenario saved successfully."), parent=self)
        self.parent.toggle_scenario_buttons() # Refresh main window button states
        self.destroy()

class InterventionEditor(Toplevel):
    def __init__(self, master, app, labels):
        super().__init__(master)
        self.app = app
        self.labels = labels
        self.title(self.labels.get("interventions_title", "Intervention Scenario Editor"))
        self.geometry("800x600")

        self.interventions = []
        self.editor_vars = {}
        self.editor_entries = {}
        self.current_edit_index = None

        self._setup_layout()
        self._setup_editor_widgets()
        self._load_initial_data()

        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        if self.listbox.size() > 0:
            self.listbox.selection_set(0)
            self.listbox.event_generate("<<ListboxSelect>>")

    def _setup_layout(self):
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame = ttk.Frame(main_pane); main_pane.add(left_frame, weight=2)
        right_frame = ttk.Frame(main_pane); main_pane.add(right_frame, weight=3)
        button_frame = ttk.Frame(self); button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        list_frame = ttk.LabelFrame(left_frame, text=self.labels.get("interventions_list_title", "Interventions"))
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_frame, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        editor_frame = ttk.LabelFrame(right_frame, text=self.labels.get("intervention_editor_pane_title", "Edit Intervention"))
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        editor_frame.grid_columnconfigure(1, weight=1)

        self.editor_vars['day'] = tk.StringVar()
        self.editor_vars['multiplier'] = tk.StringVar()
        self.editor_vars['reason'] = tk.StringVar()

        ttk.Label(editor_frame, text=self.labels.get("intervention_day_label", "Day:")).grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(editor_frame, textvariable=self.editor_vars['day']).grid(row=0, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(editor_frame, text=self.labels.get("intervention_multiplier_label", "Beta Multiplier:")).grid(row=1, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(editor_frame, textvariable=self.editor_vars['multiplier']).grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(editor_frame, text=self.labels.get("intervention_reason_label", "Reason:")).grid(row=2, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(editor_frame, textvariable=self.editor_vars['reason']).grid(row=2, column=1, sticky="ew", padx=5, pady=3)
        
        for var in self.editor_vars.values():
            var.trace_add("write", self._on_editor_update)

        ttk.Button(button_frame, text=self.labels.get("add_intervention_button", "Add"), command=self._add).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=self.labels.get("remove_intervention_button", "Remove"), command=self._remove).pack(side=tk.LEFT)
        ttk.Button(button_frame, text=self.labels.get("save_and_close_button", "Save & Close"), command=self._save_and_close).pack(side=tk.RIGHT)

    def _load_initial_data(self):
        try:
            with open(resource_path('interventions.json'), 'r', encoding='utf-8') as f:
                self.interventions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.interventions = []
        self._update_listbox()

    def _update_listbox(self):
        current_selection_index = self.listbox.curselection()[0] if self.listbox.curselection() else None
        self.interventions.sort(key=lambda v: v.get('day', 0))
        self.listbox.delete(0, tk.END)
        for item in self.interventions:
            self.listbox.insert(tk.END, f"Day {item.get('day', 0)}: {item.get('reason', 'Unnamed')}")
        if current_selection_index is not None and current_selection_index < self.listbox.size():
            self.listbox.selection_set(current_selection_index)

    def _on_select(self, event):
        selection_indices = self.listbox.curselection()
        if not selection_indices:
            self.current_edit_index = None
            return
        
        self.current_edit_index = selection_indices[0]
        item = self.interventions[self.current_edit_index]

        for var in self.editor_vars.values(): var.trace_remove('write', var.trace_info()[1])
        self.editor_vars['day'].set(item.get('day', ''))
        self.editor_vars['multiplier'].set(item.get('multiplier', ''))
        self.editor_vars['reason'].set(item.get('reason', ''))
        for var in self.editor_vars.values(): var.trace_add("write", self._on_editor_update)

    def _on_editor_update(self, *args):
        if self.current_edit_index is None: return
        item = self.interventions[self.current_edit_index]
        try:
            item['day'] = int(self.editor_vars['day'].get())
            item['multiplier'] = float(self.editor_vars['multiplier'].get())
            item['reason'] = self.editor_vars['reason'].get()
        except (ValueError, TypeError):
            pass
        self._update_listbox()

    def _add(self):
        new_day = 30
        if len(self.interventions) > 0:
            new_day = self.interventions[-1].get('day', 0) + 30
        new_item = {'day': new_day, 'multiplier': 1.0, 'reason': 'New Intervention'}
        self.interventions.append(new_item)
        self._update_listbox()
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(tk.END)
        self.listbox.event_generate("<<ListboxSelect>>")

    def _remove(self):
        if self.current_edit_index is None: return
        del self.interventions[self.current_edit_index]
        self._update_listbox()
        new_selection = max(0, self.current_edit_index - 1)
        if self.listbox.size() > 0:
            self.listbox.selection_set(new_selection)
            self.listbox.event_generate("<<ListboxSelect>>")
        else:
            self.current_edit_index = None
            for var in self.editor_vars.values(): var.set("")

    def _save_and_close(self):
        try:
            path = resource_path('interventions.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.interventions, f, indent=4, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save interventions.json:\n{e}", parent=self)
            return
        messagebox.showinfo(self.labels.get("save_success_title", "Success"), self.labels.get("save_interventions_success_text", "Intervention scenario saved successfully."), parent=self)
        self.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SeirsGuiApp(root)
    root.mainloop()
    app.mainloop()