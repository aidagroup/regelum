# Prompt For Reproducing Dubuisson 2019 Fig. 9 In Regelum

Ты работаешь в репозитории `/home/user/Repos/regelum`.

Цель: честно воспроизвести Dubuisson et al. 2019 Fig. 9 на Regelum как физическую Simulink-like модель standalone microgrid, а не как подгонку графиков и не как перерисовку оцифрованных target-кривых.

Итоговые графики должны строиться только из честного simulation trace после `PhasedReactiveSystem.run(...)`. Все токи, напряжения, частота, SOC и мощности на Fig. 9 должны быть выходами физических нод модели.

## Important PDF Reading Rule

PDF надо читать как набор картинок/страниц, а не только как текст.

Формулы, блок-схемы, подписи на схемах и таблицы в этой статье плохо и неполно извлекаются текстовым парсером. Поэтому перед инженерной реализацией обязательно:

- отрендерить нужные страницы PDF в PNG с достаточным DPI;
- визуально прочитать Fig. 1, Fig. 2, Fig. 3, Fig. 4, Fig. 5, Fig. 6, Fig. 7, Fig. 8, Fig. 9 и Table II;
- проверять формулы по изображению страницы, а не доверять OCR/текстовому извлечению;
- при сомнении смотреть саму картинку страницы.

Основной PDF:

- `references/dubuisson2019.pdf`

Существующие изображения/артефакты могут быть в:

- `artifacts/dubuisson2019/pdf_pages/`
- `artifacts/dubuisson2019/simulink_reconstruction/`

Если PNG страниц отсутствуют или плохого качества, отрендерить заново, например через `pdftoppm`.

## What To Read First

Прочитай и визуально проверь:

1. `references/dubuisson2019.pdf`
2. Fig. 1: topology standalone microgrid.
3. Fig. 2: DG governor/AVR control.
4. Fig. 3: two-level inverter control.
5. Fig. 4: DC-DC buck-boost battery controller.
6. Fig. 5: WT MPPT.
7. Fig. 6: power flow supervisor.
8. Fig. 7-8: DC voltage regulation transfer function and PI gains.
9. Fig. 9: target scenario/result layout.
10. Appendix Table II: full system parameters.
11. Existing code:
    - `examples/dubuisson2019_simulink_reconstruction.py`
    - `examples/dubuisson2019_water_treatment.py`
    - `src/regelum/examples/dubuisson2019_simulink_reconstruction.py`
    - `references/dubuisson2019_targets/README.md`
    - `references/dubuisson2019_targets/*.csv`

## Hard Rule: What Can Be External

External scenario inputs allowed:

- load demand profile reconstructed from the load/load-current scenario in Fig. 9;
- wind scenario, preferably wind speed / mechanical available power / available WT input reconstructed from the article scenario;
- initial conditions, if needed and documented.

External simulated outputs are not allowed.

Do not inject these from digitized traces:

- battery current;
- WT current;
- inverter current;
- DC bus voltage;
- SOC;
- frequency;
- load voltage magnitude;
- DG current.

Those channels are validation targets only. They must be computed by the physical model.

## Required Architecture

Implement or repair the reconstruction in a separate dedicated file, not by mixing hacks into unrelated examples.

Use:

- `examples/dubuisson2019_simulink_reconstruction.py` for the runnable example, or create a clearly named new dedicated file if needed;
- keep plotting/output code isolated enough that simulation logic remains readable;
- save output under `artifacts/dubuisson2019/simulink_reconstruction/`.

The core model must be a Regelum `PhasedReactiveSystem` made of physical nodes matching Fig. 1.

## Node Specification

### 1. ScenarioProfile Node

Purpose: external disturbances only.

Inputs/outputs:

- simulation time;
- load demand profile `P_load(t)` or equivalent load scenario;
- wind input profile, preferably wind speed or available mechanical power.

This node must not provide simulated outputs like `I_bat`, `Vdc`, `SOC`, `frequency`, etc.

### 2. WindTurbinePMSG Node

Use Appendix Table II WECS parameters:

- `P = 50 kW`;
- `Vdc = 288 V`;
- `omega = 12500 rpm`;
- `Rs = 0.0041 Ohm`;
- `Ld = 8.7079e-05 H`;
- `Lq = 1.4634e-04 H`;
- `flux linkage = 0.07 V.s`;
- `J = 0.089 kg.m^2`;
- `F = 0.005 N.m.s`;
- `Tf = 4 N.m`.

Implement PMSG dq dynamics and mechanical rotor dynamics.

Expected physical relationships:

- dq stator voltage/current equations for PMSG;
- electromagnetic torque from dq currents and flux linkage;
- rotor mechanical equation using inertia, friction, turbine torque and electromagnetic torque;
- generated electrical power from dq states.

Outputs:

- PMSG dq currents;
- PMSG dq voltages or equivalent terminal quantities;
- electromagnetic torque;
- rotor speed;
- electrical power into rectifier.

If a reduced averaged PMSG is used temporarily, document exactly what was reduced and why. The final engineering target should include real states, not only a power-source surrogate.

### 3. DiodeRectifier Node

Purpose: represent PMSG AC to DC rectification from Fig. 1.

Inputs:

- PMSG terminal voltages/currents or equivalent dq power.

Outputs:

- rectified DC voltage/current/power into the WT boost converter.

Acceptable implementation:

- full diode switching model, or
- averaged rectifier model with energy conservation and documented assumptions.

Not acceptable:

- arbitrary `wind_available_kw -> dc_power_kw` without rectifier physics.

### 4. WTBoostConverter Node

Match Fig. 5 MPPT strategy.

States:

- boost inductor current;
- duty `d`;
- previous `V_DC-WT`;
- previous `I_DC-WT`;
- previous `P_WT`.

MPPT equations from the article:

- `P_WT(n) = V_WT(n) * I_WT(n)`;
- `P_WT(n-1) = V_WT(n-1) * I_WT(n-1)`;
- sign of `P_WT(n) - P_WT(n-1)` selects `k = +1` or `k = -1`;
- `d_next = d_prev + k * dD`;
- `dD = 0.5%`.

Physical averaged boost converter:

- include boost inductor current state;
- integrate inductor current honestly;
- compute output current into common DC bus from converter equations/power conservation;
- no manual shaping to match Fig. 9.

Outputs:

- WT DC current into common DC bus;
- WT DC power;
- boost duty;
- boost inductor current.

### 5. BatteryThevenin Node

Use Appendix Table II battery parameters exactly:

- nominal voltage `250 V`;
- cut-off voltage `187.5 V`;
- full charge voltage `286 V`;
- energy capacity `100 kWh`;
- nominal discharge current `80 A`;
- internal resistance `0.00625 Ohm`.

States:

- SOC;
- open circuit voltage;
- terminal voltage.

Equations:

- SOC derivative from measured battery current and `100 kWh` capacity;
- `V_terminal = OCV(SOC) - R_int * I_bat` with a clearly documented sign convention.

Outputs:

- `SOC`;
- `V_bat`;
- measured `I_bat`;
- battery power.

Do not change battery capacity to force DG-off timing unless the change is explicitly documented as a calibration experiment. The default engineering model must use Table II.

### 6. BatteryBuckBoostConverter Node

Match Fig. 4 and equations (8)-(9).

Inputs:

- `V_DCref`;
- measured `V_DC`;
- measured `I_BAT`;
- battery terminal voltage;
- common DC bus voltage.

Outer voltage PI:

- `I_BATref = (kp1 + ki1/s) * (V_DCref - V_DC)`.

Inner current PI:

- `d = (kp2 + ki2/s) * (I_BATref - I_BAT)`.

Use published DC voltage loop values from Fig. 7-8:

- `Cdc = 500e-6 F`;
- `zeta = 0.7`;
- `omega = 439.82 rad/s`;
- `kp = 0.3079`;
- `ki = 96.7208`.

If inner current PI gains are not published, estimate them physically and document:

- bandwidth;
- damping;
- saturation;
- anti-windup;
- why the selected gains are reasonable.

Physical averaged buck-boost:

- include inductor current state;
- integrate converter current honestly;
- compute battery current from converter state;
- compute DC-link current contribution from converter equations.

Important:

- A power-supervisor feedforward may be added only if it is explicitly modeled as a supervisor/current scheduling layer and documented.
- It must not replace the Fig. 4 PI loop.

Outputs:

- battery current;
- battery power;
- duty;
- `I_BATref`;
- PI states;
- DC-link current contribution.

### 7. DcLinkCapacitor Node

Use Appendix/Fig. 7 value:

- `Cdc = 500e-6 F`.

Physical equation:

- `Cdc * dVdc/dt = I_wt + I_battery - I_vsi - I_dump`;

or equivalently:

- `Cdc * Vdc * dVdc/dt = P_wt + P_battery - P_vsi - P_dump`.

Requirements:

- integrate this directly;
- no arbitrary decay layer;
- no postprocessed target correction;
- protection clamps are allowed only as physical saturation/protection and must be documented.

Outputs:

- `Vdc`;
- net DC current;
- net DC power.

### 8. VoltageSourceInverter Node

Match Fig. 3 and equations (1)-(7).

Implement:

- `abc -> dq0` Park transform from equation (1);
- PLL/unit templates from equations (3)-(6);
- first-level control when DG is ON:
  - compute `Id_LOAD`;
  - use `Id_DG_ref`;
  - compute `Id_INV_ref` from equation (2);
  - force q-component toward zero;
- second-level control when DG is OFF:
  - `I_INV_ref = (kp + ki/s)(V_ref - V_INV) - I_INV` from equation (7);
- current PI/hysteresis/PWM or a documented averaged equivalent.

Outputs:

- three-phase inverter voltages;
- three-phase inverter currents;
- dq currents;
- inverter current magnitude for Fig. 9;
- inverter DC current/power drawn from DC link.

Important:

- Fig. 9 inverter current must be measured from simulated inverter/filter currents.
- Do not compute inverter current directly from target power as a plotting shortcut.

### 9. LCFilterTransformer Node

Represent the filter and transformer shown in Fig. 1.

States:

- filter inductor currents;
- filter capacitor voltages.

Equations:

- `L * di/dt = v_inv - v_cap - R*i`;
- `C * dv/dt = i_L - i_load - i_damping/leakage`;
- transformer relation according to the chosen model.

Outputs:

- PCC voltage;
- load voltage waveform;
- load voltage RMS/magnitude;
- inverter/filter current.

If exact filter/transformer parameters are not published, choose physically reasonable values and document them as estimated/calibrated.

### 10. Load Node

Use the load scenario as external input, but compute current from PCC voltage.

Acceptable load models:

- constant power load:
  - `i_load_abc = P_load * v_abc / sum(v_abc^2)`;
- impedance load if better supported by the article.

Outputs:

- load current waveform;
- load current magnitude for Fig. 9;
- consumed power.

### 11. DieselSynchronousGenerator Node

Use Appendix Table II SG parameters:

- `Sn = 52.5 kVA`;
- `Vn = 460 V`;
- `fs = 60 Hz`;
- `2P = 4`;
- `Rs = 0.0181 Ohm`;
- `Ll = 0.0009622 H`;
- `Lmd = 0.02683 H`;
- `Lmq = 0.01187 H`;
- `J = 0.3987 kg.m^2`;
- `F = 0.031 N.m.s`.

Implement at least a reduced dq synchronous generator model. Full model is preferred.

Include Fig. 2 controls:

- governor compares measured speed to `omega_ref`;
- governor PI controls mechanical torque/fuel;
- AVR compares measured PCC voltage to `V_ref`;
- AVR PI controls excitation/field voltage;
- breaker/enabled state controlled by supervisor.

Outputs:

- DG phase currents;
- `I_DG` for Fig. 9;
- DG active/reactive power;
- rotor speed;
- terminal voltage;
- excitation/governor states;
- breaker state.

Not acceptable as final model:

- `P_DG = 50 kW if enabled else 0` with no SG/AVR/governor dynamics, unless explicitly marked as an intermediate reduced model.

### 12. PowerFlowSupervisor Node

Match Fig. 6 and Table I.

Rules from the article:

- DG starts when `SOC < 50%`;
- DG stops when `SOC >= 70%`;
- if `P_WT > P_LOAD` and `SOC > 50%`, energy source is WT and battery charges;
- if `P_WT < P_LOAD` and `SOC > 50%`, energy sources are WT + battery and battery discharges;
- if `P_WT < P_LOAD` and `SOC < 50%`, energy source is DG and battery charges;
- if `P_WT > P_LOAD` and `SOC < 50%`, energy sources are DG + WT and battery fast charges;
- dump load is used only when `SOC = 100%` and WT power exceeds load.

Implement:

- mode state;
- DG breaker;
- AVR/governor enable;
- timer delays from Fig. 6 if present;
- controlled switch logic.

Outputs:

- current mode;
- DG enabled;
- AVR/governor enabled;
- dump load enabled;
- breaker state.

### 13. DumpLoad Node

Represent dump load from Fig. 1.

Inputs:

- supervisor dump-load command;
- DC bus voltage;
- excess WT power if needed.

Outputs:

- dump-load current;
- dump-load power.

This is needed for completeness even if Fig. 9 does not enter the dump-load case.

### 14. Logger Node

Log every quantity needed for Fig. 9 and debugging:

- time;
- mode;
- load voltage waveform;
- load voltage magnitude;
- load current;
- DG current;
- inverter current;
- battery current;
- WT current;
- DC bus voltage;
- SOC;
- frequency;
- WT power;
- load power;
- battery power;
- DG power;
- inverter AC/DC power;
- DC net current/power;
- controller references and duty cycles.

## Plotting Requirements

Generate Fig. 9 from simulation trace only.

Use matplotlib and save PDF:

- `artifacts/dubuisson2019/simulink_reconstruction/fig9_reconstruction.pdf`

Also save trace:

- `artifacts/dubuisson2019/simulink_reconstruction/fig9_reconstruction_trace.csv`

Optional PNG preview:

- `artifacts/dubuisson2019/simulink_reconstruction/fig9_reconstruction-1.png`

Do not save final plots as SVG.

## Verification Targets

Compare the generated result against article Fig. 9.

Important event times:

- `t = 7s`: load changes;
- `t = 8s`: wind rises;
- `t = 9s`: load changes back;
- `t = 10.7s`: DG turns off / supervisor transition;
- `t = 11s`;
- `t = 15s`;
- `t = 16s`;
- `t = 18s`: load reduction and WT charging behavior.

Expected qualitative matches:

- DG current turns off around `10.7s`;
- DC bus voltage spikes/dips at the same events as the article and stabilizes near the same baseline;
- load voltage magnitude has spikes at the same event times;
- battery current is a result of buck-boost PI and matches the article shape;
- WT current follows MPPT/wind scenario and matches the article shape;
- inverter current follows the inverter/filter/load/DG states and matches the article shape;
- frequency has short transition spikes and returns to nominal.

## Engineering Acceptance Checklist

Before returning, verify and report:

- Which Table II parameters are used exactly.
- Which controller equations from Fig. 2-8 are implemented exactly.
- Which parameters are not published and had to be estimated.
- Which signals are external scenario inputs.
- Which Fig. 9 channels are computed model outputs.
- Whether any reduced-order/averaged assumptions remain.
- Why each remaining assumption is physically defensible.
- Residual mismatch against Fig. 9.

The model is not acceptable if:

- battery current is read from target CSV;
- DC bus voltage is read from target CSV;
- SOC is read from target CSV;
- inverter current is computed only from a hand-written target envelope;
- load voltage magnitude is injected from the article;
- frequency is injected from the article;
- the output plot is a calibrated drawing layer over a weak simulation.

## Self-Check: Are These Nodes Sufficient?

The node set is sufficient for the Fig. 9 scenario if and only if the following energy path is closed:

1. Wind scenario drives WT/PMSG mechanical input.
2. PMSG produces electrical power.
3. Rectifier converts PMSG AC to DC.
4. Boost converter injects WT current into DC link under MPPT.
5. Battery buck-boost exchanges current with DC link under Fig. 4 PI control.
6. DC-link capacitor integrates net current to produce `Vdc`.
7. VSI draws DC current and produces AC phase voltages under Fig. 3 control.
8. LC filter/transformer produces PCC/load voltage and inverter current.
9. Load consumes current from PCC based on external load demand.
10. DG/SG with AVR/governor supplies PCC when supervisor enables it.
11. Supervisor switches DG/AVR/governor/dump-load based on SOC and WT/load power.
12. Logger records all Fig. 9 channels from node outputs.

If any of these links is missing, the simulation is not yet a faithful Fig. 1 microgrid model.

## Final Rule

Do not return with a partial claim of success.

Work until the dedicated simulation file, trace CSV, PDF plot, and engineering verification report are produced. The final answer must point to the generated files and state honestly what still differs from the article, if anything.
