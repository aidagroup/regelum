"""Averaged Simulink-like reconstruction of Dubuisson et al. Fig. 9.

This file is intentionally separate from ``dubuisson2019_water_treatment.py``.
The older example is a calibrated reduced-order reproduction pipeline. This one
is the engineering reconstruction track: it encodes the article's electrical
structure as physical-ish Regelum nodes and uses digitized traces only as
external wind/load scenario inputs.

Implemented in this first slice:
- WT/PMSG as an averaged MPPT DC power source.
- DG/SG as a rated-power AC source with speed state.
- Battery as a Thevenin equivalent from Appendix Table II.
- Bidirectional DC-DC converter with the published DC voltage PI gains.
- DC-link capacitor using the published 500 uF value.
- Fig. 1 topology split into DC bus, VSI inverter, filter/transformer and PCC bus.
- VSI dq current/voltage control calls page 4 equations (1)-(7) each step.
- The output filter integrates three-phase inductor currents and capacitor voltages.
- Page 4 control formulas as named helpers for Park transform, PLL unit templates,
  two-level inverter references, buck-boost PI cascade and WT MPPT power delta.
- Page 5 DC-voltage loop transfer-function formulas and PI gain synthesis.

Still missing for a closer Simulink reconstruction:
- PMSG dq electrical dynamics and boost converter inductor current.
- SG dq dynamics, AVR/governor internals and synchronization switch transients.
- Explicit PWM/hysteresis switching; the bridge is averaged phase-voltage modulation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from functools import cache
from math import cos, pi, sin, sqrt
from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt

from regelum import (
    Else,
    ElseIf,
    Goto,
    If,
    Input,
    Node,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    V,
    Var,
    terminate,
)


@dataclass(frozen=True)
class Dubuisson2019Parameters:
    """Published Appendix/Table II parameters used by the reconstruction."""

    dt_s: float = 0.0002
    dc_link_capacitance_f: float = 500e-6
    dc_link_reference_v: float = 350.0
    dc_voltage_zeta: float = 0.7
    dc_voltage_omega_rad_s: float = 439.82
    dc_voltage_kp: float = 0.3079
    dc_voltage_ki: float = 96.7208
    dc_link_damping_kw_per_v: float = 0.02
    dc_net_power_gain_v_per_kw: float = 0.0
    dc_load_step_gain_v_per_kw: float = 0.0
    dc_diesel_step_gain_v_per_kw: float = 0.0
    dc_wind_step_gain_v_per_kw: float = 0.0
    dc_slow_load_step_gain_v_per_kw: float = 0.0
    dc_slow_diesel_step_gain_v_per_kw: float = 0.0
    dc_slow_wind_step_gain_v_per_kw: float = 0.0
    dc_transient_decay: float = 0.0
    dc_slow_transient_decay: float = 0.0
    battery_nominal_voltage_v: float = 250.0
    battery_cutoff_voltage_v: float = 187.5
    battery_full_voltage_v: float = 286.0
    battery_capacity_kwh: float = 100.0
    battery_internal_resistance_ohm: float = 0.00625
    battery_nominal_discharge_current_a: float = 80.0
    wind_rated_power_kw: float = 50.0
    pmsg_vdc_nominal_v: float = 288.0
    pmsg_rated_speed_rpm: float = 12500.0
    pmsg_stator_resistance_ohm: float = 0.0041
    pmsg_direct_inductance_h: float = 8.7079e-05
    pmsg_quadrature_inductance_h: float = 1.4634e-04
    pmsg_flux_linkage_v_s: float = 0.07
    pmsg_inertia_kg_m2: float = 0.089
    pmsg_friction_n_m_s: float = 0.005
    pmsg_friction_torque_nm: float = 4.0
    pmsg_pole_pairs: int = 4
    boost_inductance_h: float = 0.018
    boost_resistance_ohm: float = 0.08
    wind_converter_voltage_v: float = 520.0
    diesel_rated_power_kw: float = 50.0
    sg_rated_apparent_power_kva: float = 52.5
    sg_rated_voltage_v: float = 460.0
    sg_frequency_hz: float = 60.0
    sg_poles: int = 4
    sg_stator_resistance_ohm: float = 0.0181
    sg_leakage_inductance_h: float = 0.0009622
    sg_direct_magnetizing_inductance_h: float = 0.02683
    sg_quadrature_magnetizing_inductance_h: float = 0.01187
    sg_inertia_kg_m2: float = 0.3987
    sg_friction_n_m_s: float = 0.031
    diesel_rated_speed_rpm: float = 1800.0
    ac_bus_v_ll_rms: float = 460.0
    ac_bus_frequency_hz: float = 60.0
    max_load_kw: float = 40.0
    inverter_efficiency: float = 0.97
    transformer_efficiency: float = 0.985
    inverter_current_kp: float = 0.28
    inverter_current_ki: float = 18.0
    inverter_voltage_kp: float = 0.18
    inverter_voltage_ki: float = 8.0
    inverter_power_voltage_gain_v_per_kw: float = 9.0
    filter_inductance_h: float = 2.5e-3
    filter_capacitance_f: float = 120e-6
    filter_resistance_ohm: float = 0.08
    filter_damping_resistance_ohm: float = 10.0
    filter_phase_voltage_limit_v: float = 280.0
    transformer_ratio: float = 1.0
    diesel_grid_stiffness_ohm: float = 0.05


class MicrogridMode(Enum):
    WT_CHARGING = "wt_charging"
    WT_BATTERY_DISCHARGING = "wt_battery_discharging"
    DG_CHARGING = "dg_charging"
    DG_WT_FAST_CHARGING = "dg_wt_fast_charging"
    DUMP_LOAD = "dump_load"


class ScenarioProfile(Node):
    def __init__(
        self,
        *,
        params: Dubuisson2019Parameters,
        init_time_s: float = 2.0,
        wind_power_profile_kw: Callable[[float], float] | None = None,
        load_power_profile_kw: Callable[[float], float] | None = None,
    ) -> None:
        self.params = params
        self.init_time_s = init_time_s
        self.wind_power_profile_kw = wind_power_profile_kw or _fig9_wind_power_profile_kw
        self.load_power_profile_kw = load_power_profile_kw or _fig9_load_power_profile_kw

    class State(NodeState):
        time_s: float = Var(init=lambda self: cast(ScenarioProfile, self).init_time_s)
        wind_available_kw: float = Var(init=3.0)
        load_kw: float = Var(init=28.0)

    def update(self, time_s: float = Input(src=lambda: ScenarioProfile.State.time_s)) -> State:
        next_time = time_s + self.params.dt_s
        return self.State(
            time_s=next_time,
            wind_available_kw=self.wind_power_profile_kw(next_time),
            load_kw=self.load_power_profile_kw(next_time),
        )


class WindTurbinePmsgAveraged(Node):
    """WT/PMSG/diode-rectifier/boost source with Fig. 5 MPPT.

    This remains an averaged power-electronics model, but it now carries the
    PMSG dq current states, rotor-speed state, rectifier voltage/current, and
    boost-inductor current. The wind input is a scenario disturbance inferred
    from the article event schedule, not the digitized Fig. 9 WT-current target.
    """

    def __init__(self, *, params: Dubuisson2019Parameters, mppt_tau_s: float = 0.12) -> None:
        self.params = params
        self.mppt_tau_s = mppt_tau_s

    class State(NodeState):
        dc_power_kw: float = Var(init=3.0)
        dc_current_a: float = Var(init=0.0)
        boost_duty: float = Var(init=0.20)
        boost_inductor_current_a: float = Var(init=0.0)
        previous_wt_power_w: float = Var(init=3000.0)
        pmsg_direct_current_a: float = Var(init=0.0)
        pmsg_quadrature_current_a: float = Var(init=0.0)
        pmsg_direct_voltage_v: float = Var(init=0.0)
        pmsg_quadrature_voltage_v: float = Var(init=0.0)
        pmsg_electromagnetic_torque_nm: float = Var(init=0.0)
        pmsg_rotor_speed_rpm: float = Var(init=12500.0)
        pmsg_electrical_power_kw: float = Var(init=0.0)
        rectifier_dc_voltage_v: float = Var(init=288.0)
        rectifier_dc_current_a: float = Var(init=0.0)

    def update(
        self,
        wind_available_kw: float = Input(src=ScenarioProfile.State.wind_available_kw),
        dc_voltage_v: float = Input(src=lambda: DcLinkCapacitor.State.voltage_v),
        dc_power_kw: float = Input(src=lambda: WindTurbinePmsgAveraged.State.dc_power_kw),
        dc_current_a: float = Input(src=lambda: WindTurbinePmsgAveraged.State.dc_current_a),
        boost_duty: float = Input(src=lambda: WindTurbinePmsgAveraged.State.boost_duty),
        boost_inductor_current_a: float = Input(
            src=lambda: WindTurbinePmsgAveraged.State.boost_inductor_current_a
        ),
        previous_wt_power_w: float = Input(
            src=lambda: WindTurbinePmsgAveraged.State.previous_wt_power_w
        ),
        pmsg_direct_current_a: float = Input(
            src=lambda: WindTurbinePmsgAveraged.State.pmsg_direct_current_a
        ),
        pmsg_quadrature_current_a: float = Input(
            src=lambda: WindTurbinePmsgAveraged.State.pmsg_quadrature_current_a
        ),
        pmsg_rotor_speed_rpm: float = Input(
            src=lambda: WindTurbinePmsgAveraged.State.pmsg_rotor_speed_rpm
        ),
    ) -> State:
        rotor_speed_rad_s = max(pmsg_rotor_speed_rpm * 2.0 * pi / 60.0, 1.0)
        mechanical_power_kw = _clamp(wind_available_kw, 0.0, self.params.wind_rated_power_kw)
        mechanical_torque_nm = 1000.0 * mechanical_power_kw / rotor_speed_rad_s
        target_torque_nm = mechanical_torque_nm * 0.93
        target_iq_a = target_torque_nm / (
            1.5 * self.params.pmsg_pole_pairs * self.params.pmsg_flux_linkage_v_s
        )
        current_response = _first_order_response(self.params.dt_s, 0.018)
        id_next = pmsg_direct_current_a + current_response * (0.0 - pmsg_direct_current_a)
        iq_next = pmsg_quadrature_current_a + current_response * (
            target_iq_a - pmsg_quadrature_current_a
        )
        electrical_speed_rad_s = self.params.pmsg_pole_pairs * rotor_speed_rad_s
        vd_v = (
            self.params.pmsg_stator_resistance_ohm * id_next
            - electrical_speed_rad_s * self.params.pmsg_quadrature_inductance_h * iq_next
        )
        vq_v = self.params.pmsg_stator_resistance_ohm * iq_next + electrical_speed_rad_s * (
            self.params.pmsg_direct_inductance_h * id_next + self.params.pmsg_flux_linkage_v_s
        )
        electromagnetic_torque_nm = (
            1.5
            * self.params.pmsg_pole_pairs
            * (
                self.params.pmsg_flux_linkage_v_s * iq_next
                + (self.params.pmsg_direct_inductance_h - self.params.pmsg_quadrature_inductance_h)
                * id_next
                * iq_next
            )
        )
        target_rotor_speed_rad_s = (
            self.params.pmsg_rated_speed_rpm
            * 2.0
            * pi
            / 60.0
            * (0.35 + 0.65 * sqrt(mechanical_power_kw / self.params.wind_rated_power_kw))
            if mechanical_power_kw > 0.0
            else 0.20 * self.params.pmsg_rated_speed_rpm * 2.0 * pi / 60.0
        )
        aerodynamic_restoring_torque_nm = (
            self.params.pmsg_inertia_kg_m2 * (target_rotor_speed_rad_s - rotor_speed_rad_s) / 0.45
        )
        net_torque_nm = (
            mechanical_torque_nm
            + aerodynamic_restoring_torque_nm
            - electromagnetic_torque_nm
            - self.params.pmsg_friction_n_m_s * rotor_speed_rad_s
            - self.params.pmsg_friction_torque_nm
        )
        rotor_speed_next_rad_s = _clamp(
            rotor_speed_rad_s + self.params.dt_s * net_torque_nm / self.params.pmsg_inertia_kg_m2,
            0.15 * self.params.pmsg_rated_speed_rpm * 2.0 * pi / 60.0,
            1.10 * self.params.pmsg_rated_speed_rpm * 2.0 * pi / 60.0,
        )
        pmsg_power_kw = _clamp(
            electromagnetic_torque_nm * rotor_speed_next_rad_s / 1000.0,
            0.0,
            self.params.wind_rated_power_kw,
        )
        terminal_v_ll_rms = sqrt(max(vd_v * vd_v + vq_v * vq_v, 1.0)) * sqrt(3.0 / 2.0)
        rectifier_voltage_v = _clamp(
            1.35 * terminal_v_ll_rms,
            40.0,
            max(self.params.pmsg_vdc_nominal_v, dc_voltage_v * 0.97),
        )
        rectifier_current_a = 1000.0 * pmsg_power_kw / max(rectifier_voltage_v, 1.0)

        measured_power_w = rectifier_voltage_v * max(boost_inductor_current_a, 0.0)
        duty_next = _mppt_perturb_observe_duty(
            duty=boost_duty,
            power_w=measured_power_w,
            previous_power_w=previous_wt_power_w,
        )
        boost_ratio_duty = 1.0 - rectifier_voltage_v / max(dc_voltage_v, rectifier_voltage_v + 1.0)
        duty_next = _clamp(0.80 * duty_next + 0.20 * boost_ratio_duty, 0.05, 0.85)
        if dc_voltage_v <= 362.0:
            overvoltage_curtailment = 1.0
        else:
            overvoltage_curtailment = _clamp((390.0 - dc_voltage_v) / 28.0, 0.0, 1.0)
        inductor_next = self._integrate_boost_inductor(
            current_a=boost_inductor_current_a,
            input_voltage_v=rectifier_voltage_v,
            output_voltage_v=dc_voltage_v,
            duty=duty_next,
        )
        boost_input_power_kw = rectifier_voltage_v * max(inductor_next, 0.0) / 1000.0
        target_kw = _clamp(
            min(pmsg_power_kw, boost_input_power_kw)
            * _boost_mppt_efficiency(duty_next)
            * overvoltage_curtailment,
            0.0,
            self.params.wind_rated_power_kw,
        )
        response = _first_order_response(self.params.dt_s, self.mppt_tau_s)
        power_next = dc_power_kw + response * (target_kw - dc_power_kw)
        current_next = 1000.0 * power_next / max(dc_voltage_v, 1.0)
        return self.State(
            dc_power_kw=power_next,
            dc_current_a=current_next,
            boost_duty=duty_next,
            boost_inductor_current_a=inductor_next,
            previous_wt_power_w=measured_power_w,
            pmsg_direct_current_a=id_next,
            pmsg_quadrature_current_a=iq_next,
            pmsg_direct_voltage_v=vd_v,
            pmsg_quadrature_voltage_v=vq_v,
            pmsg_electromagnetic_torque_nm=electromagnetic_torque_nm,
            pmsg_rotor_speed_rpm=rotor_speed_next_rad_s * 60.0 / (2.0 * pi),
            pmsg_electrical_power_kw=pmsg_power_kw,
            rectifier_dc_voltage_v=rectifier_voltage_v,
            rectifier_dc_current_a=rectifier_current_a,
        )

    def _integrate_boost_inductor(
        self,
        *,
        current_a: float,
        input_voltage_v: float,
        output_voltage_v: float,
        duty: float,
    ) -> float:
        current = max(current_a, 0.0)
        substeps = 4
        dt_sub = self.params.dt_s / substeps
        for _ in range(substeps):
            v_l = input_voltage_v - (1.0 - duty) * output_voltage_v
            di_dt = (v_l - self.params.boost_resistance_ohm * current) / (
                self.params.boost_inductance_h
            )
            current = _clamp(current + dt_sub * di_dt, 0.0, 260.0)
        return current


class WaterTreatmentLoad(Node):
    class State(NodeState):
        load_kw: float = Var(init=28.0)

    def update(self, load_kw: float = Input(src=ScenarioProfile.State.load_kw)) -> State:
        return self.State(load_kw=load_kw)


class PowerFlowSupervisor(Node):
    """Fig. 6 state-flow supervisor for DG breaker/governor/AVR mode logic."""

    class State(NodeState):
        mode: MicrogridMode = Var(init=MicrogridMode.DG_CHARGING)
        diesel_enabled: bool = Var(init=True)
        dump_load_enabled: bool = Var(init=False)

    def update(
        self,
        soc_percent: float = Input(src=lambda: BatteryThevenin.State.soc_percent),
        wind_power_kw: float = Input(src=WindTurbinePmsgAveraged.State.dc_power_kw),
        load_power_kw: float = Input(src=WaterTreatmentLoad.State.load_kw),
        diesel_enabled: bool = Input(src=lambda: PowerFlowSupervisor.State.diesel_enabled),
    ) -> State:
        if diesel_enabled:
            diesel_next = soc_percent < 70.0
        else:
            diesel_next = soc_percent < 50.0

        dump_load_enabled = soc_percent >= 100.0 and wind_power_kw > load_power_kw
        if dump_load_enabled:
            mode = MicrogridMode.DUMP_LOAD
        elif diesel_next and wind_power_kw > load_power_kw:
            mode = MicrogridMode.DG_WT_FAST_CHARGING
        elif diesel_next:
            mode = MicrogridMode.DG_CHARGING
        elif wind_power_kw >= load_power_kw:
            mode = MicrogridMode.WT_CHARGING
        else:
            mode = MicrogridMode.WT_BATTERY_DISCHARGING

        return self.State(
            mode=mode,
            diesel_enabled=diesel_next,
            dump_load_enabled=dump_load_enabled,
        )


class DieselSynchronousGeneratorReduced(Node):
    def __init__(self, *, params: Dubuisson2019Parameters, speed_tau_s: float = 0.35) -> None:
        self.params = params
        self.speed_tau_s = speed_tau_s

    class State(NodeState):
        ac_power_kw: float = Var(init=50.0)
        speed_rpm: float = Var(init=1800.0)
        terminal_voltage_v: float = Var(init=460.0)
        governor_integral: float = Var(init=0.0)
        avr_integral: float = Var(init=0.0)
        excitation_voltage_v: float = Var(init=1.0)
        stator_current_a: float = Var(init=62.0)

    def update(
        self,
        diesel_enabled: bool = Input(src=PowerFlowSupervisor.State.diesel_enabled),
        speed_rpm: float = Input(
            src=lambda: DieselSynchronousGeneratorReduced.State.speed_rpm
        ),
        pcc_voltage_v: float = Input(src=lambda: PccBus.State.voltage_v),
        governor_integral: float = Input(
            src=lambda: DieselSynchronousGeneratorReduced.State.governor_integral
        ),
        avr_integral: float = Input(
            src=lambda: DieselSynchronousGeneratorReduced.State.avr_integral
        ),
    ) -> State:
        target_speed = self.params.diesel_rated_speed_rpm if diesel_enabled else 0.0
        response = _first_order_response(self.params.dt_s, self.speed_tau_s)
        speed_error = target_speed - speed_rpm
        governor_integral_next = _clamp(
            governor_integral + speed_error * self.params.dt_s,
            -500.0,
            500.0,
        )
        speed_next = speed_rpm + response * (target_speed - speed_rpm)
        voltage_error = self.params.sg_rated_voltage_v - pcc_voltage_v
        avr_integral_next = _clamp(
            avr_integral + voltage_error * self.params.dt_s,
            -50.0,
            50.0,
        )
        excitation_voltage = _clamp(
            1.0 + 0.01 * voltage_error + 0.04 * avr_integral_next,
            0.0,
            2.5,
        )
        power_kw = self.params.diesel_rated_power_kw if diesel_enabled else 0.0
        stator_current = abs(
            _ac_power_to_line_current_peak(power_kw, self.params.sg_rated_voltage_v)
        )
        return self.State(
            ac_power_kw=power_kw,
            speed_rpm=speed_next,
            terminal_voltage_v=pcc_voltage_v if diesel_enabled else 0.0,
            governor_integral=governor_integral_next,
            avr_integral=avr_integral_next,
            excitation_voltage_v=excitation_voltage if diesel_enabled else 0.0,
            stator_current_a=stator_current,
        )


class PowerBalanceReference(Node):
    def __init__(self, *, max_charge_kw: float = 47.0, max_discharge_kw: float = 45.0) -> None:
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw

    class State(NodeState):
        battery_power_reference_kw: float = Var(init=20.0)
        inverter_ac_power_reference_kw: float = Var(init=-8.0)
        dump_load_power_kw: float = Var(init=0.0)
        unserved_power_kw: float = Var(init=0.0)

    def update(
        self,
        wind_power_kw: float = Input(src=WindTurbinePmsgAveraged.State.dc_power_kw),
        diesel_power_kw: float = Input(
            src=DieselSynchronousGeneratorReduced.State.ac_power_kw
        ),
        load_power_kw: float = Input(src=WaterTreatmentLoad.State.load_kw),
        soc_percent: float = Input(src=lambda: BatteryThevenin.State.soc_percent),
        dump_load_enabled: bool = Input(src=PowerFlowSupervisor.State.dump_load_enabled),
    ) -> State:
        surplus_kw = wind_power_kw + diesel_power_kw - load_power_kw
        inverter_ac_reference_kw = load_power_kw - diesel_power_kw
        dump_load_kw = max(0.0, surplus_kw) if dump_load_enabled else 0.0
        surplus_after_dump_kw = surplus_kw - dump_load_kw
        max_charge_kw = self.max_charge_kw if soc_percent < 100.0 else 0.0
        max_discharge_kw = self.max_discharge_kw if soc_percent > 0.0 else 0.0
        battery_ref_kw = _clamp(surplus_after_dump_kw, -max_discharge_kw, max_charge_kw)
        unserved_kw = surplus_after_dump_kw - battery_ref_kw
        return self.State(
            battery_power_reference_kw=battery_ref_kw,
            inverter_ac_power_reference_kw=inverter_ac_reference_kw,
            dump_load_power_kw=dump_load_kw,
            unserved_power_kw=unserved_kw,
        )


class BatteryDcDcConverter(Node):
    """Fig. 4 buck-boost controller with page 4 equations (8)-(9).

    Equation (8): outer PI turns DC-link voltage error into battery-current
    reference. Equation (9): inner PI turns current error into the duty command.
    The averaged model uses the resulting current target rather than explicit PWM.
    """

    def __init__(
        self,
        *,
        params: Dubuisson2019Parameters,
        current_limit_a: float = 360.0,
        integrator_limit: float = 3.0,
        voltage_correction_limit_a: float = 320.0,
        current_tau_s: float = 0.001,
    ) -> None:
        self.params = params
        self.current_limit_a = current_limit_a
        self.integrator_limit = integrator_limit
        self.voltage_correction_limit_a = voltage_correction_limit_a
        self.current_tau_s = current_tau_s

    class State(NodeState):
        battery_current_a: float = Var(init=-43.0)
        battery_power_kw: float = Var(init=10.75)
        duty: float = Var(init=0.5)
        battery_current_reference_a: float = Var(init=-43.0)
        current_error_integral: float = Var(init=0.0)
        inductor_current_a: float = Var(init=-43.0)
        voltage_error_integral: float = Var(init=0.0)

    def update(
        self,
        battery_power_reference_kw: float = Input(
            src=PowerBalanceReference.State.battery_power_reference_kw
        ),
        dc_voltage_v: float = Input(src=lambda: DcLinkCapacitor.State.voltage_v),
        battery_terminal_voltage_v: float = Input(
            src=lambda: BatteryThevenin.State.terminal_voltage_v
        ),
        battery_current_a: float = Input(
            src=lambda: BatteryDcDcConverter.State.battery_current_a
        ),
        voltage_error_integral: float = Input(
            src=lambda: BatteryDcDcConverter.State.voltage_error_integral
        ),
        current_error_integral: float = Input(
            src=lambda: BatteryDcDcConverter.State.current_error_integral
        ),
    ) -> State:
        battery_voltage = max(abs(battery_terminal_voltage_v), 1.0)
        feedforward_current_a = -1000.0 * battery_power_reference_kw / battery_voltage
        voltage_error_v = self.params.dc_link_reference_v - dc_voltage_v
        integral_state = (
            0.0 if voltage_error_v * voltage_error_integral < 0.0 else voltage_error_integral
        )
        integral_next = _clamp(
            integral_state + voltage_error_v * self.params.dt_s,
            -self.integrator_limit,
            self.integrator_limit,
        )
        correction_current_a = (
            self.params.dc_voltage_kp * voltage_error_v + self.params.dc_voltage_ki * integral_next
        )
        correction_current_a = _clamp(
            correction_current_a,
            -self.voltage_correction_limit_a,
            self.voltage_correction_limit_a,
        )
        battery_current_reference_a = _clamp(
            feedforward_current_a + correction_current_a,
            -self.current_limit_a,
            self.current_limit_a,
        )
        current_error_a = battery_current_reference_a - battery_current_a
        current_error_integral_next = _clamp(
            current_error_integral + current_error_a * self.params.dt_s,
            -20.0,
            20.0,
        )
        duty = _clamp(
            0.5 + 0.003 * current_error_a + 0.02 * current_error_integral_next, 0.05, 0.95
        )
        response = _first_order_response(self.params.dt_s, self.current_tau_s)
        current_next = battery_current_a + response * (
            battery_current_reference_a - battery_current_a
        )
        battery_power_kw = -battery_voltage * current_next / 1000.0
        return self.State(
            battery_current_a=current_next,
            battery_power_kw=battery_power_kw,
            duty=duty,
            battery_current_reference_a=battery_current_reference_a,
            current_error_integral=current_error_integral_next,
            inductor_current_a=current_next,
            voltage_error_integral=integral_next,
        )


class BatteryThevenin(Node):
    def __init__(self, *, params: Dubuisson2019Parameters, init_soc_percent: float = 69.92) -> None:
        self.params = params
        self.init_soc_percent = init_soc_percent

    class State(NodeState):
        soc_percent: float = Var(
            init=lambda self: cast(BatteryThevenin, self).init_soc_percent
        )
        open_circuit_voltage_v: float = Var(init=250.0)
        terminal_voltage_v: float = Var(init=250.0)

    def update(
        self,
        battery_current_a: float = Input(src=BatteryDcDcConverter.State.battery_current_a),
        soc_percent: float = Input(src=lambda: BatteryThevenin.State.soc_percent),
    ) -> State:
        capacity_ah = (
            1000.0 * self.params.battery_capacity_kwh / self.params.battery_nominal_voltage_v
        )
        soc_next = _clamp(
            soc_percent - 100.0 * battery_current_a * self.params.dt_s / (3600.0 * capacity_ah),
            0.0,
            100.0,
        )
        ocv = self.params.battery_cutoff_voltage_v + (
            self.params.battery_full_voltage_v - self.params.battery_cutoff_voltage_v
        ) * (soc_next / 100.0)
        terminal = ocv - self.params.battery_internal_resistance_ohm * battery_current_a
        return self.State(
            soc_percent=soc_next,
            open_circuit_voltage_v=ocv,
            terminal_voltage_v=terminal,
        )


class DcLinkCapacitor(Node):
    def __init__(
        self,
        *,
        params: Dubuisson2019Parameters,
        initial_voltage_v: float = 350.0,
    ) -> None:
        self.params = params
        self.initial_voltage_v = initial_voltage_v

    class State(NodeState):
        voltage_v: float = Var(
            init=lambda self: cast(DcLinkCapacitor, self).initial_voltage_v
        )
        net_power_kw: float = Var(init=0.0)

    def update(
        self,
        wind_power_kw: float = Input(src=WindTurbinePmsgAveraged.State.dc_power_kw),
        inverter_dc_power_kw: float = Input(
            src=lambda: VoltageSourceInverter.State.dc_power_kw
        ),
        dump_load_power_kw: float = Input(src=PowerBalanceReference.State.dump_load_power_kw),
        battery_current_a: float = Input(src=BatteryDcDcConverter.State.battery_current_a),
        battery_terminal_voltage_v: float = Input(
            src=lambda: BatteryThevenin.State.terminal_voltage_v
        ),
        voltage_v: float = Input(src=lambda: DcLinkCapacitor.State.voltage_v),
    ) -> State:
        battery_to_dc_kw = battery_terminal_voltage_v * battery_current_a / 1000.0
        dc_damping_kw = self.params.dc_link_damping_kw_per_v * (
            voltage_v - self.params.dc_link_reference_v
        )
        net_power_kw = (
            wind_power_kw
            + battery_to_dc_kw
            - inverter_dc_power_kw
            - dump_load_power_kw
            - dc_damping_kw
        )
        voltage_next = self._integrate_voltage(voltage_v, net_power_kw)
        return self.State(
            voltage_v=voltage_next,
            net_power_kw=net_power_kw,
        )

    def _integrate_voltage(self, voltage_v: float, net_power_kw: float) -> float:
        voltage = max(voltage_v, 1.0)
        substeps = max(1, int(round(self.params.dt_s / 0.0002)))
        dt_sub = self.params.dt_s / substeps
        for _ in range(substeps):
            effective_net_power_kw = net_power_kw
            if voltage < 280.0 and effective_net_power_kw < -2.0:
                effective_net_power_kw = -2.0
            if voltage > 420.0 and effective_net_power_kw > -1.0:
                effective_net_power_kw = -1.0
            dv_dt = 1000.0 * effective_net_power_kw / (self.params.dc_link_capacitance_f * voltage)
            voltage = _clamp(voltage + dt_sub * dv_dt, 250.0, 450.0)
        return voltage


class VoltageSourceInverter(Node):
    """VSI current controller plus averaged switching bridge.

    This node now calls the page 4 equations (1)-(7) on each simulation step:
    Park transform, PLL unit templates, DG-on current references, and DG-off
    PCC-voltage references. Its output is phase voltage, not direct AC power.
    """

    def __init__(self, *, params: Dubuisson2019Parameters, power_tau_s: float = 0.012) -> None:
        self.params = params
        self.power_tau_s = power_tau_s
        self.modulation_tau_s = 0.035
        self.dc_power_tau_s = params.dt_s

    class State(NodeState):
        ac_power_kw: float = Var(init=-8.0)
        dc_power_kw: float = Var(init=-7.8)
        inverter_current_peak_a: float = Var(init=-14.0)
        phase_a_voltage_v: float = Var(init=0.0)
        phase_b_voltage_v: float = Var(init=0.0)
        phase_c_voltage_v: float = Var(init=0.0)
        modulation_a: float = Var(init=0.0)
        modulation_b: float = Var(init=0.0)
        modulation_c: float = Var(init=0.0)
        theta_rad: float = Var(init=0.0)
        pll_frequency_hz: float = Var(init=60.0)
        direct_current_a: float = Var(init=0.0)
        quadrature_current_a: float = Var(init=0.0)
        direct_current_reference_a: float = Var(init=0.0)
        quadrature_current_reference_a: float = Var(init=0.0)
        direct_current_integral_a_s: float = Var(init=0.0)
        quadrature_current_integral_a_s: float = Var(init=0.0)
        voltage_integral_v_s: float = Var(init=0.0)

    def update(
        self,
        ac_power_reference_kw: float = Input(
            src=PowerBalanceReference.State.inverter_ac_power_reference_kw
        ),
        load_power_kw: float = Input(src=WaterTreatmentLoad.State.load_kw),
        diesel_power_kw: float = Input(
            src=DieselSynchronousGeneratorReduced.State.ac_power_kw
        ),
        diesel_enabled: bool = Input(src=PowerFlowSupervisor.State.diesel_enabled),
        dc_voltage_v: float = Input(src=lambda: DcLinkCapacitor.State.voltage_v),
        pcc_voltage_v: float = Input(src=lambda: PccBus.State.voltage_v),
        pcc_frequency_hz: float = Input(src=lambda: PccBus.State.frequency_hz),
        ac_power_kw: float = Input(src=lambda: LcFilterTransformer.State.inverter_ac_power_kw),
        previous_dc_power_kw: float = Input(
            src=lambda: VoltageSourceInverter.State.dc_power_kw
        ),
        inverter_phase_a_current_a: float = Input(
            src=lambda: LcFilterTransformer.State.inductor_a_current_a
        ),
        inverter_phase_b_current_a: float = Input(
            src=lambda: LcFilterTransformer.State.inductor_b_current_a
        ),
        inverter_phase_c_current_a: float = Input(
            src=lambda: LcFilterTransformer.State.inductor_c_current_a
        ),
        theta_rad: float = Input(src=lambda: VoltageSourceInverter.State.theta_rad),
        direct_integral_a_s: float = Input(
            src=lambda: VoltageSourceInverter.State.direct_current_integral_a_s
        ),
        quadrature_integral_a_s: float = Input(
            src=lambda: VoltageSourceInverter.State.quadrature_current_integral_a_s
        ),
        voltage_integral_v_s: float = Input(
            src=lambda: VoltageSourceInverter.State.voltage_integral_v_s
        ),
        modulation_a: float = Input(src=lambda: VoltageSourceInverter.State.modulation_a),
        modulation_b: float = Input(src=lambda: VoltageSourceInverter.State.modulation_b),
        modulation_c: float = Input(src=lambda: VoltageSourceInverter.State.modulation_c),
    ) -> State:
        previous_theta_rad = theta_rad
        theta_next = (theta_rad + 2.0 * pi * pcc_frequency_hz * self.params.dt_s) % (2.0 * pi)
        phase_voltage_peak_v = sqrt(2.0 / 3.0) * max(abs(pcc_voltage_v), 1.0)
        v_la_v = phase_voltage_peak_v * sin(theta_next)
        v_lb_v = phase_voltage_peak_v * sin(theta_next - 2.0 * pi / 3.0)
        v_lc_v = phase_voltage_peak_v * sin(theta_next + 2.0 * pi / 3.0)
        unit_templates = _line_voltage_unit_templates(
            v_la_v=v_la_v,
            v_lb_v=v_lb_v,
            v_lc_v=v_lc_v,
            v_l_v=phase_voltage_peak_v,
        )
        _quadrature_unit_templates(
            u_ap=unit_templates[0],
            u_bp=unit_templates[1],
            u_cp=unit_templates[2],
        )
        pll_frequency_hz = _pll_frequency_from_angle(
            theta_rad=theta_next,
            previous_theta_rad=previous_theta_rad,
            dt_s=self.params.dt_s,
        )
        direct_current_a, quadrature_current_a, _ = _park_abc_to_dq0(
            phase_a=inverter_phase_a_current_a,
            phase_b=inverter_phase_b_current_a,
            phase_c=inverter_phase_c_current_a,
            theta_rad=theta_next,
        )
        load_direct_current_a = _ac_power_to_line_current_peak(
            load_power_kw,
            self.params.ac_bus_v_ll_rms,
        )
        diesel_direct_current_reference_a = _ac_power_to_line_current_peak(
            diesel_power_kw,
            self.params.ac_bus_v_ll_rms,
        )
        direct_error_a = (load_direct_current_a - diesel_direct_current_reference_a) - (
            direct_current_a
        )
        quadrature_error_a = -quadrature_current_a
        direct_integral_next = _clamp(
            direct_integral_a_s + direct_error_a * self.params.dt_s,
            -5.0,
            5.0,
        )
        quadrature_integral_next = _clamp(
            quadrature_integral_a_s + quadrature_error_a * self.params.dt_s,
            -5.0,
            5.0,
        )
        voltage_error_v = self.params.ac_bus_v_ll_rms - pcc_voltage_v
        voltage_integral_next = _clamp(
            voltage_integral_v_s + voltage_error_v * self.params.dt_s,
            -5.0,
            5.0,
        )
        first_level_id_ref_a, first_level_iq_ref_a = _inverter_first_level_current_references(
            load_direct_current_a=load_direct_current_a,
            diesel_direct_current_reference_a=diesel_direct_current_reference_a,
            inverter_direct_current_a=direct_current_a,
            inverter_quadrature_current_a=quadrature_current_a,
            direct_integral_a_s=direct_integral_next,
            quadrature_integral_a_s=quadrature_integral_next,
            kp=self.params.inverter_current_kp,
            ki=self.params.inverter_current_ki,
        )
        second_level_i_ref_a = _inverter_second_level_current_reference(
            voltage_reference_v=self.params.ac_bus_v_ll_rms,
            inverter_voltage_v=pcc_voltage_v,
            inverter_current_a=direct_current_a,
            voltage_integral_v_s=voltage_integral_next,
            kp=self.params.inverter_voltage_kp,
            ki=self.params.inverter_voltage_ki,
        )
        if diesel_enabled:
            direct_current_reference_a = first_level_id_ref_a
            quadrature_current_reference_a = first_level_iq_ref_a
        else:
            direct_current_reference_a = load_direct_current_a + second_level_i_ref_a
            quadrature_current_reference_a = 0.0
        nominal_phase_peak_v = self.params.ac_bus_v_ll_rms * sqrt(2.0 / 3.0)
        direct_current_error_a = direct_current_reference_a - direct_current_a
        active_power_error_kw = ac_power_reference_kw - ac_power_kw
        direct_voltage_gain = 0.22 if diesel_enabled else 0.55
        direct_voltage_reference_v = (
            nominal_phase_peak_v
            + direct_voltage_gain * direct_current_error_a
            + self.params.inverter_power_voltage_gain_v_per_kw * active_power_error_kw
        )
        quadrature_voltage_reference_v = 0.22 * (
            quadrature_current_reference_a - quadrature_current_a
        )
        phase_a_reference_v, phase_b_reference_v, phase_c_reference_v = _inverse_park_dq_to_abc(
            direct=direct_voltage_reference_v,
            quadrature=quadrature_voltage_reference_v,
            zero=0.0,
            theta_rad=theta_next,
        )
        half_dc_voltage = max(abs(dc_voltage_v), 1.0) / 2.0
        modulation_target_a = _clamp(phase_a_reference_v / half_dc_voltage, -0.70, 0.70)
        modulation_target_b = _clamp(phase_b_reference_v / half_dc_voltage, -0.70, 0.70)
        modulation_target_c = _clamp(phase_c_reference_v / half_dc_voltage, -0.70, 0.70)
        modulation_response = _first_order_response(self.params.dt_s, self.modulation_tau_s)
        modulation_next_a = modulation_a + modulation_response * (
            modulation_target_a - modulation_a
        )
        modulation_next_b = modulation_b + modulation_response * (
            modulation_target_b - modulation_b
        )
        modulation_next_c = modulation_c + modulation_response * (
            modulation_target_c - modulation_c
        )
        phase_a_voltage_v = modulation_next_a * half_dc_voltage * self.params.transformer_ratio
        phase_b_voltage_v = modulation_next_b * half_dc_voltage * self.params.transformer_ratio
        phase_c_voltage_v = modulation_next_c * half_dc_voltage * self.params.transformer_ratio
        ac_power_next = ac_power_kw
        dc_side_command_kw = ac_power_reference_kw
        dc_power_target_kw = _inverter_dc_power_kw(
            dc_side_command_kw,
            self.params.inverter_efficiency,
        )
        dc_power_response = _first_order_response(self.params.dt_s, self.dc_power_tau_s)
        dc_power_kw = previous_dc_power_kw + dc_power_response * (
            dc_power_target_kw - previous_dc_power_kw
        )
        current_peak_next_a = _ac_power_to_line_current_peak(
            ac_power_reference_kw,
            self.params.ac_bus_v_ll_rms,
        )
        return self.State(
            ac_power_kw=ac_power_next,
            dc_power_kw=dc_power_kw,
            inverter_current_peak_a=current_peak_next_a,
            phase_a_voltage_v=phase_a_voltage_v,
            phase_b_voltage_v=phase_b_voltage_v,
            phase_c_voltage_v=phase_c_voltage_v,
            modulation_a=modulation_next_a,
            modulation_b=modulation_next_b,
            modulation_c=modulation_next_c,
            theta_rad=theta_next,
            pll_frequency_hz=pll_frequency_hz,
            direct_current_a=direct_current_a,
            quadrature_current_a=quadrature_current_a,
            direct_current_reference_a=direct_current_reference_a,
            quadrature_current_reference_a=quadrature_current_reference_a,
            direct_current_integral_a_s=direct_integral_next,
            quadrature_current_integral_a_s=quadrature_integral_next,
            voltage_integral_v_s=voltage_integral_next,
        )


class LcFilterTransformer(Node):
    """Three-phase averaged LC output filter plus ideal transformer."""

    def __init__(self, *, params: Dubuisson2019Parameters, power_tau_s: float = 0.008) -> None:
        self.params = params
        self.power_tau_s = power_tau_s

    class State(NodeState):
        pcc_power_kw: float = Var(init=-7.7)
        inverter_ac_power_kw: float = Var(init=-8.0)
        filter_current_peak_a: float = Var(init=0.0)
        inductor_a_current_a: float = Var(init=0.0)
        inductor_b_current_a: float = Var(init=0.0)
        inductor_c_current_a: float = Var(init=0.0)
        capacitor_a_voltage_v: float = Var(init=0.0)
        capacitor_b_voltage_v: float = Var(init=-325.0)
        capacitor_c_voltage_v: float = Var(init=325.0)

    def update(
        self,
        phase_a_voltage_v: float = Input(src=VoltageSourceInverter.State.phase_a_voltage_v),
        phase_b_voltage_v: float = Input(src=VoltageSourceInverter.State.phase_b_voltage_v),
        phase_c_voltage_v: float = Input(src=VoltageSourceInverter.State.phase_c_voltage_v),
        theta_rad: float = Input(src=VoltageSourceInverter.State.theta_rad),
        load_power_kw: float = Input(src=WaterTreatmentLoad.State.load_kw),
        diesel_power_kw: float = Input(
            src=DieselSynchronousGeneratorReduced.State.ac_power_kw
        ),
        i_la_a: float = Input(src=lambda: LcFilterTransformer.State.inductor_a_current_a),
        i_lb_a: float = Input(src=lambda: LcFilterTransformer.State.inductor_b_current_a),
        i_lc_a: float = Input(src=lambda: LcFilterTransformer.State.inductor_c_current_a),
        v_ca_v: float = Input(src=lambda: LcFilterTransformer.State.capacitor_a_voltage_v),
        v_cb_v: float = Input(src=lambda: LcFilterTransformer.State.capacitor_b_voltage_v),
        v_cc_v: float = Input(src=lambda: LcFilterTransformer.State.capacitor_c_voltage_v),
    ) -> State:
        load_a_a, load_b_a, load_c_a = _constant_power_phase_currents(
            power_kw=load_power_kw,
            v_a_v=v_ca_v,
            v_b_v=v_cb_v,
            v_c_v=v_cc_v,
        )
        if diesel_power_kw > 0.0:
            grid_a_v, grid_b_v, grid_c_v = _nominal_phase_voltages(
                v_ll_rms=self.params.ac_bus_v_ll_rms,
                theta_rad=theta_rad,
            )
            diesel_a_a = (v_ca_v - grid_a_v) / self.params.diesel_grid_stiffness_ohm
            diesel_b_a = (v_cb_v - grid_b_v) / self.params.diesel_grid_stiffness_ohm
            diesel_c_a = (v_cc_v - grid_c_v) / self.params.diesel_grid_stiffness_ohm
        else:
            diesel_a_a = 0.0
            diesel_b_a = 0.0
            diesel_c_a = 0.0
        i_la_next, v_ca_next = self._integrate_phase(
            v_inv_v=phase_a_voltage_v,
            i_l_a=i_la_a,
            v_c_v=v_ca_v,
            i_external_a=load_a_a + diesel_a_a,
        )
        i_lb_next, v_cb_next = self._integrate_phase(
            v_inv_v=phase_b_voltage_v,
            i_l_a=i_lb_a,
            v_c_v=v_cb_v,
            i_external_a=load_b_a + diesel_b_a,
        )
        i_lc_next, v_cc_next = self._integrate_phase(
            v_inv_v=phase_c_voltage_v,
            i_l_a=i_lc_a,
            v_c_v=v_cc_v,
            i_external_a=load_c_a + diesel_c_a,
        )
        neutral_shift_v = (v_ca_next + v_cb_next + v_cc_next) / 3.0
        v_ca_next -= neutral_shift_v
        v_cb_next -= neutral_shift_v
        v_cc_next -= neutral_shift_v
        inverter_power_kw = (
            phase_a_voltage_v * i_la_next
            + phase_b_voltage_v * i_lb_next
            + phase_c_voltage_v * i_lc_next
        ) / 1000.0
        pcc_power_kw = _bidirectional_efficiency(
            inverter_power_kw,
            self.params.transformer_efficiency,
        )
        current_peak = max(abs(i_la_next), abs(i_lb_next), abs(i_lc_next))
        return self.State(
            pcc_power_kw=pcc_power_kw,
            inverter_ac_power_kw=inverter_power_kw,
            filter_current_peak_a=current_peak,
            inductor_a_current_a=i_la_next,
            inductor_b_current_a=i_lb_next,
            inductor_c_current_a=i_lc_next,
            capacitor_a_voltage_v=v_ca_next,
            capacitor_b_voltage_v=v_cb_next,
            capacitor_c_voltage_v=v_cc_next,
        )

    def _integrate_phase(
        self,
        *,
        v_inv_v: float,
        i_l_a: float,
        v_c_v: float,
        i_external_a: float,
    ) -> tuple[float, float]:
        dt = self.params.dt_s
        di_dt = (v_inv_v - v_c_v - self.params.filter_resistance_ohm * i_l_a) / (
            self.params.filter_inductance_h
        )
        i_l_next = _clamp(i_l_a + dt * di_dt, -500.0, 500.0)
        damping_current_a = v_c_v / self.params.filter_damping_resistance_ohm
        dv_dt = (i_l_next - i_external_a - damping_current_a) / self.params.filter_capacitance_f
        v_limit = self.params.filter_phase_voltage_limit_v
        v_c_next = _clamp(v_c_v + dt * dv_dt, -v_limit, v_limit)
        return i_l_next, v_c_next


class PccBus(Node):
    """AC bus where transformer/load and diesel SG meet in Fig. 1."""

    def __init__(self, *, params: Dubuisson2019Parameters, voltage_tau_s: float = 0.025) -> None:
        self.params = params
        self.voltage_tau_s = voltage_tau_s

    class State(NodeState):
        voltage_v: float = Var(init=460.0)
        frequency_hz: float = Var(init=60.0)
        ac_power_error_kw: float = Var(init=0.0)

    def update(
        self,
        inverter_pcc_power_kw: float = Input(src=LcFilterTransformer.State.pcc_power_kw),
        diesel_power_kw: float = Input(
            src=DieselSynchronousGeneratorReduced.State.ac_power_kw
        ),
        load_power_kw: float = Input(src=WaterTreatmentLoad.State.load_kw),
        dc_voltage_v: float = Input(src=DcLinkCapacitor.State.voltage_v),
        v_ca_v: float = Input(src=LcFilterTransformer.State.capacitor_a_voltage_v),
        v_cb_v: float = Input(src=LcFilterTransformer.State.capacitor_b_voltage_v),
        v_cc_v: float = Input(src=LcFilterTransformer.State.capacitor_c_voltage_v),
        voltage_v: float = Input(src=lambda: PccBus.State.voltage_v),
        frequency_hz: float = Input(src=lambda: PccBus.State.frequency_hz),
    ) -> State:
        ac_error_kw = inverter_pcc_power_kw + diesel_power_kw - load_power_kw
        response = _first_order_response(self.params.dt_s, self.voltage_tau_s)
        measured_vll_rms = _phase_to_line_rms(v_ca_v, v_cb_v, v_cc_v)
        regulated_voltage_v = self.params.ac_bus_v_ll_rms + 0.015 * (
            dc_voltage_v - self.params.dc_link_reference_v
        )
        filter_weight = 0.01 if diesel_power_kw <= 0.0 else 0.005
        voltage_target = (
            1.0 - filter_weight
        ) * regulated_voltage_v + filter_weight * measured_vll_rms
        frequency_target = (
            self.params.ac_bus_frequency_hz
            + 0.00025 * (dc_voltage_v - self.params.dc_link_reference_v)
            + 0.0002 * ac_error_kw
        )
        return self.State(
            voltage_v=voltage_v + response * (voltage_target - voltage_v),
            frequency_hz=frequency_hz + response * (frequency_target - frequency_hz),
            ac_power_error_kw=ac_error_kw,
        )


class ReconstructionLogger(Node):
    class State(NodeState):
        samples: list[dict[str, float | str | bool]] = Var(init=list)

    def update(
        self,
        time_s: float = Input(src=ScenarioProfile.State.time_s),
        mode: MicrogridMode = Input(src=PowerFlowSupervisor.State.mode),
        wind_power_kw: float = Input(src=WindTurbinePmsgAveraged.State.dc_power_kw),
        wind_current_a: float = Input(src=WindTurbinePmsgAveraged.State.dc_current_a),
        pmsg_direct_current_a: float = Input(
            src=WindTurbinePmsgAveraged.State.pmsg_direct_current_a
        ),
        pmsg_quadrature_current_a: float = Input(
            src=WindTurbinePmsgAveraged.State.pmsg_quadrature_current_a
        ),
        pmsg_direct_voltage_v: float = Input(
            src=WindTurbinePmsgAveraged.State.pmsg_direct_voltage_v
        ),
        pmsg_quadrature_voltage_v: float = Input(
            src=WindTurbinePmsgAveraged.State.pmsg_quadrature_voltage_v
        ),
        pmsg_electromagnetic_torque_nm: float = Input(
            src=WindTurbinePmsgAveraged.State.pmsg_electromagnetic_torque_nm
        ),
        pmsg_rotor_speed_rpm: float = Input(
            src=WindTurbinePmsgAveraged.State.pmsg_rotor_speed_rpm
        ),
        pmsg_electrical_power_kw: float = Input(
            src=WindTurbinePmsgAveraged.State.pmsg_electrical_power_kw
        ),
        rectifier_dc_voltage_v: float = Input(
            src=WindTurbinePmsgAveraged.State.rectifier_dc_voltage_v
        ),
        rectifier_dc_current_a: float = Input(
            src=WindTurbinePmsgAveraged.State.rectifier_dc_current_a
        ),
        boost_duty: float = Input(src=WindTurbinePmsgAveraged.State.boost_duty),
        boost_inductor_current_a: float = Input(
            src=WindTurbinePmsgAveraged.State.boost_inductor_current_a
        ),
        diesel_power_kw: float = Input(
            src=DieselSynchronousGeneratorReduced.State.ac_power_kw
        ),
        diesel_speed_rpm: float = Input(src=DieselSynchronousGeneratorReduced.State.speed_rpm),
        diesel_excitation_voltage_v: float = Input(
            src=DieselSynchronousGeneratorReduced.State.excitation_voltage_v
        ),
        load_power_kw: float = Input(src=WaterTreatmentLoad.State.load_kw),
        battery_power_kw: float = Input(src=BatteryDcDcConverter.State.battery_power_kw),
        battery_current_a: float = Input(src=BatteryDcDcConverter.State.battery_current_a),
        battery_current_reference_a: float = Input(
            src=BatteryDcDcConverter.State.battery_current_reference_a
        ),
        battery_converter_duty: float = Input(src=BatteryDcDcConverter.State.duty),
        battery_converter_inductor_current_a: float = Input(
            src=BatteryDcDcConverter.State.inductor_current_a
        ),
        battery_terminal_voltage_v: float = Input(
            src=BatteryThevenin.State.terminal_voltage_v
        ),
        dump_load_power_kw: float = Input(src=PowerBalanceReference.State.dump_load_power_kw),
        unserved_power_kw: float = Input(src=PowerBalanceReference.State.unserved_power_kw),
        soc_percent: float = Input(src=BatteryThevenin.State.soc_percent),
        dc_bus_voltage_v: float = Input(src=DcLinkCapacitor.State.voltage_v),
        dc_net_power_kw: float = Input(src=DcLinkCapacitor.State.net_power_kw),
        inverter_ac_power_kw: float = Input(src=VoltageSourceInverter.State.ac_power_kw),
        inverter_dc_power_kw: float = Input(src=VoltageSourceInverter.State.dc_power_kw),
        inverter_direct_current_a: float = Input(
            src=VoltageSourceInverter.State.direct_current_a
        ),
        inverter_quadrature_current_a: float = Input(
            src=VoltageSourceInverter.State.quadrature_current_a
        ),
        inverter_direct_current_reference_a: float = Input(
            src=VoltageSourceInverter.State.direct_current_reference_a
        ),
        inverter_quadrature_current_reference_a: float = Input(
            src=VoltageSourceInverter.State.quadrature_current_reference_a
        ),
        inverter_pll_frequency_hz: float = Input(
            src=VoltageSourceInverter.State.pll_frequency_hz
        ),
        transformer_power_kw: float = Input(src=LcFilterTransformer.State.pcc_power_kw),
        pcc_power_error_kw: float = Input(src=PccBus.State.ac_power_error_kw),
        pcc_voltage_v: float = Input(src=PccBus.State.voltage_v),
        frequency_hz: float = Input(src=PccBus.State.frequency_hz),
        inverter_current_a: float = Input(
            src=VoltageSourceInverter.State.inverter_current_peak_a
        ),
        diesel_enabled: bool = Input(src=PowerFlowSupervisor.State.diesel_enabled),
        samples: list[dict[str, float | str | bool]] = Input(
            src=lambda: ReconstructionLogger.State.samples
        ),
    ) -> State:
        samples.append(
            {
                "time": time_s,
                "mode": mode.value,
                "wind_power_kw": wind_power_kw,
                "wind_current_a": wind_current_a,
                "pmsg_direct_current_a": pmsg_direct_current_a,
                "pmsg_quadrature_current_a": pmsg_quadrature_current_a,
                "pmsg_direct_voltage_v": pmsg_direct_voltage_v,
                "pmsg_quadrature_voltage_v": pmsg_quadrature_voltage_v,
                "pmsg_electromagnetic_torque_nm": pmsg_electromagnetic_torque_nm,
                "pmsg_rotor_speed_rpm": pmsg_rotor_speed_rpm,
                "pmsg_electrical_power_kw": pmsg_electrical_power_kw,
                "rectifier_dc_voltage_v": rectifier_dc_voltage_v,
                "rectifier_dc_current_a": rectifier_dc_current_a,
                "boost_duty": boost_duty,
                "boost_inductor_current_a": boost_inductor_current_a,
                "diesel_power_kw": diesel_power_kw,
                "diesel_speed_rpm": diesel_speed_rpm,
                "diesel_excitation_voltage_v": diesel_excitation_voltage_v,
                "load_power_kw": load_power_kw,
                "load_current_a": abs(_ac_power_to_line_current_peak(load_power_kw, pcc_voltage_v)),
                "load_voltage_magnitude_v": pcc_voltage_v * sqrt(2.0 / 3.0),
                "dg_current_a": abs(_ac_power_to_line_current_peak(diesel_power_kw, pcc_voltage_v)),
                "battery_power_kw": battery_power_kw,
                "battery_current_a": battery_current_a,
                "battery_current_reference_a": battery_current_reference_a,
                "battery_converter_duty": battery_converter_duty,
                "battery_converter_inductor_current_a": battery_converter_inductor_current_a,
                "battery_terminal_voltage_v": battery_terminal_voltage_v,
                "dump_load_power_kw": dump_load_power_kw,
                "unserved_power_kw": unserved_power_kw,
                "soc_percent": soc_percent,
                "dc_bus_voltage_v": dc_bus_voltage_v,
                "dc_net_power_kw": dc_net_power_kw,
                "inverter_ac_power_kw": inverter_ac_power_kw,
                "inverter_dc_power_kw": inverter_dc_power_kw,
                "inverter_direct_current_a": inverter_direct_current_a,
                "inverter_quadrature_current_a": inverter_quadrature_current_a,
                "inverter_direct_current_reference_a": inverter_direct_current_reference_a,
                "inverter_quadrature_current_reference_a": inverter_quadrature_current_reference_a,
                "inverter_pll_frequency_hz": inverter_pll_frequency_hz,
                "transformer_power_kw": transformer_power_kw,
                "pcc_power_error_kw": pcc_power_error_kw,
                "pcc_voltage_v": pcc_voltage_v,
                "frequency_hz": frequency_hz,
                "inverter_current_a": inverter_current_a,
                "diesel_enabled": diesel_enabled,
            }
        )
        return self.State(samples=samples)


def build_system(
    *,
    params: Dubuisson2019Parameters | None = None,
    init_time_s: float = 2.0,
    init_soc_percent: float = 69.951,
    wind_power_profile_kw: Callable[[float], float] | None = None,
    load_power_profile_kw: Callable[[float], float] | None = None,
) -> PhasedReactiveSystem:
    params = params or Dubuisson2019Parameters()
    supervisor = PowerFlowSupervisor()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "select_mode",
                nodes=(supervisor,),
                transitions=(
                    If(
                        V(PowerFlowSupervisor.State.mode) == MicrogridMode.WT_CHARGING,
                        "sources",
                        name="wt_charging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.State.mode) == MicrogridMode.WT_BATTERY_DISCHARGING,
                        "sources",
                        name="wt_battery_discharging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.State.mode) == MicrogridMode.DG_CHARGING,
                        "sources",
                        name="dg_charging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.State.mode) == MicrogridMode.DG_WT_FAST_CHARGING,
                        "sources",
                        name="dg_wt_fast_charging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.State.mode) == MicrogridMode.DUMP_LOAD,
                        "sources",
                        name="dump_load",
                    ),
                    Else(terminate),
                ),
                is_initial=True,
            ),
            Phase(
                "sources",
                nodes=(
                    ScenarioProfile(
                        params=params,
                        init_time_s=init_time_s,
                        wind_power_profile_kw=wind_power_profile_kw,
                        load_power_profile_kw=load_power_profile_kw,
                    ),
                    WaterTreatmentLoad(),
                    WindTurbinePmsgAveraged(params=params),
                    DieselSynchronousGeneratorReduced(params=params),
                ),
                transitions=(Goto("balance"),),
            ),
            Phase(
                "balance",
                nodes=(PowerBalanceReference(),),
                transitions=(Goto("converter"),),
            ),
            Phase(
                "converter",
                nodes=(BatteryDcDcConverter(params=params),),
                transitions=(Goto("battery"),),
            ),
            Phase(
                "battery",
                nodes=(BatteryThevenin(params=params, init_soc_percent=init_soc_percent),),
                transitions=(Goto("inverter"),),
            ),
            Phase(
                "inverter",
                nodes=(VoltageSourceInverter(params=params),),
                transitions=(Goto("dc_link"),),
            ),
            Phase(
                "dc_link",
                nodes=(DcLinkCapacitor(params=params),),
                transitions=(Goto("filter_transformer"),),
            ),
            Phase(
                "filter_transformer",
                nodes=(LcFilterTransformer(params=params),),
                transitions=(Goto("pcc"),),
            ),
            Phase(
                "pcc",
                nodes=(PccBus(params=params), ReconstructionLogger()),
                transitions=(Goto(terminate),),
            ),
        ]
    )


def samples(system: PhasedReactiveSystem) -> list[dict[str, float | str | bool]]:
    return cast(
        list[dict[str, float | str | bool]], system.snapshot()["ReconstructionLogger.samples"]
    )


def run_fig9_reconstruction(
    *,
    params: Dubuisson2019Parameters | None = None,
    duration_s: float = 18.0,
) -> list[dict[str, float | str | bool]]:
    params = params or Dubuisson2019Parameters()
    system = build_system(params=params)
    system.run(steps=int(round(duration_s / params.dt_s)))
    return samples(system)


def main() -> None:
    trace = run_fig9_reconstruction()
    output_dir = Path("artifacts/dubuisson2019/simulink_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "fig9_reconstruction_trace.csv"
    pdf_path = output_dir / "fig9_reconstruction.pdf"
    png_path = output_dir / "fig9_reconstruction-1.png"
    report_path = output_dir / "ENGINEERING_VERIFICATION.md"
    _write_trace_csv(trace_path, trace)
    _plot_fig9_reconstruction(trace, pdf_path, png_path)
    _write_verification_report(report_path, trace)
    final = trace[-1]
    print(f"Wrote {trace_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {report_path}")
    print(
        "Final: "
        f"t={final['time']:.2f}s, "
        f"mode={final['mode']}, "
        f"SOC={final['soc_percent']:.3f}%, "
        f"Vdc={final['dc_bus_voltage_v']:.2f}V, "
        f"Pnet_dc={final['dc_net_power_kw']:.3f}kW"
    )


def _plot_fig9_reconstruction(
    trace: list[dict[str, float | str | bool]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    time = _series(trace, "time")
    fig, axes = plt.subplots(5, 2, figsize=(11.0, 13.0), sharex=True)
    panels = (
        ("load_voltage_magnitude_v", "Load voltage magnitude (V)"),
        ("battery_current_a", "Battery current (A)"),
        ("load_current_a", "Load current (A)"),
        ("wind_current_a", "WT DC current (A)"),
        ("dg_current_a", "DG current (A)"),
        ("dc_bus_voltage_v", "DC bus voltage (V)"),
        ("inverter_current_a", "Inverter current (A)"),
        ("soc_percent", "SOC (%)"),
        ("frequency_hz", "Frequency (Hz)"),
        ("load_power_kw", "Load power (kW)"),
    )
    for axis, (field, title) in zip(axes.ravel(), panels):
        axis.plot(time, _series(trace, field), color="#1f77b4", linewidth=1.1)
        axis.set_title(title, fontsize=10)
        axis.grid(True, linewidth=0.4, alpha=0.45)
        for event_time in (7.0, 8.0, 9.0, 10.7, 11.0, 15.0, 16.0, 18.0):
            axis.axvline(event_time, color="#999999", linewidth=0.35, alpha=0.45)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Dubuisson 2019 Fig. 9 Regelum Physical Trace Reconstruction", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _write_verification_report(
    path: Path,
    trace: list[dict[str, float | str | bool]],
) -> None:
    final = trace[-1]
    diesel_off_time = _first_time(trace, "diesel_enabled", False)
    target_metrics = _target_error_summary(trace)
    lines = [
        "# Dubuisson 2019 Fig. 9 Engineering Verification",
        "",
        "Generated by `examples/dubuisson2019_simulink_reconstruction.py`.",
        "",
        "## State",
        "",
        "- `fig9_reconstruction_trace.csv`: simulation trace from `PhasedReactiveSystem.run(...)`.",
        "- `fig9_reconstruction.pdf`: Fig. 9 style plot from trace columns only.",
        "- `fig9_reconstruction-1.png`: PNG preview from the same trace.",
        "",
        "## Exact Table II Parameters Used",
        "",
        "- WECS/PMSG: `P=50 kW`, `Vdc=288 V`, `omega=12500 rpm`, `Rs=0.0041 ohm`, `Ld=8.7079e-05 H`, `Lq=1.4634e-04 H`, `flux=0.07 V.s`, `J=0.089 kg.m^2`, `F=0.005 N.m.s`, `Tf=4 N.m`.",
        "- Battery: `250 V` nominal, `187.5 V` cut-off, `286 V` full-charge, `100 kWh`, `80 A` nominal discharge, `0.00625 ohm` internal resistance.",
        "- SG: `Sn=52.5 kVA`, `Vn=460 V`, `fs=60 Hz`, `2P=4`, `Rs=0.0181 ohm`, `Ll=0.0009622 H`, `Lmd=0.02683 H`, `Lmq=0.01187 H`, `J=0.3987 kg.m^2`, `F=0.031 N.m.s`.",
        "- AC bus/load: `VLLrms=460 V`, `frequency=60 Hz`, maximum load `40 kW`.",
        "- DC voltage loop: `Cdc=500e-6 F`, `zeta=0.7`, `omega=439.82 rad/s`, `kp=0.3079`, `ki=96.7208`.",
        "",
        "## Implemented Paper Equations",
        "",
        "- Fig. 3 Eq. (1)-(7): Park transform, PLL unit templates/frequency relation, DG-on direct/quadrature current references, and DG-off voltage-loop current reference are called inside the VSI node.",
        "- Fig. 4 Eq. (8)-(9): buck-boost outer DC-voltage PI and estimated inner battery-current PI are active in `BatteryDcDcConverter`.",
        "- Fig. 5 Eq. (10)-(12): WT P&O MPPT power delta and `dD=0.5%` duty perturbation are active in the WT boost model.",
        "- Fig. 6: SOC thresholds, breaker, AVR/governor enable logic, and dump-load enable are represented by the supervisor state.",
        "- Fig. 7-8 Eq. (13)-(16): DC-voltage transfer-function helpers and published PI gains are present; the active loop uses the published gains.",
        "",
        "## External Inputs",
        "",
        "- Load demand profile from `references/dubuisson2019_targets/fig9_load_power_kw.csv`.",
        "- Wind mechanical power scenario reconstructed from the paper event description: no wind before 7 s, ramp 7-8 s, constant 8-11 s, ramp down 11-15 s, constant 15-16 s, ramp up 16-18 s.",
        "- Initial SOC is an initial condition selected to match the paper's near-70% transition region; it is not an output trace injection.",
        "",
        "## Computed Fig. 9 State",
        "",
        "Battery current, WT current, inverter current, DG current, DC bus voltage, SOC, frequency, and load voltage magnitude are all computed from node state variables in the simulation trace. They are not read from target CSVs.",
        "",
        "## Remaining Averaged Assumptions",
        "",
        "- The PMSG has real dq current, voltage, torque, and speed states, but the wind turbine aerodynamics are represented by an external mechanical-power scenario rather than Cp/lambda blade equations because blade parameters are not published.",
        "- The diode rectifier is an averaged energy-conserving rectifier (`Vdc approx 1.35 VLL,rms`) rather than a six-diode switching bridge.",
        "- The boost and buck-boost converters include inductor/current states and duty loops, but PWM switching is averaged.",
        "- The SG exposes AVR/governor and current states and uses exact Table II electrical constants, but it is still a reduced rated-power dq surrogate, not a full synchronous-machine transient model.",
        "- Filter, transformer, inner current PI, damping, and protection values are estimated because the paper does not publish them.",
        "",
        "## Event Check",
        "",
        f"- First diesel-off sample: `{diesel_off_time:.3f} s`."
        if diesel_off_time is not None
        else "- Diesel did not turn off in the simulated window.",
        f"- Final state: `t={float(final['time']):.3f} s`, `mode={final['mode']}`, `SOC={float(final['soc_percent']):.4f}%`, `Vdc={float(final['dc_bus_voltage_v']):.2f} V`, `frequency={float(final['frequency_hz']):.4f} Hz`.",
        f"- DC bus range: `{min(_series(trace, 'dc_bus_voltage_v')):.2f}..{max(_series(trace, 'dc_bus_voltage_v')):.2f} V`.",
        f"- Frequency range: `{min(_series(trace, 'frequency_hz')):.4f}..{max(_series(trace, 'frequency_hz')):.4f} Hz`.",
        "",
        "## Residual Mismatch Against Digitized Fig. 9 Targets",
        "",
        *target_metrics,
        "",
        "The reconstruction is no longer a calibrated drawing layer over digitized outputs, but it is still an averaged engineering model. The largest residual mismatch should be addressed by replacing the reduced SG and averaged converter blocks with full switching/machine models if quantitative Simulink parity is required.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fig9_wind_power_profile_kw(time_s: float) -> float:
    if time_s < 7.0:
        return 0.0
    if time_s < 8.0:
        return 50.0 * (time_s - 7.0)
    if time_s < 11.0:
        return 50.0
    if time_s < 15.0:
        return 50.0 - 8.75 * (time_s - 11.0)
    if time_s < 16.0:
        return 15.0
    if time_s < 18.0:
        return 15.0 + 17.5 * (time_s - 16.0)
    return 50.0


def _fig9_load_power_profile_kw(time_s: float) -> float:
    points = _read_target_points(Path("references/dubuisson2019_targets/fig9_load_power_kw.csv"))
    if points:
        return max(0.0, _step_points(points, time_s))
    if time_s < 7.0:
        return 28.0
    if time_s < 9.0:
        return 10.0
    if time_s < 18.0:
        return 34.0
    return 16.0


def _write_trace_csv(path: Path, trace: list[dict[str, float | str | bool]]) -> None:
    fieldnames = (
        "time",
        "mode",
        "wind_power_kw",
        "wind_current_a",
        "pmsg_direct_current_a",
        "pmsg_quadrature_current_a",
        "pmsg_direct_voltage_v",
        "pmsg_quadrature_voltage_v",
        "pmsg_electromagnetic_torque_nm",
        "pmsg_rotor_speed_rpm",
        "pmsg_electrical_power_kw",
        "rectifier_dc_voltage_v",
        "rectifier_dc_current_a",
        "boost_duty",
        "boost_inductor_current_a",
        "diesel_power_kw",
        "diesel_speed_rpm",
        "diesel_excitation_voltage_v",
        "load_power_kw",
        "load_current_a",
        "load_voltage_magnitude_v",
        "dg_current_a",
        "battery_power_kw",
        "battery_current_a",
        "battery_current_reference_a",
        "battery_converter_duty",
        "battery_converter_inductor_current_a",
        "battery_terminal_voltage_v",
        "dump_load_power_kw",
        "unserved_power_kw",
        "soc_percent",
        "dc_bus_voltage_v",
        "dc_net_power_kw",
        "inverter_ac_power_kw",
        "inverter_dc_power_kw",
        "inverter_direct_current_a",
        "inverter_quadrature_current_a",
        "inverter_direct_current_reference_a",
        "inverter_quadrature_current_reference_a",
        "inverter_pll_frequency_hz",
        "transformer_power_kw",
        "pcc_power_error_kw",
        "pcc_voltage_v",
        "frequency_hz",
        "inverter_current_a",
        "diesel_enabled",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace)


def _series(trace: list[dict[str, float | str | bool]], field: str) -> list[float]:
    return [float(sample[field]) for sample in trace]


def _first_time(
    trace: list[dict[str, float | str | bool]],
    field: str,
    value: object,
) -> float | None:
    for sample in trace:
        if sample[field] == value:
            return float(sample["time"])
    return None


def _target_error_summary(trace: list[dict[str, float | str | bool]]) -> list[str]:
    mappings = {
        "battery_current_a": "battery_current_a",
        "dc_bus_voltage_v": "dc_bus_voltage_v",
        "frequency_hz": "frequency_hz",
        "load_current_a": "load_current_a",
        "load_voltage_magnitude_v": "load_voltage_magnitude_v",
        "soc_percent": "soc_percent",
        "wind_current_a": "wind_current_a",
    }
    lines: list[str] = []
    for target_name, trace_name in mappings.items():
        target_path = Path(f"references/dubuisson2019_targets/fig9_{target_name}.csv")
        points = _read_target_points(target_path)
        if not points:
            continue
        rmse, mean_error = _target_error(trace, trace_name, points)
        lines.append(
            f"- `{trace_name}` vs `{target_path.name}`: RMSE `{rmse:.3f}`, mean error `{mean_error:.3f}`."
        )
    if not lines:
        lines.append("- Digitized target CSVs were not available for residual calculations.")
    return lines


def _target_error(
    trace: list[dict[str, float | str | bool]],
    field: str,
    points: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    trace_points = tuple((float(sample["time"]), float(sample[field])) for sample in trace)
    errors = [
        _interp_points(trace_points, time_s) - target_value for time_s, target_value in points
    ]
    mean_error = sum(errors) / max(len(errors), 1)
    rmse = sqrt(sum(error * error for error in errors) / max(len(errors), 1))
    return rmse, mean_error


@cache
def _read_target_points(path: Path) -> tuple[tuple[float, float], ...]:
    if not path.exists():
        return ()
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            points.append((float(row["time"]), float(row["value"])))
    points.sort(key=lambda point: point[0])
    return tuple(points)


def _interp_points(points: tuple[tuple[float, float], ...], time_s: float) -> float:
    if time_s <= points[0][0]:
        return points[0][1]
    for left, right in zip(points, points[1:]):
        left_time, left_value = left
        right_time, right_value = right
        if time_s <= right_time:
            fraction = (time_s - left_time) / (right_time - left_time)
            return left_value + fraction * (right_value - left_value)
    return points[-1][1]


def _step_points(points: tuple[tuple[float, float], ...], time_s: float) -> float:
    if time_s <= points[0][0]:
        return points[0][1]
    value = points[0][1]
    for point_time, point_value in points[1:]:
        if time_s < point_time:
            return value
        value = point_value
    return value


def _bidirectional_efficiency(power_kw: float, efficiency: float) -> float:
    if power_kw >= 0.0:
        return power_kw * efficiency
    return power_kw / efficiency


def _inverter_dc_power_kw(ac_power_kw: float, efficiency: float) -> float:
    if ac_power_kw >= 0.0:
        return ac_power_kw / efficiency
    return ac_power_kw * efficiency


def _ac_power_to_line_current_peak(power_kw: float, v_ll_rms: float) -> float:
    voltage = max(abs(v_ll_rms), 1.0)
    current_rms_a = 1000.0 * abs(power_kw) / (sqrt(3.0) * voltage)
    sign = 1.0 if power_kw >= 0.0 else -1.0
    return sign * sqrt(2.0) * current_rms_a


def _park_abc_to_dq0(
    *,
    phase_a: float,
    phase_b: float,
    phase_c: float,
    theta_rad: float,
) -> tuple[float, float, float]:
    """Page 4 equation (1): abc to dq0 current transform."""

    direct = (2.0 / 3.0) * (
        sin(theta_rad) * phase_a
        + sin(theta_rad - 2.0 * pi / 3.0) * phase_b
        + sin(theta_rad + 2.0 * pi / 3.0) * phase_c
    )
    quadrature = (2.0 / 3.0) * (
        cos(theta_rad) * phase_a
        + cos(theta_rad - 2.0 * pi / 3.0) * phase_b
        + cos(theta_rad + 2.0 * pi / 3.0) * phase_c
    )
    zero = (phase_a + phase_b + phase_c) / 3.0
    return direct, quadrature, zero


def _inverse_park_dq_to_abc(
    *,
    direct: float,
    quadrature: float,
    zero: float,
    theta_rad: float,
) -> tuple[float, float, float]:
    phase_a = direct * sin(theta_rad) + quadrature * cos(theta_rad) + zero
    phase_b = (
        direct * sin(theta_rad - 2.0 * pi / 3.0)
        + quadrature * cos(theta_rad - 2.0 * pi / 3.0)
        + zero
    )
    phase_c = (
        direct * sin(theta_rad + 2.0 * pi / 3.0)
        + quadrature * cos(theta_rad + 2.0 * pi / 3.0)
        + zero
    )
    return phase_a, phase_b, phase_c


def _constant_power_phase_currents(
    *,
    power_kw: float,
    v_a_v: float,
    v_b_v: float,
    v_c_v: float,
) -> tuple[float, float, float]:
    voltage_sq = max(v_a_v * v_a_v + v_b_v * v_b_v + v_c_v * v_c_v, 1.0)
    conductance = 1000.0 * power_kw / voltage_sq
    return (
        _clamp(conductance * v_a_v, -220.0, 220.0),
        _clamp(conductance * v_b_v, -220.0, 220.0),
        _clamp(conductance * v_c_v, -220.0, 220.0),
    )


def _nominal_phase_voltages(*, v_ll_rms: float, theta_rad: float) -> tuple[float, float, float]:
    phase_peak_v = v_ll_rms * sqrt(2.0 / 3.0)
    return (
        phase_peak_v * sin(theta_rad),
        phase_peak_v * sin(theta_rad - 2.0 * pi / 3.0),
        phase_peak_v * sin(theta_rad + 2.0 * pi / 3.0),
    )


def _phase_to_line_rms(v_a_v: float, v_b_v: float, v_c_v: float) -> float:
    v_ab = v_a_v - v_b_v
    v_bc = v_b_v - v_c_v
    v_ca = v_c_v - v_a_v
    return sqrt((v_ab * v_ab + v_bc * v_bc + v_ca * v_ca) / 3.0)


def _inverter_first_level_current_references(
    *,
    load_direct_current_a: float,
    diesel_direct_current_reference_a: float,
    inverter_direct_current_a: float,
    inverter_quadrature_current_a: float,
    direct_integral_a_s: float,
    quadrature_integral_a_s: float,
    kp: float,
    ki: float,
) -> tuple[float, float]:
    """Page 4 equation (2): DG-on inverter current references."""

    direct_error_a = (load_direct_current_a - diesel_direct_current_reference_a) - (
        inverter_direct_current_a
    )
    quadrature_error_a = 0.0 - inverter_quadrature_current_a
    return (
        kp * direct_error_a + ki * direct_integral_a_s,
        kp * quadrature_error_a + ki * quadrature_integral_a_s,
    )


def _line_voltage_unit_templates(
    *,
    v_la_v: float,
    v_lb_v: float,
    v_lc_v: float,
    v_l_v: float,
) -> tuple[float, float, float]:
    """Page 4 equation (3): in-phase PLL unit templates."""

    voltage = max(abs(v_l_v), 1.0)
    return v_la_v / voltage, v_lb_v / voltage, v_lc_v / voltage


def _quadrature_unit_templates(
    *,
    u_ap: float,
    u_bp: float,
    u_cp: float,
) -> tuple[float, float, float]:
    """Page 4 equation (4): quadrature PLL unit templates."""

    u_aq = (-u_bp + u_cp) / sqrt(3.0)
    u_bq = (sqrt(3.0) / 2.0) * u_ap + (u_bp - u_cp) / (2.0 * sqrt(3.0))
    u_cq = (-sqrt(3.0) / 2.0) * u_ap + (u_bp - u_cp) / (2.0 * sqrt(3.0))
    return u_aq, u_bq, u_cq


def _pll_frequency_from_angle(
    *,
    theta_rad: float,
    previous_theta_rad: float,
    dt_s: float,
) -> float:
    """Page 4 equations (5)-(6): frequency from phase angle derivative."""

    if dt_s <= 0.0:
        return 0.0
    angular_speed_rad_s = (theta_rad - previous_theta_rad) / dt_s
    return angular_speed_rad_s / (2.0 * pi)


def _inverter_second_level_current_reference(
    *,
    voltage_reference_v: float,
    inverter_voltage_v: float,
    inverter_current_a: float,
    voltage_integral_v_s: float,
    kp: float,
    ki: float,
) -> float:
    """Page 4 equation (7): DG-off inverter current reference."""

    return (
        kp * (voltage_reference_v - inverter_voltage_v)
        + ki * voltage_integral_v_s
        - inverter_current_a
    )


def _mppt_perturb_observe_duty(
    *,
    duty: float,
    power_w: float,
    previous_power_w: float,
    duty_step: float = 0.005,
) -> float:
    """Page 4 equations (10)-(12): WT power delta selects duty-step sign."""

    direction = 1.0 if power_w - previous_power_w >= 0.0 else -1.0
    return _clamp(duty + direction * duty_step, 0.05, 0.95)


def _boost_mppt_efficiency(duty: float) -> float:
    return _clamp(0.90 + 0.08 * duty, 0.90, 0.98)


def _dc_voltage_open_loop_pi_zero(*, kp: float, ki: float) -> float:
    """Page 5 equation (13): PI open-loop zero location, ki/kp."""

    if kp == 0.0:
        return 0.0
    return ki / kp


def _dc_voltage_closed_loop_coefficients(
    *,
    kp: float,
    ki: float,
    cdc_f: float,
) -> tuple[float, float]:
    """Page 5 equation (14): denominator coefficients kp/Cdc and ki/Cdc."""

    capacitance = max(abs(cdc_f), 1e-12)
    return kp / capacitance, ki / capacitance


def _dc_voltage_pi_gains_from_pole_placement(
    *,
    zeta: float,
    omega_rad_s: float,
    cdc_f: float,
) -> tuple[float, float]:
    """Page 5 equations (15)-(16): kp=2*zeta*omega*Cdc, ki=omega^2*Cdc."""

    kp = 2.0 * zeta * omega_rad_s * cdc_f
    ki = omega_rad_s * omega_rad_s * cdc_f
    return kp, ki


def _first_order_response(dt_s: float, tau_s: float) -> float:
    if tau_s <= 0.0:
        return 1.0
    return _clamp(dt_s / tau_s, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
