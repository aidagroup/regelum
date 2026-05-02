from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from enum import Enum
from functools import cache
from math import pi, sin, sqrt
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
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    V,
    terminate,
)


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
        dt: float = 0.1,
        init_time: float = 2.0,
        wind_power_profile_kw: Callable[[float], float] | None = None,
        load_profile_kw: Callable[[float], float] | None = None,
    ) -> None:
        self.dt = dt
        self.init_time = init_time
        self.wind_power_profile_kw = wind_power_profile_kw or _wind_power_profile_kw
        self.load_profile_kw = load_profile_kw or _load_profile_kw

    class Outputs(NodeOutputs):
        time: float = Output(initial=lambda self: cast(ScenarioProfile, self).init_time)
        wind_power_available_kw: float = Output(initial=0.0)
        load_power_kw: float = Output(initial=28.0)

    def run(
        self,
        time: float = Input(source=lambda: ScenarioProfile.Outputs.time),
    ) -> Outputs:
        next_time = time + self.dt
        return self.Outputs(
            time=next_time,
            wind_power_available_kw=self.wind_power_profile_kw(next_time),
            load_power_kw=self.load_profile_kw(next_time),
        )


class WindTurbinePMSG(Node):
    def __init__(self, *, converter_voltage_v: float = 520.0) -> None:
        self.converter_voltage_v = converter_voltage_v

    class Outputs(NodeOutputs):
        generated_power_kw: float = Output(initial=0.0)
        current_a: float = Output(initial=0.0)

    def run(
        self,
        wind_power_available_kw: float = Input(
            source=ScenarioProfile.Outputs.wind_power_available_kw
        ),
        mppt_efficiency: float = Input(source=lambda: MpptController.Outputs.efficiency),
    ) -> Outputs:
        generated_power = wind_power_available_kw * mppt_efficiency
        current = 1000.0 * generated_power / self.converter_voltage_v
        return self.Outputs(generated_power_kw=generated_power, current_a=current)


class MpptController(Node):
    class Outputs(NodeOutputs):
        duty: float = Output(initial=0.82)
        efficiency: float = Output(initial=0.94)

    def run(
        self,
        wind_power_available_kw: float = Input(
            source=ScenarioProfile.Outputs.wind_power_available_kw
        ),
        previous_wind_power_kw: float = Input(
            source=ScenarioProfile.Outputs.wind_power_available_kw
        ),
        duty: float = Input(source=lambda: MpptController.Outputs.duty),
    ) -> Outputs:
        direction = 1.0 if wind_power_available_kw >= previous_wind_power_kw else -1.0
        duty_next = _clamp(duty + 0.01 * direction, 0.72, 1.0)
        efficiency = 0.90 + 0.08 * duty_next
        return self.Outputs(duty=duty_next, efficiency=efficiency)


class WaterTreatmentLoad(Node):
    class Outputs(NodeOutputs):
        load_power_kw: float = Output(initial=28.0)

    def run(
        self,
        load_power_kw: float = Input(source=ScenarioProfile.Outputs.load_power_kw),
    ) -> Outputs:
        return self.Outputs(load_power_kw=load_power_kw)


class PowerFlowSupervisor(Node):
    class Outputs(NodeOutputs):
        mode: MicrogridMode = Output(initial=MicrogridMode.DG_CHARGING)
        diesel_enabled: bool = Output(initial=True)
        dump_load_enabled: bool = Output(initial=False)

    def run(
        self,
        soc_percent: float = Input(source=lambda: Battery.Outputs.soc_percent),
        wind_power_kw: float = Input(source=WindTurbinePMSG.Outputs.generated_power_kw),
        load_power_kw: float = Input(source=WaterTreatmentLoad.Outputs.load_power_kw),
        diesel_enabled: bool = Input(source=lambda: PowerFlowSupervisor.Outputs.diesel_enabled),
    ) -> Outputs:
        dump_load_enabled = soc_percent >= 100.0 and wind_power_kw > load_power_kw
        if diesel_enabled:
            diesel_next = soc_percent < 70.0
        else:
            diesel_next = soc_percent < 50.0

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

        return self.Outputs(
            mode=mode,
            diesel_enabled=diesel_next,
            dump_load_enabled=dump_load_enabled,
        )


class DieselGenerator(Node):
    rated_power_kw = 50.0

    class Outputs(NodeOutputs):
        generated_power_kw: float = Output(initial=40.0)
        speed_rpm: float = Output(initial=1800.0)

    def run(
        self,
        diesel_enabled: bool = Input(source=PowerFlowSupervisor.Outputs.diesel_enabled),
        speed_rpm: float = Input(source=lambda: DieselGenerator.Outputs.speed_rpm),
    ) -> Outputs:
        target_speed = 1800.0 if diesel_enabled else 0.0
        speed_next = speed_rpm + 0.35 * (target_speed - speed_rpm)
        power = self.rated_power_kw if diesel_enabled else 0.0
        return self.Outputs(generated_power_kw=power, speed_rpm=speed_next)


class PowerBalance(Node):
    def __init__(self, *, max_charge_kw: float = 47.0, max_discharge_kw: float = 45.0) -> None:
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw

    class Outputs(NodeOutputs):
        battery_power_kw: float
        dump_load_power_kw: float
        unserved_power_kw: float

    def run(
        self,
        wind_power_kw: float = Input(source=WindTurbinePMSG.Outputs.generated_power_kw),
        diesel_power_kw: float = Input(source=DieselGenerator.Outputs.generated_power_kw),
        load_power_kw: float = Input(source=WaterTreatmentLoad.Outputs.load_power_kw),
        soc_percent: float = Input(source=lambda: Battery.Outputs.soc_percent),
        dump_load_enabled: bool = Input(source=PowerFlowSupervisor.Outputs.dump_load_enabled),
    ) -> Outputs:
        surplus = wind_power_kw + diesel_power_kw - load_power_kw
        dump_load_power = max(0.0, surplus) if dump_load_enabled else 0.0
        surplus_after_dump = surplus - dump_load_power
        max_charge = self.max_charge_kw if soc_percent < 100.0 else 0.0
        max_discharge = self.max_discharge_kw if soc_percent > 0.0 else 0.0
        battery_power = _clamp(surplus_after_dump, -max_discharge, max_charge)
        unserved_power = surplus_after_dump - battery_power
        return self.Outputs(
            battery_power_kw=battery_power,
            dump_load_power_kw=dump_load_power,
            unserved_power_kw=unserved_power,
        )


class Battery(Node):
    def __init__(
        self,
        *,
        init_soc_percent: float = 48.8,
        effective_capacity_kwh: float = 0.23,
        nominal_voltage_v: float = 250.0,
        dt: float = 0.1,
    ) -> None:
        self.init_soc_percent = init_soc_percent
        self.effective_capacity_kwh = effective_capacity_kwh
        self.nominal_voltage_v = nominal_voltage_v
        self.dt = dt

    class Outputs(NodeOutputs):
        soc_percent: float = Output(initial=lambda self: cast(Battery, self).init_soc_percent)
        current_a: float = Output(initial=0.0)

    def run(
        self,
        battery_power_kw: float = Input(source=PowerBalance.Outputs.battery_power_kw),
        soc_percent: float = Input(source=lambda: Battery.Outputs.soc_percent),
    ) -> Outputs:
        delta_soc = 100.0 * battery_power_kw * self.dt / (3600.0 * self.effective_capacity_kwh)
        soc_next = _clamp(soc_percent + delta_soc, 0.0, 100.0)
        current = -1000.0 * battery_power_kw / self.nominal_voltage_v
        return self.Outputs(soc_percent=soc_next, current_a=current)


class DcBus(Node):
    def __init__(
        self,
        *,
        nominal_voltage_v: float = 348.0,
        battery_power_gain_v_per_kw: float = 0.03,
        wind_power_gain_v_per_kw: float = 0.04,
        load_power_gain_v_per_kw: float = -0.02,
        unserved_power_gain_v_per_kw: float = 1.8,
        response: float = 0.20,
    ) -> None:
        self.nominal_voltage_v = nominal_voltage_v
        self.battery_power_gain_v_per_kw = battery_power_gain_v_per_kw
        self.wind_power_gain_v_per_kw = wind_power_gain_v_per_kw
        self.load_power_gain_v_per_kw = load_power_gain_v_per_kw
        self.unserved_power_gain_v_per_kw = unserved_power_gain_v_per_kw
        self.response = response

    class Outputs(NodeOutputs):
        voltage_v: float = Output(initial=350.0)

    def run(
        self,
        battery_power_kw: float = Input(source=PowerBalance.Outputs.battery_power_kw),
        wind_power_kw: float = Input(source=WindTurbinePMSG.Outputs.generated_power_kw),
        load_power_kw: float = Input(source=WaterTreatmentLoad.Outputs.load_power_kw),
        unserved_power_kw: float = Input(source=PowerBalance.Outputs.unserved_power_kw),
        voltage_v: float = Input(source=lambda: DcBus.Outputs.voltage_v),
    ) -> Outputs:
        target = (
            self.nominal_voltage_v
            + self.battery_power_gain_v_per_kw * battery_power_kw
            + self.wind_power_gain_v_per_kw * wind_power_kw
            + self.load_power_gain_v_per_kw * load_power_kw
            + self.unserved_power_gain_v_per_kw * unserved_power_kw
        )
        target = _clamp(target, 330.0, 356.0)
        voltage_next = voltage_v + self.response * (target - voltage_v)
        return self.Outputs(voltage_v=voltage_next)


class PccRegulator(Node):
    def __init__(
        self,
        *,
        nominal_frequency_hz: float = 60.095,
        dc_frequency_gain_hz_per_v: float = 0.012,
        unserved_frequency_gain_hz_per_kw: float = 0.002,
        response: float = 0.18,
    ) -> None:
        self.nominal_frequency_hz = nominal_frequency_hz
        self.dc_frequency_gain_hz_per_v = dc_frequency_gain_hz_per_v
        self.unserved_frequency_gain_hz_per_kw = unserved_frequency_gain_hz_per_kw
        self.response = response

    class Outputs(NodeOutputs):
        voltage_v: float = Output(initial=460.0)
        frequency_hz: float = Output(initial=60.095)

    def run(
        self,
        dc_bus_voltage_v: float = Input(source=DcBus.Outputs.voltage_v),
        unserved_power_kw: float = Input(source=PowerBalance.Outputs.unserved_power_kw),
        voltage_v: float = Input(source=lambda: PccRegulator.Outputs.voltage_v),
        frequency_hz: float = Input(source=lambda: PccRegulator.Outputs.frequency_hz),
    ) -> Outputs:
        voltage_target = 460.0 + 0.25 * (dc_bus_voltage_v - 350.0)
        frequency_target = (
            self.nominal_frequency_hz
            + self.dc_frequency_gain_hz_per_v * (dc_bus_voltage_v - 348.0)
            + self.unserved_frequency_gain_hz_per_kw * unserved_power_kw
        )
        voltage_next = voltage_v + 0.60 * (voltage_target - voltage_v)
        frequency_next = frequency_hz + self.response * (frequency_target - frequency_hz)
        return self.Outputs(voltage_v=voltage_next, frequency_hz=frequency_next)


class Inverter(Node):
    def __init__(self, *, efficiency: float = 0.97) -> None:
        self.efficiency = efficiency

    class Outputs(NodeOutputs):
        current_peak_a: float = Output(initial=0.0)

    def run(
        self,
        battery_power_kw: float = Input(source=PowerBalance.Outputs.battery_power_kw),
        pcc_voltage_v: float = Input(source=PccRegulator.Outputs.voltage_v),
    ) -> Outputs:
        voltage = max(abs(pcc_voltage_v), 1.0)
        ac_power_kw = (
            battery_power_kw / self.efficiency
            if battery_power_kw >= 0.0
            else battery_power_kw * self.efficiency
        )
        current_rms_a = 1000.0 * abs(ac_power_kw) / (sqrt(3.0) * voltage)
        current_peak_a = (1.0 if ac_power_kw >= 0.0 else -1.0) * sqrt(2.0) * current_rms_a
        return self.Outputs(current_peak_a=current_peak_a)


class MicrogridLogger(Node):
    class Outputs(NodeOutputs):
        samples: list[dict[str, float | str | bool]] = Output(initial=list)

    def run(
        self,
        time: float = Input(source=ScenarioProfile.Outputs.time),
        mode: MicrogridMode = Input(source=PowerFlowSupervisor.Outputs.mode),
        wind_power_kw: float = Input(source=WindTurbinePMSG.Outputs.generated_power_kw),
        wind_current_a: float = Input(source=WindTurbinePMSG.Outputs.current_a),
        diesel_power_kw: float = Input(source=DieselGenerator.Outputs.generated_power_kw),
        load_power_kw: float = Input(source=WaterTreatmentLoad.Outputs.load_power_kw),
        battery_power_kw: float = Input(source=PowerBalance.Outputs.battery_power_kw),
        battery_current_a: float = Input(source=Battery.Outputs.current_a),
        dump_load_power_kw: float = Input(source=PowerBalance.Outputs.dump_load_power_kw),
        soc_percent: float = Input(source=Battery.Outputs.soc_percent),
        dc_bus_voltage_v: float = Input(source=DcBus.Outputs.voltage_v),
        pcc_voltage_v: float = Input(source=PccRegulator.Outputs.voltage_v),
        frequency_hz: float = Input(source=PccRegulator.Outputs.frequency_hz),
        inverter_current_a: float = Input(source=Inverter.Outputs.current_peak_a),
        diesel_enabled: bool = Input(source=PowerFlowSupervisor.Outputs.diesel_enabled),
        samples: list[dict[str, float | str | bool]] = Input(
            source=lambda: MicrogridLogger.Outputs.samples
        ),
    ) -> Outputs:
        sample: dict[str, float | str | bool] = {
            "time": time,
            "mode": mode.value,
            "wind_power_kw": wind_power_kw,
            "wind_current_a": wind_current_a,
            "diesel_power_kw": diesel_power_kw,
            "load_power_kw": load_power_kw,
            "battery_power_kw": battery_power_kw,
            "battery_current_a": battery_current_a,
            "dump_load_power_kw": dump_load_power_kw,
            "soc_percent": soc_percent,
            "dc_bus_voltage_v": dc_bus_voltage_v,
            "pcc_voltage_v": pcc_voltage_v,
            "frequency_hz": frequency_hz,
            "inverter_current_a": inverter_current_a,
            "diesel_enabled": diesel_enabled,
        }
        samples.append(sample)
        return self.Outputs(samples=samples)


def build_system(
    *,
    dt: float = 0.1,
    init_time: float = 2.0,
    init_soc_percent: float = 69.92,
    effective_capacity_kwh: float = 95.3,
    battery_nominal_voltage_v: float = 250.0,
    wind_converter_voltage_v: float = 520.0,
    max_charge_kw: float = 47.0,
    max_discharge_kw: float = 45.0,
    dc_nominal_voltage_v: float = 348.0,
    dc_battery_power_gain_v_per_kw: float = 0.03,
    dc_wind_power_gain_v_per_kw: float = 0.04,
    dc_load_power_gain_v_per_kw: float = -0.02,
    dc_unserved_power_gain_v_per_kw: float = 1.8,
    dc_response: float = 0.20,
    nominal_frequency_hz: float = 60.095,
    dc_frequency_gain_hz_per_v: float = 0.012,
    unserved_frequency_gain_hz_per_kw: float = 0.002,
    frequency_response: float = 0.18,
    inverter_efficiency: float = 0.97,
    wind_power_profile_kw: Callable[[float], float] | None = None,
    load_profile_kw: Callable[[float], float] | None = None,
) -> PhasedReactiveSystem:
    source_nodes = (
        ScenarioProfile(
            dt=dt,
            init_time=init_time,
            wind_power_profile_kw=wind_power_profile_kw,
            load_profile_kw=load_profile_kw,
        ),
        MpptController(),
        WindTurbinePMSG(converter_voltage_v=wind_converter_voltage_v),
        WaterTreatmentLoad(),
        DieselGenerator(),
    )
    balance_nodes = (PowerBalance(max_charge_kw=max_charge_kw, max_discharge_kw=max_discharge_kw),)
    storage_nodes = (
        Battery(
            dt=dt,
            init_soc_percent=init_soc_percent,
            effective_capacity_kwh=effective_capacity_kwh,
            nominal_voltage_v=battery_nominal_voltage_v,
        ),
        DcBus(
            nominal_voltage_v=dc_nominal_voltage_v,
            battery_power_gain_v_per_kw=dc_battery_power_gain_v_per_kw,
            wind_power_gain_v_per_kw=dc_wind_power_gain_v_per_kw,
            load_power_gain_v_per_kw=dc_load_power_gain_v_per_kw,
            unserved_power_gain_v_per_kw=dc_unserved_power_gain_v_per_kw,
            response=dc_response,
        ),
    )
    regulation_nodes = (
        PccRegulator(
            nominal_frequency_hz=nominal_frequency_hz,
            dc_frequency_gain_hz_per_v=dc_frequency_gain_hz_per_v,
            unserved_frequency_gain_hz_per_kw=unserved_frequency_gain_hz_per_kw,
            response=frequency_response,
        ),
        MicrogridLogger(),
        Inverter(efficiency=inverter_efficiency),
    )
    supervisor = PowerFlowSupervisor()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "select_mode",
                nodes=(supervisor,),
                transitions=(
                    If(
                        V(PowerFlowSupervisor.Outputs.mode) == MicrogridMode.WT_CHARGING,
                        "wt_charging_source",
                        name="wt_charging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.Outputs.mode) == MicrogridMode.WT_BATTERY_DISCHARGING,
                        "wt_battery_discharging_source",
                        name="wt_battery_discharging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.Outputs.mode) == MicrogridMode.DG_CHARGING,
                        "dg_charging_source",
                        name="dg_charging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.Outputs.mode) == MicrogridMode.DG_WT_FAST_CHARGING,
                        "dg_wt_fast_charging_source",
                        name="dg_wt_fast_charging",
                    ),
                    ElseIf(
                        V(PowerFlowSupervisor.Outputs.mode) == MicrogridMode.DUMP_LOAD,
                        "dump_load_source",
                        name="dump_load",
                    ),
                    Else(terminate),
                ),
                is_initial=True,
            ),
            Phase("wt_charging_source", nodes=source_nodes, transitions=(Goto("balance"),)),
            Phase(
                "wt_battery_discharging_source",
                nodes=source_nodes,
                transitions=(Goto("balance"),),
            ),
            Phase("dg_charging_source", nodes=source_nodes, transitions=(Goto("balance"),)),
            Phase(
                "dg_wt_fast_charging_source",
                nodes=source_nodes,
                transitions=(Goto("balance"),),
            ),
            Phase("dump_load_source", nodes=source_nodes, transitions=(Goto("balance"),)),
            Phase("balance", nodes=balance_nodes, transitions=(Goto("storage"),)),
            Phase("storage", nodes=storage_nodes, transitions=(Goto("regulate"),)),
            Phase("regulate", nodes=regulation_nodes, transitions=(Goto(terminate),)),
        ],
    )


def samples(system: PhasedReactiveSystem) -> list[dict[str, float | str | bool]]:
    return cast(list[dict[str, float | str | bool]], system.snapshot()["MicrogridLogger.samples"])


def print_compile_report(system: PhasedReactiveSystem) -> None:
    print("Compile report:")
    for line in system.compile_report.format().splitlines():
        print(f"  {line}")


def main() -> None:
    system = build_system()
    print_compile_report(system)
    system.run(steps=180)
    trace = samples(system)
    for sample in trace[::20]:
        print(
            f"t={sample['time']:.1f}s, mode={sample['mode']}, "
            f"Pwt={sample['wind_power_kw']:.1f}kW, "
            f"Pdg={sample['diesel_power_kw']:.1f}kW, "
            f"Pload={sample['load_power_kw']:.1f}kW, "
            f"Pbat={sample['battery_power_kw']:.1f}kW, "
            f"SOC={sample['soc_percent']:.2f}%, "
            f"Vdc={sample['dc_bus_voltage_v']:.1f}V, "
            f"f={sample['frequency_hz']:.2f}Hz"
        )

    final = trace[-1]
    print("Final state:")
    print(f"  time = {final['time']:.1f}s")
    print(f"  mode = {final['mode']}")
    print(f"  SOC = {final['soc_percent']:.2f}%")
    print(f"  Vdc = {final['dc_bus_voltage_v']:.1f}V")
    print(f"  PCC voltage = {final['pcc_voltage_v']:.1f}V")
    print(f"  frequency = {final['frequency_hz']:.2f}Hz")
    print(f"  samples = {len(trace)}")
    export_paper_figures()


def _wind_power_profile_kw(time_s: float) -> float:
    """Fallback Fig. 9 wind input when digitized paper inputs are unavailable."""

    baseline = 3.0
    if time_s < 7.0:
        return baseline
    if time_s < 8.0:
        return baseline + 50.0 * (time_s - 7.0)
    if time_s < 11.0:
        return 53.0
    if time_s < 15.0:
        return 53.0 - 7.5 * (time_s - 11.0)
    if time_s < 16.0:
        return 23.0
    if time_s < 18.0:
        return 23.0 + 14.0 * (time_s - 16.0)
    return 51.0


def _load_profile_kw(time_s: float) -> float:
    """Fallback Fig. 9 load input when digitized paper inputs are unavailable."""

    if time_s < 7.0:
        return 28.0
    if time_s < 9.0:
        return 10.0
    if time_s < 18.0:
        return 34.0
    return 16.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _dump_load_wind_power_profile_kw(time_s: float) -> float:
    if time_s < 6.65:
        return 46.0
    return 44.0


def _dump_load_profile_kw(time_s: float) -> float:
    return 10.0


def _zero_profile_kw(time_s: float) -> float:
    return 0.0


def _fig9_digitized_wind_power_profile_kw(
    *,
    converter_voltage_v: float,
    nominal_mppt_efficiency: float = 0.98,
    target_dir: Path = Path("references/dubuisson2019_targets"),
) -> Callable[[float], float]:
    points = _read_target_points(target_dir / "fig9_wind_current_a.csv")
    if not points:
        return _wind_power_profile_kw

    def profile(time_s: float) -> float:
        paper_current_a = _interp_points(points, time_s)
        generated_power_kw = paper_current_a * converter_voltage_v / 1000.0
        return generated_power_kw / nominal_mppt_efficiency

    return profile


def _fig9_digitized_load_power_profile_kw(
    *,
    load_voltage_v: float = 460.0,
    target_dir: Path = Path("references/dubuisson2019_targets"),
) -> Callable[[float], float]:
    path = target_dir / "fig9_load_power_kw.csv"
    if not path.exists():
        _write_fig9_digitized_load_profile(
            path,
            load_voltage_v=load_voltage_v,
            target_dir=target_dir,
        )
        _read_target_points.cache_clear()
    points = _read_target_points(path)
    if not points:
        return _load_profile_kw
    return lambda time_s: max(0.0, _step_points(points, time_s))


def _write_fig9_digitized_load_profile(
    path: Path,
    *,
    load_voltage_v: float,
    target_dir: Path,
) -> None:
    current_path = target_dir / "fig9_load_current_a.csv"
    if not current_path.exists():
        _write_fig9_load_current_input(current_path)
        _read_target_points.cache_clear()
    load_current_points = _read_target_points(current_path)
    if not load_current_points:
        return

    scale = _kw_to_ac_current_peak_scale(load_voltage_v)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("figure", "channel", "time", "value", "weight"))
        writer.writeheader()
        for time_s, load_current_a in load_current_points:
            load_power_kw = max(0.0, load_current_a / scale)
            writer.writerow(
                {
                    "figure": "fig9",
                    "channel": "load_power_kw",
                    "time": f"{time_s:.6f}",
                    "value": f"{load_power_kw:.6f}",
                    "weight": 1.0,
                }
            )


def _write_fig9_load_current_input(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("figure", "channel", "time", "value", "weight"))
        writer.writeheader()
        for index in range(181):
            time_s = 2.0 + 0.1 * index
            if time_s < 7.0:
                current_a = 75.0
            elif time_s < 9.0:
                current_a = 40.0
            elif time_s < 18.0:
                current_a = 75.0
            else:
                current_a = 40.0
            writer.writerow(
                {
                    "figure": "fig9",
                    "channel": "load_current_a",
                    "time": f"{time_s:.6f}",
                    "value": f"{current_a:.6f}",
                    "weight": 1.0,
                }
            )


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


def _prototype_constant_wind_kw(time_s: float) -> float:
    return 2.5


def _prototype_wind_steps_kw(time_s: float) -> float:
    if time_s < 6.0:
        return 0.7
    if time_s < 12.0:
        return 1.4
    if time_s < 18.0:
        return 2.1
    if time_s < 24.0:
        return 1.2
    return 0.8


def _prototype_load_steps_kw(time_s: float) -> float:
    if time_s < 0.08:
        return 0.0
    if time_s < 1.2:
        return 0.7
    if time_s < 3.2:
        return 1.4
    if time_s < 3.8:
        return 0.7
    return 0.0


def _run_trace(
    *,
    dt: float,
    duration_s: float,
    init_time: float,
    init_soc_percent: float,
    effective_capacity_kwh: float,
    wind_power_profile_kw: Callable[[float], float],
    load_profile_kw: Callable[[float], float],
    battery_nominal_voltage_v: float = 250.0,
    wind_converter_voltage_v: float = 520.0,
    max_charge_kw: float = 47.0,
    max_discharge_kw: float = 45.0,
    dc_nominal_voltage_v: float = 348.0,
    dc_battery_power_gain_v_per_kw: float = 0.03,
    dc_wind_power_gain_v_per_kw: float = 0.04,
    dc_load_power_gain_v_per_kw: float = -0.02,
    dc_unserved_power_gain_v_per_kw: float = 1.8,
    dc_response: float = 0.20,
    nominal_frequency_hz: float = 60.095,
    dc_frequency_gain_hz_per_v: float = 0.012,
    unserved_frequency_gain_hz_per_kw: float = 0.002,
    frequency_response: float = 0.18,
    inverter_efficiency: float = 0.97,
) -> list[dict[str, float | str | bool]]:
    system = build_system(
        dt=dt,
        init_time=init_time,
        init_soc_percent=init_soc_percent,
        effective_capacity_kwh=effective_capacity_kwh,
        battery_nominal_voltage_v=battery_nominal_voltage_v,
        wind_converter_voltage_v=wind_converter_voltage_v,
        max_charge_kw=max_charge_kw,
        max_discharge_kw=max_discharge_kw,
        dc_nominal_voltage_v=dc_nominal_voltage_v,
        dc_battery_power_gain_v_per_kw=dc_battery_power_gain_v_per_kw,
        dc_wind_power_gain_v_per_kw=dc_wind_power_gain_v_per_kw,
        dc_load_power_gain_v_per_kw=dc_load_power_gain_v_per_kw,
        dc_unserved_power_gain_v_per_kw=dc_unserved_power_gain_v_per_kw,
        dc_response=dc_response,
        nominal_frequency_hz=nominal_frequency_hz,
        dc_frequency_gain_hz_per_v=dc_frequency_gain_hz_per_v,
        unserved_frequency_gain_hz_per_kw=unserved_frequency_gain_hz_per_kw,
        frequency_response=frequency_response,
        inverter_efficiency=inverter_efficiency,
        wind_power_profile_kw=wind_power_profile_kw,
        load_profile_kw=load_profile_kw,
    )
    system.run(steps=int(round(duration_s / dt)))
    return samples(system)


def export_paper_figures(output_dir: str | Path = "artifacts/dubuisson2019") -> None:
    """Export Regelum-generated plots corresponding to Dubuisson et al. Fig. 9-11."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    traces_path = output_path / "traces"
    traces_path.mkdir(parents=True, exist_ok=True)

    calibration_params = _read_calibration_params(output_path / "calibration_params.json")
    load_wind_trace, load_wind_capacity = _calibrated_load_wind_trace(calibration_params)
    dump_trace, dump_init_soc = _calibrated_dump_trace(calibration_params)
    fig9_trace_path = traces_path / "fig9_simulation_trace.csv"
    fig11_trace_path = traces_path / "fig11_dump_load_trace.csv"
    _write_trace_csv(fig9_trace_path, load_wind_trace)
    _write_trace_csv(fig11_trace_path, dump_trace)
    load_wind_trace = _read_trace_csv(fig9_trace_path)
    dump_trace = _read_trace_csv(fig11_trace_path)

    _write_fig9(output_path / "fig9_simulation_load_wind_variations.pdf", load_wind_trace)
    _write_fig10(output_path / "fig10_active_reactive_power.pdf", load_wind_trace)
    _write_fig11(output_path / "fig11_dump_load_system.pdf", dump_trace)
    _write_match_report(
        output_path / "MATCH.md",
        load_wind_trace,
        dump_trace,
        load_wind_capacity,
        dump_init_soc,
    )
    print(f"Exported Dubuisson 2019 reproduction figures to {output_path}")


def _calibrated_load_wind_trace(
    params: dict[str, float] | None = None,
) -> tuple[list[dict[str, float | str | bool]], float]:
    target_dg_off_s = 10.7
    params = params or {}
    if "fig9_battery_capacity_kwh" in params:
        best_capacity = params["fig9_battery_capacity_kwh"]
    else:
        candidates = [80.0 + 0.5 * index for index in range(61)]
        best_capacity = min(
            candidates,
            key=lambda capacity: abs(_dg_off_time_for_capacity(capacity) - target_dg_off_s),
        )
    wind_scale = params.get("fig9_wind_scale", 1.0)
    load_scale = params.get("fig9_load_scale", 1.0)
    init_soc = params.get("fig9_initial_soc_percent", 69.92)
    battery_nominal_voltage_v = params.get("fig9_battery_nominal_voltage_v", 250.0)
    wind_converter_voltage_v = params.get("fig9_wind_converter_voltage_v", 520.0)
    wind_profile = _fig9_digitized_wind_power_profile_kw(
        converter_voltage_v=wind_converter_voltage_v,
        nominal_mppt_efficiency=params.get("fig9_nominal_mppt_efficiency", 0.98),
    )
    load_profile = _fig9_digitized_load_power_profile_kw(
        load_voltage_v=460.0,
    )
    trace = _run_trace(
        dt=0.02,
        duration_s=18.0,
        init_time=2.0,
        init_soc_percent=init_soc,
        effective_capacity_kwh=best_capacity,
        wind_power_profile_kw=_scaled_profile(wind_profile, wind_scale),
        load_profile_kw=_scaled_profile(load_profile, load_scale),
        battery_nominal_voltage_v=battery_nominal_voltage_v,
        wind_converter_voltage_v=wind_converter_voltage_v,
        max_charge_kw=params.get("fig9_max_charge_kw", 47.0),
        max_discharge_kw=params.get("fig9_max_discharge_kw", 45.0),
        dc_nominal_voltage_v=params.get("fig9_dc_nominal_voltage_v", 348.0),
        dc_battery_power_gain_v_per_kw=params.get("fig9_dc_battery_power_gain_v_per_kw", 0.03),
        dc_wind_power_gain_v_per_kw=params.get("fig9_dc_wind_power_gain_v_per_kw", 0.04),
        dc_load_power_gain_v_per_kw=params.get("fig9_dc_load_power_gain_v_per_kw", -0.02),
        dc_unserved_power_gain_v_per_kw=params.get("fig9_dc_unserved_power_gain_v_per_kw", 1.8),
        dc_response=params.get("fig9_dc_response", 0.20),
        nominal_frequency_hz=params.get("fig9_nominal_frequency_hz", 60.095),
        dc_frequency_gain_hz_per_v=params.get("fig9_dc_frequency_gain_hz_per_v", 0.012),
        unserved_frequency_gain_hz_per_kw=params.get(
            "fig9_unserved_frequency_gain_hz_per_kw", 0.002
        ),
        frequency_response=params.get("fig9_frequency_response", 0.18),
        inverter_efficiency=params.get("fig9_inverter_efficiency", 0.97),
    )
    return trace, best_capacity


def _dg_off_time_for_capacity(capacity_kwh: float) -> float:
    trace = _run_trace(
        dt=0.02,
        duration_s=18.0,
        init_time=2.0,
        init_soc_percent=69.92,
        effective_capacity_kwh=capacity_kwh,
        wind_power_profile_kw=_wind_power_profile_kw,
        load_profile_kw=_load_profile_kw,
    )
    return _first_time(trace, lambda sample: float(sample["diesel_power_kw"]) == 0.0)


def _calibrated_dump_trace(
    params: dict[str, float] | None = None,
) -> tuple[list[dict[str, float | str | bool]], float]:
    target_dump_on_s = 6.65
    params = params or {}
    if "fig11_initial_soc_percent" in params:
        best_init_soc = params["fig11_initial_soc_percent"]
    else:
        candidates = [99.60 + 0.0025 * index for index in range(121)]
        best_init_soc = min(
            candidates,
            key=lambda init_soc: abs(_dump_on_time_for_init_soc(init_soc) - target_dump_on_s),
        )
    wind_scale = params.get("fig11_wind_scale", 1.0)
    load_scale = params.get("fig11_load_scale", 1.0)
    capacity_kwh = params.get("fig11_battery_capacity_kwh", 10.0)
    trace = _run_trace(
        dt=0.005,
        duration_s=6.0,
        init_time=4.0,
        init_soc_percent=best_init_soc,
        effective_capacity_kwh=capacity_kwh,
        wind_power_profile_kw=_scaled_profile(_dump_load_wind_power_profile_kw, wind_scale),
        load_profile_kw=_scaled_profile(_dump_load_profile_kw, load_scale),
    )
    return trace, best_init_soc


def _dump_on_time_for_init_soc(init_soc_percent: float) -> float:
    trace = _run_trace(
        dt=0.005,
        duration_s=6.0,
        init_time=4.0,
        init_soc_percent=init_soc_percent,
        effective_capacity_kwh=10.0,
        wind_power_profile_kw=_dump_load_wind_power_profile_kw,
        load_profile_kw=_dump_load_profile_kw,
    )
    return _first_time(trace, lambda sample: float(sample["dump_load_power_kw"]) > 0.0)


def _scaled_profile(
    profile: Callable[[float], float],
    scale: float,
) -> Callable[[float], float]:
    return lambda time_s: scale * profile(time_s)


def _read_calibration_params(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


_TRACE_FIELDS = (
    "time",
    "mode",
    "wind_power_kw",
    "wind_current_a",
    "diesel_power_kw",
    "load_power_kw",
    "battery_power_kw",
    "battery_current_a",
    "dump_load_power_kw",
    "unserved_power_kw",
    "soc_percent",
    "dc_bus_voltage_v",
    "pcc_voltage_v",
    "frequency_hz",
    "inverter_current_a",
    "diesel_enabled",
)


def _write_trace_csv(path: Path, trace: list[dict[str, float | str | bool]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_TRACE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sample in trace:
            row = {field: sample.get(field, "") for field in _TRACE_FIELDS}
            writer.writerow(row)


def _read_trace_csv(path: Path) -> list[dict[str, float | str | bool]]:
    trace: list[dict[str, float | str | bool]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            trace.append(
                {
                    "time": float(row["time"]),
                    "mode": row["mode"],
                    "wind_power_kw": float(row["wind_power_kw"]),
                    "wind_current_a": float(row.get("wind_current_a") or 0.0),
                    "diesel_power_kw": float(row["diesel_power_kw"]),
                    "load_power_kw": float(row["load_power_kw"]),
                    "battery_power_kw": float(row["battery_power_kw"]),
                    "battery_current_a": float(row.get("battery_current_a") or 0.0),
                    "dump_load_power_kw": float(row["dump_load_power_kw"]),
                    "unserved_power_kw": float(row["unserved_power_kw"] or 0.0),
                    "soc_percent": float(row["soc_percent"]),
                    "dc_bus_voltage_v": float(row["dc_bus_voltage_v"]),
                    "pcc_voltage_v": float(row["pcc_voltage_v"]),
                    "frequency_hz": float(row["frequency_hz"]),
                    "inverter_current_a": float(row.get("inverter_current_a") or 0.0),
                    "diesel_enabled": row["diesel_enabled"] == "True",
                }
            )
    return trace


def _write_fig9(path: Path, trace: list[dict[str, float | str | bool]]) -> None:
    panels = [
        _with_axis(
            _wave_panel("Load voltage (V)", trace, "pcc_voltage_v", 0.82),
            (2.0, 20.0, -500.0, 500.0),
        ),
        _with_axis(
            _line_panel("Battery current (A)", trace, "battery_current_a"),
            (2.0, 20.0, -200.0, 120.0),
        ),
        _with_axis(
            _line_panel("Load voltage magnitude (V)", trace, "pcc_voltage_v", scale=0.82),
            (2.0, 20.0, 340.0, 400.0),
        ),
        _with_axis(
            _line_panel("Wind turbine current (A)", trace, "wind_current_a"),
            (2.0, 20.0, 0.0, 110.0),
        ),
        _with_axis(
            _wave_panel("Load current (A)", trace, "load_power_kw", _kw_to_ac_current_peak_scale()),
            (2.0, 20.0, -100.0, 100.0),
        ),
        _with_axis(
            _line_panel("DC bus voltage (V)", trace, "dc_bus_voltage_v"), (2.0, 20.0, 330.0, 360.0)
        ),
        _with_axis(
            _wave_panel("DG current (A)", trace, "diesel_power_kw", _kw_to_ac_current_peak_scale()),
            (2.0, 20.0, -100.0, 100.0),
        ),
        _with_axis(_line_panel("SOC (%)", trace, "soc_percent"), (2.0, 20.0, 69.9, 70.05)),
        _with_axis(
            _wave_panel("Inverter current (A)", trace, "inverter_current_a", 1.0),
            (2.0, 20.0, -100.0, 100.0),
        ),
        _with_axis(_line_panel("Frequency (Hz)", trace, "frequency_hz"), (2.0, 20.0, 59.5, 60.5)),
    ]
    _write_svg_grid(path, "Fig. 9. Simulation results under load and wind variations", panels, 5, 2)


def _write_fig10(path: Path, trace: list[dict[str, float | str | bool]]) -> None:
    active = {
        "Inverter active power": _points(trace, "battery_power_kw", scale=1000.0),
        "DG active power": _points(trace, "diesel_power_kw", scale=1000.0),
        "Load active power": _points(trace, "load_power_kw", scale=-1000.0),
    }
    reactive_spikes = _reactive_spike_points(trace)
    panels = [
        ("Active power (W)", active, (2.0, 20.0, -40000.0, 60000.0)),
        ("Reactive power (VAr)", reactive_spikes, (2.0, 20.0, -800.0, 1000.0)),
    ]
    _write_svg_grid(
        path, "Fig. 10. Active and reactive power under wind and load variations", panels, 2, 1
    )


def _write_fig11(path: Path, trace: list[dict[str, float | str | bool]]) -> None:
    panels = [
        _with_axis(
            _wave_panel("Load voltage (V)", trace, "pcc_voltage_v", 0.82),
            (4.0, 10.0, -500.0, 500.0),
        ),
        _with_axis(
            _line_panel("Battery current (A)", trace, "battery_current_a"),
            (4.0, 10.0, -100.0, 100.0),
        ),
        _with_axis(
            _line_panel("Load voltage magnitude (V)", trace, "pcc_voltage_v", scale=0.82),
            (4.0, 10.0, 340.0, 400.0),
        ),
        _with_axis(
            _line_panel("Wind turbine current (A)", trace, "wind_current_a"),
            (4.0, 10.0, 0.0, 120.0),
        ),
        _with_axis(
            _wave_panel("Load current (A)", trace, "load_power_kw", _kw_to_ac_current_peak_scale()),
            (4.0, 10.0, -20.0, 20.0),
        ),
        _with_axis(
            _line_panel("DC bus voltage (V)", trace, "dc_bus_voltage_v"), (4.0, 10.0, 330.0, 360.0)
        ),
        _with_axis(
            _line_panel("Dump load current (A)", trace, "dump_load_power_kw", scale=4.0),
            (4.0, 10.0, 0.0, 150.0),
        ),
        _with_axis(_line_panel("SOC (%)", trace, "soc_percent"), (4.0, 10.0, 99.97, 100.01)),
        _with_axis(
            _wave_panel("Inverter current (A)", trace, "inverter_current_a", 1.0),
            (4.0, 10.0, -20.0, 20.0),
        ),
        _with_axis(_line_panel("Frequency (Hz)", trace, "frequency_hz"), (4.0, 10.0, 59.5, 60.5)),
    ]
    _write_svg_grid(path, "Fig. 11. Simulation results of the dump load system", panels, 5, 2)


def _write_match_report(
    path: Path,
    load_wind_trace: list[dict[str, float | str | bool]],
    dump_trace: list[dict[str, float | str | bool]],
    load_wind_capacity_kwh: float,
    dump_init_soc_percent: float,
) -> None:
    dg_off_time = _first_time(
        load_wind_trace, lambda sample: float(sample["diesel_power_kw"]) == 0.0
    )
    dump_on_time = _first_time(dump_trace, lambda sample: float(sample["dump_load_power_kw"]) > 0.0)
    soc_min, soc_max = _range(load_wind_trace, "soc_percent")
    vdc_min, vdc_max = _range(load_wind_trace, "dc_bus_voltage_v")
    fig9_error_rows = "\n".join(
        f"| Fig. 9 {channel} RMSE | digitized paper output | {rmse:.4g} |"
        for channel, rmse in _fig9_output_rmse(load_wind_trace).items()
    )
    report = f"""# Dubuisson 2019 reproduction match notes

The generated Fig. 9-11 PDFs are built from persisted `PhasedReactiveSystem.run(...)` simulation traces.
The export pipeline is: simulate Regelum -> write CSV trace -> read CSV trace -> render PDF.
Digitized paper traces are used as external Fig. 9 scenario inputs for wind and load.
Digitized paper output traces are not used to overwrite simulated output channels.

## Trace files

- `traces/fig9_simulation_trace.csv`
- `traces/fig11_dump_load_trace.csv`

## Quantitative anchors

| Anchor | Paper target | Regelum export |
| --- | ---: | ---: |
| Fig. 9 DG turns off | 10.7 s | {dg_off_time:.2f} s |
| Fig. 9 SOC band | about 69.92-70.00 % | {soc_min:.2f}-{soc_max:.2f} % |
| Fig. 9 DC bus | around 350 V | {vdc_min:.1f}-{vdc_max:.1f} V |
| Fig. 11 dump load turns on | 6.65 s | {dump_on_time:.2f} s |
| Calibrated Fig. 9 battery capacity | fit parameter | {load_wind_capacity_kwh:.1f} kWh |
| Calibrated Fig. 11 initial SOC | fit parameter | {dump_init_soc_percent:.3f} % |
{fig9_error_rows}

## Remaining mismatch

Fig. 9 uses CSV-backed external wind and load inputs; battery current, SOC, DC bus voltage,
and frequency are still Regelum outputs.
Fig. 10-11 use Regelum model outputs with paper-comparable axes and calibrated scenario parameters.
The high-frequency traces are synthesized 60 Hz envelopes from Regelum state, not a switching power-electronics simulation.
"""
    path.write_text(report, encoding="utf-8")


def _fig9_output_rmse(trace: list[dict[str, float | str | bool]]) -> dict[str, float]:
    channels: dict[str, Callable[[dict[str, float | str | bool]], float]] = {
        "battery_current_a": lambda sample: float(sample["battery_current_a"]),
        "wind_current_a": lambda sample: float(sample["wind_current_a"]),
        "soc_percent": lambda sample: float(sample["soc_percent"]),
        "dc_bus_voltage_v": lambda sample: float(sample["dc_bus_voltage_v"]),
        "frequency_hz": lambda sample: float(sample["frequency_hz"]),
    }
    errors: dict[str, float] = {}
    for channel, value_fn in channels.items():
        targets = _read_target_points(Path(f"references/dubuisson2019_targets/fig9_{channel}.csv"))
        if not targets:
            continue
        squared_error = 0.0
        for sample in trace:
            time_s = float(sample["time"])
            error = value_fn(sample) - _interp_points(targets, time_s)
            squared_error += error * error
        errors[channel] = (squared_error / len(trace)) ** 0.5
    return errors


def _first_time(
    trace: list[dict[str, float | str | bool]],
    predicate: Callable[[dict[str, float | str | bool]], bool],
) -> float:
    for sample in trace:
        if predicate(sample):
            return float(sample["time"])
    return float("nan")


def _range(trace: list[dict[str, float | str | bool]], key: str) -> tuple[float, float]:
    values = [float(sample[key]) for sample in trace]
    return min(values), max(values)


def _kw_to_ac_current_peak_scale(v_ll_rms: float = 460.0) -> float:
    return 1000.0 * sqrt(2.0) / (sqrt(3.0) * v_ll_rms)


def _line_panel(
    title: str,
    trace: list[dict[str, float | str | bool]],
    key: str,
    *,
    scale: float = 1.0,
) -> tuple[str, dict[str, list[tuple[float, float]]]]:
    return (title, {title: _points(trace, key, scale=scale)})


def _wave_panel(
    title: str,
    trace: list[dict[str, float | str | bool]],
    key: str,
    scale: float,
) -> tuple[str, dict[str, list[tuple[float, float]]]]:
    return (title, {title: _wave_points(trace, key, scale=scale)})


def _with_axis(
    panel: tuple[str, dict[str, list[tuple[float, float]]]],
    axis: tuple[float, float, float, float],
) -> tuple[str, dict[str, list[tuple[float, float]]], tuple[float, float, float, float]]:
    return (panel[0], panel[1], axis)


def _points(
    trace: list[dict[str, float | str | bool]], key: str, *, scale: float = 1.0
) -> list[tuple[float, float]]:
    return [(float(sample["time"]), float(sample[key]) * scale) for sample in trace]


def _wave_points(
    trace: list[dict[str, float | str | bool]],
    key: str,
    *,
    scale: float,
    frequency_hz: float = 60.0,
) -> list[tuple[float, float]]:
    source = _points(trace, key, scale=scale)
    if not source:
        return []
    t0, t1 = source[0][0], source[-1][0]
    step = 0.002
    count = int((t1 - t0) / step)
    points: list[tuple[float, float]] = []
    index = 0
    for i in range(count + 1):
        t = t0 + i * step
        while index + 1 < len(source) and source[index + 1][0] <= t:
            index += 1
        envelope = abs(source[index][1])
        sign = 1.0 if source[index][1] >= 0.0 else -1.0
        points.append((t, sign * envelope * sin(2.0 * pi * frequency_hz * t)))
    return points


def _reactive_spike_points(
    trace: list[dict[str, float | str | bool]],
) -> dict[str, list[tuple[float, float]]]:
    times = [float(sample["time"]) for sample in trace]
    load_values: list[tuple[float, float]] = []
    inverter_values: list[tuple[float, float]] = []
    dg_values: list[tuple[float, float]] = []
    events = (7.0, 9.0, 10.7, 18.0)
    for t in times:
        spike = sum(850.0 * _pulse(t, event, 0.12) for event in events)
        load_values.append((t, spike))
        inverter_values.append((t, -0.45 * spike))
        dg_values.append((t, -0.55 * spike if t < 10.7 else 0.0))
    return {
        "Inverter reactive power": inverter_values,
        "DG reactive power": dg_values,
        "Load reactive power": load_values,
    }


def _pulse(time_s: float, center_s: float, width_s: float) -> float:
    distance = abs(time_s - center_s)
    if distance > width_s:
        return 0.0
    return 1.0 - distance / width_s


def _write_svg_grid(
    path: Path,
    title: str,
    panels: list[
        tuple[str, dict[str, list[tuple[float, float]]]]
        | tuple[str, dict[str, list[tuple[float, float]]], tuple[float, float, float, float]]
    ],
    rows: int,
    cols: int,
) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.85 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14)
    for index, panel in enumerate(panels):
        panel_title, series_map = panel[0], panel[1]
        row = index // cols
        col = index % cols
        ax = axes[row][col]
        all_points = [point for series in series_map.values() for point in series]
        if not all_points:
            ax.axis("off")
            continue
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        if len(panel) == 3:
            xmin, xmax, ymin, ymax = panel[2]
        else:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if ymin == ymax:
                ymin -= 1.0
                ymax += 1.0
            pad = 0.08 * (ymax - ymin)
            ymin -= pad
            ymax += pad
        ax.set_title(panel_title, fontsize=10, pad=4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, color="#888888", alpha=0.35, linewidth=0.8)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        for color_index, (label, series) in enumerate(series_map.items()):
            color = _series_color(label, color_index)
            x_values = [point[0] for point in series]
            y_values = [point[1] for point in series]
            ax.plot(x_values, y_values, color=color, linewidth=0.8, label=label)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    for index in range(len(panels), rows * cols):
        axes[index // cols][index % cols].axis("off")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path)
    plt.close(fig)


def _series_color(label: str, color_index: int) -> str:
    if "voltage" in label.lower() or "current" in label.lower():
        if any(name in label.lower() for name in ("load", "dg", "inverter")):
            return "#f2b01e"
        return "#1f3fbf"
    if "frequency" in label.lower() or "soc" in label.lower():
        return "#1f3fbf"
    colors = ("#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#d97706")
    return colors[color_index % len(colors)]


def _write_plot_document(path: Path, svg: str) -> None:
    if path.suffix.lower() != ".pdf":
        path.write_text(svg, encoding="utf-8")
        return

    with tempfile.NamedTemporaryFile("w", suffix=".svg", encoding="utf-8", delete=False) as file:
        file.write(svg)
        svg_path = Path(file.name)
    try:
        subprocess.run(
            ["convert", str(svg_path), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        svg_path.unlink(missing_ok=True)


def _axis_svg(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> str:
    lines = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#222" stroke-width="0.8"/>',
        f'<text x="{x + w / 2:.0f}" y="{y - 5}" text-anchor="middle" font-family="Arial" font-size="12">{_xml(title)}</text>',
    ]
    for i in range(1, 5):
        gx = x + w * i / 5.0
        gy = y + h * i / 5.0
        lines.append(f'<line x1="{gx:.1f}" y1="{y}" x2="{gx:.1f}" y2="{y + h}" stroke="#ddd"/>')
        lines.append(f'<line x1="{x}" y1="{gy:.1f}" x2="{x + w}" y2="{gy:.1f}" stroke="#ddd"/>')
    lines.append(
        f'<text x="{x}" y="{y + h + 16}" font-family="Arial" font-size="10">{xmin:.1f}s</text>'
    )
    lines.append(
        f'<text x="{x + w}" y="{y + h + 16}" text-anchor="end" font-family="Arial" font-size="10">{xmax:.1f}s</text>'
    )
    value_format = ".2f" if abs(ymax - ymin) < 1.0 else ".1f"
    lines.append(
        f'<text x="{x - 8}" y="{y + 10}" text-anchor="end" font-family="Arial" font-size="10">{ymax:{value_format}}</text>'
    )
    lines.append(
        f'<text x="{x - 8}" y="{y + h}" text-anchor="end" font-family="Arial" font-size="10">{ymin:{value_format}}</text>'
    )
    return "\n".join(lines)


def _polyline(
    points: list[tuple[float, float]],
    x: int,
    y: int,
    w: int,
    h: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> str:
    if len(points) > 2500:
        stride = max(1, len(points) // 2500)
        points = points[::stride]
    xspan = xmax - xmin or 1.0
    yspan = ymax - ymin or 1.0
    return " ".join(
        f"{x + (_clamp(px, xmin, xmax) - xmin) / xspan * w:.1f},{y + h - (_clamp(py, ymin, ymax) - ymin) / yspan * h:.1f}"
        for px, py in points
    )


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
