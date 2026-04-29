from __future__ import annotations

from enum import Enum
from typing import cast

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
    def __init__(self, *, dt: float = 0.1, init_time: float = 2.0) -> None:
        self.dt = dt
        self.init_time = init_time

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
            wind_power_available_kw=_wind_power_profile_kw(next_time),
            load_power_kw=_load_profile_kw(next_time),
        )


class WindTurbinePMSG(Node):
    class Outputs(NodeOutputs):
        generated_power_kw: float = Output(initial=0.0)

    def run(
        self,
        wind_power_available_kw: float = Input(
            source=ScenarioProfile.Outputs.wind_power_available_kw
        ),
        mppt_efficiency: float = Input(source=lambda: MpptController.Outputs.efficiency),
    ) -> Outputs:
        generated_power = wind_power_available_kw * mppt_efficiency
        return self.Outputs(generated_power_kw=generated_power)


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
        dump_load_enabled = soc_percent >= 99.8 and wind_power_kw > load_power_kw
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
    rated_power_kw = 40.0

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
        max_charge = 45.0 if soc_percent < 100.0 else 0.0
        max_discharge = 45.0 if soc_percent > 0.0 else 0.0
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
        current = 1000.0 * battery_power_kw / self.nominal_voltage_v
        return self.Outputs(soc_percent=soc_next, current_a=current)


class DcBus(Node):
    nominal_voltage_v = 288.0

    class Outputs(NodeOutputs):
        voltage_v: float = Output(initial=288.0)

    def run(
        self,
        unserved_power_kw: float = Input(source=PowerBalance.Outputs.unserved_power_kw),
        voltage_v: float = Input(source=lambda: DcBus.Outputs.voltage_v),
    ) -> Outputs:
        target = self.nominal_voltage_v + 1.8 * unserved_power_kw
        target = _clamp(target, 245.0, 315.0)
        voltage_next = voltage_v + 0.55 * (target - voltage_v)
        return self.Outputs(voltage_v=voltage_next)


class PccRegulator(Node):
    class Outputs(NodeOutputs):
        voltage_v: float = Output(initial=460.0)
        frequency_hz: float = Output(initial=60.0)

    def run(
        self,
        dc_bus_voltage_v: float = Input(source=DcBus.Outputs.voltage_v),
        unserved_power_kw: float = Input(source=PowerBalance.Outputs.unserved_power_kw),
        voltage_v: float = Input(source=lambda: PccRegulator.Outputs.voltage_v),
        frequency_hz: float = Input(source=lambda: PccRegulator.Outputs.frequency_hz),
    ) -> Outputs:
        voltage_target = 460.0 + 0.25 * (dc_bus_voltage_v - DcBus.nominal_voltage_v)
        frequency_target = 60.0 + 0.01 * unserved_power_kw
        voltage_next = voltage_v + 0.60 * (voltage_target - voltage_v)
        frequency_next = frequency_hz + 0.55 * (frequency_target - frequency_hz)
        return self.Outputs(voltage_v=voltage_next, frequency_hz=frequency_next)


class MicrogridLogger(Node):
    class Outputs(NodeOutputs):
        samples: list[dict[str, float | str | bool]] = Output(initial=list)

    def run(
        self,
        time: float = Input(source=ScenarioProfile.Outputs.time),
        mode: MicrogridMode = Input(source=PowerFlowSupervisor.Outputs.mode),
        wind_power_kw: float = Input(source=WindTurbinePMSG.Outputs.generated_power_kw),
        diesel_power_kw: float = Input(source=DieselGenerator.Outputs.generated_power_kw),
        load_power_kw: float = Input(source=WaterTreatmentLoad.Outputs.load_power_kw),
        battery_power_kw: float = Input(source=PowerBalance.Outputs.battery_power_kw),
        dump_load_power_kw: float = Input(source=PowerBalance.Outputs.dump_load_power_kw),
        soc_percent: float = Input(source=Battery.Outputs.soc_percent),
        dc_bus_voltage_v: float = Input(source=DcBus.Outputs.voltage_v),
        pcc_voltage_v: float = Input(source=PccRegulator.Outputs.voltage_v),
        frequency_hz: float = Input(source=PccRegulator.Outputs.frequency_hz),
        diesel_enabled: bool = Input(source=PowerFlowSupervisor.Outputs.diesel_enabled),
        samples: list[dict[str, float | str | bool]] = Input(
            source=lambda: MicrogridLogger.Outputs.samples
        ),
    ) -> Outputs:
        sample: dict[str, float | str | bool] = {
            "time": time,
            "mode": mode.value,
            "wind_power_kw": wind_power_kw,
            "diesel_power_kw": diesel_power_kw,
            "load_power_kw": load_power_kw,
            "battery_power_kw": battery_power_kw,
            "dump_load_power_kw": dump_load_power_kw,
            "soc_percent": soc_percent,
            "dc_bus_voltage_v": dc_bus_voltage_v,
            "pcc_voltage_v": pcc_voltage_v,
            "frequency_hz": frequency_hz,
            "diesel_enabled": diesel_enabled,
        }
        samples.append(sample)
        return self.Outputs(samples=samples)


def build_system(*, dt: float = 0.1) -> PhasedReactiveSystem:
    source_nodes = (
        ScenarioProfile(dt=dt),
        MpptController(),
        WindTurbinePMSG(),
        WaterTreatmentLoad(),
        DieselGenerator(),
    )
    balance_nodes = (PowerBalance(),)
    storage_nodes = (Battery(dt=dt), DcBus())
    regulation_nodes = (PccRegulator(), MicrogridLogger())
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


def _wind_power_profile_kw(time_s: float) -> float:
    if time_s < 7.0:
        return 0.0
    if time_s < 8.0:
        return 48.0 * (time_s - 7.0)
    if time_s < 11.0:
        return 48.0
    if time_s < 15.0:
        return 48.0 - 7.0 * (time_s - 11.0)
    if time_s < 16.0:
        return 20.0
    if time_s < 18.0:
        return 20.0 + 13.0 * (time_s - 16.0)
    return 46.0


def _load_profile_kw(time_s: float) -> float:
    if time_s < 7.0:
        return 28.0
    if time_s < 9.0:
        return 10.0
    if time_s < 18.0:
        return 34.0
    return 16.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
