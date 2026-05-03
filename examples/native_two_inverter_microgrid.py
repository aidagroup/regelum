from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from regelum import (
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    terminate,
)

PhaseVector = tuple[float, float, float]
VoltageSample = tuple[float, float, float, float]


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "regelum").exists():
            return parent
    raise RuntimeError("Could not locate the regelum repository root.")


def zeros3() -> PhaseVector:
    return (0.0, 0.0, 0.0)


def add3(left: PhaseVector, right: PhaseVector) -> PhaseVector:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def sub3(left: PhaseVector, right: PhaseVector) -> PhaseVector:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def scale3(value: PhaseVector, gain: float) -> PhaseVector:
    return tuple(component * gain for component in value)  # type: ignore[return-value]


def euler3(state: PhaseVector, derivative: PhaseVector, dt: float) -> PhaseVector:
    return add3(state, scale3(derivative, dt))


def dot3(left: PhaseVector, right: PhaseVector) -> float:
    return sum(left[index] * right[index] for index in range(3))


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clip3(value: PhaseVector, lower: float, upper: float) -> PhaseVector:
    return tuple(clip(component, lower, upper) for component in value)  # type: ignore[return-value]


def vector_norm(value: PhaseVector) -> float:
    return math.sqrt(dot3(value, value))


def inst_rms(value: PhaseVector) -> float:
    return vector_norm(value) / math.sqrt(3.0)


def normalise_abc(value: PhaseVector) -> PhaseVector:
    magnitude = inst_rms(value)
    if magnitude == 0:
        return value
    return scale3(value, 1.0 / magnitude)


def abc_to_alpha_beta(abc: PhaseVector) -> tuple[float, float]:
    alpha = (2.0 / 3.0) * (abc[0] - 0.5 * abc[1] - 0.5 * abc[2])
    beta = (2.0 / 3.0) * (0.866 * abc[1] - 0.866 * abc[2])
    return alpha, beta


def cos_sin(theta: float) -> tuple[float, float]:
    return math.cos(theta), math.sin(theta)


def dq0_to_abc_cos_sin(dq0: PhaseVector, cos_value: float, sin_value: float) -> PhaseVector:
    a = cos_value * dq0[0] - sin_value * dq0[1] + dq0[2]
    cos_shift = cos_value * (-0.5) - sin_value * (-0.866)
    sin_shift = sin_value * (-0.5) + cos_value * (-0.866)
    b = cos_shift * dq0[0] - sin_shift * dq0[1] + dq0[2]
    cos_shift = cos_value * (-0.5) - sin_value * 0.866
    sin_shift = sin_value * (-0.5) + cos_value * 0.866
    c = cos_shift * dq0[0] - sin_shift * dq0[1] + dq0[2]
    return a, b, c


def dq0_to_abc(dq0: PhaseVector, theta: float) -> PhaseVector:
    return dq0_to_abc_cos_sin(dq0, *cos_sin(theta))


def abc_to_dq0_cos_sin(abc: PhaseVector, cos_value: float, sin_value: float) -> PhaseVector:
    cos_shift_neg = cos_value * (-0.5) - sin_value * (-0.866)
    sin_shift_neg = sin_value * (-0.5) + cos_value * (-0.866)
    cos_shift_pos = cos_value * (-0.5) - sin_value * 0.866
    sin_shift_pos = sin_value * (-0.5) + cos_value * 0.866
    d = (2.0 / 3.0) * (
        cos_value * abc[0] + cos_shift_neg * abc[1] + cos_shift_pos * abc[2]
    )
    q = (2.0 / 3.0) * (
        -sin_value * abc[0] - sin_shift_neg * abc[1] - sin_shift_pos * abc[2]
    )
    z = sum(abc) / 3.0
    return d, q, z


def abc_to_dq0(abc: PhaseVector, theta: float) -> PhaseVector:
    return abc_to_dq0_cos_sin(abc, *cos_sin(theta))


def inst_power(voltage: PhaseVector, current: PhaseVector) -> float:
    return dot3(voltage, current)


def inst_reactive(voltage: PhaseVector, current: PhaseVector) -> float:
    rolled_left = (voltage[1], voltage[2], voltage[0])
    rolled_right = (voltage[2], voltage[0], voltage[1])
    return -0.5773502691896258 * dot3(sub3(rolled_left, rolled_right), current)


def balanced_sine(time_s: float, amplitude: float, frequency_hz: float, phase_shift: float) -> PhaseVector:
    omega_t = 2.0 * math.pi * frequency_hz * time_s + phase_shift
    return (
        amplitude * math.sin(omega_t),
        amplitude * math.sin(omega_t - 2.0 * math.pi / 3.0),
        amplitude * math.sin(omega_t + 2.0 * math.pi / 3.0),
    )


class PIController:
    def __init__(self, *, kp: float, ki: float, limits: tuple[float, float], kb: float = 1.0, dt: float) -> None:
        self.kp = kp
        self.ki = ki
        self.limits = limits
        self.kb = kb
        self.dt = dt
        self.integral = 0.0
        self.windup_compensation = 0.0

    def step(self, error: float, feedforward: float = 0.0) -> float:
        self.integral += (self.ki * error + self.windup_compensation) * self.dt
        output = self.kp * error + self.integral
        clipped = clip(output + feedforward, *self.limits)
        self.windup_compensation = (output + feedforward - clipped) * self.kb
        return clipped


class MultiPhasePIController:
    def __init__(self, *, kp: float, ki: float, limits: tuple[float, float], dt: float) -> None:
        self.controllers = [
            PIController(kp=kp, ki=ki, limits=limits, dt=dt),
            PIController(kp=kp, ki=ki, limits=limits, dt=dt),
            PIController(kp=kp, ki=ki, limits=limits, dt=dt),
        ]

    def step(self, setpoint: PhaseVector, measured: PhaseVector, feedforward: PhaseVector | None = None) -> PhaseVector:
        if feedforward is None:
            feedforward = zeros3()
        return tuple(
            controller.step(setpoint[index] - measured[index], feedforward[index])
            for index, controller in enumerate(self.controllers)
        )  # type: ignore[return-value]


class PT1Filter:
    def __init__(self, *, gain: float, tau: float, dt: float) -> None:
        self.gain = gain
        self.tau = tau
        self.dt = dt
        self.integral = 0.0

    def step(self, value: float) -> float:
        output = value * self.gain - self.integral
        if self.tau != 0:
            self.integral += output / self.tau * self.dt
            return self.integral
        if self.gain != 0:
            self.integral = 0.0
            return output
        return 0.0


class DroopController:
    def __init__(self, *, gain: float, tau: float, nominal: float, dt: float) -> None:
        inverted_gain = 1.0 / gain if gain != 0 else 0.0
        self.nominal = nominal
        self.filter = PT1Filter(gain=inverted_gain, tau=tau, dt=dt)

    def step(self, value: float) -> float:
        return self.filter.step(value) + self.nominal


class InverseDroopController:
    def __init__(self, *, gain: float, tau: float, nominal: float, tau_filter: float, dt: float) -> None:
        self.gain = gain
        self.tau = tau
        self.nominal = nominal
        self.dt = dt
        self.previous = 0.0
        self.input_filter = PT1Filter(gain=1.0, tau=tau_filter, dt=dt)

    def step(self, value: float) -> float:
        filtered = self.input_filter.step(value - self.nominal)
        derivative = (filtered - self.previous) / self.dt * self.tau
        self.previous = filtered
        if self.gain == 0:
            return 0.0
        return filtered * self.gain + derivative


class DDS:
    def __init__(self, *, dt: float, theta_0: float = 0.0) -> None:
        self.dt = dt
        self.integral = theta_0

    def step(self, frequency_hz: float) -> float:
        self.integral += self.dt * frequency_hz
        if self.integral > 1.0:
            self.integral -= 1.0
        return self.integral * 2.0 * math.pi


class PLL:
    def __init__(self, *, kp: float, ki: float, f_nom: float, dt: float, theta_0: float = 0.0) -> None:
        self.pi = PIController(kp=kp, ki=ki, limits=(-math.inf, math.inf), dt=dt)
        self.dds = DDS(dt=dt, theta_0=theta_0)
        self.previous_cos_sin = cos_sin(theta_0)
        self.f_nom = f_nom

    def step(self, voltage: PhaseVector) -> tuple[tuple[float, float], float, float]:
        normalised = normalise_abc(voltage)
        alpha_beta = abc_to_alpha_beta(normalised)
        dphi = alpha_beta[1] * self.previous_cos_sin[0] - alpha_beta[0] * self.previous_cos_sin[1]
        frequency = self.pi.step(dphi) + self.f_nom
        theta = self.dds.step(frequency)
        self.previous_cos_sin = cos_sin(theta)
        return self.previous_cos_sin, frequency, theta


class SimulationClock(Node):
    def __init__(self, *, dt: float = 0.5e-4) -> None:
        self.dt = dt

    class Inputs(NodeInputs):
        tick: int = Input(source="SimulationClock.Outputs.tick")
        time_s: float = Input(source="SimulationClock.Outputs.time_s")

    class Outputs(NodeOutputs):
        tick: int = Output(initial=0)
        time_s: float = Output(initial=0.0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(tick=inputs.tick + 1, time_s=inputs.time_s + self.dt)


class NativeDroopController(Node):
    def __init__(
        self,
        *,
        dt: float = 0.5e-4,
        v_nom: float = 230.0 * math.sqrt(2.0),
        freq_nom: float = 50.0,
        v_dc: float = 1000.0,
        i_lim: float = 30.0,
    ) -> None:
        self.v_dc = v_dc
        self.i_lim = i_lim
        self.master_voltage_pi = MultiPhasePIController(kp=0.025, ki=60.0, limits=(-i_lim, i_lim), dt=dt)
        self.master_current_pi = MultiPhasePIController(kp=0.012, ki=90.0, limits=(-1.0, 1.0), dt=dt)
        self.master_p_droop = DroopController(gain=40000.0, tau=0.005, nominal=freq_nom, dt=dt)
        self.master_q_droop = DroopController(gain=1000.0, tau=0.002, nominal=v_nom, dt=dt)
        self.master_phase = DDS(dt=dt)

        self.slave_current_pi = MultiPhasePIController(kp=0.005, ki=200.0, limits=(-1.0, 1.0), dt=dt)
        self.slave_pll = PLL(kp=10.0, ki=200.0, f_nom=freq_nom, dt=dt)
        self.slave_p_droop = InverseDroopController(
            gain=40000.0,
            tau=dt,
            nominal=freq_nom,
            tau_filter=0.04,
            dt=dt,
        )
        self.slave_q_droop = InverseDroopController(
            gain=50.0,
            tau=dt,
            nominal=v_nom,
            tau_filter=0.01,
            dt=dt,
        )

    class Inputs(NodeInputs):
        master_current: PhaseVector = Input(source="MasterLcFilter.Outputs.inductor_i")
        master_voltage: PhaseVector = Input(source="MicrogridBus.Outputs.bus_v")
        slave_current: PhaseVector = Input(source="SlaveLclFilter.Outputs.inverter_side_i")
        slave_voltage: PhaseVector = Input(source="SlaveLclFilter.Outputs.capacitor_v")

    class Outputs(NodeOutputs):
        inverter1_v: PhaseVector = Output(initial=zeros3)
        inverter2_v: PhaseVector = Output(initial=zeros3)
        inverter1_modulation: PhaseVector = Output(initial=zeros3)
        inverter2_modulation: PhaseVector = Output(initial=zeros3)
        master_frequency_hz: float = Output(initial=50.0)
        slave_frequency_hz: float = Output(initial=50.0)

    def run(self, inputs: Inputs) -> Outputs:
        master_m, master_frequency = self._master_control(
            current=inputs.master_current,
            voltage=inputs.master_voltage,
        )
        slave_m, slave_frequency = self._slave_control(
            current=inputs.slave_current,
            voltage=inputs.slave_voltage,
        )
        return self.Outputs(
            inverter1_v=scale3(master_m, self.v_dc),
            inverter2_v=scale3(slave_m, self.v_dc),
            inverter1_modulation=master_m,
            inverter2_modulation=slave_m,
            master_frequency_hz=master_frequency,
            slave_frequency_hz=slave_frequency,
        )

    def _master_control(self, *, current: PhaseVector, voltage: PhaseVector) -> tuple[PhaseVector, float]:
        instant_power = -inst_power(voltage, current)
        frequency = self.master_p_droop.step(instant_power)
        phase = self.master_phase.step(frequency)
        instant_q = -inst_reactive(voltage, current)
        voltage_setpoint = self.master_q_droop.step(instant_q)

        current_dq0 = abc_to_dq0(current, phase)
        voltage_dq0 = abc_to_dq0(voltage, phase)
        voltage_setpoint_dq0 = (voltage_setpoint, 0.0, 0.0)
        current_setpoint_dq0 = self.master_voltage_pi.step(voltage_setpoint_dq0, voltage_dq0)
        modulation_dq0 = self.master_current_pi.step(current_setpoint_dq0, current_dq0)
        return clip3(dq0_to_abc(modulation_dq0, phase), -1.0, 1.0), frequency

    def _slave_control(self, *, current: PhaseVector, voltage: PhaseVector) -> tuple[PhaseVector, float]:
        v_inst = inst_rms(voltage)
        cos_sin_pair, frequency, _theta = self.slave_pll.step(voltage)
        current_dq0 = abc_to_dq0_cos_sin(current, *cos_sin_pair)

        droop = (0.0, 0.0)
        if v_inst > 150.0:
            active_current = self.slave_p_droop.step(frequency) / v_inst
            reactive_current = self.slave_q_droop.step(v_inst) / v_inst
            droop = (
                clip(active_current / 3.0 * math.sqrt(2.0), -self.i_lim, self.i_lim),
                clip(reactive_current / 3.0 * math.sqrt(2.0), -self.i_lim, self.i_lim),
            )

        current_setpoint_dq0 = (-droop[0], droop[1], 0.0)
        modulation_dq0 = self.slave_current_pi.step(current_setpoint_dq0, current_dq0)
        return clip3(dq0_to_abc_cos_sin(modulation_dq0, *cos_sin_pair), -1.0, 1.0), frequency


class MasterLcFilter(Node):
    def __init__(self, *, dt: float = 0.5e-4, inductance: float = 0.001, resistance: float = 0.08) -> None:
        self.dt = dt
        self.inductance = inductance
        self.resistance = resistance

    class Inputs(NodeInputs):
        inverter_v: PhaseVector = Input(source=NativeDroopController.Outputs.inverter1_v)
        bus_v: PhaseVector = Input(source="MicrogridBus.Outputs.bus_v")
        inductor_i: PhaseVector = Input(source="MasterLcFilter.Outputs.inductor_i")

    class Outputs(NodeOutputs):
        inductor_i: PhaseVector = Output(initial=zeros3)
        current_to_bus: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        voltage = sub3(sub3(inputs.inverter_v, inputs.bus_v), scale3(inputs.inductor_i, self.resistance))
        di = scale3(voltage, 1.0 / self.inductance)
        next_i = euler3(inputs.inductor_i, di, self.dt)
        return self.Outputs(inductor_i=next_i, current_to_bus=next_i)


class SlaveLclFilter(Node):
    def __init__(
        self,
        *,
        dt: float = 0.5e-4,
        left_inductance: float = 0.001,
        right_inductance: float = 0.001,
        capacitance: float = 1.0e-5,
        resistance: float = 0.08,
    ) -> None:
        self.dt = dt
        self.left_inductance = left_inductance
        self.right_inductance = right_inductance
        self.capacitance = capacitance
        self.resistance = resistance

    class Inputs(NodeInputs):
        inverter_v: PhaseVector = Input(source=NativeDroopController.Outputs.inverter2_v)
        bus_v: PhaseVector = Input(source="MicrogridBus.Outputs.bus_v")
        inverter_side_i: PhaseVector = Input(source="SlaveLclFilter.Outputs.inverter_side_i")
        capacitor_v: PhaseVector = Input(source="SlaveLclFilter.Outputs.capacitor_v")
        grid_side_i: PhaseVector = Input(source="SlaveLclFilter.Outputs.grid_side_i")

    class Outputs(NodeOutputs):
        inverter_side_i: PhaseVector = Output(initial=zeros3)
        capacitor_v: PhaseVector = Output(initial=zeros3)
        grid_side_i: PhaseVector = Output(initial=zeros3)
        current_to_bus: PhaseVector = Output(initial=zeros3)
        lcl1_capacitor1_v: float = Output(initial=0.0)
        lcl1_capacitor2_v: float = Output(initial=0.0)
        lcl1_capacitor3_v: float = Output(initial=0.0)

    def run(self, inputs: Inputs) -> Outputs:
        left_voltage = sub3(
            sub3(inputs.inverter_v, inputs.capacitor_v),
            scale3(inputs.inverter_side_i, self.resistance),
        )
        right_voltage = sub3(
            sub3(inputs.capacitor_v, inputs.bus_v),
            scale3(inputs.grid_side_i, self.resistance),
        )
        next_left_i = euler3(
            inputs.inverter_side_i,
            scale3(left_voltage, 1.0 / self.left_inductance),
            self.dt,
        )
        next_right_i = euler3(
            inputs.grid_side_i,
            scale3(right_voltage, 1.0 / self.right_inductance),
            self.dt,
        )
        cap_current = sub3(next_left_i, next_right_i)
        next_cap_v = euler3(inputs.capacitor_v, scale3(cap_current, 1.0 / self.capacitance), self.dt)
        return self.Outputs(
            inverter_side_i=next_left_i,
            capacitor_v=next_cap_v,
            grid_side_i=next_right_i,
            current_to_bus=next_right_i,
            lcl1_capacitor1_v=next_cap_v[0],
            lcl1_capacitor2_v=next_cap_v[1],
            lcl1_capacitor3_v=next_cap_v[2],
        )


class LoadSideLcAndRlLoad(Node):
    def __init__(
        self,
        *,
        dt: float = 0.5e-4,
        filter_inductance: float = 0.001,
        filter_capacitance: float = 1.0e-5,
        load_inductance: float = 0.001,
        filter_resistance: float = 0.08,
        base_resistance: float = 20.0,
    ) -> None:
        self.dt = dt
        self.filter_inductance = filter_inductance
        self.filter_capacitance = filter_capacitance
        self.load_inductance = load_inductance
        self.filter_resistance = filter_resistance
        self.base_resistance = base_resistance

    class Inputs(NodeInputs):
        time_s: float = Input(source=SimulationClock.Outputs.time_s)
        bus_v: PhaseVector = Input(source="MicrogridBus.Outputs.bus_v")
        filter_i: PhaseVector = Input(source="LoadSideLcAndRlLoad.Outputs.filter_i")
        load_bus_v: PhaseVector = Input(source="LoadSideLcAndRlLoad.Outputs.load_bus_v")
        load_i: PhaseVector = Input(source="LoadSideLcAndRlLoad.Outputs.load_i")

    class Outputs(NodeOutputs):
        filter_i: PhaseVector = Output(initial=zeros3)
        load_bus_v: PhaseVector = Output(initial=zeros3)
        load_i: PhaseVector = Output(initial=zeros3)
        current_from_bus: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        resistance = self.base_resistance if inputs.time_s < 0.2 else 2.0 * self.base_resistance
        filter_voltage = sub3(
            sub3(inputs.bus_v, inputs.load_bus_v),
            scale3(inputs.filter_i, self.filter_resistance),
        )
        load_voltage = sub3(inputs.load_bus_v, scale3(inputs.load_i, resistance))
        next_filter_i = euler3(
            inputs.filter_i,
            scale3(filter_voltage, 1.0 / self.filter_inductance),
            self.dt,
        )
        next_load_i = euler3(inputs.load_i, scale3(load_voltage, 1.0 / self.load_inductance), self.dt)
        cap_current = sub3(next_filter_i, next_load_i)
        next_load_bus_v = euler3(
            inputs.load_bus_v,
            scale3(cap_current, 1.0 / self.filter_capacitance),
            self.dt,
        )
        return self.Outputs(
            filter_i=next_filter_i,
            load_bus_v=next_load_bus_v,
            load_i=next_load_i,
            current_from_bus=next_filter_i,
        )


class MicrogridBus(Node):
    def __init__(self, *, dt: float = 0.5e-4, bus_capacitance: float = 1.0e-5) -> None:
        self.dt = dt
        self.bus_capacitance = bus_capacitance

    class Inputs(NodeInputs):
        bus_v: PhaseVector = Input(source="MicrogridBus.Outputs.bus_v")
        master_i: PhaseVector = Input(source=MasterLcFilter.Outputs.current_to_bus)
        slave_i: PhaseVector = Input(source=SlaveLclFilter.Outputs.current_to_bus)
        loadside_i: PhaseVector = Input(source=LoadSideLcAndRlLoad.Outputs.current_from_bus)

    class Outputs(NodeOutputs):
        bus_v: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        bus_current = sub3(add3(inputs.master_i, inputs.slave_i), inputs.loadside_i)
        bus_v = euler3(inputs.bus_v, scale3(bus_current, 1.0 / self.bus_capacitance), self.dt)
        return self.Outputs(bus_v=bus_v)


class NativeMicrogridLogger(Node):
    class Inputs(NodeInputs):
        time_s: float = Input(source=SimulationClock.Outputs.time_s)
        lcl1_capacitor1_v: float = Input(source=SlaveLclFilter.Outputs.lcl1_capacitor1_v)
        lcl1_capacitor2_v: float = Input(source=SlaveLclFilter.Outputs.lcl1_capacitor2_v)
        lcl1_capacitor3_v: float = Input(source=SlaveLclFilter.Outputs.lcl1_capacitor3_v)
        samples: tuple[VoltageSample, ...] = Input(source="NativeMicrogridLogger.Outputs.samples")

    class Outputs(NodeOutputs):
        samples: tuple[VoltageSample, ...] = Output(initial=())

    def run(self, inputs: Inputs) -> Outputs:
        sample = (
            inputs.time_s,
            inputs.lcl1_capacitor1_v,
            inputs.lcl1_capacitor2_v,
            inputs.lcl1_capacitor3_v,
        )
        return self.Outputs(samples=inputs.samples + (sample,))


def build_system() -> PhasedReactiveSystem:
    clock = SimulationClock()
    controller = NativeDroopController()
    master_lc = MasterLcFilter()
    slave_lcl = SlaveLclFilter()
    loadside = LoadSideLcAndRlLoad()
    bus = MicrogridBus()
    logger = NativeMicrogridLogger()

    return PhasedReactiveSystem(
        phases=[
            Phase("control", nodes=(controller,), transitions=(Goto("branches"),), is_initial=True),
            Phase("branches", nodes=(master_lc, slave_lcl, loadside), transitions=(Goto("bus"),)),
            Phase("bus", nodes=(bus,), transitions=(Goto("clock"),)),
            Phase("clock", nodes=(clock,), transitions=(Goto("log"),)),
            Phase("log", nodes=(logger,), transitions=(Goto(terminate),)),
        ],
    )


def save_lcl1_plot(samples: tuple[VoltageSample, ...], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    time = [row[0] for row in samples]
    v1 = [row[1] for row in samples]
    v2 = [row[2] for row in samples]
    v3 = [row[3] for row in samples]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    ax.plot(time, v1, label="lcl1.capacitor1.v")
    ax.plot(time, v2, label="lcl1.capacitor2.v")
    ax.plot(time, v3, label="lcl1.capacitor3.v")
    ax.set_xlim(0, 0.05)
    ax.legend()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root()
        / "artifacts"
        / "openmodelica_microgrid_gym"
        / "native_droop_split_lcl1_capacitor_voltages.png",
    )
    args = parser.parse_args()

    system = build_system()
    system.run(args.steps)
    snapshot = system.snapshot()
    samples = snapshot["NativeMicrogridLogger.samples"]
    save_lcl1_plot(samples, args.output)
    print(f"steps={args.steps}")
    print(f"samples={len(samples)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
