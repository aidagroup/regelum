from __future__ import annotations

import argparse
import math
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt

import regelum as rg

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
    d = (2.0 / 3.0) * (cos_value * abc[0] + cos_shift_neg * abc[1] + cos_shift_pos * abc[2])
    q = (2.0 / 3.0) * (-sin_value * abc[0] - sin_shift_neg * abc[1] - sin_shift_pos * abc[2])
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


def pi_update(
    *,
    setpoint: PhaseVector,
    measured: PhaseVector,
    integral: PhaseVector,
    windup: PhaseVector,
    kp: float,
    ki: float,
    limits: tuple[float, float],
    dt: float,
    feedforward: PhaseVector | None = None,
    kb: float = 1.0,
) -> tuple[PhaseVector, PhaseVector, PhaseVector]:
    if feedforward is None:
        feedforward = zeros3()
    next_integral = tuple(
        integral[index] + (ki * (setpoint[index] - measured[index]) + windup[index]) * dt
        for index in range(3)
    )
    raw = tuple(
        kp * (setpoint[index] - measured[index]) + next_integral[index] + feedforward[index]
        for index in range(3)
    )
    output = clip3(raw, *limits)
    next_windup = tuple((raw[index] - output[index]) * kb for index in range(3))
    return output, next_integral, next_windup  # type: ignore[return-value]


def pt1_update(*, value: float, integral: float, gain: float, tau: float, dt: float) -> float:
    output = value * gain - integral
    if tau != 0:
        return integral + output / tau * dt
    if gain != 0:
        return 0.0
    return 0.0


def advance_phase_turns(phase_turns: float, frequency_hz: float, dt: float) -> float:
    next_phase = phase_turns + dt * frequency_hz
    if next_phase > 1.0:
        next_phase -= 1.0
    return next_phase


def save_lcl1_plot(samples: list[VoltageSample], output: Path) -> None:
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


class MasterDroop(rg.Node):
    def __init__(
        self,
        *,
        dt: float = 0.5e-4,
        v_nom: float = 230.0 * math.sqrt(2.0),
        freq_nom: float = 50.0,
    ) -> None:
        self.dt = dt
        self.v_nom = v_nom
        self.freq_nom = freq_nom

    class Inputs(rg.NodeInputs):
        current: PhaseVector = rg.Input(src=lambda: Lc1Filter.State.inductor_i)
        voltage: PhaseVector = rg.Input(src=lambda: Lc1Filter.State.capacitor_v)

    class State(rg.NodeState):
        frequency_hz: float = rg.Var(init=50.0)
        voltage_setpoint: float = rg.Var(init=230.0 * math.sqrt(2.0))
        phase: float = rg.Var(init=0.0)
        phase_turns: float = rg.Var(init=0.0)
        p_filter: float = rg.Var(init=0.0)
        q_filter: float = rg.Var(init=0.0)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        instant_power = -inst_power(inputs.voltage, inputs.current)
        p_filter = pt1_update(
            value=instant_power,
            integral=prev_state.p_filter,
            gain=1.0 / 40000.0,
            tau=0.005,
            dt=self.dt,
        )
        frequency_hz = p_filter + self.freq_nom

        instant_q = -inst_reactive(inputs.voltage, inputs.current)
        q_filter = pt1_update(
            value=instant_q,
            integral=prev_state.q_filter,
            gain=1.0 / 1000.0,
            tau=0.002,
            dt=self.dt,
        )
        voltage_setpoint = q_filter + self.v_nom
        phase_turns = advance_phase_turns(prev_state.phase_turns, frequency_hz, self.dt)
        return self.State(
            frequency_hz=frequency_hz,
            voltage_setpoint=voltage_setpoint,
            phase=phase_turns * 2.0 * math.pi,
            phase_turns=phase_turns,
            p_filter=p_filter,
            q_filter=q_filter,
        )


class MasterVoltagePI(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4, i_lim: float = 30.0) -> None:
        self.dt = dt
        self.i_lim = i_lim

    class Inputs(rg.NodeInputs):
        current: PhaseVector = rg.Input(src=lambda: Lc1Filter.State.inductor_i)
        voltage: PhaseVector = rg.Input(src=lambda: Lc1Filter.State.capacitor_v)
        phase: float = rg.Input(src=MasterDroop.State.phase)
        voltage_setpoint: float = rg.Input(src=MasterDroop.State.voltage_setpoint)

    class State(rg.NodeState):
        current_setpoint_dq0: PhaseVector = rg.Var(init=zeros3)
        integral: PhaseVector = rg.Var(init=zeros3)
        windup: PhaseVector = rg.Var(init=zeros3)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        voltage_dq0 = abc_to_dq0(inputs.voltage, inputs.phase)
        voltage_setpoint_dq0 = (inputs.voltage_setpoint, 0.0, 0.0)
        current_setpoint_dq0, integral, windup = pi_update(
            setpoint=voltage_setpoint_dq0,
            measured=voltage_dq0,
            integral=prev_state.integral,
            windup=prev_state.windup,
            kp=0.025,
            ki=60.0,
            limits=(-self.i_lim, self.i_lim),
            dt=self.dt,
        )
        return self.State(
            current_setpoint_dq0=current_setpoint_dq0,
            integral=integral,
            windup=windup,
        )


class MasterCurrentPI(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4) -> None:
        self.dt = dt

    class Inputs(rg.NodeInputs):
        current: PhaseVector = rg.Input(src=lambda: Lc1Filter.State.inductor_i)
        phase: float = rg.Input(src=MasterDroop.State.phase)
        current_setpoint_dq0: PhaseVector = rg.Input(src=MasterVoltagePI.State.current_setpoint_dq0)

    class State(rg.NodeState):
        modulation: PhaseVector = rg.Var(init=zeros3)
        integral: PhaseVector = rg.Var(init=zeros3)
        windup: PhaseVector = rg.Var(init=zeros3)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        current_dq0 = abc_to_dq0(inputs.current, inputs.phase)
        modulation_dq0, integral, windup = pi_update(
            setpoint=inputs.current_setpoint_dq0,
            measured=current_dq0,
            integral=prev_state.integral,
            windup=prev_state.windup,
            kp=0.012,
            ki=90.0,
            limits=(-1.0, 1.0),
            dt=self.dt,
        )
        modulation = clip3(dq0_to_abc(modulation_dq0, inputs.phase), -1.0, 1.0)
        return self.State(modulation=modulation, integral=integral, windup=windup)


class SlavePLL(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4, f_nom: float = 50.0) -> None:
        self.dt = dt
        self.f_nom = f_nom

    class Inputs(rg.NodeInputs):
        voltage: PhaseVector = rg.Input(src=lambda: Lcl1Filter.State.capacitor_v)

    class State(rg.NodeState):
        cos_value: float = rg.Var(init=1.0)
        sin_value: float = rg.Var(init=0.0)
        frequency_hz: float = rg.Var(init=50.0)
        theta: float = rg.Var(init=0.0)
        theta_turns: float = rg.Var(init=0.0)
        integral: float = rg.Var(init=0.0)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        normalised = normalise_abc(inputs.voltage)
        alpha_beta = abc_to_alpha_beta(normalised)
        dphi = alpha_beta[1] * prev_state.cos_value - alpha_beta[0] * prev_state.sin_value
        integral = prev_state.integral + 200.0 * dphi * self.dt
        frequency_hz = 10.0 * dphi + integral + self.f_nom
        theta_turns = advance_phase_turns(prev_state.theta_turns, frequency_hz, self.dt)
        theta = theta_turns * 2.0 * math.pi
        cos_value, sin_value = cos_sin(theta)
        return self.State(
            cos_value=cos_value,
            sin_value=sin_value,
            frequency_hz=frequency_hz,
            theta=theta,
            theta_turns=theta_turns,
            integral=integral,
        )


class SlaveInverseDroop(rg.Node):
    def __init__(
        self,
        *,
        dt: float = 0.5e-4,
        v_nom: float = 230.0 * math.sqrt(2.0),
        i_lim: float = 30.0,
    ) -> None:
        self.dt = dt
        self.v_nom = v_nom
        self.i_lim = i_lim

    class Inputs(rg.NodeInputs):
        voltage: PhaseVector = rg.Input(src=lambda: Lcl1Filter.State.capacitor_v)
        frequency_hz: float = rg.Input(src=SlavePLL.State.frequency_hz)

    class State(rg.NodeState):
        current_setpoint_dq0: PhaseVector = rg.Var(init=zeros3)
        p_filter: float = rg.Var(init=0.0)
        p_previous: float = rg.Var(init=0.0)
        q_filter: float = rg.Var(init=0.0)
        q_previous: float = rg.Var(init=0.0)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        v_inst = inst_rms(inputs.voltage)
        if v_inst <= 150.0:
            return self.State(
                current_setpoint_dq0=zeros3(),
                p_filter=prev_state.p_filter,
                p_previous=prev_state.p_previous,
                q_filter=prev_state.q_filter,
                q_previous=prev_state.q_previous,
            )

        p_filter = pt1_update(
            value=inputs.frequency_hz - 50.0,
            integral=prev_state.p_filter,
            gain=1.0,
            tau=0.04,
            dt=self.dt,
        )
        p_output = p_filter * 40000.0 + (p_filter - prev_state.p_previous)

        q_filter = pt1_update(
            value=v_inst - self.v_nom,
            integral=prev_state.q_filter,
            gain=1.0,
            tau=0.01,
            dt=self.dt,
        )
        q_output = q_filter * 50.0 + (q_filter - prev_state.q_previous)

        active_current = p_output / v_inst
        reactive_current = q_output / v_inst
        droop = (
            clip(active_current / 3.0 * math.sqrt(2.0), -self.i_lim, self.i_lim),
            clip(reactive_current / 3.0 * math.sqrt(2.0), -self.i_lim, self.i_lim),
        )
        return self.State(
            current_setpoint_dq0=(-droop[0], droop[1], 0.0),
            p_filter=p_filter,
            p_previous=p_filter,
            q_filter=q_filter,
            q_previous=q_filter,
        )


class SlaveCurrentPI(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4) -> None:
        self.dt = dt

    class Inputs(rg.NodeInputs):
        current: PhaseVector = rg.Input(src=lambda: Lcl1Filter.State.inverter_side_i)
        cos_value: float = rg.Input(src=SlavePLL.State.cos_value)
        sin_value: float = rg.Input(src=SlavePLL.State.sin_value)
        current_setpoint_dq0: PhaseVector = rg.Input(src=SlaveInverseDroop.State.current_setpoint_dq0)

    class State(rg.NodeState):
        modulation: PhaseVector = rg.Var(init=zeros3)
        integral: PhaseVector = rg.Var(init=zeros3)
        windup: PhaseVector = rg.Var(init=zeros3)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        current_dq0 = abc_to_dq0_cos_sin(inputs.current, inputs.cos_value, inputs.sin_value)
        modulation_dq0, integral, windup = pi_update(
            setpoint=inputs.current_setpoint_dq0,
            measured=current_dq0,
            integral=prev_state.integral,
            windup=prev_state.windup,
            kp=0.005,
            ki=200.0,
            limits=(-1.0, 1.0),
            dt=self.dt,
        )
        modulation = clip3(
            dq0_to_abc_cos_sin(modulation_dq0, inputs.cos_value, inputs.sin_value),
            -1.0,
            1.0,
        )
        return self.State(modulation=modulation, integral=integral, windup=windup)


class Inverter1(rg.Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(rg.NodeInputs):
        modulation: PhaseVector = rg.Input(src=MasterCurrentPI.State.modulation)

    class State(rg.NodeState):
        phase_v: PhaseVector = rg.Var(init=zeros3)

    def update(self, inputs: Inputs) -> State:
        return self.State(phase_v=scale3(inputs.modulation, self.gain))


class Inverter2(rg.Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(rg.NodeInputs):
        modulation: PhaseVector = rg.Input(src=SlaveCurrentPI.State.modulation)

    class State(rg.NodeState):
        phase_v: PhaseVector = rg.Var(init=zeros3)

    def update(self, inputs: Inputs) -> State:
        return self.State(phase_v=scale3(inputs.modulation, self.gain))


class Lc1Filter(rg.ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class Inputs(rg.NodeInputs):
        inverter_v: PhaseVector = rg.Input(src=Inverter1.State.phase_v)
        lcl1_grid_side_i: PhaseVector = rg.Input(src=lambda: Lcl1Filter.State.grid_side_i)
        lc2_inductor_i: PhaseVector = rg.Input(src=lambda: Lc2Filter.State.inductor_i)

    class State(rg.NodeState):
        capacitor_v: PhaseVector = rg.Var(init=zeros3)
        inductor_i: PhaseVector = rg.Var(init=zeros3)

    def dstate(self, inputs: Inputs, state: State) -> State:
        return self.State(
            capacitor_v=scale3(
                sub3(add3(state.inductor_i, inputs.lcl1_grid_side_i), inputs.lc2_inductor_i),
                1.0 / self.capacitance,
            ),
            inductor_i=scale3(
                sub3(inputs.inverter_v, state.capacitor_v),
                1.0 / self.inductance,
            ),
        )


class Lcl1Filter(rg.ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class Inputs(rg.NodeInputs):
        inverter_v: PhaseVector = rg.Input(src=Inverter2.State.phase_v)
        bus_v: PhaseVector = rg.Input(src=Lc1Filter.State.capacitor_v)

    class State(rg.NodeState):
        capacitor_v: PhaseVector = rg.Var(init=zeros3)
        inverter_side_i: PhaseVector = rg.Var(init=zeros3)
        grid_side_i: PhaseVector = rg.Var(init=zeros3)

    def dstate(self, inputs: Inputs, state: State) -> State:
        return self.State(
            capacitor_v=scale3(
                sub3(state.inverter_side_i, state.grid_side_i),
                1.0 / self.capacitance,
            ),
            inverter_side_i=scale3(
                sub3(inputs.inverter_v, state.capacitor_v),
                1.0 / self.inductance,
            ),
            grid_side_i=scale3(
                sub3(state.capacitor_v, inputs.bus_v),
                1.0 / self.inductance,
            ),
        )


class Lc2Filter(rg.ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class Inputs(rg.NodeInputs):
        bus_v: PhaseVector = rg.Input(src=Lc1Filter.State.capacitor_v)
        load_i: PhaseVector = rg.Input(src=lambda: Rl1Load.State.load_i)

    class State(rg.NodeState):
        capacitor_v: PhaseVector = rg.Var(init=zeros3)
        inductor_i: PhaseVector = rg.Var(init=zeros3)

    def dstate(self, inputs: Inputs, state: State) -> State:
        return self.State(
            capacitor_v=scale3(
                sub3(state.inductor_i, inputs.load_i),
                1.0 / self.capacitance,
            ),
            inductor_i=scale3(
                sub3(inputs.bus_v, state.capacitor_v),
                1.0 / self.inductance,
            ),
        )


class Rl1Load(rg.ODENode):
    def __init__(
        self,
        *,
        resistance: float = 20.0,
        inductance: float = 0.001,
    ) -> None:
        self.resistance = resistance
        self.inductance = inductance

    class Inputs(rg.NodeInputs):
        capacitor_v: PhaseVector = rg.Input(src=Lc2Filter.State.capacitor_v)

    class State(rg.NodeState):
        load_i: PhaseVector = rg.Var(init=zeros3)

    def dstate(self, inputs: Inputs, state: State, *, time: float) -> State:
        resistance = ca.if_else(time < 0.2, self.resistance, 2.0 * self.resistance)
        return self.State(
            load_i=scale3(
                sub3(inputs.capacitor_v, scale3(state.load_i, resistance)),
                1.0 / self.inductance,
            ),
        )


class ODEAPIMicrogridLogger(rg.Node):
    class Inputs(rg.NodeInputs):
        time_s: float = rg.Input(src=rg.Clock.time)
        lcl1_capacitor_v: PhaseVector = rg.Input(src=Lcl1Filter.State.capacitor_v)

    class State(rg.NodeState):
        samples: list[VoltageSample] = rg.Var(init=list)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        sample = (
            inputs.time_s,
            inputs.lcl1_capacitor_v[0],
            inputs.lcl1_capacitor_v[1],
            inputs.lcl1_capacitor_v[2],
        )
        prev_state.samples.append(sample)
        return self.State(samples=prev_state.samples)


def build_system() -> rg.PhasedReactiveSystem:
    master_droop = MasterDroop()
    master_voltage_pi = MasterVoltagePI()
    master_current_pi = MasterCurrentPI()
    slave_pll = SlavePLL()
    slave_inverse_droop = SlaveInverseDroop()
    slave_current_pi = SlaveCurrentPI()
    inverter1 = Inverter1()
    inverter2 = Inverter2()
    lc1 = Lc1Filter()
    lcl1 = Lcl1Filter()
    lc2 = Lc2Filter()
    rl1 = Rl1Load()
    electrical = rg.ODESystem(
        nodes=(lc1, lcl1, lc2, rl1),
        dt="0.00005",
        method="LSODA",
    )
    logger = ODEAPIMicrogridLogger()

    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "control",
                nodes=(
                    master_droop,
                    master_voltage_pi,
                    master_current_pi,
                    slave_pll,
                    slave_inverse_droop,
                    slave_current_pi,
                ),
                transitions=(rg.Goto("inverters"),),
                is_initial=True,
            ),
            rg.Phase("inverters", nodes=(inverter1, inverter2), transitions=(rg.Goto("electrical"),)),
            rg.Phase("electrical", nodes=(electrical,), transitions=(rg.Goto("log"),)),
            rg.Phase("log", nodes=(logger,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root()
        / "artifacts"
        / "two_inverter_static_droop"
        / "lcl1_capacitor_voltages.svg",
    )
    args = parser.parse_args()

    system = build_system()
    system.run(args.steps)
    snapshot = system.snapshot()
    samples = snapshot["ODEAPIMicrogridLogger.samples"]
    save_lcl1_plot(samples, args.output)
    print(f"steps={args.steps}")
    print(f"samples={len(samples)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
