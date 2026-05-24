from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import regelum as rg

VoltageResistanceSample = tuple[float, float, float, float, float]


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "regelum").exists():
            return parent
    raise RuntimeError("Could not locate the regelum repository root.")


def zero_abc() -> np.ndarray:
    return np.zeros(3, dtype=float)


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def inst_rms(value: np.ndarray) -> float:
    return float(np.linalg.norm(value) / math.sqrt(3.0))


def normalise_abc(value: np.ndarray) -> np.ndarray:
    magnitude = inst_rms(value)
    if magnitude == 0:
        return value
    return value / magnitude


def abc_to_alpha_beta(abc: np.ndarray) -> tuple[float, float]:
    alpha = (2.0 / 3.0) * (abc[0] - 0.5 * abc[1] - 0.5 * abc[2])
    beta = (2.0 / 3.0) * (0.866 * abc[1] - 0.866 * abc[2])
    return float(alpha), float(beta)


def cos_sin(theta: float) -> tuple[float, float]:
    return math.cos(theta), math.sin(theta)


def dq0_to_abc_cos_sin(dq0: np.ndarray, cos_value: float, sin_value: float) -> np.ndarray:
    transform = np.array(
        [
            [cos_value, -sin_value, 1.0],
            [
                cos_value * (-0.5) - sin_value * (-0.866),
                -(sin_value * (-0.5) + cos_value * (-0.866)),
                1.0,
            ],
            [
                cos_value * (-0.5) - sin_value * 0.866,
                -(sin_value * (-0.5) + cos_value * 0.866),
                1.0,
            ],
        ],
        dtype=float,
    )
    return transform @ dq0


def dq0_to_abc(dq0: np.ndarray, theta: float) -> np.ndarray:
    return dq0_to_abc_cos_sin(dq0, *cos_sin(theta))


def abc_to_dq0_cos_sin(abc: np.ndarray, cos_value: float, sin_value: float) -> np.ndarray:
    cos_shift_neg = cos_value * (-0.5) - sin_value * (-0.866)
    sin_shift_neg = sin_value * (-0.5) + cos_value * (-0.866)
    cos_shift_pos = cos_value * (-0.5) - sin_value * 0.866
    sin_shift_pos = sin_value * (-0.5) + cos_value * 0.866
    return np.array(
        [
            (2.0 / 3.0) * (cos_value * abc[0] + cos_shift_neg * abc[1] + cos_shift_pos * abc[2]),
            (2.0 / 3.0) * (-sin_value * abc[0] - sin_shift_neg * abc[1] - sin_shift_pos * abc[2]),
            float(np.sum(abc) / 3.0),
        ],
        dtype=float,
    )


def abc_to_dq0(abc: np.ndarray, theta: float) -> np.ndarray:
    return abc_to_dq0_cos_sin(abc, *cos_sin(theta))


def inst_power(v_abc: np.ndarray, i_abc: np.ndarray) -> float:
    return float(np.dot(v_abc, i_abc))


def inst_reactive(v_abc: np.ndarray, i_abc: np.ndarray) -> float:
    quadrature_v = np.roll(v_abc, -1) - np.roll(v_abc, -2)
    return float(-0.5773502691896258 * np.dot(quadrature_v, i_abc))


def pi_update(
    *,
    setpoint: np.ndarray,
    measured: np.ndarray,
    integral: np.ndarray,
    windup: np.ndarray,
    kp: float,
    ki: float,
    limits: tuple[float, float],
    dt: float,
    feedforward: np.ndarray | None = None,
    kb: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feedforward is None:
        feedforward = zero_abc()
    error = setpoint - measured
    next_integral = integral + (ki * error + windup) * dt
    raw = kp * error + next_integral + feedforward
    output = np.clip(raw, limits[0], limits[1])
    next_windup = (raw - output) * kb
    return output, next_integral, next_windup


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


def save_lcl1_plot(samples: list[VoltageResistanceSample], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    time = [row[0] for row in samples]
    v1 = [row[1] for row in samples]
    v2 = [row[2] for row in samples]
    v3 = [row[3] for row in samples]
    resistance = [row[4] for row in samples]

    fig, voltage_ax = plt.subplots(figsize=(8.0, 4.8), dpi=100)
    voltage_ax.plot(time, v1, label="AC phase A voltage", linewidth=1.2)
    voltage_ax.plot(time, v2, label="AC phase B voltage", linewidth=1.2)
    voltage_ax.plot(time, v3, label="AC phase C voltage", linewidth=1.2)
    voltage_ax.set_xlabel("time, s")
    voltage_ax.set_ylabel("AC phase voltage, V")
    voltage_ax.set_xlim(min(time), max(time))
    voltage_ax.grid(True, alpha=0.25)

    resistance_ax = voltage_ax.twinx()
    resistance_ax.step(
        time,
        resistance,
        where="post",
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="load resistance",
    )
    resistance_ax.set_ylabel("load resistance, Ohm")

    voltage_lines, voltage_labels = voltage_ax.get_legend_handles_labels()
    resistance_lines, resistance_labels = resistance_ax.get_legend_handles_labels()
    fig.legend(
        voltage_lines + resistance_lines,
        voltage_labels + resistance_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
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
        lc1_i_abc: np.ndarray = rg.src(lambda: Lc1.State.i_abc)
        lc1_v_abc: np.ndarray = rg.src(lambda: Lc1.State.v_abc)

    class State(rg.NodeState):
        frequency_hz: float = rg.var(init=50.0)
        voltage_setpoint_v: float = rg.var(init=230.0 * math.sqrt(2.0))
        ac_phase_rad: float = rg.var(init=0.0)
        ac_phase_turns: float = rg.var(init=0.0)
        p_filter: float = rg.var(init=0.0)
        q_filter: float = rg.var(init=0.0)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        instant_power = -inst_power(inputs.lc1_v_abc, inputs.lc1_i_abc)
        p_filter = pt1_update(
            value=instant_power,
            integral=prev_state.p_filter,
            gain=1.0 / 40000.0,
            tau=0.005,
            dt=self.dt,
        )
        frequency_hz = p_filter + self.freq_nom

        instant_q = -inst_reactive(inputs.lc1_v_abc, inputs.lc1_i_abc)
        q_filter = pt1_update(
            value=instant_q,
            integral=prev_state.q_filter,
            gain=1.0 / 1000.0,
            tau=0.002,
            dt=self.dt,
        )
        voltage_setpoint_v = q_filter + self.v_nom
        ac_phase_turns = advance_phase_turns(prev_state.ac_phase_turns, frequency_hz, self.dt)
        return self.State(
            frequency_hz=frequency_hz,
            voltage_setpoint_v=voltage_setpoint_v,
            ac_phase_rad=ac_phase_turns * 2.0 * math.pi,
            ac_phase_turns=ac_phase_turns,
            p_filter=p_filter,
            q_filter=q_filter,
        )


class MasterLc1VPI(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4, i_lim: float = 30.0) -> None:
        self.dt = dt
        self.i_lim = i_lim

    class Inputs(rg.NodeInputs):
        lc1_v_abc: np.ndarray = rg.src(lambda: Lc1.State.v_abc)
        ac_phase_rad: float = rg.src(MasterDroop.State.ac_phase_rad)
        voltage_setpoint_v: float = rg.src(MasterDroop.State.voltage_setpoint_v)

    class State(rg.NodeState):
        i_ref_dq0: np.ndarray = rg.var(init=zero_abc)
        integral: np.ndarray = rg.var(init=zero_abc)
        windup: np.ndarray = rg.var(init=zero_abc)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        voltage_dq0 = abc_to_dq0(inputs.lc1_v_abc, inputs.ac_phase_rad)
        voltage_setpoint_dq0 = np.array([inputs.voltage_setpoint_v, 0.0, 0.0], dtype=float)
        i_ref_dq0, integral, windup = pi_update(
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
            i_ref_dq0=i_ref_dq0,
            integral=integral,
            windup=windup,
        )


class MasterLc1IPI(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4) -> None:
        self.dt = dt

    class Inputs(rg.NodeInputs):
        lc1_i_abc: np.ndarray = rg.src(lambda: Lc1.State.i_abc)
        ac_phase_rad: float = rg.src(MasterDroop.State.ac_phase_rad)
        i_ref_dq0: np.ndarray = rg.src(MasterLc1VPI.State.i_ref_dq0)

    class State(rg.NodeState):
        modulation: np.ndarray = rg.var(init=zero_abc)
        integral: np.ndarray = rg.var(init=zero_abc)
        windup: np.ndarray = rg.var(init=zero_abc)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        i_dq0 = abc_to_dq0(inputs.lc1_i_abc, inputs.ac_phase_rad)
        modulation_dq0, integral, windup = pi_update(
            setpoint=inputs.i_ref_dq0,
            measured=i_dq0,
            integral=prev_state.integral,
            windup=prev_state.windup,
            kp=0.012,
            ki=90.0,
            limits=(-1.0, 1.0),
            dt=self.dt,
        )
        modulation = np.clip(dq0_to_abc(modulation_dq0, inputs.ac_phase_rad), -1.0, 1.0)
        return self.State(modulation=modulation, integral=integral, windup=windup)


class SlavePLL(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4, f_nom: float = 50.0) -> None:
        self.dt = dt
        self.f_nom = f_nom

    class Inputs(rg.NodeInputs):
        lcl1_v_abc: np.ndarray = rg.src(lambda: Lcl1.State.v_abc)

    class State(rg.NodeState):
        cos_ac_phase: float = rg.var(init=1.0)
        sin_ac_phase: float = rg.var(init=0.0)
        frequency_hz: float = rg.var(init=50.0)
        ac_phase_rad: float = rg.var(init=0.0)
        ac_phase_turns: float = rg.var(init=0.0)
        integral: float = rg.var(init=0.0)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        normalised = normalise_abc(inputs.lcl1_v_abc)
        alpha_beta = abc_to_alpha_beta(normalised)
        dphi = alpha_beta[1] * prev_state.cos_ac_phase - alpha_beta[0] * prev_state.sin_ac_phase
        integral = prev_state.integral + 200.0 * dphi * self.dt
        frequency_hz = 10.0 * dphi + integral + self.f_nom
        ac_phase_turns = advance_phase_turns(prev_state.ac_phase_turns, frequency_hz, self.dt)
        ac_phase_rad = ac_phase_turns * 2.0 * math.pi
        cos_ac_phase, sin_ac_phase = cos_sin(ac_phase_rad)
        return self.State(
            cos_ac_phase=cos_ac_phase,
            sin_ac_phase=sin_ac_phase,
            frequency_hz=frequency_hz,
            ac_phase_rad=ac_phase_rad,
            ac_phase_turns=ac_phase_turns,
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
        lcl1_v_abc: np.ndarray = rg.src(lambda: Lcl1.State.v_abc)
        frequency_hz: float = rg.src(SlavePLL.State.frequency_hz)

    class State(rg.NodeState):
        i_ref_dq0: np.ndarray = rg.var(init=zero_abc)
        p_filter: float = rg.var(init=0.0)
        p_previous: float = rg.var(init=0.0)
        q_filter: float = rg.var(init=0.0)
        q_previous: float = rg.var(init=0.0)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        v_inst = inst_rms(inputs.lcl1_v_abc)
        if v_inst <= 150.0:
            return self.State(
                i_ref_dq0=zero_abc(),
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

        active_i = p_output / v_inst
        reactive_i = q_output / v_inst
        droop = np.array(
            [
                clip(active_i / 3.0 * math.sqrt(2.0), -self.i_lim, self.i_lim),
                clip(reactive_i / 3.0 * math.sqrt(2.0), -self.i_lim, self.i_lim),
                0.0,
            ],
            dtype=float,
        )
        return self.State(
            i_ref_dq0=np.array([-droop[0], droop[1], droop[2]], dtype=float),
            p_filter=p_filter,
            p_previous=p_filter,
            q_filter=q_filter,
            q_previous=q_filter,
        )


class SlaveLcl1IPI(rg.Node):
    def __init__(self, *, dt: float = 0.5e-4) -> None:
        self.dt = dt

    class Inputs(rg.NodeInputs):
        lcl1_inv_i_abc: np.ndarray = rg.src(lambda: Lcl1.State.inv_i_abc)
        cos_ac_phase: float = rg.src(SlavePLL.State.cos_ac_phase)
        sin_ac_phase: float = rg.src(SlavePLL.State.sin_ac_phase)
        i_ref_dq0: np.ndarray = rg.src(SlaveInverseDroop.State.i_ref_dq0)

    class State(rg.NodeState):
        modulation: np.ndarray = rg.var(init=zero_abc)
        integral: np.ndarray = rg.var(init=zero_abc)
        windup: np.ndarray = rg.var(init=zero_abc)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        i_dq0 = abc_to_dq0_cos_sin(
            inputs.lcl1_inv_i_abc,
            inputs.cos_ac_phase,
            inputs.sin_ac_phase,
        )
        modulation_dq0, integral, windup = pi_update(
            setpoint=inputs.i_ref_dq0,
            measured=i_dq0,
            integral=prev_state.integral,
            windup=prev_state.windup,
            kp=0.005,
            ki=200.0,
            limits=(-1.0, 1.0),
            dt=self.dt,
        )
        modulation = np.clip(
            dq0_to_abc_cos_sin(modulation_dq0, inputs.cos_ac_phase, inputs.sin_ac_phase),
            -1.0,
            1.0,
        )
        return self.State(modulation=modulation, integral=integral, windup=windup)


class Inverter1(rg.Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(rg.NodeInputs):
        modulation: np.ndarray = rg.src(MasterLc1IPI.State.modulation)

    class State(rg.NodeState):
        v_abc: np.ndarray = rg.var(init=zero_abc)

    def update(self, inputs: Inputs) -> State:
        return self.State(v_abc=inputs.modulation * self.gain)


class Inverter2(rg.Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(rg.NodeInputs):
        modulation: np.ndarray = rg.src(SlaveLcl1IPI.State.modulation)

    class State(rg.NodeState):
        v_abc: np.ndarray = rg.var(init=zero_abc)

    def update(self, inputs: Inputs) -> State:
        return self.State(v_abc=inputs.modulation * self.gain)


class ResistanceScenario(rg.Node):
    def __init__(
        self,
        *,
        base_resistance: float = 20.0,
        first_switch_tick: int,
        second_switch_tick: int,
    ) -> None:
        self.base_resistance = base_resistance
        self.first_switch_tick = first_switch_tick
        self.second_switch_tick = second_switch_tick

    class Inputs(rg.NodeInputs):
        tick: int = rg.src(rg.Clock.tick)

    class State(rg.NodeState):
        resistance: float = rg.var(init=20.0)

    def update(self, inputs: Inputs) -> State:
        if inputs.tick < self.first_switch_tick:
            resistance = self.base_resistance
        elif inputs.tick < self.second_switch_tick:
            resistance = 2.0 * self.base_resistance
        else:
            resistance = self.base_resistance
        return self.State(resistance=resistance)


class Lc1(rg.ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class State(rg.NodeState):
        v_abc: np.ndarray = rg.var(init=zero_abc)
        i_abc: np.ndarray = rg.var(init=zero_abc)

    def dstate(
        self,
        state: State,
        inverter_v_abc: np.ndarray = rg.src(Inverter1.State.v_abc),
        lcl1_bus_i_abc: np.ndarray = rg.src(lambda: Lcl1.State.bus_i_abc),
        lc2_i_abc: np.ndarray = rg.src(lambda: Lc2.State.i_abc),
    ) -> State:  # ty: ignore[invalid-method-override]
        return self.State(
            v_abc=(state.i_abc + lcl1_bus_i_abc - lc2_i_abc) / self.capacitance,
            i_abc=(inverter_v_abc - state.v_abc) / self.inductance,
        )


class Lcl1(rg.ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class State(rg.NodeState):
        v_abc: np.ndarray = rg.var(init=zero_abc)
        inv_i_abc: np.ndarray = rg.var(init=zero_abc)
        bus_i_abc: np.ndarray = rg.var(init=zero_abc)

    def dstate(
        self,
        state: State,
        inverter_v_abc: np.ndarray = rg.src(Inverter2.State.v_abc),
        bus_v_abc: np.ndarray = rg.src(Lc1.State.v_abc),
    ) -> State:  # ty: ignore[invalid-method-override]
        return self.State(
            v_abc=(state.inv_i_abc - state.bus_i_abc) / self.capacitance,
            inv_i_abc=(inverter_v_abc - state.v_abc) / self.inductance,
            bus_i_abc=(state.v_abc - bus_v_abc) / self.inductance,
        )


class Lc2(rg.ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class State(rg.NodeState):
        v_abc: np.ndarray = rg.var(init=zero_abc)
        i_abc: np.ndarray = rg.var(init=zero_abc)

    def dstate(
        self,
        state: State,
        bus_v_abc: np.ndarray = rg.src(Lc1.State.v_abc),
        rl1_i_abc: np.ndarray = rg.src(lambda: Rl1.State.i_abc),
    ) -> State:  # ty: ignore[invalid-method-override]
        return self.State(
            v_abc=(state.i_abc - rl1_i_abc) / self.capacitance,
            i_abc=(bus_v_abc - state.v_abc) / self.inductance,
        )


class Rl1(rg.ODENode):
    def __init__(self, *, inductance: float = 0.001) -> None:
        self.inductance = inductance

    class Inputs(rg.NodeInputs):
        lc2_v_abc: np.ndarray = rg.src(Lc2.State.v_abc)
        resistance: float = rg.src(ResistanceScenario.State.resistance)

    class State(rg.NodeState):
        i_abc: np.ndarray = rg.var(init=zero_abc)

    def dstate(self, inputs: Inputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(
            i_abc=(inputs.lc2_v_abc - inputs.resistance * state.i_abc) / self.inductance,
        )


class Logger(rg.Node):
    class Inputs(rg.NodeInputs):
        time_s: float = rg.src(rg.Clock.time)
        lcl1_v_abc: np.ndarray = rg.src(Lcl1.State.v_abc)
        resistance: float = rg.src(ResistanceScenario.State.resistance)

    class State(rg.NodeState):
        samples: list[VoltageResistanceSample] = rg.var(init=list)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        sample = (
            inputs.time_s,
            float(inputs.lcl1_v_abc[0]),
            float(inputs.lcl1_v_abc[1]),
            float(inputs.lcl1_v_abc[2]),
            inputs.resistance,
        )
        prev_state.samples.append(sample)
        return self.State(samples=prev_state.samples)


def build_system(*, steps: int = 2000) -> rg.PhasedReactiveSystem:
    master_droop = MasterDroop()
    master_lc1_v_pi = MasterLc1VPI()
    master_lc1_i_pi = MasterLc1IPI()
    slave_pll = SlavePLL()
    slave_inverse_droop = SlaveInverseDroop()
    slave_lcl1_i_pi = SlaveLcl1IPI()
    inverter1 = Inverter1()
    inverter2 = Inverter2()
    resistance = ResistanceScenario(
        first_switch_tick=steps // 3,
        second_switch_tick=2 * steps // 3,
    )
    lc1 = Lc1()
    lcl1 = Lcl1()
    lc2 = Lc2()
    rl1 = Rl1()
    electrical = rg.ODESystem(
        nodes=(lc1, lcl1, lc2, rl1),
        dt="0.00005",
        backend="casadi",
        method="cvodes",
        options={"abstol": 1e-9, "reltol": 1e-8},
    )
    logger = Logger()

    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "control",
                nodes=(
                    master_droop,
                    master_lc1_v_pi,
                    master_lc1_i_pi,
                    slave_pll,
                    slave_inverse_droop,
                    slave_lcl1_i_pi,
                ),
                transitions=(rg.Goto("inverters"),),
                is_initial=True,
            ),
            rg.Phase("inverters", nodes=(inverter1, inverter2), transitions=(rg.Goto("scenario"),)),
            rg.Phase("scenario", nodes=(resistance,), transitions=(rg.Goto("electrical"),)),
            rg.Phase("electrical", nodes=(electrical,), transitions=(rg.Goto("log"),)),
            rg.Phase("log", nodes=(logger,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "lcl1_voltage_and_resistance.svg",
    )
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=repo_root()
        / "docs"
        / "assets"
        / "examples"
        / "two_inverter_static_droop"
        / "lcl1_voltage_and_resistance.svg",
    )
    args = parser.parse_args()

    system = build_system(steps=args.steps)
    system.run(args.steps)
    snapshot = system.snapshot()
    samples = snapshot["Logger.samples"]
    save_lcl1_plot(samples, args.output)
    args.docs_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.output, args.docs_output)
    print(f"steps={args.steps}")
    print(f"samples={len(samples)}")
    print(args.output.resolve())
    print(args.docs_output.resolve())


if __name__ == "__main__":
    main()
