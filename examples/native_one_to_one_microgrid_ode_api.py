from __future__ import annotations

import argparse
import math
from pathlib import Path

import casadi as ca

from examples.native_two_inverter_microgrid import (
    DDS,
    PLL,
    DroopController,
    InverseDroopController,
    MultiPhasePIController,
    PhaseVector,
    VoltageSample,
    abc_to_dq0,
    abc_to_dq0_cos_sin,
    add3,
    clip,
    clip3,
    dq0_to_abc,
    dq0_to_abc_cos_sin,
    inst_power,
    inst_reactive,
    inst_rms,
    repo_root,
    save_lcl1_plot,
    scale3,
    sub3,
    zeros3,
)
from regelum import (
    Clock,
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
from regelum.ode import NodeState, ODENode, ODESystem, StateVar


class ODEAPIDroopController(Node):
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
        self.master_voltage_pi = MultiPhasePIController(
            kp=0.025, ki=60.0, limits=(-i_lim, i_lim), dt=dt
        )
        self.master_current_pi = MultiPhasePIController(
            kp=0.012, ki=90.0, limits=(-1.0, 1.0), dt=dt
        )
        self.master_p_droop = DroopController(gain=40000.0, tau=0.005, nominal=freq_nom, dt=dt)
        self.master_q_droop = DroopController(gain=1000.0, tau=0.002, nominal=v_nom, dt=dt)
        self.master_phase = DDS(dt=dt)

        self.slave_current_pi = MultiPhasePIController(
            kp=0.005, ki=200.0, limits=(-1.0, 1.0), dt=dt
        )
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
        master_current: PhaseVector = Input(source=lambda: Lc1Filter.State.inductor_i)
        master_voltage: PhaseVector = Input(source=lambda: Lc1Filter.State.capacitor_v)
        slave_current: PhaseVector = Input(source=lambda: Lcl1Filter.State.inverter_side_i)
        slave_voltage: PhaseVector = Input(source=lambda: Lcl1Filter.State.capacitor_v)

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

    def _master_control(
        self, *, current: PhaseVector, voltage: PhaseVector
    ) -> tuple[PhaseVector, float]:
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

    def _slave_control(
        self, *, current: PhaseVector, voltage: PhaseVector
    ) -> tuple[PhaseVector, float]:
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


class Inverter1(Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(NodeInputs):
        modulation: PhaseVector = Input(source=ODEAPIDroopController.Outputs.inverter1_modulation)

    class Outputs(NodeOutputs):
        phase_v: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(phase_v=scale3(inputs.modulation, self.gain))


class Inverter2(Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(NodeInputs):
        modulation: PhaseVector = Input(source=ODEAPIDroopController.Outputs.inverter2_modulation)

    class Outputs(NodeOutputs):
        phase_v: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(phase_v=scale3(inputs.modulation, self.gain))


class Lc1Filter(ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class Inputs(NodeInputs):
        inverter_v: PhaseVector = Input(source=Inverter1.Outputs.phase_v)
        lcl1_grid_side_i: PhaseVector = Input(source=lambda: Lcl1Filter.State.grid_side_i)
        lc2_inductor_i: PhaseVector = Input(source=lambda: Lc2Filter.State.inductor_i)

    class State(NodeState):
        capacitor_v: PhaseVector = StateVar(initial=zeros3)
        inductor_i: PhaseVector = StateVar(initial=zeros3)

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


class Lcl1Filter(ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class Inputs(NodeInputs):
        inverter_v: PhaseVector = Input(source=Inverter2.Outputs.phase_v)
        bus_v: PhaseVector = Input(source=Lc1Filter.State.capacitor_v)

    class State(NodeState):
        capacitor_v: PhaseVector = StateVar(initial=zeros3)
        inverter_side_i: PhaseVector = StateVar(initial=zeros3)
        grid_side_i: PhaseVector = StateVar(initial=zeros3)

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


class Lc2Filter(ODENode):
    def __init__(self, *, inductance: float = 0.001, capacitance: float = 1.0e-5) -> None:
        self.inductance = inductance
        self.capacitance = capacitance

    class Inputs(NodeInputs):
        bus_v: PhaseVector = Input(source=Lc1Filter.State.capacitor_v)
        load_i: PhaseVector = Input(source=lambda: Rl1Load.State.load_i)

    class State(NodeState):
        capacitor_v: PhaseVector = StateVar(initial=zeros3)
        inductor_i: PhaseVector = StateVar(initial=zeros3)

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


class Rl1Load(ODENode):
    def __init__(
        self,
        *,
        resistance: float = 20.0,
        inductance: float = 0.001,
    ) -> None:
        self.resistance = resistance
        self.inductance = inductance

    class Inputs(NodeInputs):
        capacitor_v: PhaseVector = Input(source=Lc2Filter.State.capacitor_v)

    class State(NodeState):
        load_i: PhaseVector = StateVar(initial=zeros3)

    def dstate(self, inputs: Inputs, state: State, *, time: float) -> State:
        resistance = ca.if_else(time < 0.2, self.resistance, 2.0 * self.resistance)
        return self.State(
            load_i=scale3(
                sub3(inputs.capacitor_v, scale3(state.load_i, resistance)),
                1.0 / self.inductance,
            ),
        )


class ODEAPIMicrogridLogger(Node):
    class Inputs(NodeInputs):
        time_s: float = Input(source=Clock.time)
        lcl1_capacitor_v: PhaseVector = Input(source=Lcl1Filter.State.capacitor_v)
        samples: tuple[VoltageSample, ...] = Input(source="ODEAPIMicrogridLogger.Outputs.samples")

    class Outputs(NodeOutputs):
        samples: tuple[VoltageSample, ...] = Output(initial=())

    def run(self, inputs: Inputs) -> Outputs:
        sample = (
            inputs.time_s,
            inputs.lcl1_capacitor_v[0],
            inputs.lcl1_capacitor_v[1],
            inputs.lcl1_capacitor_v[2],
        )
        return self.Outputs(samples=inputs.samples + (sample,))


def build_system() -> PhasedReactiveSystem:
    controller = ODEAPIDroopController()
    inverter1 = Inverter1()
    inverter2 = Inverter2()
    lc1 = Lc1Filter()
    lcl1 = Lcl1Filter()
    lc2 = Lc2Filter()
    rl1 = Rl1Load()
    electrical = ODESystem(
        nodes=(lc1, lcl1, lc2, rl1),
        dt="0.00005",
        method="LSODA",
    )
    logger = ODEAPIMicrogridLogger()

    return PhasedReactiveSystem(
        phases=[
            Phase(
                "control", nodes=(controller,), transitions=(Goto("inverters"),), is_initial=True
            ),
            Phase("inverters", nodes=(inverter1, inverter2), transitions=(Goto("electrical"),)),
            Phase("electrical", nodes=(electrical,), transitions=(Goto("log"),)),
            Phase("log", nodes=(logger,), transitions=(Goto(terminate),)),
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
        / "openmodelica_microgrid_gym"
        / "native_one_to_one_ode_api_lcl1_capacitor_voltages.png",
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
