from __future__ import annotations

import argparse
from pathlib import Path

from scipy.integrate import solve_ivp

from examples.native_two_inverter_microgrid import (
    NativeDroopController,
    PhaseVector,
    SimulationClock,
    VoltageSample,
    add3,
    repo_root,
    save_lcl1_plot,
    scale3,
    sub3,
    zeros3,
)
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

NetworkState = tuple[float, ...]


def zeros24() -> NetworkState:
    return (0.0,) * 24


def pack_state(parts: tuple[PhaseVector, ...]) -> NetworkState:
    return tuple(value for part in parts for value in part)


def unpack_state(state: NetworkState | list[float]) -> tuple[PhaseVector, ...]:
    return tuple(tuple(state[index : index + 3]) for index in range(0, 24, 3))  # type: ignore[return-value]


class NetworkDroopController(NativeDroopController):
    def __init__(self) -> None:
        super().__init__()

    class Inputs(NodeInputs):
        master_current: PhaseVector = Input(source="Lc1Filter.Outputs.inductor_i")
        master_voltage: PhaseVector = Input(source="Lc1Filter.Outputs.capacitor_v")
        slave_current: PhaseVector = Input(source="Lcl1Filter.Outputs.inverter_side_i")
        slave_voltage: PhaseVector = Input(source="Lcl1Filter.Outputs.capacitor_v")


class Inverter1(Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(NodeInputs):
        modulation: PhaseVector = Input(source=NetworkDroopController.Outputs.inverter1_modulation)

    class Outputs(NodeOutputs):
        phase_v: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(phase_v=scale3(inputs.modulation, self.gain))


class Inverter2(Node):
    def __init__(self, *, v_dc: float = 1000.0) -> None:
        self.gain = 0.5 * v_dc

    class Inputs(NodeInputs):
        modulation: PhaseVector = Input(source=NetworkDroopController.Outputs.inverter2_modulation)

    class Outputs(NodeOutputs):
        phase_v: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(phase_v=scale3(inputs.modulation, self.gain))


class NetworkOdeStep(Node):
    """One LSODA step for the continuous electrical network from `Network.mo`."""

    def __init__(
        self,
        *,
        dt: float = 0.5e-4,
        inductance: float = 0.001,
        capacitance: float = 1.0e-5,
        load_resistance: float = 20.0,
        load_inductance: float = 0.001,
    ) -> None:
        self.dt = dt
        self.inductance = inductance
        self.capacitance = capacitance
        self.load_resistance = load_resistance
        self.load_inductance = load_inductance

    class Inputs(NodeInputs):
        time_s: float = Input(source=SimulationClock.Outputs.time_s)
        inverter1_v: PhaseVector = Input(source=Inverter1.Outputs.phase_v)
        inverter2_v: PhaseVector = Input(source=Inverter2.Outputs.phase_v)
        state: NetworkState = Input(source="NetworkOdeStep.Outputs.state")

    class Outputs(NodeOutputs):
        state: NetworkState = Output(initial=zeros24)
        lc1_capacitor_v: PhaseVector = Output(initial=zeros3)
        lc1_inductor_i: PhaseVector = Output(initial=zeros3)
        lcl1_capacitor_v: PhaseVector = Output(initial=zeros3)
        lcl1_inverter_side_i: PhaseVector = Output(initial=zeros3)
        lcl1_grid_side_i: PhaseVector = Output(initial=zeros3)
        lc2_capacitor_v: PhaseVector = Output(initial=zeros3)
        lc2_inductor_i: PhaseVector = Output(initial=zeros3)
        rl1_load_i: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        load_resistance = (
            self.load_resistance if inputs.time_s < 0.2 else 2.0 * self.load_resistance
        )
        result = solve_ivp(
            lambda t, y: self._derivative(
                t,
                y,
                inverter1_v=inputs.inverter1_v,
                inverter2_v=inputs.inverter2_v,
                load_resistance=load_resistance,
            ),
            (inputs.time_s, inputs.time_s + self.dt),
            list(inputs.state),
            method="LSODA",
            jac=lambda _t, _y: self._jacobian(load_resistance),
        )
        state = tuple(float(value) for value in result.y[:, -1])
        (
            lc1_capacitor_v,
            lc1_inductor_i,
            lcl1_capacitor_v,
            lcl1_inverter_side_i,
            lcl1_grid_side_i,
            lc2_capacitor_v,
            lc2_inductor_i,
            rl1_load_i,
        ) = unpack_state(state)
        return self.Outputs(
            state=state,
            lc1_capacitor_v=lc1_capacitor_v,
            lc1_inductor_i=lc1_inductor_i,
            lcl1_capacitor_v=lcl1_capacitor_v,
            lcl1_inverter_side_i=lcl1_inverter_side_i,
            lcl1_grid_side_i=lcl1_grid_side_i,
            lc2_capacitor_v=lc2_capacitor_v,
            lc2_inductor_i=lc2_inductor_i,
            rl1_load_i=rl1_load_i,
        )

    def _derivative(
        self,
        time_s: float,
        state: list[float],
        *,
        inverter1_v: PhaseVector,
        inverter2_v: PhaseVector,
        load_resistance: float,
    ) -> list[float]:
        (
            lc1_capacitor_v,
            lc1_inductor_i,
            lcl1_capacitor_v,
            lcl1_inverter_side_i,
            lcl1_grid_side_i,
            lc2_capacitor_v,
            lc2_inductor_i,
            rl1_load_i,
        ) = unpack_state(state)
        derivatives = (
            scale3(sub3(add3(lc1_inductor_i, lcl1_grid_side_i), lc2_inductor_i), 1.0 / self.capacitance),
            scale3(sub3(inverter1_v, lc1_capacitor_v), 1.0 / self.inductance),
            scale3(sub3(lcl1_inverter_side_i, lcl1_grid_side_i), 1.0 / self.capacitance),
            scale3(sub3(inverter2_v, lcl1_capacitor_v), 1.0 / self.inductance),
            scale3(sub3(lcl1_capacitor_v, lc1_capacitor_v), 1.0 / self.inductance),
            scale3(sub3(lc2_inductor_i, rl1_load_i), 1.0 / self.capacitance),
            scale3(sub3(lc1_capacitor_v, lc2_capacitor_v), 1.0 / self.inductance),
            scale3(
                sub3(lc2_capacitor_v, scale3(rl1_load_i, load_resistance)),
                1.0 / self.load_inductance,
            ),
        )
        return list(pack_state(derivatives))

    def _jacobian(self, load_resistance: float) -> list[list[float]]:
        jacobian = [[0.0 for _ in range(24)] for _ in range(24)]

        def set_entry(row: int, col: int, value: float) -> None:
            jacobian[row][col] = value

        inv_l = 1.0 / self.inductance
        inv_c = 1.0 / self.capacitance
        inv_load_l = 1.0 / self.load_inductance

        for phase in range(3):
            lc1_v = phase
            lc1_i = 3 + phase
            lcl_v = 6 + phase
            lcl_left_i = 9 + phase
            lcl_right_i = 12 + phase
            lc2_v = 15 + phase
            lc2_i = 18 + phase
            rl_i = 21 + phase

            set_entry(lc1_v, lc1_i, inv_c)
            set_entry(lc1_v, lcl_right_i, inv_c)
            set_entry(lc1_v, lc2_i, -inv_c)

            set_entry(lc1_i, lc1_v, -inv_l)

            set_entry(lcl_v, lcl_left_i, inv_c)
            set_entry(lcl_v, lcl_right_i, -inv_c)

            set_entry(lcl_left_i, lcl_v, -inv_l)

            set_entry(lcl_right_i, lcl_v, inv_l)
            set_entry(lcl_right_i, lc1_v, -inv_l)

            set_entry(lc2_v, lc2_i, inv_c)
            set_entry(lc2_v, rl_i, -inv_c)

            set_entry(lc2_i, lc1_v, inv_l)
            set_entry(lc2_i, lc2_v, -inv_l)

            set_entry(rl_i, lc2_v, inv_load_l)
            set_entry(rl_i, rl_i, -load_resistance * inv_load_l)

        return jacobian


class Lc1Filter(Node):
    """Modelica `lc1`: inverter-side L and bus shunt C."""

    class Inputs(NodeInputs):
        inductor_i: PhaseVector = Input(source=NetworkOdeStep.Outputs.lc1_inductor_i)
        capacitor_v: PhaseVector = Input(source=NetworkOdeStep.Outputs.lc1_capacitor_v)

    class Outputs(NodeOutputs):
        inductor_i: PhaseVector = Output(initial=zeros3)
        capacitor_v: PhaseVector = Output(initial=zeros3)
        current_to_bus: PhaseVector = Output(initial=zeros3)
        capacitor1_v: float = Output(initial=0.0)
        capacitor2_v: float = Output(initial=0.0)
        capacitor3_v: float = Output(initial=0.0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            inductor_i=inputs.inductor_i,
            capacitor_v=inputs.capacitor_v,
            current_to_bus=inputs.inductor_i,
            capacitor1_v=inputs.capacitor_v[0],
            capacitor2_v=inputs.capacitor_v[1],
            capacitor3_v=inputs.capacitor_v[2],
        )


class Lcl1Filter(Node):
    """Modelica `lcl1`: inverter L, midpoint shunt C, and grid-side L."""

    class Inputs(NodeInputs):
        inverter_side_i: PhaseVector = Input(source=NetworkOdeStep.Outputs.lcl1_inverter_side_i)
        capacitor_v: PhaseVector = Input(source=NetworkOdeStep.Outputs.lcl1_capacitor_v)
        grid_side_i: PhaseVector = Input(source=NetworkOdeStep.Outputs.lcl1_grid_side_i)

    class Outputs(NodeOutputs):
        inverter_side_i: PhaseVector = Output(initial=zeros3)
        capacitor_v: PhaseVector = Output(initial=zeros3)
        grid_side_i: PhaseVector = Output(initial=zeros3)
        capacitor1_v: float = Output(initial=0.0)
        capacitor2_v: float = Output(initial=0.0)
        capacitor3_v: float = Output(initial=0.0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            inverter_side_i=inputs.inverter_side_i,
            capacitor_v=inputs.capacitor_v,
            grid_side_i=inputs.grid_side_i,
            capacitor1_v=inputs.capacitor_v[0],
            capacitor2_v=inputs.capacitor_v[1],
            capacitor3_v=inputs.capacitor_v[2],
        )


class Lc2Filter(Node):
    """Modelica `lc2`: load-side LC between the common bus and `rl1`."""

    class Inputs(NodeInputs):
        inductor_i: PhaseVector = Input(source=NetworkOdeStep.Outputs.lc2_inductor_i)
        capacitor_v: PhaseVector = Input(source=NetworkOdeStep.Outputs.lc2_capacitor_v)

    class Outputs(NodeOutputs):
        inductor_i: PhaseVector = Output(initial=zeros3)
        capacitor_v: PhaseVector = Output(initial=zeros3)
        capacitor1_v: float = Output(initial=0.0)
        capacitor2_v: float = Output(initial=0.0)
        capacitor3_v: float = Output(initial=0.0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            inductor_i=inputs.inductor_i,
            capacitor_v=inputs.capacitor_v,
            capacitor1_v=inputs.capacitor_v[0],
            capacitor2_v=inputs.capacitor_v[1],
            capacitor3_v=inputs.capacitor_v[2],
        )


class Rl1Load(Node):
    class Inputs(NodeInputs):
        load_i: PhaseVector = Input(source=NetworkOdeStep.Outputs.rl1_load_i)

    class Outputs(NodeOutputs):
        load_i: PhaseVector = Output(initial=zeros3)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(load_i=inputs.load_i)


class OneToOneMicrogridLogger(Node):
    class Inputs(NodeInputs):
        time_s: float = Input(source=SimulationClock.Outputs.time_s)
        lcl1_capacitor1_v: float = Input(source=Lcl1Filter.Outputs.capacitor1_v)
        lcl1_capacitor2_v: float = Input(source=Lcl1Filter.Outputs.capacitor2_v)
        lcl1_capacitor3_v: float = Input(source=Lcl1Filter.Outputs.capacitor3_v)
        samples: tuple[VoltageSample, ...] = Input(source="OneToOneMicrogridLogger.Outputs.samples")

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
    controller = NetworkDroopController()
    inverter1 = Inverter1()
    lc1 = Lc1Filter()
    inverter2 = Inverter2()
    network = NetworkOdeStep()
    lcl1 = Lcl1Filter()
    lc2 = Lc2Filter()
    rl1 = Rl1Load()
    clock = SimulationClock()
    logger = OneToOneMicrogridLogger()

    return PhasedReactiveSystem(
        phases=[
            Phase("control", nodes=(controller,), transitions=(Goto("inverters"),), is_initial=True),
            Phase("inverters", nodes=(inverter1, inverter2), transitions=(Goto("network-step"),)),
            Phase("network-step", nodes=(network,), transitions=(Goto("lcl1"),)),
            Phase("lcl1", nodes=(lcl1,), transitions=(Goto("lc2"),)),
            Phase("lc2", nodes=(lc2,), transitions=(Goto("rl1"),)),
            Phase("rl1", nodes=(rl1,), transitions=(Goto("common-bus"),)),
            Phase("common-bus", nodes=(lc1,), transitions=(Goto("clock"),)),
            Phase("clock", nodes=(clock,), transitions=(Goto("log"),)),
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
        / "native_one_to_one_lcl1_capacitor_voltages.png",
    )
    args = parser.parse_args()

    system = build_system()
    system.run(args.steps)
    snapshot = system.snapshot()
    samples = snapshot["OneToOneMicrogridLogger.samples"]
    save_lcl1_plot(samples, args.output)
    print(f"steps={args.steps}")
    print(f"samples={len(samples)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
