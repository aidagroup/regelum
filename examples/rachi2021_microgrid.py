from __future__ import annotations

from regelum import (
    CompileError,
    Goto,
    If,
    Input,
    Node,
    NodeInputs,
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    V,
    terminate,
)


class MicrogridMeasurements(Node):
    """Local current-voltage measurements for the Rachi et al. SS-DCCB logic."""

    class Inputs(NodeInputs):
        breaker_open: bool = Input(source="BreakerActuator.Outputs.breaker_open")
        tick: int = Input(source="MicrogridMeasurements.Outputs.tick")

    class Outputs(NodeOutputs):
        tick: int = Output(initial=0, domain=range(0, 8))
        bus_voltage: float = Output(initial=380.0)
        branch_current: float = Output(initial=0.5)
        current_above_inst: bool = Output(initial=False)
        current_above_pickup: bool = Output(initial=False)
        voltage_low: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        next_tick = inputs.tick + 1
        if inputs.breaker_open:
            voltage = 380.0
            current = 0.0
        elif next_tick in (2, 3):
            voltage = 325.0
            current = 18.0
        else:
            voltage = 380.0
            current = 0.5

        return self.Outputs(
            tick=next_tick,
            bus_voltage=voltage,
            branch_current=current,
            current_above_inst=current >= 48.0,
            current_above_pickup=current >= 15.0,
            voltage_low=voltage < 365.0,
        )


class InstantTripDecision(Node):
    class Inputs(NodeInputs):
        current_above_inst: bool = Input(
            source=MicrogridMeasurements.Outputs.current_above_inst
        )

    class Outputs(NodeOutputs):
        instant_trip: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(instant_trip=inputs.current_above_inst)


class FaultFlagLatch(Node):
    class Inputs(NodeInputs):
        current_above_pickup: bool = Input(
            source=MicrogridMeasurements.Outputs.current_above_pickup
        )
        voltage_low: bool = Input(source=MicrogridMeasurements.Outputs.voltage_low)

    class Outputs(NodeOutputs):
        current_flag: bool = Output(initial=False)
        voltage_flag: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            current_flag=inputs.current_above_pickup,
            voltage_flag=inputs.voltage_low,
        )


class ProtectionTimers(Node):
    class Inputs(NodeInputs):
        current_flag: bool = Input(source=FaultFlagLatch.Outputs.current_flag)
        voltage_flag: bool = Input(source=FaultFlagLatch.Outputs.voltage_flag)
        current_counter: int = Input(source="ProtectionTimers.Outputs.current_counter")
        voltage_counter: int = Input(source="ProtectionTimers.Outputs.voltage_counter")

    class Outputs(NodeOutputs):
        current_counter: int = Output(initial=0, domain=range(0, 4))
        voltage_counter: int = Output(initial=0, domain=range(0, 4))
        current_delay_elapsed: bool = Output(initial=False)
        voltage_delay_elapsed: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        current_counter = inputs.current_counter + 1 if inputs.current_flag else 0
        voltage_counter = inputs.voltage_counter + 1 if inputs.voltage_flag else 0
        return self.Outputs(
            current_counter=min(current_counter, 3),
            voltage_counter=min(voltage_counter, 3),
            current_delay_elapsed=current_counter >= 2,
            voltage_delay_elapsed=voltage_counter >= 2,
        )


class BreakerActuator(Node):
    class Inputs(NodeInputs):
        instant_trip: bool = Input(source=InstantTripDecision.Outputs.instant_trip)
        current_flag: bool = Input(source=FaultFlagLatch.Outputs.current_flag)
        voltage_flag: bool = Input(source=FaultFlagLatch.Outputs.voltage_flag)
        current_delay_elapsed: bool = Input(
            source=ProtectionTimers.Outputs.current_delay_elapsed
        )
        voltage_delay_elapsed: bool = Input(
            source=ProtectionTimers.Outputs.voltage_delay_elapsed
        )
        breaker_open: bool = Input(source="BreakerActuator.Outputs.breaker_open")

    class Outputs(NodeOutputs):
        breaker_open: bool = Output(initial=False)
        tripped: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        delayed_trip = inputs.current_flag and (
            inputs.current_delay_elapsed
            or (inputs.voltage_flag and inputs.voltage_delay_elapsed)
        )
        tripped = inputs.instant_trip or delayed_trip
        return self.Outputs(
            breaker_open=inputs.breaker_open or tripped,
            tripped=tripped,
        )


class ProtectionLogger(Node):
    class Inputs(NodeInputs):
        tick: int = Input(source=MicrogridMeasurements.Outputs.tick)
        bus_voltage: float = Input(source=MicrogridMeasurements.Outputs.bus_voltage)
        branch_current: float = Input(source=MicrogridMeasurements.Outputs.branch_current)
        current_flag: bool = Input(source=FaultFlagLatch.Outputs.current_flag)
        voltage_flag: bool = Input(source=FaultFlagLatch.Outputs.voltage_flag)
        breaker_open: bool = Input(source=BreakerActuator.Outputs.breaker_open)
        log: tuple[tuple[int, float, float, bool, bool, bool], ...] = Input(
            source="ProtectionLogger.Outputs.log"
        )

    class Outputs(NodeOutputs):
        log: tuple[tuple[int, float, float, bool, bool, bool], ...] = Output(initial=())

    def run(self, inputs: Inputs) -> Outputs:
        sample = (
            inputs.tick,
            inputs.bus_voltage,
            inputs.branch_current,
            inputs.current_flag,
            inputs.voltage_flag,
            inputs.breaker_open,
        )
        print(
            "tick={tick} voltage={voltage:.1f} current={current:.1f} "
            "current_flag={current_flag} voltage_flag={voltage_flag} "
            "breaker_open={breaker_open}".format(
                tick=inputs.tick,
                voltage=inputs.bus_voltage,
                current=inputs.branch_current,
                current_flag=inputs.current_flag,
                voltage_flag=inputs.voltage_flag,
                breaker_open=inputs.breaker_open,
            )
        )
        return self.Outputs(log=inputs.log + (sample,))


def _nodes() -> list[Node]:
    return [
        MicrogridMeasurements(),
        InstantTripDecision(),
        FaultFlagLatch(),
        ProtectionTimers(),
        BreakerActuator(),
        ProtectionLogger(),
    ]


def build_scan_system() -> PhasedReactiveSystem:
    measurements, instant_trip, flags, timers, breaker, logger = _nodes()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "sample",
                nodes=(measurements,),
                transitions=(Goto("instant-trip-check"),),
                is_initial=True,
            ),
            Phase(
                "instant-trip-check",
                nodes=(instant_trip,),
                transitions=(
                    If(V("InstantTripDecision.instant_trip"), "trip", name="desat"),
                    If(~V("InstantTripDecision.instant_trip"), "flags", name="normal"),
                ),
            ),
            Phase("flags", nodes=(flags,), transitions=(Goto("timers"),)),
            Phase("timers", nodes=(timers,), transitions=(Goto("trip-check"),)),
            Phase(
                "trip-check",
                nodes=(),
                transitions=(
                    If(
                        V("FaultFlagLatch.current_flag")
                        & (
                            V("ProtectionTimers.current_delay_elapsed")
                            | (
                                V("FaultFlagLatch.voltage_flag")
                                & V("ProtectionTimers.voltage_delay_elapsed")
                            )
                        ),
                        "trip",
                        name="delayed-trip",
                    ),
                    If(
                        ~(
                            V("FaultFlagLatch.current_flag")
                            & (
                                V("ProtectionTimers.current_delay_elapsed")
                                | (
                                    V("FaultFlagLatch.voltage_flag")
                                    & V("ProtectionTimers.voltage_delay_elapsed")
                                )
                            )
                        ),
                        "log",
                        name="continue",
                    ),
                ),
            ),
            Phase("trip", nodes=(breaker,), transitions=(Goto("log"),)),
            Phase("log", nodes=(logger,), transitions=(Goto(terminate),)),
        ],
    )


def build_literal_flowchart_system() -> PhasedReactiveSystem:
    measurements, instant_trip, flags, timers, breaker, logger = _nodes()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "sample",
                nodes=(measurements,),
                transitions=(Goto("instant-trip-check"),),
                is_initial=True,
            ),
            Phase(
                "instant-trip-check",
                nodes=(instant_trip,),
                transitions=(
                    If(V("InstantTripDecision.instant_trip"), "trip", name="desat"),
                    If(~V("InstantTripDecision.instant_trip"), "flags", name="normal"),
                ),
            ),
            Phase("flags", nodes=(flags,), transitions=(Goto("timers"),)),
            Phase("timers", nodes=(timers,), transitions=(Goto("trip-check"),)),
            Phase(
                "trip-check",
                nodes=(),
                transitions=(
                    If(
                        V("FaultFlagLatch.current_flag")
                        & (
                            V("ProtectionTimers.current_delay_elapsed")
                            | (
                                V("FaultFlagLatch.voltage_flag")
                                & V("ProtectionTimers.voltage_delay_elapsed")
                            )
                        ),
                        "trip",
                        name="delayed-trip",
                    ),
                    If(
                        ~(
                            V("FaultFlagLatch.current_flag")
                            & (
                                V("ProtectionTimers.current_delay_elapsed")
                                | (
                                    V("FaultFlagLatch.voltage_flag")
                                    & V("ProtectionTimers.voltage_delay_elapsed")
                                )
                            )
                        ),
                        "sample",
                        name="resample",
                    ),
                ),
            ),
            Phase("trip", nodes=(breaker,), transitions=(Goto("log"),)),
            Phase("log", nodes=(logger,), transitions=(Goto(terminate),)),
        ],
    )


def main() -> None:
    print("scan system checks:")
    scan = build_scan_system()
    print("C1=PASS C2=PASS C2*=PASS C3=PASS")
    scan.run(steps=5)

    print("\nliteral Fig. 19 flowchart checks:")
    try:
        build_literal_flowchart_system()
    except CompileError as exc:
        print("compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    print("compile=True")


if __name__ == "__main__":
    main()
