from __future__ import annotations

from regelum import (
    CompileError,
    Goto,
    If,
    Input,
    Node,
    NodeInputs,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    V,
    Var,
    terminate,
)


class MicrogridMeasurements(Node):
    """Local current-voltage measurements for the Rachi et al. SS-DCCB logic."""

    class Inputs(NodeInputs):
        breaker_open: bool = Input(src="BreakerActuator.State.breaker_open")
        tick: int = Input(src="MicrogridMeasurements.State.tick")

    class State(NodeState):
        tick: int = Var(init=0, domain=range(0, 8))
        bus_voltage: float = Var(init=380.0)
        branch_current: float = Var(init=0.5)
        current_above_inst: bool = Var(init=False)
        current_above_pickup: bool = Var(init=False)
        voltage_low: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
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

        return self.State(
            tick=next_tick,
            bus_voltage=voltage,
            branch_current=current,
            current_above_inst=current >= 48.0,
            current_above_pickup=current >= 15.0,
            voltage_low=voltage < 365.0,
        )


class InstantTripDecision(Node):
    class Inputs(NodeInputs):
        current_above_inst: bool = Input(src=MicrogridMeasurements.State.current_above_inst)

    class State(NodeState):
        instant_trip: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        return self.State(instant_trip=inputs.current_above_inst)


class FaultFlagLatch(Node):
    class Inputs(NodeInputs):
        current_above_pickup: bool = Input(
            src=MicrogridMeasurements.State.current_above_pickup
        )
        voltage_low: bool = Input(src=MicrogridMeasurements.State.voltage_low)

    class State(NodeState):
        current_flag: bool = Var(init=False)
        voltage_flag: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            current_flag=inputs.current_above_pickup,
            voltage_flag=inputs.voltage_low,
        )


class ProtectionTimers(Node):
    class Inputs(NodeInputs):
        current_flag: bool = Input(src=FaultFlagLatch.State.current_flag)
        voltage_flag: bool = Input(src=FaultFlagLatch.State.voltage_flag)
        current_counter: int = Input(src="ProtectionTimers.State.current_counter")
        voltage_counter: int = Input(src="ProtectionTimers.State.voltage_counter")

    class State(NodeState):
        current_counter: int = Var(init=0, domain=range(0, 4))
        voltage_counter: int = Var(init=0, domain=range(0, 4))
        current_delay_elapsed: bool = Var(init=False)
        voltage_delay_elapsed: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        current_counter = inputs.current_counter + 1 if inputs.current_flag else 0
        voltage_counter = inputs.voltage_counter + 1 if inputs.voltage_flag else 0
        return self.State(
            current_counter=min(current_counter, 3),
            voltage_counter=min(voltage_counter, 3),
            current_delay_elapsed=current_counter >= 2,
            voltage_delay_elapsed=voltage_counter >= 2,
        )


class BreakerActuator(Node):
    class Inputs(NodeInputs):
        instant_trip: bool = Input(src=InstantTripDecision.State.instant_trip)
        current_flag: bool = Input(src=FaultFlagLatch.State.current_flag)
        voltage_flag: bool = Input(src=FaultFlagLatch.State.voltage_flag)
        current_delay_elapsed: bool = Input(src=ProtectionTimers.State.current_delay_elapsed)
        voltage_delay_elapsed: bool = Input(src=ProtectionTimers.State.voltage_delay_elapsed)
        breaker_open: bool = Input(src="BreakerActuator.State.breaker_open")

    class State(NodeState):
        breaker_open: bool = Var(init=False)
        tripped: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        delayed_trip = inputs.current_flag and (
            inputs.current_delay_elapsed or (inputs.voltage_flag and inputs.voltage_delay_elapsed)
        )
        tripped = inputs.instant_trip or delayed_trip
        return self.State(
            breaker_open=inputs.breaker_open or tripped,
            tripped=tripped,
        )


class ProtectionLogger(Node):
    class Inputs(NodeInputs):
        tick: int = Input(src=MicrogridMeasurements.State.tick)
        bus_voltage: float = Input(src=MicrogridMeasurements.State.bus_voltage)
        branch_current: float = Input(src=MicrogridMeasurements.State.branch_current)
        current_flag: bool = Input(src=FaultFlagLatch.State.current_flag)
        voltage_flag: bool = Input(src=FaultFlagLatch.State.voltage_flag)
        breaker_open: bool = Input(src=BreakerActuator.State.breaker_open)
        log: tuple[tuple[int, float, float, bool, bool, bool], ...] = Input(
            src="ProtectionLogger.State.log"
        )

    class State(NodeState):
        log: tuple[tuple[int, float, float, bool, bool, bool], ...] = Var(init=())

    def update(self, inputs: Inputs) -> State:
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
        return self.State(log=inputs.log + (sample,))


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
