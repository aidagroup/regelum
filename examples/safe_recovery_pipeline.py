from __future__ import annotations

from typing import Any

from regelum import (
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


class MicrogridSampler(Node):
    class Inputs(NodeInputs):
        tick: int = Input(src="MicrogridSampler.State.tick")

    class State(NodeState):
        tick: int = Var(init=0, domain=range(0, 4))
        bus_voltage_low: bool = Var(init=False)
        branch_current_high: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        tick = inputs.tick + 1
        return self.State(
            tick=tick,
            bus_voltage_low=True,
            branch_current_high=tick == 1,
        )


class StateEstimator(Node):
    class Inputs(NodeInputs):
        bus_voltage_low: bool = Input(src=MicrogridSampler.State.bus_voltage_low)
        branch_current_high: bool = Input(src=MicrogridSampler.State.branch_current_high)

    class State(NodeState):
        fault_detected: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        return self.State(fault_detected=inputs.bus_voltage_low or inputs.branch_current_high)


class NominalPolicy(Node):
    class Inputs(NodeInputs):
        fault_detected: bool = Input(src=StateEstimator.State.fault_detected)

    class State(NodeState):
        nominal_current_reference: int = Var(init=2, domain=range(0, 4))
        nominal_safe: bool = Var(init=True)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            nominal_current_reference=3,
            nominal_safe=not inputs.fault_detected,
        )


class FallbackPolicy(Node):
    class Inputs(NodeInputs):
        fault_detected: bool = Input(src=StateEstimator.State.fault_detected)

    class State(NodeState):
        fallback_current_reference: int = Var(init=0, domain=range(0, 4))
        fallback_safe: bool = Var(init=False)
        recovery_attempted: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            fallback_current_reference=1,
            fallback_safe=True,
            recovery_attempted=True,
        )


class ControlApplier(Node):
    class Inputs(NodeInputs):
        nominal_current_reference: int = Input(
            src=NominalPolicy.State.nominal_current_reference
        )
        nominal_safe: bool = Input(src=NominalPolicy.State.nominal_safe)
        fallback_current_reference: int = Input(
            src=FallbackPolicy.State.fallback_current_reference
        )
        fallback_safe: bool = Input(src=FallbackPolicy.State.fallback_safe)
        recovery_attempted: bool = Input(src=FallbackPolicy.State.recovery_attempted)

    class State(NodeState):
        applied_current_reference: int = Var(init=0, domain=range(0, 4))
        used_fallback: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        use_fallback = (
            not inputs.nominal_safe and inputs.recovery_attempted and inputs.fallback_safe
        )
        return self.State(
            applied_current_reference=(
                inputs.fallback_current_reference
                if use_fallback
                else inputs.nominal_current_reference
            ),
            used_fallback=use_fallback,
        )


class EmergencyShutdown(Node):
    class State(NodeState):
        shutdown_requested: bool = Var(init=False)

    def update(self, inputs: NodeInputs) -> State:
        return self.State(shutdown_requested=True)


class Monitor(Node):
    class Inputs(NodeInputs):
        applied_current_reference: int = Input(
            src=ControlApplier.State.applied_current_reference
        )
        shutdown_requested: bool = Input(src=EmergencyShutdown.State.shutdown_requested)

    class State(NodeState):
        committed: bool = Var(init=False)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            committed=inputs.shutdown_requested or inputs.applied_current_reference <= 1
        )


class RecoveryLogger(Node):
    class Inputs(NodeInputs):
        tick: int = Input(src=MicrogridSampler.State.tick)
        fault_detected: bool = Input(src=StateEstimator.State.fault_detected)
        nominal_safe: bool = Input(src=NominalPolicy.State.nominal_safe)
        recovery_attempted: bool = Input(src=FallbackPolicy.State.recovery_attempted)
        used_fallback: bool = Input(src=ControlApplier.State.used_fallback)
        committed: bool = Input(src=Monitor.State.committed)
        trace: tuple[tuple[int, bool, bool, bool, bool, bool], ...] = Input(
            src="RecoveryLogger.State.trace"
        )

    class State(NodeState):
        trace: tuple[tuple[int, bool, bool, bool, bool, bool], ...] = Var(init=())

    def update(self, inputs: Inputs) -> State:
        sample = (
            inputs.tick,
            inputs.fault_detected,
            inputs.nominal_safe,
            inputs.recovery_attempted,
            inputs.used_fallback,
            inputs.committed,
        )
        return self.State(trace=inputs.trace + (sample,))


def _effective_safe() -> Any:
    return V("NominalPolicy.nominal_safe") | (
        V("FallbackPolicy.recovery_attempted") & V("FallbackPolicy.fallback_safe")
    )


def _nodes() -> list[Node]:
    return [
        MicrogridSampler(),
        StateEstimator(),
        NominalPolicy(),
        FallbackPolicy(),
        ControlApplier(),
        EmergencyShutdown(),
        Monitor(),
        RecoveryLogger(),
    ]


def build_system() -> PhasedReactiveSystem:
    effective_safe = _effective_safe()
    needs_recovery = ~effective_safe & ~V("FallbackPolicy.recovery_attempted")
    unrecoverable = ~effective_safe & V("FallbackPolicy.recovery_attempted")
    (
        sampler,
        estimator,
        nominal_policy,
        fallback_policy,
        applier,
        shutdown,
        monitor,
        logger,
    ) = _nodes()

    return PhasedReactiveSystem(
        phases=[
            Phase(
                "sample",
                nodes=(sampler,),
                transitions=(Goto("estimate"),),
                is_initial=True,
            ),
            Phase("estimate", nodes=(estimator,), transitions=(Goto("policy"),)),
            Phase(
                "policy",
                nodes=(nominal_policy,),
                transitions=(Goto("safety-filter"),),
            ),
            Phase(
                "safety-filter",
                nodes=(nominal_policy, fallback_policy),
                transitions=(
                    If(effective_safe, "apply-control", name="safe"),
                    If(needs_recovery, "fallback-policy", name="recover"),
                    If(unrecoverable, "emergency-shutdown", name="shutdown"),
                ),
            ),
            Phase(
                "fallback-policy",
                nodes=(fallback_policy,),
                transitions=(
                    If(
                        V("FallbackPolicy.recovery_attempted"),
                        "safety-filter",
                        name="retry-once",
                    ),
                    If(
                        ~V("FallbackPolicy.recovery_attempted"),
                        "emergency-shutdown",
                        name="fallback-failed",
                    ),
                ),
            ),
            Phase(
                "apply-control",
                nodes=(applier,),
                transitions=(Goto("monitor"),),
            ),
            Phase(
                "emergency-shutdown",
                nodes=(shutdown,),
                transitions=(Goto("monitor"),),
            ),
            Phase("monitor", nodes=(monitor,), transitions=(Goto("log"),)),
            Phase("log", nodes=(logger,), transitions=(Goto(terminate),)),
        ],
    )


def main() -> None:
    system = build_system()
    print(f"compile={system.compile_report.ok}")
    records = system.step()
    print("phases=" + " -> ".join(record.phase for record in records))
    print(f"trace={system.snapshot()['RecoveryLogger.trace']}")


if __name__ == "__main__":
    main()
