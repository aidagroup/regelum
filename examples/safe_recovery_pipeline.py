from __future__ import annotations

from typing import Any

from regelum import (
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


class MicrogridSampler(Node):
    class Inputs(NodeInputs):
        tick: int = Input(source="MicrogridSampler.Outputs.tick")

    class Outputs(NodeOutputs):
        tick: int = Output(initial=0, domain=range(0, 4))
        bus_voltage_low: bool = Output(initial=False)
        branch_current_high: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        tick = inputs.tick + 1
        return self.Outputs(
            tick=tick,
            bus_voltage_low=True,
            branch_current_high=tick == 1,
        )


class StateEstimator(Node):
    class Inputs(NodeInputs):
        bus_voltage_low: bool = Input(source=MicrogridSampler.Outputs.bus_voltage_low)
        branch_current_high: bool = Input(
            source=MicrogridSampler.Outputs.branch_current_high
        )

    class Outputs(NodeOutputs):
        fault_detected: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            fault_detected=inputs.bus_voltage_low or inputs.branch_current_high
        )


class NominalPolicy(Node):
    class Inputs(NodeInputs):
        fault_detected: bool = Input(source=StateEstimator.Outputs.fault_detected)

    class Outputs(NodeOutputs):
        nominal_current_reference: int = Output(initial=2, domain=range(0, 4))
        nominal_safe: bool = Output(initial=True)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            nominal_current_reference=3,
            nominal_safe=not inputs.fault_detected,
        )


class FallbackPolicy(Node):
    class Inputs(NodeInputs):
        fault_detected: bool = Input(source=StateEstimator.Outputs.fault_detected)

    class Outputs(NodeOutputs):
        fallback_current_reference: int = Output(initial=0, domain=range(0, 4))
        fallback_safe: bool = Output(initial=False)
        recovery_attempted: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            fallback_current_reference=1,
            fallback_safe=True,
            recovery_attempted=True,
        )


class ControlApplier(Node):
    class Inputs(NodeInputs):
        nominal_current_reference: int = Input(
            source=NominalPolicy.Outputs.nominal_current_reference
        )
        nominal_safe: bool = Input(source=NominalPolicy.Outputs.nominal_safe)
        fallback_current_reference: int = Input(
            source=FallbackPolicy.Outputs.fallback_current_reference
        )
        fallback_safe: bool = Input(source=FallbackPolicy.Outputs.fallback_safe)
        recovery_attempted: bool = Input(source=FallbackPolicy.Outputs.recovery_attempted)

    class Outputs(NodeOutputs):
        applied_current_reference: int = Output(initial=0, domain=range(0, 4))
        used_fallback: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        use_fallback = (
            not inputs.nominal_safe
            and inputs.recovery_attempted
            and inputs.fallback_safe
        )
        return self.Outputs(
            applied_current_reference=(
                inputs.fallback_current_reference
                if use_fallback
                else inputs.nominal_current_reference
            ),
            used_fallback=use_fallback,
        )


class EmergencyShutdown(Node):
    class Outputs(NodeOutputs):
        shutdown_requested: bool = Output(initial=False)

    def run(self, inputs: NodeInputs) -> Outputs:
        return self.Outputs(shutdown_requested=True)


class Monitor(Node):
    class Inputs(NodeInputs):
        applied_current_reference: int = Input(
            source=ControlApplier.Outputs.applied_current_reference
        )
        shutdown_requested: bool = Input(source=EmergencyShutdown.Outputs.shutdown_requested)

    class Outputs(NodeOutputs):
        committed: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            committed=inputs.shutdown_requested or inputs.applied_current_reference <= 1
        )


class RecoveryLogger(Node):
    class Inputs(NodeInputs):
        tick: int = Input(source=MicrogridSampler.Outputs.tick)
        fault_detected: bool = Input(source=StateEstimator.Outputs.fault_detected)
        nominal_safe: bool = Input(source=NominalPolicy.Outputs.nominal_safe)
        recovery_attempted: bool = Input(source=FallbackPolicy.Outputs.recovery_attempted)
        used_fallback: bool = Input(source=ControlApplier.Outputs.used_fallback)
        committed: bool = Input(source=Monitor.Outputs.committed)
        trace: tuple[tuple[int, bool, bool, bool, bool, bool], ...] = Input(
            source="RecoveryLogger.Outputs.trace"
        )

    class Outputs(NodeOutputs):
        trace: tuple[tuple[int, bool, bool, bool, bool, bool], ...] = Output(initial=())

    def run(self, inputs: Inputs) -> Outputs:
        sample = (
            inputs.tick,
            inputs.fault_detected,
            inputs.nominal_safe,
            inputs.recovery_attempted,
            inputs.used_fallback,
            inputs.committed,
        )
        return self.Outputs(trace=inputs.trace + (sample,))


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
