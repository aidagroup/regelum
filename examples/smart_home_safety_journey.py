"""Smart-home safety pipeline -- iterative PRS design journey.

Inspired by Mohanty et al., "A Security-enabled Safety Assurance
Framework for IoT-based Smart Homes" (IEEE TIA, 2023). The paper
describes a 7-step design + 3-step runtime methodology for combining
several IoT safety subsystems (Fire Detection System, Leak Detection
System, ...), resolving cross-system conflicts (e.g. sprinkler vs
water valve), and emitting a coordinated action plan.

Maps onto a phased reactive system in five logical stages -- sense,
classify, attempt N command/verify rounds, finalise. This file walks
through four design iterations:

  v1: naive -- C1 cycle inside CLASSIFY (PriorityArbiter <-> RiskScorer)
  v2: C1 fixed -- semantic retry loop wires VERIFY back to COMMAND;
      compiler rejects with C2* (cycle is feasible under F_max)
  v3: tries a counter `attempts` to bound the retry cycle; under
      F_max the counter is unconstrained, compiler still rejects
  v4: retry unrolled into a chain of N attempt phases (N is a build
      hyperparameter); H is acyclic, system compiles and runs.

Run with: python -m examples.smart_home_safety_journey
"""

from __future__ import annotations

from typing import Any

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


# ────────────────────────────────────────────────────────────────────
# Domain encodings (small finite domains so C2* checks have teeth)
# ────────────────────────────────────────────────────────────────────

# FireState
FIRE_NORMAL = 0
FIRE_SMOKE_ONLY = 1
FIRE_CONFIRMED = 2
FIRE_SENSOR_FAULT = 3
FIRE_DOMAIN = (0, 1, 2, 3)

# LeakState
LEAK_NORMAL = 0
LEAK_DETECTED = 1
LEAK_SENSOR_FAULT = 2
LEAK_DOMAIN = (0, 1, 2)

# RiskLevel
RISK_LOW = 0
RISK_MEDIUM = 1
RISK_HIGH = 2
RISK_DOMAIN = (0, 1, 2)

# ActionPlan
ACT_NOOP = 0
ACT_ALARM_ONLY = 1
ACT_SPRINKLER_AND_ALARM = 2
ACT_CLOSE_VALVE = 3
ACT_FIRE_OVER_LEAK = 4
ACT_USER_ALERT = 5
ACTION_DOMAIN = (0, 1, 2, 3, 4, 5)


# ────────────────────────────────────────────────────────────────────
# SENSE phase nodes (4 independent sensors)
# ────────────────────────────────────────────────────────────────────


class SmokeSensor(Node):
    class Outputs(NodeOutputs):
        smoke_present: bool = Output(initial=False)
        smoke_health_ok: bool = Output(initial=True)

    def run(self, inputs: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(smoke_present=True, smoke_health_ok=True)


class HeatSensor(Node):
    class Outputs(NodeOutputs):
        heat_present: bool = Output(initial=False)
        heat_health_ok: bool = Output(initial=True)

    def run(self, inputs: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(heat_present=True, heat_health_ok=True)


class LeakSensor(Node):
    class Outputs(NodeOutputs):
        leak_present: bool = Output(initial=False)
        leak_health_ok: bool = Output(initial=True)

    def run(self, inputs: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(leak_present=True, leak_health_ok=True)


class TempSensor(Node):
    class Outputs(NodeOutputs):
        temperature_high: bool = Output(initial=False)
        temp_health_ok: bool = Output(initial=True)

    def run(self, inputs: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(temperature_high=True, temp_health_ok=True)


# ────────────────────────────────────────────────────────────────────
# CLASSIFY phase nodes (4-node DAG inside the phase)
# ────────────────────────────────────────────────────────────────────


class FireClassifier(Node):
    """Combines smoke, heat, temperature, and detector health into
    one of FIRE_NORMAL / FIRE_SMOKE_ONLY / FIRE_CONFIRMED /
    FIRE_SENSOR_FAULT."""

    class Inputs(NodeInputs):
        smoke_present: bool = Input(source="SmokeSensor.smoke_present")
        smoke_health_ok: bool = Input(source="SmokeSensor.smoke_health_ok")
        heat_present: bool = Input(source="HeatSensor.heat_present")
        heat_health_ok: bool = Input(source="HeatSensor.heat_health_ok")
        temperature_high: bool = Input(source="TempSensor.temperature_high")
        temp_health_ok: bool = Input(source="TempSensor.temp_health_ok")

    class Outputs(NodeOutputs):
        fire_state: int = Output(initial=FIRE_NORMAL, domain=FIRE_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        if not inputs.smoke_health_ok and not inputs.heat_health_ok:
            return self.Outputs(fire_state=FIRE_SENSOR_FAULT)
        temp_evidence = inputs.temperature_high and inputs.temp_health_ok
        if inputs.smoke_present and (inputs.heat_present or temp_evidence):
            return self.Outputs(fire_state=FIRE_CONFIRMED)
        if inputs.smoke_present:
            return self.Outputs(fire_state=FIRE_SMOKE_ONLY)
        return self.Outputs(fire_state=FIRE_NORMAL)


class LeakClassifier(Node):
    class Inputs(NodeInputs):
        leak_present: bool = Input(source="LeakSensor.leak_present")
        leak_health_ok: bool = Input(source="LeakSensor.leak_health_ok")

    class Outputs(NodeOutputs):
        leak_state: int = Output(initial=LEAK_NORMAL, domain=LEAK_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        if not inputs.leak_health_ok:
            return self.Outputs(leak_state=LEAK_SENSOR_FAULT)
        if inputs.leak_present:
            return self.Outputs(leak_state=LEAK_DETECTED)
        return self.Outputs(leak_state=LEAK_NORMAL)


class RiskScorer(Node):
    """Numerical risk score informed by both classifiers + temperature."""

    class Inputs(NodeInputs):
        fire_state: int = Input(source="FireClassifier.fire_state")
        leak_state: int = Input(source="LeakClassifier.leak_state")
        temperature_high: bool = Input(source="TempSensor.temperature_high")
        temp_health_ok: bool = Input(source="TempSensor.temp_health_ok")

    class Outputs(NodeOutputs):
        risk_level: int = Output(initial=RISK_LOW, domain=RISK_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        if inputs.fire_state == FIRE_CONFIRMED:
            return self.Outputs(risk_level=RISK_HIGH)
        if inputs.fire_state == FIRE_SMOKE_ONLY or inputs.leak_state == LEAK_DETECTED:
            return self.Outputs(risk_level=RISK_MEDIUM)
        if inputs.temperature_high and inputs.temp_health_ok:
            return self.Outputs(risk_level=RISK_MEDIUM)
        return self.Outputs(risk_level=RISK_LOW)


class PriorityArbiter(Node):
    """Resolves cross-system conflicts (FDS-vs-LDS, table III in
    Mohanty 2023). Picks one ActionPlan."""

    class Inputs(NodeInputs):
        fire_state: int = Input(source="FireClassifier.fire_state")
        leak_state: int = Input(source="LeakClassifier.leak_state")
        risk_level: int = Input(source="RiskScorer.risk_level")

    class Outputs(NodeOutputs):
        action_plan: int = Output(initial=ACT_NOOP, domain=ACTION_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        if inputs.fire_state == FIRE_SENSOR_FAULT or inputs.leak_state == LEAK_SENSOR_FAULT:
            return self.Outputs(action_plan=ACT_USER_ALERT)
        if inputs.fire_state == FIRE_CONFIRMED and inputs.leak_state == LEAK_DETECTED:
            return self.Outputs(action_plan=ACT_FIRE_OVER_LEAK)
        if inputs.fire_state == FIRE_CONFIRMED:
            return self.Outputs(action_plan=ACT_SPRINKLER_AND_ALARM)
        if inputs.leak_state == LEAK_DETECTED:
            return self.Outputs(action_plan=ACT_CLOSE_VALVE)
        if inputs.fire_state == FIRE_SMOKE_ONLY or inputs.risk_level == RISK_MEDIUM:
            return self.Outputs(action_plan=ACT_ALARM_ONLY)
        return self.Outputs(action_plan=ACT_NOOP)


# v1 buggy variants: PriorityArbiter reads RiskScorer (forward edge),
# RiskScorer also reads PriorityArbiter's action_plan (back edge).
# Both live in the same CLASSIFY phase => C1 dependency cycle.
class PriorityArbiterV1(Node):
    class Inputs(NodeInputs):
        fire_state: int = Input(source="FireClassifier.fire_state")
        leak_state: int = Input(source="LeakClassifier.leak_state")
        risk_level: int = Input(source="RiskScorerV1.risk_level")

    class Outputs(NodeOutputs):
        action_plan: int = Output(initial=ACT_NOOP, domain=ACTION_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        return PriorityArbiter().run(inputs)  # type: ignore[arg-type]


class RiskScorerV1(Node):
    """Buggy v1 RiskScorer: also reads action_plan ("feedback-aware").
    Creates a within-phase dependency cycle PriorityArbiterV1 <-> RiskScorerV1."""

    class Inputs(NodeInputs):
        fire_state: int = Input(source="FireClassifier.fire_state")
        leak_state: int = Input(source="LeakClassifier.leak_state")
        temperature_high: bool = Input(source="TempSensor.temperature_high")
        last_plan: int = Input(source="PriorityArbiterV1.action_plan")

    class Outputs(NodeOutputs):
        risk_level: int = Output(initial=RISK_LOW, domain=RISK_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        return RiskScorer().run(inputs)  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────
# COMMAND phase nodes (4 independent actuators)
# ────────────────────────────────────────────────────────────────────


class SprinklerCmd(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="PriorityArbiter.action_plan")

    class Outputs(NodeOutputs):
        sprinkler_on: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        on = inputs.action_plan in (ACT_SPRINKLER_AND_ALARM, ACT_FIRE_OVER_LEAK)
        return self.Outputs(sprinkler_on=on)


class ValveCmd(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="PriorityArbiter.action_plan")

    class Outputs(NodeOutputs):
        valve_closed: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        closed = inputs.action_plan in (ACT_CLOSE_VALVE, ACT_FIRE_OVER_LEAK)
        return self.Outputs(valve_closed=closed)


class AlarmCmd(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="PriorityArbiter.action_plan")

    class Outputs(NodeOutputs):
        alarm_on: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        on = inputs.action_plan in (
            ACT_ALARM_ONLY,
            ACT_SPRINKLER_AND_ALARM,
            ACT_FIRE_OVER_LEAK,
            ACT_USER_ALERT,
        )
        return self.Outputs(alarm_on=on)


class NotifCmd(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="PriorityArbiter.action_plan")

    class Outputs(NodeOutputs):
        notif_sent: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(notif_sent=inputs.action_plan != ACT_NOOP)


# ────────────────────────────────────────────────────────────────────
# VERIFY phase nodes (4 ack checks, one per actuator)
# ────────────────────────────────────────────────────────────────────


class SprinklerVerify(Node):
    class Inputs(NodeInputs):
        sprinkler_on: bool = Input(source="SprinklerCmd.sprinkler_on")

    class Outputs(NodeOutputs):
        sprinkler_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(sprinkler_acked=True)


class ValveVerify(Node):
    class Inputs(NodeInputs):
        valve_closed: bool = Input(source="ValveCmd.valve_closed")

    class Outputs(NodeOutputs):
        valve_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(valve_acked=True)


class AlarmVerify(Node):
    class Inputs(NodeInputs):
        alarm_on: bool = Input(source="AlarmCmd.alarm_on")

    class Outputs(NodeOutputs):
        alarm_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(alarm_acked=True)


class NotifVerify(Node):
    class Inputs(NodeInputs):
        notif_sent: bool = Input(source="NotifCmd.notif_sent")

    class Outputs(NodeOutputs):
        notif_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(notif_acked=True)


# Helper retry counter for v3
class AttemptCounter(Node):
    """Increments a retry counter. Lives in COMMAND phase for v3."""

    class Inputs(NodeInputs):
        attempts: int = Input(source="AttemptCounter.attempts")

    class Outputs(NodeOutputs):
        attempts: int = Output(initial=0, domain=(0, 1, 2, 3))

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(attempts=min(inputs.attempts + 1, 3))


# ────────────────────────────────────────────────────────────────────
# Common predicates
# ────────────────────────────────────────────────────────────────────

ALL_ACKED = (
    V(SprinklerVerify.Outputs.sprinkler_acked)
    & V(ValveVerify.Outputs.valve_acked)
    & V(AlarmVerify.Outputs.alarm_acked)
    & V(NotifVerify.Outputs.notif_acked)
)


def _sense_phase(
    *,
    target: str,
    smoke: SmokeSensor,
    heat: HeatSensor,
    leak: LeakSensor,
    temp: TempSensor,
) -> Phase:
    return Phase(
        "sense",
        nodes=(smoke, heat, leak, temp),
        transitions=(Goto(target),),
        is_initial=True,
    )


# ────────────────────────────────────────────────────────────────────
# v1: naive -- C1 cycle inside CLASSIFY
# ────────────────────────────────────────────────────────────────────


def build_v1() -> PhasedReactiveSystem:
    """v1 prototypes only the analytics half of the pipeline. The
    developer wants a 'feedback-aware' RiskScorer that takes last
    tick's action_plan into account -- and at the same time the
    PriorityArbiter reads risk_level. Both nodes live in CLASSIFY,
    so they form a within-phase dependency cycle. C1 fails."""
    smoke, heat, leak, temp = SmokeSensor(), HeatSensor(), LeakSensor(), TempSensor()
    fire, leak_classifier = FireClassifier(), LeakClassifier()
    risk, arbiter = RiskScorerV1(), PriorityArbiterV1()
    return PhasedReactiveSystem(
        phases=[
            _sense_phase(target="classify", smoke=smoke, heat=heat, leak=leak, temp=temp),
            Phase(
                "classify",
                nodes=(fire, leak_classifier, risk, arbiter),
                transitions=(Goto(terminate),),
            ),
        ],
    )


# ────────────────────────────────────────────────────────────────────
# v2: C1 fixed -- but adds a real retry cycle in H. Breaks C2*.
# ────────────────────────────────────────────────────────────────────


def build_v2() -> PhasedReactiveSystem:
    """v2 fixes the C1 cycle by removing RiskScorer's read of the
    arbiter. It then adds a retry edge VERIFY -> COMMAND so failed
    actuator acks loop back. Under F_max the verifiers can claim
    'not acked' forever, so C2* fails."""
    smoke, heat, leak, temp = SmokeSensor(), HeatSensor(), LeakSensor(), TempSensor()
    fire, leak_classifier = FireClassifier(), LeakClassifier()
    risk, arbiter = RiskScorer(), PriorityArbiter()
    sprinkler_cmd, valve_cmd, alarm_cmd, notif_cmd = SprinklerCmd(), ValveCmd(), AlarmCmd(), NotifCmd()
    sprinkler_verify, valve_verify, alarm_verify, notif_verify = (
        SprinklerVerify(),
        ValveVerify(),
        AlarmVerify(),
        NotifVerify(),
    )
    return PhasedReactiveSystem(
        phases=[
            _sense_phase(target="classify", smoke=smoke, heat=heat, leak=leak, temp=temp),
            Phase(
                "classify",
                nodes=(fire, leak_classifier, risk, arbiter),
                transitions=(Goto("command"),),
            ),
            Phase(
                "command",
                nodes=(sprinkler_cmd, valve_cmd, alarm_cmd, notif_cmd),
                transitions=(Goto("verify"),),
            ),
            Phase(
                "verify",
                nodes=(sprinkler_verify, valve_verify, alarm_verify, notif_verify),
                transitions=(
                    If(ALL_ACKED, terminate, name="ok"),
                    If(~ALL_ACKED, "command", name="retry"),
                ),
            ),
        ],
    )


# ────────────────────────────────────────────────────────────────────
# v3: counter-bounded retry. Still fails C2* under F_max.
# ────────────────────────────────────────────────────────────────────


def build_v3() -> PhasedReactiveSystem:
    """v3 adds an `attempts` counter to bound the retry cycle.
    Intuitively: 'retry only while attempts < 3'. Under F_max the
    AttemptCounter can write attempts := 0 every time, so the cycle
    is still feasible -- C2* rejects with a witness."""
    attempts_lt_3 = V(AttemptCounter.Outputs.attempts) < 3
    cycle_back = (~ALL_ACKED) & attempts_lt_3
    finalise = ALL_ACKED | (~attempts_lt_3)
    smoke, heat, leak, temp = SmokeSensor(), HeatSensor(), LeakSensor(), TempSensor()
    fire, leak_classifier = FireClassifier(), LeakClassifier()
    risk, arbiter = RiskScorer(), PriorityArbiter()
    sprinkler_cmd, valve_cmd, alarm_cmd, notif_cmd = SprinklerCmd(), ValveCmd(), AlarmCmd(), NotifCmd()
    counter = AttemptCounter()
    sprinkler_verify, valve_verify, alarm_verify, notif_verify = (
        SprinklerVerify(),
        ValveVerify(),
        AlarmVerify(),
        NotifVerify(),
    )
    return PhasedReactiveSystem(
        phases=[
            _sense_phase(target="classify", smoke=smoke, heat=heat, leak=leak, temp=temp),
            Phase(
                "classify",
                nodes=(fire, leak_classifier, risk, arbiter),
                transitions=(Goto("command"),),
            ),
            Phase(
                "command",
                nodes=(sprinkler_cmd, valve_cmd, alarm_cmd, notif_cmd, counter),
                transitions=(Goto("verify"),),
            ),
            Phase(
                "verify",
                nodes=(sprinkler_verify, valve_verify, alarm_verify, notif_verify),
                transitions=(
                    If(finalise, terminate, name="ok-or-give-up"),
                    If(cycle_back, "command", name="retry"),
                ),
            ),
        ],
    )


# ────────────────────────────────────────────────────────────────────
# v4: retry unrolled into a chain. H acyclic, C2* vacuous.
# ────────────────────────────────────────────────────────────────────


def build_v4(max_attempts: int = 3) -> PhasedReactiveSystem:
    """v4 unrolls the retry into `max_attempts` independent
    command/verify rounds. The phase chain is

      sense -> classify ->
        command_1 -> verify_1 ->
          (ok? done : command_2 -> verify_2 ->
            (ok? done : command_3 -> verify_3 -> done))

    H is a DAG, C2 holds vacuously (no cycles), C2* is trivial. The
    retry budget is encoded in graph topology, not in any runtime
    counter, so F_max cannot defeat it. `max_attempts` is a build-
    time hyperparameter."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    smoke, heat, leak, temp = SmokeSensor(), HeatSensor(), LeakSensor(), TempSensor()
    fire, leak_classifier = FireClassifier(), LeakClassifier()
    risk, arbiter = RiskScorer(), PriorityArbiter()
    sprinkler_cmd, valve_cmd, alarm_cmd, notif_cmd = SprinklerCmd(), ValveCmd(), AlarmCmd(), NotifCmd()
    sprinkler_verify, valve_verify, alarm_verify, notif_verify = (
        SprinklerVerify(),
        ValveVerify(),
        AlarmVerify(),
        NotifVerify(),
    )

    phases: list[Phase] = [
        _sense_phase(target="classify", smoke=smoke, heat=heat, leak=leak, temp=temp),
        Phase(
            "classify",
            nodes=(fire, leak_classifier, risk, arbiter),
            transitions=(Goto("command_1"),),
        ),
    ]

    for index in range(1, max_attempts + 1):
        is_last = index == max_attempts
        next_command = None if is_last else f"command_{index + 1}"
        phases.append(
            Phase(
                f"command_{index}",
                nodes=(sprinkler_cmd, valve_cmd, alarm_cmd, notif_cmd),
                transitions=(Goto(f"verify_{index}"),),
            )
        )
        verify_transitions: tuple
        if is_last:
            verify_transitions = (Goto(terminate),)
        else:
            verify_transitions = (
                If(ALL_ACKED, terminate, name="ok"),
                If(~ALL_ACKED, next_command, name=f"retry-{index + 1}"),
            )
        phases.append(
            Phase(
                f"verify_{index}",
                nodes=(sprinkler_verify, valve_verify, alarm_verify, notif_verify),
                transitions=verify_transitions,
            )
        )

    return PhasedReactiveSystem(phases=phases)


# ────────────────────────────────────────────────────────────────────
# main: walk through the four iterations
# ────────────────────────────────────────────────────────────────────


def _print_compile_attempt(
    version: str, builder: Any, *args: Any, **kwargs: Any
) -> PhasedReactiveSystem | None:
    print(f"\n=== {version} ===")
    try:
        system = builder(*args, **kwargs)
    except CompileError as exc:
        print(f"compile FAILED with {len(exc.report.issues)} issue(s):")
        for issue in exc.report.issues:
            print(f"  - [{issue.location}]")
            print(f"      {issue.message}")
        return None
    print(f"compile OK -- {len(system.phases)} phases, "
          f"{len(system.nodes)} nodes")
    return system


def main() -> None:
    print("Smart-home safety pipeline -- PRS design journey")
    print("(inspired by Mohanty et al., IEEE TIA 2023)")

    _print_compile_attempt("v1: naive (C1 trap)", build_v1)
    _print_compile_attempt("v2: retry loop (C2* trap)", build_v2)
    _print_compile_attempt("v3: counter-bounded retry (C2* trap)", build_v3)
    system = _print_compile_attempt(
        "v4: unrolled retry chain", build_v4, max_attempts=3
    )

    if system is None:
        return

    print("\n=== runtime trace of v4 (one tick) ===")
    records = system.step()
    phase_visits: list[str] = []
    for record in records:
        if not phase_visits or phase_visits[-1] != record.phase:
            phase_visits.append(record.phase)
    snap = system.snapshot()
    print(f"phase trace: {' -> '.join(phase_visits)} -> stop")
    print(f"  fire_state   = {snap['FireClassifier.fire_state']}")
    print(f"  leak_state   = {snap['LeakClassifier.leak_state']}")
    print(f"  risk_level   = {snap['RiskScorer.risk_level']}")
    print(f"  action_plan  = {snap['PriorityArbiter.action_plan']}")
    print(f"  sprinkler_on = {snap['SprinklerCmd.sprinkler_on']}")
    print(f"  valve_closed = {snap['ValveCmd.valve_closed']}")
    print(f"  alarm_on     = {snap['AlarmCmd.alarm_on']}")
    print(f"  notif_sent   = {snap['NotifCmd.notif_sent']}")


if __name__ == "__main__":
    main()
