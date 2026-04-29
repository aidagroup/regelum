"""Smart-home pipeline extended with async / network / power oracles.

Builds on `smart_home_safety_journey` (Mohanty 2023). Models the
real-world async surface around a smart-home hub via *oracle nodes*:
service nodes whose only job is to expose a boolean variable that
under F_max represents the worst-case behaviour of an environmental
factor. This lets PRS verify that the safety pipeline is structurally
sound even under arbitrary network drops, late acks, sensor-bus
glitches and cloud partitions.

Phases:
  env       -- network/power/cloud oracles (reroll every tick)
  sense     -- physical sensor nodes
  staging   -- channel gates: combine sensor health + transport
               oracles into trichotomised evidence (NEG/POS/UNKNOWN)
  classify  -- 4-node DAG, reads only evidence enums
  attempt_k -- (command, verify) pair with per-attempt ack oracles;
               unrolled into a chain of `max_attempts` rounds, so the
               retry budget is structural

Run with: python -m examples.smart_home_async_world
"""

from __future__ import annotations

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

from .smart_home_safety_journey import (
    ACT_ALARM_ONLY,
    ACT_CLOSE_VALVE,
    ACT_FIRE_OVER_LEAK,
    ACT_NOOP,
    ACT_SPRINKLER_AND_ALARM,
    ACT_USER_ALERT,
    ACTION_DOMAIN,
    HeatSensor,
    LeakSensor,
    SmokeSensor,
    TempSensor,
)

# ────────────────────────────────────────────────────────────────────
# Trichotomised evidence domain
# ────────────────────────────────────────────────────────────────────

EV_NEGATIVE = 0
EV_POSITIVE = 1
EV_UNKNOWN = 2
EV_DOMAIN = (0, 1, 2)


# ────────────────────────────────────────────────────────────────────
# ENV phase: oracle nodes for the environment.
# Each one writes a single bool. Under F_max the compiler explores
# every truth-table of these oracles -- every combination of 'network
# up/down', 'cloud reachable / not', 'sensor bus ok / not', etc.
# ────────────────────────────────────────────────────────────────────


class SensorBusOracle(Node):
    """True iff the sensor bus delivered fresh readings this tick."""

    class Outputs(NodeOutputs):
        sensor_bus_fresh: bool = Output(initial=True)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(sensor_bus_fresh=True)


class ActuatorNetOracle(Node):
    """True iff the actuator network is reachable in this attempt."""

    class Outputs(NodeOutputs):
        actuators_reachable: bool = Output(initial=True)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(actuators_reachable=True)


class CloudOracle(Node):
    class Outputs(NodeOutputs):
        cloud_up: bool = Output(initial=True)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(cloud_up=True)


class PowerOracle(Node):
    class Outputs(NodeOutputs):
        power_stable: bool = Output(initial=True)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(power_stable=True)


# ────────────────────────────────────────────────────────────────────
# STAGING phase: gates that turn (raw sensor, oracle) -> evidence enum
# A gate produces UNKNOWN whenever any reason to distrust shows up.
# ────────────────────────────────────────────────────────────────────


def _gate_evidence(present: bool, healthy: bool) -> int:
    if not healthy:
        return EV_UNKNOWN
    return EV_POSITIVE if present else EV_NEGATIVE


class SmokeGate(Node):
    class Inputs(NodeInputs):
        smoke_present: bool = Input(source="SmokeSensor.smoke_present")
        smoke_health_ok: bool = Input(source="SmokeSensor.smoke_health_ok")
        sensor_bus_fresh: bool = Input(source="SensorBusOracle.sensor_bus_fresh")
        power_stable: bool = Input(source="PowerOracle.power_stable")

    class Outputs(NodeOutputs):
        smoke_ev: int = Output(initial=EV_NEGATIVE, domain=EV_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        healthy = inputs.smoke_health_ok and inputs.sensor_bus_fresh and inputs.power_stable
        return self.Outputs(smoke_ev=_gate_evidence(inputs.smoke_present, healthy))


class HeatGate(Node):
    class Inputs(NodeInputs):
        heat_present: bool = Input(source="HeatSensor.heat_present")
        heat_health_ok: bool = Input(source="HeatSensor.heat_health_ok")
        sensor_bus_fresh: bool = Input(source="SensorBusOracle.sensor_bus_fresh")
        power_stable: bool = Input(source="PowerOracle.power_stable")

    class Outputs(NodeOutputs):
        heat_ev: int = Output(initial=EV_NEGATIVE, domain=EV_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        healthy = inputs.heat_health_ok and inputs.sensor_bus_fresh and inputs.power_stable
        return self.Outputs(heat_ev=_gate_evidence(inputs.heat_present, healthy))


class LeakGate(Node):
    class Inputs(NodeInputs):
        leak_present: bool = Input(source="LeakSensor.leak_present")
        leak_health_ok: bool = Input(source="LeakSensor.leak_health_ok")
        sensor_bus_fresh: bool = Input(source="SensorBusOracle.sensor_bus_fresh")
        power_stable: bool = Input(source="PowerOracle.power_stable")

    class Outputs(NodeOutputs):
        leak_ev: int = Output(initial=EV_NEGATIVE, domain=EV_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        healthy = inputs.leak_health_ok and inputs.sensor_bus_fresh and inputs.power_stable
        return self.Outputs(leak_ev=_gate_evidence(inputs.leak_present, healthy))


class TempGate(Node):
    class Inputs(NodeInputs):
        temperature_high: bool = Input(source="TempSensor.temperature_high")
        temp_health_ok: bool = Input(source="TempSensor.temp_health_ok")
        sensor_bus_fresh: bool = Input(source="SensorBusOracle.sensor_bus_fresh")
        power_stable: bool = Input(source="PowerOracle.power_stable")

    class Outputs(NodeOutputs):
        temp_ev: int = Output(initial=EV_NEGATIVE, domain=EV_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        healthy = inputs.temp_health_ok and inputs.sensor_bus_fresh and inputs.power_stable
        return self.Outputs(temp_ev=_gate_evidence(inputs.temperature_high, healthy))


# ────────────────────────────────────────────────────────────────────
# CLASSIFY phase: async-aware versions read the evidence enums only
# ────────────────────────────────────────────────────────────────────


# Use the same domains as the journey example to keep semantics aligned
FIRE_NORMAL = 0
FIRE_SMOKE_ONLY = 1
FIRE_CONFIRMED = 2
FIRE_SENSOR_FAULT = 3
FIRE_DOMAIN = (0, 1, 2, 3)

LEAK_NORMAL = 0
LEAK_DETECTED = 1
LEAK_SENSOR_FAULT = 2
LEAK_DOMAIN = (0, 1, 2)

RISK_LOW = 0
RISK_MEDIUM = 1
RISK_HIGH = 2
RISK_DOMAIN = (0, 1, 2)


class AsyncFireClassifier(Node):
    class Inputs(NodeInputs):
        smoke_ev: int = Input(source="SmokeGate.smoke_ev")
        heat_ev: int = Input(source="HeatGate.heat_ev")
        temp_ev: int = Input(source="TempGate.temp_ev")

    class Outputs(NodeOutputs):
        fire_state: int = Output(initial=FIRE_NORMAL, domain=FIRE_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        smoke_unknown = inputs.smoke_ev == EV_UNKNOWN
        heat_unknown = inputs.heat_ev == EV_UNKNOWN
        if smoke_unknown and heat_unknown:
            return self.Outputs(fire_state=FIRE_SENSOR_FAULT)
        smoke_pos = inputs.smoke_ev == EV_POSITIVE
        heat_pos = inputs.heat_ev == EV_POSITIVE
        temp_pos = inputs.temp_ev == EV_POSITIVE  # UNKNOWN excluded
        if smoke_pos and (heat_pos or temp_pos):
            return self.Outputs(fire_state=FIRE_CONFIRMED)
        if smoke_pos:
            return self.Outputs(fire_state=FIRE_SMOKE_ONLY)
        return self.Outputs(fire_state=FIRE_NORMAL)


class AsyncLeakClassifier(Node):
    class Inputs(NodeInputs):
        leak_ev: int = Input(source="LeakGate.leak_ev")

    class Outputs(NodeOutputs):
        leak_state: int = Output(initial=LEAK_NORMAL, domain=LEAK_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        if inputs.leak_ev == EV_UNKNOWN:
            return self.Outputs(leak_state=LEAK_SENSOR_FAULT)
        if inputs.leak_ev == EV_POSITIVE:
            return self.Outputs(leak_state=LEAK_DETECTED)
        return self.Outputs(leak_state=LEAK_NORMAL)


class AsyncRiskScorer(Node):
    class Inputs(NodeInputs):
        fire_state: int = Input(source="AsyncFireClassifier.fire_state")
        leak_state: int = Input(source="AsyncLeakClassifier.leak_state")
        temp_ev: int = Input(source="TempGate.temp_ev")

    class Outputs(NodeOutputs):
        risk_level: int = Output(initial=RISK_LOW, domain=RISK_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        if inputs.fire_state == FIRE_CONFIRMED:
            return self.Outputs(risk_level=RISK_HIGH)
        if inputs.fire_state == FIRE_SMOKE_ONLY or inputs.leak_state == LEAK_DETECTED:
            return self.Outputs(risk_level=RISK_MEDIUM)
        if inputs.temp_ev == EV_POSITIVE:
            return self.Outputs(risk_level=RISK_MEDIUM)
        return self.Outputs(risk_level=RISK_LOW)


class AsyncPriorityArbiter(Node):
    class Inputs(NodeInputs):
        fire_state: int = Input(source="AsyncFireClassifier.fire_state")
        leak_state: int = Input(source="AsyncLeakClassifier.leak_state")
        risk_level: int = Input(source="AsyncRiskScorer.risk_level")
        cloud_up: bool = Input(source="CloudOracle.cloud_up")

    class Outputs(NodeOutputs):
        action_plan: int = Output(initial=ACT_NOOP, domain=ACTION_DOMAIN)

    def run(self, inputs: Inputs) -> Outputs:
        # If both safety subsystems are blind, alert the user via local alarm only;
        # cloud may not even reach the operator.
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
        # Cloud-only telemetry mode: nothing to do locally
        _ = inputs.cloud_up
        return self.Outputs(action_plan=ACT_NOOP)


# ────────────────────────────────────────────────────────────────────
# Per-actuator ack oracles + async-aware verifiers
# Each oracle is a dedicated node owning one bool. Visited inside each
# verify_i phase, so under F_max its output rerolls per attempt.
# ────────────────────────────────────────────────────────────────────


class SprinklerAckOracle(Node):
    class Outputs(NodeOutputs):
        sprinkler_ack: bool = Output(initial=False)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(sprinkler_ack=True)


class ValveAckOracle(Node):
    class Outputs(NodeOutputs):
        valve_ack: bool = Output(initial=False)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(valve_ack=True)


class AlarmAckOracle(Node):
    class Outputs(NodeOutputs):
        alarm_ack: bool = Output(initial=False)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(alarm_ack=True)


class NotifAckOracle(Node):
    class Outputs(NodeOutputs):
        notif_ack: bool = Output(initial=False)

    def run(self, _: NodeInputs) -> Outputs:  # noqa: ARG002
        return self.Outputs(notif_ack=True)


class SprinklerVerifyAsync(Node):
    class Inputs(NodeInputs):
        sprinkler_on: bool = Input(source="SprinklerCmdAsync.sprinkler_on")
        actuators_reachable: bool = Input(source="ActuatorNetOracle.actuators_reachable")
        sprinkler_ack: bool = Input(source="SprinklerAckOracle.sprinkler_ack")

    class Outputs(NodeOutputs):
        sprinkler_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        ok = inputs.sprinkler_on and inputs.actuators_reachable and inputs.sprinkler_ack
        return self.Outputs(sprinkler_acked=ok)


class ValveVerifyAsync(Node):
    class Inputs(NodeInputs):
        valve_closed: bool = Input(source="ValveCmdAsync.valve_closed")
        actuators_reachable: bool = Input(source="ActuatorNetOracle.actuators_reachable")
        valve_ack: bool = Input(source="ValveAckOracle.valve_ack")

    class Outputs(NodeOutputs):
        valve_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        ok = inputs.valve_closed and inputs.actuators_reachable and inputs.valve_ack
        return self.Outputs(valve_acked=ok)


class AlarmVerifyAsync(Node):
    class Inputs(NodeInputs):
        alarm_on: bool = Input(source="AlarmCmdAsync.alarm_on")
        actuators_reachable: bool = Input(source="ActuatorNetOracle.actuators_reachable")
        alarm_ack: bool = Input(source="AlarmAckOracle.alarm_ack")

    class Outputs(NodeOutputs):
        alarm_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        ok = inputs.alarm_on and inputs.actuators_reachable and inputs.alarm_ack
        return self.Outputs(alarm_acked=ok)


class NotifVerifyAsync(Node):
    class Inputs(NodeInputs):
        notif_sent: bool = Input(source="NotifCmdAsync.notif_sent")
        cloud_up: bool = Input(source="CloudOracle.cloud_up")
        notif_ack: bool = Input(source="NotifAckOracle.notif_ack")

    class Outputs(NodeOutputs):
        notif_acked: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        ok = inputs.notif_sent and inputs.cloud_up and inputs.notif_ack
        return self.Outputs(notif_acked=ok)


# Async-aware actuator commanders. Standalone classes (not subclasses
# of the synchronous ones) so output paths are SprinklerCmdAsync.* etc.
class SprinklerCmdAsync(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="AsyncPriorityArbiter.action_plan")

    class Outputs(NodeOutputs):
        sprinkler_on: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        on = inputs.action_plan in (ACT_SPRINKLER_AND_ALARM, ACT_FIRE_OVER_LEAK)
        return self.Outputs(sprinkler_on=on)


class ValveCmdAsync(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="AsyncPriorityArbiter.action_plan")

    class Outputs(NodeOutputs):
        valve_closed: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        closed = inputs.action_plan in (ACT_CLOSE_VALVE, ACT_FIRE_OVER_LEAK)
        return self.Outputs(valve_closed=closed)


class AlarmCmdAsync(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="AsyncPriorityArbiter.action_plan")

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


class NotifCmdAsync(Node):
    class Inputs(NodeInputs):
        action_plan: int = Input(source="AsyncPriorityArbiter.action_plan")

    class Outputs(NodeOutputs):
        notif_sent: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(notif_sent=inputs.action_plan != ACT_NOOP)


# ────────────────────────────────────────────────────────────────────
# Build pipeline
# ────────────────────────────────────────────────────────────────────


ALL_ACKED = (
    V(SprinklerVerifyAsync.Outputs.sprinkler_acked)
    & V(ValveVerifyAsync.Outputs.valve_acked)
    & V(AlarmVerifyAsync.Outputs.alarm_acked)
    & V(NotifVerifyAsync.Outputs.notif_acked)
)


def build_async_world(max_attempts: int = 3) -> PhasedReactiveSystem:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    nodes = [
        # env oracles
        SensorBusOracle(),
        ActuatorNetOracle(),
        CloudOracle(),
        PowerOracle(),
        # physical sensors
        SmokeSensor(),
        HeatSensor(),
        LeakSensor(),
        TempSensor(),
        # gates
        SmokeGate(),
        HeatGate(),
        LeakGate(),
        TempGate(),
        # classify
        AsyncFireClassifier(),
        AsyncLeakClassifier(),
        AsyncRiskScorer(),
        AsyncPriorityArbiter(),
        # actuators
        SprinklerCmdAsync(),
        ValveCmdAsync(),
        AlarmCmdAsync(),
        NotifCmdAsync(),
        # ack oracles
        SprinklerAckOracle(),
        ValveAckOracle(),
        AlarmAckOracle(),
        NotifAckOracle(),
        # verifiers
        SprinklerVerifyAsync(),
        ValveVerifyAsync(),
        AlarmVerifyAsync(),
        NotifVerifyAsync(),
    ]
    (
        sensor_bus,
        actuator_net,
        cloud,
        power,
        smoke,
        heat,
        leak,
        temp,
        smoke_gate,
        heat_gate,
        leak_gate,
        temp_gate,
        fire_classifier,
        leak_classifier,
        risk_scorer,
        arbiter,
        sprinkler_cmd,
        valve_cmd,
        alarm_cmd,
        notif_cmd,
        sprinkler_ack,
        valve_ack,
        alarm_ack,
        notif_ack,
        sprinkler_verify,
        valve_verify,
        alarm_verify,
        notif_verify,
    ) = nodes

    phases: list[Phase] = [
        Phase(
            "env",
            nodes=(sensor_bus, actuator_net, cloud, power),
            transitions=(Goto("sense"),),
            is_initial=True,
        ),
        Phase(
            "sense",
            nodes=(smoke, heat, leak, temp),
            transitions=(Goto("staging"),),
        ),
        Phase(
            "staging",
            nodes=(smoke_gate, heat_gate, leak_gate, temp_gate),
            transitions=(Goto("classify"),),
        ),
        Phase(
            "classify",
            nodes=(
                fire_classifier,
                leak_classifier,
                risk_scorer,
                arbiter,
            ),
            transitions=(Goto("command_1"),),
        ),
    ]

    for index in range(1, max_attempts + 1):
        is_last = index == max_attempts
        next_command = None if is_last else f"command_{index + 1}"
        phases.append(
            Phase(
                f"command_{index}",
                nodes=(
                    sprinkler_cmd,
                    valve_cmd,
                    alarm_cmd,
                    notif_cmd,
                ),
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
                nodes=(
                    # ack oracles re-roll on every verify visit
                    sprinkler_ack,
                    valve_ack,
                    alarm_ack,
                    notif_ack,
                    sprinkler_verify,
                    valve_verify,
                    alarm_verify,
                    notif_verify,
                ),
                transitions=verify_transitions,
            )
        )

    return PhasedReactiveSystem(phases=phases)


# ────────────────────────────────────────────────────────────────────
# Cycle-aware variant: same nodes, but with a *real* retry edge in H
# instead of structural unrolling. Demonstrates how oracles trigger
# C2* failures even when intent is "wait for the network".
# ────────────────────────────────────────────────────────────────────


def build_async_world_with_cycle() -> PhasedReactiveSystem:
    """Same async modelling but uses a back-edge verify -> command_1.
    Compiler rejects with C2* showing that under a permanent network
    outage the retry never terminates."""
    nodes = [
        SensorBusOracle(),
        ActuatorNetOracle(),
        CloudOracle(),
        PowerOracle(),
        SmokeSensor(),
        HeatSensor(),
        LeakSensor(),
        TempSensor(),
        SmokeGate(),
        HeatGate(),
        LeakGate(),
        TempGate(),
        AsyncFireClassifier(),
        AsyncLeakClassifier(),
        AsyncRiskScorer(),
        AsyncPriorityArbiter(),
        SprinklerCmdAsync(),
        ValveCmdAsync(),
        AlarmCmdAsync(),
        NotifCmdAsync(),
        SprinklerAckOracle(),
        ValveAckOracle(),
        AlarmAckOracle(),
        NotifAckOracle(),
        SprinklerVerifyAsync(),
        ValveVerifyAsync(),
        AlarmVerifyAsync(),
        NotifVerifyAsync(),
    ]
    (
        sensor_bus,
        actuator_net,
        cloud,
        power,
        smoke,
        heat,
        leak,
        temp,
        smoke_gate,
        heat_gate,
        leak_gate,
        temp_gate,
        fire_classifier,
        leak_classifier,
        risk_scorer,
        arbiter,
        sprinkler_cmd,
        valve_cmd,
        alarm_cmd,
        notif_cmd,
        sprinkler_ack,
        valve_ack,
        alarm_ack,
        notif_ack,
        sprinkler_verify,
        valve_verify,
        alarm_verify,
        notif_verify,
    ) = nodes
    phases = [
        Phase(
            "env",
            nodes=(sensor_bus, actuator_net, cloud, power),
            transitions=(Goto("sense"),),
            is_initial=True,
        ),
        Phase(
            "sense",
            nodes=(smoke, heat, leak, temp),
            transitions=(Goto("staging"),),
        ),
        Phase(
            "staging",
            nodes=(smoke_gate, heat_gate, leak_gate, temp_gate),
            transitions=(Goto("classify"),),
        ),
        Phase(
            "classify",
            nodes=(
                fire_classifier,
                leak_classifier,
                risk_scorer,
                arbiter,
            ),
            transitions=(Goto("command"),),
        ),
        Phase(
            "command",
            nodes=(sprinkler_cmd, valve_cmd, alarm_cmd, notif_cmd),
            transitions=(Goto("verify"),),
        ),
        Phase(
            "verify",
            nodes=(
                sprinkler_ack,
                valve_ack,
                alarm_ack,
                notif_ack,
                sprinkler_verify,
                valve_verify,
                alarm_verify,
                notif_verify,
            ),
            transitions=(
                If(ALL_ACKED, terminate, name="ok"),
                If(~ALL_ACKED, "command", name="retry"),
            ),
        ),
    ]
    return PhasedReactiveSystem(phases=phases)


# ────────────────────────────────────────────────────────────────────
# main: compile-time exploration + one happy-path tick
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    from regelum import CompileError

    print("=== async-world with structural retry chain (max_attempts=3) ===")
    system = build_async_world(max_attempts=3)
    print(f"compile ok = {system.compile_report.ok}")
    print(f"phases     = {len(system.phases)}")
    print(f"nodes      = {len(system.nodes)}")

    print()
    print("=== async-world with retry back-edge (cycle in H) ===")
    try:
        build_async_world_with_cycle()
    except CompileError as exc:
        print("compile FAILED, as expected:")
        for issue in exc.report.issues:
            head = issue.message.split(", witness=")[0]
            print(f"  [{issue.location}]  {head}")

    print()
    print("=== runtime trace (deterministic happy path of v_async, one tick) ===")
    records = system.step()
    visits: list[str] = []
    for record in records:
        if not visits or visits[-1] != record.phase:
            visits.append(record.phase)
    snap = system.snapshot()
    print(f"phase trace: {' -> '.join(visits)} -> stop")
    print(f"  smoke_ev      = {snap['SmokeGate.smoke_ev']}")
    print(f"  heat_ev       = {snap['HeatGate.heat_ev']}")
    print(f"  leak_ev       = {snap['LeakGate.leak_ev']}")
    print(f"  temp_ev       = {snap['TempGate.temp_ev']}")
    print(f"  fire_state    = {snap['AsyncFireClassifier.fire_state']}")
    print(f"  leak_state    = {snap['AsyncLeakClassifier.leak_state']}")
    print(f"  action_plan   = {snap['AsyncPriorityArbiter.action_plan']}")
    print(f"  sprinkler_on  = {snap['SprinklerCmdAsync.sprinkler_on']}")
    print(f"  valve_closed  = {snap['ValveCmdAsync.valve_closed']}")
    print(f"  alarm_on      = {snap['AlarmCmdAsync.alarm_on']}")
    print(f"  notif_sent    = {snap['NotifCmdAsync.notif_sent']}")


if __name__ == "__main__":
    main()
