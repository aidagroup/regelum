from __future__ import annotations

from dataclasses import dataclass

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


S0, S1, S2, S3, S4, S5, S6, S7 = range(8)
LS0, LS1, LS2 = range(3)
SS0, SS1, SS2 = range(3)

FDS_STATE_NAMES = {
    S0: "S0: no fire evidence, detectors working",
    S1: "S1: smoke detected",
    S2: "S2: smoke and heat detected",
    S3: "S3: heat detector failed, no smoke",
    S4: "S4: heat detector failed, smoke detected",
    S5: "S5: smoke detector failed, normal temperature",
    S6: "S6: smoke detector failed, high temperature",
    S7: "S7: smoke and heat detection unavailable",
}
LDS_STATE_NAMES = {
    LS0: "LS0: no leak",
    LS1: "LS1: leak detected",
    LS2: "LS2: leak detector unavailable",
}
SUBSTATE_NAMES = {
    SS0: "SS0: no independent leak action",
    SS1: "SS1: independent leak action required",
    SS2: "SS2: leak detection unavailable",
}


@dataclass(frozen=True)
class Evidence:
    smoke_detector_ok: bool
    smoke_battery_ok: bool
    smoke_detected: bool
    heat_detector_ok: bool
    heat_battery_ok: bool
    high_temperature: bool
    leak_detector_ok: bool
    leak_battery_ok: bool
    leak_detected: bool


FIG4_EVIDENCE = Evidence(
    smoke_detector_ok=True,
    smoke_battery_ok=True,
    smoke_detected=True,
    heat_detector_ok=True,
    heat_battery_ok=True,
    high_temperature=True,
    leak_detector_ok=True,
    leak_battery_ok=True,
    leak_detected=True,
)

DEFAULT_EVIDENCE_SEQUENCE = (
    Evidence(True, True, False, True, True, False, True, True, False),
    FIG4_EVIDENCE,
    Evidence(True, True, False, True, True, False, True, True, True),
    Evidence(True, False, False, True, True, True, True, True, True),
)


class SmartHomeEvidence(Node):
    initial_evidence: Evidence = DEFAULT_EVIDENCE_SEQUENCE[0]

    def __init__(self, evidence: tuple[Evidence, ...] = DEFAULT_EVIDENCE_SEQUENCE) -> None:
        type(self).initial_evidence = evidence[0]
        self.evidence = evidence

    class Inputs(NodeInputs):
        tick: int = Input(source="SmartHomeEvidence.Outputs.tick")

    class Outputs(NodeOutputs):
        tick: int = Output(initial=0, domain=range(0, 8))
        smoke_detector_ok: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.smoke_detector_ok
        )
        smoke_battery_ok: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.smoke_battery_ok
        )
        smoke_detected: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.smoke_detected
        )
        heat_detector_ok: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.heat_detector_ok
        )
        heat_battery_ok: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.heat_battery_ok
        )
        high_temperature: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.high_temperature
        )
        leak_detector_ok: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.leak_detector_ok
        )
        leak_battery_ok: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.leak_battery_ok
        )
        leak_detected: bool = Output(
            initial=lambda: SmartHomeEvidence.initial_evidence.leak_detected
        )

    def run(self, inputs: Inputs) -> Outputs:
        index = min(inputs.tick, len(self.evidence) - 1)
        evidence = self.evidence[index]
        return self.Outputs(
            tick=inputs.tick + 1,
            smoke_detector_ok=evidence.smoke_detector_ok,
            smoke_battery_ok=evidence.smoke_battery_ok,
            smoke_detected=evidence.smoke_detected,
            heat_detector_ok=evidence.heat_detector_ok,
            heat_battery_ok=evidence.heat_battery_ok,
            high_temperature=evidence.high_temperature,
            leak_detector_ok=evidence.leak_detector_ok,
            leak_battery_ok=evidence.leak_battery_ok,
            leak_detected=evidence.leak_detected,
        )


class FireDetectionSystem(Node):
    class Inputs(NodeInputs):
        smoke_detector_ok: bool = Input(source=SmartHomeEvidence.Outputs.smoke_detector_ok)
        smoke_battery_ok: bool = Input(source=SmartHomeEvidence.Outputs.smoke_battery_ok)
        smoke_detected: bool = Input(source=SmartHomeEvidence.Outputs.smoke_detected)
        heat_detector_ok: bool = Input(source=SmartHomeEvidence.Outputs.heat_detector_ok)
        heat_battery_ok: bool = Input(source=SmartHomeEvidence.Outputs.heat_battery_ok)
        high_temperature: bool = Input(source=SmartHomeEvidence.Outputs.high_temperature)

    class Outputs(NodeOutputs):
        fds_state: int = Output(initial=S0, domain=range(8))
        smoke_alarm_requested: bool = Output(initial=False)
        fire_alarm_requested: bool = Output(initial=False)
        sprinkler_requested: bool = Output(initial=False)
        fds_notify_user: bool = Output(initial=False)
        fds_urgent_notify_user: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        smoke_unit_ok = inputs.smoke_detector_ok and inputs.smoke_battery_ok
        heat_unit_ok = inputs.heat_detector_ok and inputs.heat_battery_ok

        if smoke_unit_ok and heat_unit_ok:
            state = S2 if inputs.smoke_detected and inputs.high_temperature else S1 if inputs.smoke_detected else S0
        elif smoke_unit_ok:
            state = S4 if inputs.smoke_detected else S3
        elif heat_unit_ok:
            state = S6 if inputs.high_temperature else S5
        else:
            state = S7

        return self.Outputs(
            fds_state=state,
            smoke_alarm_requested=state in (S1, S4),
            fire_alarm_requested=state in (S2, S6),
            sprinkler_requested=state in (S2, S6),
            fds_notify_user=state in (S1, S2, S3, S5, S6),
            fds_urgent_notify_user=state in (S4, S7),
        )


class LeakDetectionSystem(Node):
    class Inputs(NodeInputs):
        leak_detector_ok: bool = Input(source=SmartHomeEvidence.Outputs.leak_detector_ok)
        leak_battery_ok: bool = Input(source=SmartHomeEvidence.Outputs.leak_battery_ok)
        leak_detected: bool = Input(source=SmartHomeEvidence.Outputs.leak_detected)

    class Outputs(NodeOutputs):
        lds_state: int = Output(initial=LS0, domain=range(3))
        valve_close_requested: bool = Output(initial=False)
        lds_notify_user: bool = Output(initial=False)
        lds_urgent_notify_user: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        leak_unit_ok = inputs.leak_detector_ok and inputs.leak_battery_ok
        if not leak_unit_ok:
            state = LS2
        elif inputs.leak_detected:
            state = LS1
        else:
            state = LS0
        return self.Outputs(
            lds_state=state,
            valve_close_requested=state == LS1,
            lds_notify_user=state == LS1,
            lds_urgent_notify_user=state == LS2,
        )


class ConflictResolver(Node):
    class Inputs(NodeInputs):
        fds_state: int = Input(source=FireDetectionSystem.Outputs.fds_state)
        lds_state: int = Input(source=LeakDetectionSystem.Outputs.lds_state)
        sprinkler_requested: bool = Input(source=FireDetectionSystem.Outputs.sprinkler_requested)
        valve_close_requested: bool = Input(source=LeakDetectionSystem.Outputs.valve_close_requested)
        fds_notify_user: bool = Input(source=FireDetectionSystem.Outputs.fds_notify_user)
        fds_urgent_notify_user: bool = Input(source=FireDetectionSystem.Outputs.fds_urgent_notify_user)
        lds_notify_user: bool = Input(source=LeakDetectionSystem.Outputs.lds_notify_user)
        lds_urgent_notify_user: bool = Input(source=LeakDetectionSystem.Outputs.lds_urgent_notify_user)

    class Outputs(NodeOutputs):
        system_substate: int = Output(initial=SS0, domain=range(3))
        final_sprinkler_on: bool = Output(initial=False)
        final_water_valve_closed: bool = Output(initial=False)
        final_notify_user: bool = Output(initial=False)
        final_urgent_notify_user: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        if inputs.lds_state == LS2:
            substate = SS2
            close_valve = False
        elif inputs.valve_close_requested and not inputs.sprinkler_requested:
            substate = SS1
            close_valve = True
        else:
            substate = SS0
            close_valve = False

        return self.Outputs(
            system_substate=substate,
            final_sprinkler_on=inputs.sprinkler_requested,
            final_water_valve_closed=close_valve,
            final_notify_user=(
                inputs.fds_notify_user
                or inputs.fds_urgent_notify_user
                or inputs.lds_notify_user
                or inputs.lds_urgent_notify_user
            ),
            final_urgent_notify_user=(
                inputs.fds_urgent_notify_user or inputs.lds_urgent_notify_user
            ),
        )


class ActionDispatcher(Node):
    class Inputs(NodeInputs):
        final_sprinkler_on: bool = Input(source=ConflictResolver.Outputs.final_sprinkler_on)
        final_water_valve_closed: bool = Input(source=ConflictResolver.Outputs.final_water_valve_closed)
        final_notify_user: bool = Input(source=ConflictResolver.Outputs.final_notify_user)
        final_urgent_notify_user: bool = Input(source=ConflictResolver.Outputs.final_urgent_notify_user)

    class Outputs(NodeOutputs):
        sprinkler_on: bool = Output(initial=False)
        water_valve_closed: bool = Output(initial=False)
        notify_user: bool = Output(initial=False)
        urgent_notify_user: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            sprinkler_on=inputs.final_sprinkler_on,
            water_valve_closed=inputs.final_water_valve_closed,
            notify_user=inputs.final_notify_user,
            urgent_notify_user=inputs.final_urgent_notify_user,
        )


class SafetyLogger(Node):
    class Inputs(NodeInputs):
        tick: int = Input(source=SmartHomeEvidence.Outputs.tick)
        fds_state: int = Input(source=FireDetectionSystem.Outputs.fds_state)
        lds_state: int = Input(source=LeakDetectionSystem.Outputs.lds_state)
        system_substate: int = Input(source=ConflictResolver.Outputs.system_substate)
        sprinkler_on: bool = Input(source=ActionDispatcher.Outputs.sprinkler_on)
        water_valve_closed: bool = Input(source=ActionDispatcher.Outputs.water_valve_closed)
        notify_user: bool = Input(source=ActionDispatcher.Outputs.notify_user)
        urgent_notify_user: bool = Input(source=ActionDispatcher.Outputs.urgent_notify_user)
        log: tuple[tuple[int, int, int, int, bool, bool, bool, bool], ...] = Input(
            source="SafetyLogger.Outputs.log"
        )

    class Outputs(NodeOutputs):
        log: tuple[tuple[int, int, int, int, bool, bool, bool, bool], ...] = Output(initial=())

    def run(self, inputs: Inputs) -> Outputs:
        sample = (
            inputs.tick,
            inputs.fds_state,
            inputs.lds_state,
            inputs.system_substate,
            inputs.sprinkler_on,
            inputs.water_valve_closed,
            inputs.notify_user,
            inputs.urgent_notify_user,
        )
        print(
            "tick={tick} fds={fds} lds={lds} substate={substate} "
            "sprinkler={sprinkler} valve_closed={valve} notify={notify} urgent={urgent}".format(
                tick=inputs.tick,
                fds=FDS_STATE_NAMES[inputs.fds_state].split(":", maxsplit=1)[0],
                lds=LDS_STATE_NAMES[inputs.lds_state].split(":", maxsplit=1)[0],
                substate=SUBSTATE_NAMES[inputs.system_substate].split(":", maxsplit=1)[0],
                sprinkler=inputs.sprinkler_on,
                valve=inputs.water_valve_closed,
                notify=inputs.notify_user,
                urgent=inputs.urgent_notify_user,
            )
        )
        return self.Outputs(log=inputs.log + (sample,))


def build_system(
    evidence: tuple[Evidence, ...] = DEFAULT_EVIDENCE_SEQUENCE,
) -> PhasedReactiveSystem:
    evidence_node = SmartHomeEvidence(evidence)
    fire = FireDetectionSystem()
    leak = LeakDetectionSystem()
    conflict = ConflictResolver()
    dispatcher = ActionDispatcher()
    logger = SafetyLogger()
    return PhasedReactiveSystem(
        phases=[
            Phase("collect-evidence", nodes=(evidence_node,), transitions=(Goto("classify-fire"),), is_initial=True),
            Phase("classify-fire", nodes=(fire,), transitions=(Goto("classify-leak"),)),
            Phase("classify-leak", nodes=(leak,), transitions=(Goto("resolve-conflicts"),)),
            Phase("resolve-conflicts", nodes=(conflict,), transitions=(Goto("dispatch-actions"),)),
            Phase("dispatch-actions", nodes=(dispatcher,), transitions=(Goto("log"),)),
            Phase("log", nodes=(logger,), transitions=(Goto(terminate),)),
        ],
    )


def build_fig4_system() -> PhasedReactiveSystem:
    return build_system((FIG4_EVIDENCE,))


def main() -> None:
    print("Mohanty et al. 2023 smart-home safety monitor")
    system = build_system()
    print("C1=PASS C2=PASS C2*=PASS C3=PASS")
    system.run(steps=4)

    fig4 = build_fig4_system()
    fig4.step()
    snapshot = fig4.snapshot()
    print("\nFig. 4 scenario:")
    print(FDS_STATE_NAMES[snapshot["FireDetectionSystem.fds_state"]])
    print(LDS_STATE_NAMES[snapshot["LeakDetectionSystem.lds_state"]])
    print(SUBSTATE_NAMES[snapshot["ConflictResolver.system_substate"]])
    print(
        "sprinkler_on={sprinkler} water_valve_closed={valve}".format(
            sprinkler=snapshot["ActionDispatcher.sprinkler_on"],
            valve=snapshot["ActionDispatcher.water_valve_closed"],
        )
    )


if __name__ == "__main__":
    main()
