from __future__ import annotations

from typing import cast

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


class SensorFusion(Node):
    def __init__(self, target: float = 1.0) -> None:
        self.target = target

    class Inputs(NodeInputs):
        position: float = Input(source="Plant.Outputs.position")
        velocity: float = Input(source="Plant.Outputs.velocity")

    class Outputs(NodeOutputs):
        position_estimate: float = Output(initial=0.0)
        velocity_estimate: float = Output(initial=0.0)
        target: float = Output(initial=lambda self: cast(SensorFusion, self).target)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(
            position_estimate=inputs.position,
            velocity_estimate=inputs.velocity,
            target=self.target,
        )


class Supervisor(Node):
    def __init__(self, kp: float = 4.0, kd: float = 1.2, limit: float = 8.0) -> None:
        self.kp = kp
        self.kd = kd
        self.limit = limit

    class Inputs(NodeInputs):
        position: float = Input(source=SensorFusion.Outputs.position_estimate)
        velocity: float = Input(source=SensorFusion.Outputs.velocity_estimate)
        target: float = Input(source=SensorFusion.Outputs.target)

    class Outputs(NodeOutputs):
        force: float = Output(initial=0.0)
        saturated: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        raw = self.kp * (inputs.target - inputs.position) - self.kd * inputs.velocity
        force = max(-self.limit, min(self.limit, raw))
        return self.Outputs(force=force, saturated=abs(raw) > self.limit)


class Plant(Node):
    def __init__(
        self,
        init_position: float = 0.0,
        init_velocity: float = 0.0,
        dt: float = 0.05,
    ) -> None:
        self.init_position = init_position
        self.init_velocity = init_velocity
        self.dt = dt

    class Inputs(NodeInputs):
        force: float = Input(source="Supervisor.Outputs.force")
        position: float = Input(source="Plant.Outputs.position")
        velocity: float = Input(source="Plant.Outputs.velocity")

    class Outputs(NodeOutputs):
        position: float = Output(initial=lambda self: cast(Plant, self).init_position)
        velocity: float = Output(initial=lambda self: cast(Plant, self).init_velocity)

    def run(self, inputs: Inputs) -> Outputs:
        velocity = inputs.velocity + self.dt * inputs.force
        position = inputs.position + self.dt * velocity
        return self.Outputs(position=position, velocity=velocity)


class SafetyMonitor(Node):
    def __init__(self, position_limit: float = 2.5) -> None:
        self.position_limit = position_limit

    class Inputs(NodeInputs):
        position: float = Input(source=Plant.Outputs.position)
        saturated: bool = Input(source=Supervisor.Outputs.saturated)

    class Outputs(NodeOutputs):
        fault: bool = Output(initial=False)
        reason: str = Output(initial="ok")

    def run(self, inputs: Inputs) -> Outputs:
        out_of_bounds = abs(inputs.position) > self.position_limit
        fault = out_of_bounds or inputs.saturated
        reason = "position_limit" if out_of_bounds else "saturated" if inputs.saturated else "ok"
        return self.Outputs(fault=fault, reason=reason)


class Alarm(Node):
    class Inputs(NodeInputs):
        reason: str = Input(source=SafetyMonitor.Outputs.reason)

    class Outputs(NodeOutputs):
        active: bool = Output(initial=False)
        message: str = Output(initial="")

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(active=True, message=f"fault:{inputs.reason}")


class TraceLogger(Node):
    class Inputs(NodeInputs):
        position: float = Input(source=Plant.Outputs.position)
        velocity: float = Input(source=Plant.Outputs.velocity)
        force: float = Input(source=Supervisor.Outputs.force)
        fault: bool = Input(source=SafetyMonitor.Outputs.fault)
        alarm: bool = Input(source=Alarm.Outputs.active)
        trace: tuple[tuple[float, float, float, bool, bool], ...] = Input(
            source="TraceLogger.Outputs.trace"
        )

    class Outputs(NodeOutputs):
        trace: tuple[tuple[float, float, float, bool, bool], ...] = Output(initial=())

    def run(self, inputs: Inputs) -> Outputs:
        sample = (
            inputs.position,
            inputs.velocity,
            inputs.force,
            inputs.fault,
            inputs.alarm,
        )
        return self.Outputs(trace=inputs.trace + (sample,))


def build_ok_system() -> PhasedReactiveSystem:
    sensor_fusion = SensorFusion()
    supervisor = Supervisor()
    plant = Plant()
    safety_monitor = SafetyMonitor()
    alarm = Alarm()
    logger = TraceLogger()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "sense",
                nodes=(sensor_fusion,),
                transitions=(Goto("decide"),),
                is_initial=True,
            ),
            Phase(
                "decide",
                nodes=(supervisor, safety_monitor),
                transitions=(
                    If(V("SafetyMonitor.fault"), "alarm", name="fault"),
                    If(~V("SafetyMonitor.fault"), "apply", name="normal"),
                ),
            ),
            Phase(
                "apply",
                nodes=(plant, logger),
                transitions=(Goto(terminate),),
            ),
            Phase(
                "alarm",
                nodes=(alarm, logger),
                transitions=(Goto(terminate),),
            ),
        ],
    )


def build_bad_c1_system() -> PhasedReactiveSystem:
    sensor_fusion = SensorFusion()
    supervisor = Supervisor()
    plant = Plant()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "bad-coupled-control",
                nodes=(sensor_fusion, supervisor, plant),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )


def main() -> None:
    ok_system = build_ok_system()
    print(f"ok system: compile={ok_system.compile_report.ok}")
    ok_system.run(steps=3)
    print(f"ok trace length: {len(ok_system.snapshot()['TraceLogger.trace'])}")

    try:
        build_bad_c1_system()
    except CompileError as exc:
        print("bad system: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    print("bad system: compile=True")


if __name__ == "__main__":
    main()
