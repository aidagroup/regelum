from __future__ import annotations

from typing import cast

import regelum as rg


class SensorFusion(rg.Node):
    def __init__(self, target: float = 1.0) -> None:
        self.target = target

    class Inputs(rg.NodeInputs):
        position: float = rg.src("Plant.State.position")
        velocity: float = rg.src("Plant.State.velocity")

    class State(rg.NodeState):
        position_estimate: float = rg.var(init=0.0)
        velocity_estimate: float = rg.var(init=0.0)
        target: float = rg.var(init=lambda self: cast(SensorFusion, self).target)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            position_estimate=inputs.position,
            velocity_estimate=inputs.velocity,
            target=self.target,
        )


class Supervisor(rg.Node):
    def __init__(self, kp: float = 4.0, kd: float = 1.2, limit: float = 8.0) -> None:
        self.kp = kp
        self.kd = kd
        self.limit = limit

    class Inputs(rg.NodeInputs):
        position: float = rg.src(SensorFusion.State.position_estimate)
        velocity: float = rg.src(SensorFusion.State.velocity_estimate)
        target: float = rg.src(SensorFusion.State.target)

    class State(rg.NodeState):
        force: float = rg.var(init=0.0)
        saturated: bool = rg.var(init=False)

    def update(self, inputs: Inputs) -> State:
        raw = self.kp * (inputs.target - inputs.position) - self.kd * inputs.velocity
        force = max(-self.limit, min(self.limit, raw))
        return self.State(force=force, saturated=abs(raw) > self.limit)


class Plant(rg.Node):
    def __init__(
        self,
        init_position: float = 0.0,
        init_velocity: float = 0.0,
        dt: float = 0.05,
    ) -> None:
        self.init_position = init_position
        self.init_velocity = init_velocity
        self.dt = dt

    class Inputs(rg.NodeInputs):
        force: float = rg.src("Supervisor.State.force")
        position: float = rg.src("Plant.State.position")
        velocity: float = rg.src("Plant.State.velocity")

    class State(rg.NodeState):
        position: float = rg.var(init=lambda self: cast(Plant, self).init_position)
        velocity: float = rg.var(init=lambda self: cast(Plant, self).init_velocity)

    def update(self, inputs: Inputs) -> State:
        velocity = inputs.velocity + self.dt * inputs.force
        position = inputs.position + self.dt * velocity
        return self.State(position=position, velocity=velocity)


class SafetyMonitor(rg.Node):
    def __init__(self, position_limit: float = 2.5) -> None:
        self.position_limit = position_limit

    class Inputs(rg.NodeInputs):
        position: float = rg.src(Plant.State.position)
        saturated: bool = rg.src(Supervisor.State.saturated)

    class State(rg.NodeState):
        fault: bool = rg.var(init=False)
        reason: str = rg.var(init="ok")

    def update(self, inputs: Inputs) -> State:
        out_of_bounds = abs(inputs.position) > self.position_limit
        fault = out_of_bounds or inputs.saturated
        reason = "position_limit" if out_of_bounds else "saturated" if inputs.saturated else "ok"
        return self.State(fault=fault, reason=reason)


class Alarm(rg.Node):
    class Inputs(rg.NodeInputs):
        reason: str = rg.src(SafetyMonitor.State.reason)

    class State(rg.NodeState):
        active: bool = rg.var(init=False)
        message: str = rg.var(init="")

    def update(self, inputs: Inputs) -> State:
        return self.State(active=True, message=f"fault:{inputs.reason}")


class TraceLogger(rg.Node):
    class Inputs(rg.NodeInputs):
        position: float = rg.src(Plant.State.position)
        velocity: float = rg.src(Plant.State.velocity)
        force: float = rg.src(Supervisor.State.force)
        fault: bool = rg.src(SafetyMonitor.State.fault)
        alarm: bool = rg.src(Alarm.State.active)
        trace: tuple[tuple[float, float, float, bool, bool], ...] = rg.src(
            "TraceLogger.State.trace"
        )

    class State(rg.NodeState):
        trace: tuple[tuple[float, float, float, bool, bool], ...] = rg.var(init=())

    def update(self, inputs: Inputs) -> State:
        sample = (
            inputs.position,
            inputs.velocity,
            inputs.force,
            inputs.fault,
            inputs.alarm,
        )
        return self.State(trace=inputs.trace + (sample,))


def build_ok_system() -> rg.PhasedReactiveSystem:
    sensor_fusion = SensorFusion()
    supervisor = Supervisor()
    plant = Plant()
    safety_monitor = SafetyMonitor()
    alarm = Alarm()
    logger = TraceLogger()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "sense",
                nodes=(sensor_fusion,),
                transitions=(rg.Goto("decide"),),
                is_initial=True,
            ),
            rg.Phase(
                "decide",
                nodes=(supervisor, safety_monitor),
                transitions=(
                    rg.If(rg.V("SafetyMonitor.fault"), "alarm", name="fault"),
                    rg.If(~rg.V("SafetyMonitor.fault"), "apply", name="normal"),
                ),
            ),
            rg.Phase(
                "apply",
                nodes=(plant, logger),
                transitions=(rg.Goto(rg.terminate),),
            ),
            rg.Phase(
                "alarm",
                nodes=(alarm, logger),
                transitions=(rg.Goto(rg.terminate),),
            ),
        ],
    )


def build_bad_c1_system() -> rg.PhasedReactiveSystem:
    sensor_fusion = SensorFusion()
    supervisor = Supervisor()
    plant = Plant()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "bad-coupled-control",
                nodes=(sensor_fusion, supervisor, plant),
                transitions=(rg.Goto(rg.terminate),),
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
    except rg.CompileError as exc:
        print("bad system: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    print("bad system: compile=True")


if __name__ == "__main__":
    main()
