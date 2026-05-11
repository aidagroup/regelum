from __future__ import annotations

from math import sin
from typing import cast

from regelum import (
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    Var,
    terminate,
)


class PendulumPlant(Node):
    def __init__(
        self,
        init_theta: float = 0.35,
        init_omega: float = 0.0,
        init_time: float = 0.0,
        dt: float = 0.02,
        gravity: float = 9.81,
        length: float = 1.0,
        damping: float = 0.1,
    ) -> None:
        self.init_theta = init_theta
        self.init_omega = init_omega
        self.init_time = init_time
        self.dt = dt
        self.gravity = gravity
        self.length = length
        self.damping = damping

    class State(NodeState):
        theta: float = Var(init=lambda self: cast(PendulumPlant, self).init_theta)
        omega: float = Var(init=lambda self: cast(PendulumPlant, self).init_omega)
        time: float = Var(init=lambda self: cast(PendulumPlant, self).init_time)

    def update(
        self,
        torque: float = Input(src=lambda: PDController.State.torque),
        theta: float = Input(src=lambda: PendulumPlant.State.theta),
        omega: float = Input(src=lambda: PendulumPlant.State.omega),
        time: float = Input(src=lambda: PendulumPlant.State.time),
    ) -> State:
        acceleration = torque - self.gravity / self.length * sin(theta) - self.damping * omega
        omega_next = omega + self.dt * acceleration
        theta_next = theta + self.dt * omega_next
        time_next = time + self.dt
        return self.State(theta=theta_next, omega=omega_next, time=time_next)


class PDController(Node):
    def __init__(self, kp: float = 12.0, kd: float = 3.0, torque_limit: float = 10.0) -> None:
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

    class Inputs(NodeInputs):
        theta: float = Input(src=PendulumPlant.State.theta)
        omega: float = Input(src=PendulumPlant.State.omega)

    class State(NodeState):
        torque: float

    def update(self, inputs: Inputs) -> State:
        raw = -self.kp * inputs.theta - self.kd * inputs.omega
        torque = max(-self.torque_limit, min(self.torque_limit, raw))
        return self.State(torque=torque)


class Logger(Node):
    class State(NodeState):
        samples: list[tuple[float, float, float, float]] = Var(init=lambda: list())

    def update(
        self,
        time: float = Input(src=PendulumPlant.State.time),
        theta: float = Input(src=PendulumPlant.State.theta),
        omega: float = Input(src=PendulumPlant.State.omega),
        torque: float = Input(src=PDController.State.torque),
        samples: list[tuple[float, float, float, float]] = Input(
            src=lambda: Logger.State.samples
        ),
    ) -> State:
        print(
            f"time={time:.6f}, "
            f"state=(theta={theta:.6f}, omega={omega:.6f}), "
            f"action=(torque={torque:.6f})"
        )

        samples.append((time, theta, omega, torque))
        return self.State(samples=samples)


def build_system() -> PhasedReactiveSystem:
    controller = PDController()
    plant = PendulumPlant(init_theta=3.14, init_omega=0.0, init_time=10.0)
    logger = Logger()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "control",
                nodes=(controller, logger),
                transitions=(Goto("plant"),),
                is_initial=True,
            ),
            Phase(
                "plant",
                nodes=(plant,),
                transitions=(Goto(terminate),),
            ),
        ],
    )


def print_compile_report(system: PhasedReactiveSystem) -> None:
    print("Compile report:")
    for line in system.compile_report.format().splitlines():
        print(f"  {line}")


def main() -> None:
    system = build_system()
    print_compile_report(system)
    system.run(steps=10)
    snapshot = system.snapshot()
    samples = snapshot["Logger.samples"]
    print("Final state:")
    print(f"  time = {snapshot['PendulumPlant.time']:.6f}")
    print(f"  theta = {snapshot['PendulumPlant.theta']:.6f}")
    print(f"  omega = {snapshot['PendulumPlant.omega']:.6f}")
    print(f"  torque = {snapshot['PDController.torque']:.6f}")
    print(f"  samples = {len(samples)}")


if __name__ == "__main__":
    main()
