from __future__ import annotations

import math
from typing import cast

import casadi as ca

import regelum as rg

BASE_DT = "0.01"
CONTROL_DT = "0.05"
GRAVITY = 9.81
LENGTH = 1.0
MASS = 1.0
DAMPING = 0.08


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ControlledPendulum(rg.ODENode):
    def __init__(
        self,
        *,
        theta0: float = 0.2,
        omega0: float = 0.0,
        gravity: float = GRAVITY,
        length: float = LENGTH,
        mass: float = MASS,
        damping: float = DAMPING,
    ) -> None:
        self.theta0 = theta0
        self.omega0 = omega0
        self.gravity = gravity
        self.length = length
        self.mass = mass
        self.damping = damping

    class Inputs(rg.NodeInputs):
        torque: float = rg.Input(src=lambda: SwingUpController.State.torque)

    class State(rg.NodeState):
        theta: float = rg.Var(init=lambda self: cast(ControlledPendulum, self).theta0)
        omega: float = rg.Var(init=lambda self: cast(ControlledPendulum, self).omega0)

    def dstate(self, inputs: Inputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        inertia = self.mass * self.length * self.length
        theta_dot = state.omega
        omega_dot = (
            -self.gravity / self.length * ca.sin(state.theta)
            - self.damping * state.omega
            + inputs.torque / inertia
        )
        return self.State(theta=theta_dot, omega=omega_dot)


class ControlledObserver(rg.Node):
    class Inputs(rg.NodeInputs):
        theta: float = rg.Input(src=ControlledPendulum.State.theta)
        omega: float = rg.Input(src=ControlledPendulum.State.omega)

    class State(rg.NodeState):
        sin_angle: float = rg.Var(init=lambda: math.sin(0.2))
        cos_angle: float = rg.Var(init=lambda: math.cos(0.2))
        angular_velocity: float = rg.Var(init=0.0)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            sin_angle=math.sin(inputs.theta),
            cos_angle=math.cos(inputs.theta),
            angular_velocity=inputs.omega,
        )


class SwingUpController(rg.Node):
    class Inputs(rg.NodeInputs):
        sin_angle: float = rg.Input(src=ControlledObserver.State.sin_angle)
        cos_angle: float = rg.Input(src=ControlledObserver.State.cos_angle)
        angular_velocity: float = rg.Input(src=ControlledObserver.State.angular_velocity)

    class State(rg.NodeState):
        torque: float = rg.Var(init=0.0)

    def __init__(
        self,
        *,
        kp: float = 14.0,
        kd: float = 4.0,
        torque_limit: float = 8.0,
        dt: str = CONTROL_DT,
    ) -> None:
        super().__init__(dt=dt)
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

    def update(self, inputs: Inputs) -> State:
        theta = math.atan2(inputs.sin_angle, inputs.cos_angle)
        error = wrap_angle(theta - math.pi)
        raw = -self.kp * error - self.kd * inputs.angular_velocity
        torque = max(-self.torque_limit, min(self.torque_limit, raw))
        return self.State(torque=torque)


class ControlledLogger(rg.Node):
    class Inputs(rg.NodeInputs):
        samples: tuple[tuple[float, float, float, float], ...] = rg.Input(
            src=lambda: ControlledLogger.State.samples
        )
        time: float = rg.Input(src=rg.Clock.time)
        theta: float = rg.Input(src=ControlledPendulum.State.theta)
        omega: float = rg.Input(src=ControlledPendulum.State.omega)
        torque: float = rg.Input(src=SwingUpController.State.torque)

    class State(rg.NodeState):
        samples: tuple[tuple[float, float, float, float], ...] = rg.Var(init=())

    def update(self, inputs: Inputs) -> State:
        sample = (inputs.time, inputs.theta, inputs.omega, inputs.torque)
        return self.State(samples=(*inputs.samples, sample))


def build_system() -> rg.PhasedReactiveSystem:
    pendulum = ControlledPendulum()
    observer = ControlledObserver()
    controller = SwingUpController()
    logger = ControlledLogger()
    plant = rg.ODESystem(nodes=(pendulum,), dt=BASE_DT)
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "control",
                nodes=(controller,),
                transitions=(rg.Goto("plant"),),
                is_initial=True,
            ),
            rg.Phase("plant", nodes=(plant,), transitions=(rg.Goto("observe"),)),
            rg.Phase(
                "observe",
                nodes=(observer, logger),
                transitions=(rg.Goto(rg.terminate),),
            ),
        ],
        base_dt=BASE_DT,
    )


def run_response(steps: int = 1000) -> tuple[tuple[float, float, float, float], ...]:
    system = build_system()
    system.run(steps)
    return cast(
        tuple[tuple[float, float, float, float], ...],
        system.read(ControlledLogger.State.samples),
    )


def main() -> None:
    samples = run_response()
    time, theta, omega, torque = samples[-1]
    print(f"time={time:.2f}")
    print(f"theta={theta:.6f}")
    print(f"omega={omega:.6f}")
    print(f"torque={torque:.6f}")


if __name__ == "__main__":
    main()

