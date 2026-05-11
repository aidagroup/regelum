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

    class State(rg.NodeState):
        theta: float = rg.Var(init=lambda self: cast(ControlledPendulum, self).theta0)
        omega: float = rg.Var(init=lambda self: cast(ControlledPendulum, self).omega0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        state: State,
        torque: float = rg.Input(src=lambda: SwingUpController.State.torque),
    ) -> State:
        inertia = self.mass * self.length * self.length / 3.0
        theta_dot = state.omega
        omega_dot = (
            -(3.0 * self.gravity) / (2.0 * self.length) * ca.sin(state.theta)
            - self.damping * state.omega
            + torque / inertia
        )
        return self.State(theta=theta_dot, omega=omega_dot)


class Observer(rg.Node):
    class State(rg.NodeState):
        sin_angle: float
        cos_angle: float
        angular_velocity: float

    def update(
        self,
        theta: float = rg.Input(src=ControlledPendulum.State.theta),
        omega: float = rg.Input(src=ControlledPendulum.State.omega),
    ) -> State:
        return self.State(
            sin_angle=math.sin(theta),
            cos_angle=math.cos(theta),
            angular_velocity=omega,
        )


class SwingUpController(rg.Node):
    class State(rg.NodeState):
        torque: float = rg.Var(init=0.0)

    def __init__(
        self,
        *,
        kp: float = 14.0,
        kd: float = 4.0,
        torque_limit: float = 4.0,
        dt: str = CONTROL_DT,
    ) -> None:
        super().__init__(dt=dt)
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

    def update(
        self,
        sin_angle: float = rg.Input(src=Observer.State.sin_angle),
        cos_angle: float = rg.Input(src=Observer.State.cos_angle),
        angular_velocity: float = rg.Input(src=Observer.State.angular_velocity),
    ) -> State:
        theta = math.atan2(sin_angle, cos_angle)
        error = wrap_angle(theta - math.pi)
        raw = -self.kp * error - self.kd * angular_velocity
        torque = max(-self.torque_limit, min(self.torque_limit, raw))
        return self.State(torque=torque)


class Logger(rg.Node):
    class State(rg.NodeState):
        samples: list[tuple[float, float, float, float]] = rg.Var(init=list)

    def update(
        self,
        prev_state: State,
        time: float = rg.Input(src=rg.Clock.time),
        theta: float = rg.Input(src=ControlledPendulum.State.theta),
        omega: float = rg.Input(src=ControlledPendulum.State.omega),
        torque: float = rg.Input(src=SwingUpController.State.torque),
    ) -> State:
        sample = (time, theta, omega, torque)
        prev_state.samples.append(sample)
        return self.State(samples=prev_state.samples)


def build_system() -> rg.PhasedReactiveSystem:
    pendulum = ControlledPendulum()
    observer = Observer()
    controller = SwingUpController()
    logger = Logger()
    plant = rg.ODESystem(nodes=(pendulum,), dt=BASE_DT)
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "observe_and_control",
                nodes=(observer, controller, logger),
                transitions=(rg.Goto("plant"),),
                is_initial=True,
            ),
            rg.Phase(
                "plant",
                nodes=(plant,),
                transitions=(rg.Goto(rg.terminate),),
            ),
        ],
        base_dt=BASE_DT,
    )


def run_response(steps: int = 1000) -> list[tuple[float, float, float, float]]:
    system = build_system()
    system.run(steps)
    return cast(
        list[tuple[float, float, float, float]],
        system.read(Logger.State.samples),
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
