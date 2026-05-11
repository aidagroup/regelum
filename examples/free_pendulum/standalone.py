from __future__ import annotations

import math
from typing import cast

import casadi as ca

import regelum as rg

BASE_DT = "0.01"
GRAVITY = 9.81
LENGTH = 1.0
DAMPING = 0.08


class FreePendulum(rg.ODENode):
    def __init__(
        self,
        *,
        theta0: float = 0.85,
        omega0: float = 0.0,
        gravity: float = GRAVITY,
        length: float = LENGTH,
        damping: float = DAMPING,
    ) -> None:
        self.theta0 = theta0
        self.omega0 = omega0
        self.gravity = gravity
        self.length = length
        self.damping = damping

    class State(rg.NodeState):
        theta: float = rg.Var(init=lambda self: cast(FreePendulum, self).theta0)
        omega: float = rg.Var(init=lambda self: cast(FreePendulum, self).omega0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        theta_dot = state.omega
        omega_dot = (
            -self.gravity / self.length * ca.sin(state.theta)
            - self.damping * state.omega
        )
        return self.State(theta=theta_dot, omega=omega_dot)


class Observer(rg.Node):
    class Inputs(rg.NodeInputs):
        theta: float = rg.Input(src=FreePendulum.State.theta)
        omega: float = rg.Input(src=FreePendulum.State.omega)

    class State(rg.NodeState):
        sin_angle: float = rg.Var(init=0.0)
        cos_angle: float = rg.Var(init=1.0)
        angular_velocity: float = rg.Var(init=0.0)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            sin_angle=math.sin(inputs.theta),
            cos_angle=math.cos(inputs.theta),
            angular_velocity=inputs.omega,
        )


class Logger(rg.Node):
    class Inputs(rg.NodeInputs):
        samples: tuple[tuple[float, float, float, float, float], ...] = rg.Input(
            src=lambda: Logger.State.samples
        )
        time: float = rg.Input(src=rg.Clock.time)
        theta: float = rg.Input(src=FreePendulum.State.theta)
        sin_angle: float = rg.Input(src=Observer.State.sin_angle)
        cos_angle: float = rg.Input(src=Observer.State.cos_angle)
        angular_velocity: float = rg.Input(src=Observer.State.angular_velocity)

    class State(rg.NodeState):
        samples: tuple[tuple[float, float, float, float, float], ...] = rg.Var(init=())

    def update(self, inputs: Inputs) -> State:
        sample = (
            inputs.time,
            inputs.theta,
            inputs.sin_angle,
            inputs.cos_angle,
            inputs.angular_velocity,
        )
        return self.State(samples=(*inputs.samples, sample))


def build_system() -> rg.PhasedReactiveSystem:
    pendulum = FreePendulum()
    observer = Observer()
    logger = Logger()
    plant = rg.ODESystem(nodes=(pendulum,), dt=BASE_DT)
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "plant",
                nodes=(plant,),
                transitions=(rg.Goto("observe"),),
                is_initial=True,
            ),
            rg.Phase(
                "observe",
                nodes=(observer, logger),
                transitions=(rg.Goto(rg.terminate),),
            ),
        ],
        base_dt=BASE_DT,
    )


def run_response(steps: int = 700) -> tuple[tuple[float, float, float, float, float], ...]:
    system = build_system()
    system.run(steps)
    return cast(
        tuple[tuple[float, float, float, float, float], ...],
        system.read(Logger.State.samples),
    )


def main() -> None:
    samples = run_response()
    time, theta, sin_angle, cos_angle, omega = samples[-1]
    print(f"time={time:.2f}")
    print(f"theta={theta:.6f}")
    print(f"sin(theta)={sin_angle:.6f}")
    print(f"cos(theta)={cos_angle:.6f}")
    print(f"omega={omega:.6f}")


if __name__ == "__main__":
    main()
