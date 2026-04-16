from __future__ import annotations

from dataclasses import dataclass
from math import sin


@dataclass
class PendulumState:
    theta: float
    omega: float
    time: float


def controller(theta: float, omega: float) -> float:
    gain_theta = 12.0
    gain_omega = 3.0
    raw = -gain_theta * theta - gain_omega * omega
    return max(-10.0, min(10.0, raw))


def step(state: PendulumState, dt: float = 0.02) -> PendulumState:
    torque = controller(state.theta, state.omega)
    acceleration = torque - 9.81 * sin(state.theta) - 0.1 * state.omega
    omega = state.omega + dt * acceleration
    theta = state.theta + dt * omega
    return PendulumState(theta=theta, omega=omega, time=state.time + dt)


def main() -> None:
    state = PendulumState(theta=3.14, omega=0.0, time=0.0)
    for _ in range(50):
        state = step(state)
        print(
            f"time={state.time:.2f} "
            f"theta={state.theta:.3f} "
            f"omega={state.omega:.3f}"
        )


if __name__ == "__main__":
    main()
