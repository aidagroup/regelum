from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast

import casadi as ca
import numpy as np

import regelum as rg


class PendulumODE(rg.ODENode):
    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81

    def __init__(self, theta0: float, omega0: float) -> None:
        self.theta0 = theta0
        self.omega0 = omega0

    class State(rg.NodeState):
        theta: float = rg.var(init=lambda self: cast(PendulumODE, self).theta0)
        omega: float = rg.var(init=lambda self: cast(PendulumODE, self).omega0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        state: State,
        tau: float = rg.src(lambda: Controller.State.tau),
    ) -> State:
        tau_c = 3.0 / (self.mass * self.length**2)
        g_c = (3.0 * self.gravity) / (2.0 * self.length)
        return self.State(
            theta=state.omega,
            omega=g_c * ca.sin(state.theta) + tau_c * tau,
        )


class Observer(rg.Node):
    class State(rg.NodeState):
        sin_theta: float
        cos_theta: float
        omega: float

    class Inputs(rg.NodeInputs):
        theta: float = rg.src(PendulumODE.State.theta)
        omega: float = rg.src(PendulumODE.State.omega)

    def update(self, inputs: Inputs) -> State:
        return self.State(
            sin_theta=math.sin(inputs.theta),
            cos_theta=math.cos(inputs.theta),
            omega=inputs.omega,
        )


class Controller(rg.Node):
    dt: str = "0.05"
    tau_max: float = 4.0

    def __init__(self, kp: float, kd: float) -> None:
        self.kp = kp
        self.kd = kd

    class State(rg.NodeState):
        tau: float

    def update(
        self,
        sin_theta: float = rg.src(Observer.State.sin_theta),
        cos_theta: float = rg.src(Observer.State.cos_theta),
        omega: float = rg.src(Observer.State.omega),
    ) -> State:
        theta = math.atan2(sin_theta, cos_theta)
        raw = -self.kp * theta - self.kd * omega
        tau = float(np.clip(raw, -self.tau_max, self.tau_max))
        return self.State(tau=tau)


class Logger(rg.Node):
    class State(rg.NodeState):
        samples: list[tuple[float, float, float, float]] = rg.var(init=list)

    def update(
        self,
        state: State,
        time: float = rg.src(rg.Clock.time),
        theta: float = rg.src(PendulumODE.State.theta),
        omega: float = rg.src(PendulumODE.State.omega),
        tau: float = rg.src(Controller.State.tau),
    ) -> State:
        state.samples.append((time, theta, omega, tau))
        return self.State(samples=state.samples)


def plot(
    samples: list[tuple[float, float, float, float]],
    *,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    time = [sample[0] for sample in samples]
    theta = [sample[1] for sample in samples]
    omega = [sample[2] for sample in samples]
    tau = [sample[3] for sample in samples]

    plt.style.use("default")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True)

    axes[0].plot(time, theta, label=r"$\theta$", color="#2563eb", linewidth=1.8)
    axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9, label="target")
    axes[0].set_ylabel(r"$\theta$ [rad]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(time, omega, label=r"$\omega$", color="#16a34a", linewidth=1.8)
    axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel(r"$\omega$ [rad/s]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", frameon=False)

    axes[2].plot(time, tau, drawstyle="steps-post", label=r"$\tau$", color="#dc2626", linewidth=1.8)
    axes[2].axhline(4.0, color="#6b7280", linestyle=":", linewidth=0.9)
    axes[2].axhline(-4.0, color="#6b7280", linestyle=":", linewidth=0.9)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel(r"$\tau$ [N m]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right", frameon=False)

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run(
    theta0: float = math.pi,
    omega0: float = 0.0,
    kp: float = 14.0,
    kd: float = 4.0,
    steps: int = 700,
) -> list[tuple[float, float, float, float]]:
    pendulum = PendulumODE(theta0=theta0, omega0=omega0)
    plant = rg.ODESystem(nodes=(pendulum,), dt="0.01")
    observer = Observer()
    controller = Controller(kp=kp, kd=kd)
    logger = Logger()

    system = rg.PhasedReactiveSystem(
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
    )
    system.run(steps)
    return cast(list[tuple[float, float, float, float]], system.read(Logger.State.samples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled pendulum example.")
    parser.add_argument("--theta0", type=float, default=math.pi)
    parser.add_argument("--omega0", type=float, default=0.0)
    parser.add_argument("--kp", type=float, default=14.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    samples = run(
        theta0=args.theta0,
        omega0=args.omega0,
        kp=args.kp,
        kd=args.kd,
        steps=args.steps,
    )
    plot(samples, output_path=args.output, show=args.show)
    time, theta, omega, tau = samples[-1]
    print(f"time={time:.2f}")
    print(f"theta={theta:.6f}")
    print(f"omega={omega:.6f}")
    print(f"tau={tau:.6f}")


if __name__ == "__main__":
    main()
