# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.5",
#     "matplotlib>=3.10.0",
#     "regelum>=0.3.1",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import math
    from typing import cast

    import casadi as ca
    import marimo as mo
    import matplotlib.pyplot as plt

    import regelum as rg

    return ca, cast, math, mo, plt, rg


@app.cell
def _(ca, cast, math, rg):
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

    return (run_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Free pendulum

    The plant is a damped pendulum without external torque:

    \[
    \dot{\theta} = \omega,\qquad
    \dot{\omega} = -\frac{g}{\ell}\sin(\theta) - d\omega .
    \]

    The observer node publishes \(\sin(\theta)\), \(\cos(\theta)\), and
    angular velocity from the plant state.
    """)
    return


@app.cell
def _(plt, run_response):
    samples = run_response()
    time = [sample[0] for sample in samples]
    sin_angle = [sample[2] for sample in samples]
    cos_angle = [sample[3] for sample in samples]
    omega = [sample[4] for sample in samples]

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)
    axes[0].plot(time, sin_angle, label=r"$\sin\theta$", color="#2f6fed")
    axes[0].plot(time, cos_angle, label=r"$\cos\theta$", color="#d97706")
    axes[0].set_ylabel("observer output")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[1].plot(time, omega, label=r"$\omega$", color="#15803d")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("angular velocity [rad/s]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig
    return

if __name__ == "__main__":
    app.run()
