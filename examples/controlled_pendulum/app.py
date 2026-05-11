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

    return (run_response,)


@app.cell
def _(mo):
    mo.md(r"""
    # Controlled pendulum

    The second plant adds an external torque \(\tau\):

    \[
    \dot{\theta} = \omega,\qquad
    \dot{\omega} =
    -\frac{g}{\ell}\sin(\theta) - d\omega + \frac{\tau}{m\ell^2}.
    \]

    The plant integrates at `dt=0.01`. The controller runs at `dt=0.05`, so
    its torque state is sampled and held between controller updates.
    """)
    return


@app.cell
def _(math, plt, run_response):
    samples = run_response()
    time = [sample[0] for sample in samples]
    theta = [sample[1] for sample in samples]
    omega = [sample[2] for sample in samples]
    torque = [sample[3] for sample in samples]

    plt.style.use("default")
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)
    axes[0].plot(time, theta, label=r"$\theta$", color="#7c3aed")
    axes[0].axhline(math.pi, color="#111318", linestyle="--", linewidth=1.0, label=r"$\pi$")
    axes[0].set_ylabel("angle [rad]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[1].plot(time, omega, label=r"$\omega$", color="#15803d")
    axes[1].set_ylabel("angular velocity [rad/s]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[2].plot(time, torque, drawstyle="steps-post", label=r"$\tau$", color="#d97706")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("torque [N m]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
