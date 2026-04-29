"""Adaptive-bitrate video player as a feedback control loop.

Each tick the player asks one question: do I have enough buffered video to
keep playing at the current quality?  If yes, the short branch just plays the
next chunk.  If no, the long branch first lowers the target bitrate, then
plays.  Buffer and bitrate persist across ticks, so the branching pattern is
driven by the closed loop between the network, the buffer, and the policy.

Phase graph::

    measure (init) -> decide -+-[healthy]--> play -> bottom
                              |
                              +-[stalling]-> drop_quality -> play -> bottom
"""

from __future__ import annotations

from regelum import (
    Else,
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

TICK_DT_SECONDS = 1.0
BITRATE_LADDER_KBPS = (240, 480, 720, 1080, 2160)
TOP_BITRATE_KBPS = BITRATE_LADDER_KBPS[-1]
STALL_HORIZON_SECONDS = 4.0


class Clock(Node):
    """Counts ticks; lets other nodes parameterize their behavior in time."""

    class Outputs(NodeOutputs):
        tick: int = Output(initial=0)

    def run(
        self,
        tick: int = Input(source=lambda: Clock.Outputs.tick),
    ) -> Outputs:
        return self.Outputs(tick=tick + 1)


class Network(Node):
    """Stochastic-looking but deterministic bandwidth model.

    Drops to a slow link in the middle of the run so the policy has to react,
    then recovers so the buffer can refill.  The schedule is fixed to keep the
    example reproducible.
    """

    class Inputs(NodeInputs):
        tick: int = Input(source=Clock.Outputs.tick)

    class Outputs(NodeOutputs):
        bandwidth_kbps: float = Output(initial=float(TOP_BITRATE_KBPS))

    def run(self, inputs: Inputs) -> Outputs:
        if inputs.tick < 6:
            value = 2400.0
        elif inputs.tick < 14:
            value = 600.0
        elif inputs.tick < 22:
            value = 1100.0
        else:
            value = 2400.0
        return self.Outputs(bandwidth_kbps=value)


class QualityPolicy(Node):
    """Decides whether the player is about to stall.

    The estimated drain rate is ``1 - bandwidth / bitrate`` seconds of video
    lost per wall-second.  If buffered seconds will not survive
    ``STALL_HORIZON_SECONDS`` at that drain rate, mark the tick as stalling.
    """

    class Inputs(NodeInputs):
        buffer_seconds: float = Input(source=lambda: MediaSession.Outputs.buffer_seconds)
        bitrate_kbps: int = Input(source=lambda: BitrateController.Outputs.value)
        bandwidth_kbps: float = Input(source=Network.Outputs.bandwidth_kbps)

    class Outputs(NodeOutputs):
        stalling: bool = Output(initial=False)

    def run(self, inputs: Inputs) -> Outputs:
        bitrate = max(inputs.bitrate_kbps, 1)
        drain = max(0.0, 1.0 - inputs.bandwidth_kbps / bitrate)
        if drain <= 0.0:
            return self.Outputs(stalling=False)
        time_to_empty = inputs.buffer_seconds / drain
        return self.Outputs(stalling=time_to_empty < STALL_HORIZON_SECONDS)


class BitrateController(Node):
    """Owns the current target bitrate.  Drops one rung when invoked."""

    class Inputs(NodeInputs):
        current: int = Input(source=lambda: BitrateController.Outputs.value)

    class Outputs(NodeOutputs):
        value: int = Output(initial=TOP_BITRATE_KBPS)

    def run(self, inputs: Inputs) -> Outputs:
        try:
            index = BITRATE_LADDER_KBPS.index(inputs.current)
        except ValueError:
            index = len(BITRATE_LADDER_KBPS) - 1
        next_index = max(0, index - 1)
        return self.Outputs(value=BITRATE_LADDER_KBPS[next_index])


class Decoder(Node):
    """Models how many seconds of video can be downloaded in one tick.

    With ``bandwidth_kbps`` of throughput and a video encoded at
    ``bitrate_kbps``, one wall-second of downloading produces
    ``bandwidth / bitrate`` seconds of playable content.
    """

    class Inputs(NodeInputs):
        bandwidth_kbps: float = Input(source=Network.Outputs.bandwidth_kbps)
        bitrate_kbps: int = Input(source=lambda: BitrateController.Outputs.value)

    class Outputs(NodeOutputs):
        fetched_seconds: float

    def run(self, inputs: Inputs) -> Outputs:
        bitrate = max(inputs.bitrate_kbps, 1)
        return self.Outputs(fetched_seconds=inputs.bandwidth_kbps / bitrate * TICK_DT_SECONDS)


class MediaSession(Node):
    """The plant.  Buffer fills with newly fetched video, drains with playback."""

    class Inputs(NodeInputs):
        previous: float = Input(source=lambda: MediaSession.Outputs.buffer_seconds)
        fetched: float = Input(source=Decoder.Outputs.fetched_seconds)

    class Outputs(NodeOutputs):
        buffer_seconds: float = Output(initial=10.0)

    def run(self, inputs: Inputs) -> Outputs:
        next_buffer = inputs.previous + inputs.fetched - TICK_DT_SECONDS
        return self.Outputs(buffer_seconds=max(0.0, next_buffer))


class Logger(Node):
    """Appends a per-tick record so the trajectory is visible after the run."""

    Sample = tuple[int, float, int, float, bool]

    class Inputs(NodeInputs):
        tick: int = Input(source=Clock.Outputs.tick)
        bandwidth_kbps: float = Input(source=Network.Outputs.bandwidth_kbps)
        bitrate_kbps: int = Input(source=lambda: BitrateController.Outputs.value)
        buffer_seconds: float = Input(source=lambda: MediaSession.Outputs.buffer_seconds)
        stalling: bool = Input(source=QualityPolicy.Outputs.stalling)
        history: list["Logger.Sample"] = Input(source=lambda: Logger.Outputs.history)

    class Outputs(NodeOutputs):
        history: list["Logger.Sample"] = Output(initial=lambda: [])

    def run(self, inputs: Inputs) -> Outputs:
        record: Logger.Sample = (
            inputs.tick,
            inputs.bandwidth_kbps,
            inputs.bitrate_kbps,
            inputs.buffer_seconds,
            inputs.stalling,
        )
        inputs.history.append(record)
        return self.Outputs(history=inputs.history)


def build_system() -> PhasedReactiveSystem:
    clock = Clock()
    network = Network()
    policy = QualityPolicy()
    controller = BitrateController()
    decoder = Decoder()
    session = MediaSession()
    logger = Logger()

    return PhasedReactiveSystem(
        phases=[
            Phase(
                "measure",
                nodes=(clock, network),
                transitions=(Goto("decide"),),
                is_initial=True,
            ),
            Phase(
                "decide",
                nodes=(policy,),
                transitions=(
                    If(V(policy.Outputs.stalling), "drop_quality", name="stalling"),
                    Else("play", name="healthy"),
                ),
            ),
            Phase(
                "drop_quality",
                nodes=(controller,),
                transitions=(Goto("play"),),
            ),
            Phase(
                "play",
                nodes=(decoder, session, logger),
                transitions=(Goto(terminate),),
            ),
        ],
    )


def main() -> None:
    system = build_system()
    print(f"compile_ok = {system.compile_report.ok}")
    print(
        "phase schedules: "
        + " | ".join(
            f"{name}={schedule}" for name, schedule in system.compile_report.phase_schedules.items()
        )
    )
    print()
    print("tick | bw(kbps) | bitrate | buffer(s) | stall? | path")
    print("-----+----------+---------+-----------+--------+----------------------")
    for _ in range(30):
        records = system.step()
        path_phases: list[str] = []
        for record in records:
            if not path_phases or path_phases[-1] != record.phase:
                path_phases.append(record.phase)
        path = " -> ".join(path_phases)
        snapshot = system.snapshot()
        print(
            f"{snapshot['Clock.tick']:4d} | "
            f"{snapshot['Network.bandwidth_kbps']:8.0f} | "
            f"{snapshot['BitrateController.value']:7d} | "
            f"{snapshot['MediaSession.buffer_seconds']:9.2f} | "
            f"{str(snapshot['QualityPolicy.stalling']):>6} | "
            f"{path}"
        )


if __name__ == "__main__":
    main()
