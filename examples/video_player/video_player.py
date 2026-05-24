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

import regelum as rg

TICK_DT_SECONDS = 1.0
BITRATE_LADDER_KBPS = (240, 480, 720, 1080, 2160)
TOP_BITRATE_KBPS = BITRATE_LADDER_KBPS[-1]
STALL_HORIZON_SECONDS = 4.0


class Network(rg.Node):
    """Stochastic-looking but deterministic bandwidth model.

    Drops to a slow link in the middle of the run so the policy has to react,
    then recovers so the buffer can refill.  The schedule is fixed to keep the
    example reproducible.
    """

    class Inputs(rg.NodeInputs):
        tick: int = rg.src(rg.Clock.tick)

    class State(rg.NodeState):
        bandwidth_kbps: float = rg.var(init=float(TOP_BITRATE_KBPS))

    def update(self, inputs: Inputs) -> State:
        if inputs.tick < 6:
            value = 2400.0
        elif inputs.tick < 14:
            value = 600.0
        elif inputs.tick < 22:
            value = 1100.0
        else:
            value = 2400.0
        return self.State(bandwidth_kbps=value)


class QualityPolicy(rg.Node):
    """Decides whether the player is about to stall.

    The estimated drain rate is ``1 - bandwidth / bitrate`` seconds of video
    lost per wall-second.  If buffered seconds will not survive
    ``STALL_HORIZON_SECONDS`` at that drain rate, mark the tick as stalling.
    """

    class Inputs(rg.NodeInputs):
        buffer_seconds: float = rg.src(lambda: MediaSession.State.buffer_seconds)
        bitrate_kbps: int = rg.src(lambda: BitrateController.State.value)
        bandwidth_kbps: float = rg.src(Network.State.bandwidth_kbps)

    class State(rg.NodeState):
        stalling: bool = rg.var(init=False)

    def update(self, inputs: Inputs) -> State:
        bitrate = max(inputs.bitrate_kbps, 1)
        drain = max(0.0, 1.0 - inputs.bandwidth_kbps / bitrate)
        if drain <= 0.0:
            return self.State(stalling=False)
        time_to_empty = inputs.buffer_seconds / drain
        return self.State(stalling=time_to_empty < STALL_HORIZON_SECONDS)


class BitrateController(rg.Node):
    """Owns the current target bitrate.  Drops one rung when invoked."""

    class Inputs(rg.NodeInputs):
        current: int = rg.src(lambda: BitrateController.State.value)

    class State(rg.NodeState):
        value: int = rg.var(init=TOP_BITRATE_KBPS)

    def update(self, inputs: Inputs) -> State:
        try:
            index = BITRATE_LADDER_KBPS.index(inputs.current)
        except ValueError:
            index = len(BITRATE_LADDER_KBPS) - 1
        next_index = max(0, index - 1)
        return self.State(value=BITRATE_LADDER_KBPS[next_index])


class Decoder(rg.Node):
    """Models how many seconds of video can be downloaded in one tick.

    With ``bandwidth_kbps`` of throughput and a video encoded at
    ``bitrate_kbps``, one wall-second of downloading produces
    ``bandwidth / bitrate`` seconds of playable content.
    """

    class Inputs(rg.NodeInputs):
        bandwidth_kbps: float = rg.src(Network.State.bandwidth_kbps)
        bitrate_kbps: int = rg.src(lambda: BitrateController.State.value)

    class State(rg.NodeState):
        fetched_seconds: float

    def update(self, inputs: Inputs) -> State:
        bitrate = max(inputs.bitrate_kbps, 1)
        return self.State(fetched_seconds=inputs.bandwidth_kbps / bitrate * TICK_DT_SECONDS)


class MediaSession(rg.Node):
    """The plant.  Buffer fills with newly fetched video, drains with playback."""

    class Inputs(rg.NodeInputs):
        previous: float = rg.src(lambda: MediaSession.State.buffer_seconds)
        fetched: float = rg.src(Decoder.State.fetched_seconds)

    class State(rg.NodeState):
        buffer_seconds: float = rg.var(init=10.0)

    def update(self, inputs: Inputs) -> State:
        next_buffer = inputs.previous + inputs.fetched - TICK_DT_SECONDS
        return self.State(buffer_seconds=max(0.0, next_buffer))


class Logger(rg.Node):
    """Appends a per-tick record so the trajectory is visible after the run."""

    Sample = tuple[int, float, int, float, bool]

    class Inputs(rg.NodeInputs):
        tick: int = rg.src(rg.Clock.tick)
        bandwidth_kbps: float = rg.src(Network.State.bandwidth_kbps)
        bitrate_kbps: int = rg.src(lambda: BitrateController.State.value)
        buffer_seconds: float = rg.src(lambda: MediaSession.State.buffer_seconds)
        stalling: bool = rg.src(QualityPolicy.State.stalling)
        history: list["Logger.Sample"] = rg.src(lambda: Logger.State.history)

    class State(rg.NodeState):
        history: list["Logger.Sample"] = rg.var(init=lambda: [])

    def update(self, inputs: Inputs) -> State:
        record: Logger.Sample = (
            inputs.tick,
            inputs.bandwidth_kbps,
            inputs.bitrate_kbps,
            inputs.buffer_seconds,
            inputs.stalling,
        )
        inputs.history.append(record)
        return self.State(history=inputs.history)


def build_system() -> rg.PhasedReactiveSystem:
    network = Network()
    policy = QualityPolicy()
    controller = BitrateController()
    decoder = Decoder()
    session = MediaSession()
    logger = Logger()

    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "measure",
                nodes=(network,),
                transitions=(rg.Goto("decide"),),
                is_initial=True,
            ),
            rg.Phase(
                "decide",
                nodes=(policy,),
                transitions=(
                    rg.If(rg.V(policy.State.stalling), "drop_quality", name="stalling"),
                    rg.Else("play", name="healthy"),
                ),
            ),
            rg.Phase(
                "drop_quality",
                nodes=(controller,),
                transitions=(rg.Goto("play"),),
            ),
            rg.Phase(
                "play",
                nodes=(decoder, session, logger),
                transitions=(rg.Goto(rg.terminate),),
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
            f"{system.read(rg.Clock.tick):4d} | "
            f"{snapshot['Network.bandwidth_kbps']:8.0f} | "
            f"{snapshot['BitrateController.value']:7d} | "
            f"{snapshot['MediaSession.buffer_seconds']:9.2f} | "
            f"{str(snapshot['QualityPolicy.stalling']):>6} | "
            f"{path}"
        )


if __name__ == "__main__":
    main()
