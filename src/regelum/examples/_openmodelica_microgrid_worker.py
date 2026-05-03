from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from functools import partial
from pathlib import Path

import gym  # ty: ignore[unresolved-import]
import numpy as np
from openmodelica_microgrid_gym.agents import StaticControlAgent  # ty: ignore[unresolved-import]
from openmodelica_microgrid_gym.aux_ctl import (  # ty: ignore[unresolved-import]
    DroopParams,
    InverseDroopParams,
    MultiPhaseDQ0PIPIController,
    MultiPhaseDQCurrentController,
    PI_params,
    PLLParams,
)
from openmodelica_microgrid_gym.net import Network  # ty: ignore[unresolved-import]


def load_step(t, gain):
    return 1 * gain if t < 0.2 else 2 * gain


def build_agent(net):
    droop_gain = 40000.0
    qdroop_gain = 1000.0
    delta_t = net.ts
    freq_nom = net.freq_nom
    v_nom = net.v_nom
    i_lim = net["inverter1"].i_lim

    master_voltage_pi = PI_params(kP=0.025, kI=60, limits=(-i_lim, i_lim))
    master_current_pi = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    master_droop = DroopParams(droop_gain, 0.005, freq_nom)
    master_qdroop = DroopParams(qdroop_gain, 0.002, v_nom)

    slave_current_pi = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    slave_pll = PLLParams(kP=10, kI=200, limits=None, f_nom=freq_nom)
    slave_droop = InverseDroopParams(droop_gain, delta_t, freq_nom, tau_filt=0.04)
    slave_qdroop = InverseDroopParams(50, delta_t, v_nom, tau_filt=0.01)

    controllers = [
        MultiPhaseDQ0PIPIController(
            master_voltage_pi,
            master_current_pi,
            master_droop,
            master_qdroop,
            ts_sim=delta_t,
            name="master",
        ),
        MultiPhaseDQCurrentController(
            slave_current_pi,
            slave_pll,
            i_lim,
            slave_droop,
            slave_qdroop,
            ts_sim=delta_t,
            name="slave",
        ),
    ]

    return StaticControlAgent(
        controllers,
        {
            "master": [
                [f"lc1.inductor{k}.i" for k in "123"],
                [f"lc1.capacitor{k}.v" for k in "123"],
            ],
            "slave": [
                [f"lcl1.inductor{k}.i" for k in "123"],
                [f"lcl1.capacitor{k}.v" for k in "123"],
                np.zeros(3),
            ],
        },
    )


class StaticDroopGymRuntime:
    def __init__(self, omg_root: Path, max_steps: int) -> None:
        self.omg_root = omg_root
        self.max_steps = max_steps
        self.net = Network.load(str(omg_root / "net" / "net.yaml"))
        self.env = gym.make(
            "ModelicaEnv_test-v1",
            viz_mode=None,
            log_level=logging.WARNING,
            max_episode_steps=max_steps,
            model_params={
                "rl1.resistor1.R": partial(load_step, gain=20),
                "rl1.resistor2.R": partial(load_step, gain=20),
                "rl1.resistor3.R": partial(load_step, gain=20),
                "rl1.inductor1.L": 0.001,
                "rl1.inductor2.L": 0.001,
                "rl1.inductor3.L": 0.001,
            },
            model_path=str(omg_root / "omg_grid" / "grid.network.fmu"),
            net=self.net,
        )
        self.base_env = self.env.unwrapped
        self.agent = build_agent(self.net)
        self.agent.env = self.base_env
        self.agent.obs_varnames = self.base_env.history.cols
        self.base_env.history.cols = (
            self.base_env.history.structured_cols(None) + self.agent.measurement_cols
        )
        self.base_env.measure = self.agent.measure
        self.cols = list(self.base_env.history.cols)
        self.voltage_indices = [
            self.cols.index("lcl1.capacitor1.v"),
            self.cols.index("lcl1.capacitor2.v"),
            self.cols.index("lcl1.capacitor3.v"),
        ]
        self.obs = None
        self.reward = 0
        self.done = False
        self.step_index = 0

    def reset(self):
        self.agent.reset()
        self.obs = self.env.reset()
        self.reward = 0
        self.done = False
        self.step_index = 0
        return self._sample()

    def step(self):
        if self.done or self.step_index >= self.max_steps:
            return self._sample()
        self.agent.observe(self.reward, self.done)
        action = self.agent.act(self.obs)
        self.obs, self.reward, self.done, _ = self.env.step(action)
        self.step_index += 1
        if self.step_index >= self.max_steps:
            self.done = True
        return self._sample()

    def _sample(self):
        row = self.base_env.history._data[-1]
        v1, v2, v3 = (float(row[index]) for index in self.voltage_indices)
        return {
            "step": self.step_index,
            "time": self.step_index * self.net.ts,
            "lcl1_capacitor1_v": v1,
            "lcl1_capacitor2_v": v2,
            "lcl1_capacitor3_v": v3,
            "done": bool(self.done),
            "history_rows": len(self.base_env.history._data),
        }

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def emit(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omg-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    runtime = StaticDroopGymRuntime(args.omg_root.resolve(), args.max_steps)
    emit({"ok": True, "event": "ready", "ts": runtime.net.ts})

    try:
        for line in sys.stdin:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                emit({"ok": True, "sample": runtime.reset()})
            elif command == "step":
                emit({"ok": True, "sample": runtime.step()})
            elif command == "close":
                runtime.close()
                emit({"ok": True, "event": "closed"})
                return 0
            else:
                emit({"ok": False, "error": f"unknown command: {command!r}"})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"ok": False, "error": str(exc)})
        return 1
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
