from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from regelum import (
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

VoltageSample = tuple[float, float, float, float]


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "regelum").exists():
            return parent
    raise RuntimeError("Could not locate the regelum repository root.")


def default_omg_root() -> Path:
    return Path(os.environ.get("OMG_ROOT", repo_root() / "references" / "openmodelica-microgrid-gym"))


def default_worker_python(omg_root: Path) -> Path:
    configured = os.environ.get("OMG_PYTHON")
    if configured:
        return Path(configured)
    return omg_root / ".venv-omg38" / "bin" / "python"


class StaticDroopGymBridge:
    def __init__(
        self,
        *,
        omg_root: Path,
        python: Path,
        max_steps: int,
    ) -> None:
        self.omg_root = omg_root
        self.python = python
        self.max_steps = max_steps
        self.process: subprocess.Popen[str] | None = None
        self.ts = 0.0

    def start(self) -> None:
        if self.process is not None:
            return
        if not self.python.exists():
            raise FileNotFoundError(f"OpenModelica worker Python not found: {self.python}")
        if not self.omg_root.exists():
            raise FileNotFoundError(f"openmodelica-microgrid-gym root not found: {self.omg_root}")

        env = os.environ.copy()
        fmil_home = self.omg_root / ".local" / "fmil"
        env["FMIL_HOME"] = str(fmil_home)
        env["LD_LIBRARY_PATH"] = (
            f"{fmil_home / 'lib'}:{env['LD_LIBRARY_PATH']}"
            if env.get("LD_LIBRARY_PATH")
            else str(fmil_home / "lib")
        )
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = (
            f"{self.omg_root}:{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(self.omg_root)
        )

        worker = Path(__file__).with_name("_openmodelica_microgrid_worker.py")
        self.process = subprocess.Popen(
            [
                str(self.python),
                "-u",
                str(worker),
                "--omg-root",
                str(self.omg_root),
                "--max-steps",
                str(self.max_steps),
            ],
            cwd=str(self.omg_root / "examples"),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        response = self._read_response()
        self.ts = float(response["ts"])

    def reset(self) -> dict[str, Any]:
        return self._request({"cmd": "reset"})["sample"]

    def step(self) -> dict[str, Any]:
        return self._request({"cmd": "step"})["sample"]

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self._request({"cmd": "close"})
            except Exception:
                self.process.terminate()
        self.process = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self.process is not None
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self._read_response()

    def _read_response(self) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(f"OpenModelica worker stopped before responding.\n{stderr}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "OpenModelica worker failed"))
        return response


class OpenModelicaStaticDroopGymNode(Node):
    def __init__(
        self,
        *,
        omg_root: Path | None = None,
        python: Path | None = None,
        max_steps: int = 1000,
    ) -> None:
        self.max_steps = max_steps
        resolved_omg_root = (omg_root or default_omg_root()).resolve()
        self.bridge = StaticDroopGymBridge(
            omg_root=resolved_omg_root,
            python=python or default_worker_python(resolved_omg_root),
            max_steps=max_steps,
        )
        self._started = False
        self._completed_steps = 0

    class Inputs(NodeInputs):
        samples: tuple[VoltageSample, ...] = Input(
            source="OpenModelicaStaticDroopGymNode.Outputs.samples"
        )

    class Outputs(NodeOutputs):
        step: int = Output(initial=0)
        time: float = Output(initial=0.0)
        lcl1_capacitor1_v: float = Output(initial=0.0)
        lcl1_capacitor2_v: float = Output(initial=0.0)
        lcl1_capacitor3_v: float = Output(initial=0.0)
        done: bool = Output(initial=False)
        samples: tuple[VoltageSample, ...] = Output(initial=())

    def run(self, inputs: Inputs) -> Outputs:
        samples = inputs.samples
        if not self._started:
            initial = self.bridge.reset()
            samples = samples + (self._as_tuple(initial),)
            self._started = True

        sample = self.bridge.step()
        self._completed_steps = int(sample["step"])
        samples = samples + (self._as_tuple(sample),)

        return self.Outputs(
            step=self._completed_steps,
            time=float(sample["time"]),
            lcl1_capacitor1_v=float(sample["lcl1_capacitor1_v"]),
            lcl1_capacitor2_v=float(sample["lcl1_capacitor2_v"]),
            lcl1_capacitor3_v=float(sample["lcl1_capacitor3_v"]),
            done=bool(sample["done"]) or self._completed_steps >= self.max_steps,
            samples=samples,
        )

    def close(self) -> None:
        self.bridge.close()

    @staticmethod
    def _as_tuple(sample: dict[str, Any]) -> VoltageSample:
        return (
            float(sample["time"]),
            float(sample["lcl1_capacitor1_v"]),
            float(sample["lcl1_capacitor2_v"]),
            float(sample["lcl1_capacitor3_v"]),
        )


def build_system(max_steps: int = 1000) -> tuple[PhasedReactiveSystem, OpenModelicaStaticDroopGymNode]:
    microgrid = OpenModelicaStaticDroopGymNode(max_steps=max_steps)
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "simulate-openmodelica-gym",
                nodes=(microgrid,),
                transitions=(
                    If(~V("OpenModelicaStaticDroopGymNode.done"), "simulate-openmodelica-gym"),
                    If(V("OpenModelicaStaticDroopGymNode.done"), terminate),
                ),
                is_initial=True,
            ),
        ],
        max_phase_steps=max_steps + 2,
        strict=False,
    )
    return system, microgrid


def save_lcl1_plot(samples: tuple[VoltageSample, ...], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    time = [row[0] for row in samples]
    v1 = [row[1] for row in samples]
    v2 = [row[2] for row in samples]
    v3 = [row[3] for row in samples]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    ax.plot(time, v1, label="lcl1.capacitor1.v")
    ax.plot(time, v2, label="lcl1.capacitor2.v")
    ax.plot(time, v3, label="lcl1.capacitor3.v")
    ax.set_xlim(0, 0.05)
    ax.legend()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root()
        / "artifacts"
        / "openmodelica_microgrid_gym"
        / "regelum_static_droop_lcl1_capacitor_voltages.png",
    )
    args = parser.parse_args()

    system, microgrid = build_system(max_steps=args.steps)
    try:
        system.run(steps=1)
        snapshot = system.snapshot()
        samples = snapshot["OpenModelicaStaticDroopGymNode.samples"]
        save_lcl1_plot(samples, args.output)
        print(f"steps={snapshot['OpenModelicaStaticDroopGymNode.step']}")
        print(f"samples={len(samples)}")
        print(args.output.resolve())
    finally:
        microgrid.close()


if __name__ == "__main__":
    main()
