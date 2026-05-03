from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from examples.native_one_to_one_microgrid import build_system as build_native_system
from examples.native_two_inverter_microgrid import VoltageSample, repo_root
from examples.openmodelica_microgrid_static_droop import build_system as build_fmu_system


def run_fmu_baseline(steps: int) -> tuple[VoltageSample, ...]:
    system, node = build_fmu_system(max_steps=steps)
    try:
        system.run(steps=1)
        return system.snapshot()["OpenModelicaStaticDroopGymNode.samples"]
    finally:
        node.close()


def run_native_one_to_one(steps: int) -> tuple[VoltageSample, ...]:
    system = build_native_system()
    system.run(steps)
    return system.snapshot()["OneToOneMicrogridLogger.samples"]


def compare(
    fmu_samples: tuple[VoltageSample, ...],
    native_samples: tuple[VoltageSample, ...],
    *,
    fmu_offset: int = 1,
) -> dict[str, dict[str, float]]:
    pairs = []
    for index, native in enumerate(native_samples):
        fmu_index = index + fmu_offset
        if 0 <= fmu_index < len(fmu_samples):
            pairs.append((native, fmu_samples[fmu_index]))

    metrics: dict[str, dict[str, float]] = {}
    for column, name in enumerate(("lcl1.capacitor1.v", "lcl1.capacitor2.v", "lcl1.capacitor3.v"), start=1):
        diffs = [native[column] - fmu[column] for native, fmu in pairs]
        abs_diffs = [abs(value) for value in diffs]
        metrics[name] = {
            "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
            "max_abs": max(abs_diffs),
        }
    return metrics


def save_overlay(
    fmu_samples: tuple[VoltageSample, ...],
    native_samples: tuple[VoltageSample, ...],
    output: Path,
    *,
    fmu_offset: int = 1,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), dpi=120, sharex=True)
    labels = ("lcl1.capacitor1.v", "lcl1.capacitor2.v", "lcl1.capacitor3.v")
    for column, label in enumerate(labels, start=1):
        native_time = [sample[0] for sample in native_samples]
        native_values = [sample[column] for sample in native_samples]
        fmu_aligned = fmu_samples[fmu_offset : fmu_offset + len(native_samples)]
        fmu_time = [sample[0] for sample in fmu_aligned]
        fmu_values = [sample[column] for sample in fmu_aligned]
        axes[column - 1].plot(fmu_time, fmu_values, label=f"FMU {label}", linewidth=1.4)
        axes[column - 1].plot(native_time, native_values, "--", label=f"native {label}", linewidth=1.1)
        axes[column - 1].legend(loc="upper right", fontsize=8)
        axes[column - 1].grid(True, alpha=0.25)
    axes[-1].set_xlim(0, 0.05)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fmu-offset", type=int, default=1)
    parser.add_argument("--rmse-tol", type=float, default=0.01)
    parser.add_argument("--max-abs-tol", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root()
        / "artifacts"
        / "openmodelica_microgrid_gym"
        / "native_one_to_one_vs_fmu_overlay.png",
    )
    args = parser.parse_args()

    fmu_samples = run_fmu_baseline(args.steps)
    native_samples = run_native_one_to_one(args.steps)
    metrics = compare(fmu_samples, native_samples, fmu_offset=args.fmu_offset)
    save_overlay(fmu_samples, native_samples, args.output, fmu_offset=args.fmu_offset)

    print(f"fmu_samples={len(fmu_samples)}")
    print(f"native_samples={len(native_samples)}")
    for signal, values in metrics.items():
        print(f"{signal}: rmse={values['rmse']:.6f}, max_abs={values['max_abs']:.6f}")
    print(args.output.resolve())

    failed = [
        signal
        for signal, values in metrics.items()
        if values["rmse"] > args.rmse_tol or values["max_abs"] > args.max_abs_tol
    ]
    if failed:
        raise SystemExit(f"Native one-to-one comparison failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
