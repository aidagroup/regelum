from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from regelum.examples.dubuisson2019_water_treatment import (
    _dump_load_profile_kw,
    _dump_load_wind_power_profile_kw,
    _fig9_digitized_load_power_profile_kw,
    _fig9_digitized_wind_power_profile_kw,
    _first_time,
    _run_trace,
    _scaled_profile,
    _write_plot_document,
    export_paper_figures,
)

DEFAULT_TARGET_DIR = Path("references/dubuisson2019_targets")
DEFAULT_OUTPUT_DIR = Path("artifacts/dubuisson2019")


@dataclass(frozen=True)
class CalibrationParams:
    fig9_battery_capacity_kwh: float = 95.5
    fig9_initial_soc_percent: float = 69.92
    fig9_wind_scale: float = 1.0
    fig9_load_scale: float = 1.0
    fig9_battery_nominal_voltage_v: float = 250.0
    fig9_wind_converter_voltage_v: float = 520.0
    fig9_nominal_mppt_efficiency: float = 0.98
    fig9_max_charge_kw: float = 47.0
    fig9_max_discharge_kw: float = 45.0
    fig9_dc_nominal_voltage_v: float = 348.0
    fig9_dc_battery_power_gain_v_per_kw: float = 0.03
    fig9_dc_wind_power_gain_v_per_kw: float = 0.04
    fig9_dc_load_power_gain_v_per_kw: float = -0.02
    fig9_dc_unserved_power_gain_v_per_kw: float = 1.8
    fig9_dc_load_step_gain_v_per_kw: float = -0.25
    fig9_dc_diesel_step_gain_v_per_kw: float = 0.25
    fig9_dc_wind_step_gain_v_per_kw: float = 0.04
    fig9_dc_transient_decay: float = 0.88
    fig9_dc_response: float = 0.20
    fig9_nominal_frequency_hz: float = 60.095
    fig9_dc_frequency_gain_hz_per_v: float = 0.012
    fig9_unserved_frequency_gain_hz_per_kw: float = 0.002
    fig9_frequency_response: float = 0.18
    fig9_inverter_efficiency: float = 0.97
    fig11_initial_soc_percent: float = 99.7425
    fig11_battery_capacity_kwh: float = 10.0
    fig11_wind_scale: float = 1.0
    fig11_load_scale: float = 1.0


@dataclass(frozen=True)
class TargetPoint:
    figure: str
    channel: str
    time: float
    value: float
    weight: float = 1.0


@dataclass(frozen=True)
class ChannelSpec:
    figure: str
    channel: str
    simulation_key: str
    scale: float = 1.0


CHANNEL_SPECS = {
    ("fig9", "battery_current_a"): ChannelSpec("fig9", "battery_current_a", "battery_current_a"),
    ("fig9", "wind_current_a"): ChannelSpec("fig9", "wind_current_a", "wind_current_a"),
    ("fig9", "soc_percent"): ChannelSpec("fig9", "soc_percent", "soc_percent"),
    ("fig9", "dc_bus_voltage_v"): ChannelSpec("fig9", "dc_bus_voltage_v", "dc_bus_voltage_v"),
    ("fig9", "frequency_hz"): ChannelSpec("fig9", "frequency_hz", "frequency_hz"),
    ("fig11", "battery_current_a"): ChannelSpec("fig11", "battery_current_a", "battery_current_a"),
    ("fig11", "wind_current_a"): ChannelSpec("fig11", "wind_current_a", "wind_current_a"),
    ("fig11", "dump_load_current_a"): ChannelSpec(
        "fig11", "dump_load_current_a", "dump_load_power_kw", 4.0
    ),
    ("fig11", "soc_percent"): ChannelSpec("fig11", "soc_percent", "soc_percent"),
    ("fig11", "dc_bus_voltage_v"): ChannelSpec("fig11", "dc_bus_voltage_v", "dc_bus_voltage_v"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Dubuisson 2019 Regelum traces.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--init-targets-only",
        action="store_true",
        help="Create editable target CSV templates and exit.",
    )
    args = parser.parse_args()

    ensure_target_templates(args.target_dir)
    if args.init_targets_only:
        print(f"Target templates are ready in {args.target_dir}")
        return

    targets = read_targets(args.target_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, loss = calibrate(targets, args.output_dir, iterations=args.iterations)
    write_params(args.output_dir / "calibration_params.json", params)
    write_residual_report(args.output_dir / "calibration_report.md", targets, params, loss)
    export_overlay(args.output_dir / "calibration_overlay.pdf", targets, params)
    export_paper_figures(args.output_dir)
    print(f"Best loss: {loss:.6f}")
    print(f"Wrote calibration params to {args.output_dir / 'calibration_params.json'}")


def ensure_target_templates(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    templates = _template_targets()
    for (figure, channel), points in templates.items():
        path = target_dir / f"{figure}_{channel}.csv"
        if path.exists():
            continue
        _write_target_csv(path, points)


def read_targets(target_dir: Path) -> list[TargetPoint]:
    targets: list[TargetPoint] = []
    for path in sorted(target_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                targets.append(
                    TargetPoint(
                        figure=row.get("figure") or _figure_from_filename(path),
                        channel=row.get("channel") or _channel_from_filename(path),
                        time=float(row["time"]),
                        value=float(row["value"]),
                        weight=float(row.get("weight") or 1.0),
                    )
                )
    if not targets:
        raise RuntimeError(f"No target CSV files found in {target_dir}")
    return targets


def calibrate(
    targets: list[TargetPoint],
    output_dir: Path,
    *,
    iterations: int,
) -> tuple[CalibrationParams, float]:
    current = _read_existing_params(output_dir / "calibration_params.json")
    current_loss = loss_for_params(targets, current)
    steps = {
        "fig9_battery_capacity_kwh": 5.0,
        "fig9_initial_soc_percent": 0.015,
        "fig9_wind_scale": 0.04,
        "fig9_load_scale": 0.04,
        "fig9_battery_nominal_voltage_v": 25.0,
        "fig9_wind_converter_voltage_v": 25.0,
        "fig9_nominal_mppt_efficiency": 0.01,
        "fig9_max_charge_kw": 3.0,
        "fig9_max_discharge_kw": 3.0,
        "fig9_dc_nominal_voltage_v": 1.5,
        "fig9_dc_battery_power_gain_v_per_kw": 0.02,
        "fig9_dc_wind_power_gain_v_per_kw": 0.02,
        "fig9_dc_load_power_gain_v_per_kw": 0.02,
        "fig9_dc_unserved_power_gain_v_per_kw": 0.30,
        "fig9_dc_load_step_gain_v_per_kw": 0.05,
        "fig9_dc_diesel_step_gain_v_per_kw": 0.05,
        "fig9_dc_wind_step_gain_v_per_kw": 0.02,
        "fig9_dc_transient_decay": 0.04,
        "fig9_dc_response": 0.04,
        "fig9_nominal_frequency_hz": 0.01,
        "fig9_dc_frequency_gain_hz_per_v": 0.004,
        "fig9_unserved_frequency_gain_hz_per_kw": 0.001,
        "fig9_frequency_response": 0.04,
        "fig9_inverter_efficiency": 0.01,
        "fig11_initial_soc_percent": 0.02,
        "fig11_battery_capacity_kwh": 1.5,
        "fig11_wind_scale": 0.04,
        "fig11_load_scale": 0.04,
    }
    bounds = {
        "fig9_battery_capacity_kwh": (60.0, 130.0),
        "fig9_initial_soc_percent": (69.85, 69.99),
        "fig9_wind_scale": (0.65, 1.35),
        "fig9_load_scale": (0.65, 1.35),
        "fig9_battery_nominal_voltage_v": (160.0, 620.0),
        "fig9_wind_converter_voltage_v": (360.0, 760.0),
        "fig9_nominal_mppt_efficiency": (0.90, 1.00),
        "fig9_max_charge_kw": (35.0, 75.0),
        "fig9_max_discharge_kw": (20.0, 75.0),
        "fig9_dc_nominal_voltage_v": (340.0, 353.0),
        "fig9_dc_battery_power_gain_v_per_kw": (-0.25, 0.25),
        "fig9_dc_wind_power_gain_v_per_kw": (-0.25, 0.25),
        "fig9_dc_load_power_gain_v_per_kw": (-0.25, 0.25),
        "fig9_dc_unserved_power_gain_v_per_kw": (0.0, 3.0),
        "fig9_dc_load_step_gain_v_per_kw": (-1.0, 1.0),
        "fig9_dc_diesel_step_gain_v_per_kw": (-1.0, 1.0),
        "fig9_dc_wind_step_gain_v_per_kw": (-0.4, 0.4),
        "fig9_dc_transient_decay": (0.20, 0.98),
        "fig9_dc_response": (0.02, 0.80),
        "fig9_nominal_frequency_hz": (59.95, 60.15),
        "fig9_dc_frequency_gain_hz_per_v": (-0.04, 0.04),
        "fig9_unserved_frequency_gain_hz_per_kw": (-0.02, 0.02),
        "fig9_frequency_response": (0.02, 0.80),
        "fig9_inverter_efficiency": (0.85, 1.0),
        "fig11_initial_soc_percent": (99.55, 99.99),
        "fig11_battery_capacity_kwh": (4.0, 30.0),
        "fig11_wind_scale": (0.75, 1.25),
        "fig11_load_scale": (0.75, 1.25),
    }

    for _ in range(iterations):
        improved = True
        while improved:
            improved = False
            for name, step in steps.items():
                candidate_params = [
                    _with_param(
                        current, name, _clamped(getattr(current, name) - step, *bounds[name])
                    ),
                    _with_param(
                        current, name, _clamped(getattr(current, name) + step, *bounds[name])
                    ),
                ]
                for candidate in candidate_params:
                    candidate_loss = loss_for_params(targets, candidate)
                    if candidate_loss + 1e-12 < current_loss:
                        current = candidate
                        current_loss = candidate_loss
                        improved = True
        steps = {name: step * 0.5 for name, step in steps.items()}
    return current, current_loss


def loss_for_params(targets: list[TargetPoint], params: CalibrationParams) -> float:
    traces = simulate(params)
    grouped = _targets_by_channel(targets)
    total = 0.0
    total_weight = 0.0
    for key, points in grouped.items():
        spec = CHANNEL_SPECS.get(key)
        if spec is None:
            continue
        trace = traces[spec.figure]
        series = [
            (float(sample["time"]), float(sample[spec.simulation_key]) * spec.scale)
            for sample in trace
        ]
        normalizer = _target_range(points)
        for point in points:
            residual = (_interp(series, point.time) - point.value) / normalizer
            total += point.weight * residual * residual
            total_weight += point.weight
    fig9_dg_off = _first_time(
        traces["fig9"], lambda sample: float(sample["diesel_power_kw"]) == 0.0
    )
    fig11_dump_on = _first_time(
        traces["fig11"], lambda sample: float(sample["dump_load_power_kw"]) > 0.0
    )
    event_residuals = (
        (fig9_dg_off - 10.7) / 0.25,
        (fig11_dump_on - 6.65) / 0.05,
    )
    for residual in event_residuals:
        total += 25.0 * residual * residual
        total_weight += 25.0
    return total / max(total_weight, 1.0)


def simulate(params: CalibrationParams) -> dict[str, list[dict[str, float | str | bool]]]:
    fig9_wind_profile = _fig9_digitized_wind_power_profile_kw(
        converter_voltage_v=params.fig9_wind_converter_voltage_v,
        nominal_mppt_efficiency=params.fig9_nominal_mppt_efficiency,
    )
    fig9_load_profile = _fig9_digitized_load_power_profile_kw(
        battery_voltage_v=params.fig9_battery_nominal_voltage_v,
        wind_converter_voltage_v=params.fig9_wind_converter_voltage_v,
    )
    fig9 = _run_trace(
        dt=0.02,
        duration_s=18.0,
        init_time=2.0,
        init_soc_percent=params.fig9_initial_soc_percent,
        effective_capacity_kwh=params.fig9_battery_capacity_kwh,
        wind_power_profile_kw=_scaled_profile(fig9_wind_profile, params.fig9_wind_scale),
        load_profile_kw=_scaled_profile(fig9_load_profile, params.fig9_load_scale),
        battery_nominal_voltage_v=params.fig9_battery_nominal_voltage_v,
        wind_converter_voltage_v=params.fig9_wind_converter_voltage_v,
        max_charge_kw=params.fig9_max_charge_kw,
        max_discharge_kw=params.fig9_max_discharge_kw,
        dc_nominal_voltage_v=params.fig9_dc_nominal_voltage_v,
        dc_battery_power_gain_v_per_kw=params.fig9_dc_battery_power_gain_v_per_kw,
        dc_wind_power_gain_v_per_kw=params.fig9_dc_wind_power_gain_v_per_kw,
        dc_load_power_gain_v_per_kw=params.fig9_dc_load_power_gain_v_per_kw,
        dc_unserved_power_gain_v_per_kw=params.fig9_dc_unserved_power_gain_v_per_kw,
        dc_load_step_gain_v_per_kw=params.fig9_dc_load_step_gain_v_per_kw,
        dc_diesel_step_gain_v_per_kw=params.fig9_dc_diesel_step_gain_v_per_kw,
        dc_wind_step_gain_v_per_kw=params.fig9_dc_wind_step_gain_v_per_kw,
        dc_transient_decay=params.fig9_dc_transient_decay,
        dc_response=params.fig9_dc_response,
        nominal_frequency_hz=params.fig9_nominal_frequency_hz,
        dc_frequency_gain_hz_per_v=params.fig9_dc_frequency_gain_hz_per_v,
        unserved_frequency_gain_hz_per_kw=params.fig9_unserved_frequency_gain_hz_per_kw,
        frequency_response=params.fig9_frequency_response,
        inverter_efficiency=params.fig9_inverter_efficiency,
    )
    fig11 = _run_trace(
        dt=0.005,
        duration_s=6.0,
        init_time=4.0,
        init_soc_percent=params.fig11_initial_soc_percent,
        effective_capacity_kwh=params.fig11_battery_capacity_kwh,
        wind_power_profile_kw=_scaled_profile(
            _dump_load_wind_power_profile_kw, params.fig11_wind_scale
        ),
        load_profile_kw=_scaled_profile(_dump_load_profile_kw, params.fig11_load_scale),
    )
    return {"fig9": fig9, "fig11": fig11}


def export_overlay(path: Path, targets: list[TargetPoint], params: CalibrationParams) -> None:
    traces = simulate(params)
    panels: list[str] = []
    width = 1200
    panel_h = 190
    margin = 60
    grouped = _targets_by_channel(targets)
    for index, (key, points) in enumerate(sorted(grouped.items())):
        spec = CHANNEL_SPECS.get(key)
        if spec is None:
            continue
        trace = traces[spec.figure]
        target_series = [(point.time, point.value) for point in points]
        sim_series = [
            (float(sample["time"]), float(sample[spec.simulation_key]) * spec.scale)
            for sample in trace
        ]
        y_values = [value for _, value in target_series + sim_series]
        x_values = [time for time, _ in target_series]
        xmin, xmax = min(x_values), max(x_values)
        ymin, ymax = min(y_values), max(y_values)
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        pad = 0.08 * (ymax - ymin)
        y = 42 + index * panel_h
        panels.append(
            _overlay_panel(
                margin,
                y,
                width - 2 * margin,
                panel_h - 42,
                key,
                target_series,
                sim_series,
                xmin,
                xmax,
                ymin - pad,
                ymax + pad,
            )
        )
    height = 58 + len(panels) * panel_h
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="26" text-anchor="middle" font-family="Arial" font-size="18">Dubuisson 2019 calibration overlay</text>',
        *panels,
        "</svg>",
    ]
    _write_plot_document(path, "\n".join(svg))


def write_params(path: Path, params: CalibrationParams) -> None:
    path.write_text(json.dumps(asdict(params), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_residual_report(
    path: Path,
    targets: list[TargetPoint],
    params: CalibrationParams,
    loss: float,
) -> None:
    traces = simulate(params)
    fig9_dg_off = _first_time(
        traces["fig9"], lambda sample: float(sample["diesel_power_kw"]) == 0.0
    )
    fig11_dump_on = _first_time(
        traces["fig11"], lambda sample: float(sample["dump_load_power_kw"]) > 0.0
    )
    lines = [
        "# Dubuisson 2019 Calibration Report",
        "",
        f"Total normalized MSE: `{loss:.6f}`",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
    ]
    for key, value in asdict(params).items():
        lines.append(f"| `{key}` | {value:.6g} |")
    lines.extend(
        [
            "",
            "## Event Anchors",
            "",
            "| Anchor | Simulation |",
            "| --- | ---: |",
            f"| Fig. 9 DG off | {fig9_dg_off:.3f} s |",
            f"| Fig. 11 dump load on | {fig11_dump_on:.3f} s |",
            "",
            "## Target Channels",
            "",
            "| Figure | Channel | Points |",
            "| --- | --- | ---: |",
        ]
    )
    for (figure, channel), points in sorted(_targets_by_channel(targets).items()):
        lines.append(f"| {figure} | `{channel}` | {len(points)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _template_targets() -> dict[tuple[str, str], list[TargetPoint]]:
    return {
        ("fig9", "battery_current_a"): _template_channel(
            "fig9", "battery_current_a", _fig9_battery_current
        ),
        ("fig9", "wind_current_a"): _template_channel("fig9", "wind_current_a", _fig9_wind_current),
        ("fig9", "soc_percent"): _template_channel("fig9", "soc_percent", _fig9_soc),
        ("fig9", "dc_bus_voltage_v"): _template_channel("fig9", "dc_bus_voltage_v", _fig9_vdc),
        ("fig9", "frequency_hz"): _template_channel("fig9", "frequency_hz", _fig9_frequency),
        ("fig11", "battery_current_a"): _template_channel(
            "fig11", "battery_current_a", _fig11_battery_current
        ),
        ("fig11", "wind_current_a"): _template_channel(
            "fig11", "wind_current_a", _fig11_wind_current
        ),
        ("fig11", "dump_load_current_a"): _template_channel(
            "fig11", "dump_load_current_a", _fig11_dump_current
        ),
        ("fig11", "soc_percent"): _template_channel("fig11", "soc_percent", _fig11_soc),
        ("fig11", "dc_bus_voltage_v"): _template_channel("fig11", "dc_bus_voltage_v", _fig11_vdc),
    }


def _template_channel(
    figure: str,
    channel: str,
    func: Callable[[float], float],
) -> list[TargetPoint]:
    start, stop, step = (2.0, 20.0, 0.2) if figure == "fig9" else (4.0, 10.0, 0.05)
    count = int(round((stop - start) / step))
    return [
        TargetPoint(figure, channel, start + index * step, func(start + index * step))
        for index in range(count + 1)
    ]


def _write_target_csv(path: Path, points: list[TargetPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("figure", "channel", "time", "value", "weight"))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def _targets_by_channel(targets: list[TargetPoint]) -> dict[tuple[str, str], list[TargetPoint]]:
    grouped: dict[tuple[str, str], list[TargetPoint]] = {}
    for point in targets:
        grouped.setdefault((point.figure, point.channel), []).append(point)
    for points in grouped.values():
        points.sort(key=lambda point: point.time)
    return grouped


def _read_existing_params(path: Path) -> CalibrationParams:
    if not path.exists():
        return CalibrationParams()
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    return CalibrationParams(**{key: float(value) for key, value in raw.items()})


def _with_param(params: CalibrationParams, name: str, value: float) -> CalibrationParams:
    return replace(params, **{name: value})


def _clamped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _target_range(points: list[TargetPoint]) -> float:
    values = [point.value for point in points]
    return max(max(values) - min(values), 1.0)


def _interp(series: list[tuple[float, float]], time: float) -> float:
    if time <= series[0][0]:
        return series[0][1]
    for left, right in zip(series, series[1:]):
        left_time, left_value = left
        right_time, right_value = right
        if time <= right_time:
            alpha = (time - left_time) / (right_time - left_time)
            return left_value + alpha * (right_value - left_value)
    return series[-1][1]


def _overlay_panel(
    x: int,
    y: int,
    w: int,
    h: int,
    key: tuple[str, str],
    target: list[tuple[float, float]],
    simulation: list[tuple[float, float]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> str:
    title = f"{key[0]} / {key[1]}"
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#222" stroke-width="0.8"/>',
            f'<text x="{x + w / 2}" y="{y - 7}" text-anchor="middle" font-family="Arial" font-size="12">{title}</text>',
            _polyline(target, x, y, w, h, xmin, xmax, ymin, ymax, "#dc2626", 2.0),
            _polyline(simulation, x, y, w, h, xmin, xmax, ymin, ymax, "#2563eb", 1.3),
            f'<text x="{x + 8}" y="{y + 16}" font-family="Arial" font-size="11" fill="#dc2626">target</text>',
            f'<text x="{x + 70}" y="{y + 16}" font-family="Arial" font-size="11" fill="#2563eb">regelum</text>',
        ]
    )


def _polyline(
    points: list[tuple[float, float]],
    x: int,
    y: int,
    w: int,
    h: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    color: str,
    stroke_width: float,
) -> str:
    xspan = xmax - xmin or 1.0
    yspan = ymax - ymin or 1.0
    coords = " ".join(
        f"{x + (_clamped(px, xmin, xmax) - xmin) / xspan * w:.1f},{y + h - (_clamped(py, ymin, ymax) - ymin) / yspan * h:.1f}"
        for px, py in points
    )
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{stroke_width}"/>'
    )


def _figure_from_filename(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def _channel_from_filename(path: Path) -> str:
    return path.stem.split("_", 1)[1]


def _smooth_step(time_s: float, start_s: float, end_s: float) -> float:
    if time_s <= start_s:
        return 0.0
    if time_s >= end_s:
        return 1.0
    x = (time_s - start_s) / (end_s - start_s)
    return x * x * (3.0 - 2.0 * x)


def _pulse(time_s: float, center_s: float, width_s: float) -> float:
    distance = abs(time_s - center_s)
    if distance > width_s:
        return 0.0
    return 1.0 - distance / width_s


def _decay_pulse(time_s: float, center_s: float, decay_s: float) -> float:
    if time_s < center_s:
        return 0.0
    return max(0.0, 1.0 - (time_s - center_s) / decay_s)


def _piecewise(time_s: float, points: tuple[tuple[float, float], ...]) -> float:
    if time_s <= points[0][0]:
        return points[0][1]
    for left, right in zip(points, points[1:]):
        if time_s <= right[0]:
            return left[1] + (right[1] - left[1]) * (time_s - left[0]) / (right[0] - left[0])
    return points[-1][1]


def _fig9_battery_current(time_s: float) -> float:
    if time_s < 7.0:
        return -35.0
    if time_s < 7.45:
        return -35.0 - 175.0 * (time_s - 7.0) / 0.45
    if time_s < 8.2:
        return -210.0 + 75.0 * (time_s - 7.45) / 0.75
    if time_s < 10.7:
        return -135.0 + 12.0 * _pulse(time_s, 9.8, 0.55)
    if time_s < 11.1:
        return 55.0
    if time_s < 16.0:
        return 55.0 + 55.0 * _pulse(time_s, 15.5, 1.8)
    if time_s < 18.0:
        return -45.0 + 10.0 * _pulse(time_s, 17.0, 0.65)
    return -35.0


def _fig9_wind_current(time_s: float) -> float:
    if time_s < 7.0:
        return 0.0
    if time_s < 8.0:
        return 92.0 * (time_s - 7.0)
    if time_s < 11.0:
        return 84.0 + 8.0 * _pulse(time_s, 10.0, 0.35)
    if time_s < 15.0:
        return 84.0 - 14.5 * (time_s - 11.0)
    if time_s < 16.0:
        return 26.0
    if time_s < 18.0:
        return 26.0 + 32.0 * (time_s - 16.0)
    return 90.0


def _fig9_soc(time_s: float) -> float:
    return _piecewise(
        time_s,
        ((2.0, 69.92), (7.0, 69.94), (10.7, 70.0), (12.0, 70.0), (18.0, 69.94), (20.0, 69.945)),
    )


def _fig9_vdc(time_s: float) -> float:
    return (
        350.0
        + 5.0 * _decay_pulse(time_s, 7.0, 0.5)
        - 7.0 * _decay_pulse(time_s, 9.0, 0.45)
        - 17.0 * _decay_pulse(time_s, 10.7, 0.35)
        + 10.0 * _decay_pulse(time_s, 18.0, 0.35)
    )


def _fig9_frequency(time_s: float) -> float:
    return (
        60.0
        + 0.05 * _decay_pulse(time_s, 7.0, 0.2)
        - 0.45 * _decay_pulse(time_s, 10.7, 0.15)
        + 0.06 * _decay_pulse(time_s, 18.0, 0.18)
    )


def _fig11_battery_current(time_s: float) -> float:
    return -82.0 + 172.0 * _smooth_step(time_s, 6.65, 7.2)


def _fig11_wind_current(time_s: float) -> float:
    return 92.0 - 10.0 * _smooth_step(time_s, 6.65, 7.3)


def _fig11_dump_current(time_s: float) -> float:
    return 0.0 if time_s < 6.65 else 140.0


def _fig11_soc(time_s: float) -> float:
    return min(100.0, 99.98 + 0.02 * _smooth_step(time_s, 4.0, 6.65))


def _fig11_vdc(time_s: float) -> float:
    return 350.0 - 18.0 * _decay_pulse(time_s, 6.65, 0.55)


if __name__ == "__main__":
    main()
