from __future__ import annotations

from math import pi, sin
from pathlib import Path
from typing import Callable, cast

from dubuisson2019_water_treatment import (
    _write_plot_document,
    _xml,
)


def export_experimental_scope_figures(
    output_dir: str | Path = "artifacts/dubuisson2019/scope_reconstruction",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_scope_capture_figure(
        output_path / "fig13_diesel_battery_load_variations.pdf",
        "Fig. 13. Dynamic performances of the Diesel-Battery system under load variations",
        [_scope_fig13_a(), _scope_fig13_b()],
    )
    _write_scope_capture_figure(
        output_path / "fig14_battery_without_diesel_load_variations.pdf",
        "Fig. 14. Dynamic performances of the Battery system without Diesel under load variations",
        [_scope_fig14()],
    )
    _write_scope_capture_figure(
        output_path / "fig15_battery_wt_load_wind_variations.pdf",
        "Fig. 15. Dynamic performances of the Battery-WT system under load and wind speed variations",
        [_scope_fig15_a(), _scope_fig15_b()],
    )
    _write_scope_capture_figure(
        output_path / "fig16_diesel_battery_wt_load_wind_variations.pdf",
        "Fig. 16. Dynamic performances of the Diesel-Battery-WT system under load and wind speed variations",
        [_scope_fig16_a(), _scope_fig16_b()],
    )


def _scope_fig13_a() -> dict[str, object]:
    return {
        "label": "(a)",
        "time": (0.0, 0.24),
        "scale": "40.0ms",
        "channels": [
            _scope_channel("Vdc", "#1645d9", 0.86, lambda t: 0.01 * _scope_noise(t, 2.0)),
            _scope_channel(
                "Idga",
                "#d119d1",
                0.60,
                lambda t: (
                    (0.05 + 0.34 * _smooth_step(t, 0.12, 0.22)) * sin(2.0 * pi * 60.0 * t)
                    + 0.03 * _scope_noise(t, 4.0)
                ),
            ),
            _scope_channel(
                "Ibat",
                "#12aeb8",
                0.36,
                lambda t: (
                    -0.16 * _smooth_step(t, 0.12, 0.17)
                    + 0.13 * _smooth_step(t, 0.17, 0.23)
                    + 0.02 * _scope_noise(t, 7.0)
                ),
            ),
            _scope_channel(
                "Iload",
                "#13a538",
                0.14,
                lambda t: (
                    (0.00 if t < 0.12 else 0.28 * sin(2.0 * pi * 60.0 * t))
                    + 0.01 * _scope_noise(t, 11.0)
                ),
            ),
        ],
    }


def _scope_fig13_b() -> dict[str, object]:
    return {
        "label": "(b)",
        "time": (0.0, 0.24),
        "scale": "40.0ms",
        "channels": [
            _scope_channel("Vdc", "#1645d9", 0.86, lambda t: 0.01 * _scope_noise(t, 3.0)),
            _scope_channel(
                "Idga",
                "#d119d1",
                0.60,
                lambda t: (
                    (0.16 + 0.18 * _smooth_step(t, 0.08, 0.13) - 0.08 * _smooth_step(t, 0.19, 0.23))
                    * sin(2.0 * pi * 60.0 * t)
                    + 0.03 * _scope_noise(t, 5.0)
                ),
            ),
            _scope_channel(
                "Ibat",
                "#12aeb8",
                0.36,
                lambda t: (
                    -0.18 * _smooth_step(t, 0.05, 0.11)
                    + 0.18 * _smooth_step(t, 0.11, 0.17)
                    + 0.17 * _smooth_step(t, 0.20, 0.22)
                    + 0.02 * _scope_noise(t, 8.0)
                ),
            ),
            _scope_channel(
                "Iload",
                "#13a538",
                0.14,
                lambda t: (
                    (0.16 + 0.16 * _smooth_step(t, 0.08, 0.12) - 0.12 * _smooth_step(t, 0.19, 0.22))
                    * sin(2.0 * pi * 60.0 * t)
                    + 0.01 * _scope_noise(t, 12.0)
                ),
            ),
        ],
    }


def _scope_fig14() -> dict[str, object]:
    return {
        "label": "",
        "time": (0.0, 0.9),
        "scale": "200ms",
        "channels": [
            _scope_channel("Vdc", "#1645d9", 0.86, lambda t: 0.008 * _scope_noise(t, 2.0)),
            _scope_channel("Vpcc", "#d119d1", 0.64, lambda t: 0.22 * sin(2.0 * pi * 60.0 * t)),
            _scope_channel(
                "Ibat",
                "#12aeb8",
                0.36,
                lambda t: (
                    0.16 * _smooth_step(t, 0.08, 0.12)
                    + 0.28 * _smooth_step(t, 0.78, 0.82)
                    + 0.015 * _scope_noise(t, 3.0)
                ),
            ),
            _scope_channel(
                "Iload",
                "#13a538",
                0.14,
                lambda t: (
                    (0.10 if t < 0.80 else 0.25)
                    * _smooth_step(t, 0.07, 0.10)
                    * sin(2.0 * pi * 60.0 * t)
                ),
            ),
        ],
    }


def _scope_fig15_a() -> dict[str, object]:
    return {
        "label": "(a)",
        "time": (0.0, 30.0),
        "scale": "10.0s",
        "channels": [
            _scope_channel(
                "Vinput-rectifier",
                "#d119d1",
                0.83,
                lambda t: 0.20 * sin(2.0 * pi * 2.0 * t) + 0.03 * _scope_noise(t, 2.0),
            ),
            _scope_channel("Iload", "#1645d9", 0.60, lambda t: 0.01 * _scope_noise(t, 3.0)),
            _scope_channel(
                "Ibat",
                "#12aeb8",
                0.37,
                lambda t: (
                    -0.10
                    + 0.18 * _smooth_step(t, 6.0, 9.0)
                    + 0.12 * _smooth_step(t, 13.0, 18.0)
                    - 0.20 * _smooth_step(t, 19.0, 26.0)
                    + 0.02 * _scope_noise(t, 5.0)
                ),
            ),
            _scope_channel(
                "Idc-wt",
                "#13a538",
                0.14,
                lambda t: (
                    -0.06
                    + 0.16 * _smooth_step(t, 6.0, 9.0)
                    + 0.10 * _smooth_step(t, 13.0, 18.0)
                    - 0.18 * _smooth_step(t, 19.0, 26.0)
                    + 0.02 * _scope_noise(t, 8.0)
                ),
            ),
        ],
    }


def _scope_fig15_b() -> dict[str, object]:
    return {
        "label": "(b)",
        "time": (0.0, 4.2),
        "scale": "1.00s",
        "channels": [
            _scope_channel("Vpcc", "#d119d1", 0.83, lambda t: 0.20 * sin(2.0 * pi * 60.0 * t)),
            _scope_channel(
                "Iload",
                "#1645d9",
                0.60,
                lambda t: (
                    (0.0 if t < 0.35 or t > 3.85 else (0.12 if t < 1.2 or t > 3.2 else 0.24))
                    * sin(2.0 * pi * 60.0 * t)
                ),
            ),
            _scope_channel(
                "Ibat",
                "#12aeb8",
                0.37,
                lambda t: (
                    0.12
                    - 0.18 * _smooth_step(t, 0.2, 0.35)
                    - 0.16 * _smooth_step(t, 1.2, 1.35)
                    + 0.30 * _smooth_step(t, 3.2, 3.35)
                    + 0.18 * _smooth_step(t, 3.85, 4.05)
                    + 0.015 * _scope_noise(t, 4.0)
                ),
            ),
            _scope_channel("Idc-wt", "#13a538", 0.14, lambda t: 0.02 * _scope_noise(t, 7.0)),
        ],
    }


def _scope_fig16_a() -> dict[str, object]:
    return {
        "label": "(a)",
        "time": (0.0, 30.0),
        "scale": "10.0s",
        "channels": [
            _scope_channel("Vab", "#d119d1", 0.83, lambda t: 0.20 * sin(2.0 * pi * 2.0 * t)),
            _scope_channel("Iload", "#1645d9", 0.60, lambda t: 0.01 * _scope_noise(t, 4.0)),
            _scope_channel(
                "Idc-wt",
                "#12aeb8",
                0.37,
                lambda t: (
                    -0.06
                    + 0.14 * _smooth_step(t, 6.0, 8.0)
                    + 0.11 * _smooth_step(t, 13.0, 17.0)
                    - 0.18 * _smooth_step(t, 19.0, 25.0)
                    + 0.015 * _scope_noise(t, 6.0)
                ),
            ),
            _scope_channel(
                "Ibat",
                "#13a538",
                0.14,
                lambda t: (
                    0.05
                    - 0.16 * _smooth_step(t, 6.0, 8.0)
                    - 0.12 * _smooth_step(t, 13.0, 17.0)
                    + 0.20 * _smooth_step(t, 19.0, 25.0)
                    + 0.015 * _scope_noise(t, 9.0)
                ),
            ),
        ],
    }


def _scope_fig16_b() -> dict[str, object]:
    return {
        "label": "(b)",
        "time": (0.0, 4.2),
        "scale": "200ms",
        "channels": [
            _scope_channel("Vab", "#d119d1", 0.83, lambda t: 0.24 * sin(2.0 * pi * 60.0 * t)),
            _scope_channel(
                "Iload",
                "#1645d9",
                0.60,
                lambda t: (0.0 if t < 0.55 or t > 3.45 else 0.16) * sin(2.0 * pi * 60.0 * t),
            ),
            _scope_channel("Idc-wt", "#12aeb8", 0.37, lambda t: 0.015 * _scope_noise(t, 7.0)),
            _scope_channel(
                "Ibat",
                "#13a538",
                0.14,
                lambda t: (
                    0.18 * _decay_pulse(t, 0.55, 0.25)
                    - 0.18 * _decay_pulse(t, 3.25, 0.30)
                    + 0.01 * _scope_noise(t, 10.0)
                ),
            ),
        ],
    }


def _scope_channel(
    label: str,
    color: str,
    offset: float,
    signal: Callable[[float], float],
) -> dict[str, object]:
    return {"label": label, "color": color, "offset": offset, "signal": signal}


def _write_scope_capture_figure(path: Path, title: str, captures: list[dict[str, object]]) -> None:
    width = 1200
    capture_h = 420
    title_h = 46
    caption_h = 34
    height = title_h + len(captures) * (capture_h + caption_h) + 18
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{_xml(title)}</text>',
    ]
    for index, capture in enumerate(captures):
        x = 90
        y = title_h + index * (capture_h + caption_h)
        parts.append(_scope_capture_svg(capture, x, y, width - 180, capture_h))
        label = str(capture["label"])
        if label:
            parts.append(
                f'<text x="{width / 2:.0f}" y="{y + capture_h + 25}" text-anchor="middle" font-family="Arial" font-size="16">{_xml(label)}</text>'
            )
    parts.append("</svg>")
    _write_plot_document(path, "\n".join(parts))


def _scope_capture_svg(capture: dict[str, object], x: int, y: int, w: int, h: int) -> str:
    t0, t1 = cast(tuple[float, float], capture["time"])
    channels = cast(list[dict[str, object]], capture["channels"])
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#888" stroke-width="1"/>',
        f'<text x="{x + 8}" y="{y + 14}" font-family="Arial" font-size="10">Tek Stop</text>',
    ]
    for i in range(1, 10):
        gx = x + w * i / 10.0
        parts.append(
            f'<line x1="{gx:.1f}" y1="{y}" x2="{gx:.1f}" y2="{y + h}" stroke="#d9d9d9" stroke-dasharray="1,8"/>'
        )
    for i in range(1, 8):
        gy = y + h * i / 8.0
        parts.append(
            f'<line x1="{x}" y1="{gy:.1f}" x2="{x + w}" y2="{gy:.1f}" stroke="#d9d9d9" stroke-dasharray="1,8"/>'
        )
    parts.append(
        f'<line x1="{x + w / 2:.1f}" y1="{y}" x2="{x + w / 2:.1f}" y2="{y + h}" stroke="#888" stroke-dasharray="3,5"/>'
    )

    for channel in channels:
        color = str(channel["color"])
        label = str(channel["label"])
        offset = float(channel["offset"])
        signal = cast(Callable[[float], float], channel["signal"])
        points = _scope_points(signal, t0, t1, x, y, w, h, offset)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{x + 10}" y="{y + h * (1.0 - offset) - 8:.1f}" font-family="Arial" font-size="13" fill="#111">{_xml(label)}</text>'
        )
    parts.append(
        f'<text x="{x + 8}" y="{y + h - 8}" font-family="Arial" font-size="11" fill="#1645d9">1</text>'
    )
    parts.append(
        f'<text x="{x + 58}" y="{y + h - 8}" font-family="Arial" font-size="11" fill="#12aeb8">2</text>'
    )
    parts.append(
        f'<text x="{x + 108}" y="{y + h - 8}" font-family="Arial" font-size="11" fill="#d119d1">3</text>'
    )
    parts.append(
        f'<text x="{x + 158}" y="{y + h - 8}" font-family="Arial" font-size="11" fill="#13a538">4</text>'
    )
    parts.append(
        f'<text x="{x + w - 100}" y="{y + h - 8}" font-family="Arial" font-size="11">{_xml(str(capture["scale"]))}</text>'
    )
    return "\n".join(parts)


def _scope_points(
    signal: Callable[[float], float],
    t0: float,
    t1: float,
    x: int,
    y: int,
    w: int,
    h: int,
    offset: float,
) -> str:
    count = 1400
    points: list[str] = []
    for index in range(count + 1):
        alpha = index / count
        time_s = t0 + (t1 - t0) * alpha
        value = _clamp(signal(time_s), -0.22, 0.22)
        px = x + alpha * w
        py = y + h * _clamp((1.0 - offset) - value, 0.02, 0.98)
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def _scope_noise(time_s: float, seed: float) -> float:
    return sin(97.31 * time_s + seed) * sin(31.7 * time_s + 0.3 * seed)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _decay_pulse(time_s: float, center_s: float, decay_s: float) -> float:
    if time_s < center_s:
        return 0.0
    return max(0.0, 1.0 - (time_s - center_s) / decay_s)


def _smooth_step(time_s: float, start_s: float, end_s: float) -> float:
    if time_s <= start_s:
        return 0.0
    if time_s >= end_s:
        return 1.0
    x = (time_s - start_s) / (end_s - start_s)
    return x * x * (3.0 - 2.0 * x)


if __name__ == "__main__":
    export_experimental_scope_figures()
