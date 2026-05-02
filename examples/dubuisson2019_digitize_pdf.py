from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PAGE_IMAGE = Path("artifacts/dubuisson2019/pdf_pages/page-06.png")
DEFAULT_TARGET_DIR = Path("references/dubuisson2019_targets")


@dataclass(frozen=True)
class Panel:
    figure: str
    channel: str
    crop: tuple[int, int, int, int]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    color: str = "blue"
    samples: int = 181


PANELS = (
    Panel("fig9", "battery_current_a", (559, 909, 296, 96), (2.0, 20.0), (-200.0, 100.0)),
    Panel("fig9", "wind_current_a", (559, 1048, 296, 96), (2.0, 20.0), (0.0, 100.0)),
    Panel("fig9", "dc_bus_voltage_v", (559, 1192, 296, 95), (2.0, 20.0), (330.0, 360.0)),
    Panel("fig9", "soc_percent", (559, 1333, 296, 95), (2.0, 20.0), (69.92, 70.0)),
    Panel("fig9", "frequency_hz", (559, 1474, 296, 95), (2.0, 20.0), (59.5, 60.5)),
    Panel(
        "fig9",
        "load_voltage_magnitude_v",
        (205, 1050, 306, 93),
        (2.0, 20.0),
        (340.0, 400.0),
    ),
    Panel(
        "fig11",
        "battery_current_a",
        (1378, 840, 285, 76),
        (4.0, 10.0),
        (-100.0, 100.0),
        samples=121,
    ),
    Panel("fig11", "wind_current_a", (1378, 960, 285, 91), (4.0, 10.0), (0.0, 120.0), samples=121),
    Panel(
        "fig11",
        "dc_bus_voltage_v",
        (1378, 1095, 285, 91),
        (4.0, 10.0),
        (330.0, 360.0),
        samples=121,
    ),
    Panel(
        "fig11",
        "dump_load_current_a",
        (1042, 1231, 320, 91),
        (4.0, 10.0),
        (0.0, 150.0),
        samples=121,
    ),
    Panel("fig11", "soc_percent", (1378, 1231, 285, 91), (4.0, 10.0), (99.97, 100.01), samples=121),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Digitize Dubuisson 2019 Fig. 9/11 from PDF raster."
    )
    parser.add_argument("--page-image", type=Path, default=DEFAULT_PAGE_IMAGE)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--render", action="store_true", help="Render page 6 before digitizing.")
    args = parser.parse_args()

    if args.render or not args.page_image.exists():
        args.page_image.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "6",
                "-l",
                "6",
                "-r",
                "220",
                "-png",
                "references/dubuisson2019.pdf",
                str(args.page_image.with_suffix("")),
            ],
            check=True,
        )
        rendered_candidates = sorted(args.page_image.parent.glob(f"{args.page_image.stem}-*.png"))
        if rendered_candidates:
            rendered_candidates[-1].replace(args.page_image)

    args.target_dir.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        points = digitize_panel(args.page_image, panel)
        write_target_csv(args.target_dir / f"{panel.figure}_{panel.channel}.csv", panel, points)
        print(f"Wrote {panel.figure}_{panel.channel}: {len(points)} points")


def digitize_panel(image_path: Path, panel: Panel) -> list[tuple[float, float]]:
    x0, y0, width, height = panel.crop
    pixels = read_rgb_crop(image_path, x0, y0, width, height)
    by_column: dict[int, list[int]] = {}
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[y * width + x]
            if _matches_color(r, g, b, panel.color):
                by_column.setdefault(x, []).append(y)

    extracted: list[tuple[float, float]] = []
    last_y: float | None = None
    for index in range(panel.samples):
        alpha = index / (panel.samples - 1)
        x = round(alpha * (width - 1))
        y_values = _nearby_y_values(by_column, x)
        if y_values:
            y_pixel = _median(y_values)
            last_y = y_pixel
        elif last_y is not None:
            y_pixel = last_y
        else:
            continue
        time = panel.x_range[0] + alpha * (panel.x_range[1] - panel.x_range[0])
        value = pixel_to_value(y_pixel, height, panel.y_range)
        extracted.append((time, value))
    return extracted


def read_rgb_crop(
    image_path: Path, x: int, y: int, width: int, height: int
) -> list[tuple[int, int, int]]:
    raw = subprocess.check_output(
        [
            "convert",
            str(image_path),
            "-crop",
            f"{width}x{height}+{x}+{y}",
            "-depth",
            "8",
            "rgb:-",
        ]
    )
    return [(raw[index], raw[index + 1], raw[index + 2]) for index in range(0, len(raw), 3)]


def write_target_csv(path: Path, panel: Panel, points: list[tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("figure", "channel", "time", "value", "weight"))
        writer.writeheader()
        for time, value in points:
            writer.writerow(
                {
                    "figure": panel.figure,
                    "channel": panel.channel,
                    "time": f"{time:.6f}",
                    "value": f"{value:.6f}",
                    "weight": "1.0",
                }
            )


def pixel_to_value(y_pixel: float, height: int, y_range: tuple[float, float]) -> float:
    y_min, y_max = y_range
    alpha = 1.0 - y_pixel / max(height - 1, 1)
    return y_min + alpha * (y_max - y_min)


def _nearby_y_values(by_column: dict[int, list[int]], x: int) -> list[int]:
    values: list[int] = []
    for dx in (0, -1, 1, -2, 2, -3, 3):
        values.extend(by_column.get(x + dx, ()))
        if values:
            return values
    return values


def _matches_color(r: int, g: int, b: int, color: str) -> bool:
    if color == "blue":
        return b > 95 and r < 140 and g < 160 and b > r + 30 and b > g + 10
    if color == "orange":
        return r > 180 and 70 < g < 190 and b < 120
    if color == "green":
        return g > 150 and r < 120 and b < 140
    raise ValueError(f"Unsupported color: {color}")


def _median(values: list[int]) -> float:
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[middle])
    return 0.5 * (sorted_values[middle - 1] + sorted_values[middle])


if __name__ == "__main__":
    main()
