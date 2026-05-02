# Dubuisson 2019 Calibration Targets

This directory contains digitized target traces for calibrating the Regelum reproduction
against Dubuisson et al. Fig. 9 and Fig. 11. The CSV files are extracted from the
rendered PDF page using `uv run regelum-dubuisson2019-digitize`.
`fig9_load_power_kw.csv` is a reconstructed external scenario input derived from
digitized Fig. 9 wind and battery currents plus the diesel-generator on/off event.
It is used as an input profile, not as a simulated output target.

Each CSV uses this schema:

```csv
figure,channel,time,value,weight
fig9,battery_current_a,2.0,-35.0,1.0
```

To rerun the PDF raster extraction:

```bash
uv run regelum-dubuisson2019-digitize --render
```

To recalibrate the Regelum simulation against the current targets:

```bash
uv run regelum-dubuisson2019-calibrate
```

Some channels contain fewer points because the PDF raster has occluded or low-contrast
segments where the blue trace is not separable from grid/text pixels.

The calibrator writes:

- `artifacts/dubuisson2019/calibration_params.json`
- `artifacts/dubuisson2019/calibration_report.md`
- `artifacts/dubuisson2019/calibration_overlay.pdf`
- regenerated Fig. 9-11 PDFs and simulation CSV traces
