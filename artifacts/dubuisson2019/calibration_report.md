# Dubuisson 2019 Calibration Report

Total normalized MSE: `0.034343`

## Parameters

| Parameter | Value |
| --- | ---: |
| `fig9_battery_capacity_kwh` | 85.5 |
| `fig9_initial_soc_percent` | 69.92 |
| `fig9_wind_scale` | 0.8 |
| `fig9_load_scale` | 1.2 |
| `fig11_initial_soc_percent` | 99.7425 |
| `fig11_battery_capacity_kwh` | 10 |
| `fig11_wind_scale` | 1 |
| `fig11_load_scale` | 1 |

## Event Anchors

| Anchor | Simulation |
| --- | ---: |
| Fig. 9 DG off | 10.720 s |
| Fig. 11 dump load on | 6.650 s |

## Target Channels

| Figure | Channel | Points |
| --- | --- | ---: |
| fig11 | `battery_current_a` | 121 |
| fig11 | `dc_bus_voltage_v` | 70 |
| fig11 | `dump_load_current_a` | 121 |
| fig11 | `soc_percent` | 120 |
| fig11 | `wind_current_a` | 121 |
| fig9 | `battery_current_a` | 181 |
| fig9 | `dc_bus_voltage_v` | 181 |
| fig9 | `frequency_hz` | 181 |
| fig9 | `soc_percent` | 181 |
| fig9 | `wind_current_a` | 181 |
