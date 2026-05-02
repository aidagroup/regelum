# Dubuisson 2019 reproduction match notes

The generated Fig. 9-11 PDFs are built from persisted `PhasedReactiveSystem.run(...)` simulation traces.
The export pipeline is: simulate Regelum -> write CSV trace -> read CSV trace -> render PDF.
Digitized paper traces are used as external Fig. 9 scenario inputs for wind and load.
Digitized paper output traces are not used to overwrite simulated output channels.

## Trace files

- `traces/fig9_simulation_trace.csv`
- `traces/fig11_dump_load_trace.csv`

## Quantitative anchors

| Anchor | Paper target | Regelum export |
| --- | ---: | ---: |
| Fig. 9 DG turns off | 10.7 s | 10.70 s |
| Fig. 9 SOC band | about 69.92-70.00 % | 69.92-70.00 % |
| Fig. 9 DC bus | around 350 V | 346.8-356.0 V |
| Fig. 11 dump load turns on | 6.65 s | 6.65 s |
| Calibrated Fig. 9 battery capacity | fit parameter | 80.0 kWh |
| Calibrated Fig. 11 initial SOC | fit parameter | 99.743 % |
| Fig. 9 battery_current_a RMSE | digitized paper output | 35.59 |
| Fig. 9 wind_current_a RMSE | digitized paper output | 0.006656 |
| Fig. 9 soc_percent RMSE | digitized paper output | 0.0157 |
| Fig. 9 dc_bus_voltage_v RMSE | digitized paper output | 3.503 |
| Fig. 9 frequency_hz RMSE | digitized paper output | 0.04268 |

## Remaining mismatch

Fig. 9 uses CSV-backed external wind and load inputs; battery current, SOC, DC bus voltage,
and frequency are still Regelum outputs.
Fig. 10-11 use Regelum model outputs with paper-comparable axes and calibrated scenario parameters.
The high-frequency traces are synthesized 60 Hz envelopes from Regelum state, not a switching power-electronics simulation.
