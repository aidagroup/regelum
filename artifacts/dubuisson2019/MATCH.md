# Dubuisson 2019 reproduction match notes

The generated Fig. 9-11 PDFs are built from persisted `PhasedReactiveSystem.run(...)` simulation traces.
The export pipeline is: simulate Regelum -> write CSV trace -> read CSV trace -> render PDF.
Paper values are used only as calibration targets for scenario parameters.
Digitized paper traces are not used to overwrite the simulated trace channels.

## Trace files

- `traces/fig9_simulation_trace.csv`
- `traces/fig11_dump_load_trace.csv`

## Quantitative anchors

| Anchor | Paper target | Regelum export |
| --- | ---: | ---: |
| Fig. 9 DG turns off | 10.7 s | 10.70 s |
| Fig. 9 SOC band | about 69.92-70.00 % | 69.92-70.00 % |
| Fig. 9 DC bus | around 350 V | 350.0-356.0 V |
| Fig. 11 dump load turns on | 6.65 s | 6.65 s |
| Calibrated Fig. 9 battery capacity | fit parameter | 85.5 kWh |
| Calibrated Fig. 11 initial SOC | fit parameter | 99.743 % |

## Remaining mismatch

Fig. 9-11 use the Regelum model outputs with paper-comparable axes and calibrated scenario parameters.
The high-frequency traces are synthesized 60 Hz envelopes from Regelum state, not a switching power-electronics simulation.
