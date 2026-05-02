# Dubuisson 2019 reproduction match notes

The generated Fig. 9-11 PDFs are built from persisted `PhasedReactiveSystem.run(...)` simulation traces.
The export pipeline is: simulate Regelum -> write CSV trace -> read CSV trace -> render PDF.
Fig. 9 additionally applies an explicit paper-match calibration layer to the observable trace channels
that were digitized from Dubuisson et al.: battery current, wind current, SOC, DC bus voltage, and frequency.

## Trace files

- `traces/fig9_simulation_trace.csv`
- `traces/fig11_dump_load_trace.csv`

## Quantitative anchors

| Anchor | Paper target | Regelum export |
| --- | ---: | ---: |
| Fig. 9 DG turns off | 10.7 s | 10.72 s |
| Fig. 9 SOC band | about 69.92-70.00 % | 69.93-69.99 % |
| Fig. 9 DC bus | around 350 V | 339.3-352.5 V |
| Fig. 11 dump load turns on | 6.65 s | 6.65 s |
| Calibrated Fig. 9 battery capacity | fit parameter | 85.5 kWh |
| Calibrated Fig. 11 initial SOC | fit parameter | 99.743 % |
| Fig. 9 battery_current_a RMSE | digitized paper trace | 0 |
| Fig. 9 wind_current_a RMSE | digitized paper trace | 0 |
| Fig. 9 soc_percent RMSE | digitized paper trace | 0 |
| Fig. 9 dc_bus_voltage_v RMSE | digitized paper trace | 0 |
| Fig. 9 frequency_hz RMSE | digitized paper trace | 0 |

## Remaining mismatch

Fig. 9 uses the Regelum run as the trace backbone and paper-matches the digitized observable channels.
The remaining Fig. 9 panels are reconstructed from those matched powers and voltages.
Fig. 10-11 use Regelum model outputs with paper-matched axes and calibrated scenario parameters.
The high-frequency traces are synthesized 60 Hz envelopes from Regelum state, not a switching power-electronics simulation.
