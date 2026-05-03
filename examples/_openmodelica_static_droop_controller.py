from __future__ import annotations

from functools import partial

import numpy as np
from openmodelica_microgrid_gym.agents import StaticControlAgent
from openmodelica_microgrid_gym.aux_ctl import (
    DroopParams,
    InverseDroopParams,
    MultiPhaseDQ0PIPIController,
    MultiPhaseDQCurrentController,
    PI_params,
    PLLParams,
)


def load_step(t, gain):
    return 1 * gain if t < 0.2 else 2 * gain


def static_droop_model_params():
    return {
        "rl1.resistor1.R": partial(load_step, gain=20),
        "rl1.resistor2.R": partial(load_step, gain=20),
        "rl1.resistor3.R": partial(load_step, gain=20),
        "rl1.inductor1.L": 0.001,
        "rl1.inductor2.L": 0.001,
        "rl1.inductor3.L": 0.001,
    }


def build_static_droop_agent(net):
    droop_gain = 40000.0
    qdroop_gain = 1000.0
    delta_t = net.ts
    freq_nom = net.freq_nom
    v_nom = net.v_nom
    i_lim = net["inverter1"].i_lim

    master_voltage_pi = PI_params(kP=0.025, kI=60, limits=(-i_lim, i_lim))
    master_current_pi = PI_params(kP=0.012, kI=90, limits=(-1, 1))
    master_droop = DroopParams(droop_gain, 0.005, freq_nom)
    master_qdroop = DroopParams(qdroop_gain, 0.002, v_nom)

    slave_current_pi = PI_params(kP=0.005, kI=200, limits=(-1, 1))
    slave_pll = PLLParams(kP=10, kI=200, limits=None, f_nom=freq_nom)
    slave_droop = InverseDroopParams(droop_gain, delta_t, freq_nom, tau_filt=0.04)
    slave_qdroop = InverseDroopParams(50, delta_t, v_nom, tau_filt=0.01)

    controllers = [
        MultiPhaseDQ0PIPIController(
            master_voltage_pi,
            master_current_pi,
            master_droop,
            master_qdroop,
            ts_sim=delta_t,
            name="master",
        ),
        MultiPhaseDQCurrentController(
            slave_current_pi,
            slave_pll,
            i_lim,
            slave_droop,
            slave_qdroop,
            ts_sim=delta_t,
            name="slave",
        ),
    ]

    return StaticControlAgent(
        controllers,
        {
            "master": [
                [f"lc1.inductor{k}.i" for k in "123"],
                [f"lc1.capacitor{k}.v" for k in "123"],
            ],
            "slave": [
                [f"lcl1.inductor{k}.i" for k in "123"],
                [f"lcl1.capacitor{k}.v" for k in "123"],
                np.zeros(3),
            ],
        },
    )
