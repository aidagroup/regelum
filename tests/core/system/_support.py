from typing import Any

import regelum as rg


def _tick_system(
    nodes: list[rg.Node],
    **kwargs: Any,
) -> rg.PhasedReactiveSystem:
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "tick",
                nodes=tuple(nodes),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
        **kwargs,
    )
