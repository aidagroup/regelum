import pytest
import numpy as np
from regelum.environment.node.base import State, Node, Inputs
from regelum.environment.transistor import CasADiTransistor
from regelum.utils import rg


class TestNode(Node):
    state = State(
        "test_state",
        None,
        [
            State("substate1", (1,), np.array([1.0])),
            State("substate2", (1,), np.array([2.0])),
        ],
    )
    inputs = Inputs([])

    def compute_state_dynamics(self):
        return {
            "test_state": rg.vstack(
                (
                    -rg.exp(self.state["test_state/substate1"].data),
                    -rg.exp(self.state["test_state/substate2"].data),
                )
            )
        }


@pytest.fixture
def test_node():
    return TestNode(is_root=True, step_size=0.01)


def test_casadi_transistor_with_dynamic_variable_paths(test_node):
    transistor = CasADiTransistor(
        node=test_node,
        dynamic_variable_paths=["test_state/substate1", "test_state/substate2"],
    )

    for _ in range(10):
        transistor.step()


def test_casadi_transistor_without_dynamic_variable_paths_raises_error(test_node):
    with pytest.raises(
        ValueError,
        match="Tree-like state requires dynamic_variable_paths",
    ):
        CasADiTransistor(node=test_node)
