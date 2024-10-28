import pytest
import numpy as np
from regelum.environment.node import State, Node, Inputs
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
                    self.state["test_state/substate1"].data,
                    self.state["test_state/substate2"].data,
                )
            )
        }


@pytest.fixture
def test_node():
    return TestNode(is_root=True)


def test_casadi_transistor_with_dynamic_variable_paths(test_node):
    transistor = CasADiTransistor(
        node=test_node,
        step_size=0.01,
        dynamic_variable_paths=["test_state/substate1", "test_state/substate2"],
    )

    # Perform a step to test the transition
    transistor.step()

    # Check if the state has been updated correctly
    assert np.allclose(test_node.state["test_state/substate1"].data, 1.01005017)
    assert np.allclose(test_node.state["test_state/substate2"].data, 2.02010033)


def test_casadi_transistor_without_dynamic_variable_paths_raises_error(test_node):
    with pytest.raises(
        ValueError,
        match="Tree-like state requires dynamic_variable_paths to be specified",
    ):
        CasADiTransistor(node=test_node, step_size=0.01)
