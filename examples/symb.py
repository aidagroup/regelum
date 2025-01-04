from regelum.environment.node.base_new import Node, Graph, Reset
import numpy as np


class Node1(Node):
    def __init__(self, step_size: float):
        super().__init__(step_size=step_size)
        self.x = self.define_variable("x", np.zeros(1), None, (1,))

    def step(self):
        self.x.value = self.x.value + 1


class ResetNode(Reset):
    def __init__(self, steps_to_reset: int, name: str):
        super().__init__(name=name, inputs=["clock_1.time", "step_counter_1.counter"])
        self.reset_flag = self.define_variable("reset_flag", value=False)
        self.steps_to_reset = steps_to_reset

    def step(self):
        self.reset_flag.value = (
            self.resolved_inputs.find("step_counter_1.counter").value
            % self.steps_to_reset
            == 0
        )


reset_node = ResetNode(steps_to_reset=5, name="reset_node1_1")

node1 = Node1(step_size=0.3)
node2 = Node1(step_size=0.4)

graph = Graph(
    [node1, node2, reset_node],
    initialize_inner_time=True,
    debug=False,
    states_to_log=["clock_1.time", "node1_1.x", "node1_2.x"],
)
print(node1.step_size)
print(node2.step_size)

graph.resolve(graph.variables)
graph.detect_subgraphs()
for _ in range(10):
    graph.step()
