# Regelum

![Pylint](https://img.shields.io/badge/pylint-9.40%2F10-brightgreen)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Regelum (German: "to control") is a powerful framework for reinforcement learning, simulation, and control systems. It provides a modular, node-based architecture for building dynamic simulation pipelines with minimal overhead.

## Key Features

- **🔌 Modular Node Architecture**
  - Build complex systems using independent, composable nodes
  - Easy to extend and modify without affecting other components
  - Shared global variable scope for seamless data flow

- **🔄 Synchronous Loop Design**
  - Optimized for reinforcement learning tasks
  - Predictable execution flow
  - Built-in state management

- **📊 Integrated Tools**
  - Real-time logging and visualization
  - MathJax-powered documentation
  - Jupyter notebook integration
  - Automatic code quality checks

- **🚀 Production Ready**
  - Type-checked with MyPy
  - Comprehensive linting (Pylint score: 9.5+/10)
  - Pre-commit hooks for code quality
  - Extensive documentation

## Installation

### For Users
```bash
pip install regelum
```

### For Developers
```bash
pip install -e ".[dev]"
```

## Quick Start

Here's a practical example of controlling a pendulum using LQR with visualization:

```python
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.lqr import LQRController
from regelum.node.visualization.pygame_renderer import PendulumRenderer
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps

# Create pendulum system
pendulum = Pendulum(
    control_signal_name="lqr_1.action",
    initial_state=np.array([np.pi, 0.0])
)

# Configure LQR controller
A = np.array([[0, 1], [-3 * pendulum.gravity_acceleration / (2 * pendulum.length), 0]])
B = np.array([[0], [1 / pendulum.mass]])
Q = np.diag([10.0, 1.0])  # State cost matrix
R = np.array([[0.1]])      # Control cost matrix

lqr = LQRController(
    controlled_state=pendulum.state,
    system_matrices=(A, B),
    cost_matrices=(Q, R),
    control_limits=(-10.0, 10.0),
    step_size=0.01
)

# Add visualization
viz = PendulumRenderer(
    state_variable=pendulum.state,
    fps=60.0,
    window_size=(1200, 400),
    visible_history=1000
)

# Reset pendulum periodically
reset_node = ResetEachNSteps(
    node_name_to_reset=pendulum.external_name,
    n_steps=1000
)

# Create and run computation graph
graph = Graph(
    [pendulum, lqr, viz, reset_node],
    initialize_inner_time=True,
    states_to_log=[pendulum.state.full_name, lqr.action.full_name]
)
graph.resolve(graph.variables)

# Run simulation
for _ in range(5000):
    graph.step()
```

This example demonstrates key features of Regelum:
- Modular node-based architecture
- Built-in controllers (LQR, PID, MPC)
- Real-time visualization
- Automatic state management
- Easy system composition

For more advanced examples, check out:
- State estimation with UKF
- Model predictive control
- Custom reward functions
- Advanced visualization

## Documentation

Visit our [documentation](https://aidagroup.github.io/regelum/) for:
- Detailed tutorials and examples
- API reference
- Mathematical foundations
- Best practices

## Development

We use several tools to maintain code quality:

- **Type Checking**: MyPy with strict settings
- **Linting**: Ruff and Pylint
- **Formatting**: Black
- **Testing**: Pytest
- **Pre-commit**: Automated checks before commits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality:
   ```bash
   pre-commit run --all-files
   pytest
   ```
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- Georgiy Malaniya
- Pavel Osinenko
- Anton Bolychev
- Grigory Yaremenko

## Origin

Created by Pavel Osinenko, 2015-2016  
Dresden/Chemnitz, Germany

## Citation

If you use Regelum in your research, please cite:

```bibtex
@software{regelum2024,
  author = {Malaniya, Georgiy and Osinenko, Pavel and Bolychev, Anton and Yaremenko, Grigory},
  title = {Regelum: A Framework for Reinforcement Learning and Control},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/aidagroup/regelum}
}
```
