# Deep Dive: The Node System in Regelum

The Node system forms the foundation of Regelum's computational architecture. This document provides an in-depth exploration of Node's design, implementation details, and advanced usage patterns.

## Table of Contents
1. [Node Architecture](#node-architecture)
2. [State and Variable Management](#state-and-variable-management)
3. [Input System](#input-system)
4. [Lifecycle Management](#lifecycle-management)
5. [Continuous vs. Discrete Dynamics](#continuous-vs-discrete-dynamics)
6. [Reset Mechanisms](#reset-mechanisms)
7. [Node Identification and Naming](#node-identification-and-naming)
8. [Node Internals](#node-internals)
9. [Advanced Node Patterns](#advanced-node-patterns)

## Node Architecture

Nodes are the fundamental computational units in Regelum, encapsulating both state and behavior. Each node follows a single-responsibility principle, handling a specific aspect of the overall system.

### Inheritance Hierarchy

```
INode (interface)
  ↑
  Node (base implementation)
    ↑
    Custom Node Types (User implementations)
```

### Interface Design

The `INode` interface defines the core operations that all nodes must support:

```python
class INode(IResettable, IResolvable[IInputs, IVariable], Protocol):
    """Interface for computational nodes in the graph."""
    
    @abstractmethod
    def step(self) -> None:
        """Execute one computational step."""
        
    @property
    @abstractmethod
    def variables(self) -> Sequence[IVariable]:
        """Get list of variables owned by this node."""
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Get internal name of the node."""
        
    @property
    @abstractmethod
    def external_name(self) -> str:
        """Get external name of the node."""
        
    @property
    @abstractmethod
    def is_resolved(self) -> bool:
        """Check if node's inputs are resolved."""
        
    @abstractmethod
    def get_full_names(self) -> List[str]:
        """Get fully qualified names of all variables."""
```

This interface establishes a contract that all nodes must fulfill, ensuring compatibility across the system.

### Core Functionality

The base `Node` class implements this interface with these primary responsibilities:

1. **State Management**: Defining and tracking variables
2. **Computation**: Executing steps based on inputs and internal state
3. **Input Handling**: Managing dependencies on other nodes
4. **Lifecycle**: Supporting initialization, reset, and cleanup
5. **Time Management**: Handling time steps for simulation

## State and Variable Management

One of the node's primary responsibilities is managing state through variables.

### Variable Declaration

Variables are created using the `define_variable` method:

```python
def define_variable(
    self,
    name: str,
    value: Optional[Value] = None,
    metadata: Optional[Dict[str, Any]] = None,
    reset_modifier: Optional[Callable[[Any], Any]] = None,
) -> Variable:
    """Define new variable for this node.
    
    Args:
        name: Local name for the variable
        value: Initial value
        metadata: Additional information about the variable
        reset_modifier: Function to modify reset behavior
        
    Returns:
        New variable instance
    """
    if metadata is None:
        metadata = {}
    
    # Set current and initial values
    metadata["current_value"] = value
    metadata["initial_value"] = deepcopy(value)
    
    # Add shape information if available
    if value is not None and hasattr(value, "shape"):
        metadata["shape"] = value.shape
    
    # Add reset modifier if provided
    if reset_modifier is not None:
        metadata["reset_modifier"] = reset_modifier
    
    # Create the variable
    var = Variable(name=name, metadata=metadata)
    var.node_name = self.external_name
    
    # Register the variable with this node
    self._variables.append(var)
    
    return var
```

This method not only creates the variable but also:
- Handles metadata setup
- Sets initial values
- Configures reset behavior
- Establishes the node-variable relationship
- Registers the variable with the node

### Variable Ownership

Variables are always owned by a specific node:

```python
var.node_name = self.external_name
```

This ownership is crucial for:
- Dependency resolution
- Proper variable access
- Reset cascading
- Full name generation

### Value Propagation

Nodes must actively update their variables during the `step()` method:

```python
def step(self) -> None:
    # Example implementation
    if self.resolved_inputs is None:
        return
    
    input_value = self.resolved_inputs.find(self.inputs.inputs[0]).value
    self.output.value = process_input(input_value)
```

There is no automatic notification system - variable updates must be explicitly performed.

## Input System

Nodes declare dependencies on other nodes through the input system, which consists of unresolved input specifications and resolved input references.

### Input Declaration

Inputs are declared during node initialization:

```python
def __init__(
    self,
    inputs: Optional[Inputs | List[str]] = None,
    *,
    step_size: Optional[float] = None,
    is_continuous: bool = False,
    is_root: bool = False,
    name: Optional[str] = None,
) -> None:
    self._inputs = self._normalize_inputs(inputs)
    # ...rest of initialization...
```

Inputs can be declared as:
- String list: `["sensor_1.measurement", "controller_1.action"]`
- Inputs instance: `Inputs(["sensor_1.measurement", "controller_1.action"])`

### Input Normalization

The inputs are normalized to a standard format:

```python
def _normalize_inputs(self, inputs: Optional[IInputs | List[str]]) -> IInputs:
    """Convert inputs to standardized Inputs instance."""
    if not hasattr(self, "_inputs"):
        if inputs is None:
            return Inputs([])
        return Inputs(inputs) if isinstance(inputs, list) else inputs
    return Inputs(self._inputs) if isinstance(self._inputs, list) else self._inputs
```

This ensures consistent handling regardless of the input format.

### Input Resolution

During graph construction, string-based inputs are resolved to actual variable references:

```python
def get_resolved_inputs(
    self, variables: List[IVariable]
) -> Tuple[IResolvedInputs, Set[str]]:
    """Get resolved inputs and unresolved names."""
    return self._inputs.resolve(variables)
```

The resolution process:
1. Takes a list of all available variables
2. Matches input strings to available variables
3. Creates a `ResolvedInputs` instance with actual references
4. Returns unresolved inputs for error reporting

### Accessing Input Values

Once resolved, input values are accessed during the `step()` method:

```python
def step(self) -> None:
    if self.resolved_inputs is None:
        return  # Cannot operate without inputs
        
    # Find specific input by full name
    input1 = self.resolved_inputs.find("sensor_1.measurement")
    if input1 is not None:
        measurement = input1.value
        
    # Or by variable name only (using fuzzy matching)
    input2 = self.resolved_inputs.find("action")
    if input2 is not None:
        action = input2.value
```

The `resolved_inputs` attribute provides access to the actual variable instances.

## Lifecycle Management

Nodes have a well-defined lifecycle that includes creation, dependency resolution, execution, and reset.

### Instance Registry

Nodes maintain a class-level registry of all instances:

```python
_instances: ClassVar[Dict[str, List["Node"]]] = {}

def __new__(cls, *args: Any, **kwargs: Any) -> Self:
    """Create new node instance and track it."""
    instance = super().__new__(cls)
    cls._instances.setdefault(cls.__name__, []).append(instance)
    return instance
```

This registry enables:
- Automatic instance counting
- Unique name generation
- Node lookup by class
- System-wide state inspection

### Step Execution

The core computational behavior is implemented in the `step()` method:

```python
@abstractmethod
def step(self) -> None:
    """Execute one computational step.
    
    This is the main computation method that must be implemented by derived nodes.
    For continuous nodes, this typically implements numerical integration.
    For discrete nodes, this implements the state update logic.
    """
    pass
```

Derived nodes must implement this method to define their behavior.

### Reset Behavior

Nodes support resetting their state to initial conditions:

```python
def reset(self, *, apply_reset_modifier: bool = True) -> None:
    """Reset node to initial state.
    
    Args:
        apply_reset_modifier: Whether to apply custom reset modifiers
    """
    for var in self.variables:
        var.reset(apply_reset_modifier=apply_reset_modifier)
```

The reset operation:
1. Resets all variables to their initial values
2. Optionally applies custom reset modifiers
3. Can be customized through overrides
4. Can be triggered by the containing graph

## Continuous vs. Discrete Dynamics

Regelum supports both continuous-time and discrete-time dynamics, which are handled differently in nodes.

### Continuous Nodes

Continuous nodes represent continuous-time dynamics governed by differential equations:

```python
class ContinuousSystem(Node):
    def __init__(self, control_signal_name: str):
        super().__init__(
            inputs=[control_signal_name],
            is_continuous=True,  # Mark as continuous
            step_size=0.01
        )
        self.state = self.define_variable("state", value=np.array([0.0, 0.0]))
        
    def state_transition_map(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Define the system dynamics dx/dt = f(x, u)."""
        x1, x2 = state
        u = action[0]
        return np.array([x2, -0.1*x2 - x1 + u])
        
    def step(self) -> None:
        """Implement numerical integration."""
        if self.resolved_inputs is None:
            return
            
        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        
        # Euler integration
        derivative = self.state_transition_map(self.state.value, action)
        self.state.value += derivative * self.step_size
```

Continuous nodes must:
1. Set `is_continuous=True` in initialization
2. Implement a `state_transition_map` method that returns derivatives
3. Implement a `step` method that performs numerical integration

### Discrete Nodes

Discrete nodes represent discrete-time dynamics or event-based behavior:

```python
class DiscreteController(Node):
    def __init__(self, state_name: str):
        super().__init__(
            inputs=[state_name],
            is_continuous=False,  # Mark as discrete
            step_size=0.1         # Can have larger step size
        )
        self.action = self.define_variable("action", value=np.array([0.0]))
        
    def step(self) -> None:
        """Compute control action based on state."""
        if self.resolved_inputs is None:
            return
            
        state = self.resolved_inputs.find(self.inputs.inputs[0]).value
        
        # Discrete control law
        self.action.value = np.array([-10.0 * state[0] - 2.0 * state[1]])
```

Discrete nodes:
1. Set `is_continuous=False` (or omit, as it's the default)
2. Do not need a `state_transition_map` method
3. Implement a `step` method that updates state directly
4. Often use larger step sizes than continuous nodes

### Time Step Handling

The `step_size` parameter defines the time discretization:

```python
def __init__(
    self,
    inputs: Optional[Inputs | List[str]] = None,
    *,
    step_size: Optional[float] = None,  # Time step
    # ...other parameters...
) -> None:
    self._step_size = step_size
    # ...rest of initialization...
```

This parameter is used by:
- The node itself during integration
- The containing graph for time synchronization
- Logging systems for timestamps
- Animation and visualization

## Reset Mechanisms

Nodes support sophisticated reset mechanisms to handle different simulation scenarios.

### Basic Reset

The simplest reset operation resets all variables to their initial values:

```python
def reset(self, *, apply_reset_modifier: bool = True) -> None:
    """Reset node to initial state."""
    for var in self.variables:
        var.reset(apply_reset_modifier=apply_reset_modifier)
```

### Reset Modifiers

Variables can have custom reset modifiers that transform their reset values:

```python
# Random initialization for Monte Carlo simulations
state = self.define_variable(
    "state",
    value=np.array([0.0, 0.0]),
    reset_modifier=lambda x: np.random.normal(0, 0.1, size=2)
)

# Constrained initialization
position = self.define_variable(
    "position",
    value=np.array([0.0, 0.0]),
    reset_modifier=lambda x: np.clip(x + np.random.normal(0, 0.2, size=2), -1, 1)
)
```

Reset modifiers enable:
- Randomized initializations
- Constrained values
- Learning-based modifications
- Progressive difficulty

### Selective Reset

Nodes can implement selective reset based on conditions:

```python
def reset(self, *, apply_reset_modifier: bool = True) -> None:
    """Custom reset implementation."""
    # Only reset some variables
    for var in [self.state, self.velocity]:
        var.reset(apply_reset_modifier=apply_reset_modifier)
        
    # Keep accumulated values
    # self.total_reward.reset(apply_reset_modifier=False)
```

### Reset Hijacking

Graphs can hijack a node's reset behavior using reset modifiers:

```python
def apply_reset_modifier(self, target_node: Node, reset_semaphore: Reset) -> None:
    """Modify reset behavior of a node based on a reset semaphore."""
    def modified_reset(...):
        # Custom reset logic
        
    if target_node._original_reset is None:
        target_node._original_reset = target_node.reset
    target_node._modified_reset = modified_reset
    target_node.reset = target_node._modified_reset
```

This technique enables:
- Conditional resets
- Scheduled resets
- Event-triggered resets
- Reset coordination

## Node Identification and Naming

Nodes use a sophisticated naming system to handle multiple instances and hierarchical relationships.

### Naming Convention

Each node has two names:
- `_internal_name`: Base name without instance numbers
- `_external_name`: Unique name including instance number

```python
def __init__(
    self,
    # ...other parameters...
    name: Optional[str] = None,
) -> None:
    # ...other initialization...
    self._internal_name = name or self.__class__.__name__.lower()
    self._external_name = f"{self._internal_name}_{self.get_instance_count()}"
```

Example names:
- `pendulum_1` (first pendulum instance)
- `pendulum_2` (second pendulum instance)
- `controller_1` (first controller instance)

### Instance Counting

The instance count is determined from the class registry:

```python
@classmethod
def get_instance_count(cls) -> int:
    """Get count of instances for this class."""
    return len(cls._instances.get(cls.__name__, []))
```

This ensures unique external names for each node instance.

### Full Variable Names

Variables have full names that combine the node's external name and the variable's local name:

```python
@property
def full_name(self) -> str:
    """Get fully qualified name."""
    return f"{self.node_name}.{self.name}"
```

Example full names:
- `pendulum_1.state`
- `controller_2.action`
- `sensor_3.measurement`

### Name Usage

These names serve different purposes:
- `internal_name`: Used for internal references and debugging
- `external_name`: Used for variable registration and lookup
- `full_name`: Used for dependency declaration and resolution

## Node Internals

Understanding the internal implementation details of nodes helps with advanced usage and debugging.

### Constructor Flow

The node initialization sequence is:

1. **Instance Registration**: `__new__` registers the instance
2. **Input Normalization**: `_normalize_inputs` standardizes inputs
3. **Parameter Storage**: Step size, continuity flag, etc. are saved
4. **Continuity Validation**: Continuous nodes must have state_transition_map
5. **Name Generation**: Internal and external names are created
6. **Custom Initialization**: Derived class constructors run

### Protected Attributes

The Node class maintains several protected attributes:

```python
_instances: ClassVar[Dict[str, List["Node"]]]  # Instance registry
_inputs: IInputs                              # Input dependencies
_step_size: Optional[float]                   # Time step
_is_continuous: bool                          # Continuity flag
_is_root: bool                                # Root node flag
_variables: List[IVariable]                   # Variable list
_resolved_inputs: Optional[ResolvedInputs]    # Resolved inputs
_internal_name: str                           # Base name
_external_name: str                           # Unique name
_reset_with_modifier: Optional[Callable]      # Reset function
last_update_time: Optional[float]             # Last update time
_original_reset: Optional[Callable]           # Original reset
_modified_reset: Optional[Callable]           # Modified reset
```

These attributes should not be accessed directly by users.

### Property Methods

The Node class provides property methods for safe attribute access:

```python
@property
def variables(self) -> Sequence[IVariable]:
    """Get list of node variables."""
    return self._variables

@property
def name(self) -> str:
    """Get internal name."""
    return self._internal_name

@property
def external_name(self) -> str:
    """Get external name."""
    return self._external_name

@property
def is_resolved(self) -> bool:
    """Check if inputs are resolved."""
    return self._resolved_inputs is not None
    
@property
def step_size(self) -> Optional[float]:
    """Get step size."""
    return self._step_size
    
@step_size.setter
def step_size(self, value: Optional[float]) -> None:
    """Set step size."""
    self._step_size = value
    
@property
def is_continuous(self) -> bool:
    """Check if node is continuous."""
    return self._is_continuous
    
@property
def is_root(self) -> bool:
    """Check if node is a root node."""
    return self._is_root
```

These properties provide controlled access to node attributes.

### Deep Copy Support

Nodes support deep copying for cloning and parallelization:

```python
def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
    """Create an independent deep copy of the node."""
    if id(self) in memo:
        return memo[id(self)]
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    
    # Copy basic attributes
    result._internal_name = self._internal_name
    result._external_name = f"{result._internal_name}_{len(cls._instances[cls.__name__])}"
    
    # Copy all other attributes
    for k, v in self.__dict__.items():
        if k not in ["_external_name", "_variables", "_resolved_inputs"]:
            setattr(result, k, deepcopy(v, memo))
            
    # Handle variables specially
    result._variables = []
    for var in self._variables:
        new_var = deepcopy(var, memo)
        new_var.node_name = result._external_name
        result._variables.append(new_var)
        
    # Clear resolved inputs (will be re-resolved)
    result._resolved_inputs = None
    
    return result
```

This enables:
- Node cloning for parallel execution
- Graph duplication
- Monte Carlo simulations
- Node reuse in different contexts

## Advanced Node Patterns

### 1. State Observer Pattern

Nodes can observe and process other nodes' state:

```python
class StateObserver(Node):
    def __init__(self, observed_state_name: str):
        super().__init__(inputs=[observed_state_name])
        self.observed_value = self.define_variable("observed_value", value=None)
        self.estimated_state = self.define_variable("estimated_state", value=None)
        
    def step(self) -> None:
        if self.resolved_inputs is None:
            return
            
        # Get observed state
        observed = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.observed_value.value = observed
        
        # Process observation (e.g., filtering, estimation)
        self.estimated_state.value = process_observation(observed)
```

### 2. Event Generator Pattern

Nodes can generate events based on conditions:

```python
class EventGenerator(Node):
    def __init__(self, monitored_state_name: str):
        super().__init__(inputs=[monitored_state_name])
        self.event = self.define_variable("event", value=False)
        self.event_counter = self.define_variable("event_counter", value=0)
        
    def step(self) -> None:
        if self.resolved_inputs is None:
            return
            
        # Get monitored state
        state = self.resolved_inputs.find(self.inputs.inputs[0]).value
        
        # Check event condition
        if condition_met(state):
            self.event.value = True
            self.event_counter.value += 1
        else:
            self.event.value = False
```

### 3. Finite State Machine Pattern

Nodes can implement state machines:

```python
class StateMachine(Node):
    def __init__(self, event_name: str):
        super().__init__(inputs=[event_name])
        self.state = self.define_variable("state", value="IDLE")
        self.transitions = {
            "IDLE": {"start": "RUNNING"},
            "RUNNING": {"pause": "PAUSED", "stop": "IDLE"},
            "PAUSED": {"resume": "RUNNING", "stop": "IDLE"}
        }
        
    def step(self) -> None:
        if self.resolved_inputs is None:
            return
            
        # Get event
        event = self.resolved_inputs.find(self.inputs.inputs[0]).value
        
        # Process transition
        if event in self.transitions[self.state.value]:
            self.state.value = self.transitions[self.state.value][event]
```

### 4. Memory Buffer Pattern

Nodes can implement memory buffers for history-dependent processing:

```python
class MemoryBuffer(Node):
    def __init__(self, signal_name: str, buffer_size: int = 10):
        super().__init__(inputs=[signal_name])
        self.buffer = self.define_variable(
            "buffer", 
            value=np.zeros((buffer_size, signal_dim))
        )
        self.buffer_index = self.define_variable("buffer_index", value=0)
        self.buffer_size = buffer_size
        
    def step(self) -> None:
        if self.resolved_inputs is None:
            return
            
        # Get signal
        signal = self.resolved_inputs.find(self.inputs.inputs[0]).value
        
        # Update buffer
        idx = self.buffer_index.value % self.buffer_size
        self.buffer.value[idx] = signal
        self.buffer_index.value += 1
```

### 5. Multi-Input Processor Pattern

Nodes can process multiple inputs with different roles:

```python
class MultiInputProcessor(Node):
    def __init__(self, command_name: str, feedback_name: str):
        super().__init__(inputs=[command_name, feedback_name])
        self.output = self.define_variable("output", value=np.array([0.0]))
        
    def step(self) -> None:
        if self.resolved_inputs is None:
            return
            
        # Get inputs by role
        command = self.resolved_inputs.find(self.inputs.inputs[0]).value
        feedback = self.resolved_inputs.find(self.inputs.inputs[1]).value
        
        # Process based on role
        self.output.value = process_command_and_feedback(command, feedback)
```

## Conclusion

The Node system is the cornerstone of Regelum's composable architecture. By understanding its internal mechanisms, you can effectively design nodes that encapsulate specific behaviors, maintain proper state, and interact correctly with other components.

Whether you're building standard controllers, complex observers, or specialized processors, the Node class provides the foundation for building reliable, modular components that can be composed into sophisticated systems. 