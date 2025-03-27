# Deep Dive: The Graph System in Regelum

The Graph system is the central orchestration mechanism in Regelum that manages node dependencies, execution order, and data flow. This document provides an in-depth exploration of how the Graph system works and how to leverage its advanced features.

## Table of Contents
1. [Graph Architecture](#graph-architecture)
2. [Dependency Resolution](#dependency-resolution)
3. [Execution Engine](#execution-engine)
4. [Time Synchronization](#time-synchronization)
5. [Hierarchical Composition](#hierarchical-composition)
6. [State Management](#state-management)
7. [Subgraph Detection and Extraction](#subgraph-detection-and-extraction)
8. [Graph Visualization](#graph-visualization)
9. [Advanced Use Cases](#advanced-use-cases)

## Graph Architecture

At its core, the Graph class in Regelum is both a container for nodes and a node itself. This dual nature enables hierarchical composition and recursive operations across the node network.

### Inheritance Hierarchy

```
INode (interface)
  ↑
  Node (base implementation)
    ↑
    Graph (node container + orchestrator)
      ↑
      ParallelGraph (distributed execution)
```

As a subclass of `Node`, a Graph inherits:
- Variable management
- Reset behavior
- Step execution
- Input/output handling

However, it extends these capabilities with node management and orchestration features:

### Key Components

The Graph class includes several key components:

1. **Node Registry**: List of managed nodes with metadata
   ```python
   self._nodes: List[Node]  # List of nodes in the graph
   ```

2. **Clock and Timing**: Optional time synchronization
   ```python
   self._clock: Optional[Clock]  # Clock node for time management
   ```

3. **Dependency Management**: Internal structures to track node relationships
   ```python
   # Built dynamically via the _build_dependency_graph method
   dependencies: Dict[str, Set[str]]  # Maps node names to dependency node names
   ```

4. **Execution Pipeline**: Internal sorting of nodes for execution
   ```python
   # Result of _sort_nodes_by_dependencies method
   ordered_nodes: List[Node]  # Nodes sorted for correct execution order
   ```

## Dependency Resolution

One of the most critical functions of the Graph is dependency resolution - the process of connecting node outputs to inputs across the entire system.

### Resolution Process

1. **Variable Collection**: First, the graph collects all variables from its nodes
   ```python
   def _collect_node_data(self) -> None:
       self._variables = []
       for node in self.nodes:
           self._variables.extend(node.variables)
   ```

2. **Dependency Mapping**: Next, it builds a dependency graph representing node relationships
   ```python
   def _build_dependency_graph(self) -> Dict[str, Set[str]]:
       dependencies = {node.external_name: set() for node in self.nodes}
       providers = {}
       
       # Register providers
       for node in self.nodes:
           for var in node.variables:
               providers[var.full_name] = node.external_name
               
       # Build dependencies
       for node in self.nodes:
           if node.inputs and not node.is_root:
               for input_name in node.inputs.inputs:
                   if input_name in providers:
                       dependencies[node.external_name].add(providers[input_name])
                       
       return dependencies
   ```

3. **Resolution**: Then, it resolves each node's inputs to actual variable references
   ```python
   def resolve(self, variables: List[IVariable]) -> ResolveStatus:
       resolve_status = ResolveStatus.SUCCESS
       
       for node in self.nodes:
           resolved_inputs, unresolved = node.get_resolved_inputs(variables)
           node.resolved_inputs = resolved_inputs
           
           if unresolved and not node.is_root:
               resolve_status = ResolveStatus.PARTIAL
               # Handle unresolved inputs...
               
       self.resolve_status = resolve_status
       return resolve_status
   ```

4. **Circular Dependency Detection**: It detects and handles circular dependencies
   ```python
   def _detect_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[List[str]]:
       # Uses Kosaraju's algorithm to find strongly connected components
       return find_scc(dependencies)
   ```

### Resolution Status

The resolve operation returns one of three statuses:
- `SUCCESS`: All dependencies were successfully resolved
- `PARTIAL`: Some nodes have unresolved dependencies but can still function (roots)
- `FAILURE`: Critical dependencies remain unresolved

### Variables vs. Full Names

A key concept in dependency resolution is the distinction between:
- **Variable objects**: In-memory instances used for value access
- **Full names**: String identifiers used for dependency declaration

Full names follow the pattern `node_external_name.variable_name`, such as:
- `pendulum_1.state`
- `controller_2.action`
- `sensor_3.measurement`

## Execution Engine

The graph's execution engine is responsible for running nodes in the correct order based on their dependencies.

### Dependency-Based Sorting

Before execution, nodes are sorted using a modified topological sort:

```python
def _sort_nodes_by_dependencies(self, dependencies: Dict[str, Set[str]]) -> List[Node]:
    root_nodes = [node for node in self.nodes if node.is_root]
    non_root_nodes = [node for node in self.nodes if not node.is_root]
    ordered = []
    visited = set()
    
    def visit(node: Node) -> None:
        if node.external_name in visited:
            return
        visited.add(node.external_name)
        
        for dep_name in dependencies.get(node.external_name, set()):
            dep_node = next(n for n in self.nodes if n.external_name == dep_name)
            if not dep_node.is_root:
                visit(dep_node)
                
        ordered.append(node)
    
    # Add root nodes first
    ordered.extend(root_nodes)
    visited.update(node.external_name for node in root_nodes)
    
    # Visit remaining nodes
    for node in non_root_nodes:
        visit(node)
        
    return ordered
```

This sorting ensures that:
1. Root nodes execute first (no dependencies)
2. For non-root nodes, all dependencies are executed before the node itself

### Execution Mechanism

The `step()` method orchestrates the execution of all nodes:

```python
def step(self) -> None:
    # Get sorted nodes if not already cached
    if not hasattr(self, "_sorted_nodes") or self._sorted_nodes is None:
        dependencies = self._build_dependency_graph()
        self._sorted_nodes = self._sort_nodes_by_dependencies(dependencies)
    
    # Execute each node, potentially multiple times based on n_step_repeats
    for _ in range(self.n_step_repeats):
        for node in self._sorted_nodes:
            # Skip nodes without proper dependencies
            if node.resolved_inputs is None and not node.is_root:
                continue
                
            # Execute the node's step method
            node.step()
            
            # Update last update time if using a clock
            if self._clock is not None:
                node.last_update_time = self._clock.time.value
```

### Repeated Execution

The `n_step_repeats` parameter allows a graph to execute multiple iterations for each call to `step()`. This is useful for:
- Simulating faster dynamics for some nodes
- Improving numerical stability via smaller effective time steps
- Supporting multi-rate control systems

## Time Synchronization

Graphs support complex time synchronization across nodes with different time scales:

### Step Size Mechanism

The `_validate_and_set_step_sizes` method ensures consistent time steps:

```python
def _validate_and_set_step_sizes(self, nodes: List[Node]) -> float:
    defined_step_sizes = [node.step_size for node in nodes if node.step_size is not None]
    if not defined_step_sizes:
        raise ValueError("At least one node must have a defined step_size")
    
    # Find the greatest common divisor of all step sizes
    fundamental_step_size = self.define_fundamental_step_size(nodes)
    
    # Set step sizes for nodes without defined step sizes
    for node in nodes:
        if node.step_size is None:
            node.step_size = fundamental_step_size
            
    return fundamental_step_size
```

### GCD Algorithm for Step Sizes

For nodes with different step sizes, the graph finds a common time step using a modified GCD algorithm:

```python
def define_fundamental_step_size(self, nodes: List[Node]) -> float:
    step_sizes = [node.step_size for node in nodes if node.step_size is not None]
    
    def float_gcd(a: float, b: float) -> float:
        precision = 1e-9
        a, b = round(a / precision), round(b / precision)
        return gcd(int(a), int(b)) * precision
    
    return reduce(float_gcd, step_sizes) if len(set(step_sizes)) > 1 else step_sizes[0]
```

### Aligning Discrete Nodes

The graph also ensures that discrete nodes execute at the right frequency:

```python
def _align_discrete_nodes_execution_with_step_size(self, discrete_nodes: List[Node]) -> None:
    for node in discrete_nodes:
        if node.step_size % self.step_size != 0:
            if self.debug:
                logger.warning(
                    f"Discrete node {node.external_name} has step size {node.step_size} "
                    f"which is not divisible by graph step size {self.step_size}. "
                    f"Rounding to nearest multiple."
                )
            node.step_size = round(node.step_size / self.step_size) * self.step_size
```

### Time Management Through Clock Node

When `initialize_inner_time=True`, a Graph automatically creates:
1. A `Clock` node that tracks simulation time
2. A `StepCounter` node that counts simulation steps

This enables time-based behaviors like:
- Scheduled resets
- Time-varying parameters
- Synchronized multi-rate systems
- Event triggering based on time

## Hierarchical Composition

One of the most powerful features of Regelum's Graph system is its support for hierarchical composition - graphs within graphs.

### Nested Graph Resolution

When a Graph contains another Graph, the resolution process works recursively:

```python
def resolve(self, variables: List[IVariable]) -> ResolveStatus:
    for node in self.nodes:
        # Handle nested graphs recursively
        if isinstance(node, Graph):
            # First resolve the inner graph with its own variables
            inner_vars = list(node.variables)
            node.resolve(inner_vars)
            
            # Then resolve with external variables too
            node.resolve(inner_vars + variables)
    
    # Continue with normal resolution...
```

### Variable Visibility

In hierarchical composition, variables follow these visibility rules:
1. Inner graph nodes can access variables within that graph
2. Inner graph nodes cannot access sibling graphs' variables
3. Outer graph nodes can access any variable in any contained graph

### Namespace Management

The hierarchical structure creates an implicit namespace system where:
- Node and variable names in different subgraphs can be the same
- Full names ensure uniqueness through the parent graph's name
- Examples:
  - `graph_1.subgraph_1.controller_1.action`
  - `graph_1.subgraph_2.controller_1.action`

## State Management

Graphs coordinate state management across all nodes, particularly for reset operations.

### Collective Reset

The `reset()` method resets all nodes in the graph:

```python
def reset(self, *, apply_reset_modifier: bool = True) -> None:
    for node in self.nodes:
        node.reset(apply_reset_modifier=apply_reset_modifier)
```

### Reset Modifiers

Graphs can apply reset modifiers to customize reset behavior:

```python
def apply_reset_modifier(
    self, target_node: Node, reset_semaphore: Reset
) -> None:
    """Modify reset behavior of a node based on a reset semaphore."""
    def modified_reset(node_names_to_reset: Optional[List[str]] = None, apply_reset_modifier: bool = True) -> None:
        """Modified reset function that checks the reset flag."""
        if reset_semaphore.flag.value:
            # Original reset
            if node._original_reset is not None:
                node._original_reset(node_names_to_reset, apply_reset_modifier)
            else:
                node.reset(apply_reset_modifier=apply_reset_modifier)
            reset_semaphore.flag.value = False
    
    node = target_node
    if node._original_reset is None:
        node._original_reset = node.reset
    node._modified_reset = modified_reset
    node.reset = node._modified_reset  # type: ignore
```

### Reset Types

Regelum supports several reset patterns:
1. **Manual Reset**: Directly calling `graph.reset()`
2. **Periodic Reset**: Using `ResetEachNSteps` nodes
3. **Condition-Based Reset**: Using custom reset nodes
4. **Cascading Reset**: Resetting parent graphs triggers child resets

## Subgraph Detection and Extraction

Graphs can analyze their node structure to detect independent subgraphs and extract targeted paths.

### Detecting Subgraphs

The `detect_subgraphs()` method identifies independently executable groups:

```python
def detect_subgraphs(self) -> List[List[Node]]:
    """Detect independent subgraphs in the node network."""
    subgraphs = []
    dependencies = self._build_dependency_graph()
    bidirectional = self._build_bidirectional_dependencies(providers)
    
    # Find connected components in the bidirectional graph
    used_nodes = set()
    remaining = list(self.nodes)
    
    while remaining:
        node = remaining[0]
        group = [node]
        node_deps = dependencies.get(node.external_name, set())
        
        for other in remaining[1:]:
            if dependencies.get(other.external_name, set()) == node_deps:
                group.append(other)
                
        subgraphs.append(group)
        used_nodes.update(n.external_name for n in group)
        remaining = [n for n in remaining if n.external_name not in used_nodes]
    
    return subgraphs
```

This enables:
- Parallel execution of independent subgraphs
- Modular debugging of specific components
- Performance optimization through selective execution

### Path Extraction

The `extract_path_as_graph()` method creates a minimal subgraph containing a specific path:

```python
def extract_path_as_graph(self, path: str, n_step_repeats: int = 1) -> Graph:
    """Extract a minimal subgraph containing specified nodes."""
    if not path or "->" not in path:
        raise ValueError("Path must be in format: 'node1 -> node2 -> node3'")
        
    node_names = [name.strip() for name in path.split("->")]
    
    # Verify nodes exist
    name_to_node = {node.external_name: node for node in self.nodes}
    missing = [name for name in node_names if name not in name_to_node]
    if missing:
        raise ValueError(f"Could not find nodes: {missing}")
        
    # Build dependency graph
    dependencies = self._build_dependency_graph()
    required_nodes = self._find_required_nodes(node_names, dependencies)
    subgraph_nodes = [node for node in self.nodes if node.external_name in required_nodes]
    
    result = Graph(subgraph_nodes, debug=self.debug, n_step_repeats=n_step_repeats)
    return result
```

This enables:
- Focused analysis of specific execution paths
- Debugging complex dependency chains
- Creating specialized subgraphs for specific tasks

## Graph Visualization

The Graph system supports visualization to help understand complex node networks:

### String Representation

The `__str__()` method provides a text-based visualization:

```python
def __str__(self) -> str:
    """Return string representation of the graph."""
    nodes_str = "\n".join(
        f"{i:2d}. {node.__class__.__name__:<20} (root={node.is_root}, name={node.external_name})"
        for i, node in enumerate(self.nodes, 1)
    )
    return f"Graph with {len(self.nodes)} nodes:\n{nodes_str}"
```

### Dependency Visualization

For debugging and analysis, the graph can log detailed dependency information:

```python
def _log_dependency_analysis(self, dependencies: Dict[str, Set[str]]) -> None:
    """Log detailed dependency information."""
    if not self.debug:
        return
        
    logger.info("Node dependency analysis:")
    for node_name, deps in sorted(dependencies.items()):
        if deps:
            logger.info(f"  {node_name} depends on: {', '.join(sorted(deps))}")
        else:
            logger.info(f"  {node_name} is a root node")
            
    # Log circular dependencies
    circles = self._detect_circular_dependencies(dependencies)
    if circles:
        logger.warning(f"Detected {len(circles)} circular dependencies:")
        for i, circle in enumerate(circles, 1):
            if len(circle) > 1:
                logger.warning(f"  Circle {i}: {' -> '.join(circle)} -> {circle[0]}")
```

### Task Graph Visualization

In ParallelGraph mode, the system can generate a visual representation of the task graph:

```python
def _log_debug_info(self, node_futures: Dict[Node, NodeFuture]) -> None:
    logger.info(f"Submitting {len(node_futures)} tasks to Dask...")
    visualize(*node_futures.values(), filename="task_graph")
    logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
```

## Advanced Use Cases

### 1. Dynamic Graph Modification

Graphs support dynamic modification during execution:

```python
# Adding nodes
graph.add_node(new_node)
graph.resolve(graph.variables)  # Re-resolve dependencies

# Removing nodes
graph.remove_node(existing_node)
graph.resolve(graph.variables)  # Re-resolve dependencies

# Replacing nodes
graph.remove_node(old_node)
graph.add_node(new_node)
graph.resolve(graph.variables)
```

### 2. Monte Carlo Simulations

Graphs support Monte Carlo simulations through reset modifiers:

```python
# Create system with random reset behavior
pendulum = Pendulum(
    control_signal_name="controller_1.action",
    initial_state=np.array([np.pi, 0.0]),
    state_reset_modifier=lambda x: np.array([np.pi + np.random.normal(0, 0.1), 0.0])
)

# Create reset node
reset_node = ResetEachNSteps(
    node_name_to_reset=pendulum.external_name, 
    n_steps=100
)

# Run multiple episodes
for episode in range(100):
    for _ in range(100):  # 100 steps per episode
        graph.step()
    # Analyze results and reset for next episode
    analyze_episode_results()
```

### 3. Multi-Rate Systems

Graphs support systems with different time scales:

```python
# Fast dynamics (0.001s step)
fast_system = FastDynamics(step_size=0.001)

# Medium dynamics (0.01s step)
medium_system = MediumDynamics(step_size=0.01)

# Slow controller (0.1s step)
slow_controller = SlowController(step_size=0.1)

# Graph automatically handles different rates
graph = Graph([fast_system, medium_system, slow_controller])
```

### 4. Hierarchical Control

Graphs support complex hierarchical control structures:

```python
# Inner loop - fast dynamics (position control)
inner_controller = PositionController(...)
inner_plant = PositionSystem(...)
inner_graph = Graph([inner_controller, inner_plant])

# Outer loop - slow dynamics (force control)
outer_controller = ForceController(...)
outer_plant = ForceSystem(...)
outer_graph = Graph([outer_controller, outer_plant])

# Meta-controller (coordination)
meta_controller = CoordinationController(...)

# Complete system
main_graph = Graph([inner_graph, outer_graph, meta_controller])
```

### 5. Hybrid Systems

Graphs support hybrid systems mixing continuous and discrete dynamics:

```python
# Continuous dynamics
continuous_system = ContinuousDynamics(is_continuous=True)

# Event detector
event_detector = EventDetector(inputs=[continuous_system.state.full_name])

# Discrete state machine
state_machine = StateMachine(inputs=[event_detector.event.full_name])

# Hybrid controller
hybrid_controller = HybridController(
    inputs=[continuous_system.state.full_name, state_machine.mode.full_name]
)

# Connect everything
graph = Graph([continuous_system, event_detector, state_machine, hybrid_controller])
```

## Conclusion

The Graph system is the foundation of Regelum's composable, modular architecture. By understanding its internal mechanisms, you can leverage its full power to build complex, hierarchical systems with proper dependency management, efficient execution, and robust synchronization.

Whether you're building simple control loops or complex multi-rate hybrid systems, the Graph system provides the tools to organize, connect, and orchestrate your components for efficient and reliable execution. 