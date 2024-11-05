# Welcome to the Regelum Documentation

## First variant

### Overview  
Regelum is a powerful tool designed for researchers and engineers working with reinforcement learning (RL) and optimal control. It streamlines the creation of dynamic simulation pipelines by removing the need for cumbersome, monolithic architectures. With our modular, node-based approach, users can easily build, expand, and customize their RL workflows. Each component operates as an independent node with shared access to a global variable scope, enabling easy scaling and smooth integration of new features without reworking the entire architecture.

### Key Features
- **Modular Node-Based Architecture**  
  - Each functional component is represented as an independent node, making it easy to add or replace parts of the pipeline.

- **Global Variable Scope**  
  - Nodes can interact with a shared global variable scope, simplifying data coordination and ensuring consistency across components.

- **Synchronous Loop**  
  - Built for RL tasks, Regelum runs in a synchronous loop, providing optimized performance and predictability.

- **Easy Feature Integration**  
  - New functions, such as logging (e.g., MLFlow integration), Kalman filtering, or custom constraints, can be added as separate nodes with minimal configuration.

- **Scalability**  
  - Regelum supports horizontal scaling, allowing for distributed loads across nodes without architectural overhauls.

- **Reduced Development Overhead**  
  - Quickly design and launch RL pipelines without extensive rework, reducing both development and maintenance time.

Regelum makes it easier than ever to develop, scale, and manage RL simulations, empowering users to focus on their research rather than the complexities of implementation.


--- 
## Second variant

### Overview 
Regelum is a versatile tool for researchers and engineers in reinforcement learning (RL) and optimal control, offering a streamlined way to build and manage dynamic simulation pipelines. Its modular, node-based design makes complex RL workflows simpler, more scalable, and fully customizable.

### Key Features

- **Modular Design**: Each component functions independently as a node, allowing users to modify or add parts without disrupting the entire system.
- **Global Variable Scope**: All nodes access a shared variable scope, enabling seamless data flow and collaboration between nodes.
- **Scalable and Flexible**: Regelum’s architecture is built to support growth, adapting easily to increasingly complex simulations.
- **Easy Integration**: New features and custom elements can be introduced without needing to rebuild existing workflows.
- **Efficient RL Workflow Customization**: Customize reinforcement learning processes effortlessly, enhancing experimentation and development speeds.