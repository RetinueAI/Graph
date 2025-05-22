# Graph

## Description

Graph is a Python library developed as part of the Mimir autonomous AI agent by RetinueAI. It provides a flexible framework for creating and managing graph data structures to support decision-making processes. The library enables modeling of actions, states, or concepts as nodes, with edges representing relationships or transitions. It includes features for tracking engagement and interaction metrics, supporting hierarchical subgraphs, and synchronizing data with MongoDB for persistent storage. As a work-in-progress project, the API and functionality may evolve. This is work in progress.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Customizable Nodes and Edges**: Define nodes and edges with attributes like engagement scores and interaction strengths to represent decision elements.
- **Metric Tracking**: Update and retrieve metrics to prioritize actions or concepts based on user interactions.
- **Hierarchical Subgraphs**: Model complex scenarios with nested graphs within nodes.
- **MongoDB Synchronization**: Persist graph data using asynchronous operations for scalability.
- **Advanced Operations**: Identify high-interest nodes or optimal edges to inform AI agent decisions.

## Requirements

- Python 3.7 or later
- MongoDB (for synchronization features)
- Dependencies: numpy, torch, torch-geometric, pydantic, networkx, pymongo, motor

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RetinueAI/Graph.git
   cd Graph
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install numpy torch torch-geometric pydantic networkx pymongo motor
   ```

   Note: For `torch-geometric`, follow the official installation guide at [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to ensure compatibility with your PyTorch version. Ensure MongoDB is installed and running for synchronization features.

## Usage

Below is an example of using the Graph library to model a decision graph:

```python
from graph import Graph, Node, Edge

# Initialize a graph for a specific user
graph = Graph(user_id="user123")

# Create nodes representing actions or states
action_a = Node(id=1, name="Send Email")
action_b = Node(id=2, name="Schedule Meeting")

# Add nodes to the graph
graph.add_node(action_a)
graph.add_node(action_b)

# Create an edge representing a transition
transition = Edge(from_node=1, to_node=2)
graph.add_edge(transition)

# Update engagement for an action
graph.update_node_engagement(node_id=1, engagement_delta=1.0)

# Find the node with the highest interest
top_node = graph.get_highest_interest_node()
print(f"Recommended action: {top_node.name}")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the [Graph Repository](https://github.com/RetinueAI/Graph) to discuss enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

## Contact

For any questions or feedback, contact marius.hanssen@retinueai.com or josefjameshard@retinueai.com
