from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple, Any, Union
import uuid

import torch
from torch_geometric.data import Data
from pydantic import BaseModel, Field
import networkx as nx

class Node(BaseModel):
    """
    Represents a node in the global graph. Each node has a unique ID across all graphs.
    """
    id: int
    name: str
    interest_frequency: int = Field(default=0)
    eigenvector_centrality: float = Field(default=0.0)
    last_engagement: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engagement_score: float = Field(default=0.0)
    graph_ids: List[str] = Field(default_factory=list)

    def update_engagement(self):
        self.interest_frequency += 1
        self.last_engagement = datetime.now(timezone.utc)
        self.update_engagement_score()

    def update_engagement_score(self):
        recency = (datetime.now(timezone.utc) - self.last_engagement).total_seconds()
        self.engagement_score = (self.interest_frequency * self.eigenvector_centrality) / (recency + 1)

    def to_feature_vector(self):
        return [
            self.interest_frequency,
            self.eigenvector_centrality,
            self.engagement_score,
            (datetime.now(timezone.utc) - self.last_engagement).total_seconds()
        ]

class Edge(BaseModel):
    """
    Represents an edge in the global graph. Each edge connects two nodes with unique IDs across all graphs.
    """
    from_node: int
    to_node: int
    interaction_strength: int = Field(default=0)
    last_interaction: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    contextual_similarity: float = Field(default=0.0)
    sequential_relation: float = Field(default=0.0)
    graph_ids: List[str] = Field(default_factory=list)

    def update_interaction(self):
        self.interaction_strength += 1
        self.last_interaction = datetime.now(timezone.utc)

    def to_feature_vector(self):
        return [
            self.interaction_strength,
            self.contextual_similarity,
            self.sequential_relation,
            (datetime.now(timezone.utc) - self.last_interaction).total_seconds()
        ]

class GlobalGraph(BaseModel):
    """
    Represents the global graph containing all nodes and edges.
    """
    nodes: Dict[int, Node] = Field(default_factory=dict)
    edges: Dict[Tuple[int, int], Edge] = Field(default_factory=dict)
    graphs: Dict[int, 'Graph'] = Field(default_factory=dict)

    def add_node(self, node_id: str, name: str, graph_id: str) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id, name=name, graph_ids=[graph_id])
        else:
            if graph_id not in self.nodes[node_id].graph_ids:
                self.nodes[node_id].graph_ids.append(graph_id)

        return self.nodes[node_id]

    def add_edge(self, from_node: str, to_node: str, graph_id: str) -> Edge:
        edge_key = (from_node, to_node)
        if from_node == to_node:
            raise ValueError("Self-loops are not allowed")
        if edge_key not in self.edges:
            self.edges[edge_key] = Edge(from_node=from_node, to_node=to_node, graph_ids=[graph_id])
        else:
            if graph_id not in self.edges[edge_key].graph_ids:
                self.edges[edge_key].graph_ids.append(graph_id)
        return self.edges[edge_key]
    
    def add_graph(self, graph: 'Graph') -> None:
        self.graphs[len(self.graphs)] = graph

    def get_graph_edges(self, graph_id: str) -> List[Tuple[str, str]]:
        return [(from_node, to_node) for (from_node, to_node), edge in self.edges.items() if graph_id in edge.graph_ids]

class Graph(BaseModel):
    """
    Represents a subgraph within the global graph.
    """
    id: int = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    node_ids: List[int] = Field(default_factory=list)
    global_graph: GlobalGraph

    class Config:
        arbitrary_types_allowed = True

    def add_node(self, name: str) -> Node:
        node_id = str(uuid.uuid4())
        node = self.global_graph.add_node(node_id, name, self.id)
        self.node_ids.append(node_id)
        return node

    def add_edge(self, from_node: str, to_node: str) -> Edge:
        if from_node not in self.node_ids or to_node not in self.node_ids:
            raise ValueError("Both nodes must exist in the graph")
        
        if from_node == to_node:
            raise ValueError("Self-loops are not allowed")
        
        return self.global_graph.add_edge(from_node, to_node, self.id)

    def get_nodes(self) -> List[Node]:
        return [self.global_graph.nodes[node_id] for node_id in self.node_ids]

    def get_edges(self) -> List[Edge]:
        return [self.global_graph.edges[edge_key] for edge_key in self.global_graph.get_graph_edges(self.id)]

    def to_pyg_data(self) -> Data:
        nodes = self.get_nodes()
        edges = self.get_edges()
        node_features = [node.to_feature_vector() for node in nodes]
        edge_index = [[self.node_ids.index(edge.from_node), self.node_ids.index(edge.to_node)] for edge in edges]
        edge_attr = [edge.to_feature_vector() for edge in edges]
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        )

    def calculate_eigenvector_centrality(self):
        G = nx.DiGraph()
        for edge in self.get_edges():
            G.add_edge(edge.from_node, edge.to_node)
        
        try:
            centrality = nx.eigenvector_centrality(G)
        except nx.PowerIterationFailedConvergence:
            centrality = nx.degree_centrality(G)
        
        for node_id, centrality_value in centrality.items():
            self.global_graph.nodes[node_id].eigenvector_centrality = centrality_value

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "node_ids": self.node_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], global_graph: GlobalGraph):
        return cls(id=data['id'], user_id=data['user_id'], node_ids=data['node_ids'], global_graph=global_graph)