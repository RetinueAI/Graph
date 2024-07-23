from datetime import datetime, timezone
from dateutil import parser
import os
import json
from typing import Dict, Optional, List, Tuple, Any
import uuid

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from pydantic import BaseModel, Field
import networkx as nx
from pymongo.results import InsertOneResult, InsertManyResult, DeleteResult, UpdateResult

from mongo import MongoHandler
    


class RecursiveData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, subgraphs=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        self.subgraphs = subgraphs if subgraphs is not None else {}

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'subgraphs':
            return {k: 0 for k in self.subgraphs.keys()}
        return super().__inc__(key, value, *args, **kwargs)


class Node(BaseModel):
    id: int
    name: str
    interest_frequency: int = Field(default=0)
    eigenvector_centrality: float = Field(default=0.0)
    last_engagement: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engagement_score: float = Field(default=0.0)
    subgraph: Optional['Graph'] = None

    def update_engagement(self):
        self.interest_frequency += 1
        self.last_engagement = datetime.now(timezone.utc)
        self.update_engagement_score()

    def update_engagement_score(self):
        recency = (datetime.now(timezone.utc) - self.last_engagement).total_seconds()
        self.engagement_score = (self.interest_frequency * self.eigenvector_centrality) / (recency + 1)

    def set_subgraph(self, subgraph: 'Graph'):
        self.subgraph = subgraph

    def to_feature_vector(self):
        return [
            self.interest_frequency,
            self.eigenvector_centrality,
            self.engagement_score,
            (datetime.now(timezone.utc) - self.last_engagement).total_seconds()
        ]

class Edge(BaseModel):
    from_node: int
    to_node: int
    interaction_strength: int = Field(default=0)
    last_interaction: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    contextual_similarity: float = Field(default=0.0)
    sequential_relation: float = Field(default=0.0)

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

class Graph(BaseModel):
    version: int = 1
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    nodes: Dict[int, Node] = Field(default_factory=dict)
    edges: Dict[Tuple[int, int], Edge] = Field(default_factory=dict)
    pyg_data: Optional[RecursiveData] = None

    class Config:
        arbitrary_types_allowed = True

    def add_node(self, id: int, name: str) -> Node:
        node = Node(id=id, name=name)
        self.nodes[id] = node
        self._update_pyg_data()
        return node

    def add_edge(self, from_node: int, to_node: int) -> Edge:
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")
        edge_key = (from_node, to_node)
        if edge_key not in self.edges:
            edge = Edge(from_node=from_node, to_node=to_node)
            self.edges[edge_key] = edge
            self._update_pyg_data()
        return self.edges[edge_key]

    def get_edge(self, from_node: int, to_node: int) -> Edge:
        return self.edges.get((from_node, to_node))

    def get_node(self, node_id: int) -> Node:
        return self.nodes.get(node_id)

    def remove_edge(self, from_node: int, to_node: int):
        self.edges.pop((from_node, to_node), None)
        self._update_pyg_data()

    def remove_node(self, node_id: int):
        self.nodes.pop(node_id, None)
        self.edges = {k: v for k, v in self.edges.items() if node_id not in k}
        self._update_pyg_data()

    def update_node_engagement(self, node_id: int):
        if node_id in self.nodes:
            self.nodes[node_id].update_engagement()
            self._update_pyg_data()

    def update_edge_interaction(self, from_node: int, to_node: int):
        edge_key = (from_node, to_node)
        if edge_key in self.edges:
            self.edges[edge_key].update_interaction()
            self._update_pyg_data()

    def calculate_eigenvector_centrality_pyg(self, max_iterations=100, tolerance=1e-6):
        def calculate_for_graph(graph):
            # Convert edge_index to dense adjacency matrix
            adj_matrix = to_dense_adj(graph.edge_index)[0]
            
            # Initialize centrality vector
            centrality = torch.ones(graph.num_nodes, device=graph.x.device)
            
            for _ in range(max_iterations):
                prev_centrality = centrality.clone()
                
                # Compute new centrality
                centrality = torch.matmul(adj_matrix, centrality)
                
                # Normalize
                centrality = centrality / centrality.norm(p=2)
                
                # Check for convergence
                if torch.allclose(centrality, prev_centrality, rtol=tolerance):
                    break
            
            return centrality

        def recursive_centrality(graph):
            # Calculate centrality for current graph
            centrality = calculate_for_graph(graph)
            
            # Update node features with centrality
            graph.x[:, 1] = centrality  # Assuming eigenvector_centrality is the second feature

            # Recursively calculate for subgraphs
            for node_id, subgraph in graph.subgraphs.items():
                if subgraph is not None:
                    recursive_centrality(subgraph)
                    
                    # Propagate subgraph centrality to parent graph
                    subgraph_centrality = subgraph.x[:, 1].mean()
                    graph.x[node_id, 1] = (graph.x[node_id, 1] + subgraph_centrality) / 2

            return graph

        # Start the recursive calculation from the top-level graph
        updated_graph = recursive_centrality(self.pyg_data)
        
        # Update the Graph object with new centrality values
        for node_id, centrality in enumerate(updated_graph.x[:, 1]):
            self.nodes[node_id].eigenvector_centrality = centrality.item()

    def calculate_eigenvector_centrality(self):
        nx_graph = self.to_networkx()
        centrality = nx.eigenvector_centrality_numpy(nx_graph)
        for node_id, centrality_value in centrality.items():
            self.nodes[node_id].eigenvector_centrality = centrality_value
        self._update_pyg_data()

    def set_node_subgraph(self, node_id: int, subgraph: 'Graph'|Dict):
        if node_id in self.nodes:

            if isinstance(subgraph, Dict):
                subgraph = Graph(user_id=self.user_id).from_dict(subgraph)

            self.nodes[node_id].set_subgraph(subgraph)
            self._update_pyg_data()

    def _update_pyg_data(self):
        node_features = [node.to_feature_vector() for node in self.nodes.values()]
        edge_index = [[from_node, to_node] for from_node, to_node in self.edges.keys()]
        edge_attr = [edge.to_feature_vector() for edge in self.edges.values()]
        subgraphs = {
            node_id: node.subgraph.pyg_data if node.subgraph else None
            for node_id, node in self.nodes.items() if node.subgraph
        }
        self.pyg_data = RecursiveData(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            subgraphs=subgraphs
        )

    def to_networkx(self):
        G = nx.DiGraph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.dict())
        for (from_node, to_node), edge in self.edges.items():
            G.add_edge(from_node, to_node, **edge.dict())
        return G

    @classmethod
    def from_networkx(cls, nx_graph):
        graph = cls()
        for node, data in nx_graph.nodes(data=True):
            graph.add_node(node, data['name'])
            for key, value in data.items():
                if key != 'name':
                    setattr(graph.nodes[node], key, value)
        for from_node, to_node, data in nx_graph.edges(data=True):
            edge = graph.add_edge(from_node, to_node)
            for key, value in data.items():
                setattr(edge, key, value)
        graph._update_pyg_data()
        return graph
    

    def to_dict(self):
        return {
            "version": self.version,
            "id": self.id,
            "user_id": self.user_id,
            "nodes": {
                str(node_id): {
                    **{k: (v.isoformat() if isinstance(v, datetime) else v) 
                       for k, v in node.model_dump(exclude={"subgraph"}).items()},
                    "subgraph_id": node.subgraph.id if node.subgraph else None
                } for node_id, node in self.nodes.items()
            },
            "edges": {
                f"{from_node},{to_node}": {
                    k: (v.isoformat() if isinstance(v, datetime) else v) 
                    for k, v in edge.model_dump().items()
                } 
                for (from_node, to_node), edge in self.edges.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        try:
            graph = cls(version=data['version'], id=data['id'], user_id=data['user_id'])
            for node_id, node_data in data['nodes'].items():
                parsed_node_data = {
                    k: (parser.isoparse(v) if isinstance(v, str) and k in ['last_engagement'] else v)
                    for k, v in node_data.items() if k != 'subgraph_id'
                }
                node = Node(**parsed_node_data)
                graph.nodes[int(node_id)] = node
            
            for edge_key, edge_data in data['edges'].items():
                from_node, to_node = map(int, edge_key.split(','))
                parsed_edge_data = {
                    k: (parser.isoparse(v) if isinstance(v, str) and k in ['last_interaction'] else v)
                    for k, v in edge_data.items()
                }
                edge = Edge(**parsed_edge_data)
                graph.edges[(from_node, to_node)] = edge
            
            graph._update_pyg_data()
        except Exception as e:
            print(f"Error in from_dict: {e}")
            raise

        return graph


class GraphSync(BaseModel):
    graph: Graph
    graph_map: Dict = Field(default={})
    mongo_handler: MongoHandler
    local_storage_path: str
    last_sync_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True

    async def apply_changes(self, changes: List[Dict]):
        for change in changes:
            await self.apply_change_recursive(self.graph, change)
        self.graph._update_pyg_data()

    async def apply_change_recursive(self, graph: Graph, change: Dict[str, Any]):
        if change['type'] == 'add_node':
            node = graph.add_node(change['node_id'], change['node_data']['name'])
            for key, value in change['node_data'].items():
                if key != 'name' and key != 'subgraph':
                    setattr(node, key, value)
            if 'subgraph' in change['node_data']:
                subgraph = Graph()
                node.set_subgraph(subgraph)
                for subchange in change['node_data']['subgraph']:
                    await self.apply_change_recursive(subgraph, subchange)
        elif change['type'] == 'update_node':
            node = graph.nodes[change['node_id']]
            for key, value in change['node_data'].items():
                if key != 'subgraph':
                    setattr(node, key, value)
            if 'subgraph' in change['node_data']:
                if node.subgraph is None:
                    node.set_subgraph(Graph())
                for subchange in change['node_data']['subgraph']:
                    await self.apply_change_recursive(node.subgraph, subchange)
        elif change['type'] == 'add_edge':
            graph.add_edge(change['from_node'], change['to_node'])
        elif change['type'] == 'update_edge':
            edge = graph.edges[(change['from_node'], change['to_node'])]
            for key, value in change['edge_data'].items():
                setattr(edge, key, value)
        elif change['type'] == 'delete_node':
            graph.nodes.pop(change['node_id'], None)
        elif change['type'] == 'delete_edge':
            graph.edges.pop((change['from_node'], change['to_node']), None)


    async def save_graphs_to_database(self, main_graph: Graph) -> InsertManyResult:
        graphs_to_save = self.collect_graphs(main_graph)
        
        return await self.mongo_handler.insert_many(
            entries=graphs_to_save,
            db_name='Graph',
            collection_name='Graphs'
        )

    def collect_graphs(self, graph: Graph) -> List[Dict[str, Any]]:
        graphs = [{'_id': graph.id, 'graph': graph.to_dict()}]

        for node in graph.nodes.values():
            if node.subgraph:
                graphs.extend(self.collect_graphs(node.subgraph))

        return graphs
    
    async def save_map_to_database(self) -> InsertOneResult:
        exists = await self.mongo_handler.document_exists(
            db_name='Graph', 
            collection_name='Maps', 
            filter={'_id': self.graph.user_id}
        )

        if exists:
            return await self.mongo_handler.update_document(
                entry={'_id': self.graph.user_id, 'map': self.graph_map}, 
                query={'_id': self.graph.user_id}, 
                db_name='Graph', 
                collection_name='Maps'
            )
        else:
            return await self.mongo_handler.insert(
                entry={'_id': self.graph.user_id, 'map': self.graph_map}, 
                db_name='Graph', 
                collection_name='Maps'
            )
        

    async def load_graph_map_from_database(self) -> Optional[Dict]:
        result = await self.mongo_handler.get_document(
            db_name='Graph', 
            collection_name='Maps', 
            filter={'_id': self.graph.user_id}
        )
         
        return result['map']


    async def load_graphs_from_database(self, user_id: str) -> bool:
        try:
            graphs = await self.mongo_handler.get_documents(
                db_name='Graph',
                collection_name='Graphs',
                query={"graph.user_id": user_id},
                length=None
            )

            if not graphs:
                raise ValueError(f"No graphs found for user {user_id}")

            graphs = {graph['_id']: graph['graph'] for graph in graphs}

            graph = Graph(user_id=user_id).from_dict(graphs[self.graph.id])
            graph_map = await self.load_graph_map_from_database()
            self.graph_map = graph_map

            try:
                await self.reconstruct_graph(graph=graph, graphs=graphs, graph_map=graph_map[self.graph.id])
            except Exception as e:
                print(e)
                raise

            self.graph = graph
            return True
        except Exception as e:
            print(e)
            raise

    async def reconstruct_graph(self, graph: Graph, graphs: Dict[str, Graph], graph_map: Dict[str, str]):
        for node in graph_map.keys():
            try:
                graph_id = list(graph_map[node].keys())[0] 
                graph.set_node_subgraph(int(node), subgraph=graphs[graph_id])
            except Exception as e:
                print(e)
                raise

            await self.reconstruct_graph(graph=graph.nodes[int(node)].subgraph, graphs=graphs, graph_map=graph_map[node][graph_id])