from datetime import datetime, timezone, timedelta
from dateutil import parser
import os
import json
from typing import Dict, Optional, List, Tuple, Any, Union
import uuid
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from pydantic import BaseModel, Field
import networkx as nx
from pymongo.results import InsertOneResult, InsertManyResult, DeleteResult, UpdateResult

from mongo import MongoHandler
    


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
    from_node: List[int]
    to_node: List[int]
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


class NodeMap(BaseModel):
    nodes: Dict[int, Node] = Field(default_factory=dict)
    node_map: Dict[str, int] = Field(default_factory=dict)

    def get_node(self, node: int) -> Node:
        return self.nodes[node]

    def get_node_id(self, node: str):
        return self.node_map[node]
    
    class Config:
        arbitrary_types_allowed = True
    

class Edges(BaseModel):
    edges: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Edge] = Field(default_factory=dict)


    def get_edge(self, _from: Tuple[int, ...], _to: Tuple[int, ...]) -> Edge:
        return self.edges[(_from, _to)]


    def convert_str_list(self, str_list: List[str], graph: 'Graph') -> Tuple[int]:
        if len(str_list) < 1:
            return []
        
        int_list = []

        int_list.append(graph.node_map.node_map[str_list[0]])

        if int_list[0] is None:
            raise ValueError(f"Node {str_list[0]} doesn't exist in graph {graph.id}.")

        if len(str_list) > 1:
            subgraph = graph.node_map.nodes[int_list[0]].subgraph

            if subgraph is None:
                raise ValueError(f"The subgraph in node {int_list[0]} doesn't exist in graph {graph.id}")
            
            int_list.extend(
                self.convert_str_list(str_list[1:], subgraph)
            )

        return tuple(int_list)


    def generate_edge_key(self, from_: List[str], to_: List[str], root_graph: 'Graph') -> Tuple[Tuple[int],Tuple[int]]:
        return (self.convert_str_list(from_, root_graph), self.convert_str_list(to_, root_graph))


    def add_edge_from_str(self, from_: List[str], to_: List[str], root_graph: 'Graph') -> None:
        edge_key = self.generate_edge_key(from_, to_, root_graph)

        
        if edge_key in self.edges:
            raise ValueError("Edge already exists")

        edge = Edge(from_node=edge_key[0], to_node=edge_key[1])
        self.edges[edge_key] = edge


    class Config:
        arbitrary_types_allowed = True


class Graph(BaseModel):
    version: int = 1
    root: bool = Field(default=False)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    node_map: NodeMap = Field(default_factory=NodeMap)
    edges: Edges = Field(default_factory=Edges)
    parent: Optional['Graph'] = None
    depth: int = 0

    class Config:
        arbitrary_types_allowed = True

    def add_node(self, id: int, name: str) -> Node:
        node = Node(id=id, name=name)
        self.node_map.nodes[id] = node
        self.node_map.node_map[name] = id
        return node

    def get_edge(self, _from: Tuple[int, ...], _to: Tuple[int, ...]) -> Edge:
        return self.edges.get_edge((_from, _to))

    def get_node(self, node_id: int) -> Node:
        return self.node_map.get_node(node_id)

    def remove_edge(self, _from: Tuple[int, ...], _to: Tuple[int, ...]):
        self.edges.edges.pop((_from, _to), None)

    def remove_node(self, node_id: int):
        self.node_map.nodes.pop(node_id, None)
        self.edges = {k: v for k, v in self.edges.edges.items() if node_id not in k}

    def update_node_engagement(self, node_id: int):
        if node_id in self.node_map.nodes:
            self.node_map.nodes[node_id].update_engagement()

    def update_edge_interaction(self, _from: Tuple[int, ...], _to: Tuple[int, ...]):
        edge_key = (_from, _to)
        if edge_key in self.edges:
            self.edges.edges[edge_key].update_interaction()


    def set_node_subgraph(self, node_id: int, subgraph: 'Graph'|Dict):
        if node_id in self.node_map.nodes:

            if isinstance(subgraph, Dict):
                subgraph = Graph(user_id=self.user_id).from_dict(subgraph)

            subgraph.parent = self
            subgraph.depth = self.depth + 1
            self.node_map.nodes[node_id].set_subgraph(subgraph)
    

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
                } for node_id, node in self.node_map.nodes.items()
            },
            "edges": {
                f"{from_node},{to_node}": {
                    k: (v.isoformat() if isinstance(v, datetime) else v) 
                    for k, v in edge.model_dump().items()
                } 
                for (from_node, to_node), edge in self.edges.edges.items()
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
                graph.node_map.nodes[int(node_id)] = node
            
            for edge_key, edge_data in data['edges'].items():
                from_node, to_node = map(int, edge_key.split(','))
                parsed_edge_data = {
                    k: (parser.isoparse(v) if isinstance(v, str) and k in ['last_interaction'] else v)
                    for k, v in edge_data.items()
                }
                edge = Edge(**parsed_edge_data)
                graph.edges[(from_node, to_node)] = edge
            

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
            node = graph.node_map.nodes[change['node_id']]
            for key, value in change['node_data'].items():
                if key != 'subgraph':
                    setattr(node, key, value)
            if 'subgraph' in change['node_data']:
                if node.subgraph is None:
                    node.set_subgraph(Graph())
                for subchange in change['node_data']['subgraph']:
                    await self.apply_change_recursive(node.subgraph, subchange)

        elif change['type'] == 'add_edge':
            graph.edges.add_edge_from_str(change['from_node'], change['to_node'])
        elif change['type'] == 'update_edge':
            edge = graph.edges[(change['from_node'], change['to_node'])]
            for key, value in change['edge_data'].items():
                setattr(edge, key, value)
        elif change['type'] == 'delete_node':
            graph.node_map.nodes.pop(change['node_id'], None)
        elif change['type'] == 'delete_edge':
            graph.edges.edges.pop((change['from_node'], change['to_node']), None)


    async def save_graphs_to_database(self, main_graph: Graph) -> InsertManyResult:
        graphs_to_save = self.collect_graphs(main_graph)
        
        return await self.mongo_handler.insert_many(
            entries=graphs_to_save,
            db_name='Graph',
            collection_name='Graphs'
        )


    def collect_graphs(self, graph: Graph) -> List[Dict[str, Any]]:
        graphs = [{'_id': graph.id, 'graph': graph.to_dict()}]

        for node in graph.node_map.nodes.values():
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
        # TODO Make sense of what's going on here.
        # TODO Needs to work with the new Edge implementation.
        for node in graph_map.keys():
            try:
                graph_id = list(graph_map[node].keys())[0] 
                graph.set_node_subgraph(int(node), subgraph=graphs[graph_id])
            except Exception as e:
                print(e)
                raise

            await self.reconstruct_graph(graph=graph.node_map.nodes[int(node)].subgraph, graphs=graphs, graph_map=graph_map[node][graph_id])


class GraphRandomizer(BaseModel):
    graph_ids: List = Field(default=[])

    def simulate_usage(self, graph: Graph, days=30, max_daily_interactions=10):
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        for _ in range(days):
            start_date += timedelta(days=1)
            daily_interactions = random.randint(1, max_daily_interactions)
            
            for _ in range(daily_interactions):
                # Randomly select a node to update
                node = random.choice(list(graph.node_map.nodes.values()))
                graph.update_node_engagement(node.id)
                node.last_engagement = start_date
                
                # Randomly update an edge if it exists
                if graph.edges.edges:
                    edge = random.choice(list(graph.edges.edges.values()))
                    graph.update_edge_interaction(edge.from_node, edge.to_node)
                    edge.last_interaction = start_date
                    
                    # Update contextual similarity and sequential relation
                    edge.contextual_similarity = min(1.0, edge.contextual_similarity + random.uniform(0, 0.1))
                    edge.sequential_relation = min(1.0, edge.sequential_relation + random.uniform(0, 0.1))
            
        
        # Update engagement scores for all nodes
        for node in graph.node_map.nodes.values():
            node.update_engagement_score()

    def simulate_usage_recursive(self, graph: Graph, days=30, max_daily_interactions=10):
        self.simulate_usage(graph=graph, days=days, max_daily_interactions=max_daily_interactions)
        for node in graph.node_map.nodes.values():
            if node.subgraph:
                self.simulate_usage_recursive(graph=node.subgraph, days=days, max_daily_interactions=max_daily_interactions)