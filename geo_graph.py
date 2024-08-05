from datetime import datetime, timezone, timedelta
import os
import json
import gc
import time
from bson import encode, ObjectId
from bson.json_util import dumps, loads
from typing import Dict, Optional, List, Tuple, Any, Union, Generator, AsyncGenerator
import uuid
import random
import asyncio
import psutil
import struct

from tqdm import tqdm
from dateutil import parser
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


class NodeMap(BaseModel):
    nodes: Dict[int, Node] = Field(default_factory=dict)
    node_map: Dict[str, int] = Field(default_factory=dict)

    def get_node(self, node: int) -> Node:
        return self.nodes[node]

    def get_node_id(self, node: str):
        return self.node_map[node]
    
    class Config:
        arbitrary_types_allowed = True


class EdgeMap:
    def __init__(self):
        self.path_to_id = {}
        self.id_to_path = {}
        self.counter = 0

    def get_id(self, path: Tuple[int, ...]) -> int:
        if path not in self.path_to_id:
            self.counter += 1
            self.path_to_id[path] = self.counter
            self.id_to_path[self.counter] = path
        return self.path_to_id[path]

    def get_path(self, id: int) -> Tuple[int, ...]:
        return self.id_to_path[id]

class Edge(BaseModel):
    from_node: int
    to_node: int
    data: bytearray = Field(default=bytearray(10))

    class Config:
        arbitrary_types_allowed = True

    def update_interaction(self):
        strength = min(65535, self.get_interaction_strength() + 1)
        self.set_interaction_strength(strength)
        self.set_last_interaction(int(datetime.now(timezone.utc).timestamp()))

    def get_interaction_strength(self) -> int:
        return struct.unpack_from('H', self.data, 0)[0]

    def set_interaction_strength(self, value: int):
        struct.pack_into('H', self.data, 0, value)

    def get_last_interaction(self) -> int:
        return struct.unpack_from('I', self.data, 2)[0]

    def set_last_interaction(self, value: int):
        struct.pack_into('I', self.data, 2, value)

    def get_contextual_similarity(self) -> float:
        return struct.unpack_from('H', self.data, 6)[0] / 65535.0

    def set_contextual_similarity(self, value: float):
        struct.pack_into('H', self.data, 6, int(value * 65535))

    def get_sequential_relation(self) -> float:
        return struct.unpack_from('H', self.data, 8)[0] / 65535.0

    def set_sequential_relation(self, value: float):
        struct.pack_into('H', self.data, 8, int(value * 65535))

    def to_dict(self):
        return {
            "f": self.from_node,
            "t": self.to_node,
            "d": self.data.hex()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        edge = cls(data['f'], data['t'])
        edge.data = bytearray.fromhex(data['d'])
        return edge

class Edges(BaseModel):
    edges: Dict[Tuple[int, int], Edge] = Field(default_factory=dict)
    edge_map: EdgeMap = Field(default_factory=EdgeMap)

    class Config:
        arbitrary_types_allowed = True

    def get_edge(self, _from: Tuple[int, ...], _to: Tuple[int, ...]) -> Edge:
        from_id = self.edge_map.get_id(_from)
        to_id = self.edge_map.get_id(_to)
        return self.edges[(from_id, to_id)]

    def add_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> None:
        from_id = self.edge_map.get_id(from_)
        to_id = self.edge_map.get_id(to_)
        if (from_id, to_id) in self.edges:
            raise ValueError("Edge already exists")
        edge = Edge(from_node=from_id, to_node=to_id)
        self.edges[(from_id, to_id)] = edge

    def to_dict(self):
        return {
            "edges": {f"{k[0]},{k[1]}": v.to_dict() for k, v in self.edges.items()},
            "mapping": {str(k): v for k, v in self.edge_map.id_to_path.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        edges = cls()
        for key, value in data["edges"].items():
            from_id, to_id = map(int, key.split(','))
            edges.edges[(from_id, to_id)] = Edge.from_dict(value)
        edges.edge_map.id_to_path = {int(k): tuple(v) for k, v in data["mapping"].items()}
        edges.edge_map.path_to_id = {v: k for k, v in edges.edge_map.id_to_path.items()}
        edges.edge_map.counter = max(edges.edge_map.id_to_path.keys(), default=0)
        return edges

    def generate_edge_key(self, from_: List[str], to_: List[str], root_graph: 'Graph') -> Tuple[int, int]:
        from_path = self.convert_str_list(from_, root_graph)
        to_path = self.convert_str_list(to_, root_graph)
        return (self.edge_map.get_id(from_path), self.edge_map.get_id(to_path))

    def add_edge_from_str(self, from_: List[str], to_: List[str], root_graph: 'Graph') -> None:
        edge_key = self.generate_edge_key(from_, to_, root_graph)
        
        if edge_key in self.edges:
            raise ValueError("Edge already exists")

        edge = Edge(from_node=edge_key[0], to_node=edge_key[1])
        self.edges[edge_key] = edge

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
        if edge_key in self.edges.edges:
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
            

        except Exception as e:
            print(f"Error in from_dict: {e}")
            raise

        return graph


class GraphSync(BaseModel):
    graph: Graph
    user_id: str 
    graph_map: Dict = Field(default={})
    mongo_handler: MongoHandler
    local_storage_path: str
    last_sync_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True

    async def _apply_changes(self, changes: List[Dict]):
        for change in changes:
            await self._apply_change_recursive(self.graph, change)

    async def _apply_change_recursive(self, graph: Graph, change: Dict[str, Any]):
        if change['type'] == 'add_node':
            node = graph.add_node(change['node_id'], change['node_data']['name'])
            for key, value in change['node_data'].items():
                if key != 'name' and key != 'subgraph':
                    setattr(node, key, value)
            if 'subgraph' in change['node_data']:
                subgraph = Graph()
                node.set_subgraph(subgraph)
                for subchange in change['node_data']['subgraph']:
                    await self._apply_change_recursive(subgraph, subchange)

        elif change['type'] == 'update_node':
            node = graph.node_map.nodes[change['node_id']]
            for key, value in change['node_data'].items():
                if key != 'subgraph':
                    setattr(node, key, value)
            if 'subgraph' in change['node_data']:
                if node.subgraph is None:
                    node.set_subgraph(Graph())
                for subchange in change['node_data']['subgraph']:
                    await self._apply_change_recursive(node.subgraph, subchange)

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


    async def _save_graphs_to_database(self, main_graph: Graph) -> List[InsertManyResult]:
        graph_size = len(encode({"_id": main_graph.id, "graph": main_graph.to_dict()}))
        max_batch_size = 48 * 1024 * 1024  # 48 MB
        batch_size = max(1, max_batch_size // graph_size)

        batch_generator = self._collect_graphs(graph=main_graph, batch_size=batch_size)
        
        results = []

        for batch in batch_generator:
            result = await self.mongo_handler.insert_many(
                entries=batch,
                db_name='Graph',
                collection_name='Graphs'
            )
            results.append(result)
            # Explicitly call garbage collection to free up memory
            gc.collect()
            await asyncio.sleep(0)  # Yield control to the event loop

        return results


    def _collect_graphs(self, graph: Graph, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        def _collect(graph: Graph) -> Generator[Dict[str, Any], None, None]:
            yield {'_id': graph.id, 'graph': graph.to_dict()}

            for node in graph.node_map.nodes.values():
                if node.subgraph:
                    yield from _collect(node.subgraph)

        batch = []
        for g in _collect(graph):
            batch.append(g)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


    async def _save_edges_to_database(self, edges: Edges) -> List[InsertManyResult]:
        (from_node, to_node), edge = next(iter(self.graph.edges.edges.items()))
        edge_dict = edge.to_dict()
        graph_size = len(encode({"from_node": from_node, "to_node": to_node, "user_id": self.user_id, "edge": edge_dict}))
        max_batch_size = 48 * 1024 * 1024  # 48 MB
        batch_size = max(1, max_batch_size // graph_size)

        batch_generator = self._collect_edges(edges=edges, batch_size=batch_size)
        total_edges = len(edges.edges)

        results = []
        with tqdm(total=total_edges, desc="Saving edges") as pbar:
            for batch in batch_generator:
                result = await self.mongo_handler.insert_many(
                    entries=batch,
                    db_name='Graph',
                    collection_name='Edges'
                )
                results.append(result)
                pbar.update(len(batch))
                gc.collect()
                await asyncio.sleep(0)

        return results

    def _collect_edges(self, edges: Edges, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        batch = []

        for (from_node, to_node), edge in edges.edges.items():
            batch.append({'from_node': from_node, 'to_node': to_node, 'user_id': self.user_id, 'edge': edge.to_dict()})
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
    

    async def _save_map_to_database(self) -> InsertOneResult:
        exists = await self.mongo_handler.document_exists(
            db_name='Graph', 
            collection_name='Maps', 
            filter={'_id': self.user_id}
        )

        if exists:
            return await self.mongo_handler.update_document(
                entry={'_id': self.user_id, 'map': self.graph_map}, 
                query={'_id': self.user_id}, 
                db_name='Graph', 
                collection_name='Maps'
            )
        else:
            return await self.mongo_handler.insert(
                entry={'_id': self.user_id, 'map': self.graph_map}, 
                db_name='Graph', 
                collection_name='Maps'
            )
        

    async def _save_edge_mapper(self, edge_map: EdgeMap) -> InsertOneResult:
        exists = await self.mongo_handler.document_exists(
            db_name='Graph',
            collection_name='EdgeMap',
            filter={'_id': self.user_id}
        )

        map_entry = {
            '_id': self.user_id,
            'id_to_path': edge_map.id_to_path,
            'counter': edge_map.counter
        }

        if exists:
            return await self.mongo_handler.update_document(
                entry={'_id': self.user_id, 'map': map_entry},
                query={'_id': self.user_id},
                db_name='Graph',
                collection_name='EdgeMap'
            )
        else:
            return await self.mongo_handler.insert(
                entry={'_id': self.user_id, 'map': map_entry},
                db_name='Graph',
                collection_name='EdgeMap'
            )


    async def _load_graphs_from_database(self) -> bool:
        try:
            await self._load_graph_map_from_database()

            main_graph_id = self.graph.id
            main_graph_map = self.graph_map[main_graph_id]

            self.graph = await self._load_and_reconstruct_graph(main_graph_id, main_graph_map)
            return True
        except Exception as e:
            print(f"Error loading graphs: {e}")
            raise

    async def _load_and_reconstruct_graph(self, graph_id: str, graph_map: Dict) -> Graph:
        # Fetch the graph document
        graph_doc = await self.mongo_handler.get_document(
            db_name='Graph',
            collection_name='Graphs',
            filter={'_id': graph_id}
        )

        if not graph_doc:
            raise ValueError(f"Graph with id {graph_id} not found")

        # Create the graph object
        graph = Graph(user_id=self.user_id).from_dict(graph_doc['graph'])

        # Recursively load and set subgraphs
        for node, subgraph_info in graph_map.items():
            subgraph_id = list(subgraph_info.keys())[0]
            subgraph = await self._load_and_reconstruct_graph(subgraph_id, subgraph_info[subgraph_id])
            graph.set_node_subgraph(int(node), subgraph=subgraph)

        return graph
    

    async def _load_edges_from_database(self, user_id: str) -> Edges:
        edges = Edges()
        batch_size = 1000  # Adjust this value based on your system's memory constraints
        
        total_edges = await self.mongo_handler.count_documents(db_name='Graph', collection_name='Edges')

        with tqdm(total=total_edges, desc="Loading edges") as pbar:
            async for edge_batch in self._load_edges_in_batches(user_id, batch_size):
                for edge_doc in edge_batch:
                    from_node = edge_doc['from_node']
                    to_node = edge_doc['to_node']
                    edge = Edge.from_dict(edge_doc['edge'])
                    edges.edges[(from_node, to_node)] = edge
                
                pbar.update(len(edge_batch))
                gc.collect()
                await asyncio.sleep(0)
        
        return edges

    async def _load_edges_in_batches(self, user_id: str, batch_size: int) -> AsyncGenerator[List[Dict[str, Any]], None]:
        cursor = self.mongo_handler.get_documents(
            db_name='Graph',
            collection_name='Edges',
            query={'user_id': user_id},
            batch_size=batch_size
        )

        batch = []
        async for edge_doc in cursor:
            batch.append(edge_doc)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


    async def _load_graph_map_from_database(self) -> Optional[Dict]:
        result = await self.mongo_handler.get_document(
            db_name='Graph', 
            collection_name='Maps', 
            filter={'_id': self.user_id}
        )

        if not result['map']:
            raise ValueError(f"No graph map found for user {self.user_id}")
         
        self.graph_map = result['map']


    async def _load_edge_mapper(self, user_id: str) -> EdgeMap:
        result = await self.mongo_handler.get_document(
            db_name='Graph',
            collection_name='EdgeMap',
            filter={'_id': user_id}
        )
        
        if not result:
            raise ValueError(f"EdgeMap for user {user_id} not found")

        edge_map = EdgeMap()
        map_data = result['map']
        edge_map.id_to_path = map_data['id_to_path']
        edge_map.path_to_id = {tuple(v): k for k, v in edge_map.id_to_path.items()}
        edge_map.counter = map_data['counter']
        
        return edge_map


    async def save_to_database(self) -> Tuple[List[InsertManyResult],List[InsertManyResult], InsertOneResult]:
        graphs_result = await self._save_graphs_to_database(main_graph=self.graph)
        edges_result = await self._save_edges_to_database(edges=self.graph.edges)
        map_result = await self._save_map_to_database()

        return (graphs_result, edges_result, map_result)


    async def load_from_database(self):
        await self._load_graphs_from_database()
        self.graph.edges = await self._load_edges_from_database(user_id=self.user_id)
    


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