from datetime import datetime, timezone, timedelta
import os
import json
import gc
import time
import bson
from bson import encode, ObjectId
from bson.json_util import dumps, loads
from typing import Dict, Optional, List, Tuple, Any, Union, Generator, AsyncGenerator
import uuid
import random
import asyncio
import bson.json_util
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
    parent: str
    data: bytearray = Field(default=bytearray(10))
    subgraph: Optional[Union['Graph',str]] = None

    class Config:
        arbitrary_types_allowed = True

    def update_engagement(self) -> None:
        self.set_interest_frequency()
        self.set_last_engagement()

    def update_engagement_score(self):
        recency = (datetime.now(timezone.utc) - self.get_last_engagement()).total_seconds()
        self.engagement_score = (self.get_interest_frequency() * self.get_eigenvector_centrality()) / (recency + 1)

    def set_interest_frequency(self) -> None:
        new_value = 1 + self.get_interest_frequency()
        struct.pack_into('H', self.data, 0, new_value)

    def get_interest_frequency(self) -> int:
        return struct.unpack_from('H', self.data, 0)[0]

    def set_eigenvector_centrality(self, value: float) -> None:
        struct.pack_into('H', self.data, 2, int(value * 65535))

    def get_eigenvector_centrality(self) -> float:
        return struct.unpack_from('H', self.data, 2)[0] / 65535.0
    
    def set_engagement_score(self, value: float) -> None:
        struct.pack_into('H', self.data, 4, int(value * 65535))

    def get_engagement_score(self) -> float:
        return struct.unpack_from('H', self.data, 4)[0] / 65535.0
    
    def set_last_engagement(self) -> None:
        value = int(datetime.now(timezone.utc).timestamp())
        struct.pack_into('I', self.data, 6, value)

    def get_last_engagement(self) -> datetime:
        return datetime.fromtimestamp(struct.unpack_from('I', self.data, 6)[0])
    
    def set_subgraph(self, subgraph: 'Graph') -> None:
        self.subgraph = subgraph

    def to_feature_vector(self):
        return [
            self.get_interest_frequency(),
            self.get_eigenvector_centrality(),
            self.get_engagement_score(),
            (datetime.now(timezone.utc) - self.get_last_engagement()).total_seconds()
        ]

    def to_dict(self) -> Dict:
        return {
            'i': self.id,
            'n': self.name,
            'p': self.parent,
            'd': bson.Binary(bytes(self.data)),
            's': self.subgraph.id if self.subgraph else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        node = cls(id=data['i'], name=data['n'], parent=data['p'])
        node.data = bytearray.fromhex(data['d'].hex())
        if data['s']:
            node.subgraph = data['s']
        return node


class Nodes(BaseModel):
    nodes: Dict[int, Node] = Field(default={})
    node_map: Dict[str, int] = Field(default={})

    def get_node(self, node: int) -> Node:
        return self.nodes[node]

    def get_node_id(self, node: str) -> int:
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

    def to_dict(self) -> Dict:
        return {
            "f": self.from_node,
            "t": self.to_node,
            "d": bson.Binary(bytes(self.data))
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        edge = cls(data['f'], data['t'])
        edge.data = bytearray.fromhex(data['d'].hex())
        return edge


class Edges(BaseModel):
    edges: Dict[Tuple[int, int], Edge] = Field(default={})
    edge_map: EdgeMap = Field(default=EdgeMap())

    class Config:
        arbitrary_types_allowed = True

    def get_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> Edge:
        from_id = self.edge_map.get_id(from_)
        to_id = self.edge_map.get_id(to_)
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

    def generate_edge_key(self, from_: List[str], to_: List[str], root_graph: 'Graph') -> Tuple[int, ...]:
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

        int_list.append(graph.nodes.node_map[str_list[0]])

        if int_list[0] == None:
            raise ValueError(f"Node {str_list[0]} doesn't exist in graph {graph.id}.")

        if len(str_list) > 1:
            subgraph = graph.nodes.nodes[int_list[0]].subgraph

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
    nodes: Nodes = Field(default_factory=Nodes)
    edges: Edges = Field(default_factory=Edges)
    parent: Optional['Graph'] = None
    depth: int = 0

    class Config:
        arbitrary_types_allowed = True

    def add_node(self, id: int, name: str) -> Node:
        node = Node(id=id, name=name, parent=self.id)
        self.nodes.nodes[id] = node
        self.nodes.node_map[name] = id
        return node

    def get_edge(self, _from: Tuple[int, ...], _to: Tuple[int, ...]) -> Edge:
        return self.edges.get_edge((_from, _to))

    def get_node(self, node_id: int) -> Node:
        return self.nodes.get_node(node_id)

    def remove_edge(self, _from: Tuple[int, ...], _to: Tuple[int, ...]):
        self.edges.edges.pop((_from, _to), None)

    def remove_node(self, node_id: int):
        self.nodes.nodes.pop(node_id, None)
        self.edges = {k: v for k, v in self.edges.edges.items() if node_id not in k}

    def update_node_engagement(self, node_id: int):
        if node_id in self.nodes.nodes:
            self.nodes.nodes[node_id].update_engagement()

    def update_edge_interaction(self, _from: Tuple[int, ...], _to: Tuple[int, ...]):
        edge_key = (_from, _to)
        if edge_key in self.edges.edges:
            self.edges.edges[edge_key].update_interaction()


    def set_node_subgraph(self, node_id: int, subgraph: 'Graph'):
        if node_id in self.nodes.nodes:

            subgraph.parent = self
            subgraph.depth = self.depth + 1
            self.nodes.nodes[node_id].set_subgraph(subgraph)


class GraphSync(BaseModel):
    user_id: str
    mongo_handler: MongoHandler
    local_storage_path: str = Field(default=os.path.join(os.getcwd(),'graph'))
    graph: Graph = Field(default=None)
    graph_map: Dict = Field(default={})
    last_sync_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    max_batch_size: int = Field(default=48*1024*1024)


    class Config:
        arbitrary_types_allowed = True


    async def generate_node_map(self):
        pass


    async def _save_node_locally(self, node: Node):
        path = os.path.join(self.local_storage_path, 'nodes', f'{node.parent}_{str(node.id)}.bson')
        bson_data = bson.BSON.encode(node.to_dict())

        with open(path, 'wb') as f:
            f.write(bson_data)


    async def _save_edge_locally(self, edge: Edge):
        path = os.path.join(self.local_storage_path, 'edges', f'{str(edge.from_node)}_{edge.to_node}.bson')
        bson_data = bson.BSON.encode(edge.to_dict())

        with open(path, 'wb') as f:
            f.write(bson_data)


    async def save_nodes_to_local(self) -> bool:
        n_nodes = await self._count_total_nodes(nodes=self.graph.nodes)
        return await self._save_nodes_to_local(nodes=self.graph.nodes, n_nodes=n_nodes)


    async def _save_nodes_to_local(self, nodes: Nodes, n_nodes: int, pbar = None) -> bool:

        if pbar is None:
            pbar = tqdm(total=n_nodes, desc="Saving nodes to local storage...")
        
        try:
            for node in nodes.nodes.values():
                try:
                    await self._save_node_locally(node)
                except Exception as e:
                    print(e)
                    return False
                
                if node.subgraph:
                    await self._save_nodes_to_local(node.subgraph.nodes, n_nodes=n_nodes)
                pbar.update()
        finally:
            if pbar.total == n_nodes:
                pbar.close()
        
        return True

    
    async def _save_nodes_to_database(self, nodes: Nodes) -> List[InsertManyResult]:
        node = next(iter(nodes.nodes.values()))
        document = node.to_dict()
        node_size = len(document)
        batch_size = max(1, self.max_batch_size // node_size)

        node_generator = self._generate_batched_nodes(nodes=nodes, batch_size=batch_size)
        total_nodes = await self._count_total_nodes(nodes)

        results = []
        with tqdm(total=total_nodes, desc="Saving nodes") as pbar:
            async for batch in node_generator:
                result = await self.mongo_handler.insert_many(
                    entries=batch,
                    db_name='Graph',
                    collection_name='Nodes'
                )
                results.append(result)
                pbar.update(len(batch))
                await asyncio.sleep(0)

        return results
    

    async def collect(self, node: Node) -> List[Dict]:
        nodes = [node.to_dict()]

        if node.subgraph:
            for sub_node in node.subgraph.nodes.nodes.values():
                nodes.extend(await self.collect(sub_node))

        return nodes


    async def _generate_batched_nodes(self, nodes: Nodes, batch_size: int) -> AsyncGenerator[List[Dict], None]:
        all_nodes = []
        batch = []          

        for node in nodes.nodes.values():
            print(node.parent)
            all_nodes.extend(await self.collect(node=node))

        for node in all_nodes:
            batch.append(node)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


    async def _count_total_nodes(self, nodes: Nodes) -> int:
        async def count_recursive(node_collection: Nodes):
            count = len(node_collection.nodes.values())
            for node in node_collection.nodes.values():
                if node.subgraph:
                    count += await count_recursive(node.subgraph.nodes)
            return count

        return await count_recursive(nodes)


    async def _save_edges_to_local(self, edges: Edges) -> bool:
        pass


    async def _save_edges_to_database(self, edges: Edges) -> List[InsertManyResult]:
        edge = next(iter(self.graph.edges.edges.values()))
        document = edge.to_dict()
        edge_size = len(document)
        batch_size = max(1, self.max_batch_size // edge_size)

        edge_generator = self._collect_edges(edges=edges, batch_size=batch_size)
        total_edges = len(edges.edges)

        results = []
        with tqdm(total=total_edges, desc="Saving edges") as pbar:
            for batch in edge_generator:
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

    
    def _collect_edges(self, edges: Edges, batch_size: int) -> Generator[List[Dict], None, None]:
        batch = []

        for edge in edges.edges.values():
            batch.append(edge.to_dict())
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


    async def save_to_database(self) -> Tuple[List[InsertManyResult], List[InsertManyResult]]:
        print("Saving to database.")

        print(f"self.graph.id = {self.graph.id}")
        node = next(iter(self.graph.nodes.nodes.values()))
        print(f"node parent in self.graph = {node.parent}")

        nodes_result = await self._save_nodes_to_database(nodes=self.graph.nodes)
        # edges_result = await self._save_edges_to_database(edges=self.graph.edges)

        return (nodes_result, None)
        # return (nodes_result, edges_result)


    async def _load_node_from_database(self, node_id: int, node_name: str) -> Node:
        node_document = await self.mongo_handler.get_document(
            db_name='Graph',
            collection_name='Nodes',
            filter={'i': node_id, 'n': node_name}
        )

        if node_document:
            return Node.from_dict(node_document)
        return None
        

    async def load_edge_from_database(self, from_node: int, to_node: int) -> Edge:
        edge_document = await self.mongo_handler.get_document(
            db_name='Graph',
            collection_name='Nodes',
            filter={
                'f': from_node,
                'n': to_node
            }
        )

        if edge_document:
            return Edge.from_dict(edge_document)
        return None


    async def _load_node_from_local(self, node_id: int, node_name: str) -> Node:
        node_storage_path = os.path.join(self.local_storage_path, 'nodes')
        filename = f"{str(node_id)}_{node_name}.bson"

        if filename in os.listdir(node_storage_path):
            with open(os.path.join(node_storage_path, filename), 'rb') as f:
                bson_data = f.read()

        return Node.from_dict(bson.BSON(bson_data).decode())
    

    async def _load_edge_from_local(self, from_node: int, to_node: int) -> Edge:
        node_storage_path = os.path.join(self.local_storage_path, 'nodes')
        filename = f"{str(from_node)}_{to_node}.bson"

        if filename in os.listdir(node_storage_path):
            with open(os.path.join(node_storage_path, filename), 'rb') as f:
                bson_data = f.read()

        return Edge.from_dict(bson.BSON(bson_data).decode())


    async def _load_nodes_from_database(self) -> Nodes:
        nodes = Nodes()
        batch_size = 1000

        total_nodes = await self.mongo_handler.count_documents(db_name='Graph', collection_name='Nodes')

        with tqdm(total=total_nodes, desc="Loading Nodes") as pbar:
            async for node_batch in self._load_nodes_in_batches(batch_size=batch_size):
                for node_doc in node_batch:
                    node = Node.from_dict(node_doc)
                    nodes.nodes[node.id] = node
                    pbar.update(len(node_batch))
    

    async def _load_graph_nodes_from_database(self, graph_id: str, batch_size: int) -> AsyncGenerator[List[Dict[str, Any]], None]:
        yield self.mongo_handler.get_documents(
            db_name='Graph',
            collection_name='Nodes',
            query={'p':graph_id},
            length=None
        )


    async def _load_edges_from_database(self) -> Edges:
        edges = Edges()
        batch_size = 1000
        
        total_edges = await self.mongo_handler.count_documents(db_name='Graph', collection_name='Edges')

        with tqdm(total=total_edges, desc="Loading edges") as pbar:
            async for edge_batch in self._load_edges_in_batches(batch_size=batch_size):
                for edge_doc in edge_batch:
                    from_node = edge_doc['from_node']
                    to_node = edge_doc['to_node']
                    edge = Edge.from_dict(edge_doc['edge'])
                    edges.edges[(from_node, to_node)] = edge
                
                pbar.update(len(edge_batch))
                gc.collect()
                await asyncio.sleep(0)
        
        return edges
    

    async def _load_nodes_in_batches(self, batch_size: int) -> AsyncGenerator[List[Dict[str, Any]], None]:
        return self.mongo_handler.get_documents(
            db_name='Graph',
            collection_name='Nodes',
            query={'user_id': self.user_id},
            length=batch_size
        )


    async def _load_edges_in_batches(self, batch_size: int) -> AsyncGenerator[List[Dict[str, Any]], None]:
        return await self.mongo_handler.get_documents(
            db_name='Graph',
            collection_name='Edges',
            query={'user_id': self.user_id},
            length=batch_size
        )


    async def _load_graph_map_from_database(self) -> Optional[Dict]:
        result = await self.mongo_handler.get_document(
            db_name='Graph', 
            collection_name='Maps', 
            filter={'_id': self.user_id}
        )

        if not result['map']:
            raise ValueError(f"No graph map found for user {self.user_id}")
         
        self.graph_map = result['map']


    async def load_from_database(self):
        print("Loading from database")

        nodes = await self._load_nodes_from_database()
        edges = await self._load_edges_from_database()

        print(f"Number of nodes loaded: {len(nodes.nodes)}")
        print(f"Number of edges loaded: {len(edges.edges)}")



    async def load_graph_from_database(self):
        self.graph = await self._rebuild_graph(graph_id=self.graph_map['graph']['id'], root=True)
        # self.graph.edges = await self._load_edges_from_database()


    async def _add_subgraphs(self, graph: Graph) -> None:
        for node in graph.nodes.nodes.values():
            if node.subgraph:
                node.subgraph = await self._rebuild_graph(graph_id=node.subgraph)


    async def fetch_graph_nodes(self, graph_id: str):
        batched_nodes = self.mongo_handler.get_documents(
            db_name='Graph',
            collection_name='Nodes',
            query={'p': graph_id},
            length=50
        )

        graph_nodes = []

        async for batch in batched_nodes:
            graph_nodes.extend(batch)

        return graph_nodes


    async def _rebuild_graph(self, graph_id: str, root: bool = False) -> Graph:

        graph = Graph(
            id=graph_id,
            user_id=self.user_id, 
            root=root,
        )

        graph_nodes = await self.fetch_graph_nodes(graph_id=graph_id)

        nodes_nodes = {}
        node_map = {}

        for graph_node in graph_nodes:
            node = Node.from_dict(graph_node)
            nodes_nodes[node.id] = node
            node_map[node.name] = node.id

        nodes = Nodes(
            nodes=nodes_nodes,
            node_map=node_map
        )

        graph.nodes = nodes

        await self._add_subgraphs(graph=graph)

        return graph