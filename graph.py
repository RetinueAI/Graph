from datetime import datetime, timezone, timedelta
import os
import json
import time
import bson
from bson import encode, ObjectId
from bson.json_util import dumps, loads
from typing import Dict, Optional, List, Tuple, Any, Union, Generator, AsyncGenerator, Callable
import uuid
import random
import asyncio
import bson.json_util
import psutil
import struct
import logging

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
    


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Node:
    def __init__(
        self, 
        id: int, 
        name: str, 
        parent: str, 
        data: bytearray = None,
        subgraph: Optional[Union['Graph',str]] = None,
    ):
        self.id = id
        self.name = name
        self.parent = parent
        self.data = data if data else bytearray(16)
        self.subgraph = subgraph


    async def update_engagement(self) -> None:
        await self.set_interest_frequency()
        await self.set_last_engagement()
        await self.update_engagement_score()


    async def update_engagement_score(self):
        recency = (datetime.now(timezone.utc) - await self.get_last_engagement()).total_seconds()
        value = (await self.get_interest_frequency() * await self.get_eigenvector_centrality()) / (recency + 1)
        await self.set_engagement_score(value=value)


    async def set_interest_frequency(self, value: int = None) -> None:
        if not value:
            old_value = await self.get_interest_frequency()
            value = 1 + old_value

        value = min(4294967295, 1 + value)

        struct.pack_into('I', self.data, 0, value)


    async def get_interest_frequency(self) -> int:
        return struct.unpack_from('I', self.data, 0)[0]


    async def set_eigenvector_centrality(self, value: float) -> None:
        struct.pack_into('I', self.data, 4, int(value * 4294967295))


    async def get_eigenvector_centrality(self) -> float:
        return struct.unpack_from('I', self.data, 4)[0] / 4294967295.0


    async def set_engagement_score(self, value: float) -> None:
        struct.pack_into('I', self.data, 8, int(value * 4294967295))


    async def get_engagement_score(self) -> float:
        return struct.unpack_from('I', self.data, 8)[0] / 4294967295.0


    async def set_last_engagement(self) -> None:
        value = int(datetime.now(timezone.utc).timestamp())
        struct.pack_into('I', self.data, 12, value)


    async def get_last_engagement(self) -> datetime:
        timestamp = struct.unpack_from('I', self.data, 12)[0]
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)


    async def set_subgraph(self, subgraph: 'Graph') -> None:
        self.subgraph = subgraph


    async def to_feature_vector(self):
        return [
            await self.get_interest_frequency(),
            await self.get_eigenvector_centrality(),
            await self.get_engagement_score(),
            (datetime.now(timezone.utc) - await self.get_last_engagement()).total_seconds()
        ]


    async def to_dict(self) -> Dict:
        return {
            'i': self.id,
            'n': self.name,
            'p': self.parent,
            'd': bson.Binary(bytes(self.data)),
            's': self.subgraph.id if self.subgraph else None
        }
    

    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        node = cls(
            id=data['i'], 
            name=data['n'], parent=data['p'],
            data=bytearray.fromhex(data['d'].hex())
        )
        if data['s']:
            node.subgraph = data['s']
        return node


class Nodes:
    def __init__(
        self,
        nodes: Dict[int,Node] = None,
        node_map: Dict[str,int] = None,
    ):
        self.nodes = nodes if nodes else {}
        self.node_map = node_map if node_map else {}


    async def add_node(self, id: int, name: str, parent: str)-> None:
        logger.debug(f"Creating node {id}: {name} with parent {parent}")
        node = Node(id=id, name=name, parent=parent)
        self.nodes[id] = node
        self.node_map[name] = id
        logger.debug(f"Node {id}: {name} added. Total nodes: {len(self.nodes)}")


    async def remove_node(self, node_id: int) -> None:
        pass

    async def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    async def get_node_id(self, node_name: str) -> int:
        return self.node_map[node_name]


class Edge:
    def __init__(
            self,
            from_node: int,
            to_node: int,
            data: bytearray = None,        
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.data = data if data else bytearray(10)


    async def update_interaction(self):
        strength = min(65535, await self.get_interaction_strength() + 1)
        await self.set_interaction_strength(strength)
        await self.set_last_interaction(int(datetime.now(timezone.utc).timestamp()))

    async def get_interaction_strength(self) -> int:
        return struct.unpack_from('H', self.data, 0)[0]

    async def set_interaction_strength(self, value: int = None):
        if not value:
            value = 1 + await self.get_interaction_strength()
        struct.pack_into('H', self.data, 0, value)

    async def get_last_interaction(self) -> int:
        return struct.unpack_from('I', self.data, 2)[0]

    async def set_last_interaction(self, value: int):
        struct.pack_into('I', self.data, 2, value)

    async def get_contextual_similarity(self) -> float:
        return struct.unpack_from('H', self.data, 6)[0] / 65535.0

    async def set_contextual_similarity(self, value: float):
        struct.pack_into('H', self.data, 6, int(value * 65535))

    async def get_sequential_relation(self) -> float:
        return struct.unpack_from('H', self.data, 8)[0] / 65535.0

    async def set_sequential_relation(self, value: float):
        struct.pack_into('H', self.data, 8, int(value * 65535))

    async def to_dict(self) -> Dict:
        return {
            "f": self.from_node,
            "t": self.to_node,
            "d": bson.Binary(bytes(self.data))
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        edge = cls(
            from_node=data['f'], 
            to_node=data['t'], 
            data=bytearray.fromhex(data['d'].hex())
        )
        return edge


class EdgeMap:
    def __init__(
            self, 
            path_to_id: dict = None, 
            id_to_path: dict = None,
            counter: int = None
    ):
        self.path_to_id = path_to_id if path_to_id else {}
        self.id_to_path = id_to_path if id_to_path else {}
        self.counter = counter if counter else 0

    async def get_id(self, path: Tuple[int, ...]) -> int:
        if path not in self.path_to_id:
            self.counter += 1
            self.path_to_id[path] = self.counter
            self.id_to_path[self.counter] = path
        return self.path_to_id[path]

    async def get_path(self, id: int) -> Tuple[int, ...]:
        return self.id_to_path[id]


class Edges:
    def __init__(
            self,
            edges: Dict[Tuple[int, int], Edge] = None,
            edge_map: EdgeMap = None,
    ):
        self.edges = edges if edges else {}
        self.edge_map = edge_map if edge_map else EdgeMap()


    async def get_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> Edge:
        from_id = await self.edge_map.get_id(from_)
        to_id = await self.edge_map.get_id(to_)
        return self.edges[(from_id, to_id)]


    async def add_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> None:
        from_id = await self.edge_map.get_id(from_)
        to_id = await self.edge_map.get_id(to_)
        if (from_id, to_id) in self.edges:
            raise ValueError("Edge already exists")
        edge = Edge(from_node=from_id, to_node=to_id)
        self.edges[(from_id, to_id)] = edge


    async def remove_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> None:
        pass


    async def generate_edge_key(self, from_: List[str], to_: List[str], root_graph: 'Graph') -> Tuple[int, ...]:
        from_path = await self.convert_str_list(from_, root_graph)
        to_path = await self.convert_str_list(to_, root_graph)
        return (await self.edge_map.get_id(from_path), await self.edge_map.get_id(to_path))


    async def add_edge_from_str(self, from_: List[str], to_: List[str], root_graph: 'Graph') -> None:
        edge_key = await self.generate_edge_key(from_, to_, root_graph)
        
        if edge_key in self.edges:
            raise ValueError("Edge already exists")

        edge = Edge(from_node=edge_key[0], to_node=edge_key[1])
        self.edges[edge_key] = edge


    async def convert_str_list(self, str_list: List[str], graph: 'Graph') -> Tuple[int]:
        if len(str_list) < 1:
            return []
        int_list = []

        int_list.append(graph.nodes.node_map[str_list[0]])

        if int_list[0] == None:
            raise ValueError(f"Node {str_list[0]} doesn't exist in graph {graph.id}.")

        if len(str_list) > 1:
            node = await graph.get_node(int_list[0])
            subgraph = node.subgraph

            if subgraph is None:
                raise ValueError(f"The subgraph in node {int_list[0]} doesn't exist in graph {graph.id}")
            
            int_list.extend(
                await self.convert_str_list(str_list[1:], subgraph)
            )

        return tuple(int_list)


class Graph:
    def __init__(
            self,
            user_id: str,
            version: int = 1,
            root: bool = False,
            id: str = None,
            nodes: Nodes = None,
            edges: Optional[Edges] = None,
            parent: Optional['Graph'] = None,
            depth: int = None
    ):
        self.user_id = user_id
        self.version = version
        self.root = root
        self.id = id if id else str(uuid.uuid4())
        self.edges = edges if edges else Edges()
        self.parent = parent
        self.depth = depth if depth else 0
        self.nodes = nodes if nodes else Nodes()


    async def add_node(self, id: int, name: str) -> None:
        logger.debug(f"Adding node {id}: {name} to graph {self.id}")
        await self.nodes.add_node(id=id, name=name, parent=self.id)
        logger.debug(f"Graph {self.id} now has {len(self.nodes.nodes)} nodes")
        

    async def add_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> None:
        await self.edges.add_edge(from_=from_, to_=to_)


    async def get_node(self, node_id: int) -> Node:
        return await self.nodes.get_node(node_id=node_id)
    

    async def get_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> Edge:
        return await self.edges.get_edge(from_, to_)


    async def remove_edge(self, from_: Tuple[int, ...], to_: Tuple[int, ...]) -> None:
        await self.edges.remove_edge(from_=from_, to_=to_)


    async def remove_node(self, node_id: int):
        await self.nodes.remove_node(node_id=node_id)
        self.edges = {k: v for k, v in self.edges.edges.items() if node_id not in k}


    async def update_node_engagement(self, node_id: int):
        if node_id in self.nodes.nodes:
            await self.nodes.nodes[node_id].update_engagement()


    async def update_edge_interaction(self, _from: Tuple[int, ...], _to: Tuple[int, ...]):
        edge_key = (_from, _to)
        if edge_key in self.edges.edges:
            await self.edges.edges[edge_key].update_interaction()


    async def set_node_subgraph(self, node_id: int, subgraph: 'Graph'):
        logger.debug(f"Setting subgraph {subgraph.id} for node {node_id} in graph {self.id}")
        node = await self.get_node(node_id=node_id)
        subgraph.parent = self
        # subgraph.depth = self.depth + 1
        node.subgraph = subgraph
        logger.debug(f"Subgraph set for node {node_id} in graph {self.id}")


    async def get_highest_interest_node(self):
        pass


    async def get_highest_interest_node_chain(self) -> Optional[List[str]]:

        async def get_node(nodes: Nodes) -> List[str]:
            categories = []

            score = -1
            best_node = None

            for node in nodes.nodes.values():
                interes_frequency = await node.get_interest_frequency()
                if interes_frequency > score:
                    score = interes_frequency
                    best_node = node
            
            categories.append(best_node.name)
            logger.debug(f"{best_node.name} has {await best_node.get_interest_frequency()} interactions")

            if best_node and best_node.subgraph:
                categories.extend(await get_node(nodes=best_node.subgraph.nodes))

            return categories

        return await get_node(nodes=self.nodes)


    async def get_highest_interest_node_at_depth(self, depth: int = 0) -> Optional[str]:
        """
        Returns the name of the node at the specified depth with the highest interest frequency.
        If no nodes exist at that depth, return None.
        """
        if depth < 0:
            return None

        # Traverse nodes at the specified depth
        highest_frequency = -1
        highest_node_name = None

        async def traverse(node: Node, current_depth: int):
            nonlocal highest_frequency, highest_node_name

            if current_depth == depth:
                interest_frequency = await node.get_interest_frequency()
                if interest_frequency > highest_frequency:
                    highest_frequency = interest_frequency
                    highest_node_name = node.name
            elif current_depth < depth and node.subgraph:
                for sub_node in node.subgraph.nodes.nodes.values():
                    await traverse(sub_node, current_depth + 1)

        # Start traversal from the root nodes
        for node in self.nodes.nodes.values():
            await traverse(node, 0)

        return highest_node_name
    

    async def get_best_edge(self) -> Edge:
        the_best = -1
        the_edgest = None

        for edge in self.edges.edges.values():
            interaction_strength = await edge.get_interaction_strength()
            if  interaction_strength > the_best:
                the_best =  interaction_strength
                the_edgest = edge

        return the_edgest


class GraphChanges:
    def __init__(
        self,
        change_cache: List[Dict[str,Any]] = None,
    ):
        self.change_cache = change_cache if change_cache else {}
        self.change_types = self._init_change_types()


    def _init_change_types(self) -> Dict[str,Dict]:
        return {
            'node': ['add', 'update', 'remove', 'add_subgraph', 'update_subgraph', 'remove_subgraph'],
            'edge': ['add', 'update', 'remove'],
            'graph_map': ['update'],
        }


    async def node_change(self, change_type: str, node_id: int, node_name: str, parent_id: str, changes: Dict[str,Any]):
        if change_type not in self.change_types['node']:
            logging.error("The node change type doesn't exist...")
            return

        self.change_cache.append(
            {
                'type': change_type,
                'node_id': node_id,
                'node_name': node_name,
                'parent_id': parent_id,
                'changes': changes
            }
        )


    async def edge_change(self, change_type: str, from_node: Tuple[int, ...], to_node: Tuple[int, ...], changes: Dict[str,Any]):
        if change_type not in self.change_types['edge']:
            print("The edge change type doesn't exist...")
            return
        
        self.change_cache.append(
            {
                'type': change_type,
                'from_node': from_node,
                'to_node': to_node,
                'changes': changes,
            }
        )

    
    async def graph_map_change(self, change_type: str, version: int, changes: Dict[str,Any]):
        if change_type not in self.change_types['graph_map']:
            print("The map change type doesn't exist...")
            return
        
        self.change_cache.append(
            {
                'type': change_type,
                'version': version,
                'changes': changes,
            }
        )


class GraphSync:
    def __init__(
            self,
            user_id: str,
            mongo_handler: MongoHandler,
            graph_changes: GraphChanges = None,
            graph: Graph = None,
            graph_map: Dict = None,
            sync_timestamp: datetime = None,
            max_batch_size: int = 48*1024*1024,
    ):
        self.user_id = user_id
        self.mongo_handler = mongo_handler
        self.graph_changes = graph_changes if graph_changes else GraphChanges()
        self.graph = graph
        self.graph_map = graph_map if graph_map else {}
        self.sync_timestamp = sync_timestamp
        self.max_batch_size = max_batch_size
        self.local_storage_paths = self._init_local_storage()
        self.change_cache = []


    def _init_local_storage(self) -> Dict[str,str]:
        graph_dir = os.path.join(os.getcwd(), 'graph')

        self.local_storage_paths = {
            'edges': os.path.join(graph_dir, 'edges'),
            'nodes': os.path.join(graph_dir, 'nodes'),
            'map': os.path.join(graph_dir, 'map'),
            'changelogs': os.path.join(graph_dir, 'changelogs'),
        }

        if 'graph' not in os.listdir(os.getcwd()):
            os.makedirs(graph_dir, exist_ok=True)

        if 'edges' not in os.listdir(graph_dir):
            os.makedirs(self.local_storage_paths['edges'], exist_ok=True)

        if 'nodes' not in os.listdir(graph_dir):
            os.makedirs(self.local_storage_paths['nodes'], exist_ok=True)

        if 'map' not in os.listdir(graph_dir):
            os.makedirs(self.local_storage_paths['map'], exist_ok=True)

        if 'changelogs' not in os.listdir(graph_dir):
            os.makedirs(self.local_storage_paths['changelogs'], exist_ok=True)


    async def sync(self):
        # TODO Either update from database if it has the most current, or upload to database if local is most current
        # TODO update sync timestamp with: datetime.now(timezone.utc)
        pass


    async def _cache_change(self, change: Dict[str,Any]) -> None:
        self.change_cache.append(change)


    async def _save_changelog(self, changes: List[Dict]):
        timestamp = datetime.now(timezone.utc)

        changelog = {
            'timestamp': timestamp,
            'user_id': self.user_id,
            'changes': changes
        }

        with open(os.path.join(self.local_storage_paths['changelogs'], f"{timestamp}.json"), 'w') as f:
            json.dump(changelog, f)


    async def save_node_locally(self, node: Node):
        nodes_storage_path = os.path.join(self.local_storage_paths, 'nodes')
        node_storage_path = os.path.join(nodes_storage_path, node.parent)
        node_file_path = os.path.join(node_storage_path, f'{node.id}.bson')

        if node.parent not in os.listdir(nodes_storage_path):
            os.makedirs(node_storage_path, exist_ok=True)

        bson_data = bson.BSON.encode(node.to_dict())

        with open(node_file_path, 'wb') as f:
            f.write(bson_data)


    async def save_edge_locally(self, edge: Edge):
        edges_storage_path = os.path.join(self.local_storage_paths, 'edges')
        from_edge_path = '_'.join([str(id) for id in self.graph.edges.edge_map.get_path(edge.from_node)])
        edge_storage_path = os.path.join(edges_storage_path, from_edge_path)
        to_edge_path = '_'.join([str(id) for id in self.graph.edges.edge_map.get_path(edge.to_node)])
        edge_file_path = os.path.join(edge_storage_path, f"{to_edge_path}.bson")

        if from_edge_path not in os.listdir(edges_storage_path):
            os.makedirs(edge_storage_path, exist_ok=True)

        bson_data = bson.BSON.encode(edge.to_dict())

        with open(edge_file_path, 'wb') as f:
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
                    await self.save_node_locally(node)
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
        try:
            for edge in edges.edges.values():
                await self.save_edge_locally(edge=edge)
        except Exception as e:
            print(e)
            return False

        return True


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

    async def _save_graph_map_to_database(self) -> Dict:
        return self.mongo_handler.insert(
            entry=self.graph_map,
            db_name='Graph',
            collection_name='Maps',
        )


    async def save_to_database(self) -> Tuple[List[InsertManyResult], List[InsertManyResult], InsertOneResult]:
        print("Saving to database...")

        nodes_result = await self._save_nodes_to_database(nodes=self.graph.nodes)
        edges_result = await self._save_edges_to_database(edges=self.graph.edges)
        graph_map_result = await self._save_graph_map_to_database()

        return (nodes_result, edges_result, graph_map_result)


    async def _load_node_from_database(self, node_id: int, node_name: str) -> Node:
        node_document = await self.mongo_handler.get_document(
            db_name='Graph',
            collection_name='Nodes',
            filter={'i': node_id, 'n': node_name}
        )

        if node_document:
            return Node.from_dict(node_document)
        return None
        

    async def _load_edge_from_database(self, from_node: int, to_node: int) -> Edge:
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


    async def _load_node_from_local(self, graph_id: str, node_id: int) -> Node:
        node_storage_path = os.path.join(self.local_storage_path, 'nodes', graph_id)
        filename = f"{node_id}.bson"

        if filename in os.listdir(node_storage_path):
            with open(os.path.join(node_storage_path, filename), 'rb') as f:
                bson_data = f.read()
        else:
            print("Didn't find the node in localstorage...")
            return None

        return Node.from_dict(bson.BSON(bson_data).decode())
    

    async def _load_edge_from_local(self, from_node: int, to_node: int) -> Edge:
        node_storage_path = os.path.join(self.local_storage_path, 'edges', f'{from_node}')
        filename = f"{to_node}.bson"

        if filename in os.listdir(node_storage_path):
            with open(os.path.join(node_storage_path, filename), 'rb') as f:
                bson_data = f.read()
        else:
            print("Didn't fine the edge in localstorage...")
            return None

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


    async def _load_edges_from_local(self) -> Edges:
        edges = Edges()
        edges_storage_path = os.path.join(self.local_storage_path, 'edges')
        i = 0

        for from_node in os.listdir(edges_storage_path):
            edge_storage_path = os.path.join(edges_storage_path, from_node)

            for edge_file in os.listdir(edge_storage_path):
                with open(os.path.join(edge_storage_path, edge_file), 'rb') as f:
                    bson_data = f.read()

                edge_dict = bson.BSON(bson_data).decode()
                edge = Edge.from_dict(edge_dict)
                edges.edges[(edge_dict['f'], edge_dict['t'])] = edge

                from_tuple = from_node.split('_')
                from_tuple = tuple([int(id) for id in from_tuple])

                to_tuple = edge_file.split('.')[0]
                to_tuple = to_tuple.split('_')
                to_tuple = tuple([int(id) for id in to_tuple])

                edges.edge_map.id_to_path[edge_dict['f']] = from_tuple
                edges.edge_map.path_to_id[from_tuple] = edge_dict['f']

                edges.edge_map.id_to_path[edge_dict['t']] = to_tuple
                edges.edge_map.path_to_id[to_tuple] = edge_dict['t']
                i+=1
                
        return edges


    async def _load_edges_from_database(self) -> Edges:
        edges = Edges()
        batch_size = 1000
        
        total_edges = await self.mongo_handler.count_documents(db_name='Graph', collection_name='Edges')

        with tqdm(total=total_edges, desc="Loading edges") as pbar:
            async for edge_batch in self._load_edges_in_batches(batch_size=batch_size):
                for edge_doc in edge_batch:
                    edge = Edge.from_dict(edge_doc['edge'])
                    edges.edges[(edge_doc['f'], edge_doc['t'])] = edge
                
                pbar.update(len(edge_batch))
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


    async def load_graph_from_local(self):
        self.graph = await self._rebuild_graph(graph_id=self.graph_map['graph']['id'], root=True)
        self.graph.edges = await self._load_edges_from_local()


    async def load_graph_from_database(self):
        self.graph_map = await self._load_graph_map_from_database()
        self.graph = await self._rebuild_graph(graph_id=self.graph_map['graph']['id'], root=True, local=False)
        self.graph.edges = await self._load_edges_from_database()


    async def _add_subgraphs(self, graph: Graph, local: bool = True) -> None:
        for node in graph.nodes.nodes.values():
            if node.subgraph:
                node.subgraph = await self._rebuild_graph(graph_id=node.subgraph, local=local)


    async def _fetch_graph_nodes_from_local(self, graph_id: str) -> List[Dict]:
        nodes_storage_path = os.path.join(self.local_storage_path, 'nodes')
        node_storage_path = os.path.join(nodes_storage_path, graph_id)
        graph_nodes = []

        if graph_id in os.listdir(nodes_storage_path):
            for node in os.listdir(node_storage_path):

                
                with open(os.path.join(node_storage_path, node), 'rb') as f:
                    bson_data = f.read()

                graph_node = bson.BSON(bson_data).decode()
                graph_nodes.append(graph_node)
        else:
            print(f"Nodes for graph id {graph_id} doesn't exist...")

        return graph_nodes


    async def _fetch_graph_nodes_from_database(self, graph_id: str) -> List[Dict]:
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


    async def _rebuild_graph(self, graph_id: str, root: bool = False, local: bool = True) -> Graph:

        graph = Graph(
            id=graph_id,
            user_id=self.user_id, 
            root=root,
        )

        if local:
            graph_nodes = await self._fetch_graph_nodes_from_local(graph_id=graph_id)
        else:
            graph_nodes = await self._fetch_graph_nodes_from_database(graph_id=graph_id)

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

        await self._add_subgraphs(graph=graph, local=local)

        return graph