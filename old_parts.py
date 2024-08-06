from typing import List, Dict, Any, Generator
from bson import encode
import asyncio
import gc

from pymongo.results import InsertOneResult, InsertManyResult

from graph import Graph, EdgeMap



class GraphSync:
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
            node = graph.nodes.nodes[change['node_id']]
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
            graph.nodes.nodes.pop(change['node_id'], None)
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

            for node in graph.nodes.nodes.values():
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