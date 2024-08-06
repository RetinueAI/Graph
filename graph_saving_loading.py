import os
from time import time
import json
import asyncio
from typing import Dict, Tuple

from graph_generation import generate_graph
from graph import Graph, GraphSync
from mongo import MongoHandler



def map_graph(graph: Graph, graph_map: Dict) -> None:

    for node in graph.nodes.nodes.values():
        if node.subgraph:
            graph_map[str(node.id)] = {node.subgraph.id: {}}
            map_graph(graph=node.subgraph, graph_map=graph_map[str(node.id)][node.subgraph.id])


def check_id(x: str, y: str) -> int:
    if x == y:
        return 0
    return 1


def check_mapping(graph: Graph, graph_map: Dict) -> int:

    count = 0

    for node in graph_map.keys():
        if graph.nodes.nodes[int(node)].subgraph:
            count += check_id(graph.nodes.nodes[int(node)].subgraph.id, graph_map[node])

            count += check_mapping(graph=graph.nodes.nodes[int(node)].subgraph, graph_map=graph_map[node][list(graph_map[node].keys())[0]])

    return count


def compare_graphs(x: Graph, y: Graph) -> bool:
    state = True

    x_len = len(x.nodes.nodes)
    y_len = len(y.nodes.nodes)

    if x_len == y_len:
        for i in range(x_len):
            if x.nodes.nodes[i].name == y.nodes.nodes[i].name:
                if (x.nodes.nodes[i].subgraph != None) and (y.nodes.nodes[i].subgraph != None):
                    if not compare_graphs(x.nodes.nodes[i].subgraph, y.nodes.nodes[i].subgraph):
                        return False
            else:
                return False
    else:
        state = False
    
    return state


def generate_node_map(graph: Graph, node_map: Dict[Tuple[int,str], Dict]):
    for node in graph.nodes.nodes.values():
        if node.subgraph:
            node_map[(node.id, graph.id)] = {}
            generate_node_map(node.subgraph, node_map=node_map[(node.id, graph.id)])


async def main():

    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    generate_graph(graph=graph, categories=categories)

    print("Graph generated")
    graph_map = {graph.id: {}}

    map_graph(graph=graph, graph_map=graph_map)

    if check_mapping(graph=graph, graph_map=graph_map[graph.id]) > 0:
        print("The graph map doesn't match the graph.")
        return
    else:
        print("The graph map matches the graph, continuing!")

    node_map = {}

    generate_node_map(graph=graph, node_map=node_map)

    uri = "mongodb+srv://mimir.kjfum9z.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&appName=Mimir"
    mongo_handler = MongoHandler(
        uri=uri,
        cert_path='./mongocert.pem',
    )

    if await mongo_handler.test_database_connection():
        print("Database connection established!")
    else:
        return

    # nodes_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Nodes')
    # print(f"Number of documents cleaned from the Nodes collection: {nodes_cleanup}")
    # edges_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Edges')
    # print(f"Number of documents cleaned from the Edges collection: {edges_cleanup}")

    graph_sync = GraphSync(
        graph=graph,
        user_id=user_id,
        graph_map=graph_map,
        node_map=node_map,
        mongo_handler=mongo_handler,
        local_storage_path=os.path.join(os.getcwd(), 'graph_sync_timestamp.json')
    )
    print("GraphSync initiated...")

    # saved_result = await graph_sync.save_to_database()
    # print(f"Results from saving: {saved_result}")

    await graph_sync.load_from_database()

    # nodes_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Nodes')
    # print(f"Number of documents cleaned from the Nodes collection: {nodes_cleanup}")
    # edges_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Edges')
    # print(f"Number of documents cleaned from the Edges collection: {edges_cleanup}")

    # print(type(graph))
    # print(type(graph_sync.graph))

    # compare = compare_graphs(x=graph, y=graph_sync.graph)
    # print(f"The graphs are the same: {compare}")



if __name__ == '__main__':
    asyncio.run(main())