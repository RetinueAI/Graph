import os
from time import time
import json
import asyncio
from typing import Dict

from generate_graph import generate_graph
from geo_graph import Graph, GraphSync
from mongo import MongoHandler



def map_graph(graph: Graph, graph_map: Dict) -> None:

    for node in graph.node_map.nodes.values():
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
        if graph.node_map.nodes[int(node)].subgraph:
            count += check_id(graph.node_map.nodes[int(node)].subgraph.id, graph_map[node])

            count += check_mapping(graph=graph.node_map.nodes[int(node)].subgraph, graph_map=graph_map[node][list(graph_map[node].keys())[0]])

    return count


def compare_graphs(x: Graph, y: Graph) -> bool:
    state = True

    x_len = len(x.node_map.nodes)
    y_len = len(y.node_map.nodes)

    if x_len == y_len:
        for i in range(x_len):
            if x.node_map.nodes[i].name == y.node_map.nodes[i].name:
                if (x.node_map.nodes[i].subgraph != None) and (y.node_map.nodes[i].subgraph != None):
                    if not compare_graphs(x.node_map.nodes[i].subgraph, y.node_map.nodes[i].subgraph):
                        return False
            else:
                return False
    else:
        state = False
    
    return state


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

    # uri = "mongodb+srv://mimir.kjfum9z.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&appName=Mimir"
    # mongo_handler = MongoHandler(
    #     uri=uri,
    #     cert_path='./mongocert.pem',
    # )

    # if await mongo_handler.test_database_connection():
    #     print("Database connection established!")
    # else:
    #     return

    # graphs_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Graphs')
    # edges_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Edges')
    # maps_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Maps')

    # print(f"Number of documents cleaned from the Graphs collection: {graphs_cleanup}")
    # print(f"Number of documents cleaned from the Edges collection: {edges_cleanup}")
    # print(f"Number of documents cleaned from the Maps collection: {maps_cleanup}")

    # graph_sync = GraphSync(
    #     graph=graph,
    #     user_id=user_id,
    #     graph_map=graph_map,
    #     mongo_handler=mongo_handler,
    #     local_storage_path=os.path.join(os.getcwd(), 'graph_sync_timestamp.json')
    # )
    # print("GraphSync initiated...")

    # saved_result = await graph_sync.save_to_database()
    # print(f"Results from saving: {saved_result}")

    # await graph_sync.load_from_database()

    # print(type(graph))
    # print(type(graph_sync.graph))

    # compare = compare_graphs(x=graph, y=graph_sync.graph)
    # print(f"The graphs are the same: {compare}")

    # graphs_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Graphs')
    # edges_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Edges')
    # maps_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Maps')

    # print(f"Number of documents cleaned from the Graphs collection: {graphs_cleanup}")
    # print(f"Number of documents cleaned from the Edges collection: {edges_cleanup}")
    # print(f"Number of documents cleaned from the Maps collection: {maps_cleanup}")


if __name__ == '__main__':
    asyncio.run(main())