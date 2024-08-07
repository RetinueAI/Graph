import os
from time import time
import json
import asyncio
from typing import Dict, Tuple

from graph import Graph, GraphSync
from graph_generation import generate_graph, generate_node_map, generate_graph_map
from graph_inspection import check_graph_mapping, check_id, compare_graphs
from mongo import MongoHandler



async def main():

    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    generate_graph(graph=graph, categories=categories)

    print("Graph generated")
    graph_map = {graph.id: {}}

    generate_graph_map(graph=graph, graph_map=graph_map)

    if check_graph_mapping(graph=graph, graph_map=graph_map[graph.id]) > 0:
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