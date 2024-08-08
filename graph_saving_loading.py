import os
from time import time
import json
import asyncio
from typing import Dict, Tuple

from graph import Graph, GraphSync, Edges
from graph_generation import generate_graph, load_graph_map, generate_new_graph
from graph_inspection import check_graph_mapping, check_id, compare_graphs, compare_edges
from mongo import MongoHandler



async def main():

    # graph = generate_new_graph()
    graph_map = load_graph_map()

    before = time()
    graph = Graph(user_id=graph_map['user_id'], id=graph_map['graph']['id'], edges=Edges())
    generate_graph(graph=graph, graph_map=graph_map['graph'])
    after = time()
    print(f"Graph generation took {after-before}s")

    uri = "mongodb+srv://mimir.kjfum9z.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&appName=Mimir"
    mongo_handler = MongoHandler(
        uri=uri,
        cert_path='./mongocert.pem',
    )

    if await mongo_handler.test_database_connection():
        print("Database connection established...")
    else:
        return

    # nodes_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Nodes')
    # print(f"Number of documents cleaned from the Nodes collection: {nodes_cleanup}")
    # edges_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Edges')
    # print(f"Number of documents cleaned from the Edges collection: {edges_cleanup}")

    graph_sync = GraphSync(
        graph=graph,
        user_id=graph_map['user_id'],
        graph_map=graph_map,
        mongo_handler=mongo_handler,
    )
    print("GraphSync initiated...")

    # saved_result = await graph_sync.save_to_database()
    # before = time()
    # nodes_saving = await graph_sync.save_nodes_to_local()
    # edges_saving = await graph_sync._save_edges_to_local(edges=graph_sync.graph.edges)
    # after = time()

    # if not (nodes_saving and edges_saving):
    #     return
    # print("Saving completed...")
    # print(f"Sacing took {after-before}s")

    # print("Loading the graph from local storage...")
    # before = time()
    # await graph_sync.load_graph_from_local()
    # after = time()
    # print("Graph loaded...")
    # print(f"The loading took {after-before}s")

    # print(f"Number of edges in the Graph reconstructed from the graph map: {len(graph.edges.edges)}")
    # print(f"Number of edges in the Graph inside GraphSync: {len(graph_sync.graph.edges.edges)}")


    # print("Loading the graph from the database...")
    # await graph_sync.load_graph_from_database()
    # print("Graph loaded...")

    # comparison = compare_graphs(x=graph, y=graph_sync.graph, edges=True)
    # print(f"The graphs are identical: {comparison}")

    # nodes_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Nodes')
    # print(f"Number of documents cleaned from the Nodes collection: {nodes_cleanup}")
    # edges_cleanup = await mongo_handler.cleanup(db_name='Graph', collection_name='Edges')
    # print(f"Number of documents cleaned from the Edges collection: {edges_cleanup}")



if __name__ == '__main__':
    asyncio.run(main())