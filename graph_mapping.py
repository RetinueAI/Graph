import json
import os
import asyncio
import uuid
from typing import Dict, Tuple

from geo_graph import Graph, Node, Edge, GraphSync, GraphRandomizer, GraphPresenter
from mongo import MongoHandler



def generate_graph(graph: Graph, categories: Dict):

    for i, parent in enumerate(categories.keys()):
        graph.add_node(i, parent)

        if len(categories[parent]) > 0:
            graph.set_node_subgraph(i, subgraph=Graph(user_id=graph.user_id))
            generate_graph(graph=graph.nodes[i].subgraph, categories=categories[parent])

    for j in range(len(graph.nodes)):
        for k in range(len(graph.nodes)):
            if j != k:
                graph.add_edge(j, k)
                graph.add_edge(k, j)


def map_graph(graph: Graph, graph_map: Dict) -> None:

    for node in graph.nodes.values():
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
        if graph.nodes[int(node)].subgraph:
            count += check_id(graph.nodes[int(node)].subgraph.id, graph_map[node])

            count += check_mapping(graph=graph.nodes[int(node)].subgraph, graph_map=graph_map[node][list(graph_map[node].keys())[0]])

    return count


def compare_graphs(x: Graph, y: Graph) -> bool:
    state = True

    x_len = len(x.nodes)
    y_len = len(y.nodes)

    if x_len == y_len:
        for i in range(x_len):
            if x.nodes[i].name == y.nodes[i].name:
                if (x.nodes[i].subgraph != None) and (y.nodes[i].subgraph != None):
                    if not compare_graphs(x.nodes[i].subgraph, y.nodes[i].subgraph):
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

    print(f"Number of root nodes: {len(graph.nodes)}")

    graph_map = {graph.id: {}}

    map_graph(graph=graph, graph_map=graph_map[graph.id])

    uri = "mongodb+srv://mimir.kjfum9z.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&appName=Mimir"
    mongo_handler = MongoHandler(
        uri=uri,
        cert_path='./mongocert.pem',
    )

    graph_sync = GraphSync(
        graph=graph,
        graph_map=graph_map,
        mongo_handler=mongo_handler,
        local_storage_path=os.path.join(os.getcwd(), 'graph_sync_timestamp.json')
    )

    connection = await mongo_handler.test_database_connection()
    print(f"Connection: {connection}")

    randomizer = GraphRandomizer()

    randomizer.simulate_usage_recursive(graph=graph, days=180, max_daily_interactions=60)

    presenter = GraphPresenter()

    presenter.display_simulation_results(graph=graph)

    # graphs = graph_sync.collect_graphs(graph_sync.graph)

    # print(len(graphs))

    # map_insert = await graph_sync.save_map_to_database()
    # print(f'Map saved: {map_insert.acknowledged}')

    # graphs_insert = await graph_sync.save_graphs_to_database(graph_sync.graph)
    # print(f'Graphs saved: {graphs_insert.acknowledged}')

    # loaded_map = await graph_sync.load_graph_map_from_database()
    # print(loaded_map == graph_map)

    # try: 
    #     print('Loading graph')
    #     loaded_graph = await graph_sync.load_graphs_from_database(user_id=user_id)

    #     print(f'Loaded Graph: {loaded_graph}')

    #     print(f"Number of loaded root nodes: {len(graph_sync.graph.nodes)}")

    #     count = check_mapping(graph=graph_sync.graph, graph_map=graph_sync.graph_map[graph_sync.graph.id])

    #     print(count)

    #     state = compare_graphs(x=graph, y=graph_sync.graph)

    #     print(f"Node names are identical: {state}")

    # except Exception as e:
    #     print(f'Error loading graph: {e}')
    #     pass

    # graphs_deleted = await mongo_handler.delete_documents(db_name='Graph', collection_name='Graphs', filter={'graph.user_id': graph.user_id})

    # print(f'Graphs deleted: {graphs_deleted}')

if __name__ == "__main__":
    asyncio.run(main())