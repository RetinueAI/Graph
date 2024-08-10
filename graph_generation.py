from typing import List, Dict, Tuple, Any
import json
import uuid
import logging
import random
import time

from graph import Graph, Edges, Nodes, Edge


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def generate_new_graph() -> Graph:
    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id, edges=Edges(), root=True)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    await generate_graph(graph=graph, categories=categories)

    await generate_graph_map(graph=graph)
    print("Graph Map generated...")

    return graph


async def generate_graph(graph: Graph, categories: dict[str,dict] = None, graph_map: Dict = None) -> None:
    if categories:
        await _generate_graph(graph=graph, categories=categories)
        await _generate_edges(root_graph=graph, categories=categories)
        print("Graph generated from categories...")
    else:
        if graph_map:
            await _generate_graph(graph=graph, graph_map=graph_map)
            await _generate_edges(root_graph=graph, graph_map=graph_map)
            print("Graph generated from grap map...")


async def _generate_graph(graph: Graph, categories: dict[str,dict] = None, graph_map: Dict = None) -> None:
    logger.info(f"Generating graph {graph.id}, depth: {graph.depth}")

    if graph_map:
        for node in graph_map['nodes']:
            node_id = node['id']
            logger.debug(f"Processing node {node_id}: {node['name']} in graph {graph.id}")

            await graph.add_node(
                id=node['id'],
                name=node['name']
            )

            if node['subgraph']:
                logger.debug(f"Creating subgraph for node {node_id}: {node['name']}")
                nodes = Nodes()
                subgraph = Graph(
                    user_id=graph.user_id,
                    id=node['subgraph']['id'],
                    depth=graph.depth +1,
                    nodes=nodes,
                )

                await _generate_graph(
                    graph=subgraph,
                    graph_map=node['subgraph'],
                )

                await graph.set_node_subgraph(
                    node_id=node_id,
                    subgraph=subgraph,
                )
        logger.info(f"Finished generating graph {graph.id}. Total nodes: {len(graph.nodes.nodes)}")
    else:
        if categories:
            for id, parent in enumerate(categories.keys()):
                await graph.add_node(id, parent)

                if len(categories[parent]) > 0:
                    await graph.set_node_subgraph(id, subgraph=Graph(user_id=graph.user_id, edges=graph.edges))
                    await _generate_graph(graph=graph.nodes.nodes[id].subgraph, categories=categories[parent])


async def _extract_edge_endpoints(categories: dict[str,dict] = None, graph_map: Dict = None) -> List[str]:
    edge_endpoints = []

    if categories:
        for key in categories.keys():
            if len(categories[key]) > 0:
                sub_endpoints = await _extract_edge_endpoints(categories=categories[key])

                for sub in sub_endpoints:
                    edge_endpoint = [key]
                    edge_endpoint.extend(sub)
                    edge_endpoints.append(edge_endpoint)
            else:
                edge_endpoints.append([key])
    else:
        if graph_map:
            for node in graph_map['nodes']:
                if node['subgraph']:
                    sub_endpoints = await _extract_edge_endpoints(graph_map=node['subgraph'])

                    for sub in sub_endpoints:
                        edge_endpoint = [node['name']]
                        edge_endpoint.extend(sub)
                        edge_endpoints.append(edge_endpoint)
                else:
                    edge_endpoints.append([node['name']])

    return edge_endpoints


async def _generate_edges(root_graph: Graph, categories: dict[str,dict] = None, graph_map: Dict = None) -> None:
    if categories:
        edge_endpoints = await _extract_edge_endpoints(categories=categories)
    else:
        if graph_map:
            edge_endpoints = await _extract_edge_endpoints(graph_map=graph_map)

    for i in range(len(edge_endpoints)):
        for j in range(len(edge_endpoints)):
            if i != j:
                await root_graph.edges.add_edge_from_str(from_=edge_endpoints[i], to_=edge_endpoints[j], root_graph=root_graph)


async def generate_graph_map(graph: Graph):
    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph_map = {'user_id': user_id, 'graph': {}}
    await _generate_graph_map(graph=graph, graph_map=graph_map['graph'])

    with open('graph_map.json', 'w') as f:
        json.dump(graph_map, f)


async def _generate_graph_map(graph: Graph, graph_map: Dict = {}) -> Dict:
    graph_map['id'] = graph.id

    if len(graph.nodes.nodes) > 0:
        graph_map['nodes'] = []

        for id, node in graph.nodes.nodes.items():
            node_map = {}
            node_map['id'] = id
            node_map['name'] = node.name
            node_map['parent'] = graph.id

            if node.subgraph:
                node_map['subgraph'] = await _generate_graph_map(graph=node.subgraph, graph_map={})
            else:
                node_map['subgraph'] = {}

            graph_map['nodes'].append(node_map)

    return graph_map


def load_graph_map() -> Dict:
    with open('graph_map.json', 'r') as f:
        graph_map = json.load(f)

    return graph_map


async def simulate_interactions(graph: Graph, n: int = 0):

    before = time.time()
    with open('categories.txt', 'r') as f:
        category_list = f.readlines()

    for i in range(n):
        categories = random.choice(category_list)
        categories = categories.split('/')[1:]

        await graph_interaction_update(
            graph=graph, 
            result=categories,
        )

    after = time.time()
    logger.info(f"Graph interaction simulation took {after-before}s")


async def graph_interaction_update(graph: Graph, result: List[str]):
    category = result[0].replace('\n','')
    for node in graph.nodes.nodes.values():
        if node.name == category:
            await node.set_interest_frequency()

            if len(result) > 1:
                await graph_interaction_update(graph=node.subgraph, result=result[1:])


async def simulate_edge_updates(graph: Graph, n: int = 0, k: int = 0):
    before = time.time()

    n_updates = 0

    for _ in range(n):
        edges = random.choices(list(graph.edges.edges.values()), k=k)
        for edge in edges:
            await edge.set_interaction_strength()
            n_updates += 1

    after = time.time()
    logger.info(f"Edge interaction simulation took: {after-before}s")
    logger.info(f"Number of edge update simulations: {n_updates}")