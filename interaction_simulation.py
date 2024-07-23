import json
import asyncio
from typing import Dict, List

from geo_graph import Graph, Node


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


def generate_node_name_map(graph: Graph, map: Dict):
    for idx, node in graph.nodes.items():
        map[node.name] = idx

        if node.subgraph:
            generate_node_name_map(node.subgraph, map)


def graph_interaction_update(node: Node, node_map: Dict[str,int], sub_cats: List[str] = None):
    node.interest_frequency += 1

    if sub_cats:
        graph_interaction_update(
            node=node.subgraph.nodes[node_map[sub_cats[0].replace('\n','')]],
            node_map=node_map,
            sub_cats=sub_cats[1:] if len(sub_cats) > 1 else None
        )


async def main():

    with open('categories.txt', 'r') as f:
        categoriy_list = f.readlines()


    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    generate_graph(graph=graph, categories=categories)

    node_name_map = {}

    generate_node_name_map(graph=graph, map=node_name_map)

    for cats in categoriy_list:
        valid_categoris = cats.split('/')[1:]
        graph_interaction_update(
            node=graph.nodes[node_name_map[valid_categoris[0].replace('\n','')]],
            node_map=node_name_map,
            sub_cats=valid_categoris[1:] if len(valid_categoris) > 1 else None,
        )

    print(graph.nodes[1].interest_frequency)


if __name__ == "__main__":
    asyncio.run(main())