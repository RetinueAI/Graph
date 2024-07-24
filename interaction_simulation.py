import json
import asyncio
from typing import Dict, List
import random
from time import time

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


def graph_interaction_update(graph: Graph, result: List[str]):
    category = result[0].replace('\n','')
    for node in graph.nodes.values():
        if node.name == category:
            node.interest_frequency += 1

            if len(result) > 1:
                graph_interaction_update(graph=node.subgraph, result=result[1:])

            break


def check_engagement(graph: Graph, indentation_level: int = 0):
    indentation = indentation_level * '\t'
    for node in graph.nodes.values():
        if node.interest_frequency > 0:
            print(f"{indentation}{node.name}")
            print(f"{indentation}Interactions: {node.interest_frequency}")

            if node.subgraph:
                new_indentation_level = indentation_level + 1
                check_engagement(graph=node.subgraph, indentation_level=new_indentation_level)


def simulate_interactions(graph: Graph, category_list: List[str], n: int = 0):
    for i in range(n):
        categories = random.choice(category_list)
        categories = categories.split('/')[1:]

        graph_interaction_update(
            graph=graph, 
            result=categories,
        )


async def main():

    with open('categories.txt', 'r') as f:
        categoriy_list = f.readlines()


    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    before_generation = time()
    generate_graph(graph=graph, categories=categories)
    after_generation = time()

    print("Graph generated")

    n = 1_000_000
    before_simulation = time()
    simulate_interactions(graph=graph, category_list=categoriy_list, n=n)
    after_simulation = time()

    # check_engagement(graph=graph)

    print(f"Graph generation took: {after_generation - before_generation}s")
    print(f"Number of interactions: {n}")
    print(f"Interaction simulation took: {after_simulation - before_simulation}s")

if __name__ == "__main__":
    asyncio.run(main())