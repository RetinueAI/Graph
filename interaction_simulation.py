import json
import asyncio
from typing import Dict, List
import random
from time import time

from geo_graph import Graph, Node
from generate_graph import generate_graph
from memory import get_memory_usage, human_readable_size



def graph_interaction_update(graph: Graph, result: List[str]):
    category = result[0].replace('\n','')
    for node in graph.node_map.nodes.values():
        if node.name == category:
            node.interest_frequency += 1

            if len(result) > 1:
                graph_interaction_update(graph=node.subgraph, result=result[1:])

            break


def simulate_interactions(graph: Graph, category_list: List[str], n: int = 0):
    for i in range(n):
        categories = random.choice(category_list)
        categories = categories.split('/')[1:]

        graph_interaction_update(
            graph=graph, 
            result=categories,
        )


def check_engagement(graph: Graph, indentation_level: int = 0):
    indentation = indentation_level * '\t'
    for node in graph.node_map.nodes.values():
        if node.interest_frequency > 0:
            print(f"{indentation}{node.name}")
            print(f"{indentation}Interactions: {node.interest_frequency}")

            if node.subgraph:
                new_indentation_level = indentation_level + 1
                check_engagement(graph=node.subgraph, indentation_level=new_indentation_level)


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
    
    memory_usage_before = get_memory_usage(graph)

    n = 10_000_000
    before_simulation = time()
    simulate_interactions(graph=graph, category_list=categoriy_list, n=n)
    after_simulation = time()

    before_check = time()
    print("Checking engagement:")
    check_engagement(graph=graph)
    after_check = time()

    print(f"Graph generation took: {after_generation - before_generation}s")
    print(f"Number of interactions: {n}")
    print(f"Interaction simulation took: {after_simulation - before_simulation}s")
    print(f"Engagement check took: {after_check - before_check}s")

    memory_usage_after = get_memory_usage(graph)
    print(f"The Graph uses {human_readable_size(memory_usage_before)} of memory.")
    print(f"The Graph uses {human_readable_size(memory_usage_after)} of memory after the simulated actions.")

if __name__ == "__main__":
    asyncio.run(main())