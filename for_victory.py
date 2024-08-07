import os
import base64
import json
import asyncio

from cryptography.fernet import Fernet
from graph import Graph, Node, Edge, GraphSync
from mongo import MongoHandler



# Function to generate and save a key
def generate_and_save_key(file_path):
    key = Fernet.generate_key()
    with open(file_path, 'wb') as key_file:
        key_file.write(key)
    return key

# Function to load a key
def load_key(file_path):
    with open(file_path, 'rb') as key_file:
        return key_file.read()

def generate_graph(graph: Graph, categories):

    for i, parent in enumerate(categories.keys()):
        graph.add_node(i, parent)

        if len(categories[parent]) > 0:
            graph.set_node_subgraph(i, subgraph=Graph())
            generate_graph(graph=graph.nodes[i].subgraph, categories=categories[parent])

    for j in range(len(graph.nodes)):
        for k in range(len(graph.nodes)):
            if j != k:
                graph.add_edge(j, k)
                graph.add_edge(k, j)

# Function to compare two graphs
def compare_graphs(graph1: Graph, graph2: Graph):
    if graph1.id != graph2.id:
        print("Graph IDs do not match")
        return False
    
    if set(graph1.nodes.keys()) != set(graph2.nodes.keys()):
        print("Node sets do not match")
        return False
    
    if set(graph1.edges.keys()) != set(graph2.edges.keys()):
        print("Edge sets do not match")
        return False
    
    for node_id, node1 in graph1.nodes.items():
        node2 = graph2.nodes[node_id]
        if node1.name != node2.name or node1.interest_frequency != node2.interest_frequency:
            print(f"Node {node_id} attributes do not match")
            return False
    
    for edge_key, edge1 in graph1.edges.items():
        edge2 = graph2.edges[edge_key]
        if edge1.interaction_strength != edge2.interaction_strength:
            print(f"Edge {edge_key} attributes do not match")
            return False
    
    print("Graphs match!")
    return True

# Main test function
async def main():
    # Generate and save the key
    key_file_path = 'encryption_key.key'
    # key = generate_and_save_key(key_file_path)
    # print(f"Generated key: {base64.b64encode(key).decode()}")

    # Load key
    key = load_key(key_file_path)

    # Set the key as an environment variable
    os.environ['ENCRYPTION_KEY'] = key.decode()

    original_graph = Graph()

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    # Create a sample graph

    generate_graph(graph=original_graph, categories=categories)

    # Create GraphSync instance
    uri = "mongodb+srv://mimir.kjfum9z.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&appName=Mimir"
    mongo_handler = MongoHandler(
        uri=uri,
        cert_path='./mongocert.pem',
    )


if __name__ == "__main__":
    asyncio.run(main())