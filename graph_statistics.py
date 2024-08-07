import sys
import json
import asyncio
import bson

from graph import Graph
from graph_generation import generate_graph
from graph_inspection import get_n_nodes




async def main():
    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    generate_graph(graph=graph, categories=categories)

    print("Graph generated")

    n_nodes = await get_n_nodes(graph=graph)

    print(f"Number of nodes: {n_nodes}")

    node = graph.nodes.get_node(1)
    dict_node = node.to_bson()
    dict_node_size = sys.getsizeof(dict_node)
    print(f"The size of a node when converted into a dictionary: {dict_node_size}")

    edge = next(iter(graph.edges.edges.values()))
    dict_edge_size = sys.getsizeof(edge.to_dict())
    print(f"The size of an edge when converted to a dictionary: {dict_edge_size}")

    edge_data_size = sys.getsizeof(edge.data)
    edge_data_bytes = sys.getsizeof(bytes(edge.data))
    print(F"The size of edge data: {edge_data_size}")
    print(F"The size of edge data in the native format: {edge_data_bytes}")

    from_node_size = sys.getsizeof(edge.from_node)
    from_node_bytes = sys.getsizeof(bytes(edge.from_node))
    to_node_size = sys.getsizeof(edge.to_node)
    to_node_bytes = sys.getsizeof(bytes(edge.to_node))

    print(f"From node id: {edge.from_node}")
    print(f"To node id: {edge.to_node}")

    print(f"The size of from node id: {from_node_size}")
    print(f"The size of to node id: {to_node_size}")

    print(f"The size of the from node id in the native format: {from_node_bytes}")
    print(f"The size of the from node id in the native format: {to_node_bytes}")

    print(f"Total memory usage of the information contained in a graph: {edge_data_size+from_node_size+to_node_size}")
    print(f"Total memory usage of the information contained in a graph in the native format: {edge_data_bytes+from_node_bytes+to_node_bytes}")

    document = {
        'f': edge.from_node,
        't': edge.to_node,
        'd': bson.Binary(bytes(edge.data))
    }

    byte_data = bytes(edge.data)
    restructed_data = bytearray.fromhex(byte_data.hex())

    print(f"The contents of edge data: {edge.data}")
    print(f"The contents of the edge data converted to bytes: {byte_data}")
    print(f"The contents of edge data converted to bytes, then converted to hex, then converted back to a bytearray: {bytearray.fromhex(bytes(edge.data).hex())}")

    print(f"The original data and the restructed data is the same: {edge.data == restructed_data}")

    bson_data = bson.BSON.encode(document=document)

    print(f"The size of the bson document being stored in mongodb: {len(bson_data)}")

if __name__ == '__main__':
    asyncio.run(main())