import json
import time
import os
from typing import Type, List
import random
import asyncio
import pickle
import hashlib

from motor.motor_asyncio import AsyncIOMotorCollection
from cryptography.fernet import Fernet

from geo_graph import Graph, GraphSync, Edges, Edge
from mongo import MongoHandler



def generate_graph(graph: Graph, categories: dict[str,dict]):

    for id, parent in enumerate(categories.keys()):
        graph.add_node(id, parent)

        if len(categories[parent]) > 0:
            graph.set_node_subgraph(id, subgraph=Graph(user_id=graph.user_id, edges=graph.edges))
            generate_graph(graph=graph.node_map.nodes[id].subgraph, categories=categories[parent])


def extract_edge_endpoints(categories: dict[str,dict]) -> List[str]:
    endpoints = []

    for key in categories.keys():
        if len(categories[key]) > 0:
            sub_endpoints = extract_edge_endpoints(categories=categories[key])

            for sub in sub_endpoints:
                nested_key = [key]
                nested_key.extend(sub)
                endpoints.append(nested_key)
        else:
            endpoints.append([key])

    return endpoints


def generate_edges(root_graph: Graph, categories: dict[str,dict]) -> None:
    edge_endpoints = extract_edge_endpoints(categories=categories)

    for i in range(len(edge_endpoints)):
        for j in range(len(edge_endpoints)):
            if i != j:
                root_graph.edges.add_edge_from_str(from_=edge_endpoints[i], to_=edge_endpoints[j], root_graph=root_graph)


def generate_random_changes(graph: Graph):
    changes = []
    for _ in range(random.randint(1, 5)):  # Generate 1 to 5 changes
        change_type = random.choice(["update_node", "update_edge"])
        if change_type == "update_node":
            node_id = random.choice(list(graph.nodes.keys()))
            changes.append({
                "type": "update_node",
                "node_id": node_id,
                "node_data": {
                    "interest_frequency": random.randint(0, 10),
                    "eigenvector_centrality": random.uniform(0, 1),
                    "last_engagement": time.time(),
                    "engagement_score": random.uniform(0, 5)
                }
            })
        else:  # update_edge
            edge = random.choice(list(graph.edges.edges.keys()))
            changes.append({
                "type": "update_edge",
                "from_node": edge[0],
                "to_node": edge[1],
                "edge_data": {
                    "interaction_strength": random.randint(0, 10),
                    "last_interaction": time.time(),
                    "contextual_similarity": random.uniform(0, 1),
                    "sequential_relation": random.uniform(0, 1)
                }
            })
    return changes


def verify_changes(graph, changes):
    print("Verifying changes:")
    for change in changes:
        if change["type"] == "update_node":
            node = graph.nodes[change["node_id"]]
            print(f"Node {change['node_id']}:")
            for key, value in change["node_data"].items():
                print(f"  {key}: {getattr(node, key)}")
        elif change["type"] == "update_edge":
            edge = graph.edges[(change["from_node"], change["to_node"])]
            print(f"Edge ({change['from_node']}, {change['to_node']}):")
            for key, value in change["edge_data"].items():
                print(f"  {key}: {getattr(edge, key)}")


def print_graph_stats(graph: Graph):
    print(f"\nTotal nodes: {len(graph.node_map.nodes)}")
    print(f"Total edges: {len(graph.edges.edges)}")
    

async def list_all_entries(collection: AsyncIOMotorCollection):
    cursor = collection.find({})
    documents = []
    async for document in cursor:
        documents.append(document)
    return documents


async def delete_many(collection: AsyncIOMotorCollection):
    result = await collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents")


async def main():
    # Initialize graph and generate data
    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    before_graph = time.time()
    generate_graph(graph=graph, categories=categories)
    after_graph = time.time()
    graph_time = after_graph - before_graph
    print('Graph generated')
    print(f"Graph generation took: {graph_time}s")

    before_edges = time.time()
    generate_edges(root_graph=graph, categories=categories)
    after_edges = time.time()
    edge_time = after_edges - before_edges
    print(f"Total amount of edges: {len(graph.edges.edges)}")
    print(f"Edge generation took: {edge_time}s")

    print(f"\nTotal creation time: {graph_time + edge_time}s")
    
    # print('Graph Saved')

    # # Calculate and save the checksum
    # checksum = calculate_checksum(file_path)
    # checksum_file_path = 'graph_checksum.md5'
    # save_checksum(checksum, checksum_file_path)

    # # Load the graph from the file
    # if verify_checksum(file_path, checksum_file_path):
    #     loaded_graph = load_graph_from_file(file_path)
    #     print("Checksum verified. Graph loaded successfully.")
    # else:
    #     print("Checksum verification failed. File may be corrupted.")


    # # Save graph to file
    # filename = 'victory'
    # graph.save_graph(filename)

    # # Print file size
    # file_size = os.path.getsize(file_path)
    # print(f"The size of the saved graph is {file_size} bytes")
    # print(f"The size of the saved graph is {file_size / 1024:.2f} KB")
    # print(f"The size of the saved graph is {file_size / (1024 * 1024):.2f} MB")

    # Initialize MongoHandler and GraphSync
    # uri = "mongodb+srv://mimir.kjfum9z.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&appName=Mimir"
    # mongo_handler = MongoHandler(
    #     uri=uri,
    #     cert_path='./mongocert.pem',
    # )

    # graph_sync = GraphSync(
    #     graph=graph,
    #     mongo_handler=mongo_handler,
    #     local_storage_path=os.path.join(os.getcwd(), 'graph_sync_timestamp.json')
    # )

    # # Test database connection
    # if await mongo_handler.test_database_connection():
    #     print("Successfully connected to the database.")
    # else:
    #     print("Failed to connect to the database. Please check your connection settings.")
    #     return
    
    # await graph_sync.save_to_database()

    # database = mongo_handler.client.get_database('Graph')

    # collection = database.get_collection('UserGraph')

    # col_list = await list_all_entries(collection=collection)

    # for col in col_list:
    #     print(col['_id'])

    # if await graph_sync.save_to_database():
    #     print(f"Graph saved to datanase with ID: {graph.id}")

    # try:
    #     loaded_graph = await graph_sync.load_from_database(graph.id)
    #     print(f"Loaded graph with ID: {loaded_graph.id}")

    #     print(type(graph))
    #     print(type(loaded_graph))

    #     print(len(graph.nodes[1].subgraph.nodes[2].subgraph.nodes))
    #     print(len(loaded_graph.nodes[1].subgraph.nodes[2].subgraph.nodes))
    # except Exception as e:
    #     print(f"Error loading graph from database: {e}")

    # mongo_handler.client.close()
    
    # yes = await graph_sync.save_to_database()

    # # Check if the graph exists in the database
    # graph_exists = await graph_sync.graph_exists_in_db()
    # if not graph_exists:
    #     print("Graph does not exist in the database. Saving it now.")
    #     await graph_sync.save_to_database()
    #     print("Graph saved to database.")
    # else:
    #     print("Graph exists in the database.")

    # # Verify the graph structure
    # structure_verified = await graph_sync.verify_graph_structure()
    # if structure_verified:
    #     print("Graph structure in the database matches the local graph.")
    # else:
    #     print("Graph structure in the database does not match the local graph.")
    #     print("Updating the database with the current local graph.")
    #     await graph_sync.save_to_database()
    #     print("Database updated with current local graph.")

    # # Perform 5 iterations of changes and syncing
    # for i in range(1):
    #     print(f"\nIteration {i+1}:")
        
    #     # Simulate changes
    #     changes = generate_random_changes(graph_sync.local_graph)
        
    #     # Log the changes
    #     await graph_sync.log_changes(changes)
    #     print(f"Logged {len(changes)} changes.")
        
    #     # Sync the changes
    #     await graph_sync.sync()
    #     print("Synced changes with the database.")
        
    #     # Verify changes
    #     verify_changes(graph_sync.local_graph, changes)
        
    #     # Print graph statistics
    #     print_graph_stats(graph_sync.local_graph)
        
    #     # Verify graph structure after changes
    #     structure_verified = await graph_sync.verify_graph_structure()
    #     if structure_verified:
    #         print("Graph structure in the database still matches the local graph after changes.")
    #     else:
    #         print("Warning: Graph structure in the database no longer matches the local graph after changes.")
        
    #     # Wait a bit before the next iteration
    #     await asyncio.sleep(1)

    # # Final verification
    # final_verification = await graph_sync.verify_graph_structure()
    # if final_verification:
    #     print("\nFinal verification: Graph structure in the database matches the local graph.")
    # else:
    #     print("\nFinal verification: Graph structure in the database does not match the local graph.")



# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())