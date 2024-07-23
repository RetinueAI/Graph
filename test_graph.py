import asyncio
from datetime import datetime, timezone
from geo_graph import Graph, GraphSync, MongoHandler

async def test_graph_operations():
    # Initialize MongoDB connection
     # Create GraphSync instance
    uri = "mongodb+srv://mimir.kjfum9z.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&appName=Mimir"
    mongo_handler = MongoHandler(
        uri=uri,
        cert_path='./mongocert.pem',
    )
    
    # Create a main graph
    main_graph = Graph(user_id="test_user_1", version=1, id="main_graph_id")
    
    # Add nodes to the main graph
    main_graph.add_node(1, "Node 1")
    main_graph.add_node(2, "Node 2")
    main_graph.add_node(3, "Node 3")
    
    # Add edges to the main graph
    main_graph.add_edge(1, 2)
    main_graph.add_edge(2, 3)
    main_graph.add_edge(3, 1)
    
    # Create a subgraph
    subgraph = Graph(user_id="test_user_1", version=1, id="subgraph_id")
    subgraph.add_node(1, "Subnode 1")
    subgraph.add_node(2, "Subnode 2")
    subgraph.add_edge(1, 2)
    
    # Set the subgraph for Node 3
    main_graph.set_node_subgraph(3, subgraph)
    
    # Update node engagement and edge interaction
    main_graph.update_node_engagement(1)
    main_graph.update_edge_interaction(1, 2)
    
    # Calculate eigenvector centrality
    main_graph.calculate_eigenvector_centrality()
    
    # Initialize GraphSync
    graph_sync = GraphSync(
        graph=main_graph,
        mongo_handler=mongo_handler,
        local_storage_path="./graph_sync_data.json"
    )
    
    # Save graphs to database
    success = await graph_sync.save_graphs_to_database(main_graph)
    print(f"Saved graphs to database: {success}")
    
    # Load graphs from database
    loaded_graph = await graph_sync.load_graphs_from_database("test_user_1", "main_graph_id")

    if loaded_graph is None:
        print("Failed to load graph from database")
        return
    
    if loaded_graph:
        print("Loaded graph structure:")
        print(f"Main graph nodes: {list(loaded_graph.nodes.keys())}")
        print(f"Main graph edges: {list(loaded_graph.edges.keys())}")
        
        subgraph_node = loaded_graph.get_node(3)
        if subgraph_node and subgraph_node.subgraph:
            print(f"Subgraph nodes: {list(subgraph_node.subgraph.nodes.keys())}")
            print(f"Subgraph edges: {list(subgraph_node.subgraph.edges.keys())}")
        
        # Test node and edge data
        node_1 = loaded_graph.get_node(1)
        edge_1_2 = loaded_graph.get_edge(1, 2)
        
        print(f"Node 1 engagement score: {node_1.engagement_score}")
        print(f"Edge (1, 2) interaction strength: {edge_1_2.interaction_strength}")
        
        # Test PyG data
        print(f"PyG data node features shape: {loaded_graph.pyg_data.x.shape}")
        print(f"PyG data edge index shape: {loaded_graph.pyg_data.edge_index.shape}")
    else:
        print("Failed to load graph from database")
    
    # Test sync functionality
    await graph_sync.sync()
    print("Sync completed")

    # Test graph modification and re-saving
    loaded_graph.add_node(4, "Node 4")
    loaded_graph.add_edge(3, 4)
    
    success = await graph_sync.save_graphs_to_database(loaded_graph)
    print(f"Saved modified graph to database: {success}")
    
    # Re-load and verify changes
    reloaded_graph = await graph_sync.load_graphs_from_database("test_user_1", "main_graph_id")
    if reloaded_graph:
        print(f"Reloaded graph nodes: {list(reloaded_graph.nodes.keys())}")
        print(f"Reloaded graph edges: {list(reloaded_graph.edges.keys())}")
    else:
        print("Failed to reload modified graph from database")

# Run the test
asyncio.run(test_graph_operations())