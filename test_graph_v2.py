import unittest
from datetime import datetime, timezone

from torch_geometric.data import Data

from geo_graph_v2 import Node, Graph, GlobalGraph, Edge


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node = Node(id="1", name="Test Node")

    def test_initialization(self):
        self.assertEqual(self.node.id, "1")
        self.assertEqual(self.node.name, "Test Node")
        self.assertEqual(self.node.interest_frequency, 0)
        self.assertAlmostEqual(self.node.eigenvector_centrality, 0.0)
        self.assertIsInstance(self.node.last_engagement, datetime)
        self.assertAlmostEqual(self.node.engagement_score, 0.0)
        self.assertEqual(self.node.graph_ids, [])

    def test_update_engagement(self):
        initial_engagement = self.node.last_engagement
        self.node.update_engagement()
        self.assertEqual(self.node.interest_frequency, 1)
        self.assertGreater(self.node.last_engagement, initial_engagement)

    def test_update_engagement_score(self):
        self.node.interest_frequency = 10
        self.node.eigenvector_centrality = 0.5
        self.node.update_engagement_score()
        self.assertGreater(self.node.engagement_score, 0)

    def test_to_feature_vector(self):
        feature_vector = self.node.to_feature_vector()
        self.assertEqual(len(feature_vector), 4)
        self.assertIsInstance(feature_vector[0], int)
        self.assertIsInstance(feature_vector[1], float)
        self.assertIsInstance(feature_vector[2], float)
        self.assertIsInstance(feature_vector[3], float)

class TestEdge(unittest.TestCase):
    def setUp(self):
        self.edge = Edge(from_node="1", to_node="2")

    def test_initialization(self):
        self.assertEqual(self.edge.from_node, "1")
        self.assertEqual(self.edge.to_node, "2")
        self.assertEqual(self.edge.interaction_strength, 0)
        self.assertIsInstance(self.edge.last_interaction, datetime)
        self.assertAlmostEqual(self.edge.contextual_similarity, 0.0)
        self.assertAlmostEqual(self.edge.sequential_relation, 0.0)
        self.assertEqual(self.edge.graph_ids, [])

    def test_update_interaction(self):
        initial_interaction = self.edge.last_interaction
        self.edge.update_interaction()
        self.assertEqual(self.edge.interaction_strength, 1)
        self.assertGreater(self.edge.last_interaction, initial_interaction)

    def test_to_feature_vector(self):
        feature_vector = self.edge.to_feature_vector()
        self.assertEqual(len(feature_vector), 4)
        self.assertIsInstance(feature_vector[0], int)
        self.assertIsInstance(feature_vector[1], float)
        self.assertIsInstance(feature_vector[2], float)
        self.assertIsInstance(feature_vector[3], float)


class TestGlobalGraph(unittest.TestCase):
    def setUp(self):
        self.global_graph = GlobalGraph()

    def test_add_node(self):
        node = self.global_graph.add_node("1", "Test Node", "graph1")
        self.assertIn("1", self.global_graph.nodes)
        self.assertEqual(node.name, "Test Node")
        self.assertIn("graph1", node.graph_ids)

    def test_add_existing_node_to_new_graph(self):
        self.global_graph.add_node("1", "Test Node", "graph1")
        node = self.global_graph.add_node("1", "Test Node", "graph2")
        self.assertIn("graph1", node.graph_ids)
        self.assertIn("graph2", node.graph_ids)

    def test_add_edge(self):
        self.global_graph.add_node("1", "Node 1", "graph1")
        self.global_graph.add_node("2", "Node 2", "graph1")
        edge = self.global_graph.add_edge("1", "2", "graph1")
        self.assertIn(("1", "2"), self.global_graph.edges)
        self.assertIn("graph1", edge.graph_ids)

    def test_add_existing_edge_to_new_graph(self):
        self.global_graph.add_node("1", "Node 1", "graph1")
        self.global_graph.add_node("2", "Node 2", "graph1")
        self.global_graph.add_edge("1", "2", "graph1")
        edge = self.global_graph.add_edge("1", "2", "graph2")
        self.assertIn("graph1", edge.graph_ids)
        self.assertIn("graph2", edge.graph_ids)

    def test_get_graph_edges(self):
        self.global_graph.add_node("1", "Node 1", "graph1")
        self.global_graph.add_node("2", "Node 2", "graph1")
        self.global_graph.add_edge("1", "2", "graph1")
        edges = self.global_graph.get_graph_edges("graph1")
        self.assertEqual(edges, [("1", "2")])

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.global_graph = GlobalGraph()
        self.graph = Graph(user_id="user1", global_graph=self.global_graph)

    def test_initialization(self):
        self.assertIsInstance(self.graph.id, str)
        self.assertEqual(self.graph.user_id, "user1")
        self.assertEqual(self.graph.node_ids, [])

    def test_add_node(self):
        node = self.graph.add_node("Test Node")
        self.assertIn(node.id, self.graph.node_ids)
        self.assertIn(node.id, self.global_graph.nodes)

    def test_add_edge(self):
        node1 = self.graph.add_node("Node 1")
        node2 = self.graph.add_node("Node 2")
        edge = self.graph.add_edge(node1.id, node2.id)
        self.assertIn((node1.id, node2.id), self.global_graph.edges)

    def test_add_edge_invalid_nodes(self):
        with self.assertRaises(ValueError):
            self.graph.add_edge("invalid1", "invalid2")

    def test_get_nodes(self):
        self.graph.add_node("Node 1")
        self.graph.add_node("Node 2")
        nodes = self.graph.get_nodes()
        self.assertEqual(len(nodes), 2)

    def test_get_edges(self):
        node1 = self.graph.add_node("Node 1")
        node2 = self.graph.add_node("Node 2")
        self.graph.add_edge(node1.id, node2.id)
        edges = self.graph.get_edges()
        self.assertEqual(len(edges), 1)

    def test_to_pyg_data(self):
        node1 = self.graph.add_node("Node 1")
        node2 = self.graph.add_node("Node 2")
        self.graph.add_edge(node1.id, node2.id)
        pyg_data = self.graph.to_pyg_data()
        self.assertIsInstance(pyg_data, Data)
        self.assertEqual(pyg_data.num_nodes, 2)
        self.assertEqual(pyg_data.num_edges, 1)

    def test_calculate_eigenvector_centrality(self):
        node1 = self.graph.add_node("Node 1")
        node2 = self.graph.add_node("Node 2")
        node3 = self.graph.add_node("Node 3")
        self.graph.add_edge(node1.id, node2.id)
        self.graph.add_edge(node2.id, node3.id)
        self.graph.calculate_eigenvector_centrality()
        for node in self.graph.get_nodes():
            self.assertGreater(node.eigenvector_centrality, 0)

    def test_to_dict(self):
        node1 = self.graph.add_node("Node 1")
        node2 = self.graph.add_node("Node 2")
        graph_dict = self.graph.to_dict()
        self.assertIsInstance(graph_dict, dict)
        self.assertEqual(graph_dict['user_id'], "user1")
        self.assertEqual(len(graph_dict['node_ids']), 2)

    def test_from_dict(self):
        graph_dict = {
            'id': 'test_id',
            'user_id': 'user1',
            'node_ids': ['node1', 'node2']
        }
        graph = Graph.from_dict(graph_dict, self.global_graph)
        self.assertEqual(graph.id, 'test_id')
        self.assertEqual(graph.user_id, 'user1')
        self.assertEqual(graph.node_ids, ['node1', 'node2'])

class TestComplexGraph(unittest.TestCase):
    def setUp(self):
        self.global_graph = GlobalGraph()
        self.graph = Graph(user_id="complex_user", global_graph=self.global_graph)
        
        # Create nodes
        self.nodes = {}
        for i in range(20):
            self.nodes[i] = self.graph.add_node(f"Node {i}")
        
        # Create edges (avoiding self-loops)
        for i in range(19):
            self.graph.add_edge(self.nodes[i].id, self.nodes[i+1].id)
        
        # Add some circular connections (avoiding self-loops)
        self.graph.add_edge(self.nodes[19].id, self.nodes[0].id)
        self.graph.add_edge(self.nodes[5].id, self.nodes[10].id)
        self.graph.add_edge(self.nodes[15].id, self.nodes[7].id)
        
        # Add some bidirectional connections
        self.graph.add_edge(self.nodes[3].id, self.nodes[12].id)
        self.graph.add_edge(self.nodes[12].id, self.nodes[3].id)
        
        # Update some node and edge properties
        for i in range(20):
            node = self.nodes[i]
            node.update_engagement()
            if i % 3 == 0:
                node.update_engagement()  # Some nodes are more engaged
        
        for edge in self.graph.get_edges():
            edge.update_interaction()
            if int(edge.from_node) % 2 == 0:  # Assuming node IDs are numeric strings
                edge.update_interaction()  # Some edges have more interactions
    
    def test_graph_size(self):
        self.assertEqual(len(self.graph.get_nodes()), 20)
        self.assertEqual(len(self.graph.get_edges()), 23)
    
    def test_node_connections(self):
        # Test that node 0 is connected to node 1 and node 19
        node_0_connections = [edge.to_node for edge in self.graph.get_edges() if edge.from_node == self.nodes[0].id]
        self.assertIn(self.nodes[1].id, node_0_connections)
        self.assertIn(self.nodes[19].id, node_0_connections)

    def test_bidirectional_connection(self):
        node_3_connections = [edge.to_node for edge in self.graph.get_edges() if edge.from_node == self.nodes[3].id]
        node_12_connections = [edge.to_node for edge in self.graph.get_edges() if edge.from_node == self.nodes[12].id]
        self.assertIn(self.nodes[12].id, node_3_connections)
        self.assertIn(self.nodes[3].id, node_12_connections)
    
    def test_engagement_scores(self):
        engagement_scores = [node.engagement_score for node in self.graph.get_nodes()]
        self.assertTrue(any(score > 0 for score in engagement_scores))
        self.assertTrue(max(engagement_scores) > min(engagement_scores))
    
    def test_interaction_strengths(self):
        interaction_strengths = [edge.interaction_strength for edge in self.graph.get_edges()]
        self.assertTrue(any(strength > 1 for strength in interaction_strengths))
        self.assertTrue(max(interaction_strengths) > min(interaction_strengths))
    
    def test_eigenvector_centrality(self):
        self.graph.calculate_eigenvector_centrality()
        centralities = [node.eigenvector_centrality for node in self.graph.get_nodes()]
        self.assertTrue(all(centrality > 0 for centrality in centralities))
        self.assertTrue(max(centralities) > min(centralities))
    
    def test_to_pyg_data(self):
        pyg_data = self.graph.to_pyg_data()
        self.assertEqual(pyg_data.num_nodes, 20)
        self.assertEqual(pyg_data.num_edges, 23)
        self.assertEqual(pyg_data.x.shape, (20, 4))  # 20 nodes, 4 features per node
        self.assertEqual(pyg_data.edge_attr.shape, (23, 4))  # 23 edges, 4 features per edge
    
    def test_graph_to_dict_and_back(self):
        graph_dict = self.graph.to_dict()
        reconstructed_graph = Graph.from_dict(graph_dict, self.global_graph)
        self.assertEqual(len(reconstructed_graph.get_nodes()), 20)
        self.assertEqual(len(reconstructed_graph.get_edges()), 23)
    
    def test_global_graph_consistency(self):
        global_node_count = len(self.global_graph.nodes)
        global_edge_count = len(self.global_graph.edges)
        self.assertEqual(global_node_count, 20)
        self.assertEqual(global_edge_count, 23)
    
    def test_node_feature_vector(self):
        for node in self.graph.get_nodes():
            feature_vector = node.to_feature_vector()
            self.assertEqual(len(feature_vector), 4)
            self.assertTrue(all(isinstance(feat, (int, float)) for feat in feature_vector))
    
    def test_edge_feature_vector(self):
        for edge in self.graph.get_edges():
            feature_vector = edge.to_feature_vector()
            self.assertEqual(len(feature_vector), 4)
            self.asser

if __name__ == '__main__':
    unittest.main()