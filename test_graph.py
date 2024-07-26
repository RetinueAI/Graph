# import unittest
# from datetime import datetime, timezone, timedelta
# import torch
# from torch_geometric.data import Data
# import networkx as nx
# from geo_graph import (
#     Graph, Node, Edge, GlobalNodeID, CrossHierarchyEdge,
#     RecursiveData, GraphRandomizer, GraphPresenter
# )

# class TestGeoGraph(unittest.TestCase):

#     def setUp(self):
#         self.user_id = "test_user"
#         self.graph = Graph(user_id=self.user_id)

#     def test_node_creation_and_properties(self):
#         node = Node(id=1, name="Test Node")
#         self.assertEqual(node.id, 1)
#         self.assertEqual(node.name, "Test Node")
#         self.assertEqual(node.interest_frequency, 0)
#         self.assertIsInstance(node.last_engagement, datetime)

#     def test_node_update_engagement(self):
#         node = Node(id=1, name="Test Node")
#         initial_frequency = node.interest_frequency
#         initial_engagement = node.last_engagement
#         node.update_engagement()
#         self.assertEqual(node.interest_frequency, initial_frequency + 1)
#         self.assertGreater(node.last_engagement, initial_engagement)

#     def test_edge_creation_and_properties(self):
#         edge = Edge(from_node=1, to_node=2)
#         self.assertEqual(edge.from_node, 1)
#         self.assertEqual(edge.to_node, 2)
#         self.assertEqual(edge.interaction_strength, 0)
#         self.assertIsInstance(edge.last_interaction, datetime)

#     def test_edge_update_interaction(self):
#         edge = Edge(from_node=1, to_node=2)
#         initial_strength = edge.interaction_strength
#         initial_interaction = edge.last_interaction
#         edge.update_interaction()
#         self.assertEqual(edge.interaction_strength, initial_strength + 1)
#         self.assertGreater(edge.last_interaction, initial_interaction)

#     def test_graph_add_node(self):
#         node = self.graph.add_node(1, "Test Node")
#         self.assertIn(1, self.graph.nodes)
#         self.assertEqual(node.name, "Test Node")

#     def test_graph_add_edge(self):
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_node(2, "Node 2")
#         edge = self.graph.add_edge(1, 2)
#         self.assertIn((1, 2), self.graph.edges)
#         self.assertEqual(edge.from_node, 1)
#         self.assertEqual(edge.to_node, 2)

#     def test_graph_get_edge(self):
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_node(2, "Node 2")
#         self.graph.add_edge(1, 2)
#         edge = self.graph.get_edge(1, 2)
#         self.assertIsInstance(edge, Edge)
#         self.assertEqual(edge.from_node, 1)
#         self.assertEqual(edge.to_node, 2)

#     def test_graph_remove_edge(self):
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_node(2, "Node 2")
#         self.graph.add_edge(1, 2)
#         self.graph.remove_edge(1, 2)
#         self.assertNotIn((1, 2), self.graph.edges)

#     def test_graph_remove_node(self):
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_node(2, "Node 2")
#         self.graph.add_edge(1, 2)
#         self.graph.remove_node(1)
#         self.assertNotIn(1, self.graph.nodes)
#         self.assertNotIn((1, 2), self.graph.edges)

#     def test_graph_update_node_engagement(self):
#         self.graph.add_node(1, "Test Node")
#         initial_frequency = self.graph.nodes[1].interest_frequency
#         self.graph.update_node_engagement(1)
#         self.assertEqual(self.graph.nodes[1].interest_frequency, initial_frequency + 1)

#     def test_graph_update_edge_interaction(self):
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_node(2, "Node 2")
#         self.graph.add_edge(1, 2)
#         initial_strength = self.graph.edges[(1, 2)].interaction_strength
#         self.graph.update_edge_interaction(1, 2)
#         self.assertEqual(self.graph.edges[(1, 2)].interaction_strength, initial_strength + 1)

#     def test_graph_add_cross_hierarchy_edge(self):
#         from_node = GlobalNodeID(path=[0, 1])
#         to_node = GlobalNodeID(path=[1, 0])
#         edge = self.graph.add_cross_hierarchy_edge(from_node, to_node)
#         self.assertIn((str(from_node), str(to_node)), self.graph.cross_hierarchy_edges)
#         self.assertIsInstance(edge, CrossHierarchyEdge)
#         self.assertEqual(edge.from_node, from_node)
#         self.assertEqual(edge.to_node, to_node)

#     def test_graph_set_node_subgraph(self):
#         self.graph.add_node(1, "Parent Node")
#         subgraph = Graph(user_id=self.user_id)
#         self.graph.set_node_subgraph(1, subgraph)
#         self.assertIsInstance(self.graph.nodes[1].subgraph, Graph)
#         self.assertEqual(self.graph.nodes[1].subgraph.depth, self.graph.depth + 1)

#     def test_graph_get_root_graph(self):
#         root = self.graph
#         subgraph = Graph(user_id=self.user_id)
#         root.set_node_subgraph(0, subgraph)
#         root_graph = subgraph.get_root_graph()
#         self.assertEqual(root_graph.id, root.id)
#         self.assertEqual(root_graph.user_id, root.user_id)

#     def test_graph_get_all_cross_hierarchy_edges(self):
#         self.graph.add_cross_hierarchy_edge(GlobalNodeID(path=[0]), GlobalNodeID(path=[1]))
#         subgraph = Graph(user_id=self.user_id)
#         subgraph.add_cross_hierarchy_edge(GlobalNodeID(path=[0, 0]), GlobalNodeID(path=[0, 1]))
#         self.graph.set_node_subgraph(0, subgraph)
#         edges = self.graph.get_all_cross_hierarchy_edges()
#         self.assertEqual(len(edges), 2)

#     def test_graph_calculate_eigenvector_centrality(self):
#         self.graph.add_node(0, "Node 0")
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_edge(0, 1)
#         self.graph.calculate_eigenvector_centrality()
#         for node in self.graph.nodes.values():
#             self.assertIsInstance(node.eigenvector_centrality, float)

#     def test_graph_to_networkx(self):
#         self.graph.add_node(0, "Node 0")
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_edge(0, 1)
#         nx_graph = self.graph.to_networkx()
#         self.assertIsInstance(nx_graph, nx.DiGraph)
#         self.assertEqual(len(nx_graph.nodes), 2)
#         self.assertEqual(len(nx_graph.edges), 1)

#     def test_graph_from_networkx(self):
#         nx_graph = nx.DiGraph()
#         nx_graph.add_node(0, name="Node 0")
#         nx_graph.add_node(1, name="Node 1")
#         nx_graph.add_edge(0, 1)
#         graph = Graph.from_networkx(nx_graph)
#         self.assertEqual(len(graph.nodes), 2)
#         self.assertEqual(len(graph.edges), 1)

#     def test_graph_to_dict(self):
#         self.graph.add_node(0, "Node 0")
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_edge(0, 1)
#         graph_dict = self.graph.to_dict()
#         self.assertIsInstance(graph_dict, dict)
#         self.assertIn('nodes', graph_dict)
#         self.assertIn('edges', graph_dict)

#     def test_graph_from_dict(self):
#         original_graph = Graph(user_id=self.user_id)
#         original_graph.add_node(0, "Node 0")
#         original_graph.add_node(1, "Node 1")
#         original_graph.add_edge(0, 1)
#         graph_dict = original_graph.to_dict()
#         reconstructed_graph = Graph.from_dict(graph_dict)
#         self.assertEqual(len(reconstructed_graph.nodes), len(original_graph.nodes))
#         self.assertEqual(len(reconstructed_graph.edges), len(original_graph.edges))

#     def test_recursive_data(self):
#         x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
#         edge_index = torch.tensor([[0, 1], [1, 0]])
#         recursive_data = RecursiveData(x=x, edge_index=edge_index)
#         self.assertIsInstance(recursive_data, Data)
#         self.assertTrue(torch.equal(recursive_data.x, x))
#         self.assertTrue(torch.equal(recursive_data.edge_index, edge_index))

#     def test_global_node_id(self):
#         global_id = GlobalNodeID(path=[0, 1, 2])
#         self.assertEqual(str(global_id), "0.1.2")
#         reconstructed_id = GlobalNodeID.from_string("0.1.2")
#         self.assertEqual(global_id.path, reconstructed_id.path)

#     def test_graph_randomizer(self):
#         self.graph.add_node(0, "Node 0")
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_edge(0, 1)
#         randomizer = GraphRandomizer()
#         randomizer.simulate_usage(self.graph, days=5, max_daily_interactions=5)
#         for node in self.graph.nodes.values():
#             self.assertGreater(node.interest_frequency, 0)
#             self.assertGreater(node.engagement_score, 0)

#     def test_graph_presenter(self):
#         self.graph.add_node(0, "Node 0")
#         self.graph.add_node(1, "Node 1")
#         self.graph.add_edge(0, 1)
#         presenter = GraphPresenter()
#         # This is more of a visual test, so we'll just ensure it doesn't raise an exception
#         try:
#             presenter.display_simulation_results(self.graph, max_depth=2)
#         except Exception as e:
#             self.fail(f"GraphPresenter raised an exception: {e}")

# if __name__ == '__main__':
#     unittest.main()