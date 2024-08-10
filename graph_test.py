import unittest
import asyncio
from graph import Node, Edge, Nodes, Edges, Graph

class TestGraphComponents(unittest.IsolatedAsyncioTestCase):

    async def test_node_creation(self):
        node = Node(id=1, name="Node1", parent="Root")
        self.assertEqual(node.id, 1)
        self.assertEqual(node.name, "Node1")
        self.assertEqual(node.parent, "Root")
        self.assertIsInstance(node.data, bytearray)

    async def test_node_engagement_update(self):
        node = Node(id=1, name="Node1", parent="Root")
        await node.set_interest_frequency()
        await node.set_eigenvector_centrality(0.5)
        await node.update_engagement()
        engagement_score = await node.get_engagement_score()
        self.assertGreater(engagement_score, 0)

    async def test_edge_creation(self):
        edge = Edge(from_node=1, to_node=2)
        self.assertEqual(edge.from_node, 1)
        self.assertEqual(edge.to_node, 2)
        self.assertIsInstance(edge.data, bytearray)

    async def test_edge_interaction_update(self):
        edge = Edge(from_node=1, to_node=2)
        await edge.update_interaction()
        interaction_strength = await edge.get_interaction_strength()
        self.assertGreater(interaction_strength, 0)

    async def test_nodes_add_and_get(self):
        nodes = Nodes()
        await nodes.add_node(id=1, name="Node1", parent="Root")
        node = await nodes.get_node(1)
        self.assertEqual(node.name, "Node1")

    async def test_edges_add_and_get(self):
        edges = Edges()
        from_path = (1,)
        to_path = (2,)
        await edges.add_edge(from_=from_path, to_=to_path)
        edge = await edges.get_edge(from_=from_path, to_=to_path)
        self.assertEqual(edge.from_node, 1)
        self.assertEqual(edge.to_node, 2)

    async def test_graph_add_node_and_edge(self):
        graph = Graph(user_id="user123")
        await graph.add_node(id=1, name="Node1")
        await graph.add_node(id=2, name="Node2")
        await graph.add_edge(from_=(1,), to_=(2,))
        node = await graph.get_node(1)
        edge = await graph.get_edge(from_=(1,), to_=(2,))
        self.assertEqual(node.name, "Node1")
        self.assertEqual(edge.from_node, 1)
        self.assertEqual(edge.to_node, 2)

    async def test_get_highest_interest_node_at_depth(self):
        graph = Graph(user_id="user123")
        await graph.add_node(id=1, name="RootNode")
        await graph.add_node(id=2, name="ChildNode1")
        await graph.add_node(id=3, name="ChildNode2")

        # Manually set interest frequencies
        node1 = await graph.get_node(1)
        node2 = await graph.get_node(2)
        node3 = await graph.get_node(3)

        await node1.set_interest_frequency(5)
        await node2.set_interest_frequency(10)
        await node3.set_interest_frequency(3)

        # Set subgraphs
        subgraph = Graph(user_id="user123")
        await subgraph.add_node(id=4, name="SubChildNode")
        sub_node = await subgraph.get_node(4)
        await sub_node.set_interest_frequency(15)

        await graph.set_node_subgraph(1, subgraph)

        # Test finding the highest interest node at depth 0 and 1
        highest_node_depth_0 = await graph.get_highest_interest_node_at_depth(0)
        highest_node_depth_1 = await graph.get_highest_interest_node_at_depth(1)

        self.assertEqual(highest_node_depth_0, "ChildNode1")
        self.assertEqual(highest_node_depth_1, "SubChildNode")

if __name__ == '__main__':
    unittest.main()
