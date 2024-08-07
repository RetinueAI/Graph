from typing import List, Dict, Tuple

from graph import Graph


def generate_graph(graph: Graph, categories: dict[str,dict]) -> None:
    _generate_graph(graph=graph, categories=categories)
    _generate_edges(root_graph=graph, categories=categories)

def _generate_graph(graph: Graph, categories: dict[str,dict]) -> None:

    for id, parent in enumerate(categories.keys()):
        graph.add_node(id, parent)

        if len(categories[parent]) > 0:
            graph.set_node_subgraph(id, subgraph=Graph(user_id=graph.user_id, edges=graph.edges))
            _generate_graph(graph=graph.nodes.nodes[id].subgraph, categories=categories[parent])


def _extract_edge_endpoints(categories: dict[str,dict]) -> List[str]:
    endpoints = []

    for key in categories.keys():
        if len(categories[key]) > 0:
            sub_endpoints = _extract_edge_endpoints(categories=categories[key])

            for sub in sub_endpoints:
                nested_key = [key]
                nested_key.extend(sub)
                endpoints.append(nested_key)
        else:
            endpoints.append([key])

    return endpoints


def _generate_edges(root_graph: Graph, categories: dict[str,dict]) -> None:
    edge_endpoints = _extract_edge_endpoints(categories=categories)

    for i in range(len(edge_endpoints)):
        for j in range(len(edge_endpoints)):
            if i != j:
                root_graph.edges.add_edge_from_str(from_=edge_endpoints[i], to_=edge_endpoints[j], root_graph=root_graph)


def generate_node_map(graph: Graph, node_map: Dict[Tuple[int,str], Dict]):
    for node in graph.nodes.nodes.values():
        if node.subgraph:
            node_map[(node.id, graph.id)] = {}
            generate_node_map(node.subgraph, node_map=node_map[(node.id, graph.id)])


def generate_graph_map(graph: Graph, graph_map: Dict) -> None:

    for node in graph.nodes.nodes.values():
        if node.subgraph:
            graph_map[str(node.id)] = {node.subgraph.id: {}}
            generate_graph_map(graph=node.subgraph, graph_map=graph_map[str(node.id)][node.subgraph.id])