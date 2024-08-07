from typing import List, Dict, Tuple, Any
import json
import uuid

from graph import Graph



def generate_new_graph() -> Graph:
    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph = Graph(user_id=user_id)

    with open('categories.json', 'r') as f:
        categories = json.load(f)

    generate_graph(graph=graph, categories=categories)

    generate_graph_map(graph=graph)
    print("Graph Map generated...")

    return graph


def generate_graph(graph: Graph, categories: dict[str,dict] = None, graph_map: Dict = None) -> None:
    if categories:
        _generate_graph(graph=graph, categories=categories)
        _generate_edges(root_graph=graph, categories=categories)
        print("Graph generated from categories...")
    else:
        if graph_map:
            _generate_graph(graph=graph, graph_map=graph_map)
            _generate_edges(root_graph=graph, graph_map=graph_map)
            print("Graph generated from grap map...")


def _generate_graph(graph: Graph, categories: dict[str,dict] = None, graph_map: Dict = None) -> None:
    if graph_map:
        for node in graph_map['nodes']:
            graph.add_node(
                id=node['id'],
                name=node['name']
            )

            if node['subgraph']:
                graph.set_node_subgraph(
                    node_id=node['id'],
                    subgraph=Graph(
                        user_id=graph.user_id,
                        edges=graph.edges,
                        id=node['subgraph']['id']
                    )
                )
                _generate_graph(
                    graph=graph.nodes.nodes[node['id']].subgraph,
                    graph_map=node['subgraph'],
                )
    else:
        if categories:
            for id, parent in enumerate(categories.keys()):
                graph.add_node(id, parent)

                if len(categories[parent]) > 0:
                    graph.set_node_subgraph(id, subgraph=Graph(user_id=graph.user_id, edges=graph.edges))
                    _generate_graph(graph=graph.nodes.nodes[id].subgraph, categories=categories[parent])


def _extract_edge_endpoints(categories: dict[str,dict] = None, graph_map: Dict = None) -> List[str]:
    edge_endpoints = []

    if categories:
        for key in categories.keys():
            if len(categories[key]) > 0:
                sub_endpoints = _extract_edge_endpoints(categories=categories[key])

                for sub in sub_endpoints:
                    edge_endpoint = [key]
                    edge_endpoint.extend(sub)
                    edge_endpoints.append(edge_endpoint)
            else:
                edge_endpoints.append([key])
    else:
        if graph_map:
            for node in graph_map['nodes']:
                if node['subgraph']:
                    sub_endpoints = _extract_edge_endpoints(graph_map=node['subgraph'])

                    for sub in sub_endpoints:
                        edge_endpoint = [node['name']]
                        edge_endpoint.extend(sub)
                        edge_endpoints.append(edge_endpoint)
                else:
                    edge_endpoints.append([node['name']])

    return edge_endpoints


def _generate_edges(root_graph: Graph, categories: dict[str,dict] = None, graph_map: Dict = None) -> None:
    if categories:
        edge_endpoints = _extract_edge_endpoints(categories=categories)
    else:
        if graph_map:
            edge_endpoints = _extract_edge_endpoints(graph_map=graph_map)

    for i in range(len(edge_endpoints)):
        for j in range(len(edge_endpoints)):
            if i != j:
                root_graph.edges.add_edge_from_str(from_=edge_endpoints[i], to_=edge_endpoints[j], root_graph=root_graph)


def generate_graph_map(graph: Graph):
    with open('user_id.txt', 'r') as f:
        user_id = f.read()

    graph_map = {'user_id': user_id, 'graph': {}}
    _generate_graph_map(graph=graph, graph_map=graph_map['graph'])

    with open('graph_map.json', 'w') as f:
        json.dump(graph_map, f)


def _generate_graph_map(graph: Graph, graph_map: Dict = {}) -> Dict:
    graph_map['id'] = graph.id

    if len(graph.nodes.nodes) > 0:
        graph_map['nodes'] = []

        for id, node in graph.nodes.nodes.items():
            node_map = {}
            node_map['id'] = id
            node_map['name'] = node.name
            node_map['parent'] = graph.id

            if node.subgraph:
                node_map['subgraph'] = _generate_graph_map(graph=node.subgraph, graph_map={})
            else:
                node_map['subgraph'] = {}

            graph_map['nodes'].append(node_map)

    return graph_map


def load_graph_map() -> Dict:
    with open('graph_map.json', 'r') as f:
        graph_map = json.load(f)

    return graph_map