from typing import Dict

from graph import Graph, Edge, Edges



def check_id(x: str, y: str) -> int:
    if x == y:
        return 0
    return 1


def check_graph_mapping(graph: Graph, graph_map: Dict) -> int:

    count = 0

    for node in graph_map.keys():
        if graph.nodes.nodes[int(node)].subgraph:
            count += check_id(graph.nodes.nodes[int(node)].subgraph.id, graph_map[node])

            count += check_graph_mapping(graph=graph.nodes.nodes[int(node)].subgraph, graph_map=graph_map[node][list(graph_map[node].keys())[0]])

    return count


def compare_graphs(x: Graph, y: Graph, edges: bool = True) -> bool:
    state = True

    x_len = len(x.nodes.nodes)
    y_len = len(y.nodes.nodes)

    if x_len == y_len:
        for i in range(x_len):
            if x.nodes.nodes[i].name == y.nodes.nodes[i].name:
                if (x.nodes.nodes[i].subgraph != None) and (y.nodes.nodes[i].subgraph != None):
                    if not compare_graphs(x.nodes.nodes[i].subgraph, y.nodes.nodes[i].subgraph):
                        return False
            else:
                return False
    else:
        state = False

    if edges:
        state = compare_edges(x=x.edges, y=y.edges)
    
    return state


def compare_edges(x: Edges, y: Edges) -> bool:

    if len(x.edges) == len(y.edges):
        errors = 0
        for key in x.edges.keys():
            errors += _compare_edge(x=x.edges[key], y=y.edges[key])
            
            if x.edge_map.get_id(key) != y.edge_map.get_id(key):
                errors += 1
            if x.edge_map.get_path(x.edge_map.get_id(key)) != y.edge_map.get_path(y.edge_map.get_id(key)):
                errors += 1

        if errors == 0:
            return True
    else:
        print("There's a different amount of Edges")
        return False
    return False


def _compare_edge(x: Edge, y: Edge) -> int:
    errors = 0

    if x.from_node != y.from_node:
        errors += 1
    if x.to_node != y.to_node:
        errors += 1
    if x.data != y.data:
        errors += 1
    
    return errors


async def get_n_nodes(graph: Graph) -> int:
    n_nodes = 0

    for node in graph.nodes.nodes.values():
        n_nodes += 1
        
        if node.subgraph:
            n_nodes += await get_n_nodes(node.subgraph)

    return n_nodes