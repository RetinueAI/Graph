from typing import Dict

from graph import Graph



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


def compare_graphs(x: Graph, y: Graph) -> bool:
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
    
    return state