import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import colorsys

from graph import Graph

def get_color_for_depth(depth, max_depth):
    hue = depth / max_depth
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'

def visualize_3d_static(graph, output_file: str = "category_graph_3d.png"):
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d')

    pos = nx.spring_layout(graph, dim=3, k=0.5, iterations=50)
    
    max_depth = max(nx.get_node_attributes(graph, 'depth').values())
    max_count = max(nx.get_node_attributes(graph, 'metrics').values(), key=lambda x: x.interaction_count).interaction_count

    # Assign colors to top-level categories
    top_level_categories = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    color_map = plt.cm.get_cmap('tab20')
    category_colors = {cat: color_map(i/len(top_level_categories)) for i, cat in enumerate(top_level_categories)}

    # Draw edges
    for edge in graph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        ax.plot([x0, x1], [y0, y1], [z0, z1], color='gray', alpha=0.2, linewidth=0.5)

    # Draw nodes
    for node, (x, y, z) in pos.items():
        depth = graph.nodes[node]['depth']
        count = graph.nodes[node]['metrics'].interaction_count
        
        # Find the top-level category for this node
        current = node
        while graph.in_degree(current) > 0:
            current = list(graph.predecessors(current))[0]
        
        color = category_colors[current]
        size = 20 + (count / max_count) * 1000 if max_count > 0 else 20  # Increased size scaling
        ax.scatter(x, y, z, color=color, s=size, alpha=0.6)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D Graph visualization saved as '{output_file}'")

def create_interactive_html(graph: nx.Graph, output_file: str = "category_graph_3d.html"):
    pos = nx.spring_layout(graph, dim=3, k=0.5, iterations=50)
    
    node_x, node_y, node_z = [], [], []
    node_text, node_color = [], []
    max_depth = max(nx.get_node_attributes(graph, 'depth').values())
    
    for node, data in graph.nodes(data=True):
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f"Name: {data['name']}<br>Depth: {data['depth']}")
        node_color.append(get_color_for_depth(data['depth'], max_depth))

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=3, color=node_color, opacity=0.8),
        text=node_text,
        hoverinfo='text'
    )

    edge_x, edge_y, edge_z = [], [], []
    edge_colors = []
    for edge in graph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        depth0 = graph.nodes[edge[0]]['depth']
        depth1 = graph.nodes[edge[1]]['depth']
        edge_colors.append(get_color_for_depth(max(depth0, depth1), max_depth))

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color=edge_colors, width=1),
        hoverinfo='none'
    )

    layout = go.Layout(
        title='3D Category Graph Visualization',
        scene=dict(
            xaxis=dict(title=''),
            yaxis=dict(title=''),
            zaxis=dict(title=''),
            aspectmode='data'
        ),
        margin=dict(t=50, b=0, l=0, r=0),
        showlegend=False,
        hovermode='closest'
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.write_html(output_file)
    print(f"Interactive 3D Graph visualization saved as '{output_file}'")

# In your main script:
if __name__ == "__main__":
    graph_handler = GraphHandler.from_json('categories.json')
    graph_handler.print_graph_stats()
    print("\nAnalyzing subgraphs:")
    graph_handler.analyze_subgraphs()
    
    visualize_3d_static(graph_handler.graph)
    create_interactive_html(graph_handler.graph)