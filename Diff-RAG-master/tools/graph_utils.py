import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph, save_path="graph_viz.png"):
           
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold')
    plt.savefig(save_path)
    print(f"Graph visualization saved to {save_path}")