# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def setup_canvas(parent_frame):
    """Khởi tạo không gian vẽ (Figure và Canvas) nhúng vào Tkinter"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Bản đồ lộ trình", fontsize=14)
    ax.axis("off")

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    return canvas, ax


def draw_route(canvas, ax, matrix, route, title="Bản đồ lộ trình tối ưu TSP"):
    """Vẽ lộ trình lên Canvas đã khởi tạo"""
    if matrix is None or route is None:
        return

    ax.clear()
    distance_arr = np.array(matrix)
    G = nx.from_numpy_array(distance_arr)
    pos = nx.spring_layout(G, seed=42)

    route_edges = [(route[i - 1], route[i]) for i in range(len(route))]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#87CEFA", node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=route_edges, edge_color="red", width=2.0)

    ax.set_title(title, fontsize=14, pad=10)
    ax.axis("off")
    canvas.draw()