import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def setup_canvas(parent_frame):
    """Khởi tạo không gian vẽ (1 hàng, 2 cột) nhúng vào Tkinter"""
    fig, (ax_route, ax_conv) = plt.subplots(1, 2, figsize=(10, 5))

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    return canvas, ax_route, ax_conv


def draw_both(canvas, ax_route, ax_conv, matrix, route, convergence, route_title, conv_title, conv_labels="Khoảng cách",
              color='blue'):
    """Hàm vẽ đồng thời cả Lộ trình (trái) và Biểu đồ hội tụ (phải)"""
    # 1. VẼ LỘ TRÌNH (Bên trái) - Giữ nguyên
    ax_route.clear()
    if matrix is not None and route is not None:
        distance_arr = np.array(matrix)
        G = nx.from_numpy_array(distance_arr)
        pos = nx.spring_layout(G, seed=42)
        route_edges = [(route[i - 1], route[i]) for i in range(len(route))]

        nx.draw_networkx_nodes(G, pos, ax=ax_route, node_color="#87CEFA", node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax_route, font_size=10)
        nx.draw_networkx_edges(G, pos, ax=ax_route, alpha=0.2)
        nx.draw_networkx_edges(G, pos, ax=ax_route, edgelist=route_edges, edge_color="red", width=2.0)
        ax_route.set_title(route_title, fontsize=12, pad=10)
        ax_route.axis("off")

    # 2. VẼ BIỂU ĐỒ HỘI TỤ (Bên phải)
    ax_conv.clear()
    if convergence:
        if isinstance(convergence[0], list):
            # Chế độ so sánh: HC màu đỏ, CSO màu xanh
            colors = ['red', 'blue']
            markers = ['o', 's']
            linestyles = ['-', '--']
            labels = conv_labels if isinstance(conv_labels, list) else ['HC', 'CSO']

            for i, curve in enumerate(convergence):
                ax_conv.plot(curve, color=colors[i], marker=markers[i], markersize=5,
                             linestyle=linestyles[i], linewidth=2,
                             markevery=max(1, len(curve) // 10), label=labels[i])
        else:
            # Chế độ chạy đơn lẻ: Sử dụng tham số color truyền vào
            ax_conv.plot(convergence, color=color, marker='o', markersize=4, linewidth=2, label=conv_labels)

        ax_conv.set_title(conv_title, fontsize=12, fontweight='bold', pad=10)
        ax_conv.set_xlabel("Số vòng lặp (Iteration)", fontsize=10)
        ax_conv.set_ylabel("Khoảng cách", fontsize=10)
        ax_conv.grid(True, linestyle='--', alpha=0.7)
        ax_conv.legend()

    canvas.draw()