import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np

# Giả sử bạn lưu code thuật toán của bạn bè vào file tsp_algorithms.py
# Nếu bạn gom chung vào 1 file thì không cần dòng import này
from TSP_HillClimbing import HillClimbing

class TSPGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Salesman Problem - Cat Swarm Optimization")
        self.root.geometry("1000x650")
        # --- THÊM PHẦN CHỈNH FONT CHỮ TẠI ĐÂY ---
        style = ttk.Style()
        # Chỉnh font cho tất cả các dòng chữ (Label)
        style.configure("TLabel", font=("Arial", 12))
        # Chỉnh font và độ dày cho tất cả các nút bấm (Button)
        style.configure("TButton", font=("Arial", 12, "bold"), padding=5)
        # ----------------------------------------
        # --- CHIA BỐ CỤC CHÍNH (Dùng .pack()) ---
        # Frame bên trái: Chứa các nút bấm và ô nhập liệu
        self.control_frame = ttk.Frame(self.root, width=300, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Frame bên phải: Chứa bản đồ và kết quả
        self.view_frame = ttk.Frame(self.root, padding=10)
        self.view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._setup_control_panel()
        self._setup_visualization_panel()

    def _setup_control_panel(self):
        """Thiết lập khu vực điều khiển (Dùng .grid() như Excel)"""
        ttk.Label(self.control_frame, text="THIẾT LẬP THUẬT TOÁN", font=("Arial", 20, "bold")).grid(row=0, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=(20, 20))

        # 1. Bắt lỗi nhập liệu với validate="key" và %P
        val_cmd = (self.root.register(self.validate_number), '%P')

        ttk.Label(self.control_frame, text="Số lượng thành phố:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_cities = ttk.Entry(self.control_frame, validate="key", validatecommand=val_cmd, font=("Arial", 12))
        self.entry_cities.grid(row=1, column=1, pady=5)
        self.entry_cities.insert(0, "5")  # Giá trị mặc định

        ttk.Label(self.control_frame, text="Random Seed (Tùy chọn):", font=("Arial", 12)).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.entry_seed = ttk.Entry(self.control_frame, validate="key", validatecommand=val_cmd, font=("Arial", 12))
        self.entry_seed.grid(row=2, column=1, pady=5)
        self.entry_seed.insert(0, "42")

        # 2. Các nút chạy thuật toán
        # Nút chạy Hill Climbing (Chương II - Đã có code)
        self.btn_run_hc = ttk.Button(self.control_frame, text="CHẠY HILL CLIMBING", command=self.run_hill_climbing)
        self.btn_run_hc.grid(row=3, column=0, columnspan=2, pady=(20, 5), sticky=tk.EW)

        # Nút chạy Hybrid CSO (Chương III - Chừa đường lui)
        self.btn_run_cso = ttk.Button(self.control_frame, text="CHẠY HYBRID CSO + HC", command=self.run_hybrid_cso)
        self.btn_run_cso.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # 3. Khu vực hiển thị text kết quả
        ttk.Label(self.control_frame, text="CHI TIẾT KẾT QUẢ:", font=("Arial", 20, "bold")).grid(row=5, column=0,
                                                                                                 columnspan=2,
                                                                                                 pady=(20, 5),
                                                                                                 sticky=tk.W)

        self.txt_result = tk.Text(self.control_frame, height=18, width=35, wrap=tk.WORD)
        self.txt_result.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.control_frame.grid_rowconfigure(6, weight=1)  # Cho phép hàng 6 (chứa Textbox) giãn dọc
        self.control_frame.grid_columnconfigure(0, weight=1)  # Cho phép cột 0 giãn ngang
    def _setup_visualization_panel(self):
        """Thiết lập khu vực vẽ bản đồ bằng Matplotlib"""
        # Tạo một Figure của Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Bản đồ đường đi TSP", fontsize=14)
        self.ax.axis("off")

        # Nhúng Figure vào Tkinter Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.view_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def validate_number(self, P):
        """Hàm callback kiểm tra %P: Chỉ cho phép nhập số hoặc để trống"""
        if P.isdigit() or P == "":
            return True
        return False

    def run_hill_climbing(self):
        """Hàm kích hoạt khi bấm nút chạy Hill Climbing"""
        try:
            num_cities = int(self.entry_cities.get())
            seed_val = self.entry_seed.get()
            seed = int(seed_val) if seed_val != "" else None

            # 1. Khởi tạo và gọi class của bạn bè
            hc = HillClimbing(num_cities, seed=seed)
            best_route, best_distance = hc.solve()

            # 2. Cập nhật Textbox kết quả (Lấy từ hc.result)
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.insert(tk.END, hc.result)

            # 3. Cập nhật bản đồ
            self.draw_plot(hc.matrix, best_route)

        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập số lượng thành phố hợp lệ!")

    def run_hybrid_cso(self):
        """Placeholder cho Chương III"""
        messagebox.showinfo("Thông báo",
                            "Thuật toán Hybrid CSO đang được phát triển (Chương III).\nVui lòng quay lại sau!")
        # Khi có code, bạn chỉ việc import class CSO, gọi cso.solve() và tự động vẽ lại y như hàm run_hill_climbing()

    def draw_plot(self, matrix, route):
        """Hàm tự vẽ lại đồ thị ngay trên giao diện Tkinter"""
        self.ax.clear()  # Xóa bản đồ cũ

        if matrix is None:
            return

        # Đoạn này tái sử dụng logic vẽ NetworkX từ hàm create_plot() của bạn bè
        distance_arr = np.array(matrix)
        G = nx.from_numpy_array(distance_arr)
        pos = nx.spring_layout(G, seed=42)

        # Tạo danh sách các cạnh (edges) dựa trên route tốt nhất
        route_edges = [(route[i - 1], route[i]) for i in range(len(route))]

        # Vẽ các node và lable
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color="skyblue", node_size=600)
        nx.draw_networkx_labels(G, pos, ax=self.ax)

        # Vẽ các đường nối mờ (tất cả các cạnh)
        nx.draw_networkx_edges(G, pos, ax=self.ax, alpha=0.2)

        # Vẽ đậm đường đi tốt nhất (route) với màu đỏ
        nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=route_edges, edge_color="red", width=2.0)

        self.ax.set_title("Bản đồ lộ trình tối ưu TSP", fontsize=14)
        self.ax.axis("off")

        # Lệnh quan trọng: Yêu cầu Canvas vẽ lại đồ thị mới
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = TSPGuiApp(root)
    root.mainloop()