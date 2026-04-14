import tkinter as tk
import time

# Import các module đã chia tách
from GUI import TSPGuiApp
import Visualization
import Experiment
from TSP_HillClimbing import HillClimbing
from Hybrid_CSO_HillClimbing import hybrid_algorithm_advanced, create_distance_matrix


def main():
    root = tk.Tk()

    # ==========================================
    # CÁC HÀM DRIVER XỬ LÝ LOGIC NGHIỆP VỤ
    # ==========================================

    def driver_hill_climbing():
        num_cities, seed = app.get_inputs()
        if num_cities is None: return
        try:
            # Chạy thuật toán
            hc = HillClimbing(num_cities, seed=seed)
            best_route, _ = hc.solve()

            # Cập nhật GUI
            app.update_result_text(hc.result)
            Visualization.draw_route(app.canvas, app.ax, hc.matrix, best_route, "Lộ trình: Hill Climbing")
        except Exception as e:
            app.show_error(f"Lỗi chạy HC:\n{str(e)}")

    def driver_hybrid_cso():
        num_cities, seed = app.get_inputs()
        if num_cities is None: return
        try:
            matrix = create_distance_matrix(num_cities, seed=seed)
            # Hứng 5 biến, bỏ đi các đoạn tính thời gian và format text thủ công
            best_route, best_distance, _, _, result_str = hybrid_algorithm_advanced(matrix, N=30, max_iter=100,
                                                                                    hc_rate=0.3)

            # Đẩy thẳng chuỗi format chuẩn lên giao diện
            app.update_result_text(result_str)
            Visualization.draw_route(app.canvas, app.ax, matrix, best_route, "Lộ trình: Hybrid CSO")
        except Exception as e:
            app.show_error(f"Lỗi chạy Hybrid:\n{str(e)}")

    def driver_compare():
        num_cities, seed = app.get_inputs()
        if num_cities is None: return
        try:
            # Gọi hàm thí nghiệm so sánh
            result_data = Experiment.run_comparison_experiment(num_cities, seed)

            # Cập nhật GUI
            app.update_result_text(result_data["report_text"])
            Visualization.draw_route(app.canvas, app.ax, result_data["matrix"], result_data["hybrid_route"],
                                     "So Sánh (Đang hiện Hybrid)")
        except Exception as e:
            app.show_error(f"Lỗi chạy So sánh:\n{str(e)}")

    # ==========================================
    # KHỞI TẠO VÀ LIÊN KẾT
    # ==========================================

    # Khởi tạo App và truyền 3 cái driver vào để gắn cho 3 cái nút
    app = TSPGuiApp(root,
                    cmd_hc=driver_hill_climbing,
                    cmd_cso=driver_hybrid_cso,
                    cmd_compare=driver_compare)

    # Chạy ứng dụng
    root.mainloop()


if __name__ == "__main__":
    main()