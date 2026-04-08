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
            # Tạo ma trận và chạy thuật toán
            matrix = create_distance_matrix(num_cities, seed=seed)
            start = time.perf_counter()
            best_route, best_distance, _, complexity = hybrid_algorithm_advanced(matrix, N=30, max_iter=100,
                                                                                 hc_rate=0.3)
            time_ms = round((time.perf_counter() - start) * 1000.0, 2)

            # Lấy thông số dung lượng
            space_bytes = complexity["space_measured_bytes"]
            space_kb = complexity["space_measured_kb"]

            # Định dạng chuỗi kết quả
            res = ""
            res += "Ma trận khoảng cách:\n"
            for row in matrix:
                res += f"{row}\n"

            res += f"\nGiải pháp tốt nhất: {list(best_route)}\n"
            res += f"Quãng đường ngắn nhất: {best_distance}\n"
            res += f"Thời gian đo được (ms): {time_ms}\n"
            res += f"Dung lượng đo được (bytes): {space_bytes}\n"
            res += f"Dung lượng đo được (KB): {space_kb}\n"

            # Cập nhật GUI
            app.update_result_text(res)
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