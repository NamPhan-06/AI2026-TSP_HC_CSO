import tkinter as tk
from GUI import TSPGuiApp
import Visualization
import Experiment
from TSP_HillClimbing import HillClimbing
from TSP_CSO import CSO

def main():
    root = tk.Tk()

    def driver_hill_climbing():
        num_cities, seed = app.get_inputs()
        if num_cities is None: return
        try:
            hc = HillClimbing(num_cities, seed=seed)
            best_route, _ = hc.solve()

            # Truyen ca 2 chuoi text
            app.update_result_text(hc.result, hc.history_result)
            Visualization.draw_both(app.canvas, app.ax_route, app.ax_conv, hc.matrix, best_route, hc.convergence,
                                    "Lộ trình: Hill Climbing", "Hội tụ: HC", "HC Thuần",
                                    color='red')  # Đã đổi thành red
        except Exception as e:
            app.show_error(f"Lỗi chạy HC:\n{str(e)}")

    def driver_cso():
        num_cities, seed = app.get_inputs()
        if num_cities is None: return
        try:
            cso = CSO(num_cities, seed=seed)
            best_route, _ = cso.solve()

            # Truyen ca 2 chuoi text
            app.update_result_text(cso.result, cso.history_result)
            Visualization.draw_both(app.canvas, app.ax_route, app.ax_conv, cso.matrix, best_route, cso.convergence,
                                    "Lộ trình: CSO Thuần", "Hội tụ: CSO", "CSO Thuần",
                                    color='blue')  # Giữ màu blue cho CSO
        except Exception as e:
            app.show_error(f"Lỗi chạy CSO:\n{str(e)}")

    def driver_compare():
        num_cities, seed = app.get_inputs()
        if num_cities is None: return
        try:
            result_data = Experiment.run_comparison_experiment(num_cities, seed)

            # Truyen ca 2 chuoi text
            app.update_result_text(result_data["report_text"], result_data["history_text"])
            combined_convergence = [result_data["hc_convergence"], result_data["cso_convergence"]]
            Visualization.draw_both(app.canvas, app.ax_route, app.ax_conv, result_data["matrix"], result_data["cso_route"], combined_convergence,
                                    "Lộ trình Tốt nhất", "So sánh Tốc độ Hội tụ", ["HC Thuần", "CSO Thuần"])
        except Exception as e:
            app.show_error(f"Lỗi chạy So sánh:\n{str(e)}")

    app = TSPGuiApp(root, cmd_hc=driver_hill_climbing, cmd_cso=driver_cso, cmd_compare=driver_compare)
    root.mainloop()

if __name__ == "__main__":
    main()