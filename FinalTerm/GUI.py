import tkinter as tk
from tkinter import ttk, messagebox
import Visualization


class TSPGuiApp:
    # Nhận thêm 3 tham số là 3 cái hàm do main.py truyền vào
    def __init__(self, root, cmd_hc, cmd_cso, cmd_compare):
        self.root = root
        self.root.title("TSP Optimizer - MVC Architecture")
        self.root.geometry("1000x650")

        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 11, "bold"), padding=5)

        self.cmd_hc = cmd_hc
        self.cmd_cso = cmd_cso
        self.cmd_compare = cmd_compare

        self.control_frame = ttk.Frame(self.root, width=320, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.view_frame = ttk.Frame(self.root, padding=10)
        self.view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._setup_control_panel()
        self.canvas, self.ax = Visualization.setup_canvas(self.view_frame)

    def _setup_control_panel(self):
        ttk.Label(self.control_frame, text="THIẾT LẬP THUẬT TOÁN", font=("Arial", 14, "bold")).grid(row=0, column=0,
                                                                                                    columnspan=2,
                                                                                                    pady=(0, 20))

        val_cmd = (self.root.register(self.validate_number), '%P')

        ttk.Label(self.control_frame, text="Số lượng thành phố:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_cities = ttk.Entry(self.control_frame, font=("Arial", 12), validate="key",
                                      validatecommand=val_cmd)
        self.entry_cities.grid(row=1, column=1, pady=5, sticky=tk.EW)
        self.entry_cities.insert(0, "5")  # Default là 5 như bạn yêu cầu

        ttk.Label(self.control_frame, text="Random seed:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.entry_seed = ttk.Entry(self.control_frame, font=("Arial", 12), validate="key",
                                    validatecommand=val_cmd)
        self.entry_seed.grid(row=2, column=1, pady=5, sticky=tk.EW)
        self.entry_seed.insert(0, "42")

        # Gắn 3 cái nút với 3 cái hàm nhận từ main.py
        self.btn_run_hc = ttk.Button(self.control_frame, text="CHẠY HC", command=self.cmd_hc)
        self.btn_run_hc.grid(row=3, column=0, columnspan=2, pady=(15, 5), sticky=tk.EW)

        self.btn_run_cso = ttk.Button(self.control_frame, text="CHẠY HYBRID CSO + HC", command=self.cmd_cso)
        self.btn_run_cso.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.btn_compare = ttk.Button(self.control_frame, text="CHẠY SO SÁNH", command=self.cmd_compare)
        self.btn_compare.grid(row=5, column=0, columnspan=2, pady=(5, 15), sticky=tk.EW)

        ttk.Label(self.control_frame, text="CHI TIẾT KẾT QUẢ:", font=("Arial", 12, "bold")).grid(row=6, column=0,
                                                                                                 columnspan=2,
                                                                                                 pady=(15, 5),
                                                                                                 sticky=tk.W)
        self.txt_result = tk.Text(self.control_frame, font=("Courier", 10), height=18, width=42, wrap=tk.WORD)
        self.txt_result.grid(row=7, column=0, columnspan=2, sticky="nsew")

        self.control_frame.grid_rowconfigure(7, weight=1)
        self.control_frame.grid_columnconfigure(0, weight=1)

    def validate_number(self, P):
        return P.isdigit() or P == ""

    # ==========================================
    # CÁC HÀM GIAO TIẾP VỚI MAIN.PY CHỨA DRIVER
    # ==========================================
    def get_inputs(self):
        """Main.py sẽ gọi hàm này để lấy số lượng thành phố và seed"""
        cities_input = self.entry_cities.get().strip()
        if not cities_input.isdigit() or int(cities_input) < 3:
            self.show_error("Vui lòng nhập số nguyên >= 3 cho số thành phố!")
            return None, None
        seed_val = self.entry_seed.get().strip()
        seed = int(seed_val) if seed_val.isdigit() else None
        return int(cities_input), seed

    def update_result_text(self, text):
        """Main.py truyền chuỗi kết quả vào đây để in lên màn hình"""
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, text)

    def show_error(self, message):
        messagebox.showerror("Lỗi", message)