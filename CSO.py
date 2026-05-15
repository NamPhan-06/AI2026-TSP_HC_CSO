import numpy as np
import time
import sys
from TSP_HillClimbing import create_distance_matrix


# ==========================================
# CÁC HÀM PHỤ TRỢ (Giữ nguyên bên ngoài Class cho nhẹ)
# ==========================================
def calculate_total_distance(route, matrix):
    total = 0
    for i in range(len(route)):
        total += matrix[route[i - 1]][route[i]]
    return total


def initialize_population(N, M, Pmin, Pmax):
    X = np.random.uniform(Pmin, Pmax, (N, M))
    V = np.zeros((N, M))
    return X, V


def get_global_best(X, fitness_func):
    fit = np.array([fitness_func(x) for x in X])
    idx = np.argmin(fit)
    return X[idx].copy(), fit[idx]


def assign_modes(N, MR):
    num_tracing = int(MR * N)
    indices = np.arange(N)
    np.random.shuffle(indices)
    tracing_idx = indices[:num_tracing]
    seeking_idx = indices[num_tracing:]
    return tracing_idx, seeking_idx


def seeking_mode(Xi, SMP, SRD, CDC, M, fitness_func, Pmin, Pmax, SPC):
    num_to_mutate = SMP - 1 if SPC else SMP
    copies = np.tile(Xi, (SMP, 1))
    for k in range(num_to_mutate):
        dims_to_change = np.random.choice(M, CDC, replace=False)
        for d in dims_to_change:
            R = np.random.rand()
            if np.random.rand() < 0.5:
                copies[k, d] *= (1 + SRD * R)
            else:
                copies[k, d] *= (1 - SRD * R)
        copies[k] = np.clip(copies[k], Pmin, Pmax)

    fits = np.array([fitness_func(c) for c in copies])
    if np.all(fits == fits[0]):
        prob = np.ones(SMP) / SMP
    else:
        f_max = np.max(fits)
        f_min = np.min(fits)
        prob = (f_max - fits) / (f_max - f_min + 1e-10)
        prob = prob / np.sum(prob)

    idx = np.random.choice(SMP, p=prob)
    return copies[idx]


def tracing_mode(Xi, Vi, Gbest, c1, Pmin, Pmax):
    M = len(Xi)
    for d in range(M):
        R = np.random.rand()
        Vi[d] = Vi[d] + c1 * R * (Gbest[d] - Xi[d])
        Xi[d] = Xi[d] + Vi[d]
    Xi = np.clip(Xi, Pmin, Pmax)
    return Xi, Vi


# ==========================================
# CLASS CHÍNH ĐỂ GỌI BÊN NGOÀI
# ==========================================
class CSO:
    # Gói toàn bộ tham số vào hàm khởi tạo với các giá trị mặc định chuẩn
    def __init__(self, num_cities, seed=None, N=30, max_iter=100, MR=0.3, SMP=5, SRD=0.2, CDC=2, SPC=True, c1=2.0,
                 Pmin=-10, Pmax=10):
        self.num_cities = num_cities
        self.seed = seed
        self.N = N
        self.max_iter = max_iter
        self.MR = MR
        self.SMP = SMP
        self.SRD = SRD
        self.CDC = CDC
        self.SPC = SPC
        self.c1 = c1
        self.Pmin = Pmin
        self.Pmax = Pmax

        # Các thuộc tính kết quả (Đặt tên chuẩn y hệt Hill Climbing)
        self.matrix = None
        self.best_route = None
        self.best_distance = None
        self.time_measured_ms = None
        self.space_measured_kb = None
        self.space_measured_bytes = None
        self.result = ""  # Chuỗi text xuất ra màn hình
        self.convergence = []

    def solve(self):
        # 1. Tự động khởi tạo ma trận khoảng cách giống hệt HC
        self.matrix = create_distance_matrix(self.num_cities, seed=self.seed)

        # Đồng bộ seed cho numpy nếu có
        if self.seed is not None:
            np.random.seed(self.seed)

        start_time = time.perf_counter()
        M_dims = self.num_cities  # Số chiều bằng số thành phố

        def tsp_fitness(x):
            route = np.argsort(x)
            return calculate_total_distance(route, self.matrix)

        X, V = initialize_population(self.N, M_dims, self.Pmin, self.Pmax)
        Gbest_pos, Gbest_fit = get_global_best(X, tsp_fitness)

        initial_route = np.argsort(Gbest_pos).tolist()
        initial_distance = Gbest_fit

        self.convergence = [initial_distance]

        # Vòng lặp chính
        for _ in range(self.max_iter):
            tracing_idx, seeking_idx = assign_modes(self.N, self.MR)
            for i in range(self.N):
                if i in seeking_idx:
                    X[i] = seeking_mode(X[i], self.SMP, self.SRD, self.CDC, M_dims, tsp_fitness, self.Pmin, self.Pmax,
                                        self.SPC)
                else:
                    X[i], V[i] = tracing_mode(X[i], V[i], Gbest_pos, self.c1, self.Pmin, self.Pmax)

            current_best, current_fit = get_global_best(X, tsp_fitness)
            if current_fit < Gbest_fit:
                Gbest_pos = current_best.copy()
                Gbest_fit = current_fit
            self.convergence.append(Gbest_fit)
        end_time = time.perf_counter()

        # Lưu lại kết quả
        self.best_route = np.argsort(Gbest_pos).tolist()
        self.best_distance = Gbest_fit
        self.time_measured_ms = round((end_time - start_time) * 1000.0, 4)

        # Đo bộ nhớ
        space_bytes = sys.getsizeof(X) + X.nbytes + sys.getsizeof(V) + V.nbytes + sys.getsizeof(self.matrix)
        self.space_measured_kb = round(space_bytes / 1024.0, 4)

        # Tạo chuỗi text
        self.result = ""
        self.result += "============== KẾT QUẢ CSO ===============\n"
        self.result += "Giải pháp ngẫu nhiên đầu tiên: " + str(initial_route) + "\n"
        self.result += "Q.đường ngẫu nhiên đầu tiên: " + str(initial_distance) + "\n\n"
        self.result += "Giải pháp tốt nhất      : " + str(self.best_route) + "\n"
        self.result += "Quãng đường ngắn nhất   : " + str(self.best_distance) + "\n"
        self.result += "Số vòng lặp thực hiện   : " + str(self.max_iter) + "\n"
        self.result += "Thời gian đo được (ms)  : " + str(self.time_measured_ms) + "\n"
        self.result += "Dung lượng đo được (KB) : " + str(self.space_measured_kb) + "\n"

        # TẠO LỊCH SỬ HỘI TỤ CHUẨN FORMAT BẢNG
        self.history_result = f"{'Vòng lặp':<10} | {'Quãng đường'}\n"
        self.history_result += "-" * 30 + "\n"
        for i, val in enumerate(self.convergence):
            self.history_result += f"{i:<10} | {val}\n"

        return self.best_route, self.best_distance