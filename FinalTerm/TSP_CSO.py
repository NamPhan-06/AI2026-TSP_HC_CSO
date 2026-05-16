import numpy as np
import time
from TSP_Utils import create_distance_matrix, calculate_total_distance, calculate_memory_usage, calculate_execution_time


class CSO:
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

        self.matrix = None
        self.best_route = None
        self.best_distance = None
        self.initial_route = None
        self.iterations_used = None
        self.time_measured_ms = None
        self.space_measured_bytes = None
        self.space_measured_kb = None
        self.result = ""
        self.history_result = ""
        self.convergence = []

    # ==========================================
    # CÁC HÀM NỘI BỘ (PRIVATE METHODS) CỦA CSO
    # ==========================================
    def _initialize_population(self):
        return np.random.uniform(self.Pmin, self.Pmax, (self.N, self.num_cities)), np.zeros((self.N, self.num_cities))

    def _get_global_best(self, X):
        fit = np.array([self._tsp_fitness(x) for x in X])
        idx = np.argmin(fit)
        return X[idx].copy(), fit[idx]

    def _assign_modes(self):
        num_tracing = int(self.MR * self.N)
        indices = np.arange(self.N)
        np.random.shuffle(indices)
        return indices[:num_tracing], indices[num_tracing:]

    def _seeking_mode(self, Xi):
        num_to_mutate = self.SMP - 1 if self.SPC else self.SMP
        copies = np.tile(Xi, (self.SMP, 1))
        for k in range(num_to_mutate):
            dims_to_change = np.random.choice(self.num_cities, self.CDC, replace=False)
            for d in dims_to_change:
                R = np.random.rand()
                copies[k, d] *= (1 + self.SRD * R) if np.random.rand() < 0.5 else (1 - self.SRD * R)
            copies[k] = np.clip(copies[k], self.Pmin, self.Pmax)

        fits = np.array([self._tsp_fitness(c) for c in copies])
        if np.all(fits == fits[0]):
            prob = np.ones(self.SMP) / self.SMP
        else:
            prob = (np.max(fits) - fits) / (np.max(fits) - np.min(fits) + 1e-10)
            prob = prob / np.sum(prob)
        return copies[np.random.choice(self.SMP, p=prob)]

    def _tracing_mode(self, Xi, Vi, Gbest):
        R = np.random.rand(len(Xi))
        Vi = Vi + self.c1 * R * (Gbest - Xi)
        Xi = np.clip(Xi + Vi, self.Pmin, self.Pmax)
        return Xi, Vi

    def _tsp_fitness(self, x):
        return calculate_total_distance(np.argsort(x), self.matrix)

    # ==========================================
    # HÀM CHẠY CHÍNH LÕI THUẬT TOÁN
    # ==========================================
    def solve(self):
        self.matrix = create_distance_matrix(self.num_cities, seed=self.seed)
        if self.seed is not None:
            np.random.seed(self.seed)

        start_time = time.perf_counter()

        X, V = self._initialize_population()
        Gbest_pos, Gbest_fit = self._get_global_best(X)

        self.initial_route = np.argsort(Gbest_pos).tolist()
        initial_distance = Gbest_fit
        self.convergence = [initial_distance]

        for _ in range(self.max_iter):
            tracing_idx, seeking_idx = self._assign_modes()
            for i in range(self.N):
                if i in seeking_idx:
                    X[i] = self._seeking_mode(X[i])
                else:
                    X[i], V[i] = self._tracing_mode(X[i], V[i], Gbest_pos)

            current_best, current_fit = self._get_global_best(X)
            if current_fit < Gbest_fit:
                Gbest_pos = current_best.copy()
                Gbest_fit = current_fit
            self.convergence.append(Gbest_fit)

        end_time = time.perf_counter()

        self.best_route = np.argsort(Gbest_pos).tolist()
        self.best_distance = Gbest_fit

        # --- Dùng hàm đo thời gian từ TSP_Utils ---
        self.time_measured_ms = calculate_execution_time(start_time, end_time)

        self.space_measured_bytes = calculate_memory_usage(X, V, self.matrix)
        self.space_measured_kb = round(self.space_measured_bytes / 1024.0, 4)

        self._format_output(initial_distance)
        return self.best_route, self.best_distance

    def _format_output(self, initial_distance):
        self.result = f"============== KẾT QUẢ CSO ===============\n"
        self.result += f"Giải pháp ngẫu nhiên đầu: {self.initial_route}\n"
        self.result += f"Q.đường ngẫu nhiên đầu  : {initial_distance}\n\n"
        self.result += f"Giải pháp tốt nhất      : {self.best_route}\n"
        self.result += f"Quãng đường ngắn nhất   : {self.best_distance}\n"
        self.result += f"Số vòng lặp thực hiện   : {self.max_iter}\n"
        self.result += f"Thời gian đo được (ms)  : {self.time_measured_ms}\n"
        self.result += f"Dung lượng đo được (KB) : {self.space_measured_kb}\n"

        self.history_result = f"{'Vòng lặp':<10} | {'Quãng đường'}\n"
        self.history_result += "-" * 28 + "\n"
        for i, val in enumerate(self.convergence):
            self.history_result += f"{i:<10} | {val}\n"