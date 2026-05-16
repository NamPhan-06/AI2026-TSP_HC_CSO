import random
import time
from TSP_Utils import create_distance_matrix, calculate_total_distance, generate_random_route, calculate_memory_usage, \
    calculate_execution_time


class HillClimbing:
    def __init__(self, num_cities, seed=None, max_iterations=100):
        self.num_cities = num_cities
        self.seed = seed
        self.max_iterations = max_iterations

        self.matrix = None
        self.best_route = None
        self.best_distance = None
        self.initial_route = None
        self.iterations_used = None
        self.time_measured_ms = None
        self.space_measured_bytes = None
        self.space_measured_kb = None
        self.convergence = []
        self.result = ""
        self.history_result = ""

    # ==========================================
    # CÁC HÀM NỘI BỘ (PRIVATE METHODS) CỦA HC
    # ==========================================
    def _generate_neighbors(self, route):
        neighbors = []
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                neighbor = route.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    def _get_best_neighbor(self, neighbors):
        best_neighbor = neighbors[0]
        # Dùng thẳng self.matrix, không cần truyền tham số
        best_distance = calculate_total_distance(best_neighbor, self.matrix)
        for neighbor in neighbors:
            current_distance = calculate_total_distance(neighbor, self.matrix)
            if current_distance < best_distance:
                best_neighbor = neighbor
                best_distance = current_distance
        return best_neighbor, best_distance

    # ==========================================
    # HÀM CHẠY CHÍNH LÕI THUẬT TOÁN
    # ==========================================
    def solve(self):
        if self.seed is not None:
            random.seed(self.seed)

        self.matrix = create_distance_matrix(self.num_cities, seed=self.seed)
        self.initial_route = generate_random_route(self.num_cities)
        initial_distance = calculate_total_distance(self.initial_route, self.matrix)

        current_route = self.initial_route
        current_distance = initial_distance

        start_time = time.perf_counter()
        self.convergence = [initial_distance]
        is_stuck = False

        for _ in range(self.max_iterations):
            if not is_stuck:
                # Gọi các hàm nội bộ bằng self.
                neighbors = self._generate_neighbors(current_route)
                best_neighbor, best_neighbor_distance = self._get_best_neighbor(neighbors)

                if best_neighbor_distance < current_distance:
                    current_route = best_neighbor
                    current_distance = best_neighbor_distance
                else:
                    is_stuck = True
            self.convergence.append(current_distance)

        end_time = time.perf_counter()

        self.best_route = current_route
        self.best_distance = current_distance
        self.iterations_used = self.max_iterations

        # --- Dùng hàm đo thời gian từ TSP_Utils ---
        self.time_measured_ms = calculate_execution_time(start_time, end_time)

        final_neighbors = self._generate_neighbors(self.best_route)
        self.space_measured_bytes = calculate_memory_usage(self.matrix, self.best_route, final_neighbors)
        self.space_measured_kb = round(self.space_measured_bytes / 1024.0, 4)

        self._format_output(initial_distance)
        return self.best_route, self.best_distance

    def _format_output(self, initial_distance):
        self.result = f"=============== KẾT QUẢ HC ===============\n"
        self.result += f"Giải pháp ngẫu nhiên đầu: {self.initial_route}\n"
        self.result += f"Q.đường ngẫu nhiên đầu  : {initial_distance}\n\n"
        self.result += f"Giải pháp tốt nhất      : {self.best_route}\n"
        self.result += f"Quãng đường ngắn nhất   : {self.best_distance}\n"
        self.result += f"Số vòng lặp thực hiện   : {self.iterations_used}\n"
        self.result += f"Thời gian đo được (ms)  : {self.time_measured_ms}\n"
        self.result += f"Dung lượng đo được (KB) : {self.space_measured_kb}\n"

        self.history_result = f"{'Vòng lặp':<10} | {'Quãng đường'}\n"
        self.history_result += "-" * 28 + "\n"
        for i, val in enumerate(self.convergence):
            self.history_result += f"{i:<10} | {val}\n"