import random
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# TSP FUNCTIONS

def generate_cities(num_cities, seed=None):
    if seed is not None:
        random.seed(seed)
    return list(range(num_cities))


def distance(cityA, cityB):
    return abs(cityA - cityB)


def create_distance_matrix(num_cities, max_distance=300, seed=None):
    if seed is not None:
        random.seed(seed)

    distance_matrix = []

    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            if i == j:
                row.append(0)
            elif j > i:
                row.append(random.randint(1, max_distance))
            else:
                row.append(distance_matrix[j][i])
        distance_matrix.append(row)

    return distance_matrix


def calculate_total_distance(route, distance_matrix):
    total_length = 0
    for i in range(len(route)):
        total_length += distance_matrix[route[i - 1]][route[i]]
    return total_length


def generate_random_route(num_cities):
    cities = list(range(num_cities))
    route = []

    for _ in range(num_cities):
        random_city = random.choice(cities)
        route.append(random_city)
        cities.remove(random_city)

    return route


def generate_neighbors(route):
    neighbors = []

    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            neighbor = route.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)

    return neighbors


def get_best_neighbor(distance_matrix, neighbors):
    best_neighbor = neighbors[0]
    best_distance = calculate_total_distance(best_neighbor, distance_matrix)

    for neighbor in neighbors:
        current_distance = calculate_total_distance(neighbor, distance_matrix)
        if current_distance < best_distance:
            best_neighbor = neighbor
            best_distance = current_distance

    return best_neighbor, best_distance


# HILL CLIMBING

def hill_climbing(distance_matrix, max_iterations=1000):
    num_cities = len(distance_matrix)

    current_route = generate_random_route(num_cities)
    current_distance = calculate_total_distance(current_route, distance_matrix)

    iterations = 0

    while iterations < max_iterations:
        neighbors = generate_neighbors(current_route)
        best_neighbor, best_neighbor_distance = get_best_neighbor(distance_matrix, neighbors)

        if best_neighbor_distance < current_distance:
            current_route = best_neighbor
            current_distance = best_neighbor_distance
            iterations += 1
        else:
            break

    return current_route, current_distance, iterations


def estimate_complexity(num_cities, iterations_used, runtime_seconds, distance_matrix, route, neighbors):
    """
    In ra số đo cụ thể với đơn vị chuẩn:
    - Thời gian: milliseconds (ms)
    - Dung lượng: bytes (B)

    Gồm 2 loại:
    1) time_measured_ms: thời gian chạy thực tế
    2) space_measured_bytes: bộ nhớ ước lượng theo cấu trúc dữ liệu đang giữ
    """
    time_measured_ms = runtime_seconds * 1000.0

    # Ước lượng bộ nhớ đang dùng cho các cấu trúc chính
    # distance_matrix: danh sách các hàng
    space_measured_bytes = sys.getsizeof(distance_matrix)
    for row in distance_matrix:
        space_measured_bytes += sys.getsizeof(row)
        for value in row:
            space_measured_bytes += sys.getsizeof(value)

    space_measured_bytes += sys.getsizeof(route)
    for city in route:
        space_measured_bytes += sys.getsizeof(city)

    space_measured_bytes += sys.getsizeof(neighbors)
    for neigh in neighbors:
        space_measured_bytes += sys.getsizeof(neigh)
        for city in neigh:
            space_measured_bytes += sys.getsizeof(city)

    return {
        "time_measured_ms": round(time_measured_ms, 6),
        "space_measured_bytes": int(space_measured_bytes),
        "space_measured_kb": round(space_measured_bytes / 1024.0, 6)
    }


def run_hill_climbing(num_cities=10, seed=None, max_iterations=1000):
    cities = generate_cities(num_cities, seed=seed)
    distance_matrix = create_distance_matrix(num_cities, seed=seed)
    initial_route = generate_random_route(num_cities)

    start_time = time.perf_counter()
    best_route, best_distance, iterations_used = hill_climbing(
        distance_matrix=distance_matrix,
        max_iterations=max_iterations
    )
    end_time = time.perf_counter()

    # dùng route và neighbors cuối để ước lượng bộ nhớ
    final_neighbors = generate_neighbors(best_route)
    complexity = estimate_complexity(
        num_cities=num_cities,
        iterations_used=iterations_used,
        runtime_seconds=end_time - start_time,
        distance_matrix=distance_matrix,
        route=best_route,
        neighbors=final_neighbors
    )

    return {
        "cities": cities,
        "distance_matrix": distance_matrix,
        "initial_route": initial_route,
        "best_route": best_route,
        "best_distance": best_distance,
        "iterations_used": iterations_used,
        "time_measured_ms": complexity["time_measured_ms"],
        "space_measured_bytes": complexity["space_measured_bytes"],
        "space_measured_kb": complexity["space_measured_kb"]
    }


# BEST / BASE / WORST CASE

def run_best_case():
    result = run_hill_climbing(num_cities=5, seed=1, max_iterations=100)
    result["case_name"] = "best_case"
    return result


def run_base_case():
    result = run_hill_climbing(num_cities=10, seed=2, max_iterations=500)
    result["case_name"] = "base_case"
    return result


def run_worst_case():
    result = run_hill_climbing(num_cities=30, seed=3, max_iterations=1000)
    result["case_name"] = "worst_case"
    return result


def print_case_report(case_result):
    print("Case:", case_result["case_name"])
    print("Best route:", case_result["best_route"])
    print("Best distance:", case_result["best_distance"])
    print("Iterations used:", case_result["iterations_used"])
    print("Time measured (ms):", case_result["time_measured_ms"])
    print("Space measured (bytes):", case_result["space_measured_bytes"])
    print("Space measured (KB):", case_result["space_measured_kb"])
    print("-" * 60)


# CLASS

class HillClimbing:
    def __init__(self, num_cities, seed=None):
        self.matrix = None
        self.result = None
        self.num_cities = num_cities
        self.seed = seed
        self.best_route = None
        self.best_distance = None
        self.initial_route = None
        self.iterations_used = None
        self.time_measured_ms = None
        self.space_measured_bytes = None
        self.space_measured_kb = None

    def create_plot(self):
        if self.matrix is None:
            print("Chưa có ma trận khoảng cách.")
            return

        distance = np.array(self.matrix)
        G = nx.from_numpy_array(distance)

        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_color="red", node_size=500)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        plt.title("Đồ thị TSP")
        plt.axis("off")
        plt.show()

    def solve(self):
        if self.seed is not None:
            random.seed(self.seed)

        self.matrix = create_distance_matrix(self.num_cities, seed=self.seed)

        self.initial_route = generate_random_route(self.num_cities)
        initial_distance = calculate_total_distance(self.initial_route, self.matrix)

        neighbors = generate_neighbors(self.initial_route)
        best_neighbor, best_neighbor_distance = get_best_neighbor(self.matrix, neighbors)

        current_route = self.initial_route
        current_distance = initial_distance
        iterations = 0

        start_time = time.perf_counter()

        while best_neighbor_distance < current_distance:
            current_route = best_neighbor
            current_distance = best_neighbor_distance

            neighbors = generate_neighbors(current_route)
            best_neighbor, best_neighbor_distance = get_best_neighbor(self.matrix, neighbors)

            iterations += 1

        end_time = time.perf_counter()

        self.best_route = current_route
        self.best_distance = current_distance
        self.iterations_used = iterations

        self.time_measured_ms = round((end_time - start_time) * 1000.0, 6)

        final_neighbors = generate_neighbors(self.best_route)
        space_bytes = sys.getsizeof(self.matrix)
        for row in self.matrix:
            space_bytes += sys.getsizeof(row)
            for value in row:
                space_bytes += sys.getsizeof(value)

        space_bytes += sys.getsizeof(self.best_route)
        for city in self.best_route:
            space_bytes += sys.getsizeof(city)

        space_bytes += sys.getsizeof(final_neighbors)
        for neigh in final_neighbors:
            space_bytes += sys.getsizeof(neigh)
            for city in neigh:
                space_bytes += sys.getsizeof(city)

        self.space_measured_bytes = int(space_bytes)
        self.space_measured_kb = round(space_bytes / 1024.0, 6)

        self.result = ""
        self.result += "=============== KẾT QUẢ HC ===============\n"
        self.result += "Ma trận khoảng cách:\n"
        for row in self.matrix:
            self.result += str(row) + "\n"

        self.result += "\nGiải pháp ngẫu nhiên đầu tiên là: " + str(self.initial_route) + "\n"
        self.result += "Độ dài quãng đường ngẫu nhiên đầu tiên là: " + str(initial_distance) + "\n"

        self.result += "\nGiải pháp tốt nhất: " + str(self.best_route) + "\n"
        self.result += "Quãng đường ngắn nhất: " + str(self.best_distance) + "\n"
        self.result += "Số vòng lặp thực hiện: " + str(self.iterations_used) + "\n"
        self.result += "Thời gian đo được (ms): " + str(self.time_measured_ms) + "\n"
        self.result += "Dung lượng đo được (bytes): " + str(self.space_measured_bytes) + "\n"
        self.result += "Dung lượng đo được (KB): " + str(self.space_measured_kb) + "\n"

        return self.best_route, self.best_distance

