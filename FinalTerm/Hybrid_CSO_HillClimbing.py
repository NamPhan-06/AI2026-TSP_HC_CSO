import random
import numpy as np
import time
import sys

# =========================
# ===== TSP UTILITIES =====
# =========================

def create_distance_matrix(num_cities, max_distance=300, seed=None):
    if seed is not None:
        random.seed(seed)

    matrix = []
    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            if i == j:
                row.append(0)
            elif j > i:
                val = random.randint(1, max_distance)
                row.append(val)
            else:
                row.append(matrix[j][i])
        matrix.append(row)
    return matrix


def calculate_total_distance(route, matrix):
    total = 0
    for i in range(len(route)):
        total += matrix[route[i - 1]][route[i]]
    return total


def generate_random_route(n):
    route = list(range(n))
    random.shuffle(route)
    return route


# =========================
# ===== HILL CLIMBING =====
# =========================

def generate_neighbors(route):
    neighbors = []
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            new_route = route.copy()
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbors.append(new_route)
    return neighbors


def hill_climbing(route, matrix):
    current = route
    current_dist = calculate_total_distance(current, matrix)

    while True:
        neighbors = generate_neighbors(current)
        best = current
        best_dist = current_dist

        for n in neighbors:
            d = calculate_total_distance(n, matrix)
            if d < best_dist:
                best = n
                best_dist = d

        if best_dist < current_dist:
            current = best
            current_dist = best_dist
        else:
            break

    return current, current_dist


# =========================
# ===== CSO SECTION =======
# =========================

def initialize_population(N, M):
    return np.random.rand(N, M)


def tsp_fitness(x, matrix):
    route = np.argsort(x)
    return calculate_total_distance(route, matrix)


def get_global_best(X, matrix):
    fitness = [tsp_fitness(x, matrix) for x in X]
    idx = np.argmin(fitness)
    return X[idx].copy()


def assign_modes(N, MR):
    num_tracing = int(MR * N)
    tracing = np.random.choice(N, num_tracing, replace=False)
    seeking = [i for i in range(N) if i not in tracing]
    return tracing, seeking


def seeking_mode(x, SMP, SRD, CDC, M):
    copies = np.tile(x, (SMP, 1))
    for k in range(SMP):
        dims = np.random.choice(M, CDC, replace=False)
        for d in dims:
            r = np.random.rand()
            if np.random.rand() < 0.5:
                copies[k][d] *= (1 + SRD * r)
            else:
                copies[k][d] *= (1 - SRD * r)
    return copies[np.random.randint(SMP)]


def tracing_mode(x, gbest, c1, w, v):
    for d in range(len(x)):
        r = np.random.rand()
        v[d] = w * v[d] + c1 * r * (gbest[d] - x[d])
        x[d] = x[d] + v[d]
    return x, v


# =========================
# ===== HYBRID BASIC ======
# =========================

def hybrid_algorithm(matrix, N=20, max_iter=100):
    num_cities = len(matrix)

    X = initialize_population(N, num_cities)
    V = np.zeros((N, num_cities))

    Gbest = get_global_best(X, matrix)

    for _ in range(max_iter):

        tracing, seeking = assign_modes(N, 0.3)

        for i in range(N):
            if i in seeking:
                X[i] = seeking_mode(X[i], 5, 0.2, 2, num_cities)
            else:
                X[i], V[i] = tracing_mode(X[i], Gbest, 2.0, 0.5, V[i])

        Gbest = get_global_best(X, matrix)

    # refine bằng Hill Climbing
    improved = []
    for x in X:
        route = np.argsort(x)
        route, dist = hill_climbing(route, matrix)
        improved.append((route, dist))

    return min(improved, key=lambda x: x[1])


# =========================
# ===== HYBRID ADVANCED ===
# =========================

# =========================
# ===== HYBRID ADVANCED ===
# =========================

def hybrid_algorithm_advanced(matrix, N=30, max_iter=100, hc_rate=0.3):
    # Dời đồng hồ đo thời gian vào hẳn bên trong thuật toán để đo chính xác
    start_time = time.perf_counter()
    num_cities = len(matrix)

    X = initialize_population(N, num_cities)
    V = np.zeros((N, num_cities))

    # --- Lấy giải pháp ngẫu nhiên ban đầu (Gbest của thế hệ 0) ---
    Gbest = get_global_best(X, matrix)
    initial_route = np.argsort(Gbest)
    initial_distance = calculate_total_distance(initial_route, matrix)
    # -----------------------------------------------------------

    convergence = []

    for _ in range(max_iter):
        tracing, seeking = assign_modes(N, 0.3)

        for i in range(N):
            if i in seeking:
                X[i] = seeking_mode(X[i], 5, 0.2, 2, num_cities)
            else:
                X[i], V[i] = tracing_mode(X[i], Gbest, 2.0, 0.5, V[i])

        # Apply Hill Climbing
        for i in range(N):
            if np.random.rand() < hc_rate:
                route = np.argsort(X[i])
                improved, _ = hill_climbing(route, matrix)

                old_x = np.sort(X[i])
                new_x = np.zeros_like(X[i])
                for idx, city in enumerate(improved):
                    new_x[city] = old_x[idx]
                X[i] = new_x

        Gbest = get_global_best(X, matrix)
        convergence.append(tsp_fitness(Gbest, matrix))

    best_route = np.argsort(Gbest)
    best_distance = calculate_total_distance(best_route, matrix)

    # Chốt thời gian kết thúc thuật toán
    end_time = time.perf_counter()
    time_measured_ms = round((end_time - start_time) * 1000.0, 4)

    # --- ĐO LƯỜNG DUNG LƯỢNG ---
    space_bytes = 0
    space_bytes += sys.getsizeof(X) + X.nbytes
    space_bytes += sys.getsizeof(V) + V.nbytes
    space_bytes += sys.getsizeof(matrix)
    for row in matrix:
        space_bytes += sys.getsizeof(row)
        for value in row:
            space_bytes += sys.getsizeof(value)
    space_bytes += sys.getsizeof(best_route) + best_route.nbytes
    space_bytes += sys.getsizeof(Gbest) + Gbest.nbytes
    space_bytes += sys.getsizeof(convergence)
    for val in convergence:
        space_bytes += sys.getsizeof(val)

    # Nạp thêm thời gian vào dict complexity để các hàm khác tiện dùng
    complexity = {
        "time_measured_ms": time_measured_ms,
        "space_measured_bytes": int(space_bytes),
        "space_measured_kb": round(space_bytes / 1024.0, 6)
    }

    # =========================================================
    # --- TẠO CHUỖI RESULT THEO ĐÚNG FORMAT CỦA HILL CLIMBING ---
    # =========================================================
    result_str = ""
    result_str += "=========== KẾT QUẢ HYBRID CSO ===========\n"
    result_str += "Ma trận khoảng cách:\n"
    for row in matrix:
        result_str += str(row) + "\n"

    result_str += "\nGiải pháp ngẫu nhiên đầu tiên là: " + str(initial_route.tolist()) + "\n"
    result_str += "Độ dài quãng đường ngẫu nhiên đầu tiên là: " + str(initial_distance) + "\n"

    result_str += "\nGiải pháp tốt nhất: " + str(best_route.tolist()) + "\n"
    result_str += "Quãng đường ngắn nhất: " + str(best_distance) + "\n"
    result_str += "Số vòng lặp thực hiện: " + str(max_iter) + "\n"
    result_str += "Thời gian đo được (ms): " + str(complexity["time_measured_ms"]) + "\n"
    result_str += "Dung lượng đo được (bytes): " + str(complexity["space_measured_bytes"]) + "\n"
    result_str += "Dung lượng đo được (KB): " + str(complexity["space_measured_kb"]) + "\n"

    # Trả về 5 biến (thêm result_str ở cuối)
    return best_route, best_distance, convergence, complexity, result_str





