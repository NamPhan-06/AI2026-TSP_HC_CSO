import random
import numpy as np
import time
import matplotlib.pyplot as plt
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

def hybrid_algorithm_advanced(matrix, N=30, max_iter=100, hc_rate=0.3):
    num_cities = len(matrix)

    X = initialize_population(N, num_cities)
    V = np.zeros((N, num_cities))

    Gbest = get_global_best(X, matrix)

    convergence = []

    for _ in range(max_iter):
        tracing, seeking = assign_modes(N, 0.3)

        for i in range(N):
            if i in seeking:
                X[i] = seeking_mode(X[i], 5, 0.2, 2, num_cities)
            else:
                X[i], V[i] = tracing_mode(X[i], Gbest, 2.0, 0.5, V[i])

        # 🔥 Apply Hill Climbing trong quá trình
        for i in range(N):
            if np.random.rand() < hc_rate:
                route = np.argsort(X[i])
                improved, _ = hill_climbing(route, matrix)

                # --- SỬA LỖI LOGIC: Cập nhật lại mảng số thực X[i] theo route mới ---
                # Thay vì gán thẳng mảng số nguyên, ta sắp xếp lại mảng số thực X[i] cũ
                # dựa trên thứ tự của improved route để giữ nguyên tính liên tục của CSO
                old_x = np.sort(X[i])  # Lấy các giá trị liên tục đã được sắp xếp
                new_x = np.zeros_like(X[i])
                for idx, city in enumerate(improved):
                    new_x[city] = old_x[idx]
                X[i] = new_x
                # -------------------------------------------------------------------

        Gbest = get_global_best(X, matrix)
        convergence.append(tsp_fitness(Gbest, matrix))

    best_route = np.argsort(Gbest)
    best_distance = calculate_total_distance(best_route, matrix)

    # --- ĐO LƯỜNG DUNG LƯỢNG (Bổ sung mới) ---
    space_bytes = 0

    # 1. Đo kích thước của Quần thể (Vị trí X và Vận tốc V của N con mèo)
    space_bytes += sys.getsizeof(X) + X.nbytes
    space_bytes += sys.getsizeof(V) + V.nbytes

    # 2. Đo kích thước của Ma trận khoảng cách
    space_bytes += sys.getsizeof(matrix)
    for row in matrix:
        space_bytes += sys.getsizeof(row)
        for value in row:
            space_bytes += sys.getsizeof(value)

    # 3. Đo kích thước của Route tốt nhất và Gbest
    space_bytes += sys.getsizeof(best_route) + best_route.nbytes
    space_bytes += sys.getsizeof(Gbest) + Gbest.nbytes

    # 4. Đo mảng hội tụ (convergence curve)
    space_bytes += sys.getsizeof(convergence)
    for val in convergence:
        space_bytes += sys.getsizeof(val)

    # Đóng gói thông số phức tạp thành Dictionary
    complexity = {
        "space_measured_bytes": int(space_bytes),
        "space_measured_kb": round(space_bytes / 1024.0, 6)
    }
    # -----------------------------------------

    # TRẢ VỀ THÊM DICTIONARY COMPLEXITY
    return best_route, best_distance, convergence, complexity

# =========================
# ===== EVALUATION ========
# =========================

def run_hill_climbing_only(num_cities, seed=None):
    matrix = create_distance_matrix(num_cities, seed=seed)
    route = generate_random_route(num_cities)

    start = time.perf_counter()
    route, dist = hill_climbing(route, matrix)
    end = time.perf_counter()

    return dist, (end - start) * 1000


def run_hybrid(num_cities, seed=None):
    matrix = create_distance_matrix(num_cities, seed=seed)

    start = time.perf_counter()
    route, dist, conv = hybrid_algorithm_advanced(matrix)
    end = time.perf_counter()

    # Only return distance and time for evaluation, as evaluate_algorithm expects 2 values.
    return dist, (end - start) * 1000


def evaluate_algorithm(func, runs=10):
    distances = []
    times = []

    for _ in range(runs):
        d, t = func()
        distances.append(d)
        times.append(t)

    return {
        "best": np.min(distances),
        "mean": np.mean(distances),
        "std": np.std(distances),
        "time_mean": np.mean(times)
    }


# =========================
# ===== VISUALIZATION =====
# =========================

def plot_convergence(curve, title):
    plt.figure()
    plt.plot(curve)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.grid()
    plt.show()


def plot_route(route):
    coords = np.random.rand(len(route), 2)
    ordered = coords[route]

    plt.figure()
    plt.plot(ordered[:, 0], ordered[:, 1], marker='o')
    plt.title("Best Route Visualization")
    plt.show()


# =========================
# ===== TEST CASES ========
# =========================

def run_case(size):
    print(f"\n===== SIZE {size} =====")

    hc = evaluate_algorithm(lambda: run_hill_climbing_only(size), runs=5)
    hybrid = evaluate_algorithm(lambda: run_hybrid(size), runs=5)

    print("Hill Climbing:", hc)
    print("Hybrid CSO+HC:", hybrid)




