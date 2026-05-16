import random
import sys

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
                row.append(random.randint(1, max_distance))
            else:
                row.append(matrix[j][i])
        matrix.append(row)
    return matrix

def calculate_total_distance(route, matrix):
    return sum(matrix[route[i - 1]][route[i]] for i in range(len(route)))

def generate_random_route(num_cities):
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def calculate_memory_usage(*objects):
    """Hàm tiện ích đo tổng dung lượng bộ nhớ của các object truyền vào"""
    total_bytes = 0
    for obj in objects:
        total_bytes += sys.getsizeof(obj)
        # Nếu là ma trận (list of lists)
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
            total_bytes += sum(sys.getsizeof(row) + sum(sys.getsizeof(item) for item in row) for row in obj)
        # Nếu là list bình thường 1 chiều
        elif isinstance(obj, list):
            total_bytes += sum(sys.getsizeof(item) for item in obj)
        # Bổ sung tính toán byte mảng numpy nếu có thuộc tính nbytes
        elif hasattr(obj, 'nbytes'):
            total_bytes += obj.nbytes
    return total_bytes

def calculate_execution_time(start_time, end_time, decimals=4):
    """Trả về thời gian thực thi tính bằng milliseconds (ms)"""
    return round((end_time - start_time) * 1000.0, decimals)