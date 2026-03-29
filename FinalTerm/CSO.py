# Cat swarm optimization
# N là số con mèo, M là số chiều trong không gian, max_iter là số loop
# MR là tỉ lệ con mèo ở tracing mode, 1-MR là tỉ lệ con mèo seeking
# SMP là số lần thử tìm kiếm của 1 con mèo
# SRD là tỉ lệ mở rộng phạm vi tìm kiếm quanh vị trí ban đầu
# CDC là số chiều thay đổi
# Pmin và Pmax để giới hạn giá trị fitness không bị lạc ra quá lớn
import numpy as np

# giả sử hàm mặt cầu là hàm mục
def sphere_function(x):
    return np.sum(x**2)

def initialize_population(N, M, Pmin, Pmax):
    X = np.random.uniform(Pmin, Pmax, (N, M))
    V = np.zeros((N, M))
    return X, V

def get_global_best(X, fitness_func):
    fit = np.array([fitness_func(x) for x in X])
    idx = np.argmin(fit)
    return X[idx].copy()

def assign_modes(N, MR):
    num_tracing = int(MR * N)
    tracing_idx = np.random.choice(N, num_tracing, replace=False)
    seeking_idx = [i for i in range(N) if i not in tracing_idx]
    return tracing_idx, seeking_idx

def seeking_mode(Xi, SMP, SRD, CDC, M, fitness_func, Pmin, Pmax):
    copies = np.tile(Xi, (SMP, 1))
    for k in range(SMP):
        dims = np.random.choice(M, CDC, replace=False)
        for d in dims:
            R = np.random.rand()
            if np.random.rand() < 0.5:
                copies[k, d] *= (1 + SRD * R)
            else:
                copies[k, d] *= (1 - SRD * R)
        copies[k] = np.clip(copies[k], Pmin, Pmax)
    fits = np.array([fitness_func(c) for c in copies])
    prob = 1 / (fits + 1e-10)
    prob = prob / np.sum(prob)
    idx = np.random.choice(len(copies), p=prob)
    return copies[idx]

def tracing_mode(Xi, Vi, Gbest, c1, w, Pmin, Pmax):
    M = len(Xi)
    for d in range(M):
        R = np.random.rand()
        Vi[d] = w * Vi[d] + c1 * R * (Gbest[d] - Xi[d])
        Xi[d] = Xi[d] + Vi[d]
    Xi = np.clip(Xi, Pmin, Pmax)
    return Xi, Vi

def cat_swarm_optimization(N, M, max_iter, MR, SMP, SRD, CDC, c1, w, Pmin, Pmax, fitness_func):
    X, V = initialize_population(N, M, Pmin, Pmax)
    Gbest = get_global_best(X, fitness_func)
    for _ in range(max_iter):
        tracing_idx, seeking_idx = assign_modes(N, MR)
        for i in range(N):
            if i in seeking_idx:
                X[i] = seeking_mode(X[i], SMP, SRD, CDC, M, fitness_func, Pmin, Pmax)
            else:
                X[i], V[i] = tracing_mode(X[i], V[i], Gbest, c1, w, Pmin, Pmax)
        Gbest = get_global_best(X, fitness_func)
    return Gbest, fitness_func(Gbest)

best_pos, best_fit = cat_swarm_optimization(
    N=30, M=2, max_iter=200,
    MR=0.3, SMP=10, SRD=0.1, CDC=1,
    c1=2.0, w=0.5,
    Pmin=-10, Pmax=10,
    fitness_func=sphere_function) #nhập hàm mục tiêu vào fitness_function

print("Best solution:", best_pos)
print("Best fitness:", best_fit)