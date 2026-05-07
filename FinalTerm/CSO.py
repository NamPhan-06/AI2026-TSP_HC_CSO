import numpy as np


# Giả sử hàm mặt cầu là hàm mục tiêu
def sphere_function(x):
    return np.sum(x ** 2)


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
            # Công thức: X_cn = X_c * (1 +/- SRD * R)
            if np.random.rand() < 0.5:
                copies[k, d] *= (1 + SRD * R)
            else:
                copies[k, d] *= (1 - SRD * R)
        copies[k] = np.clip(copies[k], Pmin, Pmax)

    fits = np.array([fitness_func(c) for c in copies])

    if np.all(fits == fits[0]):
        prob = np.ones(SMP) / SMP
    else:
        # Công thức (FSmax - FSi) / (FSmax - FSmin)
        f_max = np.max(fits)
        f_min = np.min(fits)
        prob = (f_max - fits) / (f_max - f_min + 1e-10)
        prob = prob / np.sum(prob)

    idx = np.random.choice(SMP, p=prob)
    return copies[idx]


# --- ĐÃ XÓA w Ở ĐÂY, TRỞ VỀ CÔNG THỨC GỐC ---
def tracing_mode(Xi, Vi, Gbest, c1, Pmin, Pmax):
    # v_{i,d} = v_{i,d} + R * c1 * (Gbest_d - Xi_d)
    R = np.random.rand(*Xi.shape)
    Vi = Vi + R * c1 * (Gbest - Xi)

    # X_{i,d,new} = X_{i,d,old} + v_{i,d}
    Xi = Xi + Vi
    Xi = np.clip(Xi, Pmin, Pmax)
    return Xi, Vi


# --- ĐÃ XÓA w KHỎI THAM SỐ HÀM CHÍNH ---
def cat_swarm_optimization(N, M, max_iter, MR, SMP, SRD, CDC, c1, Pmin, Pmax, fitness_func, SPC):
    X, V = initialize_population(N, M, Pmin, Pmax)
    Gbest, _ = get_global_best(X, fitness_func)

    for _ in range(max_iter):
        tracing_idx, seeking_idx = assign_modes(N, MR)
        for i in range(N):
            if i in seeking_idx:
                X[i] = seeking_mode(X[i], SMP, SRD, CDC, M, fitness_func, Pmin, Pmax, SPC)
            else:
                X[i], V[i] = tracing_mode(X[i], V[i], Gbest, c1, Pmin, Pmax)

        # Cập nhật Gbest (Trí nhớ lịch sử)
        current_best, current_fit = get_global_best(X, fitness_func)
        if current_fit < fitness_func(Gbest):
            Gbest = current_best.copy()

    return Gbest, fitness_func(Gbest)


# Chạy thử
best_pos, best_fit = cat_swarm_optimization(
    N=50, M=2, max_iter=200,
    MR=0.35, SMP=20, SRD=0.3, CDC=1, SPC=True,
    c1=2.0,
    Pmin=-10, Pmax=10,
    fitness_func=sphere_function)

print(f"Toạ độ tốt nhất: {best_pos}")
print(f"Giá trị tối ưu: {best_fit}")