import numpy as np

def betti_curve_over_eps(betti_fn, eps_grid):
    return np.array([betti_fn(eps) for eps in eps_grid])

def pairwise_Lp_deltas(curve_seq: np.ndarray, p: int = 2):
    return np.linalg.norm(np.diff(curve_seq, axis=0), ord=p, axis=1)

def detect_spikes(dist, z: float = 2.0):
    mu, sd = dist.mean(), dist.std() + 1e-9
    return np.where((dist - mu) / sd > z)[0]
