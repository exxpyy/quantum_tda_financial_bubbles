import numpy as np
try:
    import gudhi
except Exception:
    gudhi = None

def rips_simplex_tree(point_cloud: np.ndarray, eps: float, max_dim: int = 2):
    if gudhi is None:
        raise ImportError("gudhi not installed. Install it or use ripser baseline.")
    rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=eps)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    simplices = {k: [] for k in range(max_dim+1)}
    for s, _ in st.get_filtration():
        simplices[len(s)-1].append(tuple(sorted(s)))
    return simplices

def boundary_matrix(simplices: dict, k: int) -> np.ndarray:
    if k == 0:
        return np.zeros((0, len(simplices.get(0, []))), dtype=int)
    k_s  = simplices.get(k, [])
    k1_s = simplices.get(k-1, [])
    idx = {s:i for i,s in enumerate(k1_s)}
    B = np.zeros((len(k1_s), len(k_s)), dtype=int)
    for j, sigma in enumerate(k_s):
        for r in range(len(sigma)):
            tau = tuple([v for t,v in enumerate(sigma) if t != r])
            i = idx.get(tuple(sorted(tau)))
            if i is not None:
                B[i, j] = (-1)**r
    return B

def combinatorial_laplacian(simplices: dict, k: int) -> np.ndarray:
    Bk   = boundary_matrix(simplices, k)
    Bkp1 = boundary_matrix(simplices, k+1)
    return (Bkp1 @ Bkp1.T) + (Bk.T @ Bk)
