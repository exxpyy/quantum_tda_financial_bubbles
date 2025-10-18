import numpy as np

def takens_embedding(x: np.ndarray, m: int, d: int) -> np.ndarray:

    n = len(x) - d*(m-1)
    if n <= 0:
        raise ValueError("Series too short for given (m, d)")
    return np.stack([x[i:i+n] for i in range(0, d*m, d)], axis=1)

def sliding_windows(X: np.ndarray, w: int):

    if X.shape[0] < w:
        raise ValueError("Window size w too large")
    return [X[i:i+w] for i in range(X.shape[0]-w+1)]
