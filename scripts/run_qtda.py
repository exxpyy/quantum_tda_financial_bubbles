
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from src.takens import takens_embedding, sliding_windows
from src.betti_curves import betti_curve_over_eps, pairwise_Lp_deltas, detect_spikes


from ripser import ripser
try:
    from src.rips_laplacian import rips_simplex_tree, combinatorial_laplacian
    HAVE_GUDHI = True
except Exception:
    HAVE_GUDHI = False

try:
    from src.qpe import pad_and_rescale, unitary_from_laplacian, qpe_zero_prob
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False


def classical_betti(point_cloud: np.ndarray, eps: float, dim: int) -> int:

    res = ripser(point_cloud, maxdim=dim)
    diagrams = res["dgms"]

    return sum(1 for interval in diagrams[dim] if interval[0] < eps < interval[1])


def qpe_betti(L_k: np.ndarray, shots: int = 1024) -> int:

    if not HAVE_QISKIT:
        raise RuntimeError("Qiskit/Aer not available for QPE demo.")
    Lp = pad_and_rescale(L_k)         
    U  = unitary_from_laplacian(Lp)
    n  = Lp.shape[0]
    p0 = qpe_zero_prob(U, n_target=int(np.ceil(np.log2(n))), shots=shots)

    return int(round(p0 * n))

def load_dates_and_prices(csv_path: str, price_pref: str | None = "close"):
    import pandas as pd, numpy as np

    try:
        df = pd.read_csv(csv_path)
        has_header = isinstance(df.columns[0], str)
    except Exception:
        has_header = False

    if not has_header:

        df = pd.read_csv(csv_path, header=None, names=["date", "price"])


    date_col_candidates = ["date", "Date", "DATE", "timestamp", "Timestamp"]
    date_col = next((c for c in df.columns if str(c).strip() in date_col_candidates), None)
    dates = None
    if date_col is not None:

        dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")


    if price_pref and price_pref in df.columns:
        prices = np.asarray(df[price_pref], dtype=float)
    else:
        for cand in ["close","Close","Adj Close","adj close","price","Price"]:
            if cand in df.columns:
                prices = np.asarray(df[cand], dtype=float)
                break
        else:

            prices = np.asarray(df.iloc[:, -1], dtype=float)

    return dates, prices


def main():
    ap = argparse.ArgumentParser(description="Quantum TDA for Bubble Detection (no notebooks)")
    ap.add_argument("--csv", default="data/sp500.csv", help="CSV with 'close' or price column")
    ap.add_argument("--price-col", default="close", help="Column name for prices")
    ap.add_argument("--m", type=int, default=4, help="Takens embedding dimension")
    ap.add_argument("--d", type=int, default=5, help="Takens delay")
    ap.add_argument("--w", type=int, default=50, help="Sliding window size")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05,0.07,0.1,0.12],
                    help="Epsilon grid for VR complexes / Betti curves")
    ap.add_argument("--k", type=int, default=0, help="Betti dimension (0 or 1 typical)")
    ap.add_argument("--lp", type=int, default=2, help="L^p for pairwise distances")
    ap.add_argument("--z", type=float, default=2.0, help="Z-threshold for spike detection")
    ap.add_argument("--save-plot", action="store_true", help="Save plots to plots/ directory")
    args = ap.parse_args()

    dates, prices = load_dates_and_prices(args.csv, price_pref=args.price_col)
    x = np.log(prices.astype(float))


    X = takens_embedding(x, m=args.m, d=args.d)
    windows = sliding_windows(X, w=args.w)
    if len(windows) < 2:
        raise ValueError("Not enough windows. Adjust m/d/w or provide longer series.")


        
    n_embed = X.shape[0]
    if dates is not None:
        dates_embedded = dates.iloc[args.d * (args.m - 1):].reset_index(drop=True)
       
        center_offsets = np.arange(0, n_embed - args.w + 1) + (args.w // 2)
        window_dates = dates_embedded.iloc[center_offsets].to_numpy()
    else:
        window_dates = None


    curves = []
    for pc in windows:

        curve = []
        for eps in args.eps:
            if HAVE_GUDHI:

                simplices = rips_simplex_tree(pc, eps=eps, max_dim=max(args.k,1))

            b = classical_betti(pc, eps, dim=args.k)
            curve.append(b)
        curves.append(curve)

    curves = np.array(curves)  
    dist = pairwise_Lp_deltas(curves, p=args.lp)
    spike_idx = detect_spikes(dist, z=args.z)

    print(f"[info] windows: {len(windows)}, eps grid: {args.eps}")
    print(f"[info] detected spike indices (potential crash on following window): {spike_idx.tolist()}")


    if HAVE_GUDHI and HAVE_QISKIT:
        pc0 = windows[len(windows)//2]
        eps0 = args.eps[min(1, len(args.eps)-1)]
        simplices0 = rips_simplex_tree(pc0, eps=eps0, max_dim=max(args.k,1))
        Lk = combinatorial_laplacian(simplices0, k=args.k)
        try:
            b_qpe = qpe_betti(Lk, shots=1024)
            print(f"[qpe] demo betti estimate at eps={eps0}: {b_qpe}")
        except Exception as e:
            print(f"[qpe] skipped ({e})")


    if args.save_plot:
        Path("plots").mkdir(exist_ok=True)

 
        fig, ax = plt.subplots()
        for j in range(curves.shape[1]):
            x_axis = window_dates if window_dates is not None else np.arange(len(curves))
            ax.plot(x_axis, curves[:, j], label=f"betti@eps={args.eps[j]}")

        ax.set_title("Betti curves over windows")
        ax.set_ylabel("Betti₀ (connected components)")
        ax.set_xlabel("Date" if window_dates is not None else "Window index")
        ax.legend()

        if window_dates is not None:
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))     
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))      
            fig.autofmt_xdate()

        out1 = "plots/betti_curves.png"
        fig.savefig(out1, dpi=160)
        print(f"[save] {out1}")

  
        fig, ax = plt.subplots()
        x_axis_d = window_dates[1:] if window_dates is not None else np.arange(len(dist))
        ax.plot(x_axis_d, dist)
        if spike_idx.size:
            ax.scatter(x_axis_d[spike_idx], dist[spike_idx], marker="x")

        ax.set_title(f"Pairwise L^{args.lp} deltas (spikes flagged)")
        ax.set_ylabel("Δ (topology change)")
        ax.set_xlabel("Date" if window_dates is not None else "Window index")

        if window_dates is not None:
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            fig.autofmt_xdate()

        out2 = "plots/crash_spikes.png"
        fig.savefig(out2, dpi=160)
        print(f"[save] {out2}")



if __name__ == "__main__":
    main()
