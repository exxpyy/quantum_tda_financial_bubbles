
import argparse, sys, traceback
import numpy as np
import pandas as pd
from pathlib import Path

from ripser import ripser

try:
    from src.rips_laplacian import rips_simplex_tree, combinatorial_laplacian
    GUDHI_OK = True
except Exception:
    GUDHI_OK = False

from src.takens import takens_embedding, sliding_windows
from src.qpe import pad_and_rescale, unitary_from_laplacian, qpe_zero_prob, backend_info

def load_dates_and_prices(csv_path: str, price_pref: str | None = "close"):
    try:
        sniff = pd.read_csv(csv_path, nrows=1)
        has_header = all(isinstance(c, str) for c in sniff.columns)
    except Exception:
        has_header = False
        sniff = None
    if not has_header and sniff is not None and sniff.shape[1] in (1,2):
        names = ["price"] if sniff.shape[1] == 1 else ["date","price"]
        df = pd.read_csv(csv_path, header=None, names=names)
    else:
        df = pd.read_csv(csv_path)
    dates = None
    for cand in ["date","Date","DATE","timestamp","Timestamp"]:
        if cand in df.columns:
            dates = pd.to_datetime(df[cand], dayfirst=True, errors="coerce")
            break
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

def classical_betti(point_cloud: np.ndarray, eps: float, dim: int) -> int:
    res = ripser(point_cloud, maxdim=dim)
    diagrams = res["dgms"]
    return sum(1 for interval in diagrams[dim] if interval[0] < eps < interval[1])

def main():
    ap = argparse.ArgumentParser(description="QPE demo on Laplacian from a VR complex")
    ap.add_argument("--csv", default="data/sp500.csv")
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--w", type=int, default=48)
    ap.add_argument("--window-idx", type=int, default=None)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--k", type=int, choices=[0,1], default=0)
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--save-plot", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not GUDHI_OK:
        print("[error] GUDHI not available. Install it to run the Laplacian/QPE demo:")
        print("        pip install gudhi")
        sys.exit(1)

    backend_name, ok = backend_info()
    if args.verbose:
        print(f"[qpe] selected backend: {backend_name} (ok={ok})")

    dates, prices = load_dates_and_prices(args.csv, price_pref=args.price_col)
    x = np.log(prices.astype(float))

    X = takens_embedding(x, m=args.m, d=args.d)
    wins = sliding_windows(X, w=args.w)
    if len(wins) == 0:
        print("[error] No windows produced. Adjust m/d/w or provide longer series.")
        sys.exit(1)

    idx = (len(wins)//2) if args.window_idx is None else max(0, min(args.window_idx, len(wins)-1))
    pc = wins[idx]
    if args.verbose:
        print(f"[info] windows={len(wins)}, using window idx={idx}, eps={args.eps}, k={args.k}")
    if dates is not None:
        date_center = dates.iloc[args.d * (args.m - 1) + idx + args.w // 2]
        print(f"[info] approximate date range: centered around {date_center.date()}")


    simplices = rips_simplex_tree(pc, eps=args.eps, max_dim=max(args.k,1))
    Lk = combinatorial_laplacian(simplices, k=args.k)

    b_classical = classical_betti(pc, args.eps, dim=args.k)

    Lp = pad_and_rescale(Lk)
    n = Lp.shape[0]
    n_target = int(np.ceil(np.log2(n)))
    Ugate = unitary_from_laplacian(Lp)

    p0 = qpe_zero_prob(Ugate, n_target=n_target, shots=args.shots, verbose=args.verbose)
    b_qpe = int(round(p0 * n))

    print(f"[info] Laplacian size: {Lk.shape}, padded: {Lp.shape}, n_target={n_target}")
    print(f"[info] classical Betti_{args.k}: {b_classical}")
    print(f"[qpe ] p_zero: {p0:.4f}")
    print(f"[qpe ] Betti_{args.k} (estimated): {b_qpe}")

    if args.save_plot:
        import matplotlib.pyplot as plt
        from pathlib import Path

        Path("plots").mkdir(exist_ok=True)

        date_label = (
            date_center.strftime("%Y-%m-%d") if ("date_center" in locals() and date_center is not None)
            else f"win{idx}"
        )

        fig, ax = plt.subplots()
        ax.bar(["Classical", "QPE est."], [b_classical, b_qpe])
        ax.set_title(f"Betti_{args.k} at eps={args.eps} (center {date_label})")
        ax.set_ylabel("Betti estimate")

        safe_eps = str(args.eps).replace(".", "p")
        out = f"plots/qpe_betti_k{args.k}_eps{safe_eps}_{date_label}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"[save] {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[fatal] Unhandled error in run_qpe_demo. Full traceback below:")
        traceback.print_exc()
        sys.exit(1)
