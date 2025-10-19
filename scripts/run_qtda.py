
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

    ap.add_argument("--focus-start", type=str, default=None,
                help="Zoom: start date (YYYY-MM-DD). Example: 2019-06-01")
    ap.add_argument("--focus-end", type=str, default=None,
                    help="Zoom: end date (YYYY-MM-DD). Example: 2021-06-01")
    ap.add_argument("--annotate-topk", type=int, default=0,
                    help="Annotate top-K spikes in the Δ plot (0=off)")

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


        end_offsets = np.arange(args.w - 1, n_embed)                 
        end_offsets = end_offsets[: (n_embed - args.w + 1)]       


        window_dates = pd.to_datetime(dates_embedded.iloc[end_offsets].values)

        first_dt = pd.to_datetime(window_dates[0]).date()
        last_dt  = pd.to_datetime(window_dates[-1]).date()
        print(f"[dates] right-aligned; first={first_dt}, last={last_dt}")
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

    if window_dates is not None and (args.focus_start or args.focus_end):

        start = pd.to_datetime(args.focus_start) if args.focus_start else window_dates[0]
        end   = pd.to_datetime(args.focus_end)   if args.focus_end   else window_dates[-1]


        mask_full = (window_dates >= start) & (window_dates <= end)
        mask_delta = (window_dates[1:] >= start) & (window_dates[1:] <= end)


        if mask_full.any():
            Path("plots").mkdir(exist_ok=True)

            # betti curves zoom
            fig, ax = plt.subplots()
            x_zoom = window_dates[mask_full]
            curves_zoom = curves[mask_full, :]
            for j in range(curves_zoom.shape[1]):
                ax.plot(x_zoom, curves_zoom[:, j], label=f"betti@eps={args.eps[j]}")
            ax.set_title("Betti curves (zoom)")
            ax.set_ylabel("Betti₀ (connected components)")
            ax.set_xlabel("Date")
            ax.legend()

            # weekly ticks for visibility
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # or '%Y-W%U' for week number
            fig.autofmt_xdate()

            outz1 = "plots/betti_curves_zoom.png"
            fig.savefig(outz1, dpi=160, bbox_inches="tight")
            print(f"[save] {outz1}")


            fig, ax = plt.subplots(figsize=(8.5, 4.8))

            x_delta_zoom = window_dates[1:][mask_delta]
            dist_zoom = dist[mask_delta]

            # draw line + points so the series is readable
            ax.plot(x_delta_zoom, dist_zoom, linewidth=2, marker="o", markersize=4)

            # mark statistically significant spikes within the zoom (the “×” marks)
            if spike_idx.size:
                spike_dates_all = window_dates[1:][spike_idx]
                in_zoom = (spike_dates_all >= start) & (spike_dates_all <= end)
                spike_dates = spike_dates_all[in_zoom]
                spike_vals  = dist[spike_idx][in_zoom]
                if len(spike_dates):
                    ax.scatter(spike_dates, spike_vals, marker="x", s=60, linewidths=2)


            if args.annotate_topk and len(dist_zoom):
                topk = min(args.annotate_topk, len(dist_zoom))
                idx_local = np.argsort(dist_zoom)[-topk:] 
                idx_local = idx_local[np.argsort(x_delta_zoom[idx_local])]  

                offsets = [10, 18, 26, 14, 22]
                for i, j in enumerate(idx_local):
                    dt = x_delta_zoom[j]
                    val = dist_zoom[j]
                    ax.annotate(
                        f"{dt.strftime('%Y-%m-%d')}\nΔ={val:.2f}",
                        xy=(dt, val),
                        xytext=(0, offsets[i % len(offsets)]),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.8),
                        arrowprops=dict(arrowstyle="-", lw=0.8)
                    )


            ax.set_title(f"Pairwise L^{args.lp} deltas (zoom)", pad=10)
            ax.set_ylabel("Δ (topology change)")
            ax.set_xlabel("Date")

            # weekly ticks mondays with compact labels
            ax.set_xlim(start, end)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


            ax.grid(True, which="major", axis="both", alpha=0.25, linestyle="--", linewidth=0.8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)


            ymin = max(0, np.nanmin(dist_zoom) - 0.05)
            ymax = np.nanmax(dist_zoom) + 0.15
            ax.set_ylim(ymin, ymax)

            fig.autofmt_xdate(rotation=30, ha="right")
            fig.tight_layout()

            outz2 = "plots/crash_spikes_zoom.png"
            fig.savefig(outz2, dpi=180, bbox_inches="tight")
            print(f"[save] {outz2}")



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
