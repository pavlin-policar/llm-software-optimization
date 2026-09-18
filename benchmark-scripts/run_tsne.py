"""Runtime of openTSNE against data size: original vs. optimized build.

For each size, subsamples the Zheng et al. (2017) 10x mouse brain data (the
50-component PCA from upstream openTSNE's examples/prepare_10x.ipynb), then

1. times the stages before the optimization with each build: HNSW nearest
   neighbours (k = 90, cosine), perplexity calibration (perplexity 30), and
   spectral initialization;
2. times the optimization with each build, for FIt-SNE and Barnes-Hut and for
   every thread count: 250 iterations at exaggeration 12 and momentum 0.5, then
   500 at momentum 0.8.

Both builds optimize from the same affinities and initialization, computed
once per size and cached. Each measurement runs in its own process (the two
builds are both importable as `openTSNE`, so they cannot share one), one at a
time, under a timeout and a free-memory floor. Results are appended to a JSONL
file as they finish; rerunning skips what is already there, so an interrupted
run resumes.

Usage:
    python benchmark_scaling.py run --original PATH --optimized PATH --data 10x_mouse_zheng.pkl.gz
    python benchmark_scaling.py plot    # figure_scaling.png / .pdf
    python benchmark_scaling.py tikz    # figures.tex, pgfplots

PATH is a source checkout of openTSNE built in place
(`python setup.py build_ext --inplace`) with OpenMP enabled.
Requires numpy, scipy, hnswlib and psutil; `plot` also needs matplotlib.
"""
import argparse
import datetime
import gzip
import json
import os
import pickle
import subprocess
import sys
import time
from os.path import abspath, dirname, exists, join

HERE = dirname(abspath(__file__))

SIZES = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
METHODS = ["fft", "bh"]
THREADS = [8, 1]
BUILDS = ["original", "optimized"]

PERPLEXITY = 30
K_NEIGHBORS = 90


# --------------------------------------------------------------------------
# Measurements, each run in a child process for one build

def import_build(path):
    """Import openTSNE from a built checkout, and fail if another copy wins."""
    path = abspath(path)
    sys.path.insert(0, path)
    import openTSNE
    import openTSNE._tsne

    for mod in (openTSNE, openTSNE._tsne):
        if not abspath(mod.__file__).startswith(path + os.sep):
            raise RuntimeError(f"{mod.__name__} imported from {mod.__file__}, not {path}")
    return openTSNE


def fixture_paths(cache, n):
    return (join(cache, f"P_zheng_{n}_{PERPLEXITY}.npz"),
            join(cache, f"init_zheng_{n}_{PERPLEXITY}.npy"))


def load_subsample(data_file, n, seed=0):
    import numpy as np

    with gzip.open(data_file, "rb") as f:
        x = np.ascontiguousarray(pickle.load(f)["pca_50"], dtype=np.float64)
    if n > x.shape[0]:
        raise ValueError(f"asked for {n} points from a source with {x.shape[0]}")
    if n == x.shape[0]:
        return x
    idx = np.random.RandomState(seed).choice(x.shape[0], n, replace=False)
    idx.sort()
    return np.ascontiguousarray(x[idx])


def worker_preprocess(args):
    """Time neighbour search, affinities and initialization; cache them if missing."""
    openTSNE = import_build(args.build)
    import numpy as np
    import scipy.sparse as sp
    from openTSNE.affinity import get_knn_index, joint_probabilities_nn

    x = load_subsample(args.data, args.n)

    t0 = time.perf_counter()
    knn = get_knn_index(x, "hnsw", k=min(args.n - 1, K_NEIGHBORS), metric="cosine",
                        n_jobs=args.threads, random_state=0)
    neighbors, distances = knn.build()
    t_knn = time.perf_counter() - t0
    del knn

    t0 = time.perf_counter()
    P = joint_probabilities_nn(neighbors, distances, [PERPLEXITY], symmetrize=True,
                               n_jobs=args.threads)
    t_perplexity = time.perf_counter() - t0
    del neighbors, distances

    t0 = time.perf_counter()
    init = openTSNE.initialization.spectral(P, random_state=0)
    t_init = time.perf_counter() - t0

    p_file, init_file = fixture_paths(args.cache, args.n)
    if not (exists(p_file) and exists(init_file)):
        # P goes last, through a rename, so an interrupted write never leaves
        # a pair that looks complete
        os.makedirs(args.cache, exist_ok=True)
        np.save(init_file, np.ascontiguousarray(init, dtype=np.float64))
        sp.save_npz(p_file + ".tmp.npz", P.tocoo())
        os.replace(p_file + ".tmp.npz", p_file)

    print(json.dumps({"knn_s": t_knn, "perplexity_s": t_perplexity, "init_s": t_init,
                      "nnz": int(P.nnz)}))


class _Affinities:
    """Hands a precomputed P to TSNEEmbedding, so no neighbour search runs."""

    def __init__(self, P):
        self.P = P


def worker_optimize(args):
    """Time the 750 optimization iterations from the cached P and initialization."""
    openTSNE = import_build(args.build)
    import numpy as np
    import scipy.sparse as sp
    from openTSNE import tsne as tsne_mod

    p_file, init_file = fixture_paths(args.cache, args.n)
    P = sp.load_npz(p_file).tocsr()
    params = dict(theta=0.5, n_interpolation_points=3, min_num_intervals=50, ints_in_interval=1)
    emb = openTSNE.TSNEEmbedding(np.load(init_file), _Affinities(P),
                                 negative_gradient_method=args.method,
                                 n_jobs=args.threads, random_state=0, **params)

    t0 = time.perf_counter()
    emb.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    t_exaggeration = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb.optimize(n_iter=500, momentum=0.8, inplace=True)
    t_normal = time.perf_counter() - t0

    # The KL divergence is a sanity check that both builds converge alike; untimed
    y = np.ascontiguousarray(emb, dtype=np.float64)
    if args.method == "bh":
        kl, _ = tsne_mod.kl_divergence_bh(y, P, 1.0, {"theta": 0.5},
                                          should_eval_error=True, n_jobs=args.threads)
    else:
        fft = {k: v for k, v in params.items() if k != "theta"}
        kl, _ = tsne_mod.kl_divergence_fft(y, P, 1.0, fft,
                                           should_eval_error=True, n_jobs=args.threads)

    print(json.dumps({"exaggeration_s": t_exaggeration, "normal_s": t_normal,
                      "optimization_s": t_exaggeration + t_normal, "kl": float(kl)}))


# --------------------------------------------------------------------------
# Driver

def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def git_commit(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or None


def load_results(path):
    """Latest record per measurement; later lines supersede earlier ones."""
    results = {}
    if exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results[(r["stage"], r["build"], r["n"], r.get("method"), r["threads"])] = r
    return results


def timeout_for(n):
    # The slowest configuration, the original Barnes-Hut on one thread at 1M
    # points, takes about 2.5 h on an M1 Pro
    return 3600 if n <= 100_000 else 2 * 3600 if n <= 200_000 else 4 * 3600 if n <= 500_000 else 8 * 3600


def run_child(cmd, threads, timeout_s, min_free_gb, log_path):
    """Run one measurement; returns (status, parsed last stdout line or None, wall seconds).

    The child is killed on timeout or when available system memory falls below
    min_free_gb, before the machine starts swapping heavily."""
    import psutil

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    env.pop("PYTHONPATH", None)

    status = None
    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=logf, text=True, env=env)
        while proc.poll() is None:
            if time.perf_counter() - t0 > timeout_s:
                status = "timeout"
            elif psutil.virtual_memory().available < min_free_gb * 1024**3:
                status = "killed_memory"
            if status:
                proc.kill()
                break
            time.sleep(2)
        out, _ = proc.communicate()
        logf.write("\n--- stdout ---\n" + (out or ""))
    wall = time.perf_counter() - t0
    if status is None:
        status = "ok" if proc.returncode == 0 else "failed"
    parsed = json.loads(out.strip().splitlines()[-1]) if status == "ok" else None
    return status, parsed, wall


def cmd_run(args):
    builds = {"original": abspath(args.original), "optimized": abspath(args.optimized)}
    for label, path in builds.items():
        if not exists(join(path, "openTSNE")):
            sys.exit(f"--{label}: {path} is not an openTSNE checkout")
    commits = {b: git_commit(p) for b, p in builds.items()}
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(dirname(abspath(args.out)), exist_ok=True)

    def measure(stage, build, n, threads, method=None):
        key = (stage, build, n, method, threads)
        done = load_results(args.out)
        if key in done and done[key]["status"] == "ok":
            return True
        # A size that failed is not worth retrying at larger sizes
        if any(k[:2] == key[:2] and k[3:] == key[3:] and k[2] < n and r["status"] != "ok"
               for k, r in done.items()):
            log(f"{stage} {build} n={n} {method or ''} j{threads}: skipped, a smaller size failed")
            return False
        name = f"{stage}_{build}_{n}_{method or 'na'}_j{threads}"
        cmd = [sys.executable, abspath(__file__), f"_{stage}", "--build", builds[build],
               "--n", str(n), "--threads", str(threads), "--data", abspath(args.data),
               "--cache", abspath(args.cache)]
        if method:
            cmd += ["--method", method]
        log(f"{name}: start")
        status, res, wall = run_child(cmd, threads, timeout_for(n), args.min_free_gb,
                                      join(args.log_dir, name + ".log"))
        rec = {"stage": stage, "build": build, "commit": commits[build], "n": n,
               "method": method, "threads": threads, "status": status, "wall_s": wall,
               "finished": datetime.datetime.now().isoformat(timespec="seconds"),
               **(res or {})}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{name}: {status} in {wall:.1f} s")
        return status == "ok"

    for threads in args.threads:
        for n in args.sizes:
            # The optimized build writes the cached fixtures; the original one
            # only times the same stages on the same input
            all([measure("preprocess", b, n, args.preprocess_threads)
                      for b in ("optimized", "original")])
            if not all(exists(p) for p in fixture_paths(args.cache, n)):
                log(f"n={n}: no affinities or initialization; skipping its optimization runs")
                continue
            for method in args.methods:
                for build in BUILDS:
                    measure("optimize", build, n, threads, method)
    log("benchmark finished")


# --------------------------------------------------------------------------
# Figure

def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(args.out)
    ok = {k: r for k, r in results.items() if r["status"] == "ok"}
    style = {
        ("original", "fft"): dict(color="#1f77b4", ls="--", marker="o", label="FIt-SNE, original"),
        ("optimized", "fft"): dict(color="#1f77b4", ls="-", marker="o", label="FIt-SNE, optimized"),
        ("original", "bh"): dict(color="#d62728", ls="--", marker="s", label="Barnes–Hut, original"),
        ("optimized", "bh"): dict(color="#d62728", ls="-", marker="s", label="Barnes–Hut, optimized"),
    }
    threads_list = sorted({k[4] for k in ok if k[0] == "optimize"}, reverse=True)
    fig, axes = plt.subplots(2, len(threads_list), figsize=(5.5 * len(threads_list), 8.5),
                             squeeze=False)
    for col, threads in enumerate(threads_list):
        for row, entire in enumerate((False, True)):
            ax = axes[row][col]
            for (build, method), st in style.items():
                xs, ys = [], []
                for n in sorted({k[2] for k in ok}):
                    r = ok.get(("optimize", build, n, method, threads))
                    pre = ok.get(("preprocess", build, n, None, args.preprocess_threads))
                    if r is None or (entire and pre is None):
                        continue
                    t = r["optimization_s"]
                    if entire:
                        t += pre["knn_s"] + pre["perplexity_s"] + pre["init_s"]
                    xs.append(n / 1e3)
                    ys.append(t / 60)
                if xs:
                    ax.plot(xs, ys, markersize=5, linewidth=1.8, **st)
            tl = f"{threads} thread{'s' if threads > 1 else ''}"
            if not entire:
                title = f"optimization, {tl}"
            elif threads != args.preprocess_threads:
                title = (f"entire run, optimization on {tl}\n(neighbours, affinities, init "
                         f"on {args.preprocess_threads} threads)")
            else:
                title = f"entire run, {tl}"
            ax.set(xlabel="points (thousands)", ylabel="wall time (min)", title=title)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.figure}.{ext}", dpi=160)
    print(f"wrote {args.figure}.png and {args.figure}.pdf")


# Colour tells the method and build apart; line and marker tell the thread
# count: solid with filled marks on 8 threads, dashed with open marks on 1
TIKZ_COLUMNS = [
    ("bh_orig", "bh", "original", "Barnes--Hut (original)", "red!85!black", "*"),
    ("bh_ai", "bh", "optimized", "Barnes--Hut (AI)", "orange", "square*"),
    ("fft_orig", "fft", "original", "FIt-SNE (original)", "blue!80!black", "triangle*"),
    ("fft_ai", "fft", "optimized", "FIt-SNE (AI)", "cyan!60!black", "diamond*"),
]

TIKZ_TEMPLATE = r"""\begin{{center}}
\begin{{tikzpicture}}
\begin{{axis}}[
    xlabel={{Number of points}},
    ylabel={{Time [s]}},
    ymode=log,
    grid=both,
    legend columns=2,
    legend style={{
        at={{(0.5,-0.2)}},
        anchor=north,
        cells={{anchor=west}},
        font=\footnotesize,
    }},
    width=\columnwidth,
]
\pgfplotstableread{{
{table}
}}\datatable
{plots}
\end{{axis}}
\end{{tikzpicture}}
\captionof{{figure}}{{{caption}}}
\label{{{label}}}
\end{{center}}
"""


def cmd_tikz(args):
    ok = {k: r for k, r in load_results(args.out).items() if r["status"] == "ok"}
    sizes = sorted({k[2] for k in ok if k[0] == "optimize"})
    threads_list = sorted({k[4] for k in ok if k[0] == "optimize"}, reverse=True)
    pre_threads = args.preprocess_threads

    def seconds(build, n, method, threads, entire):
        r = ok.get(("optimize", build, n, method, threads))
        if r is None:
            return None
        t = r["optimization_s"]
        if entire:
            pre = ok.get(("preprocess", build, n, None, pre_threads))
            if pre is None:
                return None
            t += pre["knn_s"] + pre["perplexity_s"] + pre["init_s"]
        return t

    def thread_text(threads):
        return "1 thread" if threads == 1 else f"{threads} threads"

    # Columns ordered thread count first, so that with two legend columns each
    # row of the legend pairs one method and build across the thread counts
    columns = [(f"{col}_j{threads}", method, build, threads,
                f"{legend}, {thread_text(threads)}", color, mark)
               for col, method, build, legend, color, mark in TIKZ_COLUMNS
               for threads in threads_list]

    figures = []
    for entire in (False, True):
        rows = [f"{'n':<8s}" + "".join(f"{c[0]:<14s}" for c in columns).rstrip()]
        for n in sizes:
            values = [seconds(build, n, method, threads, entire)
                      for _, method, build, threads, *_ in columns]
            rows.append(f"{n:<8d}" + "".join(
                f"{'nan' if v is None else f'{v:.2f}':<14s}" for v in values).rstrip())

        plots = []
        for col, _, _, threads, legend, color, mark in columns:
            if threads == max(threads_list):
                style = f"{color}, mark={mark}"
            else:
                style = f"{color}, dashed, mark={'o' if mark == '*' else mark.rstrip('*')}, mark options={{solid}}"
            plots.append(f"\\addplot[{style}] table[x=n, y={col}] {{\\datatable}};\n"
                         f"\\addlegendentry{{{legend}}}")

        on = " and on ".join(
            "a single thread" if t == 1 else f"{t} threads" for t in threads_list)
        if not entire:
            caption = (f"Execution time of the t-SNE optimization phase (750 iterations) "
                       f"in the original and AI-optimized openTSNE on {on}, on subsamples "
                       f"of a single-cell data set of increasing size.")
            label = "fig:opentsne-optimization"
        else:
            caption = (f"Execution time of a complete t-SNE run (nearest-neighbour search, "
                       f"affinities, spectral initialization and 750 optimization "
                       f"iterations) in the original and AI-optimized openTSNE, optimizing "
                       f"on {on}, on subsamples of a single-cell data set of increasing "
                       f"size. The steps before the optimization always ran on "
                       f"{pre_threads} threads.")
            label = "fig:opentsne-entire"
        figures.append(TIKZ_TEMPLATE.format(table="\n".join(rows), plots="\n".join(plots),
                                            caption=caption, label=label))
    with open(args.tex, "w") as f:
        f.write("\n".join(figures))
    print(f"wrote {len(figures)} figures to {args.tex}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def csv(kind):
        return lambda s: [kind(v) for v in s.split(",")]

    run = sub.add_parser("run", help="run the benchmark (resumes if interrupted)")
    run.add_argument("--original", required=True, help="built checkout of the original openTSNE")
    run.add_argument("--optimized", required=True, help="built checkout of the optimized openTSNE")
    run.add_argument("--data", required=True, help="10x_mouse_zheng.pkl.gz")
    run.add_argument("--sizes", type=csv(int), default=SIZES)
    run.add_argument("--methods", type=csv(str), default=METHODS)
    run.add_argument("--threads", type=csv(int), default=THREADS,
                     help="thread counts for the optimization, in the order to run them")
    run.add_argument("--preprocess-threads", type=int, default=8)
    run.add_argument("--min-free-gb", type=float, default=1.0,
                     help="kill a run when available memory drops below this")
    run.add_argument("--cache", default=join(HERE, "cache"))
    run.add_argument("--log-dir", default=join(HERE, "logs"))
    run.add_argument("--out", default=join(HERE, "results.jsonl"))

    plot = sub.add_parser("plot", help="wall time against size, from the results file")
    plot.add_argument("--out", default=join(HERE, "results.jsonl"))
    plot.add_argument("--preprocess-threads", type=int, default=8)
    plot.add_argument("--figure", default=join(HERE, "figure_scaling"))

    tikz = sub.add_parser("tikz", help="pgfplots figures, from the results file")
    tikz.add_argument("--out", default=join(HERE, "results.jsonl"))
    tikz.add_argument("--preprocess-threads", type=int, default=8)
    tikz.add_argument("--tex", default=join(HERE, "figures.tex"))

    for stage in ("preprocess", "optimize"):
        w = sub.add_parser(f"_{stage}")
        w.add_argument("--build", required=True)
        w.add_argument("--n", type=int, required=True)
        w.add_argument("--threads", type=int, required=True)
        w.add_argument("--data", required=True)
        w.add_argument("--cache", required=True)
        w.add_argument("--method", choices=METHODS)

    args = ap.parse_args()
    {"run": cmd_run, "plot": cmd_plot, "tikz": cmd_tikz,
     "_preprocess": worker_preprocess, "_optimize": worker_optimize}[args.command](args)


if __name__ == "__main__":
    main()
