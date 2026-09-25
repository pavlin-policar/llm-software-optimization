"""Gene-set scaling of four ssGSEA implementations.

Times naive, a Python port of GSVA's .fastRndWalk, GSEApy, and the openSSGSEA
v2 kernel (ai-optimized) as the number of MSigDB GO Biological Process gene
sets grows. Samples stay fixed at the first 1000 columns of the expression
file. Each cell runs in its own process. The recorded time is the scoring call
only. Results are appended to a JSONL file; a rerun skips cells that already
succeeded, and a timeout skips larger gene-set counts for that implementation.

The expression table and the gene-set JSON are not in this repository. See the README.

Usage:
    python run_ssgsea.py run --data ssgsea/data/tcga-brca.star_tpm.tsv --genesets ssgsea/data/c5.go.bp.v2026.1.Hs.json
    python run_ssgsea.py plot
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import time
from os.path import abspath, dirname, exists, join

HERE = dirname(abspath(__file__))
ROOT = dirname(HERE)
SSGSEA = join(ROOT, "ssgsea")

IMPLEMENTATIONS = ("naive", "gsva", "gseapy", "ai-optimized")
MODULES = {
    "naive": "naive",
    "gsva": "gsva",
    "gseapy": "gseapy",
    "ai-optimized": "ai_optimized",
}
DISPLAY_NAMES = {
    "naive": "naive",
    "gsva": "R/GSVA",
    "gseapy": "GSEApy",
    "ai-optimized": "ai-optimized",
}
COLORS = {
    "naive": "#6B4C9A",
    "gsva": "#E09F3E",
    "gseapy": "#5C6B73",
    "ai-optimized": "#1B4D8C",
}

GENE_SET_COUNTS = [10, 25, 50, 100, 250, 500, 1000, 2000, 3000]
N_SAMPLES = 1000
MIN_SIZE = 15
MAX_SIZE = 500
ALPHA = 0.25
TIMEOUT_SECONDS = 120

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_results(path: str) -> dict[tuple[str, int], dict]:
    """Latest record per implementation and gene-set count."""
    results: dict[tuple[str, int], dict] = {}
    if exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    results[(record["implementation"], int(record["n_gene_sets"]))] = record
    return results


def read_gene_sets(path: str) -> dict[str, set[str]]:
    """Load ``{set name: [symbols]}`` and drop empty members."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} is not a JSON object of gene sets")
    gene_sets: dict[str, set[str]] = {}
    for name, members in payload.items():
        if not isinstance(name, str) or not isinstance(members, list):
            raise ValueError(f"{path} entry {name!r} is not a list of symbols")
        if not all(isinstance(member, str) for member in members):
            raise ValueError(f"{path} entry {name!r} contains a non-string symbol")
        gene_sets[name] = {member for member in members if member}
    return gene_sets


def qualifying_names(
    gene_sets: dict[str, set[str]],
    universe: set[str],
    *,
    min_size: int,
    max_size: int,
) -> list[str]:
    """Sorted gene-set names whose overlap with ``universe`` is in range."""
    names: list[str] = []
    for name in sorted(gene_sets):
        matched = len({str(member) for member in gene_sets[name]} & universe)
        if min_size <= matched <= max_size:
            names.append(name)
    return names


def load_expression(path: str, n_samples: int):
    """Genes-by-samples frame: first ``n_samples`` columns, symbols averaged."""
    import pandas as pd

    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    if not header or header[0] != "gene_symbol":
        raise ValueError(f"{path} header does not start with gene_symbol")
    n_available = len(header) - 1
    if n_available < n_samples:
        raise ValueError(f"{path} has {n_available} samples, need {n_samples}")
    expression = pd.read_csv(path, sep="\t", index_col=0, usecols=range(n_samples + 1))
    expression = expression.astype("float64")
    if expression.index.has_duplicates:
        expression = expression.groupby(level=0, sort=False).mean()
    expression.index = expression.index.astype(str)
    return expression


def file_stamp(path: str) -> dict[str, object]:
    stat = os.stat(path)
    return {"path": abspath(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_manifest_path(cache: str) -> str:
    return join(cache, "manifest.json")


def cache_is_current(cache: str, data: str, genesets: str, n_samples: int) -> bool:
    path = cache_manifest_path(cache)
    if not exists(path):
        return False
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "data": file_stamp(data),
        "genesets": file_stamp(genesets),
        "n_samples": n_samples,
        "min_size": MIN_SIZE,
        "max_size": MAX_SIZE,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def prepare_cache(cache: str, data: str, genesets: str, n_samples: int, n_needed: int) -> int:
    """Write the sliced matrix and the ordered qualifying gene sets. Return how many qualify."""
    import numpy as np

    if cache_is_current(cache, data, genesets, n_samples) and exists(join(cache, "expression.npy")):
        with open(cache_manifest_path(cache), encoding="utf-8") as handle:
            qualifying = int(json.load(handle)["n_qualifying"])
        if qualifying < n_needed:
            raise SystemExit(
                f"only {qualifying} gene sets overlap the expression symbols "
                f"with size in [{MIN_SIZE}, {MAX_SIZE}]; need {n_needed}"
            )
        log(f"reusing cache {cache} ({qualifying} qualifying gene sets)")
        return qualifying

    log("loading expression and gene sets")
    expression = load_expression(data, n_samples)
    gene_sets = read_gene_sets(genesets)
    universe = set(expression.index.astype(str))
    names = qualifying_names(gene_sets, universe, min_size=MIN_SIZE, max_size=MAX_SIZE)
    if len(names) < n_needed:
        raise SystemExit(
            f"only {len(names)} gene sets overlap the expression symbols "
            f"with size in [{MIN_SIZE}, {MAX_SIZE}]; need {n_needed}"
        )

    os.makedirs(cache, exist_ok=True)
    np.save(join(cache, "expression.npy"), np.ascontiguousarray(expression.to_numpy()))
    with open(join(cache, "genes.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(str(gene) for gene in expression.index) + "\n")
    with open(join(cache, "samples.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(str(sample) for sample in expression.columns) + "\n")
    ordered = [[name, sorted(gene_sets[name])] for name in names]
    with open(join(cache, "genesets.json"), "w", encoding="utf-8") as handle:
        json.dump(ordered, handle)
    manifest = {
        "data": file_stamp(data),
        "genesets": file_stamp(genesets),
        "n_samples": n_samples,
        "min_size": MIN_SIZE,
        "max_size": MAX_SIZE,
        "n_genes": int(expression.shape[0]),
        "n_qualifying": len(names),
    }
    with open(cache_manifest_path(cache), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    log(
        f"cached {expression.shape[0]} genes x {expression.shape[1]} samples, "
        f"{len(names)} qualifying gene sets"
    )
    return len(names)


def load_case(cache: str, n_gene_sets: int):
    import numpy as np
    import pandas as pd

    values = np.load(join(cache, "expression.npy"))
    with open(join(cache, "genes.txt"), encoding="utf-8") as handle:
        genes = [line.rstrip("\n") for line in handle if line.strip()]
    with open(join(cache, "samples.txt"), encoding="utf-8") as handle:
        samples = [line.rstrip("\n") for line in handle if line.strip()]
    with open(join(cache, "genesets.json"), encoding="utf-8") as handle:
        ordered = json.load(handle)
    if n_gene_sets > len(ordered):
        raise ValueError(f"cache has {len(ordered)} gene sets, need {n_gene_sets}")
    expression = pd.DataFrame(values, index=genes, columns=samples)
    gene_sets = {name: set(members) for name, members in ordered[:n_gene_sets]}
    return expression, gene_sets


def load_implementation(implementation: str):
    """Load an ssgsea module by file path.

    ``gseapy.py`` must not be imported under the name ``gseapy``: that name is
    the installed package the wrapper calls.
    """
    if implementation not in MODULES:
        raise KeyError(f"unknown implementation {implementation!r}")
    filename = MODULES[implementation] + ".py"
    module_name = "ssgsea_" + MODULES[implementation]
    spec = importlib.util.spec_from_file_location(module_name, join(SSGSEA, filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {join(SSGSEA, filename)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def score_case(implementation: str, expression, gene_sets):
    module = load_implementation(implementation)
    if implementation == "gsva":
        return module.score(
            expression,
            gene_sets,
            alpha=ALPHA,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            normalize=False,
        )
    if implementation == "gseapy":
        return module.score(
            expression,
            gene_sets,
            alpha=ALPHA,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            threads=1,
        )
    if implementation == "ai-optimized":
        return module.score(
            expression,
            gene_sets,
            alpha=ALPHA,
            normalization="raw",
            chunk_size=None,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
        )
    return module.score(expression, gene_sets, alpha=ALPHA)


def worker_score(args: argparse.Namespace) -> None:
    import numpy as np

    unavailable = load_implementation("gseapy").GseapyUnavailable
    expression, gene_sets = load_case(args.cache, args.n_gene_sets)
    try:
        started = time.perf_counter()
        result = score_case(args.implementation, expression, gene_sets)
        elapsed = time.perf_counter() - started
    except unavailable as error:
        print(json.dumps({"status": "unsupported", "error_message": str(error)}))
        return
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        )
        return
    values = result.to_numpy(dtype=np.float64)
    print(
        json.dumps(
            {
                "status": "ok",
                "elapsed_seconds": elapsed,
                "output_shape": [int(result.shape[0]), int(result.shape[1])],
                "checksum": float(np.sum(values)),
            }
        )
    )


def run_child(cmd: list[str], timeout_s: float, log_path: str) -> tuple[str, dict | None, float]:
    env = dict(os.environ)
    for variable in THREAD_ENV:
        env[variable] = "1"
    env.pop("PYTHONPATH", None)

    status = None
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_file, text=True, env=env)
        while proc.poll() is None:
            if time.perf_counter() - started > timeout_s:
                status = "timeout"
                proc.kill()
                break
            time.sleep(1)
        out, _ = proc.communicate()
        log_file.write("\n--- stdout ---\n" + (out or ""))
    wall = time.perf_counter() - started
    parsed = None
    if status is None:
        lines = [line for line in (out or "").splitlines() if line.strip()]
        if proc.returncode == 0 and lines:
            try:
                parsed = json.loads(lines[-1])
            except json.JSONDecodeError:
                parsed = None
        status = str(parsed.get("status")) if isinstance(parsed, dict) and parsed.get("status") else (
            "failed" if proc.returncode != 0 or parsed is None else "ok"
        )
        if status not in {"ok", "unsupported", "error", "timeout", "failed"}:
            status = "failed"
    return status, parsed if status != "timeout" else None, wall


def append_record(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def smaller_timeout(done: dict[tuple[str, int], dict], implementation: str, n_gene_sets: int) -> int | None:
    timed_out = [
        size
        for (name, size), record in done.items()
        if name == implementation and size < n_gene_sets and record.get("status") == "timeout"
    ]
    return max(timed_out) if timed_out else None


def cmd_run(args: argparse.Namespace) -> None:
    if not exists(args.data):
        raise SystemExit(f"missing expression file: {args.data}")
    if not exists(args.genesets):
        raise SystemExit(f"missing gene-set file: {args.genesets}")
    unknown = [name for name in args.implementations if name not in MODULES]
    if unknown:
        known = ", ".join(IMPLEMENTATIONS)
        raise SystemExit(f"unknown implementation(s): {', '.join(unknown)}; expected one of: {known}")

    os.makedirs(dirname(abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    n_needed = max(args.gene_sets)
    prepare_cache(args.cache, args.data, args.genesets, args.n_samples, n_needed)

    for n_gene_sets in args.gene_sets:
        for implementation in args.implementations:
            done = load_results(args.out)
            key = (implementation, n_gene_sets)
            if key in done and done[key].get("status") == "ok":
                log(f"{implementation} G={n_gene_sets}: already ok, skipping")
                continue
            predecessor = smaller_timeout(done, implementation, n_gene_sets)
            if predecessor is not None:
                if key not in done or done[key].get("status") != "timeout":
                    append_record(
                        args.out,
                        {
                            "implementation": implementation,
                            "n_gene_sets": n_gene_sets,
                            "n_samples": args.n_samples,
                            "status": "timeout",
                            "error_type": "skipped_predecessor_timeout",
                            "error_message": (
                                f"skipped because {implementation} timed out at n_gene_sets={predecessor}"
                            ),
                            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                log(f"{implementation} G={n_gene_sets}: skipped, timed out at G={predecessor}")
                continue

            name = f"{implementation}_{n_gene_sets}".replace("-", "_")
            cmd = [
                sys.executable,
                abspath(__file__),
                "_score",
                "--implementation",
                implementation,
                "--n-gene-sets",
                str(n_gene_sets),
                "--cache",
                abspath(args.cache),
            ]
            log(f"{implementation} G={n_gene_sets}: start")
            status, parsed, wall = run_child(cmd, args.timeout, join(args.log_dir, name + ".log"))
            record = {
                "implementation": implementation,
                "n_gene_sets": n_gene_sets,
                "n_samples": args.n_samples,
                "status": status,
                "wall_s": wall,
                "finished": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            if parsed:
                for field in ("elapsed_seconds", "output_shape", "checksum", "error_type", "error_message"):
                    if field in parsed:
                        record[field] = parsed[field]
            append_record(args.out, record)
            elapsed = record.get("elapsed_seconds")
            detail = f"{elapsed:.3f} s" if isinstance(elapsed, float) else f"wall {wall:.1f} s"
            log(f"{implementation} G={n_gene_sets}: {status} in {detail}")
    log(f"benchmark finished; results in {args.out}")


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(args.out)
    ok = [record for record in results.values() if record.get("status") == "ok" and "elapsed_seconds" in record]
    if not ok:
        raise SystemExit(f"no successful measurements in {args.out}")

    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    present = {record["implementation"] for record in ok}
    for implementation in IMPLEMENTATIONS:
        if implementation not in present:
            continue
        points = sorted(
            (int(record["n_gene_sets"]), float(record["elapsed_seconds"]))
            for record in ok
            if record["implementation"] == implementation
        )
        axis.plot(
            [n for n, _ in points],
            [seconds for _, seconds in points],
            color=COLORS[implementation],
            marker="o",
            markersize=5,
            linewidth=1.4,
            label=DISPLAY_NAMES[implementation],
        )
    axis.axhline(args.timeout, color="#8A9199", linestyle="--", linewidth=0.8, zorder=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    n_samples = int(ok[0].get("n_samples", N_SAMPLES))
    axis.set_xlabel("Gene sets")
    axis.set_ylabel("Time (s)")
    axis.set_title(f"Gene sets ({n_samples} samples)")
    axis.grid(True, which="major", color="#D0D5DA", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(f"{args.figure}.{extension}", dpi=160)
    print(f"wrote {args.figure}.png and {args.figure}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def csv_int(text: str) -> list[int]:
        return [int(value) for value in text.split(",") if value]

    run = sub.add_parser("run", help="run the gene-set scaling benchmark")
    run.add_argument("--data", required=True, help="genes-by-samples TSV, first column gene_symbol")
    run.add_argument("--genesets", required=True, help="JSON object of gene-set name to symbol list")
    run.add_argument(
        "--implementations",
        nargs="*",
        default=list(IMPLEMENTATIONS),
        help="subset of naive, gsva, gseapy, ai-optimized",
    )
    run.add_argument("--gene-sets", type=csv_int, default=GENE_SET_COUNTS)
    run.add_argument("--n-samples", type=int, default=N_SAMPLES)
    run.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    run.add_argument("--cache", default=join(HERE, "tmp", "ssgsea-cache"))
    run.add_argument("--log-dir", default=join(HERE, "tmp", "logs"))
    run.add_argument("--out", default=join(HERE, "results-ssgsea.jsonl"))

    plot = sub.add_parser("plot", help="log-log wall time against gene-set count")
    plot.add_argument("--out", default=join(HERE, "results-ssgsea.jsonl"))
    plot.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    plot.add_argument("--figure", default=join(HERE, "figure_ssgsea_genesets"))

    worker = sub.add_parser("_score")
    worker.add_argument("--implementation", required=True)
    worker.add_argument("--n-gene-sets", type=int, required=True)
    worker.add_argument("--cache", required=True)

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        worker_score(args)


if __name__ == "__main__":
    main()
