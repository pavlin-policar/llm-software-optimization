"""GSVA ``.fastRndWalk`` with the same ranks as the other ssGSEA implementations.

This is not a call into R. The walk is GSVA ``.fastRndWalk``:

1. rank each sample column with 1-based average ranks, left fractional;
2. order genes by a stable ascending argsort reversed, so ties match ``naive.py``;
3. for each gene set, match member rows into that ranking;
4. score with the ``.fastRndWalk`` closed form.

Bioconductor GSVA truncates those average ranks to integers (``type(R) <- "integer"``)
before weighting. This port leaves them fractional, so raw scores match ``naive.py``.

The benchmark calls ``score`` with ``normalize=False``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, the same convention as ``naive.py``.

    Parameters
    ----------
    values:
        Genes-by-samples dense array.
    """
    return np.asarray(rankdata(values, axis=0, method="average"), dtype=np.float64)


def match_into_ranking(member_rows: np.ndarray, gene_ranking: np.ndarray) -> np.ndarray:
    """First-position match of member row indices into ``geneRanking``.

    Mirrors ``IRanges::match(geneSetsIdx, geneRanking)`` on 0-based indices.
    """
    positions = np.empty(member_rows.size, dtype=np.int64)
    for index, row in enumerate(member_rows):
        hit = np.flatnonzero(gene_ranking == row)
        if hit.size == 0:
            raise ValueError(f"gene row {int(row)} is missing from geneRanking")
        positions[index] = int(hit[0])
    return positions


def fast_rnd_walk(
    g_set_idx: np.ndarray,
    gene_ranking: np.ndarray,
    ra_column: np.ndarray,
) -> float:
    """Port of GSVA ``.fastRndWalk`` with R argument order and indexing.

    Parameters
    ----------
    g_set_idx:
        Zero-based positions in ``geneRanking`` (R ``match(...) - 1``).
    gene_ranking:
        Row indices in decreasing rank order for one sample.
    ra_column:
        Rank weights ``R^alpha`` for one sample.
    """
    n = int(gene_ranking.size)
    k = int(g_set_idx.size)
    member_rows = gene_ranking[g_set_idx]
    weights = ra_column[member_rows]
    position_weight = n - g_set_idx.astype(np.float64)
    step_cdf_in = float(np.dot(weights, position_weight)) / float(np.sum(weights))
    step_cdf_out = (n * (n + 1) / 2.0 - float(np.sum(position_weight))) / (n - k)
    return step_cdf_in - step_cdf_out


def _mapped_gene_sets(
    gene_names: Sequence[str],
    gene_sets: Mapping[str, Collection[str]],
    *,
    min_size: int,
    max_size: int,
) -> tuple[list[str], list[np.ndarray]]:
    """Map gene-set members to unique row indices and apply size filters."""
    n_genes = len(gene_names)
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")

    symbol_rows: dict[str, list[int]] = {}
    for row, symbol in enumerate(gene_names):
        symbol_rows.setdefault(str(symbol), []).append(row)

    names: list[str] = []
    members: list[np.ndarray] = []
    for name, identifiers in gene_sets.items():
        rows: list[int] = []
        for identifier in identifiers:
            rows.extend(symbol_rows.get(str(identifier), ()))
        unique_rows = np.fromiter(dict.fromkeys(rows), dtype=np.int64, count=len(set(rows)))
        set_size = int(unique_rows.size)
        if set_size == n_genes:
            raise ValueError(f"gene set {str(name)!r} covers the complete gene universe")
        if min_size <= set_size <= max_size:
            names.append(str(name))
            members.append(unique_rows)
    if not names:
        raise ValueError("no gene sets remain after overlap and size filtering")
    return names, members


def score(
    expression: pd.DataFrame,
    gene_sets: Mapping[str, Collection[str]],
    *,
    alpha: float = 0.25,
    min_size: int = 1,
    max_size: int | None = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Score samples by gene sets using GSVA ``.fastRndWalk``.

    Parameters
    ----------
    expression:
        Genes-by-samples numeric frame.
    gene_sets:
        Mapping of gene-set name to member identifiers.
    alpha:
        Exponent applied to average ranks (default 0.25).
    min_size, max_size:
        Mapped-set size filters. ``max_size`` defaults to ``n_genes - 1``.
    normalize:
        If True, divide the score matrix by its global range (GSVA default).
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    values = expression.to_numpy(dtype=np.float64, copy=False)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("expression must have at least two genes and one sample")
    if not np.isfinite(values).all():
        raise ValueError("expression contains NaN or infinite values")

    n_genes, n_samples = values.shape
    resolved_max = int(max_size) if max_size is not None else max(1, n_genes - 1)
    names, member_rows = _mapped_gene_sets(
        [str(gene) for gene in expression.index],
        gene_sets,
        min_size=int(min_size),
        max_size=resolved_max,
    )

    ranks = average_ranks(values)
    ra = np.array(ranks, dtype=np.float64, copy=True)
    if alpha != 1.0:
        np.power(ra, alpha, out=ra)
    scores = np.empty((n_samples, len(names)), dtype=np.float64)

    for sample_index in range(n_samples):
        sample_ranks = ranks[:, sample_index]
        gene_ranking = np.argsort(sample_ranks, kind="stable")[::-1]
        ra_column = ra[:, sample_index]
        for set_index, rows in enumerate(member_rows):
            g_set_idx = match_into_ranking(rows, gene_ranking)
            scores[sample_index, set_index] = fast_rnd_walk(
                g_set_idx,
                gene_ranking,
                ra_column,
            )

    if normalize:
        finite = scores[np.isfinite(scores)]
        span = float(np.max(finite) - np.min(finite))
        if not np.isfinite(span) or span == 0.0:
            raise ValueError("cannot normalize ssGSEA scores: score range is zero")
        scores = scores / span

    return pd.DataFrame(
        scores,
        index=[str(sample) for sample in expression.columns],
        columns=names,
    )
