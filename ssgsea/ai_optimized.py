"""openSSGSEA v2 kernel: fused ranking and one CSR pass over gene-set membership.

This is the ai-optimized implementation. ``score`` keeps raw enrichment scores.
When ``chunk_size`` is omitted, samples are tiled to keep the dense working set
near a 20k-gene / 128-sample footprint.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
import pandas as pd
from scipy import sparse


class CompatibilityProfile(str, Enum):
    """Versioned numerical semantics.

    ``OPENSSGSEA`` uses stable tie ordering and defaults to rank weights.
    ``SINGLE_SAMPLE_GSEA_0_2`` retains NumPy's legacy argsort and defaults to
    z-scored rank weights.
    """

    OPENSSGSEA = "openssgsea"
    SINGLE_SAMPLE_GSEA_0_2 = "single_sample_gsea_0.2"


class CorrelNormType(str, Enum):
    """How ranked expression values are transformed before weighting."""

    RANK = "rank"
    ZSCORE = "zscore"


class Normalization(str, Enum):
    """Optional transformations applied after raw enrichment scoring."""

    RAW = "raw"
    SAMPLE_ZSCORE = "sample_zscore"
    GLOBAL_RANGE = "global_range"


def _resolve_correl_norm_type(
    correl_norm_type: CorrelNormType | str | None,
    profile: CompatibilityProfile,
) -> CorrelNormType:
    if correl_norm_type is not None:
        return CorrelNormType(correl_norm_type)
    if profile is CompatibilityProfile.SINGLE_SAMPLE_GSEA_0_2:
        return CorrelNormType.ZSCORE
    return CorrelNormType.RANK


@dataclass(frozen=True)
class CompiledGeneSets:
    """Sparse gene-set membership aligned to a specific ordered gene universe."""

    names: tuple[str, ...]
    genes: tuple[str, ...]
    membership: sparse.csr_matrix

    @property
    def sizes(self) -> np.ndarray:
        """Return the number of matched rows in each gene set."""
        return np.asarray(self.membership.sum(axis=1)).ravel()


def _coerce_expression(
    expression: pd.DataFrame | np.ndarray,
    gene_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if isinstance(expression, pd.DataFrame):
        values = expression.to_numpy(dtype=np.float64, copy=False)
        genes = tuple(str(value) for value in expression.index)
        samples = tuple(str(value) for value in expression.columns)
        if gene_names is not None and tuple(str(value) for value in gene_names) != genes:
            raise ValueError("gene_names does not match the DataFrame index")
    else:
        values = np.asarray(expression, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("expression must be a two-dimensional genes-by-samples matrix")
        if gene_names is None:
            raise ValueError("gene_names is required for NumPy input")
        genes = tuple(str(value) for value in gene_names)
        samples = tuple(f"sample_{index}" for index in range(values.shape[1]))

    if values.ndim != 2 or values.shape[0] != len(genes):
        raise ValueError("expression rows must match gene_names")
    if values.shape[1] == 0:
        raise ValueError("expression must contain at least one sample")
    if not np.isfinite(values).all():
        raise ValueError("expression contains NaN or infinite values")
    return values, genes, samples


def _normalize(scores: np.ndarray, normalization: Normalization) -> np.ndarray:
    if normalization is Normalization.RAW:
        return scores
    if normalization is Normalization.SAMPLE_ZSCORE:
        means = scores.mean(axis=1, keepdims=True)
        standard_deviations = scores.std(axis=1, keepdims=True)
        return cast(
            np.ndarray,
            np.divide(
                scores - means,
                standard_deviations,
                out=np.zeros_like(scores),
                where=standard_deviations != 0,
            ),
        )
    score_range = float(scores.max() - scores.min())
    if score_range == 0:
        return np.zeros_like(scores)
    return scores / score_range


_INT32_MAX = np.iinfo(np.int32).max
_REFERENCE_GENES = 20_000
_REFERENCE_TILE = 128


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _index_dtype(n_genes: int, nnz: int) -> type[np.int32] | type[np.int64]:
    if n_genes <= _INT32_MAX and nnz <= _INT32_MAX:
        return np.int32
    return np.int64


def compile_gene_sets(
    gene_names: Sequence[str],
    gene_sets: Mapping[str, Collection[str]],
    *,
    min_size: int = 1,
    max_size: int | None = None,
) -> CompiledGeneSets:
    """Compile gene sets into a CSR matrix.

    Duplicate identifiers in ``gene_names`` are all included. CSR column
    indices are not sorted; uniqueness is still enforced so membership is not
    double-counted.
    """
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    genes = tuple(_as_text(gene) for gene in gene_names)
    if not genes:
        raise ValueError("gene_names must not be empty")
    if not gene_sets:
        raise ValueError("gene_sets must not be empty")

    n_genes = len(genes)
    upper = n_genes if max_size is None else max_size
    if upper < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")

    symbol_rows: dict[str, list[int]] = {}
    has_duplicate_symbols = False
    for row, symbol in enumerate(genes):
        bucket = symbol_rows.get(symbol)
        if bucket is None:
            symbol_rows[symbol] = [row]
        else:
            bucket.append(row)
            has_duplicate_symbols = True

    names: list[str] = []
    indices: list[int] = []
    indptr = [0]
    unique_index = {symbol: rows[0] for symbol, rows in symbol_rows.items()}

    for name, members in gene_sets.items():
        if has_duplicate_symbols:
            rows: list[int] = []
            for member in members:
                rows.extend(symbol_rows.get(_as_text(member), ()))
            if len(rows) != len(set(rows)):
                rows = list(dict.fromkeys(rows))
        else:
            rows = []
            for member in members:
                mapped = unique_index.get(member if isinstance(member, str) else _as_text(member))
                if mapped is not None:
                    rows.append(mapped)
            if len(rows) != len(set(rows)):
                rows = list(dict.fromkeys(rows))

        if min_size <= len(rows) <= upper:
            if len(rows) == n_genes:
                raise ValueError(f"gene set {name!r} covers the complete gene universe")
            names.append(_as_text(name))
            indices.extend(rows)
            indptr.append(len(indices))

    if not names:
        raise ValueError("no gene sets remain after overlap and size filtering")

    dtype = _index_dtype(n_genes, len(indices))
    membership = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float64),
            np.asarray(indices, dtype=dtype),
            np.asarray(indptr, dtype=dtype),
        ),
        shape=(len(names), n_genes),
    )
    return CompiledGeneSets(tuple(names), genes, membership)


def _auto_tile_width(n_genes: int, n_samples: int) -> int:
    """Choose a sample tile that stays near the 20k-gene / 128-sample working set."""
    tile = max(1, (_REFERENCE_TILE * _REFERENCE_GENES) // max(n_genes, 1))
    return min(n_samples, tile)


def _average_ranks_and_q(
    values: np.ndarray,
    order: np.ndarray,
    *,
    ranks_out: np.ndarray,
    q_out: np.ndarray,
) -> None:
    """Scatter 1-based average ranks and permutation position weights ``Q=k+1``."""
    n_genes, n_samples = values.shape
    sorted_vals = np.take_along_axis(values, order, axis=0)

    starts = np.empty((n_genes, n_samples), dtype=bool)
    starts[0] = True
    if n_genes > 1:
        starts[1:] = sorted_vals[1:] != sorted_vals[:-1]

    idx = np.arange(n_genes, dtype=np.int64)[:, None]
    first = np.where(starts, idx, 0)
    np.maximum.accumulate(first, axis=0, out=first)

    sorted_rev = sorted_vals[::-1]
    starts_rev = np.empty((n_genes, n_samples), dtype=bool)
    starts_rev[0] = True
    if n_genes > 1:
        starts_rev[1:] = sorted_rev[1:] != sorted_rev[:-1]
    first_rev = np.where(starts_rev, idx, 0)
    np.maximum.accumulate(first_rev, axis=0, out=first_rev)
    last = (n_genes - 1) - first_rev[::-1]

    rank_sorted = first.astype(np.float64, copy=False)
    rank_sorted += last
    rank_sorted *= 0.5
    rank_sorted += 1.0
    np.put_along_axis(ranks_out, order, rank_sorted, axis=0)

    ordinal = np.arange(1, n_genes + 1, dtype=np.float64)[:, None]
    np.put_along_axis(q_out, order, np.broadcast_to(ordinal, q_out.shape), axis=0)


def _raise_if_constant_ranks(ranks: np.ndarray) -> np.ndarray:
    standard_deviation = ranks.std(axis=0)
    if np.any(standard_deviation == 0):
        bad = np.flatnonzero(standard_deviation == 0).tolist()
        raise ValueError(f"constant-ranked sample columns are not scoreable: {bad}")
    return standard_deviation


def _weights_from_ranks(
    ranks: np.ndarray,
    *,
    alpha: float,
    correl_norm: CorrelNormType,
    standard_deviation: np.ndarray,
) -> None:
    """Overwrite ``ranks`` with hit weights. Ranks are in ``[1, N]``."""
    if correl_norm is CorrelNormType.ZSCORE:
        ranks -= ranks.mean(axis=0)
        ranks /= standard_deviation
        np.abs(ranks, out=ranks)
        _power_inplace(ranks, alpha)
        return
    _power_inplace(ranks, alpha)


def _power_inplace(values: np.ndarray, alpha: float) -> None:
    if alpha == 1.0:
        return
    if alpha == 0.0:
        values.fill(1.0)
        return
    if alpha == 0.25:
        np.sqrt(values, out=values)
        np.sqrt(values, out=values)
        return
    if alpha == 0.5:
        np.sqrt(values, out=values)
        return
    np.power(values, alpha, out=values)


def _score_tile(
    values: np.ndarray,
    membership: sparse.csr_matrix,
    sizes: np.ndarray,
    *,
    alpha: float,
    profile: CompatibilityProfile,
    correl_norm: CorrelNormType,
    stacked: np.ndarray,
) -> np.ndarray:
    """Score one sample tile. ``stacked`` must have at least ``3 * n_samples`` columns."""
    n_genes, n_samples = values.shape
    kind = "stable" if profile is CompatibilityProfile.OPENSSGSEA else "quicksort"
    value_order = np.argsort(values, axis=0, kind=kind)

    weighted = stacked[:, :n_samples]
    wq = stacked[:, n_samples : 2 * n_samples]
    position_weight = stacked[:, 2 * n_samples : 3 * n_samples]
    _average_ranks_and_q(
        values,
        value_order,
        ranks_out=weighted,
        q_out=position_weight,
    )

    if profile is CompatibilityProfile.SINGLE_SAMPLE_GSEA_0_2:
        rank_order = np.argsort(weighted, axis=0)
        ordinal = np.arange(1, n_genes + 1, dtype=np.float64)[:, None]
        np.put_along_axis(
            position_weight,
            rank_order,
            np.broadcast_to(ordinal, position_weight.shape),
            axis=0,
        )

    standard_deviation = _raise_if_constant_ranks(weighted)
    _weights_from_ranks(
        weighted,
        alpha=alpha,
        correl_norm=correl_norm,
        standard_deviation=standard_deviation,
    )
    np.multiply(weighted, position_weight, out=wq)

    packed = stacked[:, : 3 * n_samples]
    products = np.asarray(membership @ packed, dtype=np.float64)
    total_set_weight = products[:, :n_samples]
    weighted_set_position = products[:, n_samples : 2 * n_samples]
    membership_q = products[:, 2 * n_samples : 3 * n_samples]
    if np.any(total_set_weight == 0):
        raise ValueError("a gene set has zero total rank weight")

    sum_q = n_genes * (n_genes + 1) / 2.0
    complement_position = sum_q - membership_q
    scores = weighted_set_position / total_set_weight - complement_position / (n_genes - sizes)[:, None]
    return cast(np.ndarray, np.asarray(scores.T, dtype=np.float64))


def score(
    expression: pd.DataFrame | np.ndarray,
    gene_sets: Mapping[str, Collection[str]] | CompiledGeneSets,
    *,
    gene_names: Sequence[str] | None = None,
    alpha: float = 0.25,
    normalization: Normalization | str = Normalization.RAW,
    profile: CompatibilityProfile | str = CompatibilityProfile.OPENSSGSEA,
    correl_norm_type: CorrelNormType | str | None = None,
    min_size: int = 1,
    max_size: int | None = None,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    """Score every sample and gene-set pair.

    ``expression`` is genes by samples. The result has samples as rows and gene
    sets as columns. ``normalization="raw"`` leaves the enrichment scores
    unscaled. ``chunk_size=None`` picks the cache-aware tile width.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    selected_normalization = Normalization(normalization)
    selected_profile = CompatibilityProfile(profile)
    selected_correl = _resolve_correl_norm_type(correl_norm_type, selected_profile)
    values, genes, samples = _coerce_expression(expression, gene_names)

    if isinstance(gene_sets, CompiledGeneSets):
        compiled = gene_sets
        if compiled.genes != genes:
            raise ValueError("compiled gene sets use a different ordered gene universe")
    else:
        compiled = compile_gene_sets(genes, gene_sets, min_size=min_size, max_size=max_size)

    n_genes, n_samples = values.shape
    width = _auto_tile_width(n_genes, n_samples) if chunk_size is None else chunk_size
    if width < 1:
        raise ValueError("chunk_size must be at least 1")

    sizes = np.asarray(compiled.membership.sum(axis=1), dtype=np.float64).ravel()
    stacked = np.empty((n_genes, 3 * width), dtype=np.float64)
    n_sets = len(compiled.names)
    raw_scores = np.empty((n_samples, n_sets), dtype=np.float64)
    for start in range(0, n_samples, width):
        stop = min(start + width, n_samples)
        raw_scores[start:stop] = _score_tile(
            values[:, start:stop],
            compiled.membership,
            sizes,
            alpha=alpha,
            profile=selected_profile,
            correl_norm=selected_correl,
            stacked=stacked,
        )
    normalized_scores = _normalize(raw_scores, selected_normalization)
    return cast(
        pd.DataFrame,
        pd.DataFrame(normalized_scores, index=samples, columns=compiled.names),
    )
