"""GSEApy ssGSEA with rank normalization and the raw enrichment score.

Requires the ``gseapy`` package. ``score`` raises ``GseapyUnavailable`` when
that package is not installed.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import pandas as pd


class GseapyUnavailable(RuntimeError):
    """Raised when the gseapy package cannot be imported."""


def _as_gene_set_mapping(
    gene_sets: Mapping[str, Collection[str]],
) -> dict[str, set[str]]:
    return {str(name): {str(member) for member in members} for name, members in gene_sets.items()}


def _resolve_max_size(expression: pd.DataFrame, max_size: int | None) -> int:
    if max_size is not None:
        return int(max_size)
    return max(1, int(expression.shape[0]) - 1)


def score(
    expression: pd.DataFrame,
    gene_sets: Mapping[str, Collection[str]],
    *,
    alpha: float = 0.25,
    min_size: int = 1,
    max_size: int | None = None,
    threads: int = 1,
) -> pd.DataFrame:
    """Score with ``gseapy.ssgsea`` and return samples by gene sets.

    Uses ``sample_norm_method="rank"`` and returns the raw ``ES`` column,
    reindexed to the input sample and gene-set order.
    """
    try:
        import gseapy
    except ImportError as error:
        raise GseapyUnavailable("gseapy is not installed") from error

    mapping = _as_gene_set_mapping(gene_sets)
    result = gseapy.ssgsea(
        data=expression,
        gene_sets=mapping,
        outdir=None,
        sample_norm_method="rank",
        weight=float(alpha),
        min_size=int(min_size),
        max_size=_resolve_max_size(expression, max_size),
        no_plot=True,
        threads=int(threads),
        verbose=False,
    )
    scores = result.res2d.pivot(index="Name", columns="Term", values="ES")
    scores.index = scores.index.map(str)
    scores.columns = scores.columns.map(str)
    sample_ids = [str(value) for value in expression.columns]
    set_ids = list(mapping)
    return scores.reindex(index=sample_ids, columns=set_ids)
