"""Cumulative-sum ssGSEA oracle.

The timed path uses rank weights and a stable argsort. Size filtering is the
caller's job: this function scores every gene set it is given.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import Enum
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata


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


def _resolve_correl_norm_type(
    correl_norm_type: CorrelNormType | str | None,
    profile: CompatibilityProfile,
) -> CorrelNormType:
    if correl_norm_type is not None:
        return CorrelNormType(correl_norm_type)
    if profile is CompatibilityProfile.SINGLE_SAMPLE_GSEA_0_2:
        return CorrelNormType.ZSCORE
    return CorrelNormType.RANK


def score(
    expression: pd.DataFrame,
    gene_sets: Mapping[str, Collection[str]],
    *,
    alpha: float = 0.25,
    correl_norm_type: CorrelNormType | str | None = None,
    profile: CompatibilityProfile | str = CompatibilityProfile.OPENSSGSEA,
) -> pd.DataFrame:
    """Compute raw scores with the cumulative-sum definition.

    ``expression`` is a genes-by-samples frame. The result has samples as rows
    and gene sets as columns, in the order of ``gene_sets``.
    """
    values = expression.to_numpy(dtype=np.float64, copy=False)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("expression must be a non-empty genes-by-samples DataFrame")
    if not np.isfinite(values).all():
        raise ValueError("expression contains NaN or infinite values")

    selected_profile = CompatibilityProfile(profile)
    correl_norm = _resolve_correl_norm_type(correl_norm_type, selected_profile)

    ranks = np.asarray(rankdata(values, axis=0), dtype=np.float64)
    deviations = ranks.std(axis=0)
    if np.any(deviations == 0):
        raise ValueError("constant-ranked samples are not scoreable")
    if correl_norm is CorrelNormType.ZSCORE:
        weighted = np.abs((ranks - ranks.mean(axis=0)) / deviations) ** alpha
    else:
        weighted = np.abs(ranks) ** alpha

    if selected_profile is CompatibilityProfile.OPENSSGSEA:
        order = np.argsort(ranks, axis=0, kind="stable")[::-1]
    else:
        order = np.argsort(ranks, axis=0)[::-1]

    result = np.empty((expression.shape[1], len(gene_sets)), dtype=np.float64)
    identifiers = np.asarray([str(value) for value in expression.index])
    for sample_index in range(expression.shape[1]):
        ordered_rows = order[:, sample_index]
        ordered_identifiers = identifiers[ordered_rows]
        ordered_weight = weighted[ordered_rows, sample_index]
        for set_index, members in enumerate(gene_sets.values()):
            mask = np.isin(ordered_identifiers, [str(member) for member in members])
            set_size = int(mask.sum())
            if set_size == 0 or set_size == expression.shape[0]:
                raise ValueError("each gene set must overlap but not cover the universe")
            hit_weight = ordered_weight * mask
            hit_cdf = np.cumsum(hit_weight / hit_weight.sum())
            miss_cdf = np.cumsum((~mask).astype(np.float64) / (len(mask) - set_size))
            result[sample_index, set_index] = np.sum(hit_cdf - miss_cdf)

    return cast(
        pd.DataFrame,
        pd.DataFrame(
            result,
            index=[str(value) for value in expression.columns],
            columns=[str(value) for value in gene_sets],
        ),
    )
