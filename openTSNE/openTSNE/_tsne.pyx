# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()
import numpy as np
from cython.parallel import prange, parallel
from cython cimport floating
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset

from .quad_tree cimport QuadTree, Node, TravNode2, is_close
from ._matrix_mul.matrix_mul cimport matrix_multiply_fft_1d, matrix_multiply_fft_2d


cdef double EPSILON = np.finfo(np.float64).eps


# An accepted cell reduced to what the summation over it reads. The padding
# holds the record at 32 bytes, so indexing the list is a shift -- the same
# reason `TravNode2` is that size.
cdef struct Pull2:
    double com_x
    double com_y
    double num_points
    int is_leaf
    int _pad


# How many consecutive points along the space-filling curve share one descent.
#
# Points this close together accept nearly the same cells, so the tree is walked
# once for the whole block against a criterion strict enough for every point in
# it, and the resulting list is then summed per point. Eight is where the
# measured curve turns: beyond it a block's bounding box stops being tight, the
# criterion opens most of what it looks at, and the longer list costs more than
# the descents it saves.
cdef Py_ssize_t BH_BLOCK = 8


cdef _real_dtype(floating sample):
    """The numpy dtype matching a fused floating type."""
    if floating is float:
        return np.float32
    return np.float64


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil
    double exp(double x) nogil
    double fabs(double x) nogil
    double fmax(double x, double y) nogil
    double fmin(double x, double y) nogil
    double isinf(long double) nogil
    double INFINITY


cpdef double[:, ::1] compute_gaussian_perplexity(
    double[:, :] distances,
    double[:] desired_perplexities,
    double perplexity_tol=1e-8,
    Py_ssize_t max_iter=200,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t n_samples = distances.shape[0]
        Py_ssize_t n_scales = desired_perplexities.shape[0]
        Py_ssize_t k_neighbors = distances.shape[1]
        double[:, ::1] P = np.zeros_like(distances, dtype=float, order="C")
        double[:, :, ::1] multiscale_P = np.zeros((n_samples, n_scales, k_neighbors))
        double[:, ::1] tau = np.ones((n_samples, n_scales))

        Py_ssize_t i, j, h, iteration
        double[:] desired_entropies = np.log(desired_perplexities)

        double min_tau, max_tau, sum_Pi, sum_PiDj, entropy, entropy_diff, sqrt_tau

    if num_threads < 1:
        num_threads = 1

    for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
        min_tau, max_tau = -INFINITY, INFINITY

        # For every scale find a precision tau that fits the perplexity
        for h in range(n_scales):
            for iteration in range(max_iter):
                sum_Pi, sum_PiDj = 0, 0
                sqrt_tau = sqrt(tau[i, h])

                for j in range(k_neighbors):
                    multiscale_P[i, h, j] = sqrt_tau * exp(-distances[i, j] ** 2 * tau[i, h] * 0.5)
                    sum_Pi = sum_Pi + multiscale_P[i, h, j]
                sum_Pi = sum_Pi + EPSILON

                for j in range(k_neighbors):
                    sum_PiDj = sum_PiDj + multiscale_P[i, h, j] / sum_Pi * distances[i, j] ** 2

                entropy = tau[i, h] * 0.5 * sum_PiDj + log(sum_Pi) - log(tau[i, h]) * 0.5
                entropy_diff = entropy - desired_entropies[h]

                if fabs(entropy_diff) <= perplexity_tol:
                    break

                if entropy_diff > 0:
                    min_tau = tau[i, h]
                    if isinf(max_tau):
                        tau[i, h] *= 2
                    else:
                        tau[i, h] = (tau[i, h] + max_tau) * 0.5
                else:
                    max_tau = tau[i, h]
                    if isinf(min_tau):
                        tau[i, h] /= 2
                    else:
                        tau[i, h] = (tau[i, h] + min_tau) * 0.5

        # Get the probability of the mixture of Gaussians with different precisions
        sum_Pi = 0
        for j in range(k_neighbors):
            for h in range(n_scales):
                P[i, j] = P[i, j] + multiscale_P[i, h, j]
                sum_Pi = sum_Pi + multiscale_P[i, h, j]

        # Perform row-normalization
        for j in range(k_neighbors):
            P[i, j] /= sum_Pi

    return P


cpdef void apply_gradient_update(
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double[:, ::1] update,
    double[:, ::1] gains,
    double momentum,
    double learning_rate,
    double min_gain,
    double max_step_norm,
    bint zero_mean,
    Py_ssize_t num_threads=1,
) except *:
    """Take one gradient descent step, gains and all.

    Written out as a single pass because every part of it -- the sign
    comparison, the gain update, the momentum term, the step norm and the
    subtraction of the mean -- touches the same three arrays. Expressed as
    array operations it is a dozen sweeps over them and as many temporaries,
    which at any interesting number of points costs more than the gradient
    estimate it follows.

    `max_step_norm` is ignored when non-positive, matching a `None` on the
    Python side.
    """
    cdef:
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_dims = embedding.shape[1]
        Py_ssize_t i, d, block, block_end
        double g, u, norm
        double mean[8]
        double block_sum[8]

    if n_dims > 8:
        raise ValueError("`apply_gradient_update` supports at most 8 dimensions")
    if num_threads < 1:
        num_threads = 1

    for i in prange(n_samples, nogil=True, schedule="static", num_threads=num_threads):
        for d in range(n_dims):
            g = gradient[i, d]
            u = update[i, d]
            # The gain grows while the gradient keeps its sign and decays once
            # it flips, so a coordinate that is still descending picks up speed
            # and one that is oscillating is damped. A zero counts as its own
            # sign, which is what makes the first step, taken from a zero
            # update, raise every gain
            if ((u > 0) - (u < 0)) != ((g > 0) - (g < 0)):
                gains[i, d] = gains[i, d] + 0.2
            else:
                gains[i, d] = gains[i, d] * 0.8 + min_gain
            update[i, d] = momentum * u - learning_rate * gains[i, d] * g

        if max_step_norm > 0:
            norm = 0
            for d in range(n_dims):
                norm = norm + update[i, d] * update[i, d]
            norm = sqrt(norm)
            if norm > max_step_norm:
                for d in range(n_dims):
                    update[i, d] = update[i, d] / norm * max_step_norm

        for d in range(n_dims):
            embedding[i, d] = embedding[i, d] + update[i, d]

    if not zero_mean:
        return

    # Summed in blocks so that the mean stays as well conditioned as the
    # pairwise sum this replaces
    for d in range(n_dims):
        mean[d] = 0

    for block in range(0, n_samples, 256):
        block_end = min(block + 256, n_samples)
        for d in range(n_dims):
            block_sum[d] = 0
        for i in range(block, block_end):
            for d in range(n_dims):
                block_sum[d] += embedding[i, d]
        for d in range(n_dims):
            mean[d] += block_sum[d]

    for d in range(n_dims):
        mean[d] /= n_samples

    for i in prange(n_samples, nogil=True, schedule="static", num_threads=num_threads):
        for d in range(n_dims):
            embedding[i, d] = embedding[i, d] - mean[d]


cpdef tuple estimate_positive_gradient_nn(
    sparse_index_type[:] indices,
    sparse_index_type[:] indptr,
    double[:] P_data,
    double[:, ::1] embedding,
    double[:, ::1] reference_embedding,
    double[:, ::1] gradient,
    double dof=1,
    Py_ssize_t num_threads=1,
    bint should_eval_error=False,
):
    cdef:
        Py_ssize_t n_samples = gradient.shape[0]
        Py_ssize_t n_dims = gradient.shape[1]
        double * diff
        double d_ij, p_ij, q_ij, kl_divergence = 0, sum_P = 0
        double y_i0, y_i1, y_i2, diff_0, diff_1, diff_2, pq_ij
        double grad_0, grad_1, grad_2

        Py_ssize_t i, j, k, d, k_start, k_end

        double * emb
        double * ref
        double * grad

    if num_threads < 1:
        num_threads = 1

    if n_samples == 0:
        return sum_P, kl_divergence

    # Degrees of freedom cannot be negative
    if dof <= 0:
        dof = 1e-8

    emb = &embedding[0, 0]
    ref = &reference_embedding[0, 0]
    grad = &gradient[0, 0]

    # One, two and three dimensional embeddings account for all but a handful of
    # uses, so each gets its own loop: the per-point accumulators stay in
    # registers and the coordinate loop has a compile-time trip count. Higher
    # dimensionalities, which only the Barnes-Hut method supports, fall through
    # to the general path at the bottom.
    if n_dims == 1 and not should_eval_error:
        for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
            y_i0 = emb[i]
            grad_0 = 0
            k_start = indptr[i]
            k_end = indptr[i + 1]

            if dof == 1:
                for k in range(k_start, k_end):
                    j = indices[k]
                    diff_0 = y_i0 - ref[j]
                    # q_ij * p_ij, with q_ij left unnormalized
                    pq_ij = P_data[k] / (1 + diff_0 * diff_0)
                    grad_0 = grad_0 + pq_ij * diff_0
            else:
                for k in range(k_start, k_end):
                    j = indices[k]
                    diff_0 = y_i0 - ref[j]
                    pq_ij = P_data[k] / (1 + (diff_0 * diff_0) / dof)
                    grad_0 = grad_0 + pq_ij * diff_0

            grad[i] += grad_0

        return sum_P, kl_divergence

    if n_dims == 2 and not should_eval_error:
        for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
            y_i0 = emb[2 * i]
            y_i1 = emb[2 * i + 1]
            grad_0 = 0
            grad_1 = 0
            k_start = indptr[i]
            k_end = indptr[i + 1]

            if dof == 1:
                for k in range(k_start, k_end):
                    j = indices[k]
                    diff_0 = y_i0 - ref[2 * j]
                    diff_1 = y_i1 - ref[2 * j + 1]
                    # q_ij * p_ij, with q_ij left unnormalized
                    pq_ij = P_data[k] / (1 + diff_0 * diff_0 + diff_1 * diff_1)
                    grad_0 = grad_0 + pq_ij * diff_0
                    grad_1 = grad_1 + pq_ij * diff_1
            else:
                for k in range(k_start, k_end):
                    j = indices[k]
                    diff_0 = y_i0 - ref[2 * j]
                    diff_1 = y_i1 - ref[2 * j + 1]
                    pq_ij = P_data[k] / (1 + (diff_0 * diff_0 + diff_1 * diff_1) / dof)
                    grad_0 = grad_0 + pq_ij * diff_0
                    grad_1 = grad_1 + pq_ij * diff_1

            grad[2 * i] += grad_0
            grad[2 * i + 1] += grad_1

        return sum_P, kl_divergence

    if n_dims == 3 and not should_eval_error:
        for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
            y_i0 = emb[3 * i]
            y_i1 = emb[3 * i + 1]
            y_i2 = emb[3 * i + 2]
            grad_0 = 0
            grad_1 = 0
            grad_2 = 0
            k_start = indptr[i]
            k_end = indptr[i + 1]

            if dof == 1:
                for k in range(k_start, k_end):
                    j = indices[k]
                    diff_0 = y_i0 - ref[3 * j]
                    diff_1 = y_i1 - ref[3 * j + 1]
                    diff_2 = y_i2 - ref[3 * j + 2]
                    # q_ij * p_ij, with q_ij left unnormalized
                    pq_ij = P_data[k] / (
                        1 + diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2)
                    grad_0 = grad_0 + pq_ij * diff_0
                    grad_1 = grad_1 + pq_ij * diff_1
                    grad_2 = grad_2 + pq_ij * diff_2
            else:
                for k in range(k_start, k_end):
                    j = indices[k]
                    diff_0 = y_i0 - ref[3 * j]
                    diff_1 = y_i1 - ref[3 * j + 1]
                    diff_2 = y_i2 - ref[3 * j + 2]
                    pq_ij = P_data[k] / (
                        1 + (diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2) / dof)
                    grad_0 = grad_0 + pq_ij * diff_0
                    grad_1 = grad_1 + pq_ij * diff_1
                    grad_2 = grad_2 + pq_ij * diff_2

            grad[3 * i] += grad_0
            grad[3 * i + 1] += grad_1
            grad[3 * i + 2] += grad_2

        return sum_P, kl_divergence

    # The same three, for the iterations that also evaluate the error. Here
    # q_ij is needed on its own for the divergence, so it cannot be folded into
    # the product the way it is above
    if n_dims == 1:
        for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
            y_i0 = emb[i]
            grad_0 = 0
            k_start = indptr[i]
            k_end = indptr[i + 1]

            for k in range(k_start, k_end):
                j = indices[k]
                p_ij = P_data[k]
                diff_0 = y_i0 - ref[j]
                d_ij = diff_0 * diff_0

                if dof != 1:
                    # No need exp by dof here because the terms cancel out
                    q_ij = 1 / (1 + d_ij / dof)
                else:
                    q_ij = 1 / (1 + d_ij)

                pq_ij = q_ij * p_ij
                grad_0 = grad_0 + pq_ij * diff_0

                # Note that the q_ij is unnormalized, so we need to normalize
                # once the sum of q_ij is known
                sum_P += p_ij
                if dof != 1:
                    # Now we need to do the exp by dof
                    kl_divergence += p_ij * log((p_ij / (q_ij ** dof + EPSILON)) + EPSILON)
                else:
                    kl_divergence += p_ij * log((p_ij / (q_ij + EPSILON)) + EPSILON)

            grad[i] += grad_0

        return sum_P, kl_divergence

    if n_dims == 2:
        for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
            y_i0 = emb[2 * i]
            y_i1 = emb[2 * i + 1]
            grad_0 = 0
            grad_1 = 0
            k_start = indptr[i]
            k_end = indptr[i + 1]

            for k in range(k_start, k_end):
                j = indices[k]
                p_ij = P_data[k]
                diff_0 = y_i0 - ref[2 * j]
                diff_1 = y_i1 - ref[2 * j + 1]
                d_ij = diff_0 * diff_0 + diff_1 * diff_1

                if dof != 1:
                    # No need exp by dof here because the terms cancel out
                    q_ij = 1 / (1 + d_ij / dof)
                else:
                    q_ij = 1 / (1 + d_ij)

                pq_ij = q_ij * p_ij
                grad_0 = grad_0 + pq_ij * diff_0
                grad_1 = grad_1 + pq_ij * diff_1

                # Note that the q_ij is unnormalized, so we need to normalize
                # once the sum of q_ij is known
                sum_P += p_ij
                if dof != 1:
                    # Now we need to do the exp by dof
                    kl_divergence += p_ij * log((p_ij / (q_ij ** dof + EPSILON)) + EPSILON)
                else:
                    kl_divergence += p_ij * log((p_ij / (q_ij + EPSILON)) + EPSILON)

            grad[2 * i] += grad_0
            grad[2 * i + 1] += grad_1

        return sum_P, kl_divergence

    if n_dims == 3:
        for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
            y_i0 = emb[3 * i]
            y_i1 = emb[3 * i + 1]
            y_i2 = emb[3 * i + 2]
            grad_0 = 0
            grad_1 = 0
            grad_2 = 0
            k_start = indptr[i]
            k_end = indptr[i + 1]

            for k in range(k_start, k_end):
                j = indices[k]
                p_ij = P_data[k]
                diff_0 = y_i0 - ref[3 * j]
                diff_1 = y_i1 - ref[3 * j + 1]
                diff_2 = y_i2 - ref[3 * j + 2]
                d_ij = diff_0 * diff_0 + diff_1 * diff_1 + diff_2 * diff_2

                if dof != 1:
                    # No need exp by dof here because the terms cancel out
                    q_ij = 1 / (1 + d_ij / dof)
                else:
                    q_ij = 1 / (1 + d_ij)

                pq_ij = q_ij * p_ij
                grad_0 = grad_0 + pq_ij * diff_0
                grad_1 = grad_1 + pq_ij * diff_1
                grad_2 = grad_2 + pq_ij * diff_2

                # Note that the q_ij is unnormalized, so we need to normalize
                # once the sum of q_ij is known
                sum_P += p_ij
                if dof != 1:
                    # Now we need to do the exp by dof
                    kl_divergence += p_ij * log((p_ij / (q_ij ** dof + EPSILON)) + EPSILON)
                else:
                    kl_divergence += p_ij * log((p_ij / (q_ij + EPSILON)) + EPSILON)

            grad[3 * i] += grad_0
            grad[3 * i + 1] += grad_1
            grad[3 * i + 2] += grad_2

        return sum_P, kl_divergence

    with nogil, parallel(num_threads=num_threads):
        # Use `malloc` here instead of `PyMem_Malloc` because we're in a
        # `nogil` clause and we won't be allocating much memory
        diff = <double *>malloc(n_dims * sizeof(double))
        if not diff:
            with gil:
                raise MemoryError()

        for i in prange(n_samples, schedule="guided"):
            # Iterate over all the neighbors `j` and sum up their contribution
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                p_ij = P_data[k]
                # Compute the direction of the points attraction and the
                # squared euclidean distance between the points
                d_ij = 0
                for d in range(n_dims):
                    diff[d] = emb[i * n_dims + d] - ref[j * n_dims + d]
                    d_ij = d_ij + diff[d] * diff[d]

                if dof != 1:
                    # No need exp by dof here because the terms cancel out
                    q_ij = 1 / (1 + d_ij / dof)
                else:
                    q_ij = 1 / (1 + d_ij)

                # Compute F_{attr} of point `j` on point `i`
                pq_ij = q_ij * p_ij
                for d in range(n_dims):
                    grad[i * n_dims + d] += pq_ij * diff[d]

                # Evaluating the following expressions can slow things down
                # considerably if evaluated every iteration. Note that the q_ij
                # is unnormalized, so we need to normalize once the sum of q_ij
                # is known
                if should_eval_error:
                    sum_P += p_ij
                    if dof != 1:
                        # Now we need to do the exp by dof
                        kl_divergence += p_ij * log((p_ij / (q_ij ** dof + EPSILON)) + EPSILON)
                    else:
                        kl_divergence += p_ij * log((p_ij / (q_ij + EPSILON)) + EPSILON)

        free(diff)

    return sum_P, kl_divergence


cpdef double estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double theta=0.5,
    double dof=1,
    Py_ssize_t num_threads=1,
    bint pairwise_normalization=True,
):
    """Estimate the negative t-SNE gradient using the Barnes-Hut approximation.
    
    Notes
    -----
    Changes the gradient inplace to avoid needless memory allocation. As
    such, this must be run before estimating the positive gradients, since
    the negative gradient must be normalized at the end with the sum of
    q_{ij}s.
    
    """
    cdef:
        Py_ssize_t i, j, num_points = embedding.shape[0]
        Py_ssize_t n_dims = tree.n_dims
        double sum_Q = 0
        # Every entry is written by the traversal, so it needs no clearing
        double[::1] sum_Qi = _scratch("bh/sum_Qi", (num_points,), float)
        # The Barnes-Hut criterion `length / sqrt(distance) < theta` is applied
        # in its squared form, which spares the traversal a square root per node
        double theta_sq = theta * theta
        Node * nodes
        double * center_of_mass
        TravNode2 * trav2
        Py_ssize_t * stack
        Py_ssize_t stack_size
        Py_ssize_t[::1] order
        Py_ssize_t * order_ptr
        # The two-dimensional traversal descends once for a block of points
        Py_ssize_t b, lo, hi, count, num_blocks, pull_capacity
        double center_0, center_1, half_0, half_1
        Pull2 * pulls
        Py_ssize_t failures = 0
        bint tree_holds_these_points

    if num_threads < 1:
        num_threads = 1

    # Degrees of freedom cannot be negative
    if dof <= 0:
        dof = 1e-8

    # Traversal reads the compacted layout, which also guarantees that no node
    # it visits is empty
    tree.compact()
    nodes = tree.nodes
    center_of_mass = tree.center_of_mass
    trav2 = tree.trav2

    if tree.num_nodes == 0:
        return sum_Q
    if trav2 != NULL:
        if trav2[0].num_points == 0:
            return sum_Q
    elif nodes[0].num_points == 0:
        return sum_Q

    # A depth-first traversal never holds more than one sibling group per level
    stack_size = (tree.max_depth + 2) * (tree.max_children + 1) + 2

    # Points are visited in spatial rather than input order. Nearby points
    # descend through nearly the same nodes, so whatever the previous point
    # pulled into cache is largely what the next one needs; in input order
    # consecutive points are unrelated and every traversal starts cold. Each
    # point's contribution is independent, so this only reorders the work.
    # It is also what makes a block of consecutive points tight enough to share
    # one descent.
    #
    # A Morton build sorted the points into an order of exactly this kind on
    # the way to laying out the nodes, and its key is the finer one, so where
    # it ran there is nothing left to compute here
    tree_holds_these_points = tree.point_order != NULL and tree.n_points == num_points
    if tree_holds_these_points:
        order_ptr = tree.point_order
    else:
        order = _scratch("bh/order", (num_points,), np.intp)
        _spatial_order(embedding, order)
        order_ptr = &order[0]

    # A shared descent is only available where the points being moved are the
    # ones the tree was built over. Embedding new points into a fixed reference
    # leaves the caller free to hand over any subset at all, and a block would
    # make each point's result depend on which others came with it -- so those
    # descend one point at a time, as they always have.
    if n_dims == 2 and trav2 != NULL and tree_holds_these_points:
        num_blocks = (num_points + BH_BLOCK - 1) // BH_BLOCK

        with nogil, parallel(num_threads=num_threads):
            pull_capacity = 1024
            pulls = <Pull2 *>malloc(pull_capacity * sizeof(Pull2))
            if not pulls:
                with gil:
                    raise MemoryError()

            for b in prange(num_blocks, schedule="guided"):
                lo = b * BH_BLOCK
                hi = lo + BH_BLOCK
                if hi > num_points:
                    hi = num_points

                _bh_block_bounds_2d(
                    embedding, order_ptr, lo, hi,
                    &center_0, &center_1, &half_0, &half_1,
                )
                count = _bh_block_list_2d(
                    trav2, tree.num_nodes, center_0, center_1, half_0, half_1,
                    theta_sq, &pulls, &pull_capacity,
                )
                if count < 0:
                    failures += 1
                else:
                    _bh_block_sum_2d(
                        pulls, count, embedding, gradient, sum_Qi,
                        order_ptr, lo, hi, dof,
                    )

            free(pulls)

        if failures > 0:
            raise MemoryError()
    else:
        with nogil, parallel(num_threads=num_threads):
            stack = <Py_ssize_t *>malloc(stack_size * sizeof(Py_ssize_t))
            if not stack:
                with gil:
                    raise MemoryError()

            if n_dims == 2 and trav2 != NULL:
                for j in prange(num_points, schedule="guided"):
                    i = order_ptr[j]
                    sum_Qi[i] = _estimate_negative_gradient_single_2d(
                        trav2, tree.num_nodes,
                        &embedding[i, 0], &gradient[i, 0], theta_sq, dof,
                    )
            else:
                for j in prange(num_points, schedule="guided"):
                    i = order_ptr[j]
                    sum_Qi[i] = _estimate_negative_gradient_single(
                        nodes, center_of_mass, n_dims, stack,
                        &embedding[i, 0], &gradient[i, 0], theta_sq, dof,
                    )

            free(stack)

    for i in range(num_points):
        sum_Q += sum_Qi[i]

    # Normalize q_{ij}s
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            if pairwise_normalization:
                gradient[i, j] /= sum_Q + EPSILON
            else:
                gradient[i, j] /= sum_Qi[i] + EPSILON

    return sum_Q


cdef void _spatial_order(double[:, ::1] points, Py_ssize_t[::1] order):
    """Order point indices along a Morton curve over a coarse regular grid.

    The grid is only used to group points that lie close together; the exact
    resolution is irrelevant, so a fixed 16-bit key is split evenly across the
    dimensions. Above three dimensions the curve stops being a useful proxy for
    locality and the input order is kept.
    """
    cdef:
        Py_ssize_t n = points.shape[0]
        Py_ssize_t n_dims = points.shape[1]
        Py_ssize_t bits, n_cells, i, d, b, cell, key
        double lo, hi, scale
        Py_ssize_t[::1] keys
        Py_ssize_t[::1] counts

    if n_dims > 3 or n == 0:
        for i in range(n):
            order[i] = i
        return

    bits = 16 // n_dims
    n_cells = 1 << (bits * n_dims)

    keys = np.zeros(n, dtype=np.intp)
    counts = np.zeros(n_cells + 1, dtype=np.intp)

    for d in range(n_dims):
        lo = INFINITY
        hi = -INFINITY
        for i in range(n):
            if points[i, d] < lo:
                lo = points[i, d]
            if points[i, d] > hi:
                hi = points[i, d]

        scale = 0 if hi <= lo else ((1 << bits) - 1) / (hi - lo)

        for i in range(n):
            cell = <Py_ssize_t>((points[i, d] - lo) * scale)
            if cell < 0:
                cell = 0
            elif cell >= (1 << bits):
                cell = (1 << bits) - 1

            key = keys[i]
            for b in range(bits):
                key |= ((cell >> b) & 1) << (b * n_dims + d)
            keys[i] = key

    for i in range(n):
        counts[keys[i] + 1] += 1
    for i in range(n_cells):
        counts[i + 1] += counts[i]
    for i in range(n):
        order[counts[keys[i]]] = i
        counts[keys[i]] += 1


cdef void _bh_block_bounds_2d(
    double[:, ::1] embedding,
    Py_ssize_t * order,
    Py_ssize_t lo,
    Py_ssize_t hi,
    double * center_0,
    double * center_1,
    double * half_0,
    double * half_1,
) noexcept nogil:
    """The center of a block's bounding box, and its half-extent on each axis."""
    cdef:
        Py_ssize_t b, i = order[lo]
        double x_lo = embedding[i, 0], x_hi = embedding[i, 0]
        double y_lo = embedding[i, 1], y_hi = embedding[i, 1]
        double x, y, half_x, half_y

    for b in range(lo + 1, hi):
        i = order[b]
        x = embedding[i, 0]
        y = embedding[i, 1]
        if x < x_lo:
            x_lo = x
        elif x > x_hi:
            x_hi = x
        if y < y_lo:
            y_lo = y
        elif y > y_hi:
            y_hi = y

    half_x = (x_hi - x_lo) / 2
    half_y = (y_hi - y_lo) / 2
    center_0[0] = x_lo + half_x
    center_1[0] = y_lo + half_y
    half_0[0] = half_x
    half_1[0] = half_y


cdef Py_ssize_t _bh_block_list_2d(
    TravNode2 * trav,
    Py_ssize_t num_nodes,
    double center_0,
    double center_1,
    double half_0,
    double half_1,
    double theta_sq,
    Pull2 ** pulls,
    Py_ssize_t * capacity,
) noexcept nogil:
    """Collect the cells a whole block may treat as single bodies, in pre-order.

    A cell is accepted when its size over `d_near` falls below theta, where
    `d_near` is the distance from the cell's center of mass to the block's
    bounding box. Every point in the block is at least that far from the center
    of mass, so accepting a cell here implies accepting it for each of those
    points on its own, and the list refines what per-point descent produces
    rather than coarsening it.

    The box is measured per axis, which gives `d_near` squared without a square
    root and is tighter than the sphere around it. A center of mass inside the
    box has no distance to it, so such a cell is always opened. Leaves have
    nothing to open and are accepted whatever the criterion says, which is what
    terminates the walk.

    Returns the length of the list, or -1 if the buffer could not be grown.
    """
    cdef:
        Py_ssize_t node = 0, count = 0, escape
        double edge_0, edge_1
        Pull2 * grown

    while node < num_nodes:
        escape = trav[node].escape

        if escape > node + 1:
            edge_0 = fabs(trav[node].com_x - center_0) - half_0
            edge_1 = fabs(trav[node].com_y - center_1) - half_1
            if edge_0 < 0:
                edge_0 = 0
            if edge_1 < 0:
                edge_1 = 0
            if trav[node].length_sq >= theta_sq * (edge_0 * edge_0 + edge_1 * edge_1):
                node = node + 1
                continue

        if count == capacity[0]:
            grown = <Pull2 *>realloc(pulls[0], 2 * capacity[0] * sizeof(Pull2))
            if not grown:
                return -1
            pulls[0] = grown
            capacity[0] = 2 * capacity[0]

        pulls[0][count].com_x = trav[node].com_x
        pulls[0][count].com_y = trav[node].com_y
        pulls[0][count].num_points = trav[node].num_points
        pulls[0][count].is_leaf = escape == node + 1
        count = count + 1
        node = escape

    return count


cdef void _bh_block_sum_2d(
    Pull2 * pulls,
    Py_ssize_t count,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double[::1] sum_Qi,
    Py_ssize_t * order,
    Py_ssize_t lo,
    Py_ssize_t hi,
    double dof,
) noexcept nogil:
    """Sum one accepted-cell list into each point of the block that produced it.

    Cell for cell this is what the per-point traversal computes, in the order it
    would have computed it, so a block whose list matches a point's own descent
    reproduces that point's gradient to the last bit.
    """
    cdef:
        Py_ssize_t b, k, i
        double p_0, p_1, com_0, com_1, diff_0, diff_1
        double distance, q_ij, num_points, sum_Q
        double grad_0, grad_1
        bint dof_is_one = dof == 1

    for b in range(lo, hi):
        i = order[b]
        p_0 = embedding[i, 0]
        p_1 = embedding[i, 1]
        grad_0 = 0
        grad_1 = 0
        sum_Q = 0

        for k in range(count):
            com_0 = pulls[k].com_x
            com_1 = pulls[k].com_y
            diff_0 = com_0 - p_0
            diff_1 = com_1 - p_1
            distance = EPSILON + diff_0 * diff_0 + diff_1 * diff_1
            num_points = pulls[k].num_points

            if pulls[k].is_leaf and fabs(diff_0) < EPSILON \
                    and fabs(diff_1) < EPSILON:
                # A leaf sitting on the point holds the point itself, which
                # interacts with everything but itself. The rest of what it
                # holds are duplicates of it, and those do interact
                num_points = num_points - 1

            if dof_is_one:
                q_ij = 1 / (1 + distance)
            else:
                q_ij = 1 / (1 + distance / dof) ** dof

            sum_Q = sum_Q + num_points * q_ij

            if dof_is_one:
                q_ij = q_ij * q_ij
            else:
                q_ij = q_ij ** ((dof + 1) / dof)

            grad_0 = grad_0 - num_points * q_ij * (p_0 - com_0)
            grad_1 = grad_1 - num_points * q_ij * (p_1 - com_1)

        gradient[i, 0] += grad_0
        gradient[i, 1] += grad_1
        sum_Qi[i] = sum_Q


cdef double _estimate_negative_gradient_single_2d(
    TravNode2 * trav,
    Py_ssize_t num_nodes,
    double * point,
    double * gradient,
    double theta_sq,
    double dof,
) noexcept nogil:
    """Two-dimensional case of `_estimate_negative_gradient_single`.

    Worth writing out separately for two reasons. The point coordinates and the
    two gradient accumulators stay in registers for the whole traversal, rather
    than being re-read and written back at every node that gets summarized. And
    and the nodes are laid out in the order a depth-first descent reaches them,
    so the traversal needs no stack: descending is a step to the next node, and
    summarizing is a jump to the index past the subtree. It visits the same
    nodes in the same order that pushing children onto a stack would.
    """
    cdef:
        Py_ssize_t node = 0, escape
        double distance, q_ij, num_points, sum_Q = 0
        double p_0 = point[0], p_1 = point[1]
        double com_0, com_1, diff_0, diff_1
        double grad_0 = 0, grad_1 = 0
        bint dof_is_one = dof == 1

    while node < num_nodes:
        com_0 = trav[node].com_x
        com_1 = trav[node].com_y

        diff_0 = com_0 - p_0
        diff_1 = com_1 - p_1
        distance = EPSILON + diff_0 * diff_0 + diff_1 * diff_1

        escape = trav[node].escape
        num_points = trav[node].num_points
        if escape > node + 1:
            if trav[node].length_sq >= theta_sq * distance:
                node = node + 1
                continue
        elif fabs(diff_0) < EPSILON and fabs(diff_1) < EPSILON:
            # A leaf sitting on the point holds the point itself, which
            # interacts with everything but itself. The rest of what it holds
            # are duplicates of it, and those do interact
            num_points = num_points - 1

        node = escape

        if dof_is_one:
            q_ij = 1 / (1 + distance)
        else:
            q_ij = 1 / (1 + distance / dof) ** dof

        sum_Q = sum_Q + num_points * q_ij

        if dof_is_one:
            q_ij = q_ij * q_ij
        else:
            q_ij = q_ij ** ((dof + 1) / dof)

        grad_0 = grad_0 - num_points * q_ij * (p_0 - com_0)
        grad_1 = grad_1 - num_points * q_ij * (p_1 - com_1)

    gradient[0] += grad_0
    gradient[1] += grad_1

    return sum_Q


cdef double _estimate_negative_gradient_single(
    Node * nodes,
    double * center_of_mass,
    Py_ssize_t n_dims,
    Py_ssize_t * stack,
    double * point,
    double * gradient,
    double theta_sq,
    double dof,
) noexcept nogil:
    """Accumulate one point's repulsive force, returning its share of sum_Q.

    Nodes are taken off the stack in the same order a recursive descent would
    reach them, so the summation order -- and with it the result down to the
    last bit -- is unchanged.
    """
    cdef:
        Py_ssize_t top = 1, node, first_child, num_children, c, d
        double distance, q_ij, tmp, num_points, sum_Q = 0
        double * com
        bint dof_is_one = dof == 1

    stack[0] = 0

    while top > 0:
        top = top - 1
        node = stack[top]
        com = &center_of_mass[node * n_dims]

        # Compute the squared euclidean distance in the embedding space from the
        # new point to the center of mass
        distance = EPSILON
        for d in range(n_dims):
            tmp = com[d] - point[d]
            distance = distance + (tmp * tmp)

        num_points = nodes[node].num_points
        num_children = nodes[node].num_children
        if num_children > 0:
            # Check whether we can use this node as a summary
            if nodes[node].length_sq >= theta_sq * distance:
                first_child = nodes[node].first_child
                # Pushed in reverse so that the first child comes off the stack
                # first
                for c in range(num_children - 1, -1, -1):
                    stack[top] = first_child + c
                    top = top + 1
                continue
        elif is_close(com, point, EPSILON, n_dims):
            # A leaf sitting on the point holds the point itself, which
            # interacts with everything but itself. The rest of what it holds
            # are duplicates of it, and those do interact
            num_points = num_points - 1

        if dof_is_one:
            q_ij = 1 / (1 + distance)
        else:
            q_ij = 1 / (1 + distance / dof) ** dof

        sum_Q = sum_Q + num_points * q_ij

        # These two expressions are the same, but multiplication with itself is
        # faster (dof=1: (1 + 1) / 1 = 2
        if dof_is_one:
            q_ij = q_ij * q_ij
        else:
            q_ij = q_ij ** ((dof + 1) / dof)

        for d in range(n_dims):
            gradient[d] -= num_points * q_ij * (point[d] - com[d])

    return sum_Q


# The kernel matrix is rebuilt every iteration because the grid follows the
# embedding's extent, and at 45k points its quadrant is a 568x568 array.
# Reusing the allocation avoids both the allocation and the page faults that
# come with touching fresh memory. One buffer per kernel, since when dof != 1
# the plain and squared kernels are both live at once.
cdef dict _kernel_buffers = {}


cdef _kernel_buffer_2d(Py_ssize_t rows, Py_ssize_t cols, bint exp1p, dtype):
    cdef object buf = _kernel_buffers.get((exp1p, dtype))
    if buf is None or buf.shape != (rows, cols):
        buf = np.zeros((rows, cols), dtype=dtype)
        _kernel_buffers[(exp1p, dtype)] = buf
    return buf


# FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d 11^e
# 13^f, where e+f is either 0 or 1, and the other exponents are arbitrary
cdef Py_ssize_t[::1] _RECOMMENDED_BOXES = np.array([
    20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50,
    52, 54, 55, 56, 60, 63, 64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96,
    98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 125, 126, 128, 130, 132, 135, 140,
    144, 147, 150, 154, 156, 160, 162, 165, 168, 175, 176, 180, 182, 189, 192, 195, 196,
    198, 200, 208, 210, 216, 220, 224, 225, 231, 234, 240, 243, 245, 250, 252, 256, 260,
    264, 270, 273, 275, 280, 288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336,
    343, 350, 351, 352, 360, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416, 420,
    432, 440, 441, 448, 450, 455, 462, 468, 480, 486, 490, 495, 500, 504, 512, 520, 525,
    528, 539, 540, 546, 550, 560, 567, 576, 585, 588, 594, 600, 616, 624, 625, 630, 637,
    640, 648, 650, 660, 672, 675, 686, 693, 700, 702, 704, 720, 728, 729, 735, 750, 756,
    768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 864, 875, 880, 882, 891, 896,
    900, 910, 924, 936, 945, 960, 972, 975, 980, 990, 1000,
], dtype=np.intp)


cdef Py_ssize_t _round_up_boxes(Py_ssize_t n_boxes) noexcept nogil:
    """The smallest box count at or above `n_boxes` that transforms well."""
    cdef Py_ssize_t i = 0
    if n_boxes >= 1000:
        return 1000
    while n_boxes > _RECOMMENDED_BOXES[i]:
        i += 1
    return _RECOMMENDED_BOXES[i]


cdef _kernel_buffer_1d(Py_ssize_t size, bint exp1p, dtype):
    cdef object buf = _kernel_buffers.get(("1d", exp1p, dtype))
    if buf is None or buf.shape[0] != size:
        buf = np.zeros(size, dtype=dtype)
        _kernel_buffers[("1d", exp1p, dtype)] = buf
    return buf


# Everything the interpolation gradient works in is proportional either to the
# number of points or to the size of the grid, and at 45k points and a
# converged embedding that is twenty megabytes an iteration. Allocating it
# fresh every time means faulting in twenty megabytes of untouched pages, which
# costs about twice what writing over the previous iteration's does, so the
# arrays are kept and handed out again.
#
# Only the last shape asked for is kept: the grid follows the embedding's
# extent and grows with it, and holding on to the shapes it has outgrown would
# waste more than the allocations cost.
cdef dict _scratch_buffers = {}


cdef _scratch(key, shape, dtype, bint zero=False):
    """A buffer of the given shape, reused between calls."""
    cdef object buf = _scratch_buffers.get(key)
    if buf is None or buf.shape != shape or buf.dtype != np.dtype(dtype):
        buf = np.empty(shape, dtype=dtype)
        _scratch_buffers[key] = buf
        if not zero:
            return buf
    if zero:
        buf[...] = 0
    return buf


cdef _padded_grid(key, Py_ssize_t grid_x, Py_ssize_t row_len,
                  Py_ssize_t n_terms, dtype):
    """A grid of `grid_x` rows laid out `row_len` wide, with zeroed padding.

    The columns beyond the grid are the transform's zero padding. Nothing ever
    writes to them, so they are zeroed once here and the grid itself is cleared
    per call by `_clear_grid`.
    """
    cdef tuple shape = (grid_x * row_len, n_terms)
    cdef object buf = _scratch_buffers.get(key)
    if buf is None or buf.shape != shape or buf.dtype != np.dtype(dtype):
        buf = np.zeros(shape, dtype=dtype)
        _scratch_buffers[key] = buf
    return buf


cdef void _clear_grid(floating * base, Py_ssize_t grid_x, Py_ssize_t grid_y,
                      Py_ssize_t row_len, Py_ssize_t n_terms,
                      Py_ssize_t num_threads) noexcept nogil:
    """Zero the grid inside a padded buffer, leaving the padding alone."""
    cdef Py_ssize_t r
    cdef size_t span = grid_y * n_terms * sizeof(floating)

    with parallel(num_threads=num_threads):
        for r in prange(grid_x, schedule="static"):
            memset(base + r * row_len * n_terms, 0, span)


cdef void _group_points_by_grid_row(
    int * point_box_idx,
    Py_ssize_t n_samples,
    Py_ssize_t n_boxes_x,
    Py_ssize_t[::1] row_start,
    Py_ssize_t[::1] sorted_idx,
):
    """Counting sort of point indices by the grid rows their box writes to.

    A box at `(box_x, box_y)` writes to the interpolation nodes in grid rows
    `box_x * n_interpolation_points` upwards and to no others, so grouping the
    points by `box_x` alone gives each group a range of rows that no other
    group touches. That is as much separation as the scatter needs, and there
    are `n_boxes_x` groups rather than one for every box, which leaves the
    groups large enough to be worth handing out even once the embedding has
    spread out far enough that a box holds a point or two.

    Fills `row_start` with each group's offset into `sorted_idx`. The sort is
    stable, so a node still sums its points in input order.
    """
    cdef:
        Py_ssize_t i, b, running = 0
        Py_ssize_t[::1] cursor = _scratch("2d/row_cursor", (n_boxes_x,), np.intp)

    for i in range(n_boxes_x + 1):
        row_start[i] = 0
    for i in range(n_samples):
        row_start[point_box_idx[i] % n_boxes_x + 1] += 1

    for b in range(n_boxes_x):
        running += row_start[b + 1]
        row_start[b + 1] = running
        cursor[b] = row_start[b]

    for i in range(n_samples):
        b = point_box_idx[i] % n_boxes_x
        sorted_idx[cursor[b]] = i
        cursor[b] += 1


cdef void _scatter_2d_parallel(
    floating[:, ::1] w_coefficients,
    double[:, ::1] x_interpolated_values,
    double[:, ::1] y_interpolated_values,
    double[:, ::1] q_j,
    int * point_box_idx,
    Py_ssize_t[::1] row_start,
    Py_ssize_t[::1] sorted_idx,
    Py_ssize_t n_boxes_x,
    Py_ssize_t n_interpolation_points,
    Py_ssize_t row_stride,
    Py_ssize_t col_offset,
    Py_ssize_t n_terms,
    Py_ssize_t num_threads,
) noexcept nogil:
    """Accumulate the w coefficients, one column of boxes per thread.

    The points arrive grouped by the column of boxes they fall in, and a column
    owns a range of grid rows, so the accumulation has no write conflicts.
    Grouping is stable, so each node sums its points in input order and the
    result is the one the serial loop produces.
    """
    cdef Py_ssize_t box_i, box_j, k, i, interp_i, interp_j, idx, d

    with parallel(num_threads=num_threads):
        for box_i in prange(n_boxes_x, schedule="guided"):
            for k in range(row_start[box_i], row_start[box_i + 1]):
                i = sorted_idx[k]
                box_j = point_box_idx[i] // n_boxes_x
                for interp_i in range(n_interpolation_points):
                    for interp_j in range(n_interpolation_points):
                        idx = (box_i * n_interpolation_points + interp_i) * row_stride + \
                              col_offset + box_j * n_interpolation_points + interp_j
                        for d in range(n_terms):
                            w_coefficients[idx, d] = w_coefficients[idx, d] + \
                                x_interpolated_values[i, interp_i] * \
                                y_interpolated_values[i, interp_j] * \
                                q_j[i, d]


cdef void _gather_2d_parallel(
    floating[:, ::1] phi,
    double[:, ::1] x_interpolated_values,
    double[:, ::1] y_interpolated_values,
    floating[:, ::1] y_tilde_values,
    int * point_box_idx,
    Py_ssize_t n_samples,
    Py_ssize_t n_boxes_x,
    Py_ssize_t n_interpolation_points,
    Py_ssize_t row_stride,
    Py_ssize_t col_offset,
    Py_ssize_t n_terms,
    Py_ssize_t num_threads,
) noexcept nogil:
    """Read the potentials back, one point per thread.

    Each point writes only its own row of `phi`, so this needs no grouping.
    """
    cdef Py_ssize_t i, box_idx, box_i, box_j, interp_i, interp_j, idx, d

    with parallel(num_threads=num_threads):
        for i in prange(n_samples, schedule="static"):
            box_idx = point_box_idx[i]
            box_i = box_idx % n_boxes_x
            box_j = box_idx // n_boxes_x
            # A point owns its row, so it can start it rather than add to it
            for d in range(n_terms):
                phi[i, d] = 0
            for interp_i in range(n_interpolation_points):
                for interp_j in range(n_interpolation_points):
                    idx = (box_i * n_interpolation_points + interp_i) * row_stride + \
                          col_offset + box_j * n_interpolation_points + interp_j
                    for d in range(n_terms):
                        phi[i, d] = phi[i, d] + \
                            x_interpolated_values[i, interp_i] * \
                            y_interpolated_values[i, interp_j] * \
                            y_tilde_values[idx, d]


cdef void _interpolate_parallel(
    double[::1] y_in_box,
    double[::1] y_tilde,
    double[::1] denominator,
    double[:, ::1] interpolated_values,
    Py_ssize_t num_threads,
) noexcept nogil:
    cdef:
        Py_ssize_t N = y_in_box.shape[0]
        Py_ssize_t n_interpolation_points = y_tilde.shape[0]
        Py_ssize_t i, j, k
        double value

    with parallel(num_threads=num_threads):
        for i in prange(N, schedule="static"):
            for j in range(n_interpolation_points):
                value = 1
                for k in range(n_interpolation_points):
                    if j != k:
                        value = value * (y_in_box[i] - y_tilde[k])
                interpolated_values[i, j] = value / denominator[j]


cdef double[::1] _lagrange_denominator(double[::1] y_tilde):
    """The constant each Lagrange basis polynomial is divided through by."""
    cdef:
        Py_ssize_t n = y_tilde.shape[0]
        double[::1] denominator = np.empty(n, dtype=float)
        Py_ssize_t i, j

    for i in range(n):
        denominator[i] = 1
        for j in range(n):
            if i != j:
                denominator[i] *= y_tilde[i] - y_tilde[j]
    return denominator


cdef void _prepare_points_2d(
    floating[:, ::1] embedding,
    int * point_box_idx,
    double[:, ::1] q_j,
    double[:, ::1] x_interpolated_values,
    double[:, ::1] y_interpolated_values,
    double[::1] y_tilde,
    double[::1] denominator,
    floating[::1] box_lower_bounds_x,
    floating[::1] box_lower_bounds_y,
    double coord_min,
    double box_width,
    Py_ssize_t n_boxes,
    Py_ssize_t n_boxes_x,
    Py_ssize_t box_offset_x,
    Py_ssize_t box_offset_y,
    Py_ssize_t n_terms,
    Py_ssize_t num_threads,
) noexcept nogil:
    """Everything a point contributes to the grid, in one pass over it.

    The box a point falls in, the charges it carries and the Lagrange weights
    for where it sits inside that box are all read off the same two
    coordinates. Taken separately they are four sweeps over the embedding for
    one pass of arithmetic, and none of them can be reached without waiting on
    the memory the previous one has just walked past.
    """
    cdef:
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_interpolation_points = y_tilde.shape[0]
        Py_ssize_t i, j, k
        int box_x_idx, box_y_idx
        double x_in_box, y_in_box, vx, vy

    with parallel(num_threads=num_threads):
        for i in prange(n_samples, schedule="static"):
            box_x_idx = <int>((embedding[i, 0] - coord_min) / box_width)
            box_y_idx = <int>((embedding[i, 1] - coord_min) / box_width)
            # The right most point maps directly into `n_boxes`, while it
            # should belong to the last box
            if box_x_idx >= n_boxes:
                box_x_idx = <int>n_boxes - 1
            if box_y_idx >= n_boxes:
                box_y_idx = <int>n_boxes - 1

            # Where the axis starts, in boxes
            box_x_idx = box_x_idx - <int>box_offset_x
            box_y_idx = box_y_idx - <int>box_offset_y

            point_box_idx[i] = box_y_idx * <int>n_boxes_x + box_x_idx

            q_j[i, 0] = 1
            q_j[i, 1] = embedding[i, 0]
            q_j[i, 2] = embedding[i, 1]
            if n_terms > 3:
                q_j[i, 3] = 1

            x_in_box = (embedding[i, 0] - box_lower_bounds_x[box_x_idx]) / box_width
            y_in_box = (embedding[i, 1] - box_lower_bounds_y[box_y_idx]) / box_width

            for j in range(n_interpolation_points):
                vx = 1
                vy = 1
                for k in range(n_interpolation_points):
                    if j != k:
                        vx = vx * (x_in_box - y_tilde[k])
                        vy = vy * (y_in_box - y_tilde[k])
                x_interpolated_values[i, j] = vx / denominator[j]
                y_interpolated_values[i, j] = vy / denominator[j]


cdef double[:, ::1] interpolate(double[::1] y_in_box, double[::1] y_tilde,
                                Py_ssize_t num_threads=1, key=None):
    """Lagrangian polynomial interpolation.

    `key` names a buffer to reuse for the result; without one the result is
    freshly allocated.
    """
    cdef Py_ssize_t N = y_in_box.shape[0]
    cdef Py_ssize_t n_interpolation_points = y_tilde.shape[0]

    cdef double[:, ::1] interpolated_values
    if key is None:
        interpolated_values = np.empty((N, n_interpolation_points), dtype=float)
    else:
        interpolated_values = _scratch(key, (N, n_interpolation_points), float)
    cdef double[::1] denominator = _lagrange_denominator(y_tilde)
    cdef Py_ssize_t i, j, k

    if num_threads > 1:
        _interpolate_parallel(
            y_in_box, y_tilde, denominator, interpolated_values, num_threads
        )
        return interpolated_values

    for i in range(N):
        for j in range(n_interpolation_points):
            interpolated_values[i, j] = 1
            for k in range(n_interpolation_points):
                if j != k:
                    interpolated_values[i, j] *= y_in_box[i] - y_tilde[k]
            interpolated_values[i, j] /= denominator[j]

    return interpolated_values


cdef void compute_kernel_tilde_1d(
    floating[::1] kernel_tilde,
    bint exp1p,
    Py_ssize_t n_interpolation_points_1d,
    double coord_min,
    double coord_spacing,
    double dof,
):
    cdef:
        double[::1] y_tilde = np.empty(n_interpolation_points_1d, dtype=float)

        Py_ssize_t i
        double y_diff, tmp, exponent
        bint dof_is_one = dof == 1

    y_tilde[0] = coord_spacing / 2 + coord_min
    for i in range(1, n_interpolation_points_1d):
        y_tilde[i] = y_tilde[i - 1] + coord_spacing

    if exp1p:
        exponent = -(dof + 1)
    else:
        exponent = -dof

    # What comes out is the half of the generating vector of the circulant the
    # transform multiplies by that a real-even transform reads; the other half
    # is this one reversed. Every entry but the first is written, and that one
    # is zero in the circulant, so a reused buffer needs no clearing
    for i in range(n_interpolation_points_1d):
        y_diff = y_tilde[0] - y_tilde[i]

        if dof_is_one:
            tmp = 1 + y_diff * y_diff
            if exp1p:
                tmp = 1 / (tmp * tmp)
            else:
                tmp = 1 / tmp
        else:
            tmp = (1 + (y_diff * y_diff) / dof) ** exponent

        kernel_tilde[n_interpolation_points_1d - i] = tmp


cpdef double estimate_negative_gradient_fft_1d(
    floating[::1] embedding,
    floating[::1] gradient,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    Py_ssize_t num_threads=1,
):
    cdef Py_ssize_t i, j, d, box_idx, n_samples = embedding.shape[0]
    cdef double y_max = -INFINITY, y_min = INFINITY

    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    for i in range(n_samples):
        if embedding[i] < y_min:
            y_min = embedding[i]
        elif embedding[i] > y_max:
            y_max = embedding[i]

    cdef int n_boxes = <int>fmax(min_num_intervals, (y_max - y_min) / ints_in_interval)
    # FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    # 11^e 13^f, where e+f is either 0 or 1, and the other exponents are arbitrary
    cdef list recommended_boxes = [
        20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50,
        52, 54, 55, 56, 60, 63, 64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96,
        98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 125, 126, 128, 130, 132, 135, 140,
        144, 147, 150, 154, 156, 160, 162, 165, 168, 175, 176, 180, 182, 189, 192, 195, 196,
        198, 200, 208, 210, 216, 220, 224, 225, 231, 234, 240, 243, 245, 250, 252, 256, 260,
        264, 270, 273, 275, 280, 288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336,
        343, 350, 351, 352, 360, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416, 420,
        432, 440, 441, 448, 450, 455, 462, 468, 480, 486, 490, 495, 500, 504, 512, 520, 525,
        528, 539, 540, 546, 550, 560, 567, 576, 585, 588, 594, 600, 616, 624, 625, 630, 637,
        640, 648, 650, 660, 672, 675, 686, 693, 700, 702, 704, 720, 728, 729, 735, 750, 756,
        768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 864, 875, 880, 882, 891, 896,
        900, 910, 924, 936, 945, 960, 972, 975, 980, 990, 1000 
    ]
    if n_boxes < recommended_boxes[205]:
        i = 0
        while n_boxes > recommended_boxes[i]:
            i += 1
        n_boxes = recommended_boxes[i]
    else:
        n_boxes = 1000

    cdef double box_width = (y_max - y_min) / n_boxes

    # Compute the box bounds
    cdef floating[::1] box_lower_bounds = np.empty(n_boxes, dtype=_real_dtype(<floating>0))
    cdef floating[::1] box_upper_bounds = np.empty(n_boxes, dtype=_real_dtype(<floating>0))
    for box_idx in range(n_boxes):
        box_lower_bounds[box_idx] = box_idx * box_width + y_min
        box_upper_bounds[box_idx] = (box_idx + 1) * box_width + y_min

    # Determine which box each point belongs to
    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))
    for i in range(n_samples):
        box_idx = <int>((embedding[i] - y_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_idx >= n_boxes:
            box_idx = n_boxes - 1

        point_box_idx[i] = box_idx

    cdef int n_interpolation_points_1d = n_interpolation_points * n_boxes
    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # Evaluate the the squared cauchy kernel at the interpolation nodes
    cdef floating[::1] sq_kernel_tilde = _kernel_buffer_1d(
        n_interpolation_points_1d + 1, True, _real_dtype(<floating>0)
    )
    compute_kernel_tilde_1d(
        sq_kernel_tilde, True, n_interpolation_points_1d, y_min, h * box_width, dof
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef floating[::1] kernel_tilde
    if dof != 1:
        kernel_tilde = _kernel_buffer_1d(
            n_interpolation_points_1d + 1, False, _real_dtype(<floating>0)
        )
        compute_kernel_tilde_1d(
            kernel_tilde, False, n_interpolation_points_1d, y_min, h * box_width, dof
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    #
    # With dof = 1 the normalization term needs a third potential, for
    # q_j = y_j^2. The interpolated operator is symmetric, so the sum of that
    # potential over all points equals sum_j y_j^2 phi_0[j] -- a weighted sum
    # of a potential already being computed. Carrying the term through the
    # transform would cost a third of it for a number available for free.
    cdef int n_terms = 3 if dof != 1 else 2
    cdef double[:, ::1] q_j = np.empty((n_samples, n_terms), dtype=float)
    if dof != 1:
        for i in range(n_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = embedding[i]
            q_j[i, 2] = 1
    else:
        for i in range(n_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = embedding[i]

    # Compute the relative position of each reference point in its box
    cdef double[::1] y_in_box = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        y_in_box[i] = (embedding[i] - box_lower_bounds[box_idx]) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] interpolated_values = interpolate(y_in_box, y_tilde, num_threads)

    # Actually compute w_{ij}s
    cdef floating[:, ::1] w_coefficients = np.zeros((n_interpolation_points_1d, n_terms), dtype=_real_dtype(<floating>0))
    for i in range(n_samples):
        box_idx = point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                w_coefficients[box_idx + j, d] += interpolated_values[i, j] * q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef floating[:, ::1] y_tilde_values = np.empty((n_interpolation_points_1d, n_terms), dtype=_real_dtype(<floating>0))
    if dof != 1:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients[:, :2], y_tilde_values[:, :2], num_threads)
        matrix_multiply_fft_1d(kernel_tilde, w_coefficients[:, 2:], y_tilde_values[:, 2:], num_threads)
    else:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)

    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    cdef floating[:, ::1] phi = np.zeros((n_samples, n_terms), dtype=_real_dtype(<floating>0))
    for i in range(n_samples):
        box_idx = point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                phi[i, d] += interpolated_values[i, j] * y_tilde_values[box_idx + j, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double sum_Q = 0
    if dof != 1:
        for i in range(n_samples):
            sum_Q += phi[i, 2]
    else:
        for i in range(n_samples):
            sum_Q += (1 + 2 * embedding[i] ** 2) * phi[i, 0] - \
                     2 * embedding[i] * phi[i, 1]

    sum_Q -= n_samples

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i] -= (embedding[i] * phi[i, 0] - phi[i, 1]) / (sum_Q + EPSILON)

    return sum_Q


cpdef tuple prepare_negative_gradient_fft_interpolation_grid_1d(
    floating[::1] reference_embedding,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    double padding=0,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_reference_samples = reference_embedding.shape[0]

        double y_max = -INFINITY, y_min = INFINITY

    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    # First, check the existing embedding
    for i in range(n_reference_samples):
        if reference_embedding[i] < y_min:
            y_min = reference_embedding[i]
        elif reference_embedding[i] > y_max:
            y_max = reference_embedding[i]

    # We assume here that the embedding is centered and we want to generate an
    # equal grid in both negative and positive lines
    if fabs(y_min) > fabs(y_max):
        coord_max = -y_min
    elif fabs(y_max) > fabs(y_min):
        coord_min = -y_max

    # Apply padding to the min/max coordinates
    y_min *= 1 + padding
    y_max *= 1 + padding

    cdef int n_boxes = <int>fmax(min_num_intervals, (y_max - y_min) / ints_in_interval)
    cdef double box_width = (y_max - y_min) / n_boxes

    # Compute the box bounds
    cdef floating[::1] box_lower_bounds = np.empty(n_boxes, dtype=_real_dtype(<floating>0))
    cdef floating[::1] box_upper_bounds = np.empty(n_boxes, dtype=_real_dtype(<floating>0))
    for box_idx in range(n_boxes):
        box_lower_bounds[box_idx] = box_idx * box_width + y_min
        box_upper_bounds[box_idx] = (box_idx + 1) * box_width + y_min

    # Determine which box each reference point belongs to
    cdef int *reference_point_box_idx = <int *>PyMem_Malloc(n_reference_samples * sizeof(int))
    for i in range(n_reference_samples):
        box_idx = <int>((reference_embedding[i] - y_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_idx >= n_boxes:
            box_idx = n_boxes - 1

        reference_point_box_idx[i] = box_idx

    cdef int n_interpolation_points_1d = n_interpolation_points * n_boxes
    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # Evaluate the squared cauchy kernel at the interpolation nodes
    cdef floating[::1] sq_kernel_tilde = _kernel_buffer_1d(
        n_interpolation_points_1d + 1, True, _real_dtype(<floating>0)
    )
    compute_kernel_tilde_1d(
        sq_kernel_tilde, True, n_interpolation_points_1d, y_min, h * box_width, dof
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef floating[::1] kernel_tilde
    if dof != 1:
        kernel_tilde = _kernel_buffer_1d(
            n_interpolation_points_1d + 1, False, _real_dtype(<floating>0)
        )
        compute_kernel_tilde_1d(
            kernel_tilde, False, n_interpolation_points_1d, y_min, h * box_width, dof
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    cdef int n_terms = 3
    cdef double[:, ::1] q_j = np.empty((n_reference_samples, n_terms), dtype=float)
    if dof != 1:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i]
            q_j[i, 2] = 1
    else:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i]
            q_j[i, 2] = reference_embedding[i] ** 2

    # Compute the relative position of each reference point in its box
    cdef double[::1] reference_y_in_box = np.empty(n_reference_samples, dtype=float)
    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i]
        reference_y_in_box[i] = (reference_embedding[i] - box_lower_bounds[box_idx]) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] reference_interpolated_values = interpolate(reference_y_in_box, y_tilde, num_threads)

    # Actually compute w_{ij}s
    cdef floating[:, ::1] w_coefficients = np.zeros((n_interpolation_points_1d, n_terms), dtype=_real_dtype(<floating>0))
    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                w_coefficients[box_idx + j, d] += reference_interpolated_values[i, j] * q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef floating[:, ::1] y_tilde_values = np.empty((n_interpolation_points_1d, n_terms), dtype=_real_dtype(<floating>0))
    if dof != 1:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients[:, :2], y_tilde_values[:, :2], num_threads)
        matrix_multiply_fft_1d(kernel_tilde, w_coefficients[:, 2:], y_tilde_values[:, 2:], num_threads)
    else:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)

    PyMem_Free(reference_point_box_idx)

    return np.asarray(y_tilde_values), np.asarray(box_lower_bounds)


cpdef double estimate_negative_gradient_fft_1d_with_grid(
    floating[::1] embedding,
    floating[::1] gradient,
    floating[:, ::1] y_tilde_values,
    floating[::1] box_lower_bounds,
    Py_ssize_t n_interpolation_points,
    double dof,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_terms = y_tilde_values.shape[1]
        Py_ssize_t n_boxes = box_lower_bounds.shape[0]
        double y_min = box_lower_bounds[0]
        double box_width = box_lower_bounds[1] - box_lower_bounds[0]

    # Determine which box each point belongs to
    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))
    for i in range(n_samples):
        box_idx = <int>((embedding[i] - y_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_idx >= n_boxes:
            box_idx = n_boxes - 1

        point_box_idx[i] = box_idx

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    # Compute the relative position of each new embedding point in its box
    cdef double[::1] y_in_box = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        y_in_box[i] = (embedding[i] - box_lower_bounds[box_idx]) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] interpolated_values = interpolate(y_in_box, y_tilde)

    # Actually compute \tilde{\phi(y_i)}
    cdef floating[:, ::1] phi = np.zeros((n_samples, n_terms), dtype=_real_dtype(<floating>0))
    for i in range(n_samples):
        box_idx = point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                phi[i, d] += interpolated_values[i, j] * y_tilde_values[box_idx + j, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double[::1] sum_Qi = np.empty(n_samples, dtype=float)
    if dof != 1:
        for i in range(n_samples):
            sum_Qi[i] = phi[i, 2]
    else:
        for i in range(n_samples):
            sum_Qi[i] = (1 + embedding[i] ** 2) * phi[i, 0] - \
                         2 * embedding[i] * phi[i, 1] + \
                         phi[i, 2]

    cdef double sum_Q = 0
    for i in range(n_samples):
        sum_Q += sum_Qi[i]

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i] -= (embedding[i] * phi[i, 0] - phi[i, 1]) / (sum_Qi[i] + EPSILON)

    return sum_Q


cdef void compute_kernel_tilde_2d(
    floating[:, ::1] kernel_tilde,
    bint exp1p,
    Py_ssize_t grid_x,
    Py_ssize_t grid_y,
    double coord_spacing,
    double dof,
):
    """Evaluate the kernel at every pair of interpolation nodes.

    What comes out is one quadrant of the generating matrix of the circulant
    the transform multiplies by. The whole matrix is that quadrant mirrored
    about its first row and column, and a real-even transform reads the
    quadrant directly, so the other three are never formed.

    `exp1p` selects the squared Cauchy kernel over the plain one. It is a flag
    rather than a function pointer because this is a quarter of a million
    evaluations per call and an indirect call cannot be inlined; with the
    kernel written out here, the `dof` branch is hoisted out of the loop and
    the whole body collapses to a handful of instructions.
    """
    cdef:
        Py_ssize_t i, j
        double y_diff, x_diff, tmp, exponent
        bint dof_is_one = dof == 1
        floating * row

    if exp1p:
        exponent = -(dof + 1)
    else:
        exponent = -dof

    # Everything outside the first row and column is written, and those two
    # are zero in the circulant, so a reused buffer needs no clearing
    for i in range(grid_x):
        y_diff = -i * coord_spacing
        row = &kernel_tilde[grid_x - i, 0]

        for j in range(grid_y):
            x_diff = -j * coord_spacing

            if dof_is_one:
                tmp = 1 + y_diff * y_diff + x_diff * x_diff
                if exp1p:
                    tmp = 1 / (tmp * tmp)
                else:
                    tmp = 1 / tmp
            else:
                tmp = (1 + (y_diff * y_diff + x_diff * x_diff) / dof) ** exponent

            row[grid_y - j] = tmp


cpdef double estimate_negative_gradient_fft_2d(
    floating[:, ::1] embedding,
    floating[:, ::1] gradient,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_dims = embedding.shape[1]

        double coord_min_x = INFINITY, coord_max_x = -INFINITY
        double coord_min_y = INFINITY, coord_max_y = -INFINITY
        double extent

    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    for i in range(n_samples):
        if embedding[i, 0] < coord_min_x:
            coord_min_x = embedding[i, 0]
        if embedding[i, 0] > coord_max_x:
            coord_max_x = embedding[i, 0]
        if embedding[i, 1] < coord_min_y:
            coord_min_y = embedding[i, 1]
        if embedding[i, 1] > coord_max_y:
            coord_max_y = embedding[i, 1]

    # The box width comes off the extent of the embedding as a whole, and the
    # boxes are laid out from `coord_min` as they always were
    cdef double coord_min = fmin(coord_min_x, coord_min_y)
    extent = fmax(coord_max_x, coord_max_y) - coord_min
    cdef Py_ssize_t n_boxes = _round_up_boxes(
        <Py_ssize_t>fmax(min_num_intervals, extent / ints_in_interval))
    cdef double box_width = extent / n_boxes

    # An embedding that is not square leaves whole boxes at the ends of an axis
    # with nothing in them, and each axis can start and stop where its own
    # points do. Both ends move by a whole number of boxes, so every point
    # keeps the box it was in and its place inside it: the grid loses only the
    # boxes that were empty.
    cdef Py_ssize_t box_offset_x = 0, box_offset_y = 0
    cdef Py_ssize_t hi_x = n_boxes - 1, hi_y = n_boxes - 1
    if box_width > 0:
        box_offset_x = <Py_ssize_t>((coord_min_x - coord_min) / box_width)
        box_offset_y = <Py_ssize_t>((coord_min_y - coord_min) / box_width)
        hi_x = <Py_ssize_t>((coord_max_x - coord_min) / box_width)
        hi_y = <Py_ssize_t>((coord_max_y - coord_min) / box_width)
        if hi_x >= n_boxes:
            hi_x = n_boxes - 1
        if hi_y >= n_boxes:
            hi_y = n_boxes - 1

    cdef Py_ssize_t n_boxes_x = _round_up_boxes(hi_x + 1 - box_offset_x)
    cdef Py_ssize_t n_boxes_y = _round_up_boxes(hi_y + 1 - box_offset_y)

    cdef floating[::1] box_lower_bounds_x = _scratch(
        "2d/box_lo_x", (n_boxes_x,), _real_dtype(<floating>0))
    cdef floating[::1] box_lower_bounds_y = _scratch(
        "2d/box_lo_y", (n_boxes_y,), _real_dtype(<floating>0))
    for i in range(n_boxes_x):
        box_lower_bounds_x[i] = (i + box_offset_x) * box_width + coord_min
    for i in range(n_boxes_y):
        box_lower_bounds_y[i] = (i + box_offset_y) * box_width + coord_min

    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    cdef:
        Py_ssize_t grid_x = n_boxes_x * n_interpolation_points
        Py_ssize_t grid_y = n_boxes_y * n_interpolation_points

    # Evaluate the squared cauchy kernel at the interpolation nodes
    cdef floating[:, ::1] sq_kernel_tilde = _kernel_buffer_2d(
        grid_x + 1, grid_y + 1, True, _real_dtype(<floating>0)
    )
    compute_kernel_tilde_2d(
        sq_kernel_tilde, True, grid_x, grid_y, h * box_width, dof,
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef floating[:, ::1] kernel_tilde
    if dof != 1:
        kernel_tilde = _kernel_buffer_2d(
            grid_x + 1, grid_y + 1, False, _real_dtype(<floating>0)
        )
        compute_kernel_tilde_2d(
            kernel_tilde, False, grid_x, grid_y, h * box_width, dof,
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    #
    # With dof = 1 the normalization term needs a fourth potential, for
    # q_j = |y_j|^2. The interpolated operator is symmetric, so the sum of that
    # potential over all points equals sum_j |y_j|^2 phi_0[j] -- a weighted sum
    # of a potential already being computed. Carrying the term through the
    # transform would cost a quarter of it for a number available for free.
    cdef int n_terms = 4 if dof != 1 else 3

    # Which box a point falls in, the charges it carries and the Lagrange
    # weights for where it sits inside its box all come off the same two
    # coordinates, so they are read once
    cdef:
        double[:, ::1] q_j = _scratch("2d/q_j", (n_samples, n_terms), float)
        double[:, ::1] x_interpolated_values = _scratch(
            "2d/interp_x", (n_samples, n_interpolation_points), float)
        double[:, ::1] y_interpolated_values = _scratch(
            "2d/interp_y", (n_samples, n_interpolation_points), float)
        double[::1] denominator = _lagrange_denominator(y_tilde)

    with nogil:
        _prepare_points_2d(
            embedding, point_box_idx, q_j, x_interpolated_values,
            y_interpolated_values, y_tilde, denominator, box_lower_bounds_x,
            box_lower_bounds_y, coord_min, box_width, n_boxes, n_boxes_x,
            box_offset_x, box_offset_y, n_terms, num_threads,
        )

    # Actually compute w_{ij}s
    #
    # The grid is laid out along the rows of the transform rather than along
    # rows of its own: `n_fft_y` wide instead of `grid_y`, with the
    # zero padding the circulant embedding needs already in place. That is what
    # lets the transform write its result straight into these arrays, and
    # writing over them is worth about twice what writing into fresh ones is.
    # The padding is written once, so only the grid itself is cleared per call.
    cdef:
        Py_ssize_t box_i, box_j, interp_i, interp_j, idx
        Py_ssize_t n_fft_y = 2 * grid_y
        floating[:, ::1] w_coefficients = _padded_grid(
            "2d/w", grid_x, n_fft_y, n_terms, _real_dtype(<floating>0))
        Py_ssize_t[::1] row_start
        Py_ssize_t[::1] sorted_idx

    with nogil:
        _clear_grid(&w_coefficients[0, 0], grid_x, grid_y, n_fft_y, n_terms,
                    num_threads)

    if num_threads > 1:
        row_start = _scratch("2d/row_start", (n_boxes_x + 1,), np.intp)
        sorted_idx = _scratch("2d/sorted_idx", (n_samples,), np.intp)
        _group_points_by_grid_row(
            point_box_idx, n_samples, n_boxes_x, row_start, sorted_idx,
        )
        with nogil:
            _scatter_2d_parallel(
                w_coefficients, x_interpolated_values, y_interpolated_values,
                q_j, point_box_idx, row_start, sorted_idx, n_boxes_x,
                n_interpolation_points, n_fft_y, 0, n_terms, num_threads,
            )
    else:
        for i in range(n_samples):
            box_idx = point_box_idx[i]
            box_i = box_idx % n_boxes_x
            box_j = box_idx // n_boxes_x
            for interp_i in range(n_interpolation_points):
                for interp_j in range(n_interpolation_points):
                    idx = (box_i * n_interpolation_points + interp_i) * \
                          n_fft_y + \
                          (box_j * n_interpolation_points) + \
                          interp_j
                    for d in range(n_terms):
                        w_coefficients[idx, d] += \
                            x_interpolated_values[i, interp_i] * \
                            y_interpolated_values[i, interp_j] * \
                            q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef floating[:, ::1] y_tilde_values = _scratch(
        "2d/y_tilde_values", (grid_x * n_fft_y, n_terms),
        _real_dtype(<floating>0))
    if dof != 1:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients[:, :3], y_tilde_values[:, :3], num_threads)
        matrix_multiply_fft_2d(kernel_tilde, w_coefficients[:, 3:], y_tilde_values[:, 3:], num_threads)
    else:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)

    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    cdef floating[:, ::1] phi = _scratch(
        "2d/phi", (n_samples, n_terms), _real_dtype(<floating>0))
    if num_threads > 1:
        with nogil:
            _gather_2d_parallel(
                phi, x_interpolated_values, y_interpolated_values,
                y_tilde_values, point_box_idx, n_samples, n_boxes_x,
                n_interpolation_points, n_fft_y, grid_y, n_terms, num_threads,
            )
    else:
        for i in range(n_samples):
            box_idx = point_box_idx[i]
            box_i = box_idx % n_boxes_x
            box_j = box_idx // n_boxes_x
            for d in range(n_terms):
                phi[i, d] = 0
            for interp_i in range(n_interpolation_points):
                for interp_j in range(n_interpolation_points):
                    idx = (box_i * n_interpolation_points + interp_i) * \
                          n_fft_y + grid_y + \
                          (box_j * n_interpolation_points) + \
                          interp_j
                    for d in range(n_terms):
                        phi[i, d] += x_interpolated_values[i, interp_i] * \
                                     y_interpolated_values[i, interp_j] * \
                                     y_tilde_values[idx, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double sum_Q = 0, y1, y2
    if dof != 1:
        for i in range(n_samples):
            sum_Q += phi[i, 3]
    else:
        for i in range(n_samples):
            y1 = embedding[i, 0]
            y2 = embedding[i, 1]

            sum_Q += (1 + 2 * (y1 ** 2 + y2 ** 2)) * phi[i, 0] - \
                     2 * (y1 * phi[i, 1] + y2 * phi[i, 2])

    sum_Q -= n_samples

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i, 0] -= (embedding[i, 0] * phi[i, 0] - phi[i, 1]) / (sum_Q + EPSILON)
        gradient[i, 1] -= (embedding[i, 1] * phi[i, 0] - phi[i, 2]) / (sum_Q + EPSILON)

    return sum_Q


cpdef tuple prepare_negative_gradient_fft_interpolation_grid_2d(
    floating[:, ::1] reference_embedding,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    double padding=0,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_reference_samples = reference_embedding.shape[0]

        double coord_max = -INFINITY, coord_min = INFINITY

    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    # First, check the existing embedding
    for i in range(n_reference_samples):
        if reference_embedding[i, 0] < coord_min:
            coord_min = reference_embedding[i, 0]
        elif reference_embedding[i, 0] > coord_max:
            coord_max = reference_embedding[i, 0]
        if reference_embedding[i, 1] < coord_min:
            coord_min = reference_embedding[i, 1]
        elif reference_embedding[i, 1] > coord_max:
            coord_max = reference_embedding[i, 1]

    # We assume here that the embedding is centered and we want to generate an
    # equal grid in all quadrants
    if fabs(coord_min) > fabs(coord_max):
        coord_max = -coord_min
    elif fabs(coord_max) > fabs(coord_min):
        coord_min = -coord_max

    # Apply padding to the min/max coordinates
    coord_min *= 1 + padding
    coord_max *= 1 + padding

    cdef int n_boxes_1d = <int>fmax(min_num_intervals, (coord_max - coord_min) / ints_in_interval)
    cdef int n_total_boxes = n_boxes_1d ** 2
    cdef double box_width = (coord_max - coord_min) / n_boxes_1d

    # Compute the box bounds
    cdef:
        floating[::1] box_x_lower_bounds = np.empty(n_total_boxes, dtype=_real_dtype(<floating>0))
        floating[::1] box_x_upper_bounds = np.empty(n_total_boxes, dtype=_real_dtype(<floating>0))
        floating[::1] box_y_lower_bounds = np.empty(n_total_boxes, dtype=_real_dtype(<floating>0))
        floating[::1] box_y_upper_bounds = np.empty(n_total_boxes, dtype=_real_dtype(<floating>0))

    for i in range(n_boxes_1d):
        for j in range(n_boxes_1d):
            box_x_lower_bounds[i * n_boxes_1d + j] = j * box_width + coord_min
            box_x_upper_bounds[i * n_boxes_1d + j] = (j + 1) * box_width + coord_min

            box_y_lower_bounds[i * n_boxes_1d + j] = i * box_width + coord_min
            box_y_upper_bounds[i * n_boxes_1d + j] = (i + 1) * box_width + coord_min

    # Determine which box each reference point belongs to
    cdef int *reference_point_box_idx = <int *>PyMem_Malloc(n_reference_samples * sizeof(int))
    cdef int box_x_idx, box_y_idx
    for i in range(n_reference_samples):
        box_x_idx = <int>((reference_embedding[i, 0] - coord_min) / box_width)
        box_y_idx = <int>((reference_embedding[i, 1] - coord_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_x_idx >= n_boxes_1d:
            box_x_idx = n_boxes_1d - 1
        if box_y_idx >= n_boxes_1d:
            box_y_idx = n_boxes_1d - 1

        reference_point_box_idx[i] = box_y_idx * n_boxes_1d + box_x_idx

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # Evaluate the the squared cauchy kernel at the interpolation nodes
    cdef Py_ssize_t grid_1d = n_interpolation_points * n_boxes_1d
    cdef floating[:, ::1] sq_kernel_tilde = _kernel_buffer_2d(
        grid_1d + 1, grid_1d + 1, True, _real_dtype(<floating>0)
    )
    compute_kernel_tilde_2d(
        sq_kernel_tilde, True, grid_1d, grid_1d, h * box_width, dof,
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef floating[:, ::1] kernel_tilde
    if dof != 1:
        kernel_tilde = _kernel_buffer_2d(
            grid_1d + 1, grid_1d + 1, False, _real_dtype(<floating>0)
        )
        compute_kernel_tilde_2d(
            kernel_tilde, False, grid_1d, grid_1d, h * box_width, dof,
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    cdef int n_terms = 4
    cdef double[:, ::1] q_j = np.empty((n_reference_samples, n_terms), dtype=float)
    if dof != 1:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i, 0]
            q_j[i, 2] = reference_embedding[i, 1]
            q_j[i, 3] = 1
    else:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i, 0]
            q_j[i, 2] = reference_embedding[i, 1]
            q_j[i, 3] = reference_embedding[i, 0] ** 2 + reference_embedding[i, 1] ** 2

    # Compute the relative position of each reference point in its box
    cdef:
        double[::1] reference_x_in_box = np.empty(n_reference_samples, dtype=float)
        double[::1] reference_y_in_box = np.empty(n_reference_samples, dtype=float)
        double y_min, x_min

    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i]
        x_min = box_x_lower_bounds[box_idx]
        y_min = box_y_lower_bounds[box_idx]
        reference_x_in_box[i] = (reference_embedding[i, 0] - x_min) / box_width
        reference_y_in_box[i] = (reference_embedding[i, 1] - y_min) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] reference_x_interpolated_values = interpolate(reference_x_in_box, y_tilde, num_threads)
    cdef double[:, ::1] reference_y_interpolated_values = interpolate(reference_y_in_box, y_tilde, num_threads)

    # Actually compute w_{ij}s
    cdef:
        int total_interpolation_points = n_total_boxes * n_interpolation_points ** 2
        floating[:, ::1] w_coefficients = np.zeros((total_interpolation_points, n_terms), dtype=_real_dtype(<floating>0))
        Py_ssize_t box_i, box_j, interp_i, interp_j, idx

    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i]
        box_i = box_idx % n_boxes_1d
        box_j = box_idx // n_boxes_1d
        for interp_i in range(n_interpolation_points):
            for interp_j in range(n_interpolation_points):
                idx = (box_i * n_interpolation_points + interp_i) * \
                      (n_boxes_1d * n_interpolation_points) + \
                      (box_j * n_interpolation_points) + \
                      interp_j
                for d in range(n_terms):
                    w_coefficients[idx, d] += \
                        reference_x_interpolated_values[i, interp_i] * \
                        reference_y_interpolated_values[i, interp_j] * \
                        q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef floating[:, ::1] y_tilde_values = np.empty((total_interpolation_points, n_terms), dtype=_real_dtype(<floating>0))
    if dof != 1:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients[:, :3], y_tilde_values[:, :3], num_threads)
        matrix_multiply_fft_2d(kernel_tilde, w_coefficients[:, 3:], y_tilde_values[:, 3:], num_threads)
    else:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)

    return (
        np.asarray(y_tilde_values),
        np.asarray(box_x_lower_bounds),
        np.asarray(box_y_lower_bounds),
    )


cpdef double estimate_negative_gradient_fft_2d_with_grid(
    floating[:, ::1] embedding,
    floating[:, ::1] gradient,
    floating[:, ::1] y_tilde_values,
    floating[::1] box_x_lower_bounds,
    floating[::1] box_y_lower_bounds,
    Py_ssize_t n_interpolation_points,
    double dof,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_terms = y_tilde_values.shape[1]
        Py_ssize_t n_boxes_1d = int(sqrt(box_x_lower_bounds.shape[0]))
        double coord_min = box_x_lower_bounds[0]
        double box_width = box_x_lower_bounds[1] - box_x_lower_bounds[0]

    # Determine which box each point belongs to
    cdef int box_x_idx, box_y_idx
    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))
    for i in range(n_samples):
        box_x_idx = <int>((embedding[i, 0] - coord_min) / box_width)
        box_y_idx = <int>((embedding[i, 1] - coord_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_x_idx >= n_boxes_1d:
            box_x_idx = n_boxes_1d - 1
        if box_y_idx >= n_boxes_1d:
            box_y_idx = n_boxes_1d - 1

        point_box_idx[i] = box_y_idx * n_boxes_1d + box_x_idx

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    # Compute the relative position of each new embedding point in its box
    cdef:
        double[::1] x_in_box = np.empty(n_samples, dtype=float)
        double[::1] y_in_box = np.empty(n_samples, dtype=float)

    cdef double y_min, x_min
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        x_min = box_x_lower_bounds[box_idx]
        y_min = box_y_lower_bounds[box_idx]
        x_in_box[i] = (embedding[i, 0] - x_min) / box_width
        y_in_box[i] = (embedding[i, 1] - y_min) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] x_interpolated_values = interpolate(x_in_box, y_tilde)
    cdef double[:, ::1] y_interpolated_values = interpolate(y_in_box, y_tilde)

    # Actually compute \tilde{\phi(y_i)}
    cdef Py_ssize_t box_i, box_j, interp_i, interp_j, idx

    cdef floating[:, ::1] phi = np.zeros((n_samples, n_terms), dtype=_real_dtype(<floating>0))
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        box_i = box_idx % n_boxes_1d
        box_j = box_idx // n_boxes_1d
        for interp_i in range(n_interpolation_points):
            for interp_j in range(n_interpolation_points):
                idx = (box_i * n_interpolation_points + interp_i) * \
                      (n_boxes_1d * n_interpolation_points) + \
                      (box_j * n_interpolation_points) + \
                      interp_j
                for d in range(n_terms):
                    phi[i, d] += x_interpolated_values[i, interp_i] * \
                                 y_interpolated_values[i, interp_j] * \
                                 y_tilde_values[idx, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double[::1] sum_Qi = np.empty(n_samples, dtype=float)
    cdef double y1, y2
    if dof != 1:
        for i in range(n_samples):
            sum_Qi[i] = phi[i, 3]
    else:
        for i in range(n_samples):
            y1 = embedding[i, 0]
            y2 = embedding[i, 1]

            sum_Qi[i] = (1 + y1 ** 2 + y2 ** 2) * phi[i, 0] - \
                        2 * (y1 * phi[i, 1] + y2 * phi[i, 2]) + \
                        phi[i, 3]

    cdef sum_Q = 0
    for i in range(n_samples):
        sum_Q += sum_Qi[i]

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i, 0] -= (embedding[i, 0] * phi[i, 0] - phi[i, 1]) / (sum_Qi[i] + EPSILON)
        gradient[i, 1] -= (embedding[i, 1] * phi[i, 0] - phi[i, 2]) / (sum_Qi[i] + EPSILON)

    return sum_Q
