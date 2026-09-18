# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Implements a quad/oct-tree space partitioning algorithm primarily used in
efficiently estimating the t-SNE negative gradient. Lowers the time complexity
from the naive O(n^2) to O(n * log(n)).

Notes
-----
I list here several implementation details. Many of these improve efficiency.

  - The tree is not a linked structure of individually allocated nodes. Nodes
    live in one flat pool and refer to their children by index, and the centers
    of mass live in a second pool indexed the same way. Traversal is entirely
    memory bound, so the layout is what determines its speed: one node is 32
    bytes, half a cache line, and a node's children are always one contiguous
    run.

  - Node centers and side lengths are not stored. Both are exactly recoverable
    while descending from the root -- halving a side length and offsetting a
    center are exact in binary floating point -- so keeping them out of the
    node shrinks the structure that the traversal has to stream through.

  - In one, two and three dimensions the tree is built by sorting the points
    along a Morton curve. A cell is then a contiguous run of the sorted array,
    the tree is a recursive partition of it, and one pass writes the nodes out
    already laid out for traversal. Inserting points one at a time instead
    costs a descent through the tree per point and leaves the pool in insertion
    order, which says nothing about where a node sits in space, so a second
    pass has to re-lay it out.

  - The Morton key has to separate any two points the tree would otherwise
    split, so the number of bits it can spare per dimension is what decides
    whether it is usable. Above three dimensions, and whenever the embedding is
    wide enough that a 64-bit key would resolve less finely than
    `MERGE_TOLERANCE`, the insertion build is used instead.

  - Structs do not support memoryviews, therefore pointers must be used.

  - Prefer pointers over memoryviews where speed is essential. Memoryview
    indexing and slicing is slow compared to raw memory access. We can easily
    convert a memory view to a pointer like so: `&mv[0]` however care must be
    taken to ensure the memoryview is a C contigous array. This can be ensured
    by the type declaration `double[:, ::1]` for 2d arrays.

References
----------
.. [1] Van Der Maaten, Laurens. "Accelerating t-SNE using tree-based
   algorithms." Journal of machine learning research 15.1 (2014): 3221-3245.

"""
cimport numpy as cnp
cnp.import_array()
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdint cimport INT32_MAX, uint64_t
from libc.string cimport memcpy, memset


cdef extern from "math.h":
    double fabs(double x) nogil


# A point falling within this distance of a node's center of mass is merged
# into that node instead of being sunk further down, which keeps the tree
# shallow when a data set contains duplicate points
cdef double MERGE_TOLERANCE = 1e-6

# Bits of a 64-bit Morton key available to each dimension. Two are left spare
# so the key stays in the non-negative range of a signed integer
cdef Py_ssize_t MORTON_BITS[4]
MORTON_BITS[:] = [0, 62, 31, 20]

# Digit width of the radix sort over the keys, chosen so 64 bits take four
# passes and the counters stay inside the last level of cache
cdef Py_ssize_t RADIX_BITS = 16


cdef inline uint64_t _interleave_2d(uint64_t v) noexcept nogil:
    """Insert a zero bit between each of the low 32 bits of `v`."""
    v &= <uint64_t>0x00000000FFFFFFFF
    v = (v | (v << 16)) & <uint64_t>0x0000FFFF0000FFFF
    v = (v | (v << 8)) & <uint64_t>0x00FF00FF00FF00FF
    v = (v | (v << 4)) & <uint64_t>0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & <uint64_t>0x3333333333333333
    v = (v | (v << 1)) & <uint64_t>0x5555555555555555
    return v


cdef void _radix_sort(
    uint64_t * keys, Py_ssize_t * idx,
    uint64_t * key_buf, Py_ssize_t * idx_buf,
    Py_ssize_t n, Py_ssize_t total_bits, Py_ssize_t * counts,
) noexcept nogil:
    """Sort `keys`, carrying `idx` along, least significant digit first.

    A pass whose digit is the same for every key would only copy the arrays, so
    it is skipped; on clustered data that is most of them.
    """
    cdef:
        Py_ssize_t n_buckets = 1 << RADIX_BITS
        uint64_t mask = (<uint64_t>1 << RADIX_BITS) - 1
        Py_ssize_t shift, i, b, total, count, pos
        uint64_t digit

    shift = 0
    while shift < total_bits:
        memset(counts, 0, n_buckets * sizeof(Py_ssize_t))
        for i in range(n):
            counts[<Py_ssize_t>((keys[i] >> shift) & mask)] += 1

        total = 0
        for b in range(n_buckets):
            count = counts[b]
            counts[b] = total
            total += count
            if count == n:
                break

        if count != n:
            for i in range(n):
                digit = (keys[i] >> shift) & mask
                pos = counts[<Py_ssize_t>digit]
                counts[<Py_ssize_t>digit] = pos + 1
                key_buf[pos] = keys[i]
                idx_buf[pos] = idx[i]

            memcpy(keys, key_buf, n * sizeof(uint64_t))
            memcpy(idx, idx_buf, n * sizeof(Py_ssize_t))

        shift += RADIX_BITS


cdef inline Py_ssize_t get_child_idx_for(
    double * point, double * center, Py_ssize_t n_dims
) noexcept nogil:
    cdef Py_ssize_t idx = 0, d

    for d in range(n_dims):
        idx |= (point[d] > center[d]) << d

    return idx


cdef inline void update_center_of_mass(
    Node * node, double * com, double * point, Py_ssize_t n_dims
) noexcept nogil:
    cdef Py_ssize_t d
    for d in range(n_dims):
        com[d] = (com[d] * node.num_points + point[d]) / (node.num_points + 1)
    node.num_points += 1


cdef class QuadTree:
    def __cinit__(self):
        self.build_nodes = NULL
        self.build_center_of_mass = NULL
        self.build_num_nodes = 0
        self.build_capacity = 0
        self.points = NULL
        self.n_points = 0
        self.point_capacity = 0
        self.point_order = NULL
        self.use_morton = False
        self.morton_bits = 0
        self.nodes = NULL
        self.center_of_mass = NULL
        self.trav2 = NULL
        self.trav2_capacity = 0
        self.num_nodes = 0
        self.capacity = 0
        self.root_center = NULL
        self._descent_center = NULL
        self.max_depth = 0
        self.max_children = 0
        self.is_compacted = False

    def __init__(self, double[:, ::1] data):
        cdef:
            Py_ssize_t n_dim = data.shape[1]
            Py_ssize_t n_points = data.shape[0]
            double[:] x_min = np.min(data, axis=0)
            double[:] x_max = np.max(data, axis=0)

            double[:] center = np.zeros(n_dim)
            double length = 0
            Py_ssize_t d, bits

        for d in range(n_dim):
            center[d] = (x_max[d] + x_min[d]) / 2
            if x_max[d] - x_min[d] > length:
                length = x_max[d] - x_min[d]

        self.n_dims = n_dim
        self.root_length = length

        self.root_center = <double *>PyMem_Malloc(n_dim * sizeof(double))
        self._descent_center = <double *>PyMem_Malloc(n_dim * sizeof(double))
        if not self.root_center or not self._descent_center:
            raise MemoryError()
        for d in range(n_dim):
            self.root_center[d] = center[d]

        # The Morton build is only equivalent to sinking points one at a time
        # while the key resolves at least as finely as the tolerance at which
        # the insertion build gives up and merges
        if 1 <= n_dim <= 3:
            bits = MORTON_BITS[n_dim]
            self.use_morton = length / <double>(<uint64_t>1 << bits) <= MERGE_TOLERANCE
            self.morton_bits = bits

        if not self.use_morton:
            # Every split adds `2 ** n_dims` nodes and there are at most as many
            # splits as there are points, but in practice the tree is far
            # smaller, so start with a modest guess and grow geometrically
            self._reserve(max(64, 4 * n_points))
            self._new_node(length * length)

        self.add_points(data)

    cdef int _reserve(self, Py_ssize_t capacity) except -1:
        if capacity <= self.build_capacity:
            return 0

        self.build_nodes = <Node *>PyMem_Realloc(
            self.build_nodes, capacity * sizeof(Node)
        )
        self.build_center_of_mass = <double *>PyMem_Realloc(
            self.build_center_of_mass, capacity * self.n_dims * sizeof(double)
        )
        if not self.build_nodes or not self.build_center_of_mass:
            raise MemoryError()
        self.build_capacity = capacity
        return 0

    cdef Py_ssize_t _new_node(self, double length_sq) except -1:
        """Append a single empty node to the pool and return its index."""
        cdef Py_ssize_t idx = self.build_num_nodes, d

        if idx >= self.build_capacity:
            self._reserve(2 * self.build_capacity)

        self.build_nodes[idx].num_points = 0
        self.build_nodes[idx].first_child = -1
        self.build_nodes[idx].num_children = 0
        self.build_nodes[idx].length_sq = length_sq

        for d in range(self.n_dims):
            self.build_center_of_mass[idx * self.n_dims + d] = 0

        self.build_num_nodes = idx + 1
        return idx

    cdef Py_ssize_t _split(self, Py_ssize_t node, double child_length) except -1:
        """Give `node` a full complement of empty children."""
        cdef:
            Py_ssize_t num_children = 1 << self.n_dims
            Py_ssize_t first = self.build_num_nodes
            Py_ssize_t i

        if first + num_children > self.build_capacity:
            self._reserve(max(2 * self.build_capacity, first + num_children))

        for i in range(num_children):
            self._new_node(child_length * child_length)

        self.build_nodes[node].first_child = first
        self.build_nodes[node].num_children = num_children
        return first

    cdef int _reserve_nodes(self, Py_ssize_t capacity) except -1:
        if capacity <= self.capacity:
            return 0

        self.nodes = <Node *>PyMem_Realloc(self.nodes, capacity * sizeof(Node))
        self.center_of_mass = <double *>PyMem_Realloc(
            self.center_of_mass, capacity * self.n_dims * sizeof(double)
        )
        if not self.nodes or not self.center_of_mass:
            raise MemoryError()
        self.capacity = capacity
        return 0

    cdef int _store_point(self, double * point) except -1:
        cdef Py_ssize_t capacity, d

        if self.n_points >= self.point_capacity:
            capacity = max(64, 2 * self.point_capacity)
            self.points = <double *>PyMem_Realloc(
                self.points, capacity * self.n_dims * sizeof(double)
            )
            if not self.points:
                raise MemoryError()
            self.point_capacity = capacity

        for d in range(self.n_dims):
            self.points[self.n_points * self.n_dims + d] = point[d]
        self.n_points += 1
        return 0

    cpdef void add_points(self, double[:, ::1] points) except *:
        cdef Py_ssize_t i

        if self.use_morton:
            # Growing geometrically from nothing would copy the points several
            # times over, and the common case hands them all over at once
            if self.n_points + points.shape[0] > self.point_capacity:
                self.points = <double *>PyMem_Realloc(
                    self.points,
                    (self.n_points + points.shape[0]) * self.n_dims * sizeof(double),
                )
                if not self.points:
                    raise MemoryError()
                self.point_capacity = self.n_points + points.shape[0]

        for i in range(points.shape[0]):
            self._add_point(&points[i, 0])

    cpdef void add_point(self, double[::1] point) except *:
        self._add_point(&point[0])

    cdef int _add_point(self, double * point) except -1:
        cdef:
            Py_ssize_t n_dims = self.n_dims
            Py_ssize_t node = 0, child, d
            double length = self.root_length, child_length
            double * center = self._descent_center
            double * com
            Node * nd

        # The tree is only valid for traversal once compacted, and adding a
        # point invalidates that
        self.is_compacted = False

        if self.use_morton:
            self._store_point(point)
            return 0

        for d in range(n_dims):
            center[d] = self.root_center[d]

        while True:
            nd = &self.build_nodes[node]
            com = &self.build_center_of_mass[node * n_dims]

            # An empty leaf takes the point directly. So does any node whose
            # center of mass the point practically coincides with, which is
            # what stops duplicate points from growing an unbounded tree
            if (nd.first_child < 0 and nd.num_points == 0) or \
                    is_close(com, point, MERGE_TOLERANCE, n_dims):
                update_center_of_mass(nd, com, point, n_dims)
                return 0

            child_length = length / 2

            if nd.first_child < 0:
                # Split, and sink the point already living here into the
                # appropriate child
                self._split(node, child_length)
                # `_split` may have moved the pools
                nd = &self.build_nodes[node]
                com = &self.build_center_of_mass[node * n_dims]

                child = nd.first_child + get_child_idx_for(com, center, n_dims)
                update_center_of_mass(
                    &self.build_nodes[child],
                    &self.build_center_of_mass[child * n_dims],
                    com,
                    n_dims,
                )

            update_center_of_mass(nd, com, point, n_dims)

            # Descend, deriving the child's center rather than storing it
            child = get_child_idx_for(point, center, n_dims)
            for d in range(n_dims):
                if child & (1 << d):
                    center[d] = center[d] + child_length / 2
                else:
                    center[d] = center[d] - child_length / 2

            node = nd.first_child + child
            length = child_length

    cdef int compact(self) except -1:
        """Make the tree ready for traversal."""
        if self.is_compacted:
            return 0
        if self.use_morton:
            return self._build_morton()
        self._compact_build_pool()
        return self._build_trav2()

    cdef int _reserve_trav2(self, Py_ssize_t capacity) except -1:
        cdef TravNode2 * trav
        if capacity <= self.trav2_capacity:
            return 0
        trav = <TravNode2 *>PyMem_Realloc(
            self.trav2, capacity * sizeof(TravNode2))
        if not trav:
            raise MemoryError()
        self.trav2 = trav
        self.trav2_capacity = capacity
        return 0

    cdef int _build_trav2(self) except -1:
        """Gather what the two-dimensional traversal reads into one array.

        The nodes are re-laid-out so that each is immediately followed by its
        own subtree, which is what lets the traversal find a node's first child
        by stepping forward and skip its subtree by jumping to a single stored
        index. The build pools cannot be laid out that way: they keep a node's
        children in one contiguous run so that a node can name them all with a
        first index and a count.

        A subtree's size is the same whichever order the pool is in, so it is
        counted once over the existing pool -- parents come before their
        children there, so counting backwards suffices -- and the descent then
        knows every escape index as it writes the nodes out.

        The indices and the point count are narrowed to 32 bits. A tree too
        large for that is traversed with a stack over the pools it was built
        in.
        """
        cdef:
            Py_ssize_t count = self.num_nodes
            Py_ssize_t node, old, new, first, c, top, size
            Py_ssize_t stack_capacity
            Py_ssize_t * sizes
            Py_ssize_t * stack
            TravNode2 * trav

        if self.n_dims != 2 or count == 0 \
                or count > INT32_MAX or self.nodes[0].num_points > INT32_MAX:
            return 0

        self._reserve_trav2(count)

        stack_capacity = (self.max_depth + 2) * (self.max_children + 1) + 2
        sizes = <Py_ssize_t *>PyMem_Malloc(count * sizeof(Py_ssize_t))
        stack = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        if not sizes or not stack:
            PyMem_Free(sizes)
            PyMem_Free(stack)
            raise MemoryError()

        for node in range(count - 1, -1, -1):
            size = 1
            first = self.nodes[node].first_child
            for c in range(self.nodes[node].num_children):
                size += sizes[first + c]
            sizes[node] = size

        trav = self.trav2
        stack[0] = 0
        top = 1
        new = 0

        while top > 0:
            top -= 1
            old = stack[top]

            trav[new].com_x = self.center_of_mass[2 * old]
            trav[new].com_y = self.center_of_mass[2 * old + 1]
            trav[new].length_sq = self.nodes[old].length_sq
            trav[new].num_points = <int>self.nodes[old].num_points
            trav[new].escape = <int>(new + sizes[old])
            new += 1

            # Pushed in reverse so the first child is popped first, and so
            # lands immediately after this node
            first = self.nodes[old].first_child
            for c in range(self.nodes[old].num_children - 1, -1, -1):
                stack[top] = first + c
                top += 1

        PyMem_Free(sizes)
        PyMem_Free(stack)
        return 0

    cdef int _compact_build_pool(self) except -1:
        """Drop empty children and re-lay-out the pool for traversal.

        Nodes come out in the order a depth-first traversal reaches them, with
        each node immediately followed by its children, so that both the
        sibling scan and the descent into a subtree read forwards through
        memory.
        """
        cdef:
            Py_ssize_t n_dims = self.n_dims
            Node * new_nodes
            double * new_com
            Py_ssize_t * stack_old
            Py_ssize_t * stack_new
            Py_ssize_t * stack_depth
            Py_ssize_t top = 0, count = 1, max_depth = 0, max_children = 0
            Py_ssize_t old, new, depth, i, c, child, kept, first_kept
            Py_ssize_t stack_capacity

        if self.build_num_nodes == 0:
            self.is_compacted = True
            return 0

        # A depth-first traversal holds at most one sibling group per level, so
        # the stacks stay small next to the pools; sizing them by the node count
        # instead would add half again as much as the compacted tree costs
        stack_capacity = 64 * (1 << self.n_dims) + 64

        new_nodes = <Node *>PyMem_Malloc(self.build_num_nodes * sizeof(Node))
        new_com = <double *>PyMem_Malloc(self.build_num_nodes * n_dims * sizeof(double))
        stack_old = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        stack_new = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        stack_depth = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        if not new_nodes or not new_com or not stack_old or not stack_new \
                or not stack_depth:
            raise MemoryError()

        stack_old[0] = 0
        stack_new[0] = 0
        stack_depth[0] = 0
        top = 1

        while top > 0:
            top -= 1
            old = stack_old[top]
            new = stack_new[top]
            depth = stack_depth[top]
            if depth > max_depth:
                max_depth = depth

            new_nodes[new] = self.build_nodes[old]
            memcpy(
                &new_com[new * n_dims],
                &self.build_center_of_mass[old * n_dims],
                n_dims * sizeof(double),
            )
            new_nodes[new].first_child = -1
            new_nodes[new].num_children = 0

            if self.build_nodes[old].first_child < 0:
                continue

            kept = 0
            for i in range(self.build_nodes[old].num_children):
                if self.build_nodes[self.build_nodes[old].first_child + i].num_points > 0:
                    kept += 1
            if kept == 0:
                continue

            first_kept = count
            new_nodes[new].first_child = first_kept
            new_nodes[new].num_children = kept
            count += kept
            if kept > max_children:
                max_children = kept

            # Push in reverse so the first child is popped first, which lays
            # its subtree out immediately after the sibling block
            if top + kept > stack_capacity:
                stack_capacity = 2 * (top + kept)
                stack_old = <Py_ssize_t *>PyMem_Realloc(
                    stack_old, stack_capacity * sizeof(Py_ssize_t))
                stack_new = <Py_ssize_t *>PyMem_Realloc(
                    stack_new, stack_capacity * sizeof(Py_ssize_t))
                stack_depth = <Py_ssize_t *>PyMem_Realloc(
                    stack_depth, stack_capacity * sizeof(Py_ssize_t))
                if not stack_old or not stack_new or not stack_depth:
                    raise MemoryError()

            c = kept
            for i in range(self.build_nodes[old].num_children - 1, -1, -1):
                child = self.build_nodes[old].first_child + i
                if self.build_nodes[child].num_points == 0:
                    continue
                c -= 1
                stack_old[top] = child
                stack_new[top] = first_kept + c
                stack_depth[top] = depth + 1
                top += 1

        PyMem_Free(stack_old)
        PyMem_Free(stack_new)
        PyMem_Free(stack_depth)
        PyMem_Free(self.nodes)
        PyMem_Free(self.center_of_mass)

        self.nodes = new_nodes
        self.center_of_mass = new_com
        self.capacity = self.build_num_nodes
        self.num_nodes = count
        self.max_depth = max_depth
        self.max_children = max_children
        self.is_compacted = True
        return 0

    cdef int _build_morton(self) except -1:
        """Build the traversal pools from the points sorted along a Morton curve.

        Interleaving the bits of the quantized coordinates gives every point a
        key whose leading `n_dims * k` bits name the cell it occupies at depth
        `k`. Sorting by that key therefore lays the points out so that every
        cell of the tree, at every depth, is one contiguous run, and the tree
        is exactly the recursive partition of the sorted array by successive
        groups of `n_dims` bits.

        Nodes are written in the order a depth-first traversal reaches them,
        which is the layout `_compact_build_pool` has to make a second pass to
        produce.
        """
        cdef:
            Py_ssize_t n_dims = self.n_dims
            Py_ssize_t n = self.n_points
            Py_ssize_t bits = self.morton_bits
            Py_ssize_t total_bits = bits * n_dims
            Py_ssize_t n_children = 1 << n_dims
            Py_ssize_t n_buckets = 1 << RADIX_BITS

            uint64_t * keys
            uint64_t * key_buf
            Py_ssize_t * order
            Py_ssize_t * order_buf
            Py_ssize_t * counts
            Py_ssize_t * parent
            Py_ssize_t * stack_lo
            Py_ssize_t * stack_hi
            Py_ssize_t * stack_depth
            double * lengths
            double * point

            Py_ssize_t * stack_parent
            TravNode2 * trav

            Py_ssize_t max_levels, stack_capacity
            Py_ssize_t i, d, b, k, lo, hi, depth, node, top, count
            Py_ssize_t child, child_lo, first_kept, kept, cursor, capacity
            Py_ssize_t max_depth = 0, max_children = 0
            bint direct
            Py_ssize_t bounds[9]
            double scale, lo_corner, quantum, scaled
            double * com
            uint64_t q, coded, shift

        self.num_nodes = 0
        self.max_depth = 0
        self.max_children = 0
        if n == 0:
            self.is_compacted = True
            return 0

        # Splitting stops once a cell can no longer hold two points that the
        # insertion build would have separated
        max_levels = 0
        quantum = self.root_length
        while max_levels < bits and quantum > MERGE_TOLERANCE:
            quantum /= 2
            max_levels += 1

        # A depth-first traversal holds at most one sibling group per level
        stack_capacity = (max_levels + 2) * n_children + n_children

        keys = <uint64_t *>PyMem_Malloc(n * sizeof(uint64_t))
        key_buf = <uint64_t *>PyMem_Malloc(n * sizeof(uint64_t))
        order = <Py_ssize_t *>PyMem_Malloc(n * sizeof(Py_ssize_t))
        order_buf = <Py_ssize_t *>PyMem_Malloc(n * sizeof(Py_ssize_t))
        counts = <Py_ssize_t *>PyMem_Malloc(n_buckets * sizeof(Py_ssize_t))
        lengths = <double *>PyMem_Malloc((max_levels + 1) * sizeof(double))
        stack_lo = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        stack_hi = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        stack_depth = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        stack_node = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        stack_parent = <Py_ssize_t *>PyMem_Malloc(stack_capacity * sizeof(Py_ssize_t))
        if not keys or not key_buf or not order or not order_buf or not counts \
                or not lengths or not stack_lo or not stack_hi \
                or not stack_depth or not stack_node or not stack_parent:
            raise MemoryError()

        lengths[0] = self.root_length
        for k in range(max_levels):
            lengths[k + 1] = lengths[k] / 2

        # --- Morton keys --------------------------------------------------
        scale = 0
        if self.root_length > 0:
            scale = <double>(<uint64_t>1 << bits) / self.root_length

        with nogil:
            for i in range(n):
                coded = 0
                for d in range(n_dims):
                    lo_corner = self.root_center[d] - self.root_length / 2
                    scaled = (self.points[i * n_dims + d] - lo_corner) * scale
                    if scaled <= 0:
                        q = 0
                    elif scaled >= <double>((<uint64_t>1 << bits) - 1):
                        q = (<uint64_t>1 << bits) - 1
                    else:
                        q = <uint64_t>scaled

                    if n_dims == 2:
                        coded |= _interleave_2d(q) << d
                    elif n_dims == 1:
                        coded = q
                    else:
                        for b in range(bits):
                            coded |= ((q >> b) & <uint64_t>1) << (b * n_dims + d)
                keys[i] = coded
                order[i] = i

            _radix_sort(keys, order, key_buf, order_buf, n, total_bits, counts)

        # --- partition the sorted array into nodes ------------------------
        # In two dimensions a node is given its index when the descent reaches
        # it rather than when its parent names it, which puts each node
        # immediately before its own subtree: the layout the traversal wants.
        # The traversal pool is then written here and the build pools are left
        # alone. Otherwise the descent fills the build pools, which name a
        # node's children by a first index and a count and so need the
        # children contiguous.
        #
        # A cell can be split at most once per level, so the tree holds fewer
        # than `bits * n` nodes and the traversal pool's 32-bit indices reach
        # every one of them.
        direct = self.n_dims == 2 and n < INT32_MAX / (bits + 1)
        if direct:
            self._reserve_trav2(max(64, 2 * n))
            capacity = self.trav2_capacity
        else:
            self._reserve_nodes(max(64, 2 * n))
            capacity = self.capacity
        parent = <Py_ssize_t *>PyMem_Malloc(capacity * sizeof(Py_ssize_t))
        if not parent:
            raise MemoryError()
        trav = self.trav2

        stack_lo[0] = 0
        stack_hi[0] = n
        stack_depth[0] = 0
        stack_node[0] = 0
        stack_parent[0] = -1
        parent[0] = -1
        top = 1
        count = 0 if direct else 1

        while top > 0:
            top -= 1
            lo = stack_lo[top]
            hi = stack_hi[top]
            depth = stack_depth[top]
            if depth > max_depth:
                max_depth = depth

            if direct:
                if count >= capacity:
                    self._reserve_trav2(2 * capacity)
                    capacity = self.trav2_capacity
                    trav = self.trav2
                    parent = <Py_ssize_t *>PyMem_Realloc(
                        parent, capacity * sizeof(Py_ssize_t))
                    if not parent:
                        raise MemoryError()
                node = count
                count += 1
                parent[node] = stack_parent[top]

                trav[node].num_points = <int>(hi - lo)
                trav[node].length_sq = lengths[depth] * lengths[depth]
                # Raised as the descent finds the last node of the subtree
                trav[node].escape = <int>(node + 1)
                trav[node].com_x = 0
                trav[node].com_y = 0
            else:
                node = stack_node[top]
                self.nodes[node].num_points = hi - lo
                self.nodes[node].length_sq = lengths[depth] * lengths[depth]
                self.nodes[node].first_child = -1
                self.nodes[node].num_children = 0
                com = &self.center_of_mass[node * n_dims]
                for d in range(n_dims):
                    com[d] = 0

            # A run of equal keys can never be separated, however deep the
            # tree goes, so it is already a leaf
            if hi - lo == 1 or depth >= max_levels or keys[lo] == keys[hi - 1]:
                if direct:
                    for i in range(lo, hi):
                        point = &self.points[order[i] * 2]
                        trav[node].com_x += point[0]
                        trav[node].com_y += point[1]
                else:
                    for i in range(lo, hi):
                        point = &self.points[order[i] * n_dims]
                        for d in range(n_dims):
                            com[d] += point[d]
                continue

            # The children of this cell are the runs of the sorted range that
            # agree on the next `n_dims` bits of the key
            shift = <uint64_t>(total_bits - (depth + 1) * n_dims)
            cursor = lo
            kept = 0
            for child in range(n_children):
                child_lo = cursor
                while cursor < hi and \
                        <Py_ssize_t>((keys[cursor] >> shift) & (n_children - 1)) == child:
                    cursor += 1
                if cursor > child_lo:
                    bounds[kept] = child_lo
                    kept += 1
            bounds[kept] = hi

            if kept > max_children:
                max_children = kept

            if direct:
                # Pushed in reverse so the first child is popped first, and so
                # lands immediately after this node
                for k in range(kept - 1, -1, -1):
                    stack_lo[top] = bounds[k]
                    stack_hi[top] = bounds[k + 1]
                    stack_depth[top] = depth + 1
                    stack_parent[top] = node
                    top += 1
                continue

            if count + kept > capacity:
                self._reserve_nodes(max(2 * capacity, count + kept))
                capacity = self.capacity
                parent = <Py_ssize_t *>PyMem_Realloc(
                    parent, capacity * sizeof(Py_ssize_t))
                if not parent:
                    raise MemoryError()

            first_kept = count
            self.nodes[node].first_child = first_kept
            self.nodes[node].num_children = kept
            count += kept

            # Pushed in reverse so the first child is popped first, which lays
            # its subtree out immediately after the sibling block
            for k in range(kept - 1, -1, -1):
                stack_lo[top] = bounds[k]
                stack_hi[top] = bounds[k + 1]
                stack_depth[top] = depth + 1
                stack_node[top] = first_kept + k
                parent[first_kept + k] = node
                top += 1

        # --- centers of mass and escape indices ----------------------------
        # A node is always written before every node below it, so accumulating
        # from the end backwards gives each one the sum over its whole subtree,
        # and carries the last index in that subtree up to be its escape
        if direct:
            for i in range(count - 1, 0, -1):
                node = parent[i]
                trav[node].com_x += trav[i].com_x
                trav[node].com_y += trav[i].com_y
                if trav[i].escape > trav[node].escape:
                    trav[node].escape = trav[i].escape

            for i in range(count):
                trav[i].com_x /= trav[i].num_points
                trav[i].com_y /= trav[i].num_points
        else:
            for i in range(count - 1, 0, -1):
                com = &self.center_of_mass[parent[i] * n_dims]
                for d in range(n_dims):
                    com[d] += self.center_of_mass[i * n_dims + d]

            for i in range(count):
                com = &self.center_of_mass[i * n_dims]
                for d in range(n_dims):
                    com[d] /= self.nodes[i].num_points

        # A traversal wants its points in this order as much as the build did
        PyMem_Free(self.point_order)
        self.point_order = order

        PyMem_Free(keys)
        PyMem_Free(key_buf)
        PyMem_Free(order_buf)
        PyMem_Free(counts)
        PyMem_Free(parent)
        PyMem_Free(lengths)
        PyMem_Free(stack_lo)
        PyMem_Free(stack_hi)
        PyMem_Free(stack_depth)
        PyMem_Free(stack_node)
        PyMem_Free(stack_parent)

        self.num_nodes = count
        self.max_depth = max_depth
        self.max_children = max_children
        self.is_compacted = True
        if direct:
            return 0
        return self._build_trav2()

    def __dealloc__(self):
        PyMem_Free(self.build_nodes)
        PyMem_Free(self.build_center_of_mass)
        PyMem_Free(self.points)
        PyMem_Free(self.point_order)
        PyMem_Free(self.nodes)
        PyMem_Free(self.center_of_mass)
        PyMem_Free(self.trav2)
        PyMem_Free(self.root_center)
        PyMem_Free(self._descent_center)
