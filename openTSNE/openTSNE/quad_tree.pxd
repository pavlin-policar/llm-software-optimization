# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()


# What the two-dimensional traversal reads, and nothing else. A node's center
# of mass sits alongside the rest of it, so reading a node is one fetch rather
# than two, and the counts are narrowed to fit the whole record into half a
# cache line.
#
# Nodes are in the order a depth-first descent reaches them, each immediately
# followed by its whole subtree, so descending is a step to the next node and
# `escape` -- the index one past the subtree -- is where the descent goes when
# it summarizes instead. A leaf is then exactly a node whose escape is the next
# index, which is what leaves the record no field to spare.
ctypedef struct TravNode2:
    double com_x
    double com_y
    double length_sq
    int num_points
    int escape


ctypedef struct Node:
    Py_ssize_t num_points
    # Index of the first child in the node pool, or -1 for a leaf. Siblings are
    # stored consecutively, so a node's children are one contiguous run.
    Py_ssize_t first_child
    Py_ssize_t num_children
    # The squared side length, which is what the Barnes-Hut criterion needs
    double length_sq


cdef inline bint is_close(double * center_of_mass, double * point, double eps,
                          Py_ssize_t n_dims) noexcept nogil:
    cdef Py_ssize_t d
    for d in range(n_dims):
        if (center_of_mass[d] - point[d] if center_of_mass[d] > point[d]
                else point[d] - center_of_mass[d]) >= eps:
            return False
    return True


cdef class QuadTree:
    # There are two ways to get from a set of points to the traversal pools.
    #
    # Sorting the points along a Morton curve puts every cell's points in one
    # contiguous run, so the tree is a partition of the sorted array and can be
    # written out in traversal order in a single pass. This needs a key fine
    # enough to separate any two points the tree would otherwise split, which a
    # 64-bit key only affords in few enough dimensions.
    #
    # Otherwise points are inserted one at a time into the build pools, where a
    # split always creates a full complement of `2 ** n_dims` children so that
    # a child's quadrant is its offset, and `_compact_build_pool` derives the
    # traversal pools: empty children dropped, and nodes laid out so that a
    # subtree is a contiguous run.
    #
    # Either way traversal reads the same pools, and that layout is what
    # determines its speed, traversal being entirely memory bound.
    cdef Node * build_nodes
    cdef double * build_center_of_mass
    cdef Py_ssize_t build_num_nodes
    cdef Py_ssize_t build_capacity

    cdef double * points
    cdef Py_ssize_t n_points
    cdef Py_ssize_t point_capacity
    # The points in the order the Morton build sorted them, kept because a
    # traversal wants to visit them in that order too. NULL when the tree was
    # not built that way.
    cdef Py_ssize_t * point_order
    cdef readonly bint use_morton
    cdef Py_ssize_t morton_bits

    cdef Node * nodes
    cdef double * center_of_mass
    # Built only where it is used: two dimensions, and counts that fit in it
    cdef TravNode2 * trav2
    cdef Py_ssize_t trav2_capacity
    cdef Py_ssize_t n_dims
    cdef readonly Py_ssize_t num_nodes
    cdef Py_ssize_t capacity
    cdef readonly Py_ssize_t max_depth
    cdef readonly Py_ssize_t max_children
    cdef double * root_center
    cdef double * _descent_center
    cdef double root_length
    cdef bint is_compacted

    cpdef void add_points(self, double[:, ::1] points) except *
    cpdef void add_point(self, double[::1] point) except *
    cdef int compact(self) except -1
    cdef int _reserve_trav2(self, Py_ssize_t capacity) except -1
    cdef int _build_trav2(self) except -1
    cdef int _compact_build_pool(self) except -1
    cdef int _build_morton(self) except -1
    cdef int _reserve(self, Py_ssize_t capacity) except -1
    cdef int _reserve_nodes(self, Py_ssize_t capacity) except -1
    cdef int _store_point(self, double * point) except -1
    cdef Py_ssize_t _new_node(self, double length_sq) except -1
    cdef Py_ssize_t _split(self, Py_ssize_t node, double child_length) except -1
    cdef int _add_point(self, double * point) except -1
