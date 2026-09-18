# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3


from cython cimport floating


cdef void matrix_multiply_fft_1d(
    floating[::1] kernel_tilde,
    floating[:, ::1] w_coefficients,
    floating[:, ::1] out,
    Py_ssize_t num_threads=*,
)

cdef void matrix_multiply_fft_2d(
    floating[:, ::1] kernel_tilde,
    floating[:, ::1] w_coefficients,
    floating[:, ::1] out,
    Py_ssize_t num_threads=*,
)
