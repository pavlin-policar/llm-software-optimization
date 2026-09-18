# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()
import numpy as np
import scipy.fft
from cython cimport floating
from cython.parallel import prange, parallel


# The transforms dominate the interpolation-based gradient once the embedding
# has spread out -- at 45k points a converged embedding needs a 1134 x 1134
# grid, and the nine transforms per call are essentially the whole cost. scipy
# exposes pocketfft's own thread pool through `workers`, which parallelizes
# them without needing OpenMP.
#
# The buffers are large (tens of megabytes at that grid size) and the grid size
# changes only when the embedding's extent crosses a box boundary, so they are
# kept between calls. Only the last size is retained; holding on to more would
# waste more memory than the allocations cost. Everything here runs under the
# GIL, so a plain module-level cache is safe.
cdef dict _buffers = {}

# Handing all the terms to one transform call gives the backend `n_terms` times
# as much independent work to spread over its threads, which is what a single
# term of a grid this shape does not have enough of, and on one thread it
# still saves the per-call work of going round again. The cost is a column
# spectrum `n_terms` times larger, and the grid grows with the square of the
# embedding's extent, so the buffer is capped rather than left to follow it.
# At 44 808 points and a converged embedding the batched spectrum is 31 MB
# against 10 MB for one term; past the cap the terms go one at a time.
cdef Py_ssize_t MAX_BATCHED_SPECTRUM_BYTES = 48 * 1024 * 1024

# `scipy.fft` allocates its result, and at this grid size that costs about as
# much as the transform: the row passes move fifteen megabytes each, and
# writing them into a fresh array costs roughly twice what writing them over a
# buffer the previous call left behind does. The backend scipy is built on does
# take a destination, so where the caller has laid its arrays out to be written
# into directly -- padded to the transform length, so that the result lands
# where the caller already reads from -- the transform writes there.
#
# This reaches past scipy's public interface, so it is taken only if it is
# there and it works, and the public calls stand behind it unchanged. A backend
# installed with `scipy.fft.set_backend` is not consulted on this path.
cdef object _r2c = None
cdef object _c2r = None
cdef object _dct = None

cdef _find_pocketfft():
    global _r2c, _c2r, _dct
    try:
        from scipy.fft._pocketfft import pypocketfft
        probe = np.zeros((2, 4))
        out = np.empty((2, 3), dtype=np.complex128)
        pypocketfft.r2c(probe, axes=[1], forward=True, out=out, nthreads=1)
        back = np.empty((2, 4))
        pypocketfft.c2r(out, axes=[1], lastsize=4, forward=False, inorm=2,
                        out=back, nthreads=1)
        square = np.zeros((3, 3))
        pypocketfft.dct(square, 1, axes=[0, 1], out=np.empty((3, 3)), nthreads=1)
    except Exception:
        return
    _r2c = pypocketfft.r2c
    _c2r = pypocketfft.c2r
    _dct = pypocketfft.dct


_find_pocketfft()


cdef _complex_dtype(floating sample):
    """The complex type a transform of `floating` produces."""
    if floating is float:
        return np.complex64
    return np.complex128


cdef void _multiply_by_kernel(
    floating[:, :, ::1] spectrum,
    floating[:, ::1] kernel,
    Py_ssize_t num_threads,
) noexcept nogil:
    """Scale every term's column spectrum by the kernel's, in place.

    The elementwise product is the one part of the transform numpy is left
    holding, and there it moves tens of megabytes on one thread while the
    transforms around it run on all of them.

    The kernel's spectrum is real, so this is a scaling and not a complex
    product, and the terms of one grid point sit next to each other in memory
    and are all scaled by the same number. The spectrum arrives as complex
    values viewed as pairs of reals, which is what lets one loop serve both
    precisions and scale the real and imaginary parts alike.

    Only the rows up to the middle of the kernel's spectrum are stored, the
    rest being their mirror image, so the row index folds back at the middle.
    """
    cdef:
        Py_ssize_t n_rows = spectrum.shape[0]
        Py_ssize_t n_cols = spectrum.shape[1]
        Py_ssize_t n_values = spectrum.shape[2]
        Py_ssize_t half = n_rows // 2
        Py_ssize_t i, j, d, row
        floating k

    with parallel(num_threads=num_threads):
        for i in prange(n_rows, schedule="static"):
            row = i if i <= half else n_rows - i
            for j in range(n_cols):
                k = kernel[row, j]
                for d in range(n_values):
                    spectrum[i, j, d] = spectrum[i, j, d] * k


cdef _as_real_pairs(a):
    """A complex array viewed as a trailing axis of real/imaginary pairs."""
    return a.view(a.real.dtype).reshape(a.shape[0], a.shape[1], -1)


cdef _get_buffer(name, shape, dtype=float):
    """A zeroed buffer of the given shape, reused across calls.

    One buffer is kept per name, and a call that wants a different shape or
    precision replaces it. Holding on to more than the last one would waste
    more memory than the allocations cost.
    """
    cdef object buf = _buffers.get(name)
    if buf is None or buf.shape != shape or buf.dtype != np.dtype(dtype):
        buf = np.zeros(shape, dtype=dtype)
        _buffers[name] = buf
    return buf


cdef void matrix_multiply_fft_1d(
    floating[::1] kernel_tilde,
    floating[:, ::1] w_coefficients,
    floating[:, ::1] out,
    Py_ssize_t num_threads=1,
):
    """Multiply the the kernel vectr K tilde with the w coefficients.

    Parameters
    ----------
    kernel_tilde : memoryview
        The half of the generating vector of the 2d Toeplitz matrix that is
        not a mirror image of the other, i.e. the kernel evaluated at all
        interpolation points from the left most interpolation point, embedded
        in a circulant matrix (doubled in size from (n_interp, n_interp) to
        (2 * n_interp, 2 * n_interp)) and symmetrized. See how to embed
        Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    num_threads : Py_ssize_t
        Threads to hand to the FFT backend.

    """
    cdef:
        Py_ssize_t grid_1d = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = 2 * (kernel_tilde.shape[0] - 1)
        Py_ssize_t d

    if num_threads < 1:
        num_threads = 1

    w = np.asarray(w_coefficients)
    o = np.asarray(out)

    # The circulant's generating vector is even, so its spectrum is real and a
    # real-even transform reads it off half the vector
    fft_kernel_tilde = scipy.fft.dct(
        np.asarray(kernel_tilde), type=1, workers=num_threads
    )

    for d in range(n_terms):
        # The coefficients occupy the leading half of the circulant embedding;
        # asking for a longer transform zero-pads the rest, so no padded input
        # buffer is needed
        fft_w_coeffs = scipy.fft.rfft(
            w[:, d], n=n_fft_coeffs, workers=num_threads
        )
        fft_w_coeffs *= fft_kernel_tilde
        fft_out_buffer = scipy.fft.irfft(
            fft_w_coeffs, n=n_fft_coeffs, workers=num_threads
        )

        o[:, d] = fft_out_buffer[grid_1d:]


cdef void matrix_multiply_fft_2d(
    floating[:, ::1] kernel_tilde,
    floating[:, ::1] w_coefficients,
    floating[:, ::1] out,
    Py_ssize_t num_threads=1,
):
    """Multiply the the kernel matrix K tilde with the w coefficients.

    Parameters
    ----------
    kernel_tilde : memoryview
        The quadrant of the generating matrix of the 3d Toeplitz tensor that
        the other three are mirror images of, i.e. the kernel evaluated at all
        interpolation points from the top left most interpolation point,
        embedded in a circulant matrix (doubled in size from
        (n_interp, n_interp) to (2 * n_interp, 2 * n_interp)) and symmetrized.
        See how to embed Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    num_threads : Py_ssize_t
        Threads to hand to the FFT backend.

    """
    cdef:
        Py_ssize_t n_terms = w_coefficients.shape[1]
        # The kernel's quadrant carries one more node than the grid on each
        # axis, and the two axes need not be the same length
        Py_ssize_t grid_x = kernel_tilde.shape[0] - 1
        Py_ssize_t grid_y = kernel_tilde.shape[1] - 1
        Py_ssize_t n_fft_x = 2 * grid_x
        Py_ssize_t n_fft_coeffs = 2 * grid_y
        Py_ssize_t d
        floating[:, :, ::1] spectrum_pairs
        floating[:, ::1] kernel_real

    if num_threads < 1:
        num_threads = 1

    w = np.asarray(w_coefficients)
    o = np.asarray(out)

    # Either the coefficients occupy a grid of their own and the transform
    # pads them, or the caller has already laid them out along the transform's
    # rows with the padding in place, in which case the transform can be given
    # somewhere to write
    cdef Py_ssize_t row_len = n_fft_coeffs if w.shape[0] == \
        grid_x * n_fft_coeffs else grid_y

    # The circulant's generating matrix is even along both axes, so its
    # spectrum is real and a real-even transform reads it off one quadrant
    k = np.asarray(kernel_tilde)
    if _dct is not None:
        kernel_real = _dct(
            k, 1, axes=[0, 1], nthreads=num_threads,
            out=_get_buffer("2d-kernel", k.shape, k.dtype),
        )
    else:
        kernel_real = scipy.fft.dctn(k, type=1, workers=num_threads)

    # A two-dimensional transform is a pass along the rows followed by a pass
    # along the columns, and taking the two passes separately lets both of them
    # skip work that a single `rfft2`/`irfft2` cannot know about:
    #
    #   - the coefficients occupy only the top left quadrant of the circulant
    #     embedding, so the forward row pass has half its rows zero;
    #   - only the bottom right quadrant of the result is ever read, so the
    #     inverse row pass is needed for half the rows.
    #
    # Together that removes about a quarter of the transform work. The column
    # spectrum is kept between calls and the column pass runs in place on it,
    # which is worth a fifth of the whole transform at 45k points: the pass
    # writes tens of megabytes, and writing them into a freshly allocated array
    # costs twice what writing them over the previous call's does. Since the
    # pass leaves its own output there, the half of the buffer the row pass
    # does not write has to be zeroed every call rather than once.
    cdef Py_ssize_t half = n_fft_coeffs // 2 + 1
    cdef Py_ssize_t batched_bytes = (
        n_fft_x * half * n_terms
        * (8 if floating is float else 16)
    )
    # `w` and `o` are views over the term axis when dof != 1, and reshaping
    # those would copy away whatever the batching saves
    cdef bint batched = (
        batched_bytes <= MAX_BATCHED_SPECTRUM_BYTES
        and w.flags.c_contiguous and o.flags.c_contiguous
    )

    if batched:
        # The terms are the last, contiguous axis of `w_coefficients`, so all
        # of them transform together simply by keeping that axis
        spectrum = _get_buffer(
            "2d-batched",
            (n_fft_x, half, n_terms),
            dtype=_complex_dtype(<floating>0),
        )

        wr = w.reshape(grid_x, row_len, n_terms)
        if row_len == n_fft_coeffs and _r2c is not None:
            _r2c(wr, axes=[1], forward=True,
                 out=spectrum[:grid_x], nthreads=num_threads)
        else:
            spectrum[:grid_x] = scipy.fft.rfft(
                wr, n=n_fft_coeffs, axis=1, workers=num_threads,
            )
        spectrum[grid_x:] = 0

        fft_w_coefficients = scipy.fft.fft(
            spectrum, axis=0, overwrite_x=True, workers=num_threads
        )
        spectrum_pairs = _as_real_pairs(fft_w_coefficients)
        with nogil:
            _multiply_by_kernel(spectrum_pairs, kernel_real, num_threads)

        columns = scipy.fft.ifft(
            fft_w_coefficients, axis=0, overwrite_x=True, workers=num_threads
        )
        bottom = columns[grid_x:]

        if row_len == n_fft_coeffs:
            # The result the caller reads is the trailing half of each row, so
            # the whole row goes straight into the output grid
            outr = o.reshape(grid_x, row_len, n_terms)
            if _c2r is not None:
                _c2r(bottom, axes=[1], lastsize=n_fft_coeffs, forward=False,
                     inorm=2, out=outr, nthreads=num_threads)
            else:
                outr[...] = scipy.fft.irfft(
                    bottom, n=n_fft_coeffs, axis=1, workers=num_threads)
            return

        fft_out_buffer = scipy.fft.irfft(
            bottom, n=n_fft_coeffs, axis=1, workers=num_threads,
        )
        o.reshape(
            grid_x, grid_y, n_terms
        )[:] = fft_out_buffer[:, grid_y:, :]
        return

    spectrum = _get_buffer(
        "2d",
        (n_fft_x, half),
        dtype=_complex_dtype(<floating>0),
    )

    for d in range(n_terms):
        spectrum[:grid_x] = scipy.fft.rfft(
            w[:, d].reshape(grid_x, row_len),
            n=n_fft_coeffs,
            axis=1,
            workers=num_threads,
        )
        spectrum[grid_x:] = 0

        fft_w_coefficients = scipy.fft.fft(
            spectrum, axis=0, overwrite_x=True, workers=num_threads
        )
        spectrum_pairs = _as_real_pairs(fft_w_coefficients)
        with nogil:
            _multiply_by_kernel(spectrum_pairs, kernel_real, num_threads)

        columns = scipy.fft.ifft(
            fft_w_coefficients, axis=0, overwrite_x=True, workers=num_threads
        )
        fft_out_buffer = scipy.fft.irfft(
            columns[grid_x:],
            n=n_fft_coeffs,
            axis=1,
            workers=num_threads,
        )

        if row_len == n_fft_coeffs:
            o[:, d] = fft_out_buffer.ravel()
        else:
            o[:, d] = fft_out_buffer[:, grid_y:].ravel()
