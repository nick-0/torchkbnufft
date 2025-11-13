# torchkbnufft/_nufft/utils.py
# SciPy-free utilities for torchkbnufft
# - Replaces scipy.special.iv/jv with closed-form / NumPy equivalents for the cases used here
# - Replaces scipy.sparse.coo_matrix usage with a tiny local COO compat class
# NOTE: This module currently supports Kaiser–Bessel with order == 0 and d == 1 in kaiser_bessel_ft.

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

# --------------------------------------------------------------------------------------
# DTypes
# --------------------------------------------------------------------------------------

DTYPE_MAP = [
    (torch.complex128, torch.float64),
    (torch.complex64, torch.float32),
]

# --------------------------------------------------------------------------------------
# Minimal COO sparse compatibility (enough for build_table() usage)
# --------------------------------------------------------------------------------------

class _COOColView:
    """Minimal object to mimic SciPy's getcol(c).todense() behavior used in build_table()."""

    def __init__(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        data: np.ndarray,
        nrows: int,
        col: int,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._data = data
        self._nrows = nrows
        self._col = int(col)

    def todense(self) -> np.ndarray:
        """Return a (nrows, 1) dense column vector with entries for this column."""
        out = np.zeros((self._nrows, 1), dtype=self._data.dtype)
        mask = (self._cols == self._col)
        if mask.any():
            out[self._rows[mask], 0] = self._data[mask]
        return out


class _COOMatrixCompat:
    """A tiny COO-like matrix class sufficient for this module.

    Exposes a minimal SciPy-compatible surface:
      - .shape
      - .row, .col, .data (readonly views)
      - .getcol(col).todense()
      - .astype(dtype, copy=True)
      - .toarray() / .todense()
      - .T (transpose)
    """

    def __init__(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
    ) -> None:
        self._rows = np.asarray(rows, dtype=np.int64)
        self._cols = np.asarray(cols, dtype=np.int64)
        self._data = np.asarray(data)
        self._shape = (int(shape[0]), int(shape[1]))

    # --- SciPy-like attributes ---
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def row(self) -> np.ndarray:
        return self._rows

    @property
    def col(self) -> np.ndarray:
        return self._cols

    @property
    def data(self) -> np.ndarray:
        return self._data

    # --- Basic ops used by our code/tests ---
    def getcol(self, col: int) -> "_COOColView":
        return _COOColView(self._rows, self._cols, self._data, self._shape[0], int(col))

    def astype(self, dtype, copy: bool = True) -> "_COOMatrixCompat":
        data = self._data.astype(dtype, copy=copy)
        # rows/cols are integer; only copy them if requested
        rows = self._rows.copy() if copy else self._rows
        cols = self._cols.copy() if copy else self._cols
        return _COOMatrixCompat(rows, cols, data, self._shape)

    def toarray(self) -> np.ndarray:
        out = np.zeros(self._shape, dtype=self._data.dtype)
        out[self._rows, self._cols] = self._data
        return out

    def todense(self) -> np.ndarray:
        return self.toarray()

    @property
    def T(self) -> "_COOMatrixCompat":
        # transpose: swap rows/cols and shape
        return _COOMatrixCompat(self._cols, self._rows, self._data, (self._shape[1], self._shape[0]))



# --------------------------------------------------------------------------------------
# SciPy-free Bessel helpers (for the specific cases used in this module)
# --------------------------------------------------------------------------------------

def _iv0(x: np.ndarray) -> np.ndarray:
    """Modified Bessel function of the first kind, order 0: I0(x).
    NumPy provides this as np.i0, returning a real array.
    """
    return np.i0(x)


def _jv_half(z: np.ndarray) -> np.ndarray:
    """Bessel function J_{1/2}(z) in closed form: sqrt(2/(pi*z)) * sin(z).
    Input may be complex; output is complex. J_{1/2}(0) = 0.
    """
    z = z.astype(np.complex128, copy=False)
    out = np.zeros_like(z, dtype=np.complex128)
    nz = (z != 0)
    out[nz] = np.sqrt(2.0 / (np.pi * z[nz])) * np.sin(z[nz])
    # at z==0, value is 0 (already set)
    return out


# --------------------------------------------------------------------------------------
# Core utilities (SciPy-free)
# --------------------------------------------------------------------------------------

def build_numpy_spmatrix(
    omega: np.ndarray,
    numpoints: Sequence[int],
    im_size: Sequence[int],
    grid_size: Sequence[int],
    n_shift: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> _COOMatrixCompat:
    """Build a sparse (COO-like) matrix with interpolation coefficients.

    Args:
        omega: Array of coordinates to interpolate to (radians/voxel), shape (ndims, K).
        numpoints: Number of neighbors for interpolation in each dimension.
        im_size: Size of base image per dimension.
        grid_size: Size of the interpolation grid per dimension.
        n_shift: FFT shift per dimension.
        order: Order of Kaiser–Bessel kernel (only 0.0 supported here).
        alpha: KB parameter per dimension (typically kbwidth * numpoints[d]).

    Returns:
        A COO-like sparse interpolation matrix with interface:
            - .shape
            - .getcol(col).todense()
    """
    ndims = omega.shape[0]
    klength = omega.shape[1]

    # calculate interpolation coefficients using KB kernel
    def interp_coeff(om, npts, grdsz, alpha_val, order_val):
        gam = 2 * np.pi / grdsz
        interp_dist = om / gam - np.floor(om / gam - npts / 2)
        Jvec = np.reshape(np.arange(1, npts + 1), (1, npts))
        kern_in = -1 * Jvec + np.expand_dims(interp_dist, 1)

        cur_coeff = np.zeros(shape=kern_in.shape, dtype=np.complex128)
        indices = np.abs(kern_in) < npts / 2
        bess_arg = np.sqrt(1 - (kern_in[indices] / (npts / 2)) ** 2)

        # Only order==0 supported: iv(0, x) == I0(x)
        if order_val != 0.0:
            raise NotImplementedError("Kaiser–Bessel iv(order, ·) only implemented for order==0.")
        denom = _iv0(alpha_val)
        cur_coeff[indices] = _iv0(alpha_val * bess_arg) / denom

        cur_coeff = np.real(cur_coeff)  # coefficients are real
        return cur_coeff, kern_in

    full_coef: List[np.ndarray] = []
    kd: List[np.ndarray] = []

    for (
        it_om,
        it_im_size,
        it_grid_size,
        it_numpoints,
        _dup_om,          # duplicated omega in original zip
        it_alpha,
        it_order,
    ) in zip(omega, im_size, grid_size, numpoints, omega, alpha, order):
        # interpolation coefficients
        coef, kern_in = interp_coeff(it_om, it_numpoints, it_grid_size, it_alpha, it_order)

        gam = 2 * np.pi / it_grid_size
        phase_scale = 1j * gam * (it_im_size - 1) / 2
        phase = np.exp(phase_scale * kern_in)
        full_coef.append(phase * coef)

        # nufft_offset
        koff = np.expand_dims(np.floor(it_om / gam - it_numpoints / 2), 1)
        Jvec = np.reshape(np.arange(1, it_numpoints + 1), (1, it_numpoints))
        kd.append(np.mod(Jvec + koff, it_grid_size) + 1)

    for i in range(len(kd)):
        kd[i] = (kd[i] - 1) * np.prod(grid_size[i + 1 :])

    # assemble block outer sums/products across dimensions
    kk = kd[0]
    spmat_coef = full_coef[0]
    for i in range(1, ndims):
        Jprod = int(np.prod(numpoints[: i + 1]))
        # block outer sum for indices
        kk = np.reshape(
            np.expand_dims(kk, 1) + np.expand_dims(kd[i], 2), (klength, Jprod)
        )
        # block outer product for coefficients
        spmat_coef = np.reshape(
            np.expand_dims(spmat_coef, 1) * np.expand_dims(full_coef[i], 2),
            (klength, Jprod),
        )

    # build in fftshift phase
    phase = np.exp(1j * np.dot(omega.T, np.expand_dims(n_shift, 1)))
    spmat_coef = np.conj(spmat_coef) * phase

    # coordinates in sparse matrix
    trajind = np.expand_dims(np.arange(klength), 1)
    trajind = np.repeat(trajind, int(np.prod(numpoints)), axis=1)

    # build the COO-like structure
    rows = trajind.flatten().astype(np.int64, copy=False)
    cols = kk.flatten().astype(np.int64, copy=False)
    data = spmat_coef.flatten().astype(np.complex128, copy=False)
    shape = (klength, int(np.prod(grid_size)))

    return _COOMatrixCompat(rows, cols, data, shape)


def build_table(
    im_size: Sequence[int],
    grid_size: Sequence[int],
    numpoints: Sequence[int],
    table_oversamp: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> List[Tensor]:
    """Build interpolation tables for each dimension (Fessler trick)."""
    table: List[Tensor] = []

    for (
        it_im_size,
        it_grid_size,
        it_numpoints,
        it_table_oversamp,
        it_order,
        it_alpha,
    ) in zip(im_size, grid_size, numpoints, table_oversamp, order, alpha):
        # Fessler's broadcast trick to build 1D table
        t1 = (
            it_numpoints / 2
            - 1
            + np.arange(it_table_oversamp) / it_table_oversamp
        )  # [L]
        om1 = t1 * 2 * np.pi / it_grid_size  # gamma
        s1 = build_numpy_spmatrix(
            np.expand_dims(om1, 0),
            numpoints=(it_numpoints,),
            im_size=(it_im_size,),
            grid_size=(it_grid_size,),
            n_shift=(0,),
            order=(it_order,),
            alpha=(it_alpha,),
        )

        # Stack columns (it_numpoints-1 .. 0)
        h = np.array(s1.getcol(it_numpoints - 1).todense())
        for col in range(it_numpoints - 2, -1, -1):
            h = np.concatenate((h, np.array(s1.getcol(col).todense())), axis=0)
        h = np.concatenate((h.flatten(), np.array([0])))

        table.append(torch.tensor(h))

    return table


def kaiser_bessel_ft(
    omega: np.ndarray, numpoints: int, alpha: float, order: float, d: int
) -> np.ndarray:
    """Compute FT of the Kaiser–Bessel function for image-domain scaling.

    This implementation currently supports order == 0 and d == 1, which is the configuration
    used by torchkbnufft's default pipelines.

    Args:
        omega: Frequency coordinates.
        numpoints: Kernel width (J).
        alpha: KB parameter (typically kbwidth * J).
        order: KB order (only 0.0 supported here).
        d: Dimension parameter (only 1 supported here).

    Returns:
        Real-valued scaling coefficients as a NumPy array.
    """
    # Complex-valued z, as in original code
    z = np.sqrt((2 * np.pi * (numpoints / 2) * omega) ** 2 - alpha**2 + 0j)

    if order != 0.0 or d != 1:
        raise NotImplementedError("kaiser_bessel_ft is implemented for order==0 and d==1 only.")

    # With order==0 and d==1: nu = d/2 + order = 1/2
    # jv(1/2, z) = sqrt(2/(pi*z)) * sin(z), and iv(0, x) = I0(x) = np.i0(x)
    # Original scaling:
    # scaling = (2*pi)^(d/2) * ( (J/2)^d ) * (alpha^order) / iv(order, alpha) * jv(nu, z) / z^nu
    # Here: (2*pi)^(1/2) * (J/2) * 1 / I0(alpha) * jv(1/2, z) / z^(1/2)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling_coef = (
            (2 * np.pi) ** 0.5
            * (numpoints / 2)
            * (1.0 / _iv0(alpha))
            * (_jv_half(z) / (z ** 0.5))
        )

    scaling_coef = np.real(scaling_coef)
    return scaling_coef


def compute_scaling_coefs(
    im_size: Sequence[int],
    grid_size: Sequence[int],
    numpoints: Sequence[int],
    alpha: Sequence[float],
    order: Sequence[float],
) -> Tensor:
    """Compute image-domain scaling coefficients for NUFFT."""
    # Validate supported configuration early
    if any(o != 0.0 for o in order):
        raise NotImplementedError("compute_scaling_coefs supports order==0 only.")
    if len(im_size) != len(grid_size) or len(im_size) != len(numpoints):
        raise ValueError("im_size, grid_size, numpoints must have the same length.")

    num_coefs = np.arange(im_size[0]) - (im_size[0] - 1) / 2
    scaling_coef = 1 / kaiser_bessel_ft(
        num_coefs / grid_size[0], numpoints[0], alpha[0], order[0], 1
    )

    if numpoints[0] == 1:
        scaling_coef = np.ones_like(scaling_coef)

    for i in range(1, len(im_size)):
        indlist = np.arange(im_size[i]) - (im_size[i] - 1) / 2
        scaling_coef = np.expand_dims(scaling_coef, axis=-1)

        tmp = 1 / kaiser_bessel_ft(
            indlist / grid_size[i], numpoints[i], alpha[i], order[i], 1
        )

        for _ in range(i):
            tmp = tmp[np.newaxis]

        if numpoints[i] == 1:
            tmp = np.ones_like(tmp)

        scaling_coef = scaling_coef * tmp

    return torch.tensor(scaling_coef)


def init_fn(
    im_size: Sequence[int],
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2**10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[
    List[Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    """Initialization function for NUFFT objects."""
    (
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        order,
        alpha,
        dtype,
        device,
    ) = validate_args(
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        kbwidth,
        order,
        dtype,
        device,
    )

    tables = build_table(
        numpoints=numpoints,
        table_oversamp=table_oversamp,
        grid_size=grid_size,
        im_size=im_size,
        order=order,
        alpha=alpha,
    )
    assert len(tables) == len(im_size)

    # precompute interpolation offsets
    offset_list = list(itertools.product(*[range(numpoint) for numpoint in numpoints]))

    if dtype.is_floating_point:
        real_dtype = dtype
        complex_dtype = None
        for pair in DTYPE_MAP:
            if pair[1] == real_dtype:
                complex_dtype = pair[0]
                break
        if complex_dtype is None:
            raise TypeError("Unsupported floating dtype.")
    elif dtype.is_complex:
        complex_dtype = dtype
        real_dtype = None
        for pair in DTYPE_MAP:
            if pair[0] == complex_dtype:
                real_dtype = pair[1]
                break
        if real_dtype is None:
            raise TypeError("Unsupported complex dtype.")
    else:
        raise TypeError("Unrecognized dtype.")

    tables = [table.to(dtype=complex_dtype, device=device) for table in tables]

    return (
        tables,
        torch.tensor(im_size, dtype=torch.long, device=device),
        torch.tensor(grid_size, dtype=torch.long, device=device),
        torch.tensor(n_shift, dtype=real_dtype, device=device),  # type: ignore[arg-type]
        torch.tensor(numpoints, dtype=torch.long, device=device),
        torch.tensor(offset_list, dtype=torch.long, device=device),
        torch.tensor(table_oversamp, dtype=torch.long, device=device),
        torch.tensor(order, dtype=real_dtype, device=device),    # type: ignore[arg-type]
        torch.tensor(alpha, dtype=real_dtype, device=device),    # type: ignore[arg-type]
    )


def validate_args(
    im_size: Sequence[int],
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2**10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[float],
    Sequence[float],
    torch.dtype,
    torch.device,
]:
    """Validate and normalize NUFFT initialization arguments."""
    im_size = tuple(int(v) for v in im_size)

    if grid_size is None:
        grid_size = tuple([dim * 2 for dim in im_size])
    else:
        grid_size = tuple(int(v) for v in grid_size)

    if isinstance(numpoints, int):
        numpoints = tuple([int(numpoints) for _ in range(len(grid_size))])
    else:
        numpoints = tuple(int(v) for v in numpoints)

    if n_shift is None:
        n_shift = tuple([dim // 2 for dim in im_size])
    else:
        n_shift = tuple(int(v) for v in n_shift)

    if isinstance(table_oversamp, int):
        table_oversamp = tuple(int(table_oversamp) for _ in range(len(grid_size)))
    else:
        table_oversamp = tuple(int(v) for v in table_oversamp)

    alpha = tuple(float(kbwidth) * numpoint for numpoint in numpoints)

    if isinstance(order, float) or isinstance(order, int):
        order = tuple(float(order) for _ in range(len(grid_size)))
    else:
        order = tuple(float(v) for v in order)

    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None:
        device = torch.device("cpu")

    # dimension checking
    assert len(grid_size) == len(im_size)
    assert len(n_shift) == len(im_size)
    assert len(numpoints) == len(im_size)
    assert len(alpha) == len(im_size)
    assert len(order) == len(im_size)
    assert len(table_oversamp) == len(im_size)

    return (
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        order,
        alpha,
        dtype,
        device,
    )
