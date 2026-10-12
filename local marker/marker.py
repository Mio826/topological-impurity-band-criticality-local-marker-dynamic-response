"""Exact OBC local Chern marker, reusing the Stage-4 projector convention.

C_i = 4*pi*Im <i|Q x P y Q|i> = 4*pi*Im <i|P x P y P|i>
for ordinary finite-sample Cartesian coordinate operators x and y.
We use the unnormalized site marker (no area normalization).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MarkerResult:
    marker: np.ndarray
    energies: np.ndarray
    n_occupied: int
    gap: float


def marker_from_occupied(occ: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Site-resolved marker from an orthonormal occupied-state basis."""
    x, y = np.asarray(positions).T
    xocc = x[:, None] * occ
    yocc = y[:, None] * occ
    qx = xocc - occ @ (occ.conj().T @ xocc)
    qy = yocc - occ @ (occ.conj().T @ yocc)
    return 4 * np.pi * np.imag(np.sum(qx * np.conj(qy), axis=1))


def diagonalize(H: sparse.spmatrix | np.ndarray, max_dim: int = 3000):
    """Dense exact eigenpairs; refuses accidentally huge matrices."""
    n = H.shape[0]
    if n > max_dim:
        raise MemoryError(f'Exact diagonalization dimension {n} > {max_dim}. '
                          'Choose a smaller L or explicitly raise max_dim with enough RAM.')
    arr = H.toarray() if sparse.issparse(H) else np.asarray(H)
    return np.linalg.eigh(arr)


def exact_marker(H: sparse.spmatrix | np.ndarray, positions: np.ndarray,
                 mu: float = 0.0, max_dim: int = 3000) -> MarkerResult:
    energies, U = diagonalize(H, max_dim=max_dim)
    nocc = int(np.searchsorted(energies, mu, side='left'))
    if not 0 < nocc < len(energies):
        raise ValueError('Fermi level has no occupied or empty subspace.')
    val = marker_from_occupied(U[:, :nocc], positions)
    return MarkerResult(val, energies, nocc, float(energies[nocc]-energies[nocc-1]))


def defect_marker(H, H0, positions, mu: float = 0.0, max_dim: int = 3000):
    """Return (delta C, defective, pristine); pristine may be pre-cached separately."""
    background = exact_marker(H0, positions, mu, max_dim)
    result = exact_marker(H, positions, mu, max_dim)
    return result.marker-background.marker, result, background
