"""Small set of core Stage-4 calculations, distilled from prior exact runs.

The finite-rank secular equation finds E=0 crossings for H(h)=H0+hW.
It does not by itself establish a thermodynamic bulk topological transition.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .marker import exact_marker, diagonalize, marker_from_occupied


def secular_roots(H0: sparse.spmatrix, W: sparse.spmatrix, *,
                  imaginary_tolerance: float = 1e-6,
                  deduplicate_tolerance: float = 1e-7) -> np.ndarray:
    """All real candidate zero-energy roots, without assuming a sign/window.

    det(H0+hW)=det(H0)det(I+h H0^{-1}W) for invertible H0.
    Only the nonzero support of W is used. H0 must be invertible at E=0.
    """
    W = W.tocsr()
    support = np.flatnonzero(np.diff(W.indptr) != 0)
    if len(support) == 0:
        return np.array([], dtype=float)
    if H0.shape != W.shape:
        raise ValueError('Inconsistent Hamiltonian dimensions.')
    sources = np.zeros((H0.shape[0], len(support)), dtype=np.complex128)
    sources[support, np.arange(len(support))] = 1
    G = splu(H0.tocsc()).solve(sources)[support, :]
    Wsmall = W[support, :][:, support].toarray()
    eigenvalues = np.linalg.eigvals(G @ Wsmall)
    candidates = sorted(float(-1.0 / lam.real) for lam in eigenvalues
                        if abs(lam.imag) <= imaginary_tolerance
                        and abs(lam.real) > 1e-10)
    unique = []
    for root in candidates:
        if not unique or abs(root - unique[-1]) > deduplicate_tolerance:
            unique.append(root)
    return np.asarray(unique)


def root_in_window(roots: np.ndarray, lo: float, hi: float) -> float:
    """Choose central candidate; inspect candidates when multiple distinct roots exist."""
    found = np.asarray([x for x in roots if lo < x < hi])
    if len(found) == 0:
        raise ValueError(f'No real secular root in ({lo}, {hi}). Roots: {roots}')
    if len(found) > 1 and np.ptp(found) > 1e-5:
        raise ValueError(f'Multiple distinct roots in ({lo}, {hi}): {found}; narrow window.')
    return float(np.median(found))


def site_masks(regions, *, edge_layers: int = 2):
    """Mutually exclusive partitions for a centered defect sample."""
    m = regions.model
    Lx, Ly = m.Lx, m.Ly
    b0, b1, c0, c1 = regions.defect_bounds
    B = np.zeros((Ly, Lx), dtype=bool)
    B[c0:c1, b0:b1] = True
    E = np.zeros((Ly, Lx), dtype=bool)
    k = edge_layers
    if k:
        E[:k] = E[-k:] = True
        E[:, :k] = E[:, -k:] = True
    R = regions.integration_mask()
    bb = np.repeat(B.ravel(), 2)
    edge = np.repeat(E.ravel(), 2)
    if np.any(R & edge):
        raise ValueError('Integration region overlaps the edge partition.')
    return {'B': bb, 'R_minus_B': R & ~bb,
            'interior_outside_R': ~R & ~edge,
            'edge': edge, 'R': R}


def region_sums(delta_marker: np.ndarray, masks: dict) -> dict[str, float]:
    keys = ('B', 'R_minus_B', 'interior_outside_R', 'edge', 'R')
    return {k: float(np.sum(delta_marker[masks[k]])) for k in keys} | {
        'full_sample': float(np.sum(delta_marker))}


@dataclass
class CrossingSample:
    h: float
    energies: np.ndarray
    eigvecs: np.ndarray
    delta_marker: np.ndarray
    integrated_marker: float
    n_occupied: int


def exact_crossing_sample(H0, W, positions, integration_mask, pristine_marker,
                          h: float, *, mu: float = 0.0, max_dim: int = 3000):
    energies, U = diagonalize(H0 + h*W, max_dim=max_dim)
    nocc = int(np.searchsorted(energies, mu, side='left'))
    d = marker_from_occupied(U[:, :nocc], positions)-pristine_marker
    return CrossingSample(float(h), energies, U, d, float(d[integration_mask].sum()), nocc)


def occupation_swap_jump(minus: CrossingSample, plus: CrossingSample,
                         positions: np.ndarray, pristine_marker: np.ndarray,
                         integration_mask: np.ndarray) -> dict:
    """Replace ONLY highest occupied state on minus side by lowest empty.

    The rest of the full occupied projector is held fixed; this is a diagnostic,
    not a self-contained two-state model. Must use strict one-sided points.
    """
    no = minus.n_occupied
    if no != plus.n_occupied:
        raise ValueError('Occupied count differs across crossing; swap is not applicable.')
    occ = minus.eigvecs[:, :no].copy()
    occ[:, -1] = minus.eigvecs[:, no]
    marker_swapped = marker_from_occupied(occ, positions)-pristine_marker
    pred = float(marker_swapped[integration_mask].sum()-minus.integrated_marker)
    actual = float(plus.integrated_marker-minus.integrated_marker)
    return dict(predicted_jump=pred, exact_jump=actual,
                relative_error=abs(pred-actual)/max(abs(actual), 1e-12),
                occupied_after_overlap=float(abs(np.vdot(minus.eigvecs[:, no], plus.eigvecs[:, no-1]))**2))


def frozen_reference_reconstruction(H0, W, positions, pristine_marker, integration_mask,
                                    hc: float, h_values, *,
                                    reference_offset: float = -0.002,
                                    dimension: int = 4, max_dim: int = 3000):
    """No-fit fixed-reference 2/4-level model with frozen deep occupied states.

    Returns predicted M_R. This approximation had realization-dependent accuracy
    in the original project; it is NOT a general low-energy exact mapping.
    """
    if dimension not in (2, 4):
        raise ValueError('Use a 2- or 4-state subspace.')
    h_ref = hc + reference_offset
    E, U = diagonalize(H0 + h_ref*W, max_dim=max_dim)
    no = int(np.searchsorted(E, 0.0, side='left'))
    count = dimension // 2
    indices = np.arange(no-count, no+count)
    low = U[:, indices]
    deep = U[:, :no-count]
    D = np.diag(E[indices]); velocity = low.conj().T @ (W @ low)
    out = []
    for h in h_values:
        _, z = np.linalg.eigh(D+(h-h_ref)*velocity)
        occupied = np.column_stack((deep, low@z[:, :count]))
        d = marker_from_occupied(occupied, positions)-pristine_marker
        out.append(float(d[integration_mask].sum()))
    return np.asarray(out)
