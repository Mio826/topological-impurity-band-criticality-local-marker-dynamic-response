"""
optical_response.py
===================

Frequency-resolved optical response for the finite-density topological-
impurity project.

This module is designed to sit beside ``qwz.py``, ``haldane.py``,
``geometry.py``, ``disorder.py``, and ``markers.py``.  It deliberately does
not define microscopic models, sample disorder, construct A/B/C regions, or
plot data.  Its job is

    (H, positions, mu) -> current operators -> interband transitions
                       -> Re sigma_xx(omega), Re sigma_yy(omega)
                       -> circular absorption sigma_+/sigma_-
                       -> Im sigma_xy(omega) and MCD difference.

The production implementation is intentionally OBC-only.  This matches the
existing A superset B superset C framework and makes the Cartesian position
operators well defined.  PBC optical response should be implemented later
through Peierls/twist derivatives of the Hamiltonian, not through a naive
position commutator.

Conventions
-----------
* ``e_charge`` is the positive elementary-charge magnitude.
* Defaults use natural units e = hbar = 1.
* Occupied states satisfy E < mu at T=0.
* The charge-current operator is

      J_alpha = -(i e / hbar) [H, r_alpha].

* For omega > 0 the regular longitudinal spectrum is

      Re sigma_xx(omega)
        = (pi/A) sum_{o,u} |Jx_ou|^2 / (Eu-Eo)
          delta(omega - (Eu-Eo)/hbar).

* Circular currents are J_+ = (Jx + i Jy)/sqrt(2) and
  J_- = (Jx - i Jy)/sqrt(2).  With this convention

      (sigma_+ - sigma_-)/2 = Im sigma_xy(abs)

  for the absorptive interband spectrum implemented here.  Therefore the
  full circular-dichroic absorption difference is

      D_MCD(omega) = sigma_+(omega) - sigma_-(omega)
                   = 2 Im sigma_xy(abs)(omega).

* ``integrated_im_sigma_xy_exact`` is the quantity directly matched to the
  whole-sample average of the Resta integrated-MCD marker in ``markers.py``.
  The experimental circular absorption difference carries an additional
  factor of two in the above convention.

Important A/B/C note
--------------------
The Kubo conductivity produced here is a *global* optical response of the
open finite sample, normalized by the total nominal area N_cell*A_cell.
The A/B/C construction is still used to place impurities away from the edge
and to evaluate local markers in C, but a global conductivity is not the same
object as a C-window average.  For exact sum-rule checks compare the spectral
integral to the *whole-sample* average of the local MCD marker.  Edge-state
contamination in low-frequency OBC spectra should be diagnosed separately
(e.g. via state probability in an edge/bulk mask) rather than silently
removed from the Kubo definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import fftconvolve
except ImportError as exc:  # pragma: no cover
    raise ImportError("optical_response.py requires scipy") from exc


ArrayF = npt.NDArray[np.float64]
ArrayC = npt.NDArray[np.complex128]
ArrayB = npt.NDArray[np.bool_]
Kernel = Literal["gaussian", "lorentzian", "none"]


def _as_dense_hermitian(H: Any, atol: float = 1.0e-10) -> ArrayC:
    if sp is not None and sp.issparse(H):
        A = H.toarray().astype(np.complex128, copy=False)
    else:
        A = np.asarray(H, dtype=np.complex128)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"H must be square; received shape {A.shape}")

    residual = float(np.max(np.abs(A - A.conj().T))) if A.size else 0.0
    if residual > float(atol):
        raise ValueError(
            f"H must be Hermitian; max|H-H^dagger|={residual:.3e}"
        )
    return A


def _as_positions(positions: Any, dim: int) -> ArrayF:
    pos = np.asarray(positions, dtype=float)
    if pos.shape != (dim, 2):
        raise ValueError(
            f"positions must have shape ({dim},2); received {pos.shape}"
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite")
    return pos


def _uniform_grid(omega: Any) -> tuple[ArrayF, float]:
    grid = np.asarray(omega, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("omega must be a 1D grid with at least two points")
    if not np.all(np.isfinite(grid)):
        raise ValueError("omega grid must be finite")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("omega grid must be strictly increasing")
    dw = float(np.mean(np.diff(grid)))
    if not np.allclose(np.diff(grid), dw, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("omega grid must be uniformly spaced")
    return grid, dw


def _histogram_density(
    frequencies: ArrayF,
    weights: ArrayF,
    omega: ArrayF,
) -> ArrayF:
    """Histogram delta peaks onto a uniform grid as a density in omega."""
    grid, dw = _uniform_grid(omega)
    edges = np.empty(len(grid) + 1, dtype=float)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5 * dw
    edges[-1] = grid[-1] + 0.5 * dw

    hist, _ = np.histogram(frequencies, bins=edges, weights=weights)
    return np.asarray(hist, dtype=float) / dw


def _broaden_density(
    density: ArrayF,
    omega: ArrayF,
    eta: float,
    kernel: Kernel,
) -> ArrayF:
    """Broaden a histogram density without changing its continuum units."""
    grid, dw = _uniform_grid(omega)
    eta = float(eta)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if kernel not in ("gaussian", "lorentzian", "none"):
        raise ValueError("kernel must be 'gaussian', 'lorentzian', or 'none'")
    if kernel == "none" or eta == 0.0:
        return np.asarray(density, dtype=float).copy()

    if kernel == "gaussian":
        sigma_bins = eta / dw
        return np.asarray(
            gaussian_filter1d(
                np.asarray(density, dtype=float),
                sigma=sigma_bins,
                mode="constant",
                cval=0.0,
                truncate=6.0,
            ),
            dtype=float,
        )

    # Discrete Lorentzian convolution.  Kernel is normalized so that
    # sum(kernel) * dw = 1, preserving integrated spectral weight away from
    # finite-grid boundary leakage.
    offsets = (np.arange(len(grid)) - len(grid) // 2) * dw
    lor = (eta / np.pi) / (offsets**2 + eta**2)
    norm = float(np.sum(lor) * dw)
    if norm <= 0:
        raise RuntimeError("failed to normalize Lorentzian kernel")
    lor /= norm
    broadened = fftconvolve(np.asarray(density, dtype=float), lor, mode="same") * dw
    return np.asarray(broadened, dtype=float)


@dataclass(frozen=True)
class EigensystemData:
    eigenvalues: ArrayF
    eigenvectors: ArrayC
    occupied: ArrayB
    mu: float
    nearest_fermi_distance: float
    near_fermi_count: int

    @property
    def n_occupied(self) -> int:
        return int(np.sum(self.occupied))

    @property
    def n_empty(self) -> int:
        return int(len(self.occupied) - np.sum(self.occupied))


@dataclass(frozen=True)
class TransitionData:
    """All T=0 occupied -> empty interband transitions."""

    occupied_indices: npt.NDArray[np.int64]
    empty_indices: npt.NDArray[np.int64]
    delta_E: ArrayF                 # shape (Nocc, Nemp)
    omega: ArrayF                   # delta_E / hbar
    Jx_ou: ArrayC                   # <o|Jx|u>
    Jy_ou: ArrayC                   # <o|Jy|u>
    weight_xx: ArrayF               # includes pi/A and 1/delta_E
    weight_yy: ArrayF
    weight_im_sigma_xy: ArrayF      # absorptive Im sigma_xy weight
    weight_sigma_plus: ArrayF
    weight_sigma_minus: ArrayF

    @property
    def shape(self) -> tuple[int, int]:
        return self.delta_E.shape

    @property
    def n_transitions(self) -> int:
        return int(self.delta_E.size)

    @property
    def weight_mcd_difference(self) -> ArrayF:
        """sigma_+ - sigma_- = 2 Im sigma_xy in this convention."""
        return self.weight_sigma_plus - self.weight_sigma_minus

    def circular_identity_residual(self) -> float:
        lhs = self.weight_mcd_difference
        rhs = 2.0 * self.weight_im_sigma_xy
        return float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0


@dataclass(frozen=True)
class OpticalSpectra:
    omega: ArrayF
    sigma_xx: ArrayF
    sigma_yy: ArrayF
    sigma_plus: ArrayF
    sigma_minus: ArrayF
    im_sigma_xy: ArrayF
    mcd_difference: ArrayF
    eta: float
    kernel: Kernel

    def numerical_integrals(self) -> dict[str, float]:
        """Numerical omega integrals over the supplied finite grid."""
        return {
            "sigma_xx": float(np.trapezoid(self.sigma_xx, self.omega)),
            "sigma_yy": float(np.trapezoid(self.sigma_yy, self.omega)),
            "sigma_plus": float(np.trapezoid(self.sigma_plus, self.omega)),
            "sigma_minus": float(np.trapezoid(self.sigma_minus, self.omega)),
            "im_sigma_xy": float(np.trapezoid(self.im_sigma_xy, self.omega)),
            "mcd_difference": float(np.trapezoid(self.mcd_difference, self.omega)),
        }


class OpticalResponseCalculator:
    """
    Dense OBC Kubo/tQGT backend for QWZ and Haldane project models.

    The class diagonalizes H once, constructs Cartesian current operators from
    position commutators, and vectorizes all occupied-to-empty matrix elements.
    """

    def __init__(
        self,
        H: Any,
        positions: Any,
        mu: float,
        total_area: float,
        *,
        e_charge: float = 1.0,
        hbar: float = 1.0,
        fermi_tolerance: float = 1.0e-10,
        hermiticity_tolerance: float = 1.0e-10,
        eigenvalues: Optional[Any] = None,
        eigenvectors: Optional[Any] = None,
    ) -> None:
        self.H = _as_dense_hermitian(H, atol=hermiticity_tolerance)
        self.dim = self.H.shape[0]
        self.positions = _as_positions(positions, self.dim)
        self.mu = float(mu)
        self.total_area = float(total_area)
        self.e_charge = float(e_charge)
        self.hbar = float(hbar)
        self.fermi_tolerance = float(fermi_tolerance)

        for name, value in (
            ("mu", self.mu),
            ("total_area", self.total_area),
            ("e_charge", self.e_charge),
            ("hbar", self.hbar),
            ("fermi_tolerance", self.fermi_tolerance),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.total_area <= 0:
            raise ValueError("total_area must be positive")
        if self.hbar <= 0:
            raise ValueError("hbar must be positive")
        if self.fermi_tolerance < 0:
            raise ValueError("fermi_tolerance must be nonnegative")

        self.Jx, self.Jy = self._build_current_operators()
        self.eigensystem = self._build_eigensystem(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        self.transitions = self._build_transition_data()

    @classmethod
    def from_model(
        cls,
        model: Any,
        H: Any,
        mu: float,
        **kwargs: Any,
    ) -> "OpticalResponseCalculator":
        """Construct from the validated QWZ/Haldane OBC model interface."""
        if bool(getattr(model, "pbc_x", False)) or bool(getattr(model, "pbc_y", False)):
            raise ValueError(
                "OpticalResponseCalculator.from_model currently requires OBC. "
                "PBC must use a Peierls/twist-current implementation, not [H,r]."
            )
        if not hasattr(model, "positions"):
            raise TypeError("model must provide positions()")
        if not hasattr(model, "cell_area"):
            raise TypeError("model must provide cell_area")
        if not hasattr(model, "Lx") or not hasattr(model, "Ly"):
            raise TypeError("model must provide Lx and Ly")

        n_cells = int(model.Lx) * int(model.Ly)
        total_area = n_cells * float(model.cell_area)
        return cls(
            H=H,
            positions=model.positions(),
            mu=mu,
            total_area=total_area,
            **kwargs,
        )

    @classmethod
    def from_marker_calculator(
        cls,
        marker_calculator: Any,
        **kwargs: Any,
    ) -> "OpticalResponseCalculator":
        """
        Reuse the eigensystem already computed by ``markers.MarkerCalculator``.

        This avoids a second dense diagonalization when static markers and
        optical spectra are evaluated for the same H.
        """
        calc = marker_calculator
        proj = calc.projector
        total_area = float(calc.n_cells) * float(calc.cell_area)

        options = dict(
            e_charge=float(calc.e_charge),
            hbar=float(calc.hbar),
            fermi_tolerance=float(calc.fermi_tolerance),
            eigenvalues=proj.eigenvalues,
            eigenvectors=proj.eigenvectors,
        )
        options.update(kwargs)

        return cls(
            H=calc.H,
            positions=calc.positions,
            mu=float(calc.mu),
            total_area=total_area,
            **options,
        )

    def _build_current_operators(self) -> tuple[ArrayC, ArrayC]:
        x = self.positions[:, 0]
        y = self.positions[:, 1]

        # [H, X]_ij = H_ij (x_j - x_i) for diagonal X.
        comm_x = self.H * (x[None, :] - x[:, None])
        comm_y = self.H * (y[None, :] - y[:, None])
        pref = -1.0j * self.e_charge / self.hbar
        Jx = pref * comm_x
        Jy = pref * comm_y
        return np.asarray(Jx, dtype=np.complex128), np.asarray(Jy, dtype=np.complex128)

    def current_diagnostics(self) -> dict[str, float]:
        return {
            "Jx_hermiticity_residual": float(np.max(np.abs(self.Jx - self.Jx.conj().T))),
            "Jy_hermiticity_residual": float(np.max(np.abs(self.Jy - self.Jy.conj().T))),
        }

    def _build_eigensystem(
        self,
        eigenvalues: Optional[Any],
        eigenvectors: Optional[Any],
    ) -> EigensystemData:
        if (eigenvalues is None) != (eigenvectors is None):
            raise ValueError("provide both eigenvalues and eigenvectors, or neither")

        if eigenvalues is None:
            evals, evecs = np.linalg.eigh(self.H)
        else:
            evals = np.asarray(eigenvalues, dtype=float)
            evecs = np.asarray(eigenvectors, dtype=np.complex128)
            if evals.shape != (self.dim,):
                raise ValueError(
                    f"eigenvalues must have shape ({self.dim},); received {evals.shape}"
                )
            if evecs.shape != (self.dim, self.dim):
                raise ValueError(
                    f"eigenvectors must have shape ({self.dim},{self.dim}); received {evecs.shape}"
                )

            # Lightweight residual so an accidentally mismatched eigensystem is
            # caught before producing a plausible-looking spectrum.
            residual = np.max(
                np.abs(self.H @ evecs - evecs * evals[None, :])
            )
            if residual > 1.0e-8:
                raise ValueError(
                    f"provided eigensystem does not match H; residual={residual:.3e}"
                )

        occupied = np.asarray(evals < self.mu, dtype=bool)
        distances = np.abs(evals - self.mu)
        nearest = float(np.min(distances)) if len(distances) else np.inf
        near_count = int(np.sum(distances <= self.fermi_tolerance))

        return EigensystemData(
            eigenvalues=np.asarray(evals, dtype=float),
            eigenvectors=np.asarray(evecs, dtype=np.complex128),
            occupied=occupied,
            mu=self.mu,
            nearest_fermi_distance=nearest,
            near_fermi_count=near_count,
        )

    def _build_transition_data(self) -> TransitionData:
        es = self.eigensystem
        occ_idx = np.flatnonzero(es.occupied).astype(np.int64)
        emp_idx = np.flatnonzero(~es.occupied).astype(np.int64)

        if len(occ_idx) == 0 or len(emp_idx) == 0:
            raise ValueError("mu must leave at least one occupied and one empty state")

        Eo = es.eigenvalues[occ_idx]
        Eu = es.eigenvalues[emp_idx]
        delta_E = Eu[None, :] - Eo[:, None]
        if np.any(delta_E <= 0):
            raise RuntimeError("encountered nonpositive occupied-to-empty excitation energy")

        Uo = es.eigenvectors[:, occ_idx]
        Uu = es.eigenvectors[:, emp_idx]

        # Matrix elements <o|J|u>, fully vectorized through BLAS.
        Jx_ou = Uo.conj().T @ (self.Jx @ Uu)
        Jy_ou = Uo.conj().T @ (self.Jy @ Uu)

        pref = np.pi / self.total_area
        weight_xx = pref * (np.abs(Jx_ou) ** 2) / delta_E
        weight_yy = pref * (np.abs(Jy_ou) ** 2) / delta_E

        cross_im = np.imag(Jx_ou * np.conjugate(Jy_ou))
        weight_im_sigma_xy = pref * cross_im / delta_E

        Jplus_ou = (Jx_ou + 1.0j * Jy_ou) / np.sqrt(2.0)
        Jminus_ou = (Jx_ou - 1.0j * Jy_ou) / np.sqrt(2.0)
        weight_plus = pref * (np.abs(Jplus_ou) ** 2) / delta_E
        weight_minus = pref * (np.abs(Jminus_ou) ** 2) / delta_E

        return TransitionData(
            occupied_indices=occ_idx,
            empty_indices=emp_idx,
            delta_E=np.asarray(delta_E, dtype=float),
            omega=np.asarray(delta_E / self.hbar, dtype=float),
            Jx_ou=np.asarray(Jx_ou, dtype=np.complex128),
            Jy_ou=np.asarray(Jy_ou, dtype=np.complex128),
            weight_xx=np.asarray(weight_xx, dtype=float),
            weight_yy=np.asarray(weight_yy, dtype=float),
            weight_im_sigma_xy=np.asarray(weight_im_sigma_xy, dtype=float),
            weight_sigma_plus=np.asarray(weight_plus, dtype=float),
            weight_sigma_minus=np.asarray(weight_minus, dtype=float),
        )

    def exact_integrals(self) -> dict[str, float]:
        """Exact delta-function integrals before broadening/grid truncation."""
        t = self.transitions
        return {
            "sigma_xx": float(np.sum(t.weight_xx)),
            "sigma_yy": float(np.sum(t.weight_yy)),
            "sigma_plus": float(np.sum(t.weight_sigma_plus)),
            "sigma_minus": float(np.sum(t.weight_sigma_minus)),
            "im_sigma_xy": float(np.sum(t.weight_im_sigma_xy)),
            "mcd_difference": float(np.sum(t.weight_mcd_difference)),
        }

    def state_probability(self, basis_mask: Any) -> ArrayF:
        """Probability of every eigenstate in an arbitrary real-space basis mask."""
        mask = np.asarray(basis_mask, dtype=bool)
        if mask.shape != (self.dim,):
            raise ValueError(
                f"basis_mask must have shape ({self.dim},); received {mask.shape}"
            )
        if not np.any(mask):
            return np.zeros(self.dim, dtype=float)
        psi = self.eigensystem.eigenvectors
        return np.asarray(np.sum(np.abs(psi[mask, :]) ** 2, axis=0), dtype=float)

    def transition_character_factors(
        self,
        state_character: Any,
    ) -> dict[str, ArrayF]:
        """
        Soft BB/BI/II decomposition from a state character p_n in [0,1].

        Here p=1 means fully impurity-like and p=0 fully bulk-like.  This
        routine is intentionally agnostic about how p_n was defined.
        """
        p = np.asarray(state_character, dtype=float)
        if p.shape != (self.dim,):
            raise ValueError(
                f"state_character must have shape ({self.dim},); received {p.shape}"
            )
        if not np.all(np.isfinite(p)) or np.any(p < -1e-12) or np.any(p > 1.0 + 1e-12):
            raise ValueError("state_character must lie in [0,1]")
        p = np.clip(p, 0.0, 1.0)

        po = p[self.transitions.occupied_indices][:, None]
        pu = p[self.transitions.empty_indices][None, :]
        ii = po * pu
        bb = (1.0 - po) * (1.0 - pu)
        bi = po * (1.0 - pu) + (1.0 - po) * pu
        return {
            "BB": np.asarray(bb, dtype=float),
            "BI": np.asarray(bi, dtype=float),
            "II": np.asarray(ii, dtype=float),
        }

    def _spectrum_from_weight(
        self,
        omega: ArrayF,
        transition_weight: ArrayF,
        *,
        eta: float,
        kernel: Kernel,
        transition_factor: Optional[Any] = None,
    ) -> ArrayF:
        w = np.asarray(transition_weight, dtype=float)
        if w.shape != self.transitions.shape:
            raise ValueError(
                f"transition_weight must have shape {self.transitions.shape}; received {w.shape}"
            )

        if transition_factor is not None:
            factor = np.asarray(transition_factor, dtype=float)
            if factor.shape != self.transitions.shape:
                raise ValueError(
                    f"transition_factor must have shape {self.transitions.shape}; received {factor.shape}"
                )
            w = w * factor

        density = _histogram_density(
            self.transitions.omega.ravel(),
            w.ravel(),
            omega,
        )
        return _broaden_density(density, omega, eta=eta, kernel=kernel)

    def compute_spectra(
        self,
        omega: Any,
        *,
        eta: float = 0.05,
        kernel: Kernel = "gaussian",
        transition_factor: Optional[Any] = None,
    ) -> OpticalSpectra:
        """
        Evaluate broadened spectra on a uniform angular-frequency grid.

        ``transition_factor`` can be used for a soft decomposition; it must
        have shape (Nocc,Nemp).  The same factor is applied consistently to
        all optical channels.
        """
        grid, _ = _uniform_grid(omega)
        t = self.transitions

        xx = self._spectrum_from_weight(
            grid, t.weight_xx, eta=eta, kernel=kernel,
            transition_factor=transition_factor,
        )
        yy = self._spectrum_from_weight(
            grid, t.weight_yy, eta=eta, kernel=kernel,
            transition_factor=transition_factor,
        )
        plus = self._spectrum_from_weight(
            grid, t.weight_sigma_plus, eta=eta, kernel=kernel,
            transition_factor=transition_factor,
        )
        minus = self._spectrum_from_weight(
            grid, t.weight_sigma_minus, eta=eta, kernel=kernel,
            transition_factor=transition_factor,
        )
        imxy = self._spectrum_from_weight(
            grid, t.weight_im_sigma_xy, eta=eta, kernel=kernel,
            transition_factor=transition_factor,
        )

        # Use the circular channels themselves for the plotted MCD difference;
        # their exact transition weights satisfy mcd = 2*Im sigma_xy.
        mcd = plus - minus

        return OpticalSpectra(
            omega=np.asarray(grid, dtype=float),
            sigma_xx=np.asarray(xx, dtype=float),
            sigma_yy=np.asarray(yy, dtype=float),
            sigma_plus=np.asarray(plus, dtype=float),
            sigma_minus=np.asarray(minus, dtype=float),
            im_sigma_xy=np.asarray(imxy, dtype=float),
            mcd_difference=np.asarray(mcd, dtype=float),
            eta=float(eta),
            kernel=kernel,
        )

    def diagnostics(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        out.update(self.current_diagnostics())
        out.update({
            "n_occupied": self.eigensystem.n_occupied,
            "n_empty": self.eigensystem.n_empty,
            "near_fermi_count": self.eigensystem.near_fermi_count,
            "nearest_fermi_distance": self.eigensystem.nearest_fermi_distance,
            "n_transitions": self.transitions.n_transitions,
            "min_transition_energy": float(np.min(self.transitions.delta_E)),
            "max_transition_energy": float(np.max(self.transitions.delta_E)),
            "circular_identity_residual": self.transitions.circular_identity_residual(),
        })
        return out


def compute_optical_spectra(
    model: Any,
    H: Any,
    mu: float,
    omega: Any,
    *,
    eta: float = 0.05,
    kernel: Kernel = "gaussian",
    **kwargs: Any,
) -> tuple[OpticalResponseCalculator, OpticalSpectra]:
    """Convenience wrapper matching the style of ``markers.compute_markers``."""
    calc = OpticalResponseCalculator.from_model(model, H, mu, **kwargs)
    return calc, calc.compute_spectra(omega, eta=eta, kernel=kernel)
