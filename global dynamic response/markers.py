"""
markers.py
==========

Real-space projector markers for the finite-density impurity project.

This module implements three observables for a finite 2D tight-binding system:

1. Bianco--Resta local Chern marker
       C(r) = 4 pi Im <r| Q x P y Q |r>.

2. Bianco--Resta local orbital magnetization
       M(r) = (e / hbar c) Im <r|
                P x Q H Q y P
              - Q x P (H - 2 mu) P y Q
              |r>.

3. Resta frequency-integrated magnetic circular dichroism (MCD) density
       X_xy(r) = (pi e^2 / 2 hbar^2)
                 Im <r| (H-mu) [[x,P],[y,P]] |r>.

For lattice calculations, basis-resolved diagonal contributions are divided by
the primitive-cell area.  Summing all basis contributions belonging to one
unit cell gives a cell-resolved intensive marker.  Bulk observables are obtained
by averaging those cell-resolved markers over cells entirely contained in the
measurement region C.

References
----------
R. Bianco and R. Resta, Phys. Rev. B 84, 241106(R) (2011).
R. Bianco and R. Resta, Phys. Rev. Lett. 110, 087202 (2013).
R. Resta, Phys. Rev. Research 2, 023139 (2020).

Conventions
-----------
* x and y are physical Cartesian coordinates.
* e is the positive elementary-charge magnitude used in the papers.
* Defaults use lattice/natural units e = hbar = c = 1.
* The formulas require open-boundary position operators.  PBC marker variants
  are a separate problem and are intentionally not implemented here.
* P = Theta(mu-H) is implemented as all eigenstates with E < mu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


ArrayF = npt.NDArray[np.float64]
ArrayC = npt.NDArray[np.complex128]
ArrayB = npt.NDArray[np.bool_]


def _as_dense_hermitian(H: Any, atol: float = 1.0e-10) -> ArrayC:
    if sp is not None and sp.issparse(H):
        A = H.toarray().astype(np.complex128, copy=False)
    else:
        A = np.asarray(H, dtype=np.complex128)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"H must be square; received shape {A.shape}")

    residual = float(np.max(np.abs(A - A.conj().T))) if A.size else 0.0
    if residual > atol:
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


def _max_abs(A: Any) -> float:
    a = np.asarray(A)
    return 0.0 if a.size == 0 else float(np.max(np.abs(a)))


def _diag_product(A: ArrayC, B: ArrayC) -> ArrayC:
    """diag(A @ B) without materializing the full product."""
    return np.einsum("ij,ji->i", A, B, optimize=True)


@dataclass(frozen=True)
class ProjectorData:
    """Zero-temperature occupied projector and numerical diagnostics."""

    eigenvalues: ArrayF
    eigenvectors: ArrayC
    occupied: ArrayB
    P: ArrayC
    Q: ArrayC
    mu: float
    near_fermi_count: int
    nearest_fermi_distance: float

    @property
    def n_occupied(self) -> int:
        return int(np.sum(self.occupied))

    def diagnostics(self, H: ArrayC) -> dict[str, float | int | bool]:
        I = np.eye(H.shape[0], dtype=np.complex128)
        p2 = _max_abs(self.P @ self.P - self.P)
        q2 = _max_abs(self.Q @ self.Q - self.Q)
        pq = _max_abs(self.P @ self.Q)
        hp = _max_abs(H @ self.P - self.P @ H)
        herm_p = _max_abs(self.P - self.P.conj().T)
        complement = _max_abs(self.P + self.Q - I)

        return {
            "n_occupied": self.n_occupied,
            "near_fermi_count": self.near_fermi_count,
            "nearest_fermi_distance": self.nearest_fermi_distance,
            "P_idempotency_residual": p2,
            "Q_idempotency_residual": q2,
            "PQ_orthogonality_residual": pq,
            "HP_commutator_residual": hp,
            "P_hermiticity_residual": herm_p,
            "P_plus_Q_residual": complement,
        }


@dataclass(frozen=True)
class MarkerFields:
    """
    Basis- and cell-resolved real-space markers.

    orbital_lc:
        First Bianco--Resta term, P x Q H Q y P.
    orbital_ic:
        The mu-independent second term, -Q x P H P y Q.
    orbital_chern:
        Explicit chemical-potential/Chern contribution,
        +2 mu Q x P y Q.
    orbital:
        orbital_lc + orbital_ic + orbital_chern.
    """

    chern_basis: ArrayF
    orbital_basis: ArrayF
    orbital_lc_basis: ArrayF
    orbital_ic_basis: ArrayF
    orbital_chern_basis: ArrayF
    mcd_basis: ArrayF

    chern_cell: ArrayF
    orbital_cell: ArrayF
    orbital_lc_cell: ArrayF
    orbital_ic_cell: ArrayF
    orbital_chern_cell: ArrayF
    mcd_cell: ArrayF

    mu_used: float


class MarkerCalculator:
    """
    Shared projector backend for Chern marker, orbital magnetization, and MCD.

    Use ``MarkerCalculator.from_model(model, H, mu)`` for the QWZ/Haldane
    models in this project.
    """

    def __init__(
        self,
        H: Any,
        positions: Any,
        mu: float,
        cell_area: float,
        cell_ids: Any,
        *,
        e_charge: float = 1.0,
        hbar: float = 1.0,
        c_light: float = 1.0,
        fermi_tolerance: float = 1.0e-10,
        hermiticity_tolerance: float = 1.0e-10,
    ) -> None:
        self.H = _as_dense_hermitian(H, atol=hermiticity_tolerance)
        self.dim = self.H.shape[0]
        self.positions = _as_positions(positions, self.dim)

        self.mu = float(mu)
        self.cell_area = float(cell_area)
        self.e_charge = float(e_charge)
        self.hbar = float(hbar)
        self.c_light = float(c_light)
        self.fermi_tolerance = float(fermi_tolerance)

        for name, value in (
            ("mu", self.mu),
            ("cell_area", self.cell_area),
            ("e_charge", self.e_charge),
            ("hbar", self.hbar),
            ("c_light", self.c_light),
            ("fermi_tolerance", self.fermi_tolerance),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.cell_area <= 0:
            raise ValueError("cell_area must be positive")
        if self.hbar <= 0:
            raise ValueError("hbar must be positive")
        if self.c_light <= 0:
            raise ValueError("c_light must be positive")
        if self.fermi_tolerance < 0:
            raise ValueError("fermi_tolerance must be nonnegative")

        ids = np.asarray(cell_ids)
        if ids.shape != (self.dim,):
            raise ValueError(
                f"cell_ids must have shape ({self.dim},); received {ids.shape}"
            )
        if not np.issubdtype(ids.dtype, np.integer):
            if np.all(np.equal(ids, np.floor(ids))):
                ids = ids.astype(int)
            else:
                raise TypeError("cell_ids must be integer-valued")
        ids = ids.astype(int)
        if np.any(ids < 0):
            raise ValueError("cell_ids must be nonnegative")

        unique = np.unique(ids)
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError("cell_ids must be contiguous integers starting at 0")

        self.cell_ids = ids
        self.n_cells = len(unique)
        self.projector = self._build_projector()

    @classmethod
    def from_model(
        cls,
        model: Any,
        H: Any,
        mu: float,
        **kwargs: Any,
    ) -> "MarkerCalculator":
        """
        Construct from the validated QWZ/Haldane model interface.

        Both current model files use cell-major ordering:
            basis_index = n_internal * cell_index + internal_index.
        """
        if bool(getattr(model, "pbc_x", False)) or bool(
            getattr(model, "pbc_y", False)
        ):
            raise ValueError(
                "Real-space position markers here require an open-boundary model"
            )

        dim = int(model.dim)
        n_cells = int(model.Lx) * int(model.Ly)
        if dim % n_cells != 0:
            raise ValueError("model dimension is incompatible with Lx*Ly cells")

        n_internal = dim // n_cells
        cell_ids = np.repeat(np.arange(n_cells, dtype=int), n_internal)

        return cls(
            H=H,
            positions=model.positions(),
            mu=mu,
            cell_area=float(model.cell_area),
            cell_ids=cell_ids,
            **kwargs,
        )

    def _build_projector(self) -> ProjectorData:
        evals, evecs = np.linalg.eigh(self.H)
        occupied = evals < self.mu

        if np.any(occupied):
            Uocc = evecs[:, occupied]
            P = Uocc @ Uocc.conj().T
        else:
            P = np.zeros_like(self.H)

        Q = np.eye(self.dim, dtype=np.complex128) - P

        distances = np.abs(evals - self.mu)
        nearest = float(np.min(distances)) if len(distances) else np.inf
        near_count = int(np.sum(distances <= self.fermi_tolerance))

        return ProjectorData(
            eigenvalues=np.asarray(evals, dtype=float),
            eigenvectors=np.asarray(evecs, dtype=np.complex128),
            occupied=np.asarray(occupied, dtype=bool),
            P=np.asarray(P, dtype=np.complex128),
            Q=np.asarray(Q, dtype=np.complex128),
            mu=self.mu,
            near_fermi_count=near_count,
            nearest_fermi_distance=nearest,
        )

    def projector_diagnostics(self) -> dict[str, float | int | bool]:
        return self.projector.diagnostics(self.H)

    def reduce_basis_to_cells(self, values: Any) -> ArrayF:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.dim,):
            raise ValueError(
                f"values must have shape ({self.dim},); received {values.shape}"
            )
        return np.bincount(
            self.cell_ids,
            weights=values,
            minlength=self.n_cells,
        ).astype(float)

    def cell_mask_from_basis_mask(
        self,
        basis_mask: Any,
        *,
        require: str = "all",
    ) -> ArrayB:
        """
        Convert a basis-state mask to a unit-cell mask.

        require="all" is the physically conservative choice for region C:
        every basis state in the unit cell must lie inside the measurement
        region.
        """
        mask = np.asarray(basis_mask, dtype=bool)
        if mask.shape != (self.dim,):
            raise ValueError(
                f"basis_mask must have shape ({self.dim},); received {mask.shape}"
            )
        if require not in ("all", "any"):
            raise ValueError("require must be 'all' or 'any'")

        counts_total = np.bincount(
            self.cell_ids, minlength=self.n_cells
        )
        counts_true = np.bincount(
            self.cell_ids,
            weights=mask.astype(int),
            minlength=self.n_cells,
        )

        if require == "all":
            return counts_true == counts_total
        return counts_true > 0

    def bulk_average(
        self,
        cell_values: Any,
        *,
        basis_mask: Optional[Any] = None,
        cell_mask: Optional[Any] = None,
        require: str = "all",
    ) -> float:
        """
        Average a cell-resolved marker over a selected bulk region.
        """
        values = np.asarray(cell_values, dtype=float)
        if values.shape != (self.n_cells,):
            raise ValueError(
                f"cell_values must have shape ({self.n_cells},)"
            )

        if basis_mask is not None and cell_mask is not None:
            raise ValueError("provide basis_mask or cell_mask, not both")

        if basis_mask is not None:
            mask = self.cell_mask_from_basis_mask(
                basis_mask, require=require
            )
        elif cell_mask is not None:
            mask = np.asarray(cell_mask, dtype=bool)
            if mask.shape != (self.n_cells,):
                raise ValueError(
                    f"cell_mask must have shape ({self.n_cells},)"
                )
        else:
            mask = np.ones(self.n_cells, dtype=bool)

        if not np.any(mask):
            raise ValueError("selected bulk region contains no complete cells")

        return float(np.mean(values[mask]))

    def compute_fields(
        self,
        *,
        mu_override: Optional[float] = None,
    ) -> MarkerFields:
        """
        Compute all three real-space markers using the already-built projector.

        ``mu_override`` changes only the explicit mu appearing in OM/MCD,
        while keeping P fixed.  This is physically valid only if no eigenvalue
        is crossed between the original mu and the override.  It is useful for
        exact Streda/Chern consistency checks.
        """
        mu = self.mu if mu_override is None else float(mu_override)
        if not np.isfinite(mu):
            raise ValueError("mu_override must be finite")

        P = self.projector.P
        Q = self.projector.Q
        H = self.H

        x = self.positions[:, 0]
        y = self.positions[:, 1]

        # Efficient diagonal-position multiplications.
        XP = x[:, None] * P
        XQ = x[:, None] * Q
        YP = y[:, None] * P
        YQ = y[:, None] * Q

        QXP = Q @ XP
        PXQ = P @ XQ
        QYP = Q @ YP
        PYQ = P @ YQ

        # --------------------------------------------------------------
        # 1. Bianco--Resta local Chern marker
        # C_i = 4pi/Acell Im [Q x P y Q]_{ii}
        # --------------------------------------------------------------
        chern_core = np.imag(_diag_product(QXP, YQ))
        chern_basis = (4.0 * np.pi / self.cell_area) * chern_core

        # --------------------------------------------------------------
        # 2. Bianco--Resta local orbital magnetization
        #
        # M = e/(hbar c) Im [
        #       P x Q H Q y P
        #     - Q x P (H-2mu) P y Q
        # ]
        #
        # We expose the second term as
        #   IC      = - Q x P H P y Q
        #   Chernmu = + 2mu Q x P y Q.
        # --------------------------------------------------------------
        pref_m = self.e_charge / (
            self.hbar * self.c_light * self.cell_area
        )

        lc_core = np.imag(
            _diag_product(PXQ, H @ QYP)
        )
        ic_core = -np.imag(
            _diag_product(QXP, H @ PYQ)
        )
        chern_mu_core = 2.0 * mu * chern_core

        orbital_lc_basis = pref_m * lc_core
        orbital_ic_basis = pref_m * ic_core
        orbital_chern_basis = pref_m * chern_mu_core
        orbital_basis = (
            orbital_lc_basis
            + orbital_ic_basis
            + orbital_chern_basis
        )

        # --------------------------------------------------------------
        # 3. Resta MCD sum-rule density
        # X_xy = pi e^2/(2 hbar^2 Acell)
        #        Im diag[(H-mu) [[x,P],[y,P]]]
        #
        # [x,P]_ij = (x_i-x_j) P_ij.
        # --------------------------------------------------------------
        comm_x = (x[:, None] - x[None, :]) * P
        comm_y = (y[:, None] - y[None, :]) * P
        double_comm = comm_x @ comm_y - comm_y @ comm_x

        H_minus_mu = H - mu * np.eye(self.dim, dtype=np.complex128)
        mcd_core = np.imag(
            _diag_product(H_minus_mu, double_comm)
        )

        pref_i = (
            np.pi
            * self.e_charge**2
            / (2.0 * self.hbar**2 * self.cell_area)
        )
        mcd_basis = pref_i * mcd_core

        return MarkerFields(
            chern_basis=np.asarray(chern_basis, dtype=float),
            orbital_basis=np.asarray(orbital_basis, dtype=float),
            orbital_lc_basis=np.asarray(orbital_lc_basis, dtype=float),
            orbital_ic_basis=np.asarray(orbital_ic_basis, dtype=float),
            orbital_chern_basis=np.asarray(orbital_chern_basis, dtype=float),
            mcd_basis=np.asarray(mcd_basis, dtype=float),

            chern_cell=self.reduce_basis_to_cells(chern_basis),
            orbital_cell=self.reduce_basis_to_cells(orbital_basis),
            orbital_lc_cell=self.reduce_basis_to_cells(orbital_lc_basis),
            orbital_ic_cell=self.reduce_basis_to_cells(orbital_ic_basis),
            orbital_chern_cell=self.reduce_basis_to_cells(
                orbital_chern_basis
            ),
            mcd_cell=self.reduce_basis_to_cells(mcd_basis),

            mu_used=mu,
        )


def compute_markers(
    model: Any,
    H: Any,
    mu: float,
    **kwargs: Any,
) -> tuple[MarkerCalculator, MarkerFields]:
    """
    Convenience wrapper for the common project workflow.
    """
    calculator = MarkerCalculator.from_model(
        model=model,
        H=H,
        mu=mu,
        **kwargs,
    )
    return calculator, calculator.compute_fields()
