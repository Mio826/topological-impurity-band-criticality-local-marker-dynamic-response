"""
geometry.py
===========

Physical-space A ⊃ B ⊃ C geometry for the finite-density impurity project.

A : the full open finite sample.
B : the interior region in which defect centers are allowed.
C : the still-deeper region used to average real-space observables.

The implementation is lattice-independent.  It constructs the convex hull of
the model's physical basis positions and defines B and C by Euclidean
clearance from the physical boundary:

    B : d(r, ∂A) >= impurity_margin
    C : d(r, ∂A) >= impurity_margin + measurement_margin

This works for both the square QWZ sample and the oblique honeycomb sample.
It is intentionally independent of disorder sampling and observables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from scipy.spatial import ConvexHull
except ImportError as exc:  # pragma: no cover
    raise ImportError("geometry.py requires scipy.spatial.ConvexHull") from exc


ArrayF = npt.NDArray[np.float64]
ArrayB = npt.NDArray[np.bool_]


def _as_points(points: Any) -> ArrayF:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2); received {arr.shape}")
    if len(arr) == 0:
        raise ValueError("points cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points must be finite")
    return arr


def _model_is_periodic(model: Any) -> bool:
    px = bool(getattr(model, "pbc_x", False))
    py = bool(getattr(model, "pbc_y", False))
    return px or py


@dataclass
class NestedRegions:
    """
    Lattice-independent nested physical regions A ⊃ B ⊃ C.

    Parameters
    ----------
    boundary_points:
        Physical points whose convex hull defines the open sample A.
        Usually this is ``model.positions()``.
    impurity_margin:
        Physical distance from ∂A required for a point to belong to B.
    measurement_margin:
        Additional physical distance from B to the measurement region C.
        Thus the total A->C clearance is
        ``impurity_margin + measurement_margin``.
    tolerance:
        Numerical tolerance for hull-membership comparisons.
    """

    boundary_points: ArrayF
    impurity_margin: float
    measurement_margin: float
    tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        points = _as_points(self.boundary_points)

        self.impurity_margin = float(self.impurity_margin)
        self.measurement_margin = float(self.measurement_margin)
        self.tolerance = float(self.tolerance)

        for name, value in (
            ("impurity_margin", self.impurity_margin),
            ("measurement_margin", self.measurement_margin),
            ("tolerance", self.tolerance),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.impurity_margin < 0.0:
            raise ValueError("impurity_margin must be nonnegative")
        if self.measurement_margin < 0.0:
            raise ValueError("measurement_margin must be nonnegative")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")

        # Remove duplicated basis coordinates (e.g. the two QWZ orbitals).
        unique = np.unique(points, axis=0)
        if len(unique) < 3:
            raise ValueError("at least three distinct points are required")

        hull = ConvexHull(unique)

        # For scipy ConvexHull in 2D, each row is
        #     n_x x + n_y y + b <= 0
        # for points inside the hull.
        eq = np.asarray(hull.equations, dtype=float)
        normals = eq[:, :2]
        offsets = eq[:, 2]
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms == 0):
            raise RuntimeError("degenerate convex-hull facet")

        self._points = points
        self._unique_points = unique
        self._hull = hull
        self._facet_normals = normals
        self._facet_offsets = offsets
        self._facet_norms = norms

    @classmethod
    def from_model(
        cls,
        model: Any,
        impurity_margin: float,
        measurement_margin: float,
        tolerance: float = 1.0e-10,
    ) -> "NestedRegions":
        """
        Construct A/B/C directly from a microscopic lattice model.

        The A/B/C construction is intended for an open finite sample.  A
        periodic model is therefore rejected to avoid silently treating an
        arbitrary coordinate cut as a physical edge.
        """
        if _model_is_periodic(model):
            raise ValueError("NestedRegions.from_model requires an open-boundary model")
        if not hasattr(model, "positions"):
            raise TypeError("model must provide positions()")
        return cls(
            boundary_points=np.asarray(model.positions(), dtype=float),
            impurity_margin=impurity_margin,
            measurement_margin=measurement_margin,
            tolerance=tolerance,
        )

    @property
    def measurement_clearance(self) -> float:
        return self.impurity_margin + self.measurement_margin

    @property
    def hull_vertices(self) -> ArrayF:
        """Convex-hull vertices in cyclic order."""
        return self._unique_points[self._hull.vertices].copy()

    @property
    def area_A(self) -> float:
        """Area enclosed by the convex hull."""
        return float(self._hull.volume)  # scipy calls 2D area "volume"

    def signed_clearance(self, points: Any) -> ArrayF:
        """
        Signed Euclidean clearance from the convex-hull boundary.

        Positive: inside A.
        Zero: on ∂A.
        Negative: outside A.

        For a convex polygon this equals the minimum signed distance to its
        supporting boundary lines.
        """
        pts = _as_points(points)
        values = (
            pts @ self._facet_normals.T
            + self._facet_offsets[None, :]
        )
        distances = -values / self._facet_norms[None, :]
        return np.min(distances, axis=1)

    def mask_A(self, points: Any) -> ArrayB:
        return self.signed_clearance(points) >= -self.tolerance

    def mask_B(self, points: Any) -> ArrayB:
        return (
            self.signed_clearance(points)
            >= self.impurity_margin - self.tolerance
        )

    def mask_C(self, points: Any) -> ArrayB:
        return (
            self.signed_clearance(points)
            >= self.measurement_clearance - self.tolerance
        )

    def region_masks(self, points: Any) -> dict[str, ArrayB]:
        """Return A/B/C masks for any physical points."""
        pts = _as_points(points)
        a = self.mask_A(pts)
        b = self.mask_B(pts)
        c = self.mask_C(pts)
        return {"A": a, "B": b, "C": c}

    def basis_masks(self, model: Any) -> dict[str, ArrayB]:
        """A/B/C masks on the model's single-particle basis positions."""
        return self.region_masks(model.positions())

    def measurement_basis_mask(self, model: Any) -> ArrayB:
        """Boolean mask selecting basis states in C."""
        return self.mask_C(model.positions())

    def defect_centers_with_positions(
        self,
        model: Any,
    ) -> tuple[tuple[Any, ...], ArrayF]:
        """
        Return all model-defined defect centers and their physical positions.
        """
        if not hasattr(model, "available_defect_centers"):
            raise TypeError("model must provide available_defect_centers()")
        if not hasattr(model, "defect_center_position"):
            raise TypeError("model must provide defect_center_position(center)")

        centers = tuple(model.available_defect_centers())
        if not centers:
            return (), np.empty((0, 2), dtype=float)

        positions = np.asarray(
            [model.defect_center_position(c) for c in centers],
            dtype=float,
        )
        return centers, positions

    def eligible_defect_centers(self, model: Any) -> tuple[Any, ...]:
        """Model defect centers whose physical centers lie in B."""
        centers, pos = self.defect_centers_with_positions(model)
        if not centers:
            return ()
        mask = self.mask_B(pos)
        return tuple(c for c, keep in zip(centers, mask) if keep)

    def eligible_defect_positions(self, model: Any) -> ArrayF:
        """Physical positions corresponding to eligible defect centers in B."""
        centers = self.eligible_defect_centers(model)
        if not centers:
            return np.empty((0, 2), dtype=float)
        return np.asarray(
            [model.defect_center_position(c) for c in centers],
            dtype=float,
        )

    def validate_for_model(self, model: Any) -> dict[str, int | float | bool]:
        """
        Lightweight geometry diagnostics for a model.
        """
        basis_pos = np.asarray(model.positions(), dtype=float)
        masks = self.region_masks(basis_pos)
        eligible = self.eligible_defect_centers(model)

        nested_BA = bool(np.all(~masks["B"] | masks["A"]))
        nested_CB = bool(np.all(~masks["C"] | masks["B"]))

        return {
            "n_basis_A": int(np.sum(masks["A"])),
            "n_basis_B": int(np.sum(masks["B"])),
            "n_basis_C": int(np.sum(masks["C"])),
            "n_eligible_defects_B": len(eligible),
            "B_subset_A": nested_BA,
            "C_subset_B": nested_CB,
            "C_nonempty": bool(np.any(masks["C"])),
            "hull_area": self.area_A,
        }
