"""
disorder.py
===========

Finite-density defect realizations for setting 1 of the impurity-response
project.

The microscopic model owns the definition of a defect:
    QWZ     -> onsite mass defect on a lattice site
    Haldane -> flux/Haldane defect on a hexagon

This module only chooses *which allowed defect centers* are occupied.

The production default is fixed-count sampling:
    N_imp = round(rho * N_candidates),
followed by uniform sampling without replacement from region B.

For reproducibility, realization_id and base_seed deterministically generate
a derived seed.  A realization is then kept fixed while h is scanned.

Random sampling and Hamiltonian preparation live here; response markers do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


SamplingMode = Literal["fixed_count", "bernoulli"]


def _as_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _canonical_center_tuple(center: Any) -> tuple[int, ...]:
    if not isinstance(center, (tuple, list, np.ndarray)):
        raise TypeError(f"defect center must be tuple/list/array; received {center!r}")
    out = tuple(int(x) for x in center)
    if not out:
        raise ValueError("defect center cannot be empty")
    return out


@dataclass(frozen=True)
class DefectRealization:
    """
    One frozen impurity configuration.

    density is defined relative to the number of eligible defect centers in B.
    For QWZ these centers are sites; for Haldane they are hexagons.
    """

    centers: tuple[tuple[int, ...], ...]
    target_density: float
    n_candidates: int
    mode: SamplingMode
    base_seed: int
    realization_id: int
    derived_seed: int

    def __post_init__(self) -> None:
        canonical = tuple(_canonical_center_tuple(c) for c in self.centers)
        if len(set(canonical)) != len(canonical):
            raise ValueError("realization contains duplicate defect centers")
        object.__setattr__(self, "centers", canonical)

        rho = float(self.target_density)
        if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ValueError("target_density must lie in [0,1]")
        object.__setattr__(self, "target_density", rho)

        n_candidates = _as_nonnegative_int(self.n_candidates, "n_candidates")
        object.__setattr__(self, "n_candidates", n_candidates)

        if len(canonical) > n_candidates:
            raise ValueError("more selected defects than candidate centers")

        object.__setattr__(
            self, "realization_id",
            _as_nonnegative_int(self.realization_id, "realization_id")
        )
        object.__setattr__(
            self, "base_seed",
            _as_nonnegative_int(self.base_seed, "base_seed")
        )
        object.__setattr__(
            self, "derived_seed",
            _as_nonnegative_int(self.derived_seed, "derived_seed")
        )

        if self.mode not in ("fixed_count", "bernoulli"):
            raise ValueError(f"unknown sampling mode {self.mode!r}")

    @property
    def n_defects(self) -> int:
        return len(self.centers)

    @property
    def achieved_density(self) -> float:
        if self.n_candidates == 0:
            return 0.0
        return self.n_defects / self.n_candidates

    def positions(self, model: Any) -> np.ndarray:
        if not self.centers:
            return np.empty((0, 2), dtype=float)
        return np.asarray(
            [model.defect_center_position(c) for c in self.centers],
            dtype=float,
        )

    def validate_against(self, model: Any, regions: Any) -> bool:
        allowed = set(regions.eligible_defect_centers(model))
        return all(c in allowed for c in self.centers)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_density": self.target_density,
            "achieved_density": self.achieved_density,
            "n_candidates": self.n_candidates,
            "n_defects": self.n_defects,
            "base_seed": self.base_seed,
            "realization_id": self.realization_id,
            "derived_seed": self.derived_seed,
            "centers": self.centers,
        }


@dataclass
class PreparedRealization:
    """
    Cache H0 and W for one realization so an h-scan only forms H0 + h W.
    """

    model: Any
    realization: DefectRealization
    H0: Any
    W: Any
    sparse: bool = False

    def hamiltonian(self, h: float):
        h = float(h)
        if not np.isfinite(h):
            raise ValueError("h must be finite")
        H = self.H0 + h * self.W
        if self.sparse and sp is not None and sp.issparse(H):
            return H.tocsr()
        return np.asarray(H, dtype=np.complex128)

    def compare_to_model(self, h: float) -> float:
        direct = self.model.hamiltonian(
            defects=self.realization.centers,
            h=float(h),
            sparse=self.sparse,
        )
        cached = self.hamiltonian(h)

        if (
            sp is not None
            and sp.issparse(direct)
            and sp.issparse(cached)
        ):
            diff = direct - cached
            return 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))

        return float(np.max(np.abs(np.asarray(direct) - np.asarray(cached))))


@dataclass(frozen=True)
class DefectSampler:
    """
    Reproducible finite-density defect sampler.

    Parameters
    ----------
    density:
        Fraction of eligible defect centers in B that are occupied.
    mode:
        "fixed_count" (production default) or "bernoulli".
    base_seed:
        Global seed.  realization_id is mixed into it with SeedSequence so
        each sample is reproducible independently of sampling order.
    """

    density: float
    mode: SamplingMode = "fixed_count"
    base_seed: int = 0

    def __post_init__(self) -> None:
        rho = float(self.density)
        if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ValueError("density must lie in [0,1]")
        object.__setattr__(self, "density", rho)

        if self.mode not in ("fixed_count", "bernoulli"):
            raise ValueError("mode must be 'fixed_count' or 'bernoulli'")

        object.__setattr__(
            self, "base_seed",
            _as_nonnegative_int(self.base_seed, "base_seed")
        )

    def _derived_seed(self, realization_id: int) -> int:
        rid = _as_nonnegative_int(realization_id, "realization_id")
        ss = np.random.SeedSequence([self.base_seed, rid])
        return int(ss.generate_state(1, dtype=np.uint64)[0])

    def sample(
        self,
        model: Any,
        regions: Any,
        realization_id: int = 0,
    ) -> DefectRealization:
        rid = _as_nonnegative_int(realization_id, "realization_id")

        candidates = tuple(regions.eligible_defect_centers(model))
        n_candidates = len(candidates)
        if n_candidates == 0:
            raise ValueError("region B contains no eligible defect centers")

        seed = self._derived_seed(rid)
        rng = np.random.default_rng(seed)

        if self.mode == "fixed_count":
            n_imp = int(np.rint(self.density * n_candidates))
            if n_imp == 0:
                chosen_indices = np.empty(0, dtype=int)
            elif n_imp == n_candidates:
                chosen_indices = np.arange(n_candidates, dtype=int)
            else:
                chosen_indices = rng.choice(
                    n_candidates,
                    size=n_imp,
                    replace=False,
                )
        else:
            chosen_indices = np.flatnonzero(
                rng.random(n_candidates) < self.density
            )

        # Sorting makes the realization representation independent of the
        # arbitrary ordering returned by rng.choice.
        chosen_indices = np.sort(np.asarray(chosen_indices, dtype=int))
        centers = tuple(
            _canonical_center_tuple(candidates[i])
            for i in chosen_indices
        )

        return DefectRealization(
            centers=centers,
            target_density=self.density,
            n_candidates=n_candidates,
            mode=self.mode,
            base_seed=self.base_seed,
            realization_id=rid,
            derived_seed=seed,
        )

    def sample_many(
        self,
        model: Any,
        regions: Any,
        n_realizations: int,
        start_id: int = 0,
    ) -> tuple[DefectRealization, ...]:
        n = _as_nonnegative_int(n_realizations, "n_realizations")
        start = _as_nonnegative_int(start_id, "start_id")
        return tuple(
            self.sample(model, regions, realization_id=start + i)
            for i in range(n)
        )

    def prepare(
        self,
        model: Any,
        realization: DefectRealization,
        sparse: bool = False,
    ) -> PreparedRealization:
        # The microscopic model is the authoritative validator of center IDs.
        model.canonicalize_defects(realization.centers)

        H0 = model.clean_hamiltonian(sparse=sparse)
        W = model.defect_operator(realization.centers, sparse=sparse)
        return PreparedRealization(
            model=model,
            realization=realization,
            H0=H0,
            W=W,
            sparse=bool(sparse),
        )


def prepare_realization(
    model: Any,
    realization: DefectRealization,
    sparse: bool = False,
) -> PreparedRealization:
    """
    Public convenience function independent of a DefectSampler instance.
    """
    model.canonicalize_defects(realization.centers)
    H0 = model.clean_hamiltonian(sparse=sparse)
    W = model.defect_operator(realization.centers, sparse=sparse)
    return PreparedRealization(
        model=model,
        realization=realization,
        H0=H0,
        W=W,
        sparse=bool(sparse),
    )
