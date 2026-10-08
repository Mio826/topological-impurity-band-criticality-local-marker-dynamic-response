"""
qwz.py
======

Qi--Wu--Zhang (QWZ) lattice model for the topological-impurity project.

Bloch Hamiltonian:
    H0(k) =
        A sin(kx) sigma_x
      + A sin(ky) sigma_y
      + [u + t cos(kx) + t cos(ky)] sigma_z.

A local defect at square-lattice site R is
    V_R = -h sigma_z.

For a fixed defect configuration D,
    H(h) = H0 + h W_D,
where W_D is a unit-strength defect operator independent of h.

Random sampling belongs in disorder.py; observables belong in markers.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, Optional, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


Site: TypeAlias = tuple[int, int]
BoundaryCondition: TypeAlias = bool | tuple[bool, bool]
DenseMatrix: TypeAlias = npt.NDArray[np.complex128]

SIGMA_0: DenseMatrix = np.eye(2, dtype=np.complex128)
SIGMA_X: DenseMatrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y: DenseMatrix = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z: DenseMatrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _coerce_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; received {value!r}")
    return int(value)


def _empty_matrix(dim: int, sparse: bool):
    if sparse:
        if sp is None:
            raise ImportError("scipy is required when sparse=True")
        return sp.lil_matrix((dim, dim), dtype=np.complex128)
    return np.zeros((dim, dim), dtype=np.complex128)


def _finalize(matrix, sparse: bool):
    return matrix.tocsr() if sparse else np.asarray(matrix, dtype=np.complex128)


def _max_abs(matrix) -> float:
    if sp is not None and sp.issparse(matrix):
        return 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))
    a = np.asarray(matrix)
    return 0.0 if a.size == 0 else float(np.max(np.abs(a)))


@dataclass
class QWZModel:
    """Finite square-lattice QWZ model with two orbitals per site."""

    Lx: int
    Ly: int
    t: float = 1.0
    u: float = -2.05
    A: float = 1.5
    pbc: BoundaryCondition = False

    norb: ClassVar[int] = 2
    cell_area: ClassVar[float] = 1.0

    def __post_init__(self) -> None:
        self.Lx = _coerce_int(self.Lx, "Lx")
        self.Ly = _coerce_int(self.Ly, "Ly")
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx and Ly must be positive")
        self.t = float(self.t)
        self.u = float(self.u)
        self.A = float(self.A)
        if not all(np.isfinite(v) for v in (self.t, self.u, self.A)):
            raise ValueError("t, u, and A must be finite")
        _ = self.boundary_conditions

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly

    @property
    def dim(self) -> int:
        return self.norb * self.n_sites

    @property
    def boundary_conditions(self) -> tuple[bool, bool]:
        if isinstance(self.pbc, (bool, np.bool_)):
            p = bool(self.pbc)
            return p, p
        if not isinstance(self.pbc, tuple) or len(self.pbc) != 2:
            raise TypeError("pbc must be bool or (pbc_x, pbc_y)")
        px, py = self.pbc
        if not isinstance(px, (bool, np.bool_)) or not isinstance(py, (bool, np.bool_)):
            raise TypeError("pbc tuple entries must be booleans")
        return bool(px), bool(py)

    @property
    def pbc_x(self) -> bool:
        return self.boundary_conditions[0]

    @property
    def pbc_y(self) -> bool:
        return self.boundary_conditions[1]

    def _normalize_axis(self, value: int, L: int, periodic: bool, name: str) -> int:
        value = _coerce_int(value, name)
        if periodic:
            return value % L
        if not 0 <= value < L:
            raise IndexError(f"{name}={value} outside [0,{L})")
        return value

    def normalize_site(self, x: int, y: int) -> Site:
        return (
            self._normalize_axis(x, self.Lx, self.pbc_x, "x"),
            self._normalize_axis(y, self.Ly, self.pbc_y, "y"),
        )

    def _neighbor(self, x: int, y: int, dx: int, dy: int) -> Optional[Site]:
        xn, yn = x + dx, y + dy
        if self.pbc_x:
            xn %= self.Lx
        elif not 0 <= xn < self.Lx:
            return None
        if self.pbc_y:
            yn %= self.Ly
        elif not 0 <= yn < self.Ly:
            return None
        return int(xn), int(yn)

    def site_index(self, x: int, y: int) -> int:
        x, y = self.normalize_site(x, y)
        return x + self.Lx * y

    def idx(self, x: int, y: int, orbital: int) -> int:
        orbital = _coerce_int(orbital, "orbital")
        if orbital not in (0, 1):
            raise ValueError("orbital must be 0 or 1")
        return 2 * self.site_index(x, y) + orbital

    basis_index = idx

    def site_positions(self) -> npt.NDArray[np.float64]:
        pos = np.empty((self.n_sites, 2), dtype=float)
        for y in range(self.Ly):
            for x in range(self.Lx):
                pos[self.site_index(x, y)] = (float(x), float(y))
        return pos

    def positions(self) -> npt.NDArray[np.float64]:
        return np.repeat(self.site_positions(), self.norb, axis=0)

    def available_defect_centers(self) -> tuple[Site, ...]:
        return tuple((x, y) for y in range(self.Ly) for x in range(self.Lx))

    def defect_center_position(self, center: Site) -> npt.NDArray[np.float64]:
        x, y = self.normalize_site(*center)
        return np.array([float(x), float(y)])

    @property
    def onsite_block(self) -> DenseMatrix:
        return self.u * SIGMA_Z

    @property
    def hopping_x(self) -> DenseMatrix:
        return 0.5 * (self.t * SIGMA_Z - 1.0j * self.A * SIGMA_X)

    @property
    def hopping_y(self) -> DenseMatrix:
        return 0.5 * (self.t * SIGMA_Z - 1.0j * self.A * SIGMA_Y)

    def bloch_hamiltonian(self, kx: float, ky: float) -> DenseMatrix:
        return np.asarray(
            self.A * np.sin(kx) * SIGMA_X
            + self.A * np.sin(ky) * SIGMA_Y
            + (self.u + self.t * np.cos(kx) + self.t * np.cos(ky)) * SIGMA_Z,
            dtype=np.complex128,
        )

    def dispersion(self, kx: float, ky: float) -> npt.NDArray[np.float64]:
        return np.linalg.eigvalsh(self.bloch_hamiltonian(kx, ky))

    def _add_onsite(self, matrix, x: int, y: int, block: DenseMatrix) -> None:
        for a in range(2):
            ia = self.idx(x, y, a)
            for b in range(2):
                ib = self.idx(x, y, b)
                matrix[ia, ib] += block[a, b]

    def _add_hopping(self, matrix, source: Site, target: Site, block: DenseMatrix) -> None:
        x1, y1 = source
        x2, y2 = target
        for a in range(2):
            i = self.idx(x1, y1, a)
            for b in range(2):
                j = self.idx(x2, y2, b)
                matrix[i, j] += block[a, b]
                matrix[j, i] += np.conjugate(block[a, b])

    def clean_hamiltonian(self, sparse: bool = False):
        H = _empty_matrix(self.dim, sparse)
        for y in range(self.Ly):
            for x in range(self.Lx):
                self._add_onsite(H, x, y, self.onsite_block)
                xp = self._neighbor(x, y, +1, 0)
                yp = self._neighbor(x, y, 0, +1)
                if xp is not None:
                    self._add_hopping(H, (x, y), xp, self.hopping_x)
                if yp is not None:
                    self._add_hopping(H, (x, y), yp, self.hopping_y)
        return _finalize(H, sparse)

    def canonicalize_defects(
        self, defects: Optional[Iterable[Site]]
    ) -> tuple[Site, ...]:
        if defects is None:
            return ()
        out: list[Site] = []
        seen: set[Site] = set()
        for d in defects:
            if not isinstance(d, (tuple, list, np.ndarray)) or len(d) != 2:
                raise TypeError(f"QWZ defect must be site (x,y); received {d!r}")
            site = self.normalize_site(int(d[0]), int(d[1]))
            if site in seen:
                raise ValueError(f"duplicate defect site: {site}")
            seen.add(site)
            out.append(site)
        return tuple(out)

    def defect_mask(self, defects: Optional[Iterable[Site]]) -> npt.NDArray[np.bool_]:
        mask = np.zeros((self.Ly, self.Lx), dtype=bool)
        for x, y in self.canonicalize_defects(defects):
            mask[y, x] = True
        return mask

    def defect_operator(
        self,
        defects: Optional[Iterable[Site]],
        sparse: bool = False,
    ):
        """Unit-strength W_D such that H(h)=H0+hW_D and V_R=-h sigma_z."""
        W = _empty_matrix(self.dim, sparse)
        for x, y in self.canonicalize_defects(defects):
            self._add_onsite(W, x, y, -SIGMA_Z)
        return _finalize(W, sparse)

    def hamiltonian(
        self,
        defects: Optional[Iterable[Site]] = None,
        h: float = 0.0,
        sparse: bool = False,
    ):
        h = float(h)
        if not np.isfinite(h):
            raise ValueError("h must be finite")
        H0 = self.clean_hamiltonian(sparse=sparse)
        ds = self.canonicalize_defects(defects)
        if not ds or h == 0.0:
            return H0
        W = self.defect_operator(ds, sparse=sparse)
        H = H0 + h * W
        return H.tocsr() if sparse else np.asarray(H, dtype=np.complex128)

    def onsite_mass_profile(
        self,
        defects: Optional[Iterable[Site]],
        h: float,
    ) -> npt.NDArray[np.float64]:
        mass = np.full((self.Ly, self.Lx), self.u, dtype=float)
        mass[self.defect_mask(defects)] -= float(h)
        return mass

    @staticmethod
    def hermiticity_residual(matrix) -> float:
        if sp is not None and sp.issparse(matrix):
            return _max_abs(matrix - matrix.getH())
        a = np.asarray(matrix)
        return _max_abs(a - a.conj().T)

    def clean_spectrum_from_bloch_grid(self) -> npt.NDArray[np.float64]:
        if not (self.pbc_x and self.pbc_y):
            raise ValueError("Bloch-grid comparison requires PBC in both directions")
        energies = []
        for nx in range(self.Lx):
            kx = 2.0 * np.pi * nx / self.Lx
            for ny in range(self.Ly):
                ky = 2.0 * np.pi * ny / self.Ly
                energies.extend(self.dispersion(kx, ky))
        return np.sort(np.asarray(energies, dtype=float))

    def realspace_bloch_spectrum_residual(self) -> float:
        er = np.sort(np.linalg.eigvalsh(self.clean_hamiltonian(sparse=False)))
        eb = self.clean_spectrum_from_bloch_grid()
        return float(np.max(np.abs(er - eb)))

    def run_sanity_checks(
        self,
        defects: Optional[Sequence[Site]] = None,
        h: float = 0.0,
        atol: float = 1.0e-10,
    ) -> dict[str, float | bool]:
        H = self.hamiltonian(defects=defects, h=h, sparse=False)
        H0 = self.clean_hamiltonian(sparse=False)
        W = self.defect_operator(defects, sparse=False)

        out: dict[str, float | bool] = {
            "hermiticity_residual": self.hermiticity_residual(H),
            "is_hermitian": self.hermiticity_residual(H) <= atol,
            "linear_decomposition_residual": _max_abs(H - (H0 + float(h) * W)),
        }

        if self.pbc_x and self.pbc_y:
            r = self.realspace_bloch_spectrum_residual()
            out["realspace_bloch_spectrum_residual"] = r
            out["realspace_bloch_spectrum_matches"] = r <= atol

        if sp is not None:
            Hs = self.hamiltonian(defects=defects, h=h, sparse=True).toarray()
            out["dense_sparse_residual"] = _max_abs(H - Hs)
            out["dense_sparse_match"] = out["dense_sparse_residual"] <= atol

        return out
