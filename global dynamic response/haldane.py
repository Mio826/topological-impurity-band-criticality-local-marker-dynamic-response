"""
haldane.py
==========

Honeycomb Haldane model with hexagon defects following Seth & Kimchi,
"Topological Impurity Bands", arXiv:2609.09118.

The clean Bloch Hamiltonian is
    H(k) = [[m + t2 d_k, h_k],
            [h_k^*,      -m - t2 d_k]],

    d_k = -2 [sin k1 + sin k2 - sin(k1+k2)],
    h_k = -t [1 + exp(i k1) + exp(-i k2)].

Second-neighbor hoppings are oriented counterclockwise around each hexagon.
A defect on hexagon R is
    V_R = -i h sum_<<ij>> in R c_i^dagger c_j + H.c.,
so positive h locally reduces t2:
    t2 -> t2-h
on the six second-neighbor bonds of that hexagon.

For fixed defects D,
    H(h) = H0 + h W_D.

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


Cell: TypeAlias = tuple[int, int]
Site: TypeAlias = tuple[int, int, int]
Hexagon: TypeAlias = tuple[int, int]
BoundaryCondition: TypeAlias = bool | tuple[bool, bool]

A_SUB = 0
B_SUB = 1


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
class HaldaneModel:
    """Finite honeycomb Haldane model; Lx,Ly count Bravais unit cells."""

    Lx: int
    Ly: int
    t: float = 1.0
    t2: float = 0.1
    m: float = 0.0
    pbc: BoundaryCondition = False

    nsub: ClassVar[int] = 2

    # Mirrored standard honeycomb embedding; nearest-neighbor distance = 1.
    a1: ClassVar[npt.NDArray[np.float64]] = np.array(
        [-np.sqrt(3.0) / 2.0, -1.5], dtype=float
    )
    a2: ClassVar[npt.NDArray[np.float64]] = np.array(
        [-np.sqrt(3.0) / 2.0, +1.5], dtype=float
    )
    basis_A: ClassVar[npt.NDArray[np.float64]] = np.array([0.0, 0.0], dtype=float)
    basis_B: ClassVar[npt.NDArray[np.float64]] = np.array([0.0, 1.0], dtype=float)

    def __post_init__(self) -> None:
        self.Lx = _coerce_int(self.Lx, "Lx")
        self.Ly = _coerce_int(self.Ly, "Ly")
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx and Ly must be positive")
        self.t = float(self.t)
        self.t2 = float(self.t2)
        self.m = float(self.m)
        if not all(np.isfinite(v) for v in (self.t, self.t2, self.m)):
            raise ValueError("t, t2, and m must be finite")
        _ = self.boundary_conditions

    @property
    def n_cells(self) -> int:
        return self.Lx * self.Ly

    @property
    def n_sites(self) -> int:
        return 2 * self.n_cells

    @property
    def dim(self) -> int:
        return self.n_sites

    @property
    def cell_area(self) -> float:
        return float(abs(np.linalg.det(np.column_stack((self.a1, self.a2)))))

    @property
    def bravais_orientation(self) -> float:
        return float(np.sign(np.linalg.det(np.column_stack((self.a1, self.a2)))))

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

    def _canonical_cell(
        self, n1: int, n2: int, allow_none: bool = False
    ) -> Optional[Cell]:
        n1 = _coerce_int(n1, "n1")
        n2 = _coerce_int(n2, "n2")

        if self.pbc_x:
            n1 %= self.Lx
        elif not 0 <= n1 < self.Lx:
            if allow_none:
                return None
            raise IndexError(f"n1={n1} outside [0,{self.Lx})")

        if self.pbc_y:
            n2 %= self.Ly
        elif not 0 <= n2 < self.Ly:
            if allow_none:
                return None
            raise IndexError(f"n2={n2} outside [0,{self.Ly})")

        return int(n1), int(n2)

    def normalize_site(self, site: Site) -> Site:
        if not isinstance(site, (tuple, list, np.ndarray)) or len(site) != 3:
            raise TypeError("site must be (n1,n2,sublattice)")
        n1, n2, sub = site
        sub = _coerce_int(sub, "sublattice")
        if sub not in (A_SUB, B_SUB):
            raise ValueError("sublattice must be 0 (A) or 1 (B)")
        cell = self._canonical_cell(int(n1), int(n2), allow_none=False)
        assert cell is not None
        return cell[0], cell[1], sub

    def cell_index(self, n1: int, n2: int) -> int:
        cell = self._canonical_cell(n1, n2, allow_none=False)
        assert cell is not None
        return cell[0] + self.Lx * cell[1]

    def idx(self, n1: int, n2: int, sublattice: int) -> int:
        sublattice = _coerce_int(sublattice, "sublattice")
        if sublattice not in (A_SUB, B_SUB):
            raise ValueError("sublattice must be 0 or 1")
        return 2 * self.cell_index(n1, n2) + sublattice

    basis_index = idx

    def cell_position(self, n1: int, n2: int) -> npt.NDArray[np.float64]:
        return float(n1) * self.a1 + float(n2) * self.a2

    def site_position(self, site: Site) -> npt.NDArray[np.float64]:
        n1, n2, sub = self.normalize_site(site)
        return self.cell_position(n1, n2) + (
            self.basis_A if sub == A_SUB else self.basis_B
        )

    def positions(self) -> npt.NDArray[np.float64]:
        out = np.empty((self.dim, 2), dtype=float)
        for n2 in range(self.Ly):
            for n1 in range(self.Lx):
                out[self.idx(n1, n2, A_SUB)] = self.cell_position(n1, n2) + self.basis_A
                out[self.idx(n1, n2, B_SUB)] = self.cell_position(n1, n2) + self.basis_B
        return out

    def _shift_site(self, site: Site, dn1: int, dn2: int) -> Optional[Site]:
        n1, n2, sub = site
        cell = self._canonical_cell(n1 + dn1, n2 + dn2, allow_none=True)
        if cell is None:
            return None
        return cell[0], cell[1], sub

    def nearest_neighbor_bonds(self) -> tuple[tuple[Site, Site], ...]:
        """
        Each NN bond once, oriented A -> B.

        A_R connects to B_R, B_{R+a1}, B_{R-a2}, giving
        h_k=-t(1+e^{ik1}+e^{-ik2}).
        """
        bonds: list[tuple[Site, Site]] = []
        for n2 in range(self.Ly):
            for n1 in range(self.Lx):
                A = (n1, n2, A_SUB)
                for dn1, dn2 in ((0, 0), (+1, 0), (0, -1)):
                    cell = self._canonical_cell(n1 + dn1, n2 + dn2, allow_none=True)
                    if cell is not None:
                        bonds.append((A, (cell[0], cell[1], B_SUB)))
        return tuple(bonds)

    def clean_nnn_bonds_ccw(self) -> tuple[tuple[Site, Site], ...]:
        """
        Each NNN bond once, oriented counterclockwise around its hexagon.

        A directions: +a2, +a1, -(a1+a2)
        B directions: +(a1+a2), -a2, -a1
        """
        bonds: list[tuple[Site, Site]] = []
        A_dirs = ((0, +1), (+1, 0), (-1, -1))
        B_dirs = ((+1, +1), (0, -1), (-1, 0))
        for n2 in range(self.Ly):
            for n1 in range(self.Lx):
                for sub, dirs in ((A_SUB, A_dirs), (B_SUB, B_dirs)):
                    src = (n1, n2, sub)
                    for dn1, dn2 in dirs:
                        dst = self._shift_site(src, dn1, dn2)
                        if dst is not None:
                            bonds.append((src, dst))
        return tuple(bonds)

    def _raw_hexagon_vertices(self, R: Hexagon) -> tuple[Site, ...]:
        """
        Six vertices in geometric counterclockwise order:
          A_R, B_R, A_{R+a2}, B_{R+a1+a2}, A_{R+a1+a2}, B_{R+a1}.
        """
        n1, n2 = int(R[0]), int(R[1])
        return (
            (n1,     n2,     A_SUB),
            (n1,     n2,     B_SUB),
            (n1,     n2 + 1, A_SUB),
            (n1 + 1, n2 + 1, B_SUB),
            (n1 + 1, n2 + 1, A_SUB),
            (n1 + 1, n2,     B_SUB),
        )

    def hexagon_vertices(self, R: Hexagon) -> tuple[Site, ...]:
        out: list[Site] = []
        for n1, n2, sub in self._raw_hexagon_vertices(R):
            cell = self._canonical_cell(n1, n2, allow_none=True)
            if cell is None:
                raise ValueError(f"hexagon {R} is incomplete under OBC")
            out.append((cell[0], cell[1], sub))
        if len(set(out)) != 6:
            raise ValueError(f"hexagon {R} collapses in a too-small periodic system")
        return tuple(out)

    def hexagon_nnn_bonds(self, R: Hexagon) -> tuple[tuple[Site, Site], ...]:
        """Six NNN bonds of hexagon R, all oriented counterclockwise."""
        v = self.hexagon_vertices(R)
        return (
            (v[0], v[2]),
            (v[2], v[4]),
            (v[4], v[0]),
            (v[1], v[3]),
            (v[3], v[5]),
            (v[5], v[1]),
        )

    def _is_valid_hexagon(self, R: Hexagon) -> bool:
        try:
            self.hexagon_vertices(R)
            return True
        except ValueError:
            return False

    def available_defect_centers(self) -> tuple[Hexagon, ...]:
        out = []
        for n2 in range(self.Ly):
            for n1 in range(self.Lx):
                R = (n1, n2)
                if self._is_valid_hexagon(R):
                    out.append(R)
        return tuple(out)

    def hexagon_center_position(self, R: Hexagon) -> npt.NDArray[np.float64]:
        n1, n2 = int(R[0]), int(R[1])
        return self.cell_position(n1, n2) + np.array(
            [-np.sqrt(3.0) / 2.0, 0.5], dtype=float
        )

    defect_center_position = hexagon_center_position

    def canonicalize_defects(
        self, defects: Optional[Iterable[Hexagon]]
    ) -> tuple[Hexagon, ...]:
        if defects is None:
            return ()
        allowed = set(self.available_defect_centers())
        out: list[Hexagon] = []
        seen: set[Hexagon] = set()

        for d in defects:
            if not isinstance(d, (tuple, list, np.ndarray)) or len(d) != 2:
                raise TypeError(f"Haldane defect must be hexagon id (n1,n2); received {d!r}")
            n1, n2 = int(d[0]), int(d[1])
            if self.pbc_x:
                n1 %= self.Lx
            if self.pbc_y:
                n2 %= self.Ly
            R = (n1, n2)
            if R not in allowed:
                raise ValueError(f"hexagon {R} is not an allowed complete defect center")
            if R in seen:
                raise ValueError(f"duplicate hexagon defect: {R}")
            seen.add(R)
            out.append(R)
        return tuple(out)

    def d_k(self, k1: float, k2: float) -> float:
        return float(
            -2.0 * (np.sin(k1) + np.sin(k2) - np.sin(k1 + k2))
        )

    def h_k(self, k1: float, k2: float) -> complex:
        return complex(
            -self.t * (1.0 + np.exp(1.0j * k1) + np.exp(-1.0j * k2))
        )

    def bloch_hamiltonian(self, k1: float, k2: float) -> npt.NDArray[np.complex128]:
        d = self.d_k(k1, k2)
        hk = self.h_k(k1, k2)
        return np.array(
            [
                [self.m + self.t2 * d, hk],
                [np.conjugate(hk), -self.m - self.t2 * d],
            ],
            dtype=np.complex128,
        )

    def dispersion(self, k1: float, k2: float) -> npt.NDArray[np.float64]:
        return np.linalg.eigvalsh(self.bloch_hamiltonian(k1, k2))

    def _add_onsite(self, matrix, site: Site, value: complex) -> None:
        s = self.normalize_site(site)
        i = self.idx(*s)
        matrix[i, i] += value

    def _add_hopping(self, matrix, source: Site, target: Site, amplitude: complex) -> None:
        s = self.normalize_site(source)
        t = self.normalize_site(target)
        i = self.idx(*s)
        j = self.idx(*t)
        matrix[i, j] += amplitude
        matrix[j, i] += np.conjugate(amplitude)

    def clean_hamiltonian(self, sparse: bool = False):
        H = _empty_matrix(self.dim, sparse)

        for n2 in range(self.Ly):
            for n1 in range(self.Lx):
                self._add_onsite(H, (n1, n2, A_SUB), +self.m)
                self._add_onsite(H, (n1, n2, B_SUB), -self.m)

        for s, t in self.nearest_neighbor_bonds():
            self._add_hopping(H, s, t, -self.t)

        for s, t in self.clean_nnn_bonds_ccw():
            self._add_hopping(H, s, t, 1.0j * self.t2)

        return _finalize(H, sparse)

    def defect_operator(
        self,
        defects: Optional[Iterable[Hexagon]],
        sparse: bool = False,
    ):
        """
        Unit-strength W_D such that H(h)=H0+hW_D.

        Each hexagon contributes -i on its six CCW NNN bonds + H.c.
        """
        W = _empty_matrix(self.dim, sparse)
        for R in self.canonicalize_defects(defects):
            for s, t in self.hexagon_nnn_bonds(R):
                self._add_hopping(W, s, t, -1.0j)
        return _finalize(W, sparse)

    def hamiltonian(
        self,
        defects: Optional[Iterable[Hexagon]] = None,
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
        for n1 in range(self.Lx):
            k1 = 2.0 * np.pi * n1 / self.Lx
            for n2 in range(self.Ly):
                k2 = 2.0 * np.pi * n2 / self.Ly
                energies.extend(self.dispersion(k1, k2))
        return np.sort(np.asarray(energies, dtype=float))

    def realspace_bloch_spectrum_residual(self) -> float:
        er = np.sort(np.linalg.eigvalsh(self.clean_hamiltonian(sparse=False)))
        eb = self.clean_spectrum_from_bloch_grid()
        return float(np.max(np.abs(er - eb)))

    def local_t2_cancellation_residual(
        self, R: Hexagon, h: Optional[float] = None
    ) -> float:
        """At h=t2 the six defect-hexagon NNN amplitudes should vanish."""
        if h is None:
            h = self.t2
        H = self.hamiltonian(defects=[R], h=h, sparse=False)
        vals = [
            H[self.idx(*s), self.idx(*t)]
            for s, t in self.hexagon_nnn_bonds(R)
        ]
        return float(np.max(np.abs(vals))) if vals else 0.0

    def run_sanity_checks(
        self,
        defects: Optional[Sequence[Hexagon]] = None,
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
