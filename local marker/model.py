"""Standalone open-boundary Haldane model with hexagonal flux defects.

Conventions inherited from the physical *definitions* in the Stage-1 Haldane
model; this module does not import the Stage-1 implementation.

Primitive cells have two orbitals (A=0,B=1); a defect is a complete hexagon
whose six oriented next-nearest-neighbour hoppings acquire -1j*h.
For clean NNN amplitude +1j*t2, this changes that hexagon to 1j*(t2-h).
Overlapping hexagons contribute additively (they need not be disjoint).

Only OBC is implemented. Cartesian coordinates have nearest-neighbour
spacing 1, and ordering is index=2*(n1+Lx*n2)+sublattice.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy import sparse

Hex = tuple[int, int]
Site = tuple[int, int, int]
_A1 = np.array([-np.sqrt(3) / 2, -1.5], dtype=float)
_A2 = np.array([-np.sqrt(3) / 2, +1.5], dtype=float)
_BASIS = np.array([[0., 0.], [0., 1.]], dtype=float)


@dataclass(frozen=True)
class HaldaneOBC:
    Lx: int
    Ly: int
    t: float = 1.0
    t2: float = 0.02
    m: float = 0.0

    def __post_init__(self):
        if self.Lx < 3 or self.Ly < 3:
            raise ValueError('Need Lx, Ly >= 3 for complete hexagons.')
        if not np.isfinite([self.t, self.t2, self.m]).all():
            raise ValueError('Model couplings must be finite.')

    @property
    def n_sites(self) -> int:
        return 2 * self.Lx * self.Ly

    def index(self, site: Site) -> int:
        n1, n2, sub = map(int, site)
        if not (0 <= n1 < self.Lx and 0 <= n2 < self.Ly and sub in (0,1)):
            raise ValueError(f'Site outside open sample: {site}')
        return 2 * (n1 + self.Lx*n2) + sub

    def positions(self) -> np.ndarray:
        n1, n2 = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        cells = n1.reshape(-1, 1)*_A1 + n2.reshape(-1, 1)*_A2
        return (cells[:,None,:] + _BASIS[None,:,:]).reshape(-1,2)

    def hex_vertices(self, R: Hex) -> tuple[Site, ...]:
        a,b = map(int,R)
        if not (0 <= a < self.Lx-1 and 0 <= b < self.Ly-1):
            raise ValueError(f'Incomplete hexagon at {R}')
        return ((a,b,0), (a,b,1), (a,b+1,0),
                (a+1,b+1,1), (a+1,b+1,0), (a+1,b,1))

    def hex_center(self, R: Hex) -> np.ndarray:
        a,b = map(int,R)
        self.hex_vertices((a,b))
        return a*_A1 + b*_A2 + np.array([-np.sqrt(3)/2, 0.5])

    def hex_nnn_bonds(self, R: Hex) -> tuple[tuple[int,int], ...]:
        v = [self.index(s) for s in self.hex_vertices(R)]
        return ((v[0],v[2]), (v[2],v[4]), (v[4],v[0]),
                (v[1],v[3]), (v[3],v[5]), (v[5],v[1]))

    def all_hexagons(self) -> tuple[Hex, ...]:
        return tuple((a,b) for b in range(self.Ly-1) for a in range(self.Lx-1))

    @staticmethod
    def _append_hop(rows: list[int], cols: list[int], values: list[complex],
                    i: int, j: int, value: complex):
        rows.extend((i,j)); cols.extend((j,i))
        values.extend((complex(value),complex(value).conjugate()))

    def pristine(self) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[complex] = []
        for b in range(self.Ly):
            for a in range(self.Lx):
                ia = self.index((a,b,0)); ib = self.index((a,b,1))
                rows.extend((ia,ib)); cols.extend((ia,ib))
                vals.extend((self.m,-self.m))
                for da,db in ((0,0),(1,0),(0,-1)):
                    a2,b2=a+da,b+db
                    if 0 <= a2 < self.Lx and 0 <= b2 < self.Ly:
                        self._append_hop(rows,cols,vals,ia,self.index((a2,b2,1)),-self.t)
                for sub, directions in (
                    (0,((0,1),(1,0),(-1,-1))),
                    (1,((1,1),(0,-1),(-1,0)))
                ):
                    source=self.index((a,b,sub))
                    for da,db in directions:
                        a2,b2=a+da,b+db
                        if 0 <= a2 < self.Lx and 0 <= b2 < self.Ly:
                            self._append_hop(rows,cols,vals,source,
                                             self.index((a2,b2,sub)),1j*self.t2)
        return sparse.coo_matrix((vals,(rows,cols)),shape=(self.n_sites,)*2,
                                 dtype=np.complex128).tocsr()

    def defect_unit_operator(self, hexagons: Iterable[Hex]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[complex] = []
        seen: set[Hex] = set()
        for center in hexagons:
            key=tuple(map(int,center))
            if key in seen:
                raise ValueError(f'Duplicate defect center: {key}')
            seen.add(key)
            for i,j in self.hex_nnn_bonds(key):
                self._append_hop(rows,cols,vals,i,j,-1j)
        return sparse.coo_matrix((vals,(rows,cols)),shape=(self.n_sites,)*2,
                                 dtype=np.complex128).tocsr()

    def hamiltonian(self, h: float, unit_defect: sparse.csr_matrix,
                    pristine: sparse.csr_matrix) -> sparse.csr_matrix:
        return (pristine + float(h)*unit_defect).tocsr()
