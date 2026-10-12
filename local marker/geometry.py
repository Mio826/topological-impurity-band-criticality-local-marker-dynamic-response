"""Three nested regions for Arnab Fig. 5: full A contains measurement R contains defects B.

The paper calls these 'boxes' but does not specify their edge convention or
exact random coordinates. We use centered Bravais-cell rectangles, which must
be recorded as a convention rather than claimed as exact pixel reproduction.
The defect candidates are complete hexagons; measurement includes both sites
of every primitive cell in a centered rectangle.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .model import HaldaneOBC, Hex


def centered_start(total: int, width: int) -> int:
    if not 1 <= width <= total:
        raise ValueError(f'Invalid centered width {width} for system {total}')
    return (total - width)//2


@dataclass(frozen=True)
class FigureRegions:
    model: HaldaneOBC
    defect_width: int
    integration_width: int

    def __post_init__(self):
        if not (1 <= self.defect_width < self.integration_width <= min(self.model.Lx,self.model.Ly)):
            raise ValueError('Require defect_width < integration_width <= min(Lx,Ly).')

    @property
    def defect_bounds(self) -> tuple[int,int,int,int]:
        a0=centered_start(self.model.Lx,self.defect_width)
        b0=centered_start(self.model.Ly,self.defect_width)
        return a0,a0+self.defect_width,b0,b0+self.defect_width

    @property
    def integration_bounds(self) -> tuple[int,int,int,int]:
        a0=centered_start(self.model.Lx,self.integration_width)
        b0=centered_start(self.model.Ly,self.integration_width)
        return a0,a0+self.integration_width,b0,b0+self.integration_width

    def integration_mask(self) -> np.ndarray:
        n1=np.tile(np.arange(self.model.Lx),self.model.Ly)
        n2=np.repeat(np.arange(self.model.Ly),self.model.Lx)
        a0,a1,b0,b1=self.integration_bounds
        inside=(n1>=a0)&(n1<a1)&(n2>=b0)&(n2<b1)
        return np.repeat(inside,2)

    def defect_candidates(self) -> tuple[Hex,...]:
        a0,a1,b0,b1=self.defect_bounds
        return tuple((a,b) for b in range(b0,b1) for a in range(a0,a1)
                     if 0<=a<self.model.Lx-1 and 0<=b<self.model.Ly-1)

    def choose_defects(self, count: int, seed: int) -> tuple[Hex,...]:
        choices=self.defect_candidates()
        if not 0 <= count <= len(choices):
            raise ValueError('Defect count exceeds central-region complete hexagons.')
        rng=np.random.default_rng(int(seed))
        indexes=np.sort(rng.choice(len(choices),size=count,replace=False))
        return tuple(choices[int(k)] for k in indexes)

    def group_site_by_cell(self, marker: np.ndarray) -> np.ndarray:
        if marker.shape != (self.model.n_sites,):
            raise ValueError('Site marker has wrong length.')
        return marker.reshape(self.model.Ly,self.model.Lx,2).sum(axis=2)

    def metadata(self, centers: tuple[Hex,...]) -> dict:
        return dict(defect_bounds=self.defect_bounds,
                    integration_bounds=self.integration_bounds,
                    n_defects=len(centers),n_candidates=len(self.defect_candidates()),
                    sampled_density=len(centers)/max(1,len(self.defect_candidates())),
                    chosen_defects=[list(x) for x in centers])
