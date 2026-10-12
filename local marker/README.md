# Local Chern markers in topological impurity-band models (Stage 4)

A **minimal research code repository** for the open-boundary Haldane model with hexagonal flux defects, inspired by the unpublished local-marker calculations discussed alongside *Topological Impurity Bands* (Seth *et al.*, [arXiv:2609.09118](https://arxiv.org/abs/2609.09118)). The numerical notebooks document our Stage-4 small-system Fig. 5 reproduction, spatial marker distributions, and robustness across fixed random defect realizations.

This repository preserves the **core physical definitions and reproducible calculation pipelines**. It deliberately excludes large eigenvector caches, temporary plotting scripts, PDFs generated for discussion, unvalidated near-resonance KPM approximations, and one-off troubleshooting code.

## Repository layout

```text
stage4-local-marker/
├── README.md
├── requirements.txt
├── .gitignore
├── src/stage4_local_marker/
│   ├── __init__.py
│   ├── model.py          # OBC Haldane Hamiltonian and hexagonal flux defects
│   ├── geometry.py       # full sample A, defect box B, integration box R
│   ├── marker.py         # exact occupied projector and site-resolved local Chern marker
│   └── analysis.py       # finite-rank resonance roots, regional sums, state-exchange diagnostics
├── notebooks/
│   ├── 01_fig5_reproduction.ipynb
│   ├── 02_spatial_and_projector.ipynb
│   └── 03_disorder_realizations.ipynb
└── data/
    └── reference_realizations.csv   # compact historical reference values, not a new result
```

## Physical definitions

The two-orbital honeycomb Hamiltonian uses real nearest-neighbor hopping `t`, purely imaginary next-nearest-neighbor hopping `i*t2`, sublattice mass `m`, and an open boundary. Each hexagonal defect modifies its six oriented next-nearest-neighbor bonds by `-i*h`:

\[
H(h)=H_0+hW,\qquad
P(h)=\sum_{E_n<\mu}|n\rangle\langle n|,\qquad Q=1-P.
\]

The **unnormalized**, site-resolved Chern marker is

\[
C_i(h)=4\pi\,\mathrm{Im}\langle i|QxPyQ|i\rangle,
\qquad \delta C_i(h)=C_i(h)-C_i(0),
\]

and the measured quantity is an **integral/sum over the central integration region**

\[
M_R(h)=\sum_{i\in R}\delta C_i(h).
\]

There is **no division by lattice area or number of sites**. The three nested regions are **\(A\supset R\supset B\)**: full open sample, marker integration region, and random-defect placement box. The whole finite OBC sample satisfies \(\sum_{i\in A}\delta C_i=0\), so spatial compensation is mandatory. This marker is **not** the physical orbital magnetization density or a directly measured bond current.

Numerical conventions have been kept consistent with the original Stage-4 `fig5_haldane.py`, `fig5_geometry.py`, and the dense exact branch of `fig5_marker.py`. The model construction allows defects whose hexagons touch or overlap, with additive bond contributions. If reproducing a different paper configuration that forbids neighboring defects, change the sampling constraint explicitly rather than assuming it here.

### Finite-rank zero-energy crossing

For a finite sample and invertible clean \(H_0\), the low-rank secular equation on the defect support gives

\[
\det[1+hR_0W]=0,\qquad R_0=H_0^{-1}.
\]

Here $R_0$ denotes the inverse of the clean Hamiltonian; the conventional retarded Green function at zero energy is $G_0(0)=(-H_0)^{-1}=-R_0$. The code reports candidate **finite-sample zero-energy crossings**. A root may be double because of particle-hole partners. These roots are *not*, by themselves, proofs of finite-density Bott-index or thermodynamic TIB transitions. Exact eigendecomposition at the two sides of a root checks the occupied-state exchange.

## Quick start

From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

Open notebooks in the following order:

1. **`01_fig5_reproduction.ipynb`** — small exact OBC spectra and the two resonance-centered local-marker features, with a manageable `L=14` default. Optionally increase `L` or change `t2`.
2. **`02_spatial_and_projector.ipynb`** — fixed `L=30`, three-defect calculation of signed spatial maps, regional compensation, one-sided projector jump, and optionally two/four-state reconstruction.
3. **`03_disorder_realizations.ipynb`** — repeats the two one-sided resonance tests for three **fixed** independent seeds `20261008`, `20261009`, `20261010`. It is computationally expensive by default.

The notebooks import only from `src/stage4_local_marker` (with an explicitly set notebook path), so they can be run either from the project root or from `notebooks/`. They write no files unless `SAVE=True` is set; outputs are placed under the Git-ignored `outputs/` directory.

## Reproducibility and numerical limits

The historical Stage-4 benchmark used `L=30`, `t=1`, `t2=0.02`, `m=mu=0`, three hexagons sampled from the centered `10×10` cell defect box and a centered `15×15` cell integration box. The three seeds and approximate one-sided jumps are recorded in [`data/reference_realizations.csv`](data/reference_realizations.csv). The original seed's roots are \(h_c^-\approx-1.227127335\) and \(h_c^+\approx0.794102487\), using the stated geometry.

The key **qualitative observations** are:

- Near each impurity-associated resonance, a particle-hole-paired state crosses zero and the occupied projector has a finite one-sided exchange, producing a sharp **jump** rather than a demonstrated smooth divergence.
- The signed marker is large in the defect region but extends beyond it; the full OBC marker sum is compensated, often substantially at the physical edge.
- The state-exchange mechanism repeats across three fixed disorder configurations. **Peak/jump sign and size are not realization-independent**, and a fixed four-state reconstruction is not equally accurate for every realization.

**Outstanding limitation:** in separate fixed-defect boundary checks, enlarging the full sample from `L=30` to `34` and `38` altered the marker jumps, including a sign reversal on one branch. Therefore this repository must **not** be cited as evidence of convergence to a boundary-independent local observable. Random-realization robustness is not a substitute for physical-boundary or integration-window convergence.

All backends here are **dense exact diagonalization**: \(N=2L^2\) orbitals, cubic eigensolver time and substantial memory. `L=30` means an 1800×1800 complex Hermitian problem; `L=40` or `L=90` should **not** be used casually in these notebooks. Before running on a laptop, close other memory-intensive applications; CPU threading can be limited before launching Jupyter, e.g. `OPENBLAS_NUM_THREADS=4 jupyter lab` on Linux.

We do not include the exploratory Chebyshev/KPM implementation because its smoothing near the zero-energy occupation crossing was **not validated** for the sharp local-marker features.

## Scientific scope and attribution

This is a **research-record repository**, not a polished public library or a published demonstration of a local topological invariant. The clean/defect model and exact-marker conventions originated from the Stage-4 numerical code; the secular-resonance and projector-swap analysis were adapted from the subsequent fixed-realization studies. The notebooks use those same formulas while removing one-off data extraction and report-generation steps.

- Arnab Seth and collaborators, *Topological Impurity Bands*, [arXiv:2609.09118](https://arxiv.org/abs/2609.09118).
- R. Bianco and R. Resta, *Mapping topological order in coordinate space*, Phys. Rev. B **84**, 241106(R) (2011), [doi:10.1103/PhysRevB.84.241106](https://doi.org/10.1103/PhysRevB.84.241106).

This code computes an OBC **finite-size local-marker response**, not a disordered critical exponent, an FSS extrapolation, a Bott-index phase diagram, or a physical orbital magnetization.
