# Finite-Density Topological Impurity Response

## 1. Project goal

This project studies the **magnetic and optical response of a finite density of topological impurities** in two-dimensional Chern/Dirac lattice models.

The immediate motivation is the progression

1. **Xu, Seth, Kimchi, PRB 113, 045151 (2026)**: a *single* Haldane-type magnetic defect on the honeycomb lattice can produce a chirality reversal near a resonant impurity bound-state crossing.
2. **Seth & Kimchi, arXiv:2609.09118 (2026), "Topological Impurity Bands"**: at finite defect density, impurity bound states hybridize into topological impurity bands.
3. **Current project**: ask what happens to experimentally relevant magnetic/optical observables when many impurities are present.

The primary observables are

- local Chern marker,
- real-space orbital magnetization,
- frequency-integrated magnetic circular dichroism (MCD).

The main active setting is **random finite-density impurities**.  Engineered fixed impurity patterns (corrals / Shiba-like settings) may be considered later.

The primary microscopic models are

- **QWZ**: onsite random mass defects;
- **Honeycomb/Haldane**: local flux defects living on hexagons.

Kagome is optional and is not part of the current core implementation.

---

## 2. Physical geometry: \(A \supset B \supset C\)

All production response calculations use an open finite sample with three nested spatial regions:

- \(A\): the full open sample;
- \(B\): the interior region in which impurities are allowed;
- \(C\): a deeper central region in which real-space observables are averaged.

The purpose is to separate three effects:

1. the physical boundary of the finite sample;
2. the artificial boundary of the impurity-containing region;
3. the bulk response we actually want to measure.

The code defines the regions using physical Cartesian coordinates and Euclidean clearance from the convex hull of the open sample:

\[
B:\qquad d(\mathbf r,\partial A)\ge d_{AB},
\]

\[
C:\qquad d(\mathbf r,\partial A)\ge d_{AB}+d_{BC}.
\]

This definition works for both the square QWZ lattice and the oblique honeycomb lattice.

For a final observable \(\mathcal O_i\), the quantity reported for one disorder realization is the central-window average

\[
\mathcal O_C
=
\frac{1}{N_C}
\sum_{i\in C}\mathcal O_i.
\]

A disorder average is performed only afterwards.

---

## 3. Microscopic models

### 3.1 QWZ

The clean Bloch Hamiltonian is

\[
H_0(\mathbf k)
=
A\sin k_x\,\sigma_x
+
A\sin k_y\,\sigma_y
+
\left[u+t\cos k_x+t\cos k_y\right]\sigma_z .
\]

A defect at square-lattice site \(R\) is an onsite mass impurity

\[
V_R=-h\sigma_z.
\]

Thus the local mass changes as

\[
u\rightarrow u-h.
\]

For a frozen defect configuration \(\mathcal D_s\),

\[
H_s(h)=H_0+hW_s,
\]

where \(W_s\) is a unit-strength defect operator built once for that realization.

The present convention gives the clean QWZ sequence

\[
C(u<-2t)=0,\qquad
C(-2t<u<0)=+1,\qquad
C(0<u<2t)=-1,\qquad
C(u>2t)=0
\]

for \(A,t>0\) with the projector/Berry-curvature sign convention used by `markers.py`.

### 3.2 Honeycomb / Haldane

The clean Bloch Hamiltonian is

\[
H(\mathbf k)
=
\begin{pmatrix}
m+t_2 d_{\mathbf k} & h_{\mathbf k}\\
h_{\mathbf k}^* & -m-t_2d_{\mathbf k}
\end{pmatrix},
\]

with

\[
d_{\mathbf k}
=
-2\left[
\sin k_1+\sin k_2-\sin(k_1+k_2)
\right],
\]

\[
h_{\mathbf k}
=
-t\left[
1+e^{ik_1}+e^{-ik_2}
\right].
\]

The clean topological transition occurs at

\[
m_c=3\sqrt{3}\,t_2.
\]

A magnetic defect lives on a **hexagon**, not on a site.  It modifies the six counterclockwise next-nearest-neighbor Haldane hoppings:

\[
V_R
=
-i h
\sum_{\langle\!\langle ij\rangle\!\rangle\in R}
c_i^\dagger c_j
+\mathrm{H.c.}
\]

so that locally

\[
t_2\rightarrow t_2-h.
\]

The real-space embedding currently used has

\[
\operatorname{sgn}\det(\mathbf a_1,\mathbf a_2)=-1.
\]

This is not an error.  Real-space observables use physical Cartesian coordinates directly, and the independent Bloch validation includes this orientation sign.  Do not change the lattice handedness without rerunning all sign-sensitive marker tests.

---

## 4. Disorder convention

The production default is **fixed-count disorder**, not Bernoulli disorder.

For a chosen density \(\rho\),

\[
N_{\rm imp}
=
\operatorname{round}
\left(
\rho N_{\rm eligible}^{(B)}
\right).
\]

Then \(N_{\rm imp}\) defect centers are sampled uniformly without replacement from the eligible centers in \(B\).

Important:

- QWZ defect centers are lattice sites.
- Haldane defect centers are complete hexagons.
- Density is defined relative to the number of **eligible defect centers in \(B\)**.
- A realization is sampled once and then **frozen for the entire \(h\)-scan**.
- The same `(base_seed, realization_id)` reproduces the same realization exactly.

For realization \(s\),

\[
\mathcal D_s
\quad\longrightarrow\quad
W_s
\quad\longrightarrow\quad
H_s(h)=H_0+hW_s.
\]

Never redraw disorder independently for each value of \(h\).

---

## 5. Real-space observables

For a finite open Hamiltonian \(H\) and chemical potential \(\mu\),

\[
P=\Theta(\mu-H),\qquad Q=1-P.
\]

The code diagonalizes \(H\) once and reuses the same projector for all three markers.

### 5.1 Local Chern marker

Bianco--Resta:

\[
\boxed{
\mathcal C(\mathbf r)
=
4\pi\,
\mathrm{Im}
\langle\mathbf r|QxPyQ|\mathbf r\rangle
}
\]

with the code dividing basis contributions by primitive-cell area and summing internal degrees of freedom into a cell-resolved marker.

For a finite open sample,

\[
\sum_{\mathbf r\in A}\mathcal C(\mathbf r)=0,
\]

so the **whole-sample average is not the bulk Chern number**.  The topological bulk value is obtained from a central-window average in \(C\).

### 5.2 Orbital magnetization

Bianco--Resta:

\[
\boxed{
\mathcal M(\mathbf r)
=
\frac{e}{\hbar c}
\mathrm{Im}
\left\langle\mathbf r\left|
PxQHQyP
-
QxP(H-2\mu)PyQ
\right|\mathbf r\right\rangle
}
\]

again normalized per primitive-cell area.

The code exposes the decomposition

\[
\mathcal M
=
\mathcal M_{\rm LC}
+
\mathcal M_{\rm IC}
+
\mathcal M_{\rm Chern}.
\]

The explicit chemical-potential contribution satisfies

\[
\mathcal M_{\rm Chern}(\mathbf r)
=
\frac{e\mu}{2\pi\hbar c}
\mathcal C(\mathbf r).
\]

The microscopic site-resolved marker should not be overinterpreted as a unique atomic magnetic moment.  The physically robust object is the bulk/coarse-grained spatial average.

### 5.3 Frequency-integrated MCD

Resta:

\[
\boxed{
\mathcal X_{xy}(\mathbf r)
=
\frac{\pi e^2}{2\hbar^2}
\mathrm{Im}
\left\langle\mathbf r\left|
(H-\mu)
[[x,P],[y,P]]
\right|\mathbf r\right\rangle
}
\]

with primitive-cell-area normalization.

This is the real-space density corresponding to the integrated antisymmetric optical response

\[
I_{xy}
=
\mathrm{Im}
\int_0^\infty d\omega\,
\sigma_{xy}^{(-)}(\omega).
\]

Orbital magnetization and MCD are related T-breaking observables but are not the same quantity.

---

## 6. Code architecture

Current core files:

```text
qwz.py
haldane.py
geometry.py
disorder.py
markers.py
```

Validation notebooks:

```text
test_lattice_models.ipynb
test_geometry_disorder.ipynb
test_markers_physics.ipynb
```

### `qwz.py`

Responsibilities:

- square-lattice QWZ geometry;
- clean real-space Hamiltonian;
- Bloch Hamiltonian;
- onsite mass defects;
- `H(h)=H0+hW`;
- physical basis coordinates;
- deterministic defect-center interface.

Important public interface:

```python
model = QWZModel(...)
H0 = model.clean_hamiltonian()
centers = model.available_defect_centers()
r = model.defect_center_position(center)
W = model.defect_operator(defects)
H = model.hamiltonian(defects=defects, h=h)
positions = model.positions()
```

### `haldane.py`

Responsibilities:

- honeycomb real-space geometry;
- Haldane NN/NNN hopping convention;
- clean Bloch Hamiltonian;
- complete hexagon enumeration;
- local hexagon flux defects;
- `t2 -> t2-h`;
- physical site and defect-center coordinates.

It intentionally exposes the same general interface as `QWZModel`.

### `geometry.py`

Main class:

```python
NestedRegions
```

Example:

```python
regions = NestedRegions.from_model(
    model,
    impurity_margin=d_AB,
    measurement_margin=d_BC,
)
```

Important methods:

```python
regions.eligible_defect_centers(model)
regions.measurement_basis_mask(model)
regions.signed_clearance(points)
regions.validate_for_model(model)
```

### `disorder.py`

Main classes:

```python
DefectSampler
DefectRealization
PreparedRealization
```

Typical use:

```python
sampler = DefectSampler(
    density=rho,
    mode="fixed_count",
    base_seed=seed,
)

realization = sampler.sample(
    model,
    regions,
    realization_id=s,
)

prepared = prepare_realization(
    model,
    realization,
)

H = prepared.hamiltonian(h)
```

### `markers.py`

Main class:

```python
MarkerCalculator
```

Typical use:

```python
calc = MarkerCalculator.from_model(
    model,
    H,
    mu=mu,
)

fields = calc.compute_fields()
mask_C = regions.measurement_basis_mask(model)

C_bulk = calc.bulk_average(
    fields.chern_cell,
    basis_mask=mask_C,
)

M_bulk = calc.bulk_average(
    fields.orbital_cell,
    basis_mask=mask_C,
)

MCD_bulk = calc.bulk_average(
    fields.mcd_cell,
    basis_mask=mask_C,
)
```

---

## 7. Validation status

### 7.1 Lattice models

Both QWZ and Haldane pass:

- Hermiticity;
- dense/sparse agreement;
- exact `H = H0 + h W`;
- real-space PBC spectrum versus Bloch spectrum;
- analytic clean gap-closing points;
- clean Chern phases;
- QWZ `u -> u-h`;
- Haldane local `t2 -> t2-h`.

Real-space/Bloch spectral residuals are at machine precision, roughly \(10^{-14}\) or below.

### 7.2 Geometry/disorder

Both models pass:

- \(C\subset B\subset A\);
- all sampled defect centers lie in \(B\);
- Haldane candidates are complete hexagons;
- exact fixed-count density;
- seed reproducibility;
- distinct realization IDs generate distinct samples;
- frozen realization reused across an \(h\)-scan;
- cached `H0+hW` exactly equals direct microscopic construction.

### 7.3 Real-space markers

The marker implementation passes substantially stronger physics tests.

#### Projector algebra

\[
P^2=P,\quad Q^2=Q,\quad PQ=0,\quad [H,P]=0
\]

to machine precision.

#### OBC Chern trace identity

\[
\sum_i \mathcal C_i=0
\]

to approximately \(10^{-13}\) or better.

#### Independent Bloch-space comparison

Representative QWZ test:

```text
C:   real = +0.999989    BZ = +0.998810
M:   real = +0.031831    BZ = +0.031793
MCD: real = -0.478914    BZ = -0.478404
```

Representative Haldane test:

```text
C:   real = +0.990715    BZ = +0.998143
M:   real = +0.023714    BZ = +0.023829
MCD: real = -0.422023    BZ = -0.421891
```

The remaining mismatch is finite-size / finite-grid convergence.

#### Exact OM--Chern identity

\[
\frac{\partial M(\mathbf r)}{\partial\mu}
=
\frac{e}{2\pi\hbar c}
\mathcal C(\mathbf r)
\]

at fixed projector, to machine precision.

#### MCD chemical-potential identity

The fully integrated MCD is unchanged by varying the explicit \(\mu\) while holding \(P\) fixed.

#### Invariances

All three markers pass:

- energy-zero invariance:
  \[
  H\to H+\Delta,\qquad \mu\to\mu+\Delta;
  \]
- coordinate-origin invariance:
  \[
  \mathbf r\to\mathbf r+\mathbf r_0;
  \]
- time-reversal/complex-conjugation sign reversal:
  \[
  H\to H^*
  \quad\Rightarrow\quad
  (\mathcal C,M,\mathrm{MCD})
  \to
  -(\mathcal C,M,\mathrm{MCD});
  \]
- TR-symmetric null test for real Haldane Hamiltonian.

The current backend is therefore considered **validated sufficiently for the first production physics scans**.

---

## 8. Standard production workflow

For one model and one density:

```python
# 1. Build open microscopic model
model = ...

# 2. Define A > B > C
regions = NestedRegions.from_model(
    model,
    impurity_margin=d_AB,
    measurement_margin=d_BC,
)

# 3. Build disorder sampler
sampler = DefectSampler(
    density=rho,
    mode="fixed_count",
    base_seed=seed,
)

# 4. Loop over disorder realizations
for realization_id in range(Nreal):

    realization = sampler.sample(
        model,
        regions,
        realization_id=realization_id,
    )

    prepared = prepare_realization(
        model,
        realization,
    )

    # 5. Keep realization fixed while scanning h
    for h in h_values:

        H = prepared.hamiltonian(h)

        calc = MarkerCalculator.from_model(
            model,
            H,
            mu=mu,
        )

        fields = calc.compute_fields()
        mask_C = regions.measurement_basis_mask(model)

        C = calc.bulk_average(
            fields.chern_cell,
            basis_mask=mask_C,
        )
        M = calc.bulk_average(
            fields.orbital_cell,
            basis_mask=mask_C,
        )
        I = calc.bulk_average(
            fields.mcd_cell,
            basis_mask=mask_C,
        )

        save(realization_id, rho, h, C, M, I)
```

Only after all realizations are complete should one form

\[
\overline C(h,\rho),\qquad
\overline M(h,\rho),\qquad
\overline I_{\rm MCD}(h,\rho),
\]

together with realization-to-realization fluctuations.

---

## 9. Important conventions and warnings

1. **Markers use OBC.**  
   Do not pass a PBC model into `MarkerCalculator.from_model`.

2. **Do not average the local Chern marker over the entire OBC sample.**  
   Its full trace is exactly zero.  Use region \(C\).

3. **Do not redraw disorder at every \(h\).**  
   Freeze the realization and scan \(h\) on the same \(W_s\).

4. **Haldane impurities live on hexagons.**  
   Never reinterpret the density as a fraction of atomic sites without explicitly converting conventions.

5. **The current marker implementation is dense.**  
   It stores full \(N\times N\) matrices such as \(P,Q\) and several intermediate products.  Computational cost and memory rise rapidly with system size.

6. **Chemical potential near an eigenvalue is dangerous.**  
   Inspect `nearest_fermi_distance` and `near_fermi_count` from projector diagnostics.

7. **Do not alter sign conventions casually.**  
   QWZ Chern sign, honeycomb Bravais handedness, Haldane NNN orientation, OM sign, and MCD sign are now cross-validated against independent Bloch-space formulas.

8. **Single-site OM/MCD textures should not automatically be interpreted as unique atomic observables.**  
   The robust quantities for the present project are coarse-grained / central-window averages.

---

## 10. Current research question

The immediate numerical question is:

> When a finite density of local mass/flux defects is present, does the anomalous single-impurity chirality/orbital response survive disorder averaging, and how does it evolve as impurity wavefunctions begin to overlap and form impurity bands?

Natural control parameters are

\[
h,\qquad
\rho_{\rm imp},
\]

with microscopic host parameters fixed initially.

The first production scans should therefore establish

1. the single-impurity limit;
2. the dilute finite-density regime;
3. the onset of overlapping impurity responses;
4. the relation among local Chern marker, orbital magnetization, and MCD;
5. sample-to-sample fluctuations.

A full finite-size scaling study is **not** the first goal of this project.

---

## 11. Computational note

The present `markers.py` implementation uses dense diagonalization and dense projector products.

If the single-particle Hilbert-space dimension is \(D\),

- Hamiltonian/projector storage scales as \(D^2\);
- dense diagonalization scales approximately as \(D^3\);
- the marker calculation forms several additional \(D\times D\) matrices.

For QWZ,

\[
D=2L_xL_y.
\]

For Haldane,

\[
D=2L_xL_y
\]

when \(L_x,L_y\) count honeycomb unit cells.

Small validation scans are appropriate for a laptop.  Production disorder averaging should normally run on a workstation or cluster until the marker backend is optimized.

---

## 12. Core references

- R. Xu, A. Seth, and I. Kimchi, **"Chirality reversal at finite magnetic impurity strength and local signatures of a topological phase transition,"** Phys. Rev. B **113**, 045151 (2026).
- A. Seth and I. Kimchi, **"Topological Impurity Bands,"** arXiv:2609.09118 (2026).
- R. Bianco and R. Resta, **"Mapping topological order in coordinate space,"** Phys. Rev. B **84**, 241106(R) (2011).
- R. Bianco and R. Resta, **"Orbital Magnetization as a Local Property,"** Phys. Rev. Lett. **110**, 087202 (2013).
- R. Resta, **"Magnetic circular dichroism versus orbital magnetization,"** Phys. Rev. Research **2**, 023139 (2020).

---

## 13. Current status

Completed and validated:

```text
[done] qwz.py
[done] haldane.py
[done] geometry.py
[done] disorder.py
[done] markers.py

[done] lattice-model validation
[done] geometry/disorder validation
[done] independent physics validation of all real-space markers
```

Next task:

```text
Design and run the first finite-density response scan:
    model parameters
    system size
    A/B/C margins
    impurity densities
    h windows
    number of disorder realizations
    convergence checks
```
