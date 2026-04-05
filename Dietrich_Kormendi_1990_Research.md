# Dietrich & Kormendi (1990): Perturbation of Plane-Wave Reflectivity

## Personal Context

This research traces directly to **T.M. Nestor, "A Practical Implementation of Waveform Inversion in a Stratified Marine Environment"** (Master of Science thesis, ~1991), which applied the Kormendi & Dietrich waveform inversion framework to real marine seismic data. The work below represents the theoretical foundation of that thesis, now revisited 35 years later in the context of modern deep learning and automatic differentiation.

---

## 1. The Paper

**Dietrich, M. and Kormendi, F. (1990).** "Perturbation of the Plane-Wave Reflectivity of a Depth-Dependent Elastic Medium by Weak Inhomogeneities."
*Geophysical Journal International*, **100**(2), 203–214.
[DOI: 10.1111/j.1365-246X.1990.tb02480.x](https://doi.org/10.1111/j.1365-246X.1990.tb02480.x)

**Kormendi, F. and Dietrich, M. (1991).** "Nonlinear Waveform Inversion of Plane-Wave Seismograms in Stratified Elastic Media."
*Geophysics*, **56**(5), 664–674.
[DOI: 10.1190/1.1443083](https://doi.org/10.1190/1.1443083)

---

## 2. Core Mathematical Contribution

### The Problem

Given a 1D stratified elastic medium (stack of horizontal layers with Vp, Vs, ρ, thickness per layer), compute the **exact sensitivity** of the full plane-wave seismic response to perturbations in each layer's elastic properties.

### The Approach

Dietrich & Kormendi derived **first-order (Born) perturbation formulae** for the plane-wave reflectivity of a depth-dependent elastic medium:

1. **Decompose** the medium into a reference model plus a small perturbation in elastic parameters (λ, μ, ρ)
2. **Expand** the elastic wave equations to first order in the perturbation
3. **Express** the resulting sensitivity as analytical **Fréchet derivatives** of the displacement field with respect to the elastic parameters at each depth
4. **Evaluate** these derivatives using the **Green's functions of the unperturbed medium**, computed via Kennett's recursive reflectivity algorithm

The Fréchet derivative at each depth takes the form:

```
δu(ω, p) = ∫ G(receiver → z) · P(z, δm) · G(z → source) dz
```

where:
- `G` = Green's functions computed by Kennett's reflectivity method
- `P` = perturbation operator (depends on δVp, δVs, δρ at depth z)
- `ω` = angular frequency, `p` = ray parameter

### Key Properties

- Captures the **full multi-layer response**: transmission losses, mode conversions, internal multiples, tuning effects
- Works in the **τ-p (intercept time – ray parameter) domain**
- Accurate for perturbations up to ~10% at all angles including the evanescent regime
- Breaks down near critical ray parameters (P-wave and S-wave transitions)
- Semi-analytical: reuses Green's functions already computed in the forward modeling step

---

## 3. Relationship to Kennett Reflectivity

The unperturbed Green's functions are computed using the **Generalized Reflection and Transmission Matrix Method of Kennett and Kerry (1979)**, which:

- Recursively builds reflection/transmission matrices from the bottom of the layer stack upward
- Is unconditionally numerically stable (unlike the original Thomson-Haskell propagator matrix method)
- Produces exact full-waveform solutions for 1D layered media in the frequency–ray parameter domain
- Includes all multiples, mode conversions, head waves, and evanescent contributions

The Fréchet derivatives are therefore computed **on top of** the Kennett forward modeling — the additional cost is modest since the Green's functions are already available.

---

## 4. The Inversion Framework (Kormendi & Dietrich, 1991)

The companion paper applied the analytical Fréchet derivatives to **nonlinear, gradient-based waveform inversion**:

- **Optimization**: Conjugate-gradient method minimizing L2 misfit between observed and synthetic τ-p seismograms
- **Gradient computation**: `∇χ = J^T · (d_obs - d_syn)` where J is the Fréchet derivative (Jacobian) matrix
- **Data partitioning strategy**: Invert near offsets / early arrivals / low frequencies first, progressively expanding to wider angles and higher frequencies
- **Iteration**: At each step, update the model, recompute the forward response and Fréchet derivatives, repeat
- **No finite-difference Jacobians needed**: The analytical derivatives are both exact and computationally efficient

### The Progression This Represents

```
Zoeppritz (1919)           — exact reflection coefficients, single interface
  → Aki-Richards (1980)    — linearized approximation, single interface
    → Dietrich & Kormendi  — linearized perturbation, full multi-layer stack via Kennett
      → Kormendi & Dietrich — nonlinear iterative FWI using the above Jacobians
        → Nestor (MSc)     — practical implementation in stratified marine environment
```

---

## 5. Extensions by the Grenoble Group

### Poroelastic Extension

**De Barros, L. and Dietrich, M. (2008).** "Perturbations of the Seismic Reflectivity of a Fluid-Saturated Depth-Dependent Poroelastic Medium."
*Journal of the Acoustical Society of America*, **123**(3), 1409–1420.
[arXiv: 0801.2442](https://arxiv.org/abs/0801.2442) | [PubMed: 18345830](https://pubmed.ncbi.nlm.nih.gov/18345830/)

- Extended the 1990 perturbation theory from elastic to **Biot poroelastic media**
- Derived Fréchet derivatives with respect to: porosity, permeability, consolidation parameter, fluid properties, solid density, mineral bulk/shear moduli
- Found porosity and consolidation parameter are the most sensitive parameters

### Poroelastic Inversion

**De Barros, L., Dietrich, M., and Valette, B. (2010).** "Full Waveform Inversion of Seismic Waves Reflected in a Stratified Porous Medium."
*Geophysical Journal International*, **182**(3), 1543–1556.
[DOI: 10.1111/j.1365-246X.2010.04696.x](https://academic.oup.com/gji/article/182/3/1543/599816)

- Quasi-Newton FWI for poroelastic parameters using the above Fréchet derivatives
- Demonstrated feasibility of extracting porosity, permeability, and fluid properties from seismic reflectivity data

---

## 6. Key Citing Works

### Sensitivity Studies

**Neves, F.A. and Singh, S.C. (1996).** "Sensitivity Study of Seismic Reflection/Refraction Data."
*Geophysical Journal International*, **126**(2), 470–488.
[Oxford Academic](https://academic.oup.com/gji/article/126/2/470/623784)

- Used the Dietrich-Kormendi perturbation framework to study sensitivity of multi-offset seismic data to elastic parameters

### Applications to Marine Geophysics

**Singh, S.C. and Dietrich, M. (1991).** "A Complete Waveform Inversion and Its Application to ECORS Data."
In *Continental Lithosphere: Deep Seismic Reflections*, AGU Geodynamics Series.

**Singh, S.C. et al. (1993, 1994).** Applied reflectivity-based FWI to **gas hydrate bottom-simulating reflectors (BSRs)**.
Published in *Science* and *JGR*. Used the Kormendi & Dietrich inversion with conjugate-gradient optimization.

### AVO Inversion for Layer Stacks

**Malovichko, L. (2015).** "Inverse AVO Problem for a Stack of Layers."
*Exploration Geophysics*, **46**(3).
[CSIRO Publishing](https://www.publish.csiro.au/eg/eg13020)

- AVO inversion for multilayered media using RT-matrices and differential seismograms

### Anisotropic Extension

**Ji, J. and Singh, S.C. (2005).** "Anisotropy from Full Waveform Inversion of Multicomponent Seismic Data Using a Hybrid Optimization Method."
*Geophysical Prospecting*.
[Wiley](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2478.2005.00476.x)

### Differential Seismograms

**Zeng, Y. and Anderson, J.G. (1995).** "A Method for Direct Computation of the Differential Seismogram with Respect to the Velocity Change in a Layered Elastic Solid."
*Bulletin of the Seismological Society of America*, **85**(1), 300–307.

### Modern Propagator-Matrix AVA Inversion

**Pan, X., Zhang, G., and Yin, X. (2019).** Pre-stack AVA inversion via propagator matrix with analytical Jacobians.
*Pure and Applied Geophysics*.
[Springer](https://link.springer.com/article/10.1007/s00024-019-02157-9)

**Pan, X. et al. (2020).** Joint PP/PS inversion via propagator matrix.
*Surveys in Geophysics*.
[Springer](https://link.springer.com/article/10.1007/s10712-020-09605-5)

---

## 7. Foundational References

**Kennett, B.L.N. (1983).** *Seismic Wave Propagation in Stratified Media.*
Cambridge University Press. [ANU Press (open access)](https://press.anu.edu.au/publications/seismic-wave-propagation-stratified-media)

**Kennett, B.L.N. and Kerry, N.J. (1979).** "Seismic Waves in a Stratified Half Space."
*Geophysical Journal of the Royal Astronomical Society*, **57**(2), 557–583.

**Thomson, W.T. (1950).** "Transmission of Elastic Waves Through a Stratified Solid Medium."
*Journal of Applied Physics*, **21**, 89–93.

**Haskell, N.A. (1953).** "The Dispersion of Surface Waves on Multilayered Media."
*Bulletin of the Seismological Society of America*, **43**, 17–34.

**Gilbert, F. and Backus, G.E. (1966).** "Propagator Matrices in Elastic Wave and Vibration Problems."
*Geophysics*, **31**, 326–332.

**Aki, K. and Richards, P.G. (1980).** *Quantitative Seismology: Theory and Methods.*
W.H. Freeman. (2nd edition 2002, University Science Books)

**Tarantola, A. (1984).** "Inversion of Seismic Reflection Data in the Acoustic Approximation."
*Geophysics*, **49**(8), 1259–1266.

**Tarantola, A. (1986).** "A Strategy for Nonlinear Elastic Inversion of Seismic Reflection Data."
*Geophysics*, **51**(10), 1893–1903.

---

## 8. Connection to Modern Automatic Differentiation

**What Dietrich & Kormendi hand-derived in 1990 is exactly what automatic differentiation computes automatically today.** Their analytical Fréchet derivatives are the manual equivalent of backpropagating through the Kennett recursion.

### Why This Matters Now

| 1990 (Manual) | 2026 (Automatic Differentiation) |
|---|---|
| Hand-derive Fréchet derivatives for each parameterization | AD computes gradients for any parameterization automatically |
| Re-derive for poroelastic, anisotropic extensions | Just change the forward code; AD handles the rest |
| Restricted to first-order Born perturbation | AD gives exact gradients through the full nonlinear forward model |
| Jacobian is a separate analytical formula | Jacobian is implicit in the computational graph |

### The Opportunity

The Kennett recursion is a **chain of small complex matrix operations** per layer — exactly the kind of computation that PyTorch/JAX handle natively. A modern implementation would:

1. **Implement the Kennett reflectivity method** as a differentiable computational graph in JAX or PyTorch
2. **Obtain exact gradients** via reverse-mode AD (backpropagation) — no need to hand-derive Fréchet derivatives
3. **Validate** the AD-computed gradients against the Dietrich & Kormendi analytical expressions
4. **Extend trivially** to attenuation (Q), anisotropy, poroelasticity, or any new parameterization
5. **Embed as a differentiable physics layer** inside a Bayesian Neural Network for uncertainty-quantified inversion

The closest existing analog is **ADsurf** (Liu et al., 2024), which implements the Thomson-Haskell transfer matrix in PyTorch for surface wave dispersion — but for surface waves, not body-wave reflectivity:

**Liu, F., Li, J., Fu, L., and Lu, L. (2024).** "Multimodal Surface Wave Inversion with Automatic Differentiation."
*Geophysical Journal International*, **238**(1), 290–312.
[GJI](https://academic.oup.com/gji/article/238/1/290/7659841) | [GitHub: ADsurf](https://github.com/liufeng2317/ADsurf)

**No published work has yet implemented the Kennett reflectivity method as a differentiable forward model for body-wave inversion.** This is a genuine gap — and a natural extension of the work begun in the Nestor MSc thesis 35 years ago.

---

## 9. Existing Codebase: `Kennett_Reflectivity/`

A complete Python 3.12 / NumPy implementation of the Kennett recursive reflectivity method already exists in this project, translated from the original Fortran `kennetslo.f`:

```
Kennett_Reflectivity/
├── layer_model.py             # LayerModel dataclass (Vp, Vs, ρ, h, Q per layer)
├── scattering_matrices.py     # P-SV interface coefficients (solid-solid + ocean-bottom)
├── kennett_reflectivity.py    # Core Kennett upward sweep with addition formula
├── kennett_reflectivity_gpu.py # GPU-accelerated batch version
├── kennett_seismogram.py      # Orchestrator: model → synthetic seismogram
├── kennett_gather.py          # Multi-offset gather computation
├── kennett_gather_gpu.py      # GPU-accelerated gather
├── source.py                  # Ricker wavelet
├── example_usage.py           # 6 usage examples
└── test_package.py            # Test suite
```

### Key Implementation Details

- **Vectorized over frequency**: All 2×2 matrix operations (inv, matmul) are batched over `nfreq` using `np.einsum`
- **Scattering coefficients are frequency-independent**: Precomputed once per interface, reused across all frequencies
- **Modified (normalized) coefficients**: Incorporates `√(η·ρ)` factors so the Kennett recursion preserves unitarity
- **Handles acoustic-elastic (ocean-bottom) interface**: Separate `ocean_bottom_interface()` with correct zero-S structure
- **Attenuation**: Complex slowness via quality factors Q_α, Q_β
- **Free surface option**: Optional reverberation operator for surface multiples

### Path to Differentiable Implementation

Every operation in `kennett_reflectivity.py` is already expressed as NumPy array operations that have direct JAX equivalents:

| NumPy Operation | JAX Equivalent | Differentiable? |
|----------------|----------------|-----------------|
| `np.sqrt()` | `jnp.sqrt()` | Yes |
| `np.exp(1j * ...)` | `jnp.exp(1j * ...)` | Yes |
| `np.einsum()` | `jnp.einsum()` | Yes |
| `batch_inv2x2()` (analytical) | Same formula with `jnp` | Yes |
| `LayerModel` (dataclass) | pytree or `NamedTuple` | Yes (as leaf values) |

The conversion from NumPy → JAX would make the entire forward model differentiable via `jax.grad()`, producing exactly the Fréchet derivatives that Dietrich & Kormendi (1990) hand-derived — but automatically, for any parameterization, and with support for higher-order derivatives.

---

## 10. Related Open-Source Code

| Repository | Description |
|-----------|-------------|
| [ADsurf (GitHub)](https://github.com/liufeng2317/ADsurf) | Thomson-Haskell in PyTorch for surface waves — closest analog |
| [ADFWI (GitHub)](https://github.com/liufeng2317/ADFWI) | AD-based full waveform inversion (finite-difference, 2D) |
| [Kennett reflectivity method (GitHub)](https://github.com/samhaug/reflectivity_method) | Modifications of Kennett's ERZSOL3 code (Fortran) |
| [ADSeismic.jl (GitHub)](https://github.com/kailaix/ADSeismic.jl) | Differentiable seismic modeling in Julia |
| [Seistorch (GitHub)](https://github.com/GeophyAI/seistorch) | PyTorch-based FWI (finite-difference wave equation) |
| [Deepwave (GitHub)](https://github.com/ar4/deepwave) | PyTorch wave propagation (1D/2D/3D, FD-based) |
| [sbi (GitHub)](https://github.com/sbi-dev/sbi) | Simulation-based Bayesian inference framework |
| [CNN Impedance Inversion (GitHub)](https://github.com/vishaldas/CNN_based_impedance_inversion) | Das et al. — uses Kennett for data generation only |
