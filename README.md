# SeismicInversion

**A modern Python reconstruction of the waveform inversion methods from:**

> T. M. Nestor, *"A Practical Implementation of Waveform Inversion in a
> Stratified Marine Environment"*, Master of Science thesis, Monash
> University, 1991.

The original work implemented the Kormendi & Dietrich (1991) nonlinear
waveform inversion framework in Fortran, applying it to marine seismic
data in the tau-p (plane-wave) domain.  This repository is a modern Python/PyTorch port of the original
Fortran code, reconstructing and extending those methods with NumPy
vectorisation, PyTorch automatic differentiation, and GPU
acceleration.

## Purpose

The 1991 thesis demonstrated that full-waveform inversion of
plane-wave seismograms could recover elastic earth models (P-wave
velocity, S-wave velocity, density, and layer thickness) from
frequency-domain reflectivity data.  The key ingredients were:

1. **Kennett's reflectivity method** for the forward model -- a
   numerically stable recursive algorithm that computes the exact
   plane-wave response of a stratified elastic half-space, including
   all internal multiples and P-SV mode conversions.

2. **Analytical Frechet derivatives** (Dietrich & Kormendi, 1990) for
   the Jacobian of the reflectivity with respect to each layer
   parameter, enabling gradient-based inversion.

3. **Levenberg-Marquardt regularised inversion** using the Hessian to
   achieve quadratic convergence near the solution.

This repository replaces the hand-derived Frechet derivatives with
**PyTorch automatic differentiation**, which computes the exact
Jacobian and Hessian through the Kennett recursion without any
analytical derivation.  It also adds the **Global Matrix Method**
(Chin, Hedstrom & Thigpen, 1984) as an alternative forward model that
yields cheaper derivatives through implicit differentiation of a
linear solve.

## Repository Structure

```
SeismicInversion/
├── Kennett_Reflectivity/        # Kennett recursive reflectivity + inversion
│   ├── layer_model.py           #   LayerModel dataclass, complex/vertical slowness
│   ├── scattering_matrices.py   #   P-SV interface reflection/transmission coefficients
│   ├── kennett_reflectivity.py  #   Recursive reflectivity (Kennett addition formulae)
│   ├── kennett_torch.py         #   Differentiable reflectivity (PyTorch AD)
│   ├── kennett_seismogram.py    #   Single-trace synthetic seismogram
│   ├── kennett_gather.py        #   Multi-offset gather (discrete wavenumber summation)
│   ├── kennett_gather_gpu.py    #   GPU-accelerated gather (MPS/CUDA)
│   ├── kennett_reflectivity_gpu.py  # Batched GPU reflectivity
│   ├── taup_inversion.py        #   Newton-LM inversion in tau-p domain
│   ├── frechet_analytical.py    #   Analytical Frechet derivatives (Dietrich & Kormendi)
│   ├── hessian_cross_terms.py   #   Hessian decomposition analysis
│   ├── source.py                #   Ricker wavelet (frequency & time domain)
│   ├── inversion_config.py      #   YAML config loader/validator
│   └── configs/                 #   YAML earth model configurations
│
├── GlobalMatrix/                # Global Matrix Method (Chin et al. 1984)
│   ├── layer_matrix.py          #   Displacement-stress E-matrices (NumPy + PyTorch)
│   ├── global_matrix.py         #   System assembly and solve (NumPy)
│   ├── gmm_torch.py             #   Differentiable version + Jacobian/Hessian (PyTorch)
│   ├── gmm_reflectivity_cli.py  #   CLI entry point
│   ├── config.py                #   YAML config loader/validator
│   └── configs/                 #   YAML earth model configurations
│
├── latex/                       # LaTeX documentation
│   └── frechet_derivatives.tex  #   Frechet derivative derivation
│
├── Dietrich_Kormendi_1990_Research.md   # Research notes on the original papers
└── BNN_Seismic_Inversion_Research.md    # Bayesian neural network extensions
```

## Features

### Forward Modelling

- **Kennett recursive reflectivity** for plane-stratified elastic
  media with anelastic attenuation (complex velocities via
  constant-Q).
- **Global Matrix Method** assembling all interface continuity
  conditions into a single dense linear system G x = b.
- Both methods validated to agree within machine precision (< 1e-12).
- Synthetic seismograms via discrete wavenumber summation with
  multiprocessing and GPU acceleration (Apple MPS, NVIDIA CUDA).
- Free-surface multiples, P-SV mode conversion, all internal
  multiples included.

### Automatic Differentiation

- Full Jacobian and Hessian of the reflectivity with respect to all
  layer parameters via `torch.func.jacrev` and `torch.func.hessian`.
- Analytical Frechet derivatives (Dietrich & Kormendi, 1990)
  implemented independently for cross-validation.
- Log-parameter space transformation for positivity and improved
  conditioning.

### Waveform Inversion

- Full-Newton Levenberg-Marquardt inversion in the tau-p domain.
- Exact Hessian (not Gauss-Newton approximation) via PyTorch AD.
- Multi-slowness data fitting across a discrete grid of ray
  parameters.
- YAML configuration for reproducible experiments.
- LaTeX/TikZ output: parameter tables, depth profiles, trace
  comparisons, convergence curves.

## Installation

```bash
conda env create -f envs/seismic.yml
conda activate seismic
```

The environment includes Python 3.12, NumPy, SciPy, matplotlib,
PyTorch, PyYAML, and ObsPy.

## Usage Examples

### Compute a synthetic seismogram

```bash
python -m Kennett_Reflectivity.kennett_seismogram \
    -p 0.2 -T 64 -n 2048 -o seismogram.png
```

### Compute a multi-offset gather

```bash
python -m Kennett_Reflectivity.kennett_gather \
    --r-min 0.5 --r-max 20.0 --dr 0.5 \
    -T 64 -n 2048 --np 2048 --p-max 0.8 -o gather.png
```

### GPU-accelerated gather

```bash
python -m Kennett_Reflectivity.kennett_gather_gpu \
    --r-min 0.5 --r-max 50.0 --dr 0.5 \
    -T 64 -n 2048 --np 8192 --p-max 1.2 \
    --free-surface --t-max 30 -o gather_gpu.png
```

### Run tau-p waveform inversion

```bash
# Default 5-layer ocean-crust model
python -m Kennett_Reflectivity.taup_inversion

# From YAML config with overrides
python -m Kennett_Reflectivity.taup_inversion \
    --config Kennett_Reflectivity/configs/default_ocean_crust.yaml \
    --max-iter 30 --perturbation 0.10
```

### Compute GMM reflectivity and compare with Kennett

```bash
python -m GlobalMatrix.gmm_reflectivity_cli \
    -p 0.1 0.2 0.3 0.4 0.6 -n 256

# From YAML config, single slowness, no comparison
python -m GlobalMatrix.gmm_reflectivity_cli \
    --config GlobalMatrix/configs/default_ocean_crust.yaml \
    -p 0.2 --no-compare
```

### Python API

```python
from Kennett_Reflectivity import (
    default_ocean_crust_model, kennett_reflectivity,
    invert_taup, model_to_tensors, jacobian, hessian,
)
from GlobalMatrix import gmm_reflectivity, gmm_jacobian
import numpy as np

# Forward model
model = default_ocean_crust_model()
omega = np.linspace(0.1, 25.0, 255)
R = kennett_reflectivity(model, p=0.2, omega=omega)

# Jacobian and Hessian via PyTorch AD
J = jacobian(model, p=0.2, omega=omega)    # (2*nfreq, n_params)
H = hessian(model, p=0.2, omega=omega)     # (n_params, n_params)

# Inversion
result = invert_taup(
    true_model=model,
    p_values=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    nfreq=64, perturbation=0.15, max_iter=50,
)
```

## Levenberg-Marquardt Inversion with the Exact Hessian

The 1991 thesis used the Gauss-Newton approximation to the Hessian,
which retains only the J^T J term and discards the second-order
residual terms.  This works well near the solution where residuals are
small, but can fail or converge slowly when the starting model is far
from the truth.

This implementation computes the **exact Hessian** via PyTorch AD,
which includes the full second-order information:

```
H = J^H J + sum_i r_i * d^2 R_i / dm^2
```

where J is the Jacobian of the complex reflectivity, r_i are the
residuals, and the second term captures the curvature of each
reflectivity element with respect to the model parameters.

### Why the exact Hessian matters

1. **Quadratic convergence**: The full Newton step converges
   quadratically near a minimum, compared to the superlinear
   convergence of Gauss-Newton.  This translates to fewer iterations
   for tight convergence.

2. **Robustness far from the solution**: The second-order residual
   terms provide curvature information that the Gauss-Newton
   approximation misses.  This is particularly important for the
   seismic inverse problem where the misfit surface is highly
   nonlinear due to cycle-skipping and velocity-thickness trade-offs.

3. **Reliable trust-region scaling**: Levenberg-Marquardt adds a
   damping term lambda * I to the Hessian.  With the exact Hessian,
   the eigenvalues correctly reflect the local curvature, so the
   damping parameter adapts to the true geometry of the misfit surface
   rather than the Gauss-Newton surrogate.

4. **Cross-parameter coupling**: The exact Hessian captures
   interactions between parameters of the same layer (e.g., the
   velocity-thickness trade-off) that are partially lost in the
   Gauss-Newton approximation.  The `hessian_cross_terms.py` module
   demonstrates this decomposition numerically.

### Levenberg-Marquardt update

At each iteration, the parameter update solves:

```
(H + lambda * I) * delta_m = -g
```

where H is the exact Hessian, g is the gradient, and lambda is
adjusted adaptively: increased when a step increases misfit (more
regularisation), decreased when a step decreases misfit (trust the
Newton direction).

The inversion operates in **log-parameter space**
(log alpha, log beta, log rho, log h) to enforce positivity and
improve conditioning.  The ocean layer is held fixed; the half-space
thickness is excluded from the parameter vector.

## Implicit Differentiation Through a Linear Solve

The Global Matrix Method reduces the forward problem to a single linear
system **G x = b**, where G is the assembled global matrix, b encodes
the source, and x contains the displacement-stress coefficients at
every interface.  Both G and b depend on the model parameters
θ = (α, β, ρ, h).

### Jacobian (first derivative)

Differentiate G x = b with respect to a model parameter θ:

```
G (dx/dθ) = db/dθ − (dG/dθ) x
```

This is a linear system with the **same coefficient matrix G** that was
already LU-factored during the forward solve.  The Jacobian column
dx/dθ costs one back-substitution — not a new factorisation.

### Hessian (second derivative)

Differentiate again with respect to a second parameter φ:

```
G (d²x/dθdφ) = d²b/dθdφ − (d²G/dθdφ) x
               − (dG/dθ)(dx/dφ) − (dG/dφ)(dx/dθ)
```

This is again a linear system with the **same G**.  The right-hand side
requires only quantities already computed: the first derivatives dx/dθ
and dx/dφ (from the Jacobian step), the solution x (from the forward
solve), and the second derivatives of G and b with respect to the model
parameters (assembled analytically from the layer matrices).

For the diagonal Hessian element (φ = θ) this simplifies to:

```
G (d²x/dθ²) = d²b/dθ² − (d²G/dθ²) x − 2 (dG/dθ)(dx/dθ)
```

### Cost summary

| Quantity | Kennett recursion | Global Matrix Method |
|----------|-------------------|----------------------|
| Forward x | Deep recursive sweep | LU factorise G, solve |
| Jacobian dx/dθ | Backprop through full recursion graph | One back-substitution per θ |
| Hessian d²x/dθdφ | Backprop through the Jacobian graph | One back-substitution per (θ,φ) pair |

The Hessian is where implicit differentiation pays off most: the
Kennett method requires AD to differentiate *through* the already
expensive Jacobian computation, doubling the graph depth, while the
Global Matrix Method reuses the same LU factorisation for every
derivative order.  Benchmarks in `GlobalMatrix/investigate_ad.py` show
a ~1.7× speedup for the Hessian, compared to ~1.2× for the Jacobian.

In PyTorch, `torch.linalg.solve` implements implicit differentiation
in its backward pass automatically — `torch.func.hessian` applies the
rule twice through the autograd graph without the user writing any
derivative formulae.

## Earth Model

The default 5-layer ocean-crust model from the original Fortran code:

| Layer | Type | Vp (km/s) | Vs (km/s) | rho (g/cm^3) | h (km) |
|-------|------|-----------|-----------|---------------|--------|
| 0 | Ocean | 1.50 | 0.00 | 1.0 | 2.0 |
| 1 | Sediment | 1.60 | 0.30 | 2.0 | 1.0 |
| 2 | Crust | 3.00 | 1.50 | 3.0 | 1.0 |
| 3 | Upper mantle | 5.00 | 3.00 | 3.0 | 1.0 |
| 4 | Half-space | 2.20 | 1.10 | 1.8 | inf |

## Tests

```bash
# All Kennett tests
conda run -n seismic pytest Kennett_Reflectivity/test_package.py -v
conda run -n seismic pytest Kennett_Reflectivity/test_taup_inversion.py -v

# All GlobalMatrix tests
conda run -n seismic pytest GlobalMatrix/test_gmm.py -v
conda run -n seismic pytest GlobalMatrix/test_gmm_gradients.py -v
```

## References

1. Nestor, T. M. (1991). *A Practical Implementation of Waveform
   Inversion in a Stratified Marine Environment.* Master of Science
   thesis, Monash University.
2. Kormendi, F. & Dietrich, M. (1991). Nonlinear waveform inversion
   of plane-wave seismograms in stratified elastic media. *Geophysics*,
   56(5), 664--674.
3. Dietrich, M. & Kormendi, F. (1990). Perturbation of the plane-wave
   reflectivity of a depth-dependent elastic medium by weak
   inhomogeneities. *Geophys. J. Int.*, 100(2), 203--214.
4. Kennett, B. L. N. (1983). *Seismic Wave Propagation in Stratified
   Media.* Cambridge University Press.
5. Chin, R. C. Y., Hedstrom, G. W. & Thigpen, L. (1984). Matrix
   methods in synthetic seismograms. *Geophys. J. R. astr. Soc.*, 77,
   483--502.
6. Marquardt, D. W. (1963). An algorithm for least-squares estimation
   of nonlinear parameters. *SIAM J. Appl. Math.*, 11(2), 431--441.
7. Aki, K. & Richards, P. G. (1980). *Quantitative Seismology.*
   W. H. Freeman.

## Licence

This repository is for academic and research purposes.
