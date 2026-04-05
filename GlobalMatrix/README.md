# GlobalMatrix

Plane-wave reflectivity computation for stratified elastic media using the
Global Matrix Method with PyTorch automatic differentiation.

## Overview

This package implements the Global Matrix Method (GMM) as an alternative to
Kennett's recursive reflectivity. The GMM assembles all interface continuity
conditions into a single dense linear system **G x = b** and solves for wave
amplitudes directly.

1. **Forward model** -- assembles the displacement-stress eigenvector matrices,
   builds the global system, and extracts PP reflectivity via `np.linalg.solve`
   (NumPy) or `torch.linalg.solve` (PyTorch).
2. **Differentiable model** -- the PyTorch version is fully differentiable via
   `torch.autograd`. Implicit differentiation through `torch.linalg.solve`
   makes backward passes cheaper than walking the full Kennett recursive graph.
3. **Jacobian and Hessian** -- computed via `torch.func.jacrev` (vectorized
   reverse-mode AD) and `torch.func.hessian`, yielding 10--14x speedup over
   the loop-based `torch.autograd.functional.jacobian`.

### Why GMM alongside Kennett?

Both methods compute identical reflectivity (validated to < 1e-12). The
tradeoffs:

| | Kennett | GMM |
|---|---|---|
| Forward speed | Faster (recursive 2x2 ops) | ~1.3x slower (15x15 solve) |
| Jacobian (torch.func) | Baseline | ~1.2x faster |
| Hessian (torch.func) | Baseline | ~1.7x faster |
| Memory | O(nfreq) | O(nfreq x N^2) |
| Extensibility | Fixed recursion | Easy to add new physics |

The GMM advantage grows for larger models where the backward pass through
the recursive graph becomes the bottleneck.

### Convention

All formulas use the **exp(-iwt)** inverse Fourier transform convention,
matching the Kennett implementation. Chin, Hedstrom & Thigpen (1984) use the
conjugate convention exp(+iwt); the E-matrices and phase factors here are
adapted accordingly.

## Installation

Uses the same conda environment as `Kennett_Reflectivity`:

```bash
conda env create -f envs/seismic.yml
conda activate seismic
```

## Package Structure

```
GlobalMatrix/
├── __init__.py               # Package init with lazy imports
├── layer_matrix.py           # Displacement-stress E-matrices (NumPy + PyTorch)
├── global_matrix.py          # System assembly and solve (NumPy)
├── gmm_torch.py              # Differentiable version + Jacobian/Hessian (PyTorch)
├── gmm_reflectivity_cli.py   # CLI entry point
├── config.py                 # YAML config loader/validator/serializer
├── configs/
│   └── default_ocean_crust.yaml  # Default 5-layer model config
├── test_gmm.py               # Forward validation tests (GMM vs Kennett)
└── test_gmm_gradients.py     # Derivative validation tests
```

## CLI Reference

### `gmm_reflectivity_cli`

Compute GMM reflectivity and optionally compare with Kennett.

```bash
python -m GlobalMatrix.gmm_reflectivity_cli \
    -p 0.1 0.2 0.3 -n 256 -o reflectivity.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | built-in model | Path to YAML config file |
| `-p`, `--slowness` | from config | Slowness values (s/km), space-separated |
| `-n`, `--nfreq` | 256 | Number of frequencies |
| `--free-surface` | off | Include free surface reflections |
| `--no-compare` | off | Skip Kennett comparison |
| `-o`, `--output` | `gmm_reflectivity.png` | Output plot filename |
| `--no-plot` | off | Skip plot generation |
| `--output-dir` | from config | Override output directory |

Output: PNG plot + `.npz` data files per slowness + `gmm_config.yaml` for
reproducibility.

### Examples

```bash
# Default model, compare with Kennett
python -m GlobalMatrix.gmm_reflectivity_cli

# From YAML config
python -m GlobalMatrix.gmm_reflectivity_cli \
    --config GlobalMatrix/configs/default_ocean_crust.yaml

# Single slowness, no comparison
python -m GlobalMatrix.gmm_reflectivity_cli -p 0.2 --no-compare

# Free surface, high frequency
python -m GlobalMatrix.gmm_reflectivity_cli \
    -p 0.1 0.2 0.3 0.4 0.6 -n 1024 --free-surface
```

## Config File Format

The YAML config specifies the earth model, computation parameters, and output
settings. Only the `model` section is required.

### Minimal example

```yaml
model:
  layers:
    - {alpha: 1.5, beta: 0.0, rho: 1.0, thickness: 2.0, Q_alpha: 20000, Q_beta: 1.0e+10}
    - {alpha: 3.0, beta: 1.5, rho: 3.0, thickness: 1.0, Q_alpha: 100, Q_beta: 100}
    - {alpha: 5.0, beta: 3.0, rho: 3.0, thickness: .inf, Q_alpha: 100, Q_beta: 100}
```

### Full example

See `configs/default_ocean_crust.yaml` for a complete annotated config.

### Schema reference

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| `model.layers[]` | `name` | string | `Layer {i}` | Layer name |
| | `alpha` | float | *required* | P-wave velocity (km/s), > 0 |
| | `beta` | float | *required* | S-wave velocity (km/s), >= 0 |
| | `rho` | float | *required* | Density (g/cm^3), > 0 |
| | `thickness` | float | *required* | Thickness (km), `.inf` for half-space |
| | `Q_alpha` | float | *required* | P-wave quality factor, > 0 |
| | `Q_beta` | float | *required* | S-wave quality factor, > 0 |
| `model` | `fixed_layers` | list[int] | `[0]` | Fixed layers (ocean) |
| `computation` | `p_values` | list[float] | `[0.1, ..., 0.6]` | Slowness values (s/km) |
| | `nfreq` | int | 256 | Number of frequencies |
| | `free_surface` | bool | false | Include free-surface multiples |
| `output` | `directory` | string | `figures` | Output directory |
| | `formats` | list[str] | all | Subset of: `reflectivity`, `comparison` |

Constraints:
- First layer must be acoustic (`beta: 0`) -- the ocean layer
- Last layer must be a half-space (`thickness: .inf`)

## Python API

### GMM reflectivity (NumPy)

```python
from GlobalMatrix import gmm_reflectivity
from Kennett_Reflectivity import default_ocean_crust_model
import numpy as np

model = default_ocean_crust_model()
omega = np.linspace(0.1, 25.0, 255)
R = gmm_reflectivity(model, p=0.2, omega=omega, free_surface=False)
```

### Differentiable reflectivity (PyTorch)

```python
from GlobalMatrix import gmm_reflectivity_torch, gmm_jacobian, gmm_hessian
from Kennett_Reflectivity import model_to_tensors, default_ocean_crust_model
import torch

model = default_ocean_crust_model()
tensors = model_to_tensors(model, requires_grad=True)
omega = torch.linspace(0.1, 25.0, 255, dtype=torch.float64)

R = gmm_reflectivity_torch(
    tensors["alpha"], tensors["beta"], tensors["rho"],
    tensors["thickness"], tensors["Q_alpha"], tensors["Q_beta"],
    p=0.2, omega=omega,
)

J = gmm_jacobian(
    tensors["alpha"], tensors["beta"], tensors["rho"],
    tensors["thickness"], tensors["Q_alpha"], tensors["Q_beta"],
    p=0.2, omega=omega,
)  # shape: (nfreq, n_params)

H = gmm_hessian(
    tensors["alpha"], tensors["beta"], tensors["rho"],
    tensors["thickness"], tensors["Q_alpha"], tensors["Q_beta"],
    p=0.2, omega=omega,
)  # shape: (n_params, n_params)
```

### Using YAML configs from Python

```python
from GlobalMatrix import load_config, save_config, compute_and_compare
from pathlib import Path

cfg = load_config(Path("GlobalMatrix/configs/default_ocean_crust.yaml"))
results = compute_and_compare(cfg)

for p, entry in results.items():
    print(f"p={p}: max|R| = {abs(entry['R_gmm']).max():.4f}")

save_config(cfg, Path("output/gmm_config.yaml"))
```

## Mathematical Formulation

### Dependent variables

Per layer, referenced to avoid exponential growth:
- **Downgoing** amplitudes referenced at layer **top** (phase decays downward)
- **Upgoing** amplitudes referenced at layer **bottom** (phase decays upward)

| Layer | Unknowns | Count |
|-------|----------|-------|
| Ocean (acoustic) | U_0^P (D_0^P = 1 specified) | 1 |
| Elastic finite | D_j^P, D_j^S, U_j^P, U_j^S | 4 |
| Half-space | D_m^P, D_m^S (no upgoing) | 2 |

For the 5-layer model: 1 + 4 + 4 + 4 + 2 = **15 unknowns**.

### E-matrices

For solid layer j with vertical P-slowness eta, S-slowness nu, density rho,
complex shear modulus mu = rho * beta^2, and gamma = rho(1 - 2 beta^2 p^2):

```
E_d = [[p,      nu      ],     E_u = [[p,       -nu     ],
       [eta,    -p      ],            [-eta,     -p     ],
       [-gamma,  2*mu*p*nu],          [-gamma,  -2*mu*p*nu],
       [2*mu*p*eta, gamma]]           [-2*mu*p*eta, gamma]]
```

Rows: [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)]. Columns: [P, S].

### Efficient derivatives (GMM advantage)

From **G x = b**, differentiating w.r.t. parameter m_j:

```
dx/dm_j = G^{-1} * (db/dm_j - dG/dm_j * x)
```

G is factored **once** (from forward solve) and reused for all parameters.
dG/dm_j is extremely sparse: nonzero only in the block-rows bounding the
layer that m_j belongs to.

## Performance

### Jacobian (nfreq=127, 5-layer model, Apple Silicon)

| Method | Kennett | GMM | Speedup |
|--------|--------:|----:|--------:|
| `autograd.functional.jacobian` | 1035 ms | 835 ms | 1.24x |
| `torch.func.jacrev` | 89 ms | 94 ms | 0.95x |

The dominant optimisation is switching from `autograd.functional.jacobian` to
`torch.func.jacrev` (**11x speedup**), not the physics method.

### Hessian (nfreq=31, 5-layer model)

| Method | Kennett | GMM | Speedup |
|--------|--------:|----:|--------:|
| `autograd.functional.hessian` | 310 ms | 182 ms | 1.70x |
| `torch.func.hessian` | 70 ms | 53 ms | 1.32x |

## Tests

```bash
# Forward validation (GMM vs Kennett)
conda run -n seismic pytest GlobalMatrix/test_gmm.py -v

# Gradient validation (GMM Jacobian/Hessian vs Kennett AD)
conda run -n seismic pytest GlobalMatrix/test_gmm_gradients.py -v

# All GlobalMatrix tests
conda run -n seismic pytest GlobalMatrix/ -v --ignore=GlobalMatrix/bench_ad.py \
    --ignore=GlobalMatrix/investigate_ad.py \
    --ignore=GlobalMatrix/investigate_vmap.py \
    --ignore=GlobalMatrix/investigate_func_hessian.py
```

## References

1. Chin, R. C. Y., Hedstrom, G. W., & Thigpen, L. (1984). Matrix methods in
   synthetic seismograms. *Geophys. J. R. astr. Soc.*, 77, 483--502.
2. Kennett, B. L. N. (1983). *Seismic Wave Propagation in Stratified Media.*
   Cambridge University Press.
3. Aki, K. & Richards, P. G. (1980). *Quantitative Seismology.* W. H. Freeman.
