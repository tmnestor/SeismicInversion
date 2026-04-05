# Kennett_Reflectivity

Synthetic seismogram computation and waveform inversion for plane-stratified
elastic media using Kennett's reflectivity method with PyTorch automatic
differentiation.

## Overview

This package implements three tightly integrated components:

1. **Kennett reflectivity** -- recursive computation of plane-wave
   reflection/transmission coefficients for layered elastic half-spaces using
   Kennett's addition formulae (Kennett, 1983).
2. **Synthetic seismograms** -- single-trace and multi-offset gather
   computation via discrete wavenumber summation, with optional free-surface
   multiples and GPU acceleration.
3. **Tau-p inversion** -- full-Newton Levenberg-Marquardt inversion of
   frequency-domain plane-wave reflectivity data using PyTorch automatic
   differentiation for exact gradient and Hessian computation.

Translated from the Fortran program `kennetslo.f` to Python 3.12 with NumPy
vectorisation, multiprocessing, and PyTorch GPU support.

## Installation

```bash
conda env create -f envs/seismic.yml
conda activate seismic
```

The environment includes Python 3.12, NumPy, SciPy, matplotlib, PyTorch,
PyYAML, ObsPy, and Devito.

## Package Structure

```
Kennett_Reflectivity/
├── __init__.py                 # Package init with lazy imports
├── layer_model.py              # LayerModel dataclass, complex/vertical slowness
├── scattering_matrices.py      # P-SV interfacial reflection/transmission coefficients
├── kennett_reflectivity.py     # Recursive reflectivity (Kennett addition formula)
├── kennett_seismogram.py       # Single-trace synthetic seismogram (fixed slowness)
├── kennett_gather.py           # Multi-offset gather (discrete wavenumber summation)
├── kennett_reflectivity_gpu.py # PyTorch GPU batched reflectivity (MPS/CUDA)
├── kennett_gather_gpu.py       # GPU-accelerated gather computation
├── kennett_torch.py            # Differentiable reflectivity (AD-ready forward model)
├── taup_inversion.py           # Newton-LM inversion in tau-p domain
├── inversion_config.py         # YAML config loader/validator/serializer
├── source.py                   # Source wavelets (Ricker, frequency & time domain)
├── configs/
│   └── default_ocean_crust.yaml  # Default 5-layer model config
├── test_package.py             # Core package tests
└── test_taup_inversion.py      # Inversion convergence tests
```

## CLI Reference

### `kennett_seismogram`

Compute a single synthetic seismogram at a fixed horizontal slowness.

```bash
python -m Kennett_Reflectivity.kennett_seismogram \
    -p 0.2 -T 64 -n 2048 -o seismogram.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `-p`, `--slowness` | 0.2 | Horizontal slowness / ray parameter (s/km) |
| `-T`, `--duration` | 64.0 | Time window (seconds) |
| `-n`, `--nw` | 2048 | Number of positive frequencies (power of 2) |
| `-o`, `--output` | `seismogram_verification.png` | Output plot filename |
| `--no-plot` | off | Skip plot, save data only |
| `--free-surface` | off | Include free surface reflections |

Output: PNG plot + `seismogram_p{slowness}.txt` (two-column ASCII: time, amplitude).

### `kennett_gather`

Multi-offset seismogram gather via discrete wavenumber summation.

```bash
python -m Kennett_Reflectivity.kennett_gather \
    --r-min 0.5 --r-max 20.0 --dr 0.5 \
    -T 64 -n 2048 --np 2048 --p-max 0.8 \
    --t-max 30 -o gather.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--r-min` | 0.5 | Minimum offset (km) |
| `--r-max` | 20.0 | Maximum offset (km) |
| `--dr` | 0.5 | Offset spacing (km) |
| `-T`, `--duration` | 64.0 | Time window (seconds) |
| `-n`, `--nw` | 2048 | Number of positive frequencies (power of 2) |
| `--np` | 2048 | Number of slowness samples |
| `--p-max` | 0.8 | Maximum slowness (s/km) |
| `--gamma` | pi/T | Complex frequency damping (rad/s) |
| `--t-max` | full window | Maximum display time (seconds) |
| `-o`, `--output` | `seismogram_gather.png` | Output plot filename |
| `--no-plot` | off | Skip plot, save data only |
| `-j`, `--workers` | all cores | Number of parallel workers |
| `--free-surface` | off | Include free surface reflections |

Output: PNG plot + `.npz` archive (`time`, `offsets`, `gather`).

### `kennett_gather_gpu`

GPU-accelerated gather (batched reflectivity over all slownesses simultaneously).

```bash
python -m Kennett_Reflectivity.kennett_gather_gpu \
    --r-min 0.5 --r-max 50.0 --dr 0.5 \
    -T 64 -n 2048 --np 8192 --p-max 1.2 \
    --free-surface --t-max 30 -o gather_gpu.png
```

Same flags as `kennett_gather` plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto-detect | Backend: `mps`, `cuda`, or `cpu` |

### `taup_inversion`

Run a tau-p waveform inversion with Newton-Levenberg-Marquardt.

```bash
# Built-in default model
python -m Kennett_Reflectivity.taup_inversion

# From YAML config
python -m Kennett_Reflectivity.taup_inversion \
    --config Kennett_Reflectivity/configs/default_ocean_crust.yaml

# With CLI overrides
python -m Kennett_Reflectivity.taup_inversion \
    --config Kennett_Reflectivity/configs/default_ocean_crust.yaml \
    --max-iter 30 --perturbation 0.10 --seed 99 --output-dir results
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | built-in model | Path to YAML config file |
| `--max-iter` | from config | Override max Newton iterations |
| `--perturbation` | from config | Override initial model perturbation |
| `--seed` | from config | Override random seed |
| `--output-dir` | from config | Override output directory |

The CLI saves an `inversion_config.yaml` to the output directory for
reproducibility. Re-run with `--config figures/inversion_config.yaml` to
reproduce identical results.

## Config File Format

The YAML config specifies the earth model, inversion parameters, and output
settings. Only the `model` section is required; all other fields have defaults.

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
| `model.layers[]` | `name` | string | `Layer {i}` | Layer name (for tables) |
| | `alpha` | float | *required* | P-wave velocity (km/s), > 0 |
| | `beta` | float | *required* | S-wave velocity (km/s), >= 0 |
| | `rho` | float | *required* | Density (g/cm^3), > 0 |
| | `thickness` | float | *required* | Thickness (km), `.inf` for half-space |
| | `Q_alpha` | float | *required* | P-wave quality factor, > 0 |
| | `Q_beta` | float | *required* | S-wave quality factor, > 0 |
| `model` | `fixed_layers` | list[int] | `[0]` | Fixed layers (must be `[0]`) |
| `inversion` | `p_values` | list[float] | 0.05--0.60 by 0.05 | Slowness grid (s/km) |
| | `nfreq` | int | 64 | Number of frequencies |
| | `perturbation` | float | 0.15 | Initial model perturbation fraction |
| | `max_iter` | int | 100 | Maximum Newton iterations |
| | `seed` | int | 42 | Random seed |
| | `tol` | float | 1e-8 | Convergence tolerance |
| `output` | `directory` | string | `figures` | Output directory path |
| | `formats` | list[str] | all four | Subset of: `table`, `profiles`, `traces`, `convergence` |
| `output.trace_display` | `t_max` | float | 15.0 | Maximum display time (s) |
| | `nw` | int | 1024 | Positive frequencies for trace synthesis |

Constraints:
- First layer must be acoustic (`beta: 0`) -- the ocean layer
- Last layer must be a half-space (`thickness: .inf`)
- `fixed_layers` must be `[0]` (only the ocean layer is fixed)
- All velocities, densities, and Q factors must be positive

## Python API

### LayerModel construction

```python
from Kennett_Reflectivity import LayerModel
import numpy as np

model = LayerModel.from_arrays(
    alpha=[1.5, 1.6, 3.0, 5.0, 2.2],
    beta=[0.0, 0.3, 1.5, 3.0, 1.1],
    rho=[1.0, 2.0, 3.0, 3.0, 1.8],
    thickness=[2.0, 1.0, 1.0, 1.0, np.inf],
    Q_alpha=[20000, 100, 100, 100, 100],
    Q_beta=[1e10, 100, 100, 100, 100],
)
```

### Synthetic seismograms

```python
from Kennett_Reflectivity import compute_seismogram, default_ocean_crust_model

model = default_ocean_crust_model()
time, seis = compute_seismogram(model, p=0.2, T=64.0, nw=2048, free_surface=True)
```

### Multi-offset gather

```python
from Kennett_Reflectivity.kennett_gather import compute_gather
import numpy as np

offsets = np.arange(0.5, 50.5, 0.5)
time, offsets, gather = compute_gather(
    model, offsets, T=64.0, nw=2048, np_slow=4096, p_max=1.2, free_surface=True,
)
```

### Running inversion from Python

```python
from Kennett_Reflectivity import invert_taup, default_ocean_crust_model

result = invert_taup(
    true_model=default_ocean_crust_model(),
    p_values=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    nfreq=64,
    perturbation=0.15,
    max_iter=50,
)
print(f"Converged: {result.converged}")
print(f"Iterations: {result.n_iterations}")
print(f"Final error: {result.param_error_history[-1]:.2e}")
```

### Using YAML configs from Python

```python
from Kennett_Reflectivity import load_config, save_config, invert_taup
from pathlib import Path

cfg = load_config(Path("Kennett_Reflectivity/configs/default_ocean_crust.yaml"))
result = invert_taup(
    true_model=cfg.true_model,
    p_values=cfg.p_values,
    nfreq=cfg.nfreq,
    perturbation=cfg.perturbation,
    max_iter=cfg.max_iter,
    seed=cfg.seed,
    tol=cfg.tol,
)
save_config(cfg, Path("results/inversion_config.yaml"))
```

### Differentiable reflectivity

```python
from Kennett_Reflectivity import (
    kennett_reflectivity_torch, forward_model_torch,
    jacobian, hessian, model_to_tensors,
)

tensors = model_to_tensors(model)
J = jacobian(model, p=0.3, omega=omega)   # (2*nfreq, n_params)
H = hessian(model, p=0.3, omega=omega)    # (n_params, n_params)
```

## Default Model

The built-in 5-layer ocean-crust model from `kennetslo.f`:

| Layer | Type | alpha (km/s) | beta (km/s) | rho (g/cm^3) | h (km) | Q_alpha | Q_beta |
|-------|------|-------------|-------------|--------------|--------|---------|--------|
| 0 | Ocean | 1.50 | 0.00 | 1.0 | 2.0 | 20000 | 1e10 |
| 1 | Sediment | 1.60 | 0.30 | 2.0 | 1.0 | 100 | 100 |
| 2 | Crust | 3.00 | 1.50 | 3.0 | 1.0 | 100 | 100 |
| 3 | Upper mantle | 5.00 | 3.00 | 3.0 | 1.0 | 100 | 100 |
| 4 | Half-space | 2.20 | 1.10 | 1.8 | inf | 100 | 100 |

Critical slownesses (head wave thresholds):

| Refractor | alpha (km/s) | p_critical (s/km) |
|-----------|-------------|--------------------|
| Ocean | 1.50 | 0.667 |
| Sediment | 1.60 | 0.625 |
| Crust | 3.00 | 0.333 |
| Upper mantle | 5.00 | 0.200 |

## Theory

### Kennett recursion

The reflectivity of a stack of elastic layers is computed recursively from the
bottom up using Kennett's addition formulae. At each interface, the downgoing
and upgoing wave systems are coupled through 2x2 scattering matrices
(reflection and transmission coefficients) derived from the Zoeppritz
equations. The recursive scheme builds the total response by combining the
interface scattering with phase propagation across each layer, avoiding
numerical instability from exponentially growing evanescent waves.

### Tau-p inversion

The inversion minimises the L2 misfit between observed and predicted
frequency-domain reflectivity R(omega, p) summed over a discrete set of
slowness values. The data domain is the plane-wave (tau-p) domain, which
decouples the slowness dependence and avoids the Bessel function integration
required for offset-domain data.

### Newton-LM with log-parameterisation

The solver uses a full-Newton method with Levenberg-Marquardt damping. Both
the gradient and exact Hessian are computed via PyTorch automatic
differentiation through the Kennett recursion. The inversion operates in
log-parameter space (`log(alpha)`, `log(beta)`, `log(rho)`, `log(h)`) to
enforce positivity and improve Hessian conditioning. The ocean layer is held
fixed; the half-space thickness is excluded from the parameter vector.

## Output Formats

The inversion CLI produces four output types (controlled by `output.formats`):

- **`table`**: LaTeX `booktabs` table comparing true, initial, and recovered
  parameters with per-layer relative errors (`taup_model_parameters.tex`)
- **`profiles`**: TikZ 3-panel depth profiles showing Vp, Vs, and rho for
  all three models (`taup_model_profiles.tex`)
- **`traces`**: PDF with variable-area wiggle comparison of true vs recovered
  traces and residual panel (`taup_trace_comparison.pdf`)
- **`convergence`**: PDF with 3-panel semilogy plots of misfit, gradient norm,
  and relative parameter error (`taup_convergence.pdf`)

## Performance

The gather computation parallelises over slowness using `multiprocessing.Pool`.
Each worker computes reflectivity for one slowness with all frequencies
vectorised via NumPy.

Typical runtimes on Apple Silicon (M-series, 10 cores):

| Configuration | Slowness samples | Frequencies | Approximate time |
|---------------|-----------------|-------------|------------------|
| Quick test | 512 | 511 | ~5s |
| Standard | 2048 | 2047 | ~30s |
| High quality | 4096 | 2047 | ~60s |
| Head waves | 4096 | 2047 | ~60s |

## GPU Acceleration

The PyTorch GPU backend batches the entire Kennett recursion across all
slowness samples simultaneously using tensors of shape
`(np_slow, nfreq, 2, 2)`.

```
CPU:  for each p_j: R[w] = kennett_reflectivity(model, p_j, omega)
GPU:  R[j, w] = kennett_reflectivity_batch(model, p_all, omega)  # single call
```

Supports Apple MPS, NVIDIA CUDA, and CPU fallback (auto-detected). The GPU
path enables much denser slowness sampling (8192--16384 vs 4096 on CPU) which
suppresses late-time aliasing artefacts.

```bash
python -m Kennett_Reflectivity.kennett_gather_gpu \
    --r-min 0.5 --r-max 50.0 --dr 0.5 \
    -T 64 -n 2048 --np 8192 --p-max 1.2 \
    --free-surface --t-max 30 --device mps -o gather_gpu.png
```

## References

1. Kennett, B. L. N. (1983). *Seismic Wave Propagation in Stratified Media.* Cambridge University Press.
2. Aki, K. & Richards, P. G. (1980). *Quantitative Seismology.* W. H. Freeman.
3. Bouchon, M. (1981). A simple method to calculate Green's functions for elastic layered media. *BSSA*, 71(4), 959--971.
4. Chapman, C. H. (2004). *Fundamentals of Seismic Wave Propagation.* Cambridge University Press.
5. Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *SIAM J. Appl. Math.*, 11(2), 431--441.
