# Kennett_Reflectivity Package - File Index

## Quick Navigation

### Core Package Files

| File | Purpose | Key Items |
|------|---------|-----------|
| `__init__.py` | Package initialization | `__version__`, `__all__`, all exports |
| `layer_model.py` | Stratified media model | `LayerModel`, `complex_slowness()`, `vertical_slowness()` |
| `scattering_matrices.py` | Interface coefficients | `ScatteringCoefficients`, `solid_solid_interface()`, `ocean_bottom_interface()` |
| `kennett_reflectivity.py` | Core algorithm | `kennett_reflectivity()`, `inv2x2()` |
| `source.py` | Wavelets | `ricker_spectrum()`, `ricker_wavelet()` |
| `kennett_seismogram.py` | Orchestrator | `compute_seismogram()`, `default_ocean_crust_model()` |

### Documentation & Examples

| File | Purpose | Contains |
|------|---------|----------|
| `README.md` | Full documentation | API reference, math, examples, references |
| `test_package.py` | Test suite | 4 test groups, comprehensive coverage |
| `example_usage.py` | Usage examples | 6 examples showing all features |
| `INDEX.md` | This file | File navigation |

## Module Dependency Graph

```
__init__.py
├── layer_model.py
│   └── (numpy)
├── scattering_matrices.py
│   ├── layer_model.py
│   └── (numpy)
├── kennett_reflectivity.py
│   ├── layer_model.py
│   ├── scattering_matrices.py
│   └── (numpy)
├── source.py
│   └── (numpy)
└── kennett_seismogram.py
    ├── layer_model.py
    ├── kennett_reflectivity.py
    ├── source.py
    └── (numpy)
```

## Function Reference

### Layer Model (`layer_model.py`)

```python
# Functions
complex_slowness(velocity: float, Q: float) -> complex
vertical_slowness(slowness: complex, p: float) -> complex

# Classes
LayerModel
  .alpha: ndarray
  .beta: ndarray
  .rho: ndarray
  .thickness: ndarray
  .Q_alpha: ndarray
  .Q_beta: ndarray
  .n_layers: int (property)
  .complex_slowness_p() -> ndarray
  .complex_slowness_s() -> ndarray
  .from_arrays(...) (classmethod)
```

### Scattering Matrices (`scattering_matrices.py`)

```python
# Functions
solid_solid_interface(p, eta1, neta1, rho1, beta1, eta2, neta2, rho2, beta2) 
  -> ScatteringCoefficients
ocean_bottom_interface(p, eta1, rho1, eta2, neta2, rho2, beta2) 
  -> ScatteringCoefficients

# Classes
ScatteringCoefficients
  .Rd: ndarray (2x2)
  .Ru: ndarray (2x2)
  .Tu: ndarray (2x2)
  .Td: ndarray (2x2)
```

### Kennett Reflectivity (`kennett_reflectivity.py`)

```python
# Functions
inv2x2(M: ndarray) -> ndarray
kennett_reflectivity(model: LayerModel, p: float, omega: ndarray) -> ndarray
```

### Source Wavelets (`source.py`)

```python
# Functions
ricker_spectrum(omega: ndarray, omega_max: float) -> ndarray
ricker_wavelet(t: ndarray, f_peak: float) -> ndarray
```

### Seismogram (`kennett_seismogram.py`)

```python
# Functions
compute_seismogram(model, p, T=64.0, nw=2048, eps=0.0, source_func=None)
  -> tuple[ndarray, ndarray]
default_ocean_crust_model() -> LayerModel
```

## Common Workflows

### Workflow 1: Quick Test with Default Model

```python
from Kennett_Reflectivity import default_ocean_crust_model, compute_seismogram

model = default_ocean_crust_model()
time, seismogram = compute_seismogram(model, p=0.2)
```

### Workflow 2: Create Custom Model

```python
from Kennett_Reflectivity import LayerModel
import numpy as np

model = LayerModel.from_arrays(
    alpha=[2.0, 3.0, 4.0],
    beta=[1.0, 1.5, 2.0],
    rho=[2.5, 2.7, 2.9],
    thickness=[1.0, 2.0, np.inf],
    Q_alpha=[100, 100, 100],
    Q_beta=[50, 50, 50],
)
```

### Workflow 3: Compute Reflectivity

```python
from Kennett_Reflectivity import kennett_reflectivity, default_ocean_crust_model
import numpy as np

model = default_ocean_crust_model()
omega = np.linspace(0.1, 10.0, 512)
R = kennett_reflectivity(model, p=0.2, omega=omega)
```

### Workflow 4: Analyze Scattering

```python
from Kennett_Reflectivity import (
    ocean_bottom_interface, 
    complex_slowness, 
    vertical_slowness
)

p = 0.2
eta1 = vertical_slowness(complex_slowness(1.5, 20000), p)
eta2 = vertical_slowness(complex_slowness(1.6, 100), p)
neta2 = vertical_slowness(complex_slowness(0.3, 100), p)

coeff = ocean_bottom_interface(
    p=p, eta1=eta1, rho1=1.0, 
    eta2=eta2, neta2=neta2, rho2=2.0, beta2=0.3
)
print(f"Reflection: {coeff.Rd}")
```

## Testing

### Run Full Test Suite

```bash
python3 test_package.py
```

### Run Examples

```bash
python3 example_usage.py
```

### Test Individual Module

```python
# In Python:
from Kennett_Reflectivity import LayerModel, default_ocean_crust_model
model = default_ocean_crust_model()
print(f"Model OK: {model.n_layers} layers")
```

## Documentation

### Getting Started
- See `README.md` for installation and quick start

### API Reference
- Each module has comprehensive docstrings
- Use `help()` in Python: `help(function_name)`

### Theory
- See `README.md` "Mathematical Details" section
- References to Kennett, Aki & Richards in README

### Examples
- See `example_usage.py` for 6 detailed examples
- See `test_package.py` for test cases showing usage

## Performance Tips

1. **Reuse model objects** - create once, use many times
2. **Vectorize slowness computations** - compute many p values together
3. **Cache scattering matrices** - they depend only on p, not frequency
4. **Use reasonable frequency resolution** - 512 samples is good default

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import fails | Add package parent to sys.path |
| Type errors | Check input types match function signatures |
| NaN in output | Check model parameters are physical (positive, finite) |
| Slow computation | Reduce `nw` parameter or use fewer slownesses |

## Version Info

- **Version:** 1.0.0
- **Python:** 3.12+
- **NumPy:** 1.20+
- **Created:** February 28, 2026
- **Status:** Production-ready

## Quick Links

- Full documentation: `README.md`
- Usage examples: `example_usage.py`
- Tests: `test_package.py`
- Default model: `default_ocean_crust_model()` in `kennett_seismogram.py`

---

For additional help, consult the docstrings in each module:
```python
import module_name
help(module_name.function_name)
```
