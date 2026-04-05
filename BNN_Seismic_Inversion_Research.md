# Bayesian Neural Networks for 1D Plane-Layer Seismic Inversion

## 1. Problem Formulation

The 1D plane-layer model represents the subsurface as **N horizontal, homogeneous, isotropic layers**, each described by P-wave velocity (Vp), S-wave velocity (Vs), density (ρ), and thickness (h). The parameter vector is:

```
m = [Vp₁, Vs₁, ρ₁, h₁, ..., Vpₙ, Vsₙ, ρₙ, hₙ]
```

### Forward Models

| Method | Description |
|--------|-------------|
| **Convolutional model** | `s(t) = w(t) * r(t) + n(t)` where reflectivity `rᵢ = (Zᵢ₊₁ - Zᵢ)/(Zᵢ₊₁ + Zᵢ)` and `Z = Vp·ρ` |
| **Propagator matrix (Haskell-Thomson)** | Matrix chain through layer stack; captures multiples and mode conversions |
| **Reflectivity method (Kennett, 1983)** | Full-waveform in frequency-wavenumber domain; unconditionally stable, includes all multiples |

### Why the Problem Is Ill-Posed

- **Non-uniqueness**: Band-limited wavelets lose information; thin layers create equivalent responses
- **Nonlinearity**: Angle-dependent reflectivity (AVO) makes the mapping highly nonlinear
- **Parameter trade-offs**: Velocity-thickness ambiguity (travel time ∝ velocity × thickness)
- **Limited bandwidth**: Seismic data typically 5–80 Hz, missing both low and high frequencies
- **Noise**: Random + coherent noise (multiples, ground roll)

Any single "best-fit" model is therefore misleading — **uncertainty quantification (UQ)** is essential for identifying which parameters are well-constrained, providing confidence intervals for risk assessment, and supporting decision-making.

---

## 2. BNN Architectures and Bayesian Treatments

### Network Architectures

| Architecture | Use Case | Key References |
|-------------|----------|----------------|
| **Fully-connected (MLP)** | Earliest approach; trace → layer properties | Roth & Tarantola (1994); Li, Grana & Liu (2024) |
| **1D CNN** | Natural for seismic traces as 1D signals | Das et al. (2019) |
| **Encoder-decoder** | Compress data → latent → reconstruct velocity model | Wu & Lin (2018) "InversionNet" |
| **RNN/LSTM** | Sequential depth-ordered layer prediction | Various |
| **U-Net with skip connections** | 2D/3D extensions | Rumpf et al. (2022) |

### Bayesian Inference Methods

**Variational Inference (Bayes by Backprop)** — Blundell et al. (2015): Learn Gaussian distributions over all weights (mean + variance per weight). Minimize ELBO = data fit + KL divergence from prior. Li, Grana & Liu (2024) applied BNN-VI and BPINN-VI (physics-informed variant) to petrophysical inversion; the physics-informed version produced lower uncertainty and better lateral continuity.

**MC Dropout** — Gal & Ghahramani (2016): Dropout at inference time approximates Bayesian inference. Multiple forward passes → mean ≈ posterior mean, variance ≈ epistemic uncertainty. Lightweight but can underestimate uncertainty.

**Deep Ensembles** — Lakshminarayanan et al. (2017): Train M independently initialized networks. Ensemble disagreement captures epistemic uncertainty. Gou et al. (2024) combined ensembles with dropout for seismic traces.

**Mixture Density Networks (MDNs)**: Output parameters of a Gaussian mixture → multimodal posteriors. Zhang et al. (2024) developed a physics-guided deep MDN (PG-DMDN) with lower computational cost than BNNs while capturing multimodality.

**Conditional Normalizing Flows**: Learn invertible mapping from Gaussian to posterior, conditioned on data. Siahkoohi et al. (2021, 2022) at SLIM Lab developed amortized variational Bayesian inference — once trained, instant posterior samples for new observations.

**HMC with Neural Surrogates**: Neural networks as differentiable surrogate forward models enable gradient-based MCMC. Sen & Biswas (2017) combined rjMCMC with HMC for transdimensional inversion.

**Diffusion/Score-Based Models** (2024–2025 frontier): Train unconditional diffusion model, incorporate physics-based likelihood during reverse diffusion. Cao et al. (2025) applied this to acoustic impedance inversion.

---

## 3. Key Papers

### Pioneering

- **Roth & Tarantola (1994)**, *J. Geophys. Res.* — First NN seismic inversion (450 training examples, 8-layer models)

### Deterministic Baselines

- **Das, Pollack, Wollner & Mukerji (2019)**, *Geophysics* 84(6) — 1D CNN for impedance inversion with ABC uncertainty
- **Wu & Lin (2018)**, *arXiv:1811.07875* — InversionNet encoder-decoder
- **Yang & Ma (2019)**, *Geophysics* 84(4) — Fully convolutional velocity model building
- **Torres et al. (2023)**, *Geophysical Prospecting* 71(6) — Deep decomposition learning for reflectivity inversion

### Bayesian / Probabilistic

- **Li, Grana & Liu (2024)**, *Geophysics* 89(6) — BNN-VI vs. BPINN-VI comparison
- **Li et al. (2024)**, *Geophysics* 89(2) — Probabilistic PINN with reparameterization
- **Zhang et al. (2024)**, *Petroleum Science* 21(1) — Physics-guided deep MDN
- **Shahraeeni & Curtis (2011)**, *Geophysics* — Fast probabilistic nonlinear petrophysical inversion
- **Gou et al. (2024)**, *arXiv:2410.06120* — Ensembles + dropout for seismic UQ
- **Fang et al. (2024)**, *arXiv:2409.06840v1* — UQ via importance sampling and ensembles

### Amortized / Simulation-Based Inference

- **Siahkoohi et al. (2020)**, *EAGE / arXiv:2001.04567* — Deep-learning based Bayesian seismic imaging
- **Siahkoohi et al. (2022)**, *EAGE / arXiv:2203.15881* — Amortized variational Bayesian inference with conditional normalizing flows
- **Siahkoohi et al. (2021)**, *SEG Technical Program* — Fast reliability-aware imaging with normalizing flows
- **Spurio Mancini et al. (2025)**, *Geophys. J. Int.* 241(3) — First SBI (neural posterior estimation) in seismology
- **Liao et al. (2025)**, *JGR: ML & Computation* 2(1) — Iterative normalizing flows addressing loss function bias

### Generative Priors

- **Mosser, Dubrule & Blunt (2020)**, *Math. Geosci.* — GAN as geological prior with MCMC in latent space

### Diffusion Models

- **Cao et al. (2025)**, *Geophysics* — Unsupervised diffusion model for impedance inversion
- **Wang et al. (2024)**, *JGR: ML & Computation* — Controllable velocity synthesis via diffusion
- **Li et al. (2025)**, *arXiv:2506.13529* — Conditional latent generative diffusion for impedance inversion

---

## 4. Training Data and Forward Modeling

### Synthetic Data Generation Pipeline

1. **Sample layer models** from prior distributions
2. **Apply rock-physics constraints** (Gardner's relation, Castagna's mudrock line, Greenberg-Castagna)
3. **Compute synthetic seismograms** via reflectivity method, propagator matrices, convolutional model, or finite difference
4. **Add noise** at various SNR levels (band-limited random, correlated, field-derived)

### Common Prior Distributions

| Parameter | Typical Prior |
|-----------|--------------|
| **Vp** | Uniform or log-uniform, 1500–5500 m/s (sometimes depth-dependent) |
| **Density** | Linked to Vp via Gardner's relation (`ρ = a·Vp^b`) + perturbation, or uniform 1.5–3.0 g/cm³ |
| **Thickness** | Uniform or log-uniform, 5–100 m per layer |
| **Number of layers** | Uniform (e.g., 2–50) for transdimensional methods |
| **Wavelet** | Ricker (25–60 Hz dominant frequency) or Ormsby band-pass |

### Dataset Sizes

- Roth & Tarantola (1994): 450 training examples
- Torres et al. (2023): 5,000 samples with 60 Hz Ricker wavelet
- Modern studies: 10,000–1,000,000+ synthetic examples

---

## 5. Comparison with Traditional Bayesian Methods

| Aspect | MCMC / rjMCMC | BNN (Variational) | MC Dropout / Ensembles | Normalizing Flows |
|--------|--------------|-------------------|----------------------|-------------------|
| **Posterior accuracy** | Asymptotically exact | Approximate (limited by variational family) | Approximate | Flexible, but trained approximation |
| **Inference cost** | Very high (re-run per dataset) | Low (single pass + sampling) | Low (M forward passes) | Very low (single pass) |
| **Training cost** | N/A | High (one-time) | Moderate | High (one-time) |
| **Multimodality** | Captured (if mixing is good) | Often missed (mean-field VI) | Limited | Well-captured |
| **Transdimensional** | Natural (rjMCMC) | Difficult | Difficult | Possible but complex |
| **Amortized** | No | Yes | Partial | Yes |
| **Scalability** | Poor (>100 params is hard) | Good | Good | Good |

**Key insight**: BNNs trade posterior exactness for massive speedup at inference — milliseconds vs. hours/days. This is critical for trace-by-trace inversion across 3D surveys.

**Hybrid approaches**: Neural surrogates replace expensive forward models within MCMC, providing gradients for HMC while maintaining sampling rigor.

---

## 6. Practical Considerations

### Noise Handling

- **Training with noise augmentation**: Adding realistic noise (random, correlated, field-derived) to synthetic training data is essential. A 2025 study in *Geophysical Journal International* demonstrated that training with field-noise-augmented data significantly improves generalization.
- **Aleatoric uncertainty modeling**: Networks can learn to predict observation noise variance as an output (heteroscedastic noise models), separating data uncertainty from model uncertainty.

### Epistemic vs. Aleatoric Uncertainty Decomposition

- **Aleatoric uncertainty**: Irreducible, arises from noise in the data. Captured by predicting variance as a network output or through MDN mixture parameters.
- **Epistemic uncertainty**: Reducible with more data, arises from limited training data or model capacity. Captured by weight uncertainty (BNN-VI), MC dropout variance, or ensemble disagreement.
- **Practical approach**: Use the decomposition to identify regions where more data would help (high epistemic) vs. regions where uncertainty is intrinsic (high aleatoric).

### Out-of-Distribution Detection

- Neural networks can produce confident but incorrect predictions when input data falls outside the training distribution.
- BNNs provide a natural OOD detection mechanism: epistemic uncertainty increases for OOD inputs.
- Deep ensembles and MC dropout provide analogous uncertainty inflation for OOD data.
- Always validate that the training data prior covers the expected range of subsurface properties in the target area.

### Physics-Informed Constraints

- Embedding physical constraints (wave equation, rock physics relations) into the loss function improves inversion quality and reduces dependence on large training datasets.
- BPINN-VI (Li, Grana & Liu, 2024) demonstrated that adding physics constraints via KL divergence terms yields lower uncertainty and better lateral continuity than pure data-driven BNNs.

### Dimensionality

- 1D plane-layer problems are relatively low-dimensional (typically 4N parameters for N layers), making them tractable for most methods.
- Ideal testbed for comparing BNN posteriors against exact MCMC posteriors.
- For high-dimensional problems (2D/3D), dimensionality reduction (DCT, PCA, autoencoders) is essential.

---

## 7. Open-Source Implementations

| Repository | Description |
|-----------|-------------|
| [SLIM Lab - Bayesian DL Imaging](https://github.com/slimgroup/Software.siahkoohi2020EAGEdlb) | CNN + stochastic gradient Langevin dynamics (PyTorch + Devito) |
| [SLIM Lab - Conditional NFs](https://github.com/slimgroup/ConditionalNFs4Imaging.jl) | Amortized variational Bayesian inference (Julia) |
| [CNN Impedance Inversion + ABC](https://github.com/vishaldas/CNN_based_impedance_inversion) | CNN with approximate Bayesian computation UQ |
| [sbi](https://github.com/sbi-dev/sbi) | General simulation-based inference framework (NPE, NRE) — directly applicable |
| [Kennett reflectivity method](https://github.com/samhaug/reflectivity_method) | Forward modeling for training data generation |
| [FCNVMB](https://github.com/YangFangShu/FCNVMB-Deep-learning-based-seismic-velocity-model-building) | Deterministic FCN baseline for velocity building |
| [nbouziani/seismic-inversion](https://github.com/nbouziani/seismic-inversion) | NN-regularized seismic inversion with Firedrake |
| [RJ_MCMC](https://github.com/alistairboyce11/RJ_MCMC) | Transdimensional inversion baseline (rjMCMC) |
| [MDN for uncertainty](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation) | Generic MDN implementation (Keras/TF) |
| [Awesome Open Geoscience](https://github.com/softwareunderground/awesome-open-geoscience) | Curated list of geoscience tools |

---

## 8. State of the Art Summary

**Mature (2019–2023)**: Deterministic CNNs + MC Dropout/ensembles for lightweight UQ. MDNs for multimodal posteriors.

**Current mainstream (2023–2025)**: BNN-VI and BPINN-VI with physics constraints. Conditional normalizing flows for amortized posterior estimation.

**Emerging frontier (2024–2026)**: Diffusion models for flexible posterior sampling. Simulation-based inference entering geophysics. Iterative normalizing flows addressing amortization bias.

**Key opportunity**: The 1D plane-layer problem is a natural testbed where BNN posteriors can be validated against exact MCMC — combining BNNs with the reflectivity method for full-waveform 1D inversion with rigorous UQ remains an active area with room for contribution.
