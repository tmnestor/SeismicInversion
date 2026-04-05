# Fréchet Derivatives of the Kennett Reflectivity

## Independent re-derivation of Dietrich & Kormendi (1990)

The PP reflectivity of a stratified elastic half-space is:

$$R(\omega, p) = e_{a,0}^2 \cdot \mathbf{RR}_d^{(0)}[0,0]$$

where $e_{a,0}^2 = \exp(2i\omega\eta_0 h_0)$ is the two-way ocean phase and
$\mathbf{RR}_d^{(0)}$ is the cumulative downgoing reflection matrix at the ocean
bottom, computed by Kennett's upward recursion.

The Fréchet derivative $\partial R / \partial m_j$ for any model parameter $m_j$
is computed by **tangent-linear differentiation** of the entire computation chain.

---

## 1. Kennett Addition Formula

At interface $i$ (between layers $i$ and $i{+}1$), the cumulative reflection
from below is updated by:

$$\mathbf{RR}_d^{(i)} = \mathbf{R}_d^{(i)} + \mathbf{T}_u^{(i)} \, \mathbf{M} \, \mathbf{U} \, \mathbf{T}_d^{(i)}$$

where:

$$\mathbf{M} = \mathbf{E}_{i+1} \, \mathbf{RR}_d^{(i+1)} \, \mathbf{E}_{i+1}, \qquad \mathbf{U} = \bigl(\mathbf{I} - \mathbf{R}_u^{(i)} \mathbf{M}\bigr)^{-1}$$

and $\mathbf{E}_j = \mathrm{diag}(e_{a,j},\, e_{b,j})$ with $e_{a,j} = \exp(i\omega\eta_j h_j)$, $e_{b,j} = \exp(i\omega\nu_j h_j)$.

Initialisation: $\mathbf{RR}_d^{(N-1)} = \mathbf{0}$ (radiation condition at the half-space).

---

## 2. Tangent-Linear Kennett Recursion

Differentiating the addition formula, the tangent $\delta\mathbf{RR}_d$ satisfies:

$$\boxed{\delta\mathbf{RR}_d^{(i)} = \delta\mathbf{R}_d + \delta\mathbf{T}_u \, \mathbf{Z}\,\mathbf{T}_d + \mathbf{T}_u \, \delta\mathbf{Z} \, \mathbf{T}_d + \mathbf{T}_u \, \mathbf{Z} \, \delta\mathbf{T}_d}$$

where $\mathbf{Z} = \mathbf{M}\,\mathbf{U}$ and:

$$\delta\mathbf{Z} = \delta\mathbf{M}\,\mathbf{U} + \mathbf{M}\,\delta\mathbf{U}$$

$$\delta\mathbf{U} = \mathbf{U}\bigl(\delta\mathbf{R}_u\,\mathbf{M} + \mathbf{R}_u\,\delta\mathbf{M}\bigr)\mathbf{U}$$

$$\delta\mathbf{M} = \delta\mathbf{E}\,\mathbf{RR}_d\,\mathbf{E} + \mathbf{E}\,\delta\mathbf{RR}_d\,\mathbf{E} + \mathbf{E}\,\mathbf{RR}_d\,\delta\mathbf{E}$$

The $\delta\mathbf{U}$ formula follows from $d(\mathbf{W}^{-1}) = \mathbf{W}^{-1}\,d\mathbf{W}\,\mathbf{W}^{-1}$ applied to $\mathbf{W} = \mathbf{I} - \mathbf{R}_u\mathbf{M}$.

Initialisation: $\delta\mathbf{RR}_d^{(N-1)} = \mathbf{0}$.

---

## 3. Primitive Derivative Chains

For a perturbation of parameter $m_j$ in layer $j$, the tangent inputs come from the following chains:

### P-wave velocity $\alpha_j$

$$\frac{\partial s_{p,j}}{\partial \alpha_j} = -\frac{s_{p,j}}{\alpha_j}$$

$$\frac{\partial \eta_j}{\partial \alpha_j} = \frac{s_{p,j}}{\eta_j} \cdot \frac{\partial s_{p,j}}{\partial \alpha_j} = -\frac{s_{p,j}^2}{\eta_j \, \alpha_j}$$

$$\frac{\partial e_{a,j}}{\partial \alpha_j} = i\omega h_j \, \frac{\partial \eta_j}{\partial \alpha_j} \, e_{a,j}$$

### S-wave velocity $\beta_j$

$$\frac{\partial s_{s,j}}{\partial \beta_j} = -\frac{s_{s,j}}{\beta_j}, \qquad
\frac{\partial \nu_j}{\partial \beta_j} = -\frac{s_{s,j}^2}{\nu_j \, \beta_j}, \qquad
\frac{\partial \tilde\beta_j}{\partial \beta_j} = \frac{\tilde\beta_j}{\beta_j}$$

$$\frac{\partial e_{b,j}}{\partial \beta_j} = i\omega h_j \, \frac{\partial \nu_j}{\partial \beta_j} \, e_{b,j}$$

where $\tilde\beta_j = 1/s_{s,j}$ is the complex S-velocity entering the scattering matrices.

### Density $\rho_j$

Enters the scattering matrices directly. No effect on phase factors.

### Thickness $h_j$

$$\frac{\partial e_{a,j}}{\partial h_j} = i\omega\eta_j \, e_{a,j}, \qquad
\frac{\partial e_{b,j}}{\partial h_j} = i\omega\nu_j \, e_{b,j}$$

No effect on scattering matrices.

### Proof of $\partial s/\partial v = -s/v$

From the Futterman attenuation model:

$$s = \frac{2Q^2}{(1+4Q^2)\,v} + \frac{2Qi}{(1+4Q^2)\,v} = \frac{2Q^2 + 2Qi}{(1+4Q^2)\,v}$$

Since $Q$ is fixed, $\partial s/\partial v = -s/v$. $\square$

### Proof of $\partial\eta/\partial s = s/\eta$

From $\eta = \sqrt{(s+p)(s-p)} = \sqrt{s^2 - p^2}$:

$$\frac{\partial\eta}{\partial s} = \frac{2s}{2\eta} = \frac{s}{\eta}$$

This holds for either sign of $\eta$ (the branch-cut flip $\eta \to -\eta$ does not affect $s/\eta$ since both numerator and denominator transform consistently). $\square$

---

## 4. Scattering Matrix Derivatives (Solid-Solid)

Define the Kennett intermediate variables at interface $i$ between layers with properties $(\eta_1, \nu_1, \rho_1, \tilde\beta_1)$ above and $(\eta_2, \nu_2, \rho_2, \tilde\beta_2)$ below:

$$\mu_k = \rho_k \tilde\beta_k^2, \quad \Delta\mu = \mu_2 - \mu_1, \quad d = 2\Delta\mu$$

$$a = \Delta\rho - p^2 d, \quad b = \rho_2 - p^2 d, \quad c = \rho_1 + p^2 d$$

$$E = b\eta_1 + c\eta_2, \quad F = b\nu_1 + c\nu_2$$

$$G = a - d\eta_1\nu_2, \quad H = a - d\eta_2\nu_1$$

$$D = EF + GHp^2$$

After normalisation $\hat{E} = E/D$ etc., the scattering elements are:

$$Q = (b\eta_1 - c\eta_2)\hat{F}, \quad R = (a + d\eta_1\nu_2)\hat{H}p^2$$

$$S = (ab + cd\eta_2\nu_2)\,p/D, \quad T = (b\nu_1 - c\nu_2)\hat{E}$$

$$U = (a + d\eta_2\nu_1)\hat{G}p^2, \quad V = (ac + bd\eta_1\nu_1)\,p/D$$

### Tangent of the intermediate variables

For a perturbation $(\delta\eta_1, \delta\nu_1, \delta\rho_1, \delta\tilde\beta_1, \delta\eta_2, \delta\nu_2, \delta\rho_2, \delta\tilde\beta_2)$:

$$\delta\mu_k = \delta\rho_k\,\tilde\beta_k^2 + 2\rho_k\tilde\beta_k\,\delta\tilde\beta_k$$

$$\delta d = 2(\delta\mu_2 - \delta\mu_1)$$

$$\delta a = \delta(\Delta\rho) - p^2\,\delta d, \quad \delta b = \delta\rho_2 - p^2\,\delta d, \quad \delta c = \delta\rho_1 + p^2\,\delta d$$

$$\delta E = \delta b\,\eta_1 + b\,\delta\eta_1 + \delta c\,\eta_2 + c\,\delta\eta_2$$

$$\delta F = \delta b\,\nu_1 + b\,\delta\nu_1 + \delta c\,\nu_2 + c\,\delta\nu_2$$

$$\delta G = \delta a - \delta d\,\eta_1\nu_2 - d\,\delta\eta_1\,\nu_2 - d\,\eta_1\,\delta\nu_2$$

$$\delta H = \delta a - \delta d\,\eta_2\nu_1 - d\,\delta\eta_2\,\nu_1 - d\,\eta_2\,\delta\nu_1$$

$$\delta D = \delta E\,F + E\,\delta F + (\delta G\,H + G\,\delta H)p^2$$

### Tangent of the normalised variables

Using $\delta(X/D) = (\delta X \cdot D - X \cdot \delta D)/D^2$:

$$\delta\hat{E} = \frac{\delta E \cdot D - E \cdot \delta D}{D^2}, \quad \text{etc. for } \hat{F}, \hat{G}, \hat{H}$$

### Tangent of the scattering elements

$$\delta Q = \delta(b\eta_1 - c\eta_2)\,\hat{F} + (b\eta_1 - c\eta_2)\,\delta\hat{F}$$

$$\delta R = \delta(a + d\eta_1\nu_2)\,\hat{H}p^2 + (a + d\eta_1\nu_2)\,\delta\hat{H}\,p^2$$

$$\delta S = \frac{\delta(ab + cd\eta_2\nu_2)\cdot D - (ab + cd\eta_2\nu_2)\cdot\delta D}{D^2}\,p$$

$$\delta T = \delta(b\nu_1 - c\nu_2)\,\hat{E} + (b\nu_1 - c\nu_2)\,\delta\hat{E}$$

$$\delta U = \delta(a + d\eta_2\nu_1)\,\hat{G}p^2 + (a + d\eta_2\nu_1)\,\delta\hat{G}\,p^2$$

$$\delta V = \frac{\delta(ac + bd\eta_1\nu_1)\cdot D - (ac + bd\eta_1\nu_1)\cdot\delta D}{D^2}\,p$$

### Tangent of the scattering matrices

Using $\hat\eta_k = \sqrt{\eta_k}$, $\hat\nu_k = \sqrt{\nu_k}$, $\hat\rho_k = \sqrt{\rho_k}$, $\hat{z}_{a,k} = \hat\eta_k\hat\rho_k$, $\hat{z}_{b,k} = \hat\nu_k\hat\rho_k$:

$$\delta\mathbf{R}_d = \begin{pmatrix} \delta Q - \delta R & -2i(\delta\hat\eta_1\,\hat\nu_1 + \hat\eta_1\,\delta\hat\nu_1)S -2i\hat\eta_1\hat\nu_1\,\delta S \\ \text{sym} & \delta T - \delta U \end{pmatrix}$$

$$\delta\mathbf{T}_d = \begin{pmatrix} 2(\delta\hat{z}_{a1}\hat{z}_{a2} + \hat{z}_{a1}\delta\hat{z}_{a2})\hat{F} + 2\hat{z}_{a1}\hat{z}_{a2}\,\delta\hat{F} & \cdots \\ \cdots & 2(\delta\hat{z}_{b1}\hat{z}_{b2} + \hat{z}_{b1}\delta\hat{z}_{b2})\hat{E} + 2\hat{z}_{b1}\hat{z}_{b2}\,\delta\hat{E} \end{pmatrix}$$

$$\delta\mathbf{T}_u = (\delta\mathbf{T}_d)^T \qquad\text{(reciprocity)}$$

$$\delta\mathbf{R}_u = \begin{pmatrix} -(\delta Q + \delta U) & -2i(\delta\hat\eta_2\hat\nu_2 + \hat\eta_2\delta\hat\nu_2)V -2i\hat\eta_2\hat\nu_2\,\delta V \\ \text{sym} & -(\delta T + \delta R) \end{pmatrix}$$

where $\delta\hat\eta_k = \delta\eta_k/(2\hat\eta_k)$, $\delta\hat\rho_k = \delta\rho_k/(2\hat\rho_k)$, etc.

---

## 5. Locality of the Perturbation

For a parameter $m_j$ belonging to layer $j$, the non-zero tangent inputs occur at:

| Source | Affected interface | Role of layer $j$ |
|--------|-------------------|-------------------|
| $\delta\mathbf{R}_d, \delta\mathbf{R}_u, \delta\mathbf{T}_u, \delta\mathbf{T}_d$ | $j{-}1$ | "layer 2" (below) |
| $\delta\mathbf{R}_d, \delta\mathbf{R}_u, \delta\mathbf{T}_u, \delta\mathbf{T}_d$ | $j$ (if $< N{-}1$) | "layer 1" (above) |
| $\delta\mathbf{E}_j$ | phase through layer $j$ | $\delta e_{a,j}, \delta e_{b,j}$ |

All other interfaces have $\delta\mathbf{R}_d = \delta\mathbf{R}_u = \delta\mathbf{T}_u = \delta\mathbf{T}_d = \mathbf{0}$ and $\delta\mathbf{E} = \mathbf{0}$. The tangent-linear recursion merely propagates $\delta\mathbf{RR}_d$ through the unperturbed Kennett addition formula at those interfaces.

---

## 6. Complete Jacobian

The full Jacobian has shape $(\text{nfreq}, 4N{-}1)$ where $N$ is the number of sub-ocean layers:

$$J_{ij} = \frac{\partial R(\omega_i)}{\partial m_j} = e_{a,0}^2(\omega_i) \cdot \delta\mathbf{RR}_d^{(0)}[0,0]\bigg|_{m_j}$$

The parameter vector is $\mathbf{m} = [\alpha_1,\ldots,\alpha_N, \beta_1,\ldots,\beta_N, \rho_1,\ldots,\rho_N, h_1,\ldots,h_{N-1}]$.

---

## 7. Equivalence to Dietrich & Kormendi (1990)

The tangent-linear recursion derived above is the **forward-mode automatic differentiation** of the Kennett recursion. It computes the exact same quantity as the Dietrich & Kormendi analytical Fréchet derivative, but expressed as a recursion over interfaces rather than an integral over Green's functions.

The equivalence holds because:
- D&K's Fréchet kernel $\delta R = \int G^\uparrow(z) \cdot \delta\mathbf{P}(z) \cdot G^\downarrow(z)\,dz$ for homogeneous layers reduces to a finite sum over interfaces
- The Green's functions $G^\uparrow, G^\downarrow$ encode the cumulative transmission/reflection operators of the Kennett recursion
- The perturbation operator $\delta\mathbf{P}$ corresponds to $(\delta\mathbf{R}_d, \delta\mathbf{R}_u, \delta\mathbf{T}_u, \delta\mathbf{T}_d, \delta\mathbf{E})$

**Validated numerically:** AD Jacobian (torch.autograd) agrees with this tangent-linear Jacobian (numpy) to $< 10^{-10}$ relative error, and both agree with finite differences to $< 10^{-5}$.

---

## 8. Ocean-Bottom Interface (Acoustic-Elastic)

The ocean-bottom interface has the same structure as solid-solid, with the simplifications:
- $\nu_1 = 0$ (no shear in ocean)
- $\tilde\beta_1 = 0$
- $F = b$ (not $b\nu_1 + c\nu_2$)
- $H = -d\eta_2$ (not $a - d\eta_2\nu_1$)
- $\mathbf{R}_d$ has only $[0,0]$ nonzero (P$\to$P only from above)
- $\mathbf{T}_d$ has column 1 zero (no downgoing S from ocean)
- $\mathbf{T}_u$ has row 1 zero (no upgoing S to ocean)

Ocean parameters (layer 0) are fixed: $\delta\eta_1 = \delta\rho_1 = 0$.
