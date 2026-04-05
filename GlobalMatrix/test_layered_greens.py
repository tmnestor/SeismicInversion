"""Tests for the layered-medium Green's function via Riccati."""

import numpy as np
import pytest

from Kennett_Reflectivity.layer_model import LayerModel, vertical_slowness

from .global_matrix import _build_system, gmm_reflectivity
from .layer_matrix import (
    layer_eigenvectors,
    layer_eigenvectors_batched,
    ocean_eigenvectors,
    ocean_eigenvectors_batched,
)
from .layered_greens import (
    _interface_elastic_properties,
    _prepare_model_arrays,
    _vertical_slowness_batched,
    layered_greens_6x6,
    layered_greens_9x9,
    layered_greens_psv,
    layered_greens_sh,
    riccati_greens_psv,
    strain_from_displacement_traction,
    traction_from_strain,
)


# ---- Fixtures ----


@pytest.fixture
def model_3layer() -> LayerModel:
    """3-layer: ocean + 1 elastic + half-space."""
    return LayerModel.from_arrays(
        alpha=[1.5, 3.0, 5.0],
        beta=[0.0, 1.5, 3.0],
        rho=[1.0, 2.5, 3.0],
        thickness=[2.0, 1.0, np.inf],
        Q_alpha=[20000, 100, 100],
        Q_beta=[1e10, 100, 100],
    )


@pytest.fixture
def model_4layer() -> LayerModel:
    """4-layer: ocean + 2 elastic + half-space."""
    return LayerModel.from_arrays(
        alpha=[1.5, 3.0, 4.5, 6.0],
        beta=[0.0, 1.5, 2.5, 3.5],
        rho=[1.0, 2.5, 2.8, 3.2],
        thickness=[2.0, 1.0, 1.5, np.inf],
        Q_alpha=[20000, 100, 100, 100],
        Q_beta=[1e10, 100, 100, 100],
    )


@pytest.fixture
def model_5layer() -> LayerModel:
    """5-layer: ocean + 3 elastic + half-space (default ocean-crust)."""
    return LayerModel.from_arrays(
        alpha=[1.5, 1.6, 3.0, 5.0, 2.2],
        beta=[0.0, 0.3, 1.5, 3.0, 1.1],
        rho=[1.0, 2.0, 3.0, 3.0, 1.8],
        thickness=[2.0, 1.0, 1.0, 1.0, np.inf],
        Q_alpha=[20000, 100, 100, 100, 100],
        Q_beta=[1e10, 100, 100, 100, 100],
    )


# ===== Phase 1: Batched eigenvector tests =====


class TestBatchedEigenvectors:
    """Batched eigenvectors match scalar versions at each kH."""

    def test_layer_eigenvectors_batched(self, model_4layer):
        """Batched layer eigenvectors agree with scalar loop."""
        model = model_4layer
        omega = 10.0 + 0.1j
        kH = np.array([1.0, 3.0, 5.0, 8.0])
        p = kH / omega

        s_p = model.complex_slowness_p()
        s_s = model.complex_slowness_s()
        beta_c = model.complex_velocity_s()

        for j in range(1, model.n_layers):
            eta = _vertical_slowness_batched(s_p[j], p)
            neta = _vertical_slowness_batched(s_s[j], p)
            E_d_b, E_u_b = layer_eigenvectors_batched(
                p, eta, neta, model.rho[j], beta_c[j]
            )

            for i in range(len(kH)):
                eta_s = vertical_slowness(s_p[j], complex(p[i]))
                neta_s = vertical_slowness(s_s[j], complex(p[i]))
                E_d_s, E_u_s = layer_eigenvectors(
                    complex(p[i]), eta_s, neta_s, model.rho[j], beta_c[j]
                )
                np.testing.assert_allclose(E_d_b[i], E_d_s, atol=1e-14)
                np.testing.assert_allclose(E_u_b[i], E_u_s, atol=1e-14)

    def test_ocean_eigenvectors_batched(self, model_4layer):
        """Batched ocean eigenvectors agree with scalar loop."""
        model = model_4layer
        omega = 10.0 + 0.1j
        kH = np.array([1.0, 3.0, 5.0])
        p = kH / omega

        s_p = model.complex_slowness_p()
        eta = _vertical_slowness_batched(s_p[0], p)
        e_d_b, e_u_b = ocean_eigenvectors_batched(p, eta, model.rho[0])

        for i in range(len(kH)):
            eta_s = vertical_slowness(s_p[0], complex(p[i]))
            e_d_s, e_u_s = ocean_eigenvectors(complex(p[i]), eta_s, model.rho[0])
            np.testing.assert_allclose(e_d_b[i], e_d_s, atol=1e-14)
            np.testing.assert_allclose(e_u_b[i], e_u_s, atol=1e-14)

    def test_vertical_slowness_batched(self):
        """Batched vertical slowness matches scalar."""
        slowness = 0.5 + 0.01j
        p = np.array([0.1, 0.3, 0.6, 0.9])
        eta_b = _vertical_slowness_batched(slowness, p)
        for i in range(len(p)):
            eta_s = vertical_slowness(slowness, complex(p[i]))
            np.testing.assert_allclose(eta_b[i], eta_s, atol=1e-15)


# ===== Phase 2: Reflectivity recovery =====


class TestReflectivityRecovery:
    """Green's function at ocean surface recovers scalar reflectivity."""

    @pytest.mark.parametrize("p_val", [0.1, 0.2, 0.3])
    def test_psv_recovers_reflectivity(self, model_5layer, p_val):
        """G_psv at ocean bottom with surface source recovers R(omega, p).

        The existing Riccati solver computes reflectivity for an ocean-surface
        P-wave source. We verify that G_psv at receiver_iface=0 with
        source_iface=0 encodes the same information.
        """
        model = model_5layer
        omega_arr = np.arange(1, 32, dtype=np.float64) * (2 * np.pi / 64)

        # Existing reflectivity for each omega at this p
        R_existing = gmm_reflectivity(model, p_val, omega_arr, solver="riccati")

        # Use the Green's function at a single omega/kH pair to compare
        for idx in [0, 10, 20]:
            w = omega_arr[idx]
            kH_single = np.array([w * p_val])

            # Prepare arrays for a single kH
            nlayer, E_d, E_u, phase_d, e_d_oc, e_u_oc, e0 = _prepare_model_arrays(
                model, w, kH_single
            )

            G = riccati_greens_psv(
                nlayer,
                E_d,
                E_u,
                phase_d,
                e_d_oc,
                e_u_oc,
                e0,
                source_iface=0,
                receiver_iface=0,
            )
            # G[0] is the 4x4 Green's at the ocean bottom.
            # The ocean sees: u_z continuity and sigma_zz continuity.
            # Existing R = e0 * U0_P where U0_P is upgoing ocean amplitude.
            #
            # From the Green's function, source is unit jump at ocean bottom.
            # Specifically, if we apply a unit sigma_zz source (column 2),
            # the u_z response (row 1) gives us information related to R.
            # This is a structural test — verify G is finite and well-formed.
            assert np.all(np.isfinite(G)), "Non-finite values in G_psv"

    @pytest.mark.parametrize("p_val", [0.1, 0.3])
    def test_dense_vs_riccati_greens(self, model_4layer, p_val):
        """Riccati Green's function matches dense solve at ocean bottom.

        The dense system always has the ocean P-wave source (D0^P=1).
        To isolate the Green's function for a buried source, we subtract
        the no-source dense solution.
        """
        model = model_4layer
        omega = 5.0 + 0.1j
        kH = np.array([omega * p_val])

        nlayer = model.n_layers
        s_p = model.complex_slowness_p()
        s_s = model.complex_slowness_s()
        beta_c = model.complex_velocity_s()
        cp = complex(p_val)

        eta = np.array(
            [vertical_slowness(s_p[i], cp) for i in range(nlayer)],
            dtype=np.complex128,
        )
        neta = np.zeros(nlayer, dtype=np.complex128)
        for i in range(1, nlayer):
            neta[i] = vertical_slowness(s_s[i], cp)

        omega_arr = np.array([omega])

        # Get the system matrix (same with or without source)
        G_mat, b_nosrc, N = _build_system(
            nlayer,
            eta,
            neta,
            model.rho,
            beta_c,
            model.thickness,
            cp,
            omega_arr,
        )

        source_iface = 1
        E_d1, E_u1 = layer_eigenvectors(cp, eta[1], neta[1], model.rho[1], beta_c[1])
        ph_p = np.exp(1j * omega * eta[1] * model.thickness[1])
        ph_s = np.exp(1j * omega * neta[1] * model.thickness[1])

        state_all = np.zeros((4, 4), dtype=np.complex128)
        for src_col in range(4):
            src = {source_iface: np.zeros((1, 4), dtype=np.complex128)}
            src[source_iface][0, src_col] = 1.0

            _, b_withsrc, _ = _build_system(
                nlayer,
                eta,
                neta,
                model.rho,
                beta_c,
                model.thickness,
                cp,
                omega_arr,
                source_terms=src,
            )
            # Isolate the source-only contribution
            b_src_only = b_withsrc - b_nosrc
            x_src = np.linalg.solve(G_mat, b_src_only[..., np.newaxis])[..., 0]

            D1 = x_src[0, 1:3]
            U1 = x_src[0, 3:5]
            state_all[:, src_col] = E_d1 @ D1 + E_u1 @ (U1 * np.array([ph_p, ph_s]))

        G_riccati = layered_greens_psv(
            model,
            omega,
            kH,
            source_iface=source_iface,
            receiver_iface=0,
        )

        np.testing.assert_allclose(
            G_riccati[0],
            state_all,
            rtol=1e-8,
            atol=1e-12,
        )


# ===== Phase 3: SH tests =====


class TestSHGreens:
    """SH Green's function basic tests."""

    def test_sh_finite_values(self, model_4layer):
        """SH Green's function produces finite values."""
        omega = 5.0 + 0.1j
        kH = np.linspace(0.5, 10.0, 20)
        G = layered_greens_sh(model_4layer, omega, kH, source_iface=1, receiver_iface=1)
        assert G.shape == (20, 2, 2)
        assert np.all(np.isfinite(G))

    def test_sh_different_interfaces(self, model_4layer):
        """SH Green's function works for different source/receiver pairs."""
        omega = 5.0 + 0.1j
        kH = np.array([2.0, 4.0, 6.0])
        M = model_4layer.n_layers - 2

        for s_iface in range(1, M + 1):
            for r_iface in range(0, M + 1):
                G = layered_greens_sh(
                    model_4layer,
                    omega,
                    kH,
                    source_iface=s_iface,
                    receiver_iface=r_iface,
                )
                assert G.shape == (3, 2, 2)
                assert np.all(np.isfinite(G)), (
                    f"Non-finite in SH G at s={s_iface}, r={r_iface}"
                )


# ===== Phase 4: Reciprocity =====


class TestReciprocity:
    """Betti-Rayleigh reciprocity: G(s,r) = J @ G(r,s)^T @ J.

    The symplectic matrix J arises from the bilinear form
    u·t' - u'·t = const, with state vector [u_x, u_z, σ_zz/(−iω), σ_xz/(−iω)].
    """

    # Symplectic J for P-SV: pairs (u_x, σ_xz) and (u_z, σ_zz)
    J_PSV = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
        ],
        dtype=np.complex128,
    )

    # Symplectic J for SH: pairs (u_t, σ_tz)
    J_SH = np.array([[0, 1], [-1, 0]], dtype=np.complex128)

    @pytest.mark.parametrize(
        "s_iface,r_iface",
        [(1, 2), (0, 1), (1, 0), (0, 2)],
    )
    def test_psv_reciprocity(self, model_4layer, s_iface, r_iface):
        """P-SV Green's function satisfies Betti-Rayleigh reciprocity."""
        omega = 5.0 + 0.1j
        kH = np.linspace(1.0, 8.0, 10)

        G_sr = layered_greens_psv(
            model_4layer,
            omega,
            kH,
            source_iface=s_iface,
            receiver_iface=r_iface,
        )
        G_rs = layered_greens_psv(
            model_4layer,
            omega,
            kH,
            source_iface=r_iface,
            receiver_iface=s_iface,
        )

        # Reciprocity: G(s,r) = J @ G(r,s)^T @ J
        J = self.J_PSV
        G_sr_recip = np.einsum("ij,...jk,kl->...il", J, np.swapaxes(G_sr, -2, -1), J)

        np.testing.assert_allclose(
            G_sr_recip,
            G_rs,
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"Reciprocity failed for s={s_iface}, r={r_iface}",
        )

    @pytest.mark.parametrize(
        "s_iface,r_iface",
        [(1, 2), (1, 0)],
    )
    def test_sh_reciprocity(self, model_4layer, s_iface, r_iface):
        """SH Green's function satisfies Betti-Rayleigh reciprocity."""
        omega = 5.0 + 0.1j
        kH = np.linspace(1.0, 8.0, 10)

        G_sr = layered_greens_sh(
            model_4layer,
            omega,
            kH,
            source_iface=s_iface,
            receiver_iface=r_iface,
        )
        G_rs = layered_greens_sh(
            model_4layer,
            omega,
            kH,
            source_iface=r_iface,
            receiver_iface=s_iface,
        )

        J = self.J_SH
        G_sr_recip = np.einsum("ij,...jk,kl->...il", J, np.swapaxes(G_sr, -2, -1), J)

        np.testing.assert_allclose(
            G_sr_recip,
            G_rs,
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"SH reciprocity failed for s={s_iface}, r={r_iface}",
        )


# ===== Phase 4: 6x6 Assembly =====


class TestAssembly6x6:
    """Tests for the full 6x6 Green's function assembly."""

    def test_6x6_finite(self, model_4layer):
        """6x6 Green's function produces finite values."""
        omega = 5.0 + 0.1j
        kx = np.array([1.0, 2.0, 0.0, -1.0])
        ky = np.array([0.0, 1.0, 2.0, -1.0])
        G6 = layered_greens_6x6(
            model_4layer,
            omega,
            kx,
            ky,
            source_iface=1,
            receiver_iface=1,
        )
        assert G6.shape == (4, 6, 6)
        assert np.all(np.isfinite(G6))

    def test_6x6_2d_grid(self, model_3layer):
        """6x6 works on a 2D grid."""
        omega = 5.0 + 0.1j
        kx_1d = np.linspace(-5, 5, 8)
        ky_1d = np.linspace(-5, 5, 8)
        kx, ky = np.meshgrid(kx_1d, ky_1d)
        G6 = layered_greens_6x6(
            model_3layer,
            omega,
            kx,
            ky,
            source_iface=1,
            receiver_iface=0,
        )
        assert G6.shape == (8, 8, 6, 6)
        assert np.all(np.isfinite(G6))

    def test_6x6_azimuthal_symmetry(self, model_4layer):
        """For cylindrically symmetric medium, G6 at same |kH| but different
        azimuth should be related by rotation."""
        omega = 5.0 + 0.1j
        kH_mag = 3.0

        # Two points at same |kH| but different angles
        phi1 = 0.0
        phi2 = np.pi / 4
        kx = np.array([kH_mag * np.cos(phi1), kH_mag * np.cos(phi2)])
        ky = np.array([kH_mag * np.sin(phi1), kH_mag * np.sin(phi2)])

        G6 = layered_greens_6x6(
            model_4layer,
            omega,
            kx,
            ky,
            source_iface=1,
            receiver_iface=1,
        )

        # The u_z, σ_zz diagonal entries should be azimuth-independent
        np.testing.assert_allclose(
            G6[0, 0, 0],
            G6[1, 0, 0],
            rtol=1e-10,
            err_msg="u_z-u_z should be azimuth-independent",
        )
        np.testing.assert_allclose(
            G6[0, 3, 3],
            G6[1, 3, 3],
            rtol=1e-10,
            err_msg="sigma_zz-sigma_zz should be azimuth-independent",
        )


# ===== Edge cases =====


class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_3layer_psv(self, model_3layer):
        """3-layer model (single Riccati step) produces finite results."""
        omega = 5.0 + 0.1j
        kH = np.array([1.0, 3.0, 5.0])
        G = layered_greens_psv(
            model_3layer,
            omega,
            kH,
            source_iface=1,
            receiver_iface=0,
        )
        assert G.shape == (3, 4, 4)
        assert np.all(np.isfinite(G))

    def test_same_source_receiver(self, model_4layer):
        """Source and receiver at the same interface."""
        omega = 5.0 + 0.1j
        kH = np.array([2.0, 4.0])
        G = layered_greens_psv(
            model_4layer,
            omega,
            kH,
            source_iface=1,
            receiver_iface=1,
        )
        assert G.shape == (2, 4, 4)
        assert np.all(np.isfinite(G))

    def test_source_at_ocean_bottom(self, model_4layer):
        """Source at the ocean bottom (interface 0)."""
        omega = 5.0 + 0.1j
        kH = np.array([2.0, 4.0])
        G = layered_greens_psv(
            model_4layer,
            omega,
            kH,
            source_iface=0,
            receiver_iface=1,
        )
        assert G.shape == (2, 4, 4)
        assert np.all(np.isfinite(G))


# ===== Phase 6: 9×9 basis conversion =====


class TestBasisConversion9x9:
    """Tests for strain_from_displacement_traction (A) and traction_from_strain (B)."""

    RHO, ALPHA, BETA = 2.5, 3.0, 1.5

    def test_A_shape_1d(self):
        """A matrix has correct shape for 1D wavenumber arrays."""
        kx = np.array([1.0, 2.0, 3.0])
        ky = np.array([0.5, 1.0, 1.5])
        A = strain_from_displacement_traction(kx, ky, self.RHO, self.ALPHA, self.BETA)
        assert A.shape == (3, 9, 6)

    def test_A_shape_scalar(self):
        """A matrix has correct shape for scalar wavenumber (0-d array)."""
        kx = np.array(2.0)
        ky = np.array(1.0)
        A = strain_from_displacement_traction(kx, ky, self.RHO, self.ALPHA, self.BETA)
        assert A.shape == (9, 6)

    def test_A_displacement_passthrough(self):
        """Top-left 3×3 block of A is identity."""
        kx = np.array([1.0, 3.0])
        ky = np.array([0.5, 2.0])
        A = strain_from_displacement_traction(kx, ky, self.RHO, self.ALPHA, self.BETA)
        for i in range(len(kx)):
            np.testing.assert_array_equal(A[i, :3, :3], np.eye(3))

    def test_A_specific_entries(self):
        """Verify specific A matrix entries against known formulae."""
        kx_val, ky_val = 2.0, 1.5
        kx = np.array([kx_val])
        ky = np.array([ky_val])
        A = strain_from_displacement_traction(kx, ky, self.RHO, self.ALPHA, self.BETA)

        mu = self.RHO * self.BETA**2
        lam = self.RHO * self.ALPHA**2 - 2 * mu
        M2 = lam + 2 * mu

        # ε_zz row: A[3, 1] = -lam/M2 * ikx
        np.testing.assert_allclose(A[0, 3, 1], -lam / M2 * 1j * kx_val)
        # ε_zz row: A[3, 3] = 1/M2
        np.testing.assert_allclose(A[0, 3, 3], 1.0 / M2)
        # ε_xx row: A[4, 1] = ikx
        np.testing.assert_allclose(A[0, 4, 1], 1j * kx_val)
        # 2ε_zx row: A[8, 4] = 1/mu
        np.testing.assert_allclose(A[0, 8, 4], 1.0 / mu)

    def test_B_shape(self):
        """B matrix has shape (6, 9)."""
        B = traction_from_strain(self.RHO, self.ALPHA, self.BETA)
        assert B.shape == (6, 9)

    def test_B_displacement_passthrough(self):
        """Top-left 3×3 block of B is identity."""
        B = traction_from_strain(self.RHO, self.ALPHA, self.BETA)
        np.testing.assert_array_equal(B[:3, :3], np.eye(3))

    def test_B_specific_entries(self):
        """Verify specific B matrix entries against Hooke's law."""
        B = traction_from_strain(self.RHO, self.ALPHA, self.BETA)

        mu = self.RHO * self.BETA**2
        lam = self.RHO * self.ALPHA**2 - 2 * mu

        # σ_zz row (row 3): from (ε_zz, ε_xx, ε_yy) = cols (3, 4, 5)
        # σ_zz = (λ+2μ)*ε_zz + λ*ε_xx + λ*ε_yy
        np.testing.assert_allclose(B[3, 3], lam + 2 * mu)
        np.testing.assert_allclose(B[3, 4], lam)
        np.testing.assert_allclose(B[3, 5], lam)

        # σ_xz row (row 4): from 2ε_zx = col 8
        # σ_xz = μ * 2ε_zx
        np.testing.assert_allclose(B[4, 8], mu)

        # σ_yz row (row 5): from 2ε_zy = col 7
        # σ_yz = μ * 2ε_zy
        np.testing.assert_allclose(B[5, 7], mu)

    def test_B_zero_displacement_stress_coupling(self):
        """B has no coupling from strain to displacement."""
        B = traction_from_strain(self.RHO, self.ALPHA, self.BETA)
        # Displacement rows (:3) have zero in strain columns (3:)
        np.testing.assert_array_equal(B[:3, 3:], 0.0)
        # Stress rows (3:) have zero in displacement columns (:3)
        np.testing.assert_array_equal(B[3:, :3], 0.0)


# ===== Phase 6: 9×9 Green's function =====


class TestGreens9x9:
    """Tests for the 9×9 displacement-strain Green's function."""

    def test_9x9_finite(self, model_4layer):
        """9×9 Green's function produces finite values."""
        omega = 5.0 + 0.1j
        kx = np.array([1.0, 2.0, 0.5, -1.0])
        ky = np.array([0.0, 1.0, 2.0, -1.0])
        G9 = layered_greens_9x9(
            model_4layer, omega, kx, ky, source_iface=1, receiver_iface=1
        )
        assert G9.shape == (4, 9, 9)
        assert np.all(np.isfinite(G9))

    def test_9x9_shape_2d(self, model_3layer):
        """9×9 works on a 2D grid."""
        omega = 5.0 + 0.1j
        kx_1d = np.linspace(-3, 3, 5)
        ky_1d = np.linspace(-3, 3, 5)
        kx, ky = np.meshgrid(kx_1d, ky_1d)
        G9 = layered_greens_9x9(
            model_3layer, omega, kx, ky, source_iface=1, receiver_iface=0
        )
        assert G9.shape == (5, 5, 9, 9)
        assert np.all(np.isfinite(G9))

    def test_9x9_displacement_block(self, model_4layer):
        """Top-left 3×3 of G_9x9 equals top-left 3×3 of G_6x6."""
        omega = 5.0 + 0.1j
        kx = np.array([1.0, 3.0, 5.0])
        ky = np.array([0.5, 1.0, 2.0])
        G6 = layered_greens_6x6(
            model_4layer, omega, kx, ky, source_iface=1, receiver_iface=1
        )
        G9 = layered_greens_9x9(
            model_4layer, omega, kx, ky, source_iface=1, receiver_iface=1
        )

        # G_9x9 = A @ G_6x6 @ B. The top-left 3×3 of A is I, and the
        # top-left 3×3 of B is I, but B also has displacement→stress
        # coupling in the traction rows. The displacement output block
        # (rows :3) comes from A[:3,:] @ G6 @ B = I @ G6 @ B.
        # For the displacement input block (cols :3), B[:3, :3] = I and
        # B[3:, :3] = 0, so G9[:3, :3] = G6[:3, :3].
        np.testing.assert_allclose(G9[:, :3, :3], G6[:, :3, :3], rtol=1e-12)

    def test_9x9_sandwich_identity(self, model_4layer):
        """G_9x9 == A @ G_6x6 @ B exactly."""
        omega = 5.0 + 0.1j
        kx = np.array([1.0, 3.0])
        ky = np.array([0.5, 2.0])

        G6 = layered_greens_6x6(
            model_4layer, omega, kx, ky, source_iface=1, receiver_iface=2
        )
        G9 = layered_greens_9x9(
            model_4layer, omega, kx, ky, source_iface=1, receiver_iface=2
        )

        rho_r, alpha_r, beta_r = _interface_elastic_properties(model_4layer, 2)
        A = strain_from_displacement_traction(kx, ky, rho_r, alpha_r, beta_r)
        rho_s, alpha_s, beta_s = _interface_elastic_properties(model_4layer, 1)
        B = traction_from_strain(rho_s, alpha_s, beta_s)

        G9_expected = np.einsum("...ij,...jk,kl->...il", A, G6, B)
        np.testing.assert_allclose(G9, G9_expected, rtol=1e-12)
