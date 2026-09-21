using Radiant
using LinearAlgebra
using Test
using Random

# Helper: matrix exponential for BigFloat 2×2 via scaling-and-squaring Taylor.
function _expm_big(M::Matrix{BigFloat})
    nrm = maximum(sum(abs.(M[i, :]) for i in 1:size(M, 1)))
    k = max(0, ceil(Int, log2(nrm + 1)) + 1)
    Ms = M / BigFloat(2)^k
    E = Matrix{BigFloat}(I, size(M))
    term = copy(E)
    for n in 1:90
        term = term * Ms / BigFloat(n)
        E = E + term
        if maximum(abs.(term)) < BigFloat(1e-60)
            break
        end
    end
    for _ in 1:k
        E = E * E
    end
    return E
end

@testset "ray_moc_kernel" begin

    @testset "Cayley-Hamilton coefficients: repeated-root 2×2" begin
        λ = 2.0
        Δs = 0.7
        c0, c1, om_c0 = Radiant._cayley_hamilton_coeffs(2λ, 0.0, Δs)
        @test c0 ≈ (1.0 + λ * Δs) * exp(-λ * Δs)
        @test c1 ≈ -Δs * exp(-λ * Δs)
        @test om_c0 ≈ 1.0 - c0
    end

    @testset "2×2 E/F against generic exp reference" begin
        rng = MersenneTwister(42)
        worst_E = 0.0
        worst_F = 0.0
        for _ in 1:20
            Sig = 1.0 + 29.0 * rand(rng)
            b1 = 100.0 * rand(rng)
            b2 = -20.0 + 40.0 * rand(rng)
            β_up = b1 + sqrt(3.0) * b2
            β_dn = b1 - sqrt(3.0) * b2
            A = [Sig + β_dn        -sqrt(3.0) * β_dn;
                 sqrt(3.0) * β_up   Sig + 3.0 * β_dn + 2.0*sqrt(3.0)*b2]
            Δs = 10.0^(-2.0 + 2.5 * rand(rng))

            E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)

            E_ref = exp(-A * Δs)
            F_ref = A \ (I - E_ref)

            worst_E = max(worst_E, norm(E - E_ref) / max(1e-15, norm(E_ref)))
            worst_F = max(worst_F, norm(F - F_ref) / max(1e-15, norm(F_ref)))
        end
        @test worst_E < 1e-12
        @test worst_F < 1e-7   # dominated by Ainv conditioning in strong anisotropy
    end

    @testset "Jm recurrence vs BigFloat recurrence reference" begin
        rng = MersenneTwister(7)
        worst = 0.0
        for _ in 1:20
            Sig = 1.0 + 29.0 * rand(rng)
            b1 = 100.0 * rand(rng)
            b2 = -20.0 + 40.0 * rand(rng)
            β_up = b1 + sqrt(3.0) * b2
            β_dn = b1 - sqrt(3.0) * b2
            A = [Sig + β_dn        -sqrt(3.0) * β_dn;
                 sqrt(3.0) * β_up   Sig + 3.0 * β_dn + 2.0*sqrt(3.0)*b2]
            Δs = 10.0^(-2.0 + 2.5 * rand(rng))

            E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)
            B = A * Δs
            J = Radiant.ray_moc_Jm(E, F, Ainv, B, Δs, 2)

            # BigFloat reference: same recurrence, J_0 = B⁻¹(I-E)
            B_big = BigFloat.(B)
            E_big = _expm_big(-B_big)
            I2 = Matrix{BigFloat}(I, 2, 2)
            Binv_big = B_big \ I2
            J0_ref = Float64.(Binv_big * (I2 - E_big))
            J1_ref = Float64.(Binv_big * (BigFloat(1) * Binv_big * (I2 - E_big) - E_big))
            J2_ref = Float64.(Binv_big * (BigFloat(2) * Binv_big * (Binv_big * (I2 - E_big) - E_big) - E_big))

            worst = max(worst, norm(J[1] - J0_ref) / max(1e-15, norm(J0_ref)))
            worst = max(worst, norm(J[2] - J1_ref) / max(1e-15, norm(J1_ref)))
            worst = max(worst, norm(J[3] - J2_ref) / max(1e-15, norm(J2_ref)))
        end
        @test worst < 1e-11
    end

    @testset "Jm small-‖B‖ series branch limits" begin
        # As B → 0, J_m → I / (m+1).  J[1]=J_0 → I, J[2]=J_1 → I/2, J[3]=J_2 → I/3.
        A = [0.01 0.005; 0.002 0.008]
        Δs = 1e-4
        E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)
        B = A * Δs
        J = Radiant.ray_moc_Jm(E, F, Ainv, B, Δs, 2)
        I2 = Matrix{Float64}(I, 2, 2)
        @test J[1] ≈ I2 rtol=1e-4
        @test J[2] ≈ 0.5 * I2 rtol=1e-4
        @test J[3] ≈ (1.0/3.0) * I2 rtol=1e-4
    end

    @testset "Jm works for 1x1 (N=1) small ‖B‖" begin
        # N=1 (scalar BTE) with small optical thickness must take the series
        # branch without a 2×2 hardcode (DimensionMismatch otherwise).
        Σt = 1e-4; Δs = 1e-3
        E1    = reshape([exp(-Σt*Δs)], 1, 1)
        F1    = reshape([-expm1(-Σt*Δs)/Σt], 1, 1)
        Ainv1 = reshape([1.0/Σt], 1, 1)
        B1    = reshape([Σt*Δs], 1, 1)
        J = Radiant.ray_moc_Jm(E1, F1, Ainv1, B1, Δs, 1)
        @test size(J[1]) == (1,1)
        @test size(J[2]) == (1,1)
        @test isapprox(J[1][1,1], 1.0; atol=1e-6)
        @test isapprox(J[2][1,1], 0.5; atol=1e-6)
    end

    @testset "Lmp Ox2 closed form vs BigFloat closed-form reference" begin
        rng = MersenneTwister(13)
        worst = 0.0
        for _ in 1:20
            Sig = 1.0 + 29.0 * rand(rng)
            b1 = 100.0 * rand(rng)
            b2 = -20.0 + 40.0 * rand(rng)
            β_up = b1 + sqrt(3.0) * b2
            β_dn = b1 - sqrt(3.0) * b2
            A = [Sig + β_dn        -sqrt(3.0) * β_dn;
                 sqrt(3.0) * β_up   Sig + 3.0 * β_dn + 2.0*sqrt(3.0)*b2]
            Δs = 10.0^(-2.0 + 2.5 * rand(rng))

            E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)
            B = A * Δs
            L00, L01, L10, L11 = Radiant.ray_moc_Lmp_Ox2(E, F, Ainv, B, Δs)

            # BigFloat closed-form reference (S4)
            B_big = BigFloat.(B)
            E_big = _expm_big(-B_big)
            I2 = Matrix{BigFloat}(I, 2, 2)
            Binv_big = B_big \ I2
            J0_big = Binv_big * (I2 - E_big)
            L00r = Float64.(Binv_big * (I2 - J0_big))
            L01r = Float64.(0.5 * Binv_big - Binv_big^2 * (J0_big - E_big))
            L10r = Float64.(0.5 * Binv_big - Binv_big^2 * (I2 - J0_big))
            L11r = Float64.((1.0/3.0) * Binv_big - 0.5 * Binv_big^2 + Binv_big^3 * (J0_big - E_big))

            worst = max(worst, norm(L00 - L00r) / max(1e-15, norm(L00r)))
            worst = max(worst, norm(L01 - L01r) / max(1e-15, norm(L01r)))
            worst = max(worst, norm(L10 - L10r) / max(1e-15, norm(L10r)))
            worst = max(worst, norm(L11 - L11r) / max(1e-15, norm(L11r)))
        end
        @test worst < 1e-11
    end

    @testset "Lmp Ox2 small-‖B‖ series branch limits" begin
        # As B → 0: L00→1/2 I, L01→1/3 I, L10→1/6 I, L11→1/8 I.
        A = [0.01 0.005; 0.002 0.008]
        Δs = 1e-4
        E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)
        B = A * Δs
        L00, L01, L10, L11 = Radiant.ray_moc_Lmp_Ox2(E, F, Ainv, B, Δs)
        I2 = Matrix{Float64}(I, 2, 2)
        @test L00 ≈ 0.5 * I2 rtol=1e-4
        @test L01 ≈ (1.0/3.0) * I2 rtol=1e-4
        @test L10 ≈ (1.0/6.0) * I2 rtol=1e-4
        @test L11 ≈ (1.0/8.0) * I2 rtol=1e-4
    end

    @testset "Polynomial source coefficients" begin
        phiE = [1.0, 2.0]
        beta_up = 0.5
        sx = 1.0
        q0, q1 = Radiant.ray_polynomial_source_qs_Ox2(phiE, beta_up, sx)
        v = [1.0, sqrt(3.0)]
        @test q0 ≈ beta_up * (1.0 - sqrt(3.0) * 2.0) * v
        @test q1 ≈ beta_up * (2.0 * sqrt(3.0) * 2.0) * v

        sx = -1.0
        q0m, q1m = Radiant.ray_polynomial_source_qs_Ox2(phiE, beta_up, sx)
        @test q0m ≈ beta_up * (1.0 + sqrt(3.0) * 2.0) * v
        @test q1m ≈ beta_up * (-2.0 * sqrt(3.0) * 2.0) * v
    end

    @testset "Legendre τ coefficients" begin
        @test Radiant.ray_legendre_tau_coeffs(1, 1.0) ≈ [1.0]
        @test Radiant.ray_legendre_tau_coeffs(2, 1.0) ≈ [-sqrt(3.0), 2*sqrt(3.0)]
        @test Radiant.ray_legendre_tau_coeffs(2, -1.0) ≈ [sqrt(3.0), -2*sqrt(3.0)]
        @test Radiant.ray_legendre_tau_coeffs(3, 1.0) ≈ [sqrt(5.0), -6*sqrt(5.0), 6*sqrt(5.0)]
    end

    @testset "Stage 2b: ray_moments_Ox2 vs BigFloat reference" begin
        rng = MersenneTwister(7)
        worst = 0.0
        for _ in 1:20
            Sig = 1.0 + 29.0 * rand(rng)
            b1 = 100.0 * rand(rng)
            b2 = -20.0 + 40.0 * rand(rng)
            β_up = b1 + sqrt(3.0) * b2
            β_dn = b1 - sqrt(3.0) * b2
            A = [Sig + β_dn        -sqrt(3.0) * β_dn;
                 sqrt(3.0) * β_up   Sig + 3.0 * β_dn + 2.0*sqrt(3.0)*b2]
            Δs = 10.0^(-2.0 + 2.5 * rand(rng))
            phi_in = rand(rng, 2)
            phiE = rand(rng, 2)
            beta_up = β_up
            sx = rand(rng) > 0.5 ? 1.0 : -1.0

            E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)
            B = A * Δs
            M = Radiant.ray_moments_Ox2(E, F, Ainv, B, Δs, phi_in, phiE, beta_up, sx)

            # BigFloat closed-form reference (S4)
            B_big = BigFloat.(B)
            E_big = _expm_big(-B_big)
            I2 = Matrix{BigFloat}(I, 2, 2)
            Binv_big = B_big \ I2
            J0_big = Binv_big * (I2 - E_big)
            J1_big = Binv_big * (J0_big - E_big)
            L00_big = Binv_big * (I2 - J0_big)
            L01_big = 0.5 * Binv_big - Binv_big^2 * (J0_big - E_big)
            L10_big = 0.5 * Binv_big - Binv_big^2 * (I2 - J0_big)
            L11_big = (1.0/3.0) * Binv_big - 0.5 * Binv_big^2 + Binv_big^3 * (J0_big - E_big)

            v = [1.0, sqrt(3.0)]
            q0s = phiE[1] - sqrt(3.0) * sx * phiE[2]
            q1s = 2.0 * sqrt(3.0) * sx * phiE[2]
            q0 = beta_up * q0s * v
            q1 = beta_up * q1s * v

            M_ref = Matrix{Float64}(undef, 2, 2)
            for jx in 1:2
                if jx == 1
                    a0, a1 = 1.0, 0.0
                else
                    a0 = -sqrt(3.0) * sx
                    a1 =  2.0 * sqrt(3.0) * sx
                end
                G  = a0 * J0_big + a1 * J1_big
                K0 = Δs * (a0 * L00_big + a1 * L01_big)
                K1 = Δs * (a0 * L10_big + a1 * L11_big)
                pin = BigFloat.(phi_in)
                M_ref[:, jx] = Float64.(G * pin + K0 * BigFloat.(q0) + K1 * BigFloat.(q1))
            end

            den = max(1e-12, norm(M_ref))
            worst = max(worst, norm(M - M_ref) / den)
        end
        @test worst < 1e-11
    end

    @testset "Stage 2b: ray_outgoing_flux_Ox2 vs BigFloat reference" begin
        rng = MersenneTwister(17)
        worst = 0.0
        for _ in 1:20
            Sig = 1.0 + 29.0 * rand(rng)
            b1 = 100.0 * rand(rng)
            b2 = -20.0 + 40.0 * rand(rng)
            β_up = b1 + sqrt(3.0) * b2
            β_dn = b1 - sqrt(3.0) * b2
            A = [Sig + β_dn        -sqrt(3.0) * β_dn;
                 sqrt(3.0) * β_up   Sig + 3.0 * β_dn + 2.0*sqrt(3.0)*b2]
            Δs = 10.0^(-2.0 + 2.5 * rand(rng))
            phi_in = rand(rng, 2)
            phiE = rand(rng, 2)
            sx = rand(rng) > 0.5 ? 1.0 : -1.0

            E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)
            B = A * Δs
            Φ_out = Radiant.ray_outgoing_flux_Ox2(E, F, Ainv, B, Δs, phi_in, phiE, β_up, sx)

            # BigFloat reference: Φ(1) = E φ_in + F q0 + Δs H1(1) q1
            B_big = BigFloat.(B)
            E_big = _expm_big(-B_big)
            F_big = BigFloat.(A) \ (I - BigFloat.(E))
            Binv_big = B_big \ Matrix{BigFloat}(I, 2, 2)
            I2 = Matrix{BigFloat}(I, 2, 2)
            H1_at_1 = Binv_big * (I2 - Binv_big * (I2 - E_big))

            v = [1.0, sqrt(3.0)]
            q0s = phiE[1] - sqrt(3.0) * sx * phiE[2]
            q1s = 2.0 * sqrt(3.0) * sx * phiE[2]
            q0 = β_up * q0s * v
            q1 = β_up * q1s * v

            Φ_ref = Float64.(E_big * BigFloat.(phi_in) + F_big * BigFloat.(q0) + BigFloat(Δs) * H1_at_1 * BigFloat.(q1))

            den = max(1e-12, norm(Φ_ref))
            worst = max(worst, norm(Φ_out - Φ_ref) / den)
        end
        @test worst < 1e-11
    end

    @testset "Stage 2b: small-‖B‖ series branch vs BigFloat reference" begin
        A = [0.01 0.005; 0.002 0.008]
        Δs = 1e-4
        phi_in = [1.0, 0.5]
        phiE = [0.8, 0.3]
        beta_up = 0.5
        sx = 1.0

        E, F, Ainv, det = Radiant.ray_moc_matrices_2x2(A, Δs)
        B = A * Δs
        M = Radiant.ray_moments_Ox2(E, F, Ainv, B, Δs, phi_in, phiE, beta_up, sx)

        # BigFloat closed-form reference (series branch is triggered because ‖B‖_1 ≪ τ_series)
        B_big = BigFloat.(B)
        E_big = _expm_big(-B_big)
        I2 = Matrix{BigFloat}(I, 2, 2)
        Binv_big = B_big \ I2
        J0_big = Binv_big * (I2 - E_big)
        J1_big = Binv_big * (J0_big - E_big)
        L00_big = Binv_big * (I2 - J0_big)
        L01_big = 0.5 * Binv_big - Binv_big^2 * (J0_big - E_big)
        L10_big = 0.5 * Binv_big - Binv_big^2 * (I2 - J0_big)
        L11_big = (1.0/3.0) * Binv_big - 0.5 * Binv_big^2 + Binv_big^3 * (J0_big - E_big)

        v = [1.0, sqrt(3.0)]
        q0s = phiE[1] - sqrt(3.0) * sx * phiE[2]
        q1s = 2.0 * sqrt(3.0) * sx * phiE[2]
        q0 = beta_up * q0s * v
        q1 = beta_up * q1s * v

        M_ref = Matrix{Float64}(undef, 2, 2)
        for jx in 1:2
            if jx == 1
                a0, a1 = 1.0, 0.0
            else
                a0 = -sqrt(3.0) * sx
                a1 =  2.0 * sqrt(3.0) * sx
            end
            G  = a0 * J0_big + a1 * J1_big
            K0 = Δs * (a0 * L00_big + a1 * L01_big)
            K1 = Δs * (a0 * L10_big + a1 * L11_big)
            M_ref[:, jx] = Float64.(G * BigFloat.(phi_in) + K0 * BigFloat.(q0) + K1 * BigFloat.(q1))
        end

        @test M ≈ M_ref rtol=1e-10
    end

end
