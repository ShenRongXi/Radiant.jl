"""
    ray_moc_kernel.jl

Characteristic-line (MOC) kernel for the 1D first-collision-source (FCS) ray sweep.
Implements Cayley-Hamilton 2×2 matrix exponentials, effective reaction matrices,
interface sources, and high-order spatial-moment kernels for the FCS ray path.

Only depends on LinearAlgebra (already imported by Radiant).
"""

# -----------------------------------------------------------------------------
# Cayley-Hamilton coefficients for exp(-A Δs) with A a 2×2 real matrix
# -----------------------------------------------------------------------------

"""
    _cayley_hamilton_coeffs(tr, D, Δs; τ_root=1e-6)

Return `(c0, c1, om_c0)` such that `exp(-A Δs) = c0 I + c1 A` for a 2×2 matrix
with trace `tr` and discriminant `D = tr² - 4 det`.  `om_c0 = 1 - c0` is formed
stably via `expm1` in the near-repeated-root branch.

The branch criterion uses the true cancellation metric `√|D|·Δs ≤ τ_root`, not
`|D|/tr²`, so that small-Δs near-repeated roots are handled by the stable branch.
"""
function _cayley_hamilton_coeffs(tr::Float64, D::Float64, Δs::Float64; τ_root::Float64=1e-6)
    if sqrt(abs(D)) * Δs ≤ τ_root        # near-repeated (covers D≷0 small)
        λ  = tr / 2
        eλ = exp(-λ * Δs)
        c1 = -Δs * eλ
        c0 = (1.0 + λ * Δs) * eλ
        om_c0 = -expm1(-λ * Δs) - λ * Δs * eλ   # stable 1-c0
    elseif D > 0                         # distinct real roots
        s  = sqrt(D)
        λ1 = (tr + s) / 2
        λ2 = (tr - s) / 2
        e1 = exp(-λ1 * Δs)
        e2 = exp(-λ2 * Δs)
        c1 = (e1 - e2) / (λ1 - λ2)
        c0 = (λ1 * e2 - λ2 * e1) / (λ1 - λ2)
        om_c0 = 1.0 - c0
    else                                 # complex conjugate roots
        a  = tr / 2
        b  = sqrt(-D) / 2
        ea = exp(-a * Δs)
        c0 = ea * (cos(b * Δs) + (a / b) * sin(b * Δs))
        c1 = -ea * sin(b * Δs) / b
        om_c0 = 1.0 - c0
    end
    return c0, c1, om_c0
end

# -----------------------------------------------------------------------------
# 2×2 matrix exponential and response matrix
# -----------------------------------------------------------------------------

"""
    ray_moc_matrices_2x2(A, Δs; ϵ_sing=1e-12, τ_root=1e-6)

Return `(E, F, Ainv, det)` where `E = exp(-A Δs)` (Cayley-Hamilton) and
`F = A⁻¹(I - E) = (1-c0) A⁻¹ - c1 I`.  The explicit 2×2 inverse is computed once
and returned for reuse.

Two independent thresholds:
- `ϵ_sing`: near-singular guard on `|det| / max(tr², 1)`.
- `τ_root`: near-repeated-root cancellation metric `√|D|·Δs`.
"""
function ray_moc_matrices_2x2(A::Matrix{Float64}, Δs::Float64; ϵ_sing::Float64=1e-12, τ_root::Float64=1e-6)
    tr  = A[1,1] + A[2,2]
    det = A[1,1] * A[2,2] - A[1,2] * A[2,1]
    if abs(det) ≤ ϵ_sing * max(tr * tr, 1.0)
        error("ray_moc 2×2: det(A)≈0 (vacuum BFP); add series branch before stage 2.")
    end
    D = tr * tr - 4.0 * det
    c0, c1, om_c0 = _cayley_hamilton_coeffs(tr, D, Δs; τ_root=τ_root)
    E = c0 * I + c1 * A
    Ainv = (1.0 / det) * [A[2,2] -A[1,2]; -A[2,1] A[1,1]]
    F = om_c0 * Ainv - c1 * I
    return E, F, Ainv, det
end

"""
    ray_moc_matrices_general(A::AbstractMatrix{Float64}, Δs::Float64)

Generic matrix exponential path for size `N ≥ 2` (high-order energy expansion).
"""
function ray_moc_matrices_general(A::AbstractMatrix{Float64}, Δs::Float64)
    E = exp(-A * Δs)
    F = A \ (I - E)
    return E, F
end

# -----------------------------------------------------------------------------
# Effective reaction matrix and boundary source
# -----------------------------------------------------------------------------

"""
    _build_A_matrix(Σt, β1, β2)

2×2 effective reaction matrix `A_g` for BFP/CSD (v5 eq. 4).  `Σt` is the
effective total cross-section.  `β1, β2` are already scaled by `1/ΔE` by the caller.
"""
function _build_A_matrix(Σt::Float64, β1::Float64, β2::Float64)
    β_up   = β1 + sqrt(3.0) * β2
    β_down = β1 - sqrt(3.0) * β2
    return [Σt + β_down            -sqrt(3.0) * β_down;
            sqrt(3.0) * β_up        Σt + 3.0 * β_down + 2.0 * sqrt(3.0) * β2]
end

"""
    _build_boundary_source(β1, β2, phi_prev_group_out; scheme_E="DG")

CSD down-scattering interface source vector from the previous-group outgoing
scalar flux at `e = -1/2`.

- DG (stage 1/2): `f = (β1+√3 β2) [1; √3] φ`
- DD (stage 3):   `f = [2√3 β2; 2√3 β1] φ`

`β1, β2` are already scaled by `1/ΔE`.
"""
function _build_boundary_source(β1::Float64, β2::Float64, phi_prev_group_out::Float64; scheme_E::String="DG")
    sc = uppercase(scheme_E)
    if sc == "DG"
        β_up = β1 + sqrt(3.0) * β2
        return β_up * [1.0; sqrt(3.0)] * phi_prev_group_out
    elseif sc == "DD"
        return [2.0 * sqrt(3.0) * β2; 2.0 * sqrt(3.0) * β1] * phi_prev_group_out
    else
        error("Unsupported energy scheme: ", scheme_E)
    end
end

# -----------------------------------------------------------------------------
# High-order spatial moment kernels
# -----------------------------------------------------------------------------

"""
    ray_moc_Jm(E, F, Ainv, B, Δs, mmax; τ_series=1e-2)

Return `J_0 … J_mmax` where `J_m = ∫_0^1 τ^m exp(-B τ) dτ`, with `B = A Δs`.
For `m ≥ 1` the recurrence uses `Binv = Ainv / Δs` when `‖B‖_1` is large.
For small `‖B‖_1` all moments use the Taylor series to avoid catastrophic
cancellation (`J_0 = F/Δs` would otherwise lose precision when `B ≈ 0`).
"""
function ray_moc_Jm(E::Matrix{Float64}, F::Matrix{Float64}, Ainv::Matrix{Float64},
                    B::Matrix{Float64}, Δs::Float64, mmax::Int; τ_series::Float64=1e-2)
    Binv = Ainv / Δs
    J = Vector{Matrix{Float64}}(undef, mmax + 1)
    Bnorm = opnorm(B, 1)
    dim = size(B, 1)
    I2 = Matrix{Float64}(I, dim, dim)
    if Bnorm ≥ τ_series
        J[1] = F / Δs
    else
        # Series: J_0 = Σ_{n=0}^∞ (-1)^n B^n / (n+1)!
        S = zeros(dim, dim)
        term = copy(I2)
        for n in 0:8
            S += ((-1.0)^n / factorial(n + 1)) * term
            term = term * B
        end
        J[1] = S
    end
    for m in 1:mmax
        if Bnorm ≥ τ_series
            J[m+1] = Binv * (m * J[m] - E)        # recurrence (v1 eq. 4)
        else
            # Series: J_m = Σ_{n=0}^∞ (-1)^n B^n / (n! (m+n+1))
            S = zeros(dim, dim)
            term = copy(I2)
            for n in 0:8
                S += ((-1.0)^n / (factorial(n) * (m + n + 1))) * term
                term = term * B
            end
            J[m+1] = S
        end
    end
    return J
end

"""
    _series_L(B, m, p; n_terms=8)

Series branch for `L_{m,p} = ∫_0^1 τ^p H_m(τ) dτ` with `H_m(τ)=∫_0^τ exp(-B(τ-s)) s^m ds`.
Used for small `‖B‖` to avoid `B^{-2}, B^{-3}` cancellation.
"""
function _series_L(B::Matrix{Float64}, m::Int, p::Int; n_terms::Int=8)
    S = zeros(size(B))
    term = Matrix{Float64}(I, size(B))
    for k in 0:n_terms-1
        if m == 0 && p == 0
            denom = factorial(k + 1) * (k + 2)
        elseif m == 0 && p == 1
            denom = factorial(k + 1) * (k + 3)
        elseif m == 1 && p == 0
            denom = factorial(k + 2) * (k + 3)
        elseif m == 1 && p == 1
            denom = factorial(k + 2) * (k + 4)
        else
            error("_series_L only implemented for m,p ∈ {0,1}")
        end
        S += ((-1.0)^k / denom) * term
        term = term * B
    end
    return S
end

"""
    ray_moc_Lmp_Ox2(E, F, Ainv, B, Δs; τ_series=1e-2)

Return `(L00, L01, L10, L11)` for the 𝒪x=2 polynomial-interface-source kernel.
Uses closed forms (v2 eq. S4) for large `‖B‖_1` and series for small `‖B‖_1`.
"""
function ray_moc_Lmp_Ox2(E::Matrix{Float64}, F::Matrix{Float64}, Ainv::Matrix{Float64},
                         B::Matrix{Float64}, Δs::Float64; τ_series::Float64=1e-2)
    I2 = Matrix{Float64}(I, 2, 2)
    Binv = Ainv / Δs
    Binv2 = Binv * Binv
    Binv3 = Binv2 * Binv
    J0 = F / Δs
    J1 = Binv * (J0 - E)

    Bnorm = opnorm(B, 1)
    if Bnorm ≥ τ_series
        L00 = Binv * (I2 - J0)
        L01 = 0.5 * Binv - Binv2 * (J0 - E)
        L10 = 0.5 * Binv - Binv2 * (I2 - J0)
        L11 = (1.0 / 3.0) * Binv - 0.5 * Binv2 + Binv3 * (J0 - E)
    else
        L00 = _series_L(B, 0, 0)
        L01 = _series_L(B, 0, 1)
        L10 = _series_L(B, 1, 0)
        L11 = _series_L(B, 1, 1)
    end
    return L00, L01, L10, L11
end

# -----------------------------------------------------------------------------
# Polynomial interface source coefficients for 𝒪x=2
# -----------------------------------------------------------------------------

"""
    ray_polynomial_source_qs_Ox2(phiE::Vector{Float64}, beta_up::Float64, sx::Float64)

For 𝒪x=2, expand the previous-group scalar flux `φ_E(τ) = φ_1 + φ_2 √3 sx(2τ-1)`
into `q_eff(τ) = q^{(0)} + q^{(1)} τ` with `q^{(m)} = β_up [1; √3] qm_scalar`.
"""
function ray_polynomial_source_qs_Ox2(phiE::AbstractVector{Float64}, beta_up::Float64, sx::Float64)
    v = [1.0, sqrt(3.0)]
    q0s = phiE[1] - sqrt(3.0) * sx * phiE[2]
    q1s = 2.0 * sqrt(3.0) * sx * phiE[2]
    q0 = beta_up * q0s * v
    q1 = beta_up * q1s * v
    return q0, q1
end

"""
    ray_legendre_tau_coeffs(jx::Int, sx::Float64)

Return the τ-power coefficients `a[1:jx]` of `P̃_{jx}(sx(2τ-1))`.
Coefficients are ordered from lowest to highest power of τ.
"""
function ray_legendre_tau_coeffs(jx::Int, sx::Float64)
    if jx == 1
        return [1.0]
    elseif jx == 2
        # √3 * sx * (2τ - 1) = -√3*sx + 2√3*sx τ
        return [-sqrt(3.0) * sx, 2.0 * sqrt(3.0) * sx]
    elseif jx == 3
        # √5 * (6τ² - 6τ + 1)
        return [sqrt(5.0), -6.0 * sqrt(5.0), 6.0 * sqrt(5.0)]
    else
        error("ray_legendre_tau_coeffs only implemented for jx ≤ 3 (stage 2)")
    end
end

# -----------------------------------------------------------------------------
# Stage 2a: high-order spatial moments with constant interface source
# -----------------------------------------------------------------------------

"""
    ray_moments_constant_source(E, F, Ainv, B, Δs, phi_in, q_eff, sx, 𝒪x)

Compute the spatial moments `M[k,jx]` for `jx = 1..𝒪x` under a constant effective
source `q_eff` (stage 2a).  Uses the recurrence `J_m` and the Legendre τ-coefficients.
"""
function ray_moments_constant_source(E::Matrix{Float64}, F::Matrix{Float64},
                                     Ainv::Matrix{Float64}, B::Matrix{Float64},
                                     Δs::Float64, phi_in::Vector{Float64},
                                     q_eff::Vector{Float64}, sx::Float64, 𝒪x::Int)
    J = ray_moc_Jm(E, F, Ainv, B, Δs, 𝒪x - 1)
    N = length(phi_in)
    M = Matrix{Float64}(undef, N, 𝒪x)
    I2 = Matrix{Float64}(I, N, N)
    for jx in 1:𝒪x
        a = ray_legendre_tau_coeffs(jx, sx)
        G = zeros(N, N)
        for (m_idx, am) in enumerate(a)
            G += am * J[m_idx]
        end
        δ = (jx == 1) ? 1.0 : 0.0
        M[:, jx] = G * phi_in + Ainv * (δ * I2 - G) * q_eff
    end
    return M
end

"""
    project_M_to_Phi_n(M, 𝒪x, 𝒪E, isFC::Bool)

Project the `(𝒪E, 𝒪x)` moment matrix `M[k,jx]` onto the `Nm[5]` vector `𝚽n`
following the non-FC or FC indexing convention of `flux_1D_BFP`.
"""
function project_M_to_Phi_n(M::Matrix{Float64}, 𝒪x::Int, 𝒪E::Int, isFC::Bool)
    if isFC
        𝚽n = Vector{Float64}(undef, 𝒪x * 𝒪E)
        for jx in 1:𝒪x, k in 1:𝒪E
            𝚽n[𝒪E*(jx-1)+k] = M[k, jx]
        end
    else
        𝚽n = Vector{Float64}(undef, 𝒪x + 𝒪E - 1)
        𝚽n[1] = M[1, 1]
        for k in 2:𝒪E
            𝚽n[k] = M[k, 1]
        end
        for jx in 2:𝒪x
            𝚽n[𝒪E + jx - 1] = M[1, jx]
        end
    end
    return 𝚽n
end

# -----------------------------------------------------------------------------
# Stage 2b: 𝒪x=2 polynomial interface source
# -----------------------------------------------------------------------------

"""
    _series_H1_at_1(B; n_terms=8)

Series for `H_1(1) = ∫_0^1 exp(-B(1-s)) s ds = Σ_{k=0}^∞ (-1)^k B^k / (k+2)!`.
Used for small `‖B‖` to avoid the `B^{-2}` cancellation in the closed form.
"""
function _series_H1_at_1(B::Matrix{Float64}; n_terms::Int=8)
    S = zeros(size(B))
    term = Matrix{Float64}(I, size(B))
    for k in 0:n_terms-1
        S += ((-1.0)^k / factorial(k + 2)) * term
        term = term * B
    end
    return S
end

"""
    ray_moments_Ox2(E, F, Ainv, B, Δs, phi_in, phiE, beta_up, sx)

Compute the spatial moments `M[k,jx]` for `𝒪x=2` under a polynomial interface
source `q_eff(τ) = q^{(0)} + q^{(1)} τ` (stage 2b, v2 eq. (S1)-(S8)).
"""
function ray_moments_Ox2(E::Matrix{Float64}, F::Matrix{Float64},
                         Ainv::Matrix{Float64}, B::Matrix{Float64},
                         Δs::Float64, phi_in::AbstractVector{Float64},
                         phiE::AbstractVector{Float64}, beta_up::Float64, sx::Float64)
    J = ray_moc_Jm(E, F, Ainv, B, Δs, 1)
    J0 = J[1]
    J1 = J[2]
    L00, L01, L10, L11 = ray_moc_Lmp_Ox2(E, F, Ainv, B, Δs)
    q0, q1 = ray_polynomial_source_qs_Ox2(phiE, beta_up, sx)

    N = length(phi_in)
    M = Matrix{Float64}(undef, N, 2)
    for jx in 1:2
        if jx == 1
            a0, a1 = 1.0, 0.0
        else
            a0 = -sqrt(3.0) * sx
            a1 =  2.0 * sqrt(3.0) * sx
        end
        G  = a0 * J0 + a1 * J1
        K0 = Δs * (a0 * L00 + a1 * L01)
        K1 = Δs * (a0 * L10 + a1 * L11)
        M[:, jx] = G * phi_in + K0 * q0 + K1 * q1
    end
    return M
end

"""
    ray_outgoing_flux_Ox2(E, F, Ainv, B, Δs, phi_in, phiE, beta_up, sx)

Return the outgoing energy-coefficient vector `Φ(τ=1)` for `𝒪x=2` with
polynomial interface source (v2 eq. (S9)-(S10)).
"""
function ray_outgoing_flux_Ox2(E::Matrix{Float64}, F::Matrix{Float64},
                               Ainv::Matrix{Float64}, B::Matrix{Float64},
                               Δs::Float64, phi_in::AbstractVector{Float64},
                               phiE::AbstractVector{Float64}, beta_up::Float64,
                               sx::Float64)
    q0, q1 = ray_polynomial_source_qs_Ox2(phiE, beta_up, sx)
    Binv = Ainv / Δs
    N = length(phi_in)
    I2 = Matrix{Float64}(I, N, N)
    Bnorm = opnorm(B, 1)
    if Bnorm ≥ 1e-2
        H1_at_1 = Binv * (I2 - Binv * (I2 - E))
    else
        H1_at_1 = _series_H1_at_1(B)
    end
    return E * phi_in + F * q0 + Δs * H1_at_1 * q1
end
