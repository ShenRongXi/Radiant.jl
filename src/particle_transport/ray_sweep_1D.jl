"""
    ray_sweep_1D.jl

1D characteristic-line (MOC) sweep for the FCS uncollided path.
Supports:
- Stage 1: 𝒪x=1, DG energy scheme, void boundaries.
- Stage 2a/2b: 𝒪x=2 high-order spatial moments with polynomial interface source
  for BFP/CSD, plus homogeneous high-order moments for BTE.

The signature mirrors `sn_sweep_1D` so it can be used as a drop-in alternative
in `_compute_uncollided_surface_flux` when `use_ray_sweep` is enabled.
"""

"""
    _ray_cell_kernel_1D(A, Δs, ΦE_in, q_eff; ϵ_sing=1e-12, τ_root=1e-6)

FSA characteristic-line solve over a segment of length `Δs`.
Returns the outgoing energy coefficients at `σ=Δs` and the segment integral
`∫₀^{Δs} Φ(σ) dσ`.

All closed-form; no per-point matrix-exponential evaluation.
"""
function _ray_cell_kernel_1D(A::AbstractMatrix{Float64}, Δs::Float64,
                             ΦE_in::Vector{Float64}, q_eff::Vector{Float64};
                             ϵ_sing::Float64=1e-12, τ_root::Float64=1e-6)
    N = length(ΦE_in)
    if N == 1
        Σt = A[1,1]
        E = exp(-Σt * Δs)
        if Σt * Δs > 1e-12
            F = -expm1(-Σt * Δs) / Σt
            ΦE_int = [F * ΦE_in[1] + (Δs - F) / Σt * q_eff[1]]
        else
            F = Δs * (1.0 - 0.5 * Σt * Δs)
            ΦE_int = [F * ΦE_in[1] + (Δs * Δs / 2.0) * q_eff[1]]
        end
        ΦE_out = [E * ΦE_in[1] + F * q_eff[1]]
        return ΦE_out, ΦE_int
    else
        E, F, Ainv, det = ray_moc_matrices_2x2(A, Δs; ϵ_sing=ϵ_sing, τ_root=τ_root)
        ΦE_out = E * ΦE_in + F * q_eff
        ΦE_int = F * ΦE_in + Ainv * ((Δs * I - F) * q_eff)
        return ΦE_out, ΦE_int
    end
end

"""
    _project_to_moments_1D(ΦE_int, Δs, 𝒪E, Nm)

Stage-1 projection (𝒪x=1): constant spatial moment `M_{k,1} = (1/Δs) ∫Φ_k dσ`.
Maps onto `Nm[5]=𝒪E` moments.
"""
function _project_to_moments_1D(ΦE_int::Vector{Float64}, Δs::Float64,
                                𝒪E::Int, Nm::Vector{Int})
    𝚽n = zeros(Nm[5])
    @inbounds for k in 1:𝒪E
        𝚽n[k] = ΦE_int[k] / Δs
    end
    return 𝚽n
end

"""
    ray_sweep_1D(...)

1D characteristic-line sweep.  Signature-compatible with `sn_sweep_1D`.

Restrictions (enforced):
- `𝒪[1] ∈ {1,2}` (spatial order 1 or 2)
- DG energy scheme
- void boundaries
- if `isCSD`, `𝒪[4] ≥ 2`
"""
function ray_sweep_1D(𝚽l::AbstractArray{Float64,3},Ql::AbstractArray{Float64,3},
                      Σt::Vector{Float64},mat::Vector{Int64},Nx::Int64,Δx::Vector{Float64},
                      μ::Float64,Mn::Vector{Float64},Dn::Vector{Float64},Np::Int64,
                      Mnx⁻::Vector{Float64},Dnx⁻::Vector{Float64},Np_surf::Int64,
                      𝒪::Vector{Int64},Nm::Vector{Int64},C::Vector{Float64},
                      ω::Vector{Array{Float64}},sources::Matrix{Union{Float64, Array{Float64}}},
                      isAdapt::Bool,isCSD::Bool,ΔE::Float64,𝚽E12::AbstractArray{Float64},
                      S⁻::Vector{Float64},S⁺::Vector{Float64},S::Array{Float64},
                      𝒲::Array{Float64},isFC::Bool,𝚽x12⁻::Array{Float64,3},
                      boundary_conditions::Vector{Int64},Np_source)

    @assert 𝒪[1] ∈ (1, 2) "ray_sweep_1D only supports 𝒪x ∈ {1,2}."
    @assert all(boundary_conditions .== 0) "ray_sweep_1D only supports void boundary."
    @assert (!isCSD) || (𝒪[4] ≥ 2) "ray_sweep_1D with CSD requires 𝒪E≥2."

    𝒪x = 𝒪[1]
    𝒪E = 𝒪[4]
    sx = sign(μ)
    x_sweep = (μ ≥ 0) ? (1:Nx) : (Nx:-1:1)

    𝚽x12 = zeros(Nm[1])
    𝚽x12⁺ = zeros(Np_surf, Nm[1], 2)

    # Incident face source (void boundaries only)
    incident_face = (μ ≥ 0) ? 1 : 2
    for p in 1:Np_source
        𝚽x12[1] += Mnx⁻[p] * sources[p, incident_face]
    end

    for ix in x_sweep
        Δs = Δx[ix] / abs(μ)

        ΦE_in = copy(𝚽x12)

        if isCSD
            β1, β2 = S[mat[ix], 1], S[mat[ix], 2]
            A = _build_A_matrix(Σt[mat[ix]], β1, β2)
            phiE = view(𝚽E12, 1:𝒪x, ix)
            beta_up = β1 + sqrt(3.0) * β2

            if 𝒪x == 1
                phi_prev_group_out = phiE[1]
                q_eff = _build_boundary_source(β1, β2, phi_prev_group_out; scheme_E="DG")
                ΦE_out, ΦE_integral = _ray_cell_kernel_1D(A, Δs, ΦE_in, q_eff)

                𝚽n = _project_to_moments_1D(ΦE_integral, Δs, 𝒪E, Nm)
                for is in 1:Nm[5], p in 1:Np
                    𝚽l[p, is, ix] += Dn[p] * 𝚽n[is]
                end

                𝚽x12 = copy(ΦE_out)
                M = ΦE_integral ./ Δs
                𝚽E12[1, ix] = M[1] - sqrt(3.0) * M[2]
            else
                E, F, Ainv, det = ray_moc_matrices_2x2(A, Δs)
                B = A * Δs
                M = ray_moments_Ox2(E, F, Ainv, B, Δs, ΦE_in, phiE, beta_up, sx)
                𝚽n = project_M_to_Phi_n(M, 𝒪x, 𝒪E, isFC)
                for is in 1:Nm[5], p in 1:Np
                    𝚽l[p, is, ix] += Dn[p] * 𝚽n[is]
                end

                𝚽x12 = ray_outgoing_flux_Ox2(E, F, Ainv, B, Δs, ΦE_in, phiE, beta_up, sx)
                for jx in 1:𝒪x
                    𝚽E12[jx, ix] = M[1, jx] - sqrt(3.0) * M[2, jx]
                end
            end
        else
            A = reshape([Σt[mat[ix]]], 1, 1)
            if 𝒪x == 1
                q_eff = zeros(1)
                ΦE_out, ΦE_integral = _ray_cell_kernel_1D(A, Δs, ΦE_in, q_eff)

                𝚽n = _project_to_moments_1D(ΦE_integral, Δs, 𝒪E, Nm)
                for is in 1:Nm[5], p in 1:Np
                    𝚽l[p, is, ix] += Dn[p] * 𝚽n[is]
                end

                𝚽x12 = copy(ΦE_out)
            else
                # BTE with 𝒪x=2: homogeneous high-order spatial moments, no interface source.
                Σt_val = Σt[mat[ix]]
                E = reshape([exp(-Σt_val * Δs)], 1, 1)
                F = reshape([-expm1(-Σt_val * Δs) / Σt_val], 1, 1)
                Ainv = reshape([1.0 / Σt_val], 1, 1)
                B = A * Δs
                q_eff = zeros(1)
                M = ray_moments_constant_source(E, F, Ainv, B, Δs, ΦE_in, q_eff, sx, 𝒪x)
                𝚽n = project_M_to_Phi_n(M, 𝒪x, 𝒪E, isFC)
                for is in 1:Nm[5], p in 1:Np
                    𝚽l[p, is, ix] += Dn[p] * 𝚽n[is]
                end

                𝚽x12 = E * ΦE_in
            end
        end
    end

    # Save boundary fluxes
    for p in 1:Np_surf, is in 1:Nm[1]
        if μ ≥ 0
            𝚽x12⁺[p, is, 2] += Dnx⁻[p] * 𝚽x12[is]
        else
            𝚽x12⁺[p, is, 1] += Dnx⁻[p] * 𝚽x12[is]
        end
    end

    return 𝚽l, 𝚽E12, 𝚽x12⁺
end
