"""
    ray_sweep_3D.jl

3D 特征线 (MOC) 扫描，用于 FCS 未碰撞表面通量路径，仅支持沿坐标轴入射的面源。
轴对齐入射 ⇒ 无横向输运耦合 ⇒ 解析退化为逐横向柱的独立 1D MOC 扫描。
复用 ray_moc_kernel.jl。签名与 sn_sweep_3D 兼容。
"""

"""
    _is_axis_aligned(Ω; tol=1e-9) -> Bool

判断方向是否恰好沿某坐标轴（恰一个分量 |·|≈1、其余≈0）。
"""
function _is_axis_aligned(Ω::AbstractVector{Float64}; tol::Float64=1e-9)
    nz = findall(c -> abs(c) > tol, Ω)
    return length(nz) == 1 && abs(abs(Ω[nz[1]]) - 1.0) ≤ tol
end

"""
    _ray_sweep_axis(Ω; tol=1e-9) -> (d, s)

返回轴对齐入射的扫描轴 `d ∈ {1,2,3}` 与符号 `s = sign(Ω[d]) = ±1`；
非轴对齐则报错。
"""
function _ray_sweep_axis(Ω::AbstractVector{Float64}; tol::Float64=1e-9)
    if !_is_axis_aligned(Ω; tol=tol)
        error("ray_sweep_3D only supports axis-aligned beams; got Ω=$(Ω).")
    end
    d = findfirst(c -> abs(c) > tol, Ω)
    return d, sign(Ω[d])
end

"""
    _ray3d_applicable(use_ray_sweep, Ω, 𝒪, is_CSD) -> Bool

Whether the 3D characteristic (MOC) sweep `ray_sweep_3D` is usable for this beam.
True only when the ray sweep is requested AND every `ray_sweep_3D` prerequisite
holds: axis-aligned direction, transverse spatial orders == 1, along-axis order
∈ {1,2}, and (for CSD) energy order == 2.  Any other configuration must fall back
to the standard SN sweep rather than tripping an assertion.
"""
function _ray3d_applicable(use_ray_sweep::Bool, Ω::AbstractVector{Float64},
                           𝒪::Vector{Int64}, is_CSD::Bool; tol::Float64=1e-9)
    use_ray_sweep || return false
    _is_axis_aligned(Ω; tol=tol) || return false
    d = findfirst(c -> abs(c) > tol, Ω)
    all(𝒪[setdiff(1:3, [d])] .== 1) || return false   # transverse order == 1
    (𝒪[d] ∈ (1, 2)) || return false                   # along-axis order ∈ {1,2}
    (!is_CSD || 𝒪[4] == 2) || return false            # CSD energy order == 2
    return true
end

"""
    project_M_to_Phi3D_n(M, 𝒪, isFC, sweep_axis, Nm5) -> 𝚽n::Vector

把逐柱矩 `M[k, j]`（`k=1..𝒪E` 能量矩，`j=1..𝒪[sweep_axis]` 沿轴空间矩，
横向阶恒为 1）按 `map_moments` 索引投影到长度 `Nm5` 的 3D 矩向量；
横向高阶矩留 0。
"""
function project_M_to_Phi3D_n(M::AbstractMatrix{Float64}, 𝒪::Vector{Int64},
                              isFC::Bool, sweep_axis::Int, Nm5::Int)
    𝒪E = 𝒪[4]; 𝒪d = 𝒪[sweep_axis]
    𝚽n = zeros(Nm5)
    for j in 1:𝒪d, k in 1:𝒪E
        jx = jy = jz = 1
        if sweep_axis == 1
            jx = j
        elseif sweep_axis == 2
            jy = j
        else
            jz = j
        end
        if isFC
            is = 𝒪[2]*𝒪[1]*𝒪[4]*(jz-1) + 𝒪[1]*𝒪[4]*(jy-1) + 𝒪[4]*(jx-1) + k
        else
            if j > 1 && k > 1; continue end          # 轴纯矩：j>1 ⇒ k==1
            is = 1 + (k-1) + (jx-1) + (jy-1) + (jz-1)
            if jx > 1; is += 𝒪[4]-1 end
            if jy > 1; is += 𝒪[4]-1 + 𝒪[1]-1 end
            if jz > 1; is += 𝒪[4]-1 + 𝒪[1]-1 + 𝒪[2]-1 end
        end
        𝚽n[is] = M[k, j]
    end
    return 𝚽n
end

@inline cell_index(d, idx, it1, it2) =
    d == 1 ? (idx, it1, it2) : d == 2 ? (it1, idx, it2) : (it1, it2, idx)

@inline function _accumulate!(𝚽l, M, 𝒪, isFC, d, Nm5, Dn, P, ix, iy, iz)
    𝚽n = project_M_to_Phi3D_n(M, 𝒪, isFC, d, Nm5)
    @inbounds for is in 1:Nm5, p in 1:P
        𝚽l[p, is, ix, iy, iz] += Dn[p] * 𝚽n[is]
    end
end

# BTE 沿轴高阶矩（横向阶=1）：复用 ray_moments_constant_source（q_eff=0）。
# 返回更新后的出射能量系数 ΦE_out = E·ΦE_in。
function _bte_highorder_cell!(𝚽l, ΦE_in, Σt_val, Δseg, 𝒪, isFC, d, s, Nm, Dn, P, ix, iy, iz)
    A    = reshape([Σt_val], 1, 1)
    E    = reshape([exp(-Σt_val*Δseg)], 1, 1)
    F    = reshape([-expm1(-Σt_val*Δseg)/Σt_val], 1, 1)
    Ainv = reshape([1.0/Σt_val], 1, 1)
    B    = A * Δseg
    M    = ray_moments_constant_source(E, F, Ainv, B, Δseg, ΦE_in, zeros(1), s, 𝒪[d])
    _accumulate!(𝚽l, M, 𝒪, isFC, d, Nm[5], Dn, P, ix, iy, iz)
    return E * ΦE_in
end

# BFP/CSD 沿轴矩（横向阶=1）。2×2 有效反应矩阵 + DG 下散射界面源。
# 原位更新 𝚽E12[:,ix,iy,iz]，返回出射能量系数 ΦE_out。
function _csd_cell!(𝚽l, 𝚽E12, ΦE_in, Σt_val, S, mat, Δseg, 𝒪, isFC, d, s, Nm, Dn, P, ix, iy, iz)
    β1, β2 = S[mat[ix,iy,iz], 1], S[mat[ix,iy,iz], 2]
    A = _build_A_matrix(Σt_val, β1, β2)              # 2×2
    if 𝒪[d] == 1
        φ_prev = 𝚽E12[1, ix, iy, iz]
        q_eff  = _build_boundary_source(β1, β2, φ_prev; scheme_E="DG")   # 长度 2
        ΦE_out, ΦE_int = _ray_cell_kernel_1D(A, Δseg, ΦE_in, q_eff)      # ΦE_in 长度 𝒪E==2
        M = reshape(ΦE_int ./ Δseg, :, 1)
        _accumulate!(𝚽l, M, 𝒪, isFC, d, Nm[5], Dn, P, ix, iy, iz)
        𝚽E12[1, ix, iy, iz] = M[1,1] - sqrt(3.0)*M[2,1]
        return ΦE_out
    else                                             # 𝒪_d=2
        beta_up = β1 + sqrt(3.0)*β2
        phiE = view(𝚽E12, 1:𝒪[d], ix, iy, iz)        # R3 ⇒ 即沿轴矩
        E, F, Ainv, det = ray_moc_matrices_2x2(A, Δseg)
        B = A * Δseg
        M = ray_moments_Ox2(E, F, Ainv, B, Δseg, ΦE_in, phiE, beta_up, s)
        _accumulate!(𝚽l, M, 𝒪, isFC, d, Nm[5], Dn, P, ix, iy, iz)
        ΦE_out = ray_outgoing_flux_Ox2(E, F, Ainv, B, Δseg, ΦE_in, phiE, beta_up, s)
        for j in 1:𝒪[d]
            𝚽E12[j, ix, iy, iz] = M[1,j] - sqrt(3.0)*M[2,j]
        end
        return ΦE_out
    end
end

# 仅沿轴出口面非零。注：驱动层 Dn_surf 恒为 0（sn_flux_fcs.jl:360-362），
# FCS 路径下此处写入恒为 0，出射面通量当前未被 FCS 使用（与 1D ray_sweep 一致）。
# 保留接口以兼容 sn_sweep_3D。
function store_axis_exit_flux!(𝚽x12⁺, 𝚽y12⁺, 𝚽z12⁺, d, s, ΦE_out, Dn_surf, Np_surf, it1, it2)
    side = (s ≥ 0) ? 2 : 1
    tgt = d == 1 ? 𝚽x12⁺ : d == 2 ? 𝚽y12⁺ : 𝚽z12⁺
    @inbounds for p in 1:Np_surf, is in 1:length(ΦE_out)
        tgt[p, is, side, it1, it2] += Dn_surf[p] * ΦE_out[is]
    end
end

"""
    ray_sweep_3D(...)

3D 特征线 (MOC) 扫描，签名与 `sn_sweep_3D` 兼容，仅支持沿坐标轴入射的面源。
逐横向柱做独立 1D MOC（复用 ray_moc_kernel.jl）。

约束（`@assert` 强制）：
- 轴对齐入射（`_ray_sweep_axis`）
- 沿轴空间阶 `𝒪[d] ∈ {1,2}`
- 真空边界
- CSD 时 `𝒪[4] == 2`
- 横向空间阶 == 1
"""
function ray_sweep_3D(𝚽l::AbstractArray{Float64,5}, Ql, Σt::Vector{Float64},
                      mat::Array{Int64,3}, Ns::Vector{Int64},
                      Δs::Vector{Vector{Float64}}, Ω::Vector{Float64},
                      Mn::Vector{Float64}, Dn::Vector{Float64}, P::Int64,
                      Mnx⁻, Dnx⁻, Mny⁻, Dny⁻, Mnz⁻, Dnz⁻, Np_surf::Int64,
                      𝒪::Vector{Int64}, Nm::Vector{Int64}, C::Vector{Float64},
                      ω::Vector{Array{Float64}}, sources, isAdapt::Bool,
                      isCSD::Bool, ΔE::Float64, 𝚽E12::AbstractArray{Float64},
                      S⁻::Vector{Float64}, S⁺::Vector{Float64}, S::Array{Float64},
                      𝒲::Array{Float64}, isFC::Bool,
                      𝚽x12⁻, 𝚽y12⁻, 𝚽z12⁻, boundary_conditions, Np_source)

    d, s = _ray_sweep_axis(Ω)
    @assert 𝒪[d] ∈ (1, 2) "ray_sweep_3D supports along-axis order ∈ {1,2}."
    @assert all(boundary_conditions .== 0) "ray_sweep_3D supports void boundary only."
    @assert (!isCSD) || (𝒪[4] == 2) "ray_sweep_3D CSD path requires 𝒪E == 2."
    let t = setdiff(1:3, [d]); @assert all(𝒪[t] .== 1) "ray_sweep_3D requires transverse spatial orders == 1." end
    @assert Np_source == 1 "ray_sweep_3D assumes Np_source == 1 (FCS beam)."

    Nx, Ny, Nz = Ns
    𝚽x12⁺ = zeros(Np_surf, Nm[1], 2, Ny, Nz)
    𝚽y12⁺ = zeros(Np_surf, Nm[2], 2, Nx, Nz)
    𝚽z12⁺ = zeros(Np_surf, Nm[3], 2, Nx, Ny)

    incident_face = 2*(d-1) + (s ≥ 0 ? 1 : 2)
    Mn_surf = (d==1) ? Mnx⁻ : (d==2) ? Mny⁻ : Mnz⁻
    Dn_surf = (d==1) ? Dnx⁻ : (d==2) ? Dny⁻ : Dnz⁻
    Nd = Ns[d]; Δd = Δs[d]
    axis_sweep = (s ≥ 0) ? (1:Nd) : (Nd:-1:1)
    t1, t2 = (d==1) ? (2,3) : (d==2) ? (1,3) : (1,2)

    for it1 in 1:Ns[t1], it2 in 1:Ns[t2]
        Φin = 0.0
        src = sources[1, incident_face][it1, it2]
        for p in 1:Np_source; Φin += Mn_surf[p] * src end
        if isCSD
            ΦE_in = zeros(𝒪[4]); ΦE_in[1] = Φin       # R1
        else
            ΦE_in = [Φin]
        end

        for idx in axis_sweep
            ix, iy, iz = cell_index(d, idx, it1, it2)
            Σt_val = Σt[mat[ix,iy,iz]]; Δseg = Δd[idx]
            if !isCSD && 𝒪[d] == 1
                A = reshape([Σt_val], 1, 1)
                ΦE_out, ΦE_int = _ray_cell_kernel_1D(A, Δseg, ΦE_in, zeros(1))
                M = reshape([ΦE_int[1] / Δseg], 1, 1)
                _accumulate!(𝚽l, M, 𝒪, isFC, d, Nm[5], Dn, P, ix, iy, iz)
                ΦE_in = ΦE_out
            elseif !isCSD
                ΦE_in = _bte_highorder_cell!(𝚽l, ΦE_in, Σt_val, Δseg, 𝒪, isFC, d, s, Nm, Dn, P, ix, iy, iz)
            else
                ΦE_in = _csd_cell!(𝚽l, 𝚽E12, ΦE_in, Σt_val, S, mat, Δseg, 𝒪, isFC, d, s, Nm, Dn, P, ix, iy, iz)
            end
        end

        store_axis_exit_flux!(𝚽x12⁺, 𝚽y12⁺, 𝚽z12⁺, d, s, ΦE_in, Dn_surf, Np_surf, it1, it2)
    end
    return 𝚽l, 𝚽E12, 𝚽x12⁺, 𝚽y12⁺, 𝚽z12⁺
end
