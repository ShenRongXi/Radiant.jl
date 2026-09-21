# First Collision Source (FCS) extension for the SN solver.

const _FACE_INDEX = Dict("X-"=>1, "X+"=>2, "Y-"=>3, "Y+"=>4, "Z-"=>5, "Z+"=>6)

"""
    build_angular_discretization(solver::SN,geometry::Geometry)

Build the angular discretization associated with `solver` and `geometry`.
The result is cached in an `SN_Angular_Discretization` object and can be reused
by both the standard `compute_flux` path and the FCS path.

# Input Argument(s)
- `solver::SN` : discrete ordinates informations.
- `geometry::Geometry` : geometry informations.

# Output Argument(s)
- `sn_angle_discre::SN_Angular_Discretization` : angular discretization data.

# Reference(s)
N/A

"""
function build_angular_discretization(solver::SN,geometry::Geometry)
    Ndims = geometry.get_dimension()
    geo_type = geometry.get_type()
    L = solver.get_legendre_order()
    N = solver.get_quadrature_order()
    quadrature_type = solver.get_quadrature_type()
    SN_type = solver.get_angular_boltzmann()
    Qdims = solver.get_quadrature_dimension(Ndims)

    Ω,w = quadrature(N,quadrature_type,Ndims,Qdims)
    if typeof(Ω) == Vector{Float64} Ω = [Ω,0*Ω,0*Ω] end
    Nd = length(w)
    Np,Mn,Dn,pl,pm = angular_polynomial_basis(Ω,w,L,SN_type,Qdims)
    if SN_type == "galerkin-direct"
        if Ndims != 1 || Qdims != 1 || quadrature_type != "gauss-lobatto"
            error("galerkin-direct surface sources require 1D Gauss-Lobatto quadrature.")
        end
        Np_surf,Mn_surf,Dn_surf,n⁺_to_n,n_to_n⁺,pl_surf,pm_surf = direct_surface_angular_basis_1D(Ω)
    else
        Np_surf,Mn_surf,Dn_surf,n⁺_to_n,n_to_n⁺,pl_surf,pm_surf = surface_angular_polynomial_basis(Ω,w,L,SN_type,Qdims,Ndims,geo_type)
    end
    return SN_Angular_Discretization(Ω,w,Nd,Np,Mn,Dn,pl,pm,Np_surf,Mn_surf,Dn_surf,n⁺_to_n,n_to_n⁺,SN_type,Qdims)
end

"""
    _beam_basis_coefficients(Ωs::Vector{Float64},loc::String,
    sn_angle_discre::SN_Angular_Discretization,SN_type::String)

Compute the beam-direction projection coefficients onto the solver angular basis.

# Input Argument(s)
- `Ωs::Vector{Float64}` : beam direction cosines [μ,η,ξ].
- `loc::String` : face location (uppercased).
- `sn_angle_discre::SN_Angular_Discretization` : precomputed angular basis.
- `SN_type::String` : angular Boltzmann discretization.

# Output Argument(s)
- `Dn_beam::Vector{Float64}` : full-range basis evaluated at Ωs (B_p(Ωs)), shape `(Np,)`.
- `M_beam::Vector{Float64}` : integral-preserving zeroth half-range moment, shape `(1,)`.

# Reference(s)
N/A

"""
function _beam_basis_coefficients(Ωs::Vector{Float64},loc::String,
                                  sn_angle_discre::SN_Angular_Discretization,SN_type::String)
    μs,ηs,ξs = Ωs
    ϕs = atan(ξs,ηs)

    # Full-range coefficients
    Lmax = maximum(sn_angle_discre.pl)
    if sn_angle_discre.Qdims == 1
        Pls = legendre_polynomials_up_to_L(Lmax,μs)
        Dn_beam = [Pls[sn_angle_discre.pl[p]+1] for p in range(1,sn_angle_discre.Np)]
    else
        Ylms = real_spherical_harmonics_up_to_L(Lmax,μs,ϕs)
        Dn_beam = [Ylms[sn_angle_discre.pl[p]+1][sn_angle_discre.pm[p]+sn_angle_discre.pl[p]+1]
                   for p in range(1,sn_angle_discre.Np)]
    end

    # Half-range coefficient: single zeroth moment carrying the incident flux density
    # [particles/(cm²·s)].  M_beam = [1.0] is the identity conversion for a mono-directional
    # beam whose source intensity is already the angular flux value at the incident face.
    M_beam = [1.0]

    return Dn_beam,M_beam
end

"""
    _build_sources_ig(intensity,loc,eg,ig,Ndims,Ns,Δs,geometry::Geometry,boundaries::Dict{String,Vector{Float64}})

Build the per-group surface source matrix `(Np_surf, 2*Ndims)` for the FCS uncollided
sweep, with `Np_surf=1` (zeroth half-range moment only). Non-zero only when `ig == eg`,
on the face given by `loc`, storing the incident flux density [particles/(cm²·s)].
In 2D/3D, only voxels inside `boundaries` receive source intensity, matching the
standard `surface_source()` filtering logic.

# Input Argument(s)
- `intensity::Float64` : beam intensity I₀.
- `loc::String` : face location (uppercased, e.g. "X-").
- `eg::Int64` : energy group of the source.
- `ig::Int64` : current energy group.
- `Ndims::Int64` : geometry dimension.
- `Ns::Vector{Int64}` : number of voxels per axis.
- `Δs::Vector{Vector{Float64}}` : voxels width per axis.
- `geometry::Geometry` : geometry informations (used to fetch voxel center positions).
- `boundaries::Dict{String,Vector{Float64}}` : source boundaries per axis.

# Output Argument(s)
- `sources_ig::Matrix{Union{Float64,Array{Float64}}}` : per-group source matrix, shape `(1, 2*Ndims)`.

# Reference(s)
N/A

"""
function _build_sources_ig(intensity,loc,eg,ig,Ndims,Ns,Δs,geometry::Geometry,boundaries::Dict{String,Vector{Float64}})
    sources_ig = Matrix{Union{Float64,Array{Float64}}}(undef,1,2*Ndims)
    # Initialize all faces to zero
    for face in range(1,2*Ndims)
        if Ndims == 1
            sources_ig[1,face] = 0.0
        elseif Ndims == 2
            if face ∈ [1,2] sources_ig[1,face] = zeros(Ns[2]) else sources_ig[1,face] = zeros(Ns[1]) end
        else
            if face ∈ [1,2] sources_ig[1,face] = zeros(Ns[2],Ns[3])
            elseif face ∈ [3,4] sources_ig[1,face] = zeros(Ns[1],Ns[3])
            else sources_ig[1,face] = zeros(Ns[1],Ns[2]) end
        end
    end
    # Inject intensity on the incident face when ig == eg
    if ig == eg
        face_idx = _FACE_INDEX[loc]
        if Ndims == 1
            sources_ig[1,face_idx] = intensity
        elseif Ndims == 2
            if loc ∈ ["X-","X+"]
                ymin = boundaries["y"][1]; ymax = boundaries["y"][2]
                y = geometry.get_voxels_position("y")
                s = zeros(Ns[2])
                for iy in range(1,Ns[2])
                    if y[iy] < ymin || y[iy] > ymax continue end
                    s[iy] = intensity
                end
                sources_ig[1,face_idx] = s
            else
                xmin = boundaries["x"][1]; xmax = boundaries["x"][2]
                x = geometry.get_voxels_position("x")
                s = zeros(Ns[1])
                for ix in range(1,Ns[1])
                    if x[ix] < xmin || x[ix] > xmax continue end
                    s[ix] = intensity
                end
                sources_ig[1,face_idx] = s
            end
        else
            if loc ∈ ["X-","X+"]
                ymin = boundaries["y"][1]; ymax = boundaries["y"][2]
                zmin = boundaries["z"][1]; zmax = boundaries["z"][2]
                y = geometry.get_voxels_position("y")
                z = geometry.get_voxels_position("z")
                s = zeros(Ns[2],Ns[3])
                for iy in range(1,Ns[2]), iz in range(1,Ns[3])
                    if y[iy] < ymin || y[iy] > ymax continue end
                    if z[iz] < zmin || z[iz] > zmax continue end
                    s[iy,iz] = intensity
                end
                sources_ig[1,face_idx] = s
            elseif loc ∈ ["Y-","Y+"]
                xmin = boundaries["x"][1]; xmax = boundaries["x"][2]
                zmin = boundaries["z"][1]; zmax = boundaries["z"][2]
                x = geometry.get_voxels_position("x")
                z = geometry.get_voxels_position("z")
                s = zeros(Ns[1],Ns[3])
                for ix in range(1,Ns[1]), iz in range(1,Ns[3])
                    if x[ix] < xmin || x[ix] > xmax continue end
                    if z[iz] < zmin || z[iz] > zmax continue end
                    s[ix,iz] = intensity
                end
                sources_ig[1,face_idx] = s
            else
                xmin = boundaries["x"][1]; xmax = boundaries["x"][2]
                ymin = boundaries["y"][1]; ymax = boundaries["y"][2]
                x = geometry.get_voxels_position("x")
                y = geometry.get_voxels_position("y")
                s = zeros(Ns[1],Ns[2])
                for ix in range(1,Ns[1]), iy in range(1,Ns[2])
                    if x[ix] < xmin || x[ix] > xmax continue end
                    if y[iy] < ymin || y[iy] > ymax continue end
                    s[ix,iy] = intensity
                end
                sources_ig[1,face_idx] = s
            end
        end
    end
    return sources_ig
end

"""
    _compute_nm_from_o(𝒪::Vector{Int64}, isFC::Bool)

Compute the number of moments `Nm` from the order vector `𝒪` for the non-FC or FC
indexing convention. This mirrors the moment-count logic in `SN.get_schemes` but does
not require a `Geometry` object.
"""
function _compute_nm_from_o(𝒪::Vector{Int64}, isFC::Bool)
    if isFC
        Nm = [𝒪[2]*𝒪[3]*𝒪[4], 𝒪[1]*𝒪[3]*𝒪[4], 𝒪[1]*𝒪[2]*𝒪[4], 𝒪[1]*𝒪[2]*𝒪[3], prod(𝒪)]
    else
        Nm = [1+(𝒪[2]-1)+(𝒪[3]-1)+(𝒪[4]-1),
              1+(𝒪[1]-1)+(𝒪[3]-1)+(𝒪[4]-1),
              1+(𝒪[1]-1)+(𝒪[2]-1)+(𝒪[4]-1),
              1+(𝒪[1]-1)+(𝒪[2]-1)+(𝒪[3]-1),
              1+sum(𝒪.-1)]
    end
    return Nm
end

"""
    _compute_uncollided_surface_flux(cross_sections::Cross_Sections,geometry::Geometry,
    solver::SN,source::Source,sn_angle_discre::SN_Angular_Discretization;
    T::Array{Float64,2}=Array{Float64}(undef,0,0),λ₀::Float64=0.0)

Compute the uncollided flux driven by surface (beam) sources, by sweeping each beam
direction directly through `sn_sweep_*D` (no scattering iteration).  When
`use_ray_sweep` is true and the geometry is 1D, the optional characteristic-line
sweep `ray_sweep_1D` is used instead.

For BFP/FP solvers the uncollided sweep uses the same augmented total cross-section
(`Σ_t + T * λ₀`) as the standard SN solve, so that the Fokker-Planck diagonal
stabilization factor cancels between the uncollided and scattered fluxes.

# Input Argument(s)
- `cross_sections::Cross_Sections` : cross section informations.
- `geometry::Geometry` : geometry informations.
- `solver::SN` : discrete ordinates informations.
- `source::Source` : source carrying original `surface_source_objects`.
- `sn_angle_discre::SN_Angular_Discretization` : precomputed angular basis.
- `T::Array{Float64,2}` : momentum transfer cross-sections `(Ng,Nmat)` (optional, for BFP/FP).
- `λ₀::Float64` : Fokker-Planck diagonal stabilization factor (optional, for BFP/FP).

# Output Argument(s)
- `φ_u_surf::Array{Float64,6}` : uncollided flux moments, shape `(Ng,Np,Nm[5],Ns[1],Ns[2],Ns[3])`.
- `𝚽cutoff_u_surf::Array{Float64,5}` : uncollided cutoff flux moments, shape `(Np,Nm[5],Ns[1],Ns[2],Ns[3])`.

# Reference(s)
N/A

"""
function _compute_uncollided_surface_flux(cross_sections::Cross_Sections,geometry::Geometry,
                                          solver::SN,source::Source,
                                          sn_angle_discre::SN_Angular_Discretization;
                                          T::Array{Float64,2}=Array{Float64}(undef,0,0),
                                          λ₀::Float64=0.0,
                                          use_ray_sweep::Bool=true)

    Ndims = geometry.get_dimension()
    Ns = geometry.get_number_of_voxels()
    Δs = geometry.get_voxels_width()
    mat = geometry.get_material_per_voxel()

    part = solver.get_particle()
    solver_type,is_CSD = solver.get_solver_type()
    isFC = solver.get_is_full_coupling()
    schemes,𝒪,Nm = solver.get_schemes(geometry,isFC)
    ω,𝒞,is_adaptive,𝒲 = scheme_weights(𝒪,schemes,Ndims,is_CSD)

    𝒪_ray_order = solver.get_fcs_ray_spatial_order()
    if 𝒪_ray_order != 0 && 𝒪_ray_order ∉ (1, 2)
        error("fcs_ray_spatial_order must be 0 (inherit solver order), 1 or 2.")
    end
    mixed_order = (Ndims == 1) && use_ray_sweep && (𝒪_ray_order ≥ 1) && (𝒪_ray_order < 𝒪[1])
    if mixed_order
        𝒪_ray = copy(𝒪)
        𝒪_ray[1] = 𝒪_ray_order
        Nm_ray = _compute_nm_from_o(𝒪_ray, isFC)
    end

    Nmat = cross_sections.get_number_of_materials()
    Ng = cross_sections.get_number_of_groups(part)

    # Total cross-section used in the uncollided sweep. For BFP/FP this must match the
    # augmented total used in the standard SN solve so that the FP stabilization factor
    # cancels between the uncollided and scattered fluxes.
    Σtot = zeros(Ng,Nmat)
    if solver_type ∈ [4,5]
        Σtot = cross_sections.get_absorption(part)
    else
        Σtot = cross_sections.get_total(part)
    end
    if solver_type ∈ [2,4]
        Σtot .+= T .* λ₀
    end

    if is_CSD
        ΔE = cross_sections.get_energy_width(part)
        Eb = cross_sections.get_energy_boundaries(part)
        Sb = cross_sections.get_boundary_stopping_powers(part)
        S⁻ = zeros(Ng,Nmat); S⁺ = zeros(Ng,Nmat)
        for n in range(1,Nmat)
            S⁻[:,n] = Sb[1:Ng,n] ; S⁺[:,n] = Sb[2:Ng+1,n]
        end
        S = zeros(Ng,Nmat,𝒪[4])
        for n in range(1,Nmat), ig in range(1,Ng)
            S[ig,n,1] = (S⁻[ig,n]+S⁺[ig,n])/2
            if (𝒪[4] > 1) S[ig,n,2] = (S⁻[ig,n]-S⁺[ig,n])/(2*sqrt(3)) end
        end
    end

    φ_u_surf = zeros(Ng,sn_angle_discre.Np,Nm[5],Ns[1],Ns[2],Ns[3])
    𝚽cutoff_u_surf = zeros(sn_angle_discre.Np,Nm[5],Ns[1],Ns[2],Ns[3])
    boundary_conditions = zeros(Int64,2*Ndims)
    Np = sn_angle_discre.Np

    for ss in source.get_surface_source_objects()
        Ωs = ss.direction
        loc = uppercase(ss.location)
        intensity = ss.intensity
        eg = ss.energy_group

        # Beam direction must enter through loc face
        if (loc == "X-" && Ωs[1] ≤ 0) || (loc == "X+" && Ωs[1] ≥ 0) ||
           (loc == "Y-" && Ωs[2] ≤ 0) || (loc == "Y+" && Ωs[2] ≥ 0) ||
           (loc == "Z-" && Ωs[3] ≤ 0) || (loc == "Z+" && Ωs[3] ≥ 0)
            error("Beam direction $(Ωs) does not enter the geometry through face $loc.")
        end

        Dn_beam,M_beam = _beam_basis_coefficients(Ωs,loc,sn_angle_discre,solver.get_angular_boltzmann())
        Np_surf = length(M_beam)

        Mnx⁻_beam = zeros(Np_surf)
        Mny⁻_beam = zeros(Np_surf)
        Mnz⁻_beam = zeros(Np_surf)
        if loc ∈ ["X-","X+"]
            Mnx⁻_beam = M_beam
        elseif loc ∈ ["Y-","Y+"]
            Mny⁻_beam = M_beam
        elseif loc ∈ ["Z-","Z+"]
            Mnz⁻_beam = M_beam
        end
        Dnx⁻_beam = zeros(Np_surf)
        Dny⁻_beam = zeros(Np_surf)
        Dnz⁻_beam = zeros(Np_surf)

        # Per-beam CSD energy-boundary flux (no Nd axis; shape depends on Ndims)
        if is_CSD
            if Ndims == 1
                𝚽E12_beam = zeros(Nm[4],Ns[1])
            elseif Ndims == 2
                𝚽E12_beam = zeros(Nm[4],Ns[1],Ns[2])
            else
                𝚽E12_beam = zeros(Nm[4],Ns[1],Ns[2],Ns[3])
            end
        else
            𝚽E12_beam = Array{Float64}(undef)
        end

        # Void incoming boundary fluxes (shape must match sn_sweep_*D)
        if Ndims == 1
            𝚽x12_in = zeros(Np_surf,Nm[1],2)
            𝚽y12_in = zeros(0); 𝚽z12_in = zeros(0)
            Dny⁻_beam = zeros(0); Dnz⁻_beam = zeros(0)
            Mny⁻_beam = zeros(0); Mnz⁻_beam = zeros(0)
        elseif Ndims == 2
            𝚽x12_in = zeros(Np_surf,Nm[1],2,Ns[2])
            𝚽y12_in = zeros(Np_surf,Nm[2],2,Ns[1])
            𝚽z12_in = zeros(0)
            Dnz⁻_beam = zeros(0)
            Mnz⁻_beam = zeros(0)
        else
            𝚽x12_in = zeros(Np_surf,Nm[1],2,Ns[2],Ns[3])
            𝚽y12_in = zeros(Np_surf,Nm[2],2,Ns[1],Ns[3])
            𝚽z12_in = zeros(Np_surf,Nm[3],2,Ns[1],Ns[2])
        end

        # 𝒪_sweep: spatial orders passed to the uncollided sweep function.
        # Defaults to the solver's 𝒪 (no allocation).  Replaced by a modified
        # copy only when force_ray_sweep or mixed_order_3d applies.
        𝒪_sweep = 𝒪

        use_ray3d = (Ndims == 3) && _ray3d_applicable(use_ray_sweep, Ωs, 𝒪, is_CSD)

        # Pre-compute beam axis for reuse in force and mixed_order blocks.
        d_beam = (Ndims == 3) ? findfirst(c -> abs(c) > 1e-9, Ωs) : 0

        # force_ray_sweep: when enabled, allow ray_sweep_3D for the uncollided
        # flux even when solver transverse spatial orders > 1.  The uncollided
        # flux of an axis-aligned beam has no transverse variation, so internally
        # clamping the transverse orders to 1 is exact.  The collision SN solve
        # keeps the solver's original 𝒪 unchanged.
        if !use_ray3d && use_ray_sweep && solver.get_force_ray_sweep() && Ndims == 3
            if _is_axis_aligned(Ωs) && (𝒪[d_beam] ∈ (1, 2)) && (!is_CSD || 𝒪[4] == 2)
                use_ray3d = true
                𝒪_sweep = copy(𝒪)
                𝒪_sweep[setdiff(1:3, [d_beam])] .= 1
                @debug "force_ray_sweep: ray_sweep_3D enabled for axis-aligned beam" Ωs 𝒪 𝒪_sweep
            end
        end

        mixed_order_3d = false
        if use_ray3d && 𝒪_ray_order ≥ 1
            if 𝒪_ray_order < 𝒪_sweep[d_beam]
                mixed_order_3d = true
                𝒪_ray_3d = copy(𝒪_sweep)
                𝒪_ray_3d[d_beam] = 𝒪_ray_order
                Nm_ray_3d = _compute_nm_from_o(𝒪_ray_3d, isFC)
            end
        end

        for ig in range(1,Ng)
            if is_CSD && (ig != 1)
                𝚽E12_beam .*= ΔE[ig]/ΔE[ig-1]
            end

            sources_ig = _build_sources_ig(intensity,loc,eg,ig,Ndims,Ns,Δs,geometry,ss.boundaries)

            if is_CSD
                Sg⁻ = S⁻[ig,:]/ΔE[ig] ; Sg⁺ = S⁺[ig,:]/ΔE[ig]
                Sg = S[ig,:,:]/ΔE[ig] ; ΔEg = ΔE[ig]
            else
                Sg⁻ = Vector{Float64}() ; Sg⁺ = Vector{Float64}()
                Sg = Vector{Float64}() ; ΔEg = 0.0
            end

            if Ndims == 1
                if mixed_order
                    𝚽l_view_low = @view φ_u_surf[ig, :, 1:Nm_ray[5], :, 1, 1]
                    Qlout_beam = zeros(Np, Nm_ray[5], Ns[1])
                    if is_CSD
                        𝚽E12_view = @view 𝚽E12_beam[1:Nm_ray[4], :]
                        _, _, _ = ray_sweep_1D(
                            𝚽l_view_low,
                            Qlout_beam,
                            Σtot[ig,:],mat[:,1,1],Ns[1],Δs[1],
                            Ωs[1],
                            zeros(Np),
                            Dn_beam,Np,
                            Mnx⁻_beam,Dnx⁻_beam,Np_surf,
                            𝒪_ray,Nm_ray,𝒞,ω,
                            sources_ig,is_adaptive,is_CSD,ΔEg,
                            𝚽E12_view,Sg⁻,Sg⁺,Sg,𝒲,isFC,
                            𝚽x12_in,boundary_conditions,1)
                    else
                        _, _, _ = ray_sweep_1D(
                            𝚽l_view_low,
                            Qlout_beam,
                            Σtot[ig,:],mat[:,1,1],Ns[1],Δs[1],
                            Ωs[1],
                            zeros(Np),
                            Dn_beam,Np,
                            Mnx⁻_beam,Dnx⁻_beam,Np_surf,
                            𝒪_ray,Nm_ray,𝒞,ω,
                            sources_ig,is_adaptive,is_CSD,ΔEg,
                            𝚽E12_beam,Sg⁻,Sg⁺,Sg,𝒲,isFC,
                            𝚽x12_in,boundary_conditions,1)
                    end
                else
                    𝚽l_view = @view φ_u_surf[ig,:,:,:,1,1]
                    Qlout_beam = zeros(Np,Nm[5],Ns[1])
                    sweep_fn = use_ray_sweep ? ray_sweep_1D : sn_sweep_1D
                    _,𝚽E12_beam,_ = sweep_fn(
                        𝚽l_view,
                        Qlout_beam,
                        Σtot[ig,:],mat[:,1,1],Ns[1],Δs[1],
                        Ωs[1],
                        zeros(Np),
                        Dn_beam,Np,
                        Mnx⁻_beam,Dnx⁻_beam,Np_surf,
                        𝒪,Nm,𝒞,ω,
                        sources_ig,is_adaptive,is_CSD,ΔEg,
                        𝚽E12_beam,Sg⁻,Sg⁺,Sg,𝒲,isFC,
                        𝚽x12_in,boundary_conditions,1)
                end
            elseif Ndims == 2
                𝚽l_view = @view φ_u_surf[ig,:,:,:,:,1]
                Qlout_beam = zeros(Np,Nm[5],Ns[1],Ns[2])
                _,𝚽E12_beam,_,_ = sn_sweep_2D(
                    𝚽l_view,
                    Qlout_beam,
                    Σtot[ig,:],mat[:,:,1],Ns[1:2],[Δs[1],Δs[2]],
                    [Ωs[1],Ωs[2]],
                    zeros(Np),
                    Dn_beam,Np,
                    Mnx⁻_beam,Dnx⁻_beam,Mny⁻_beam,Dny⁻_beam,Np_surf,
                    𝒪,Nm,𝒞,ω,
                    sources_ig,is_adaptive,is_CSD,ΔEg,
                    𝚽E12_beam,Sg⁻,Sg⁺,Sg,𝒲,isFC,
                    𝚽x12_in,𝚽y12_in,boundary_conditions,1)
            else
                𝚽l_view = @view φ_u_surf[ig,:,:,:,:,:]
                Qlout_beam = zeros(Np,Nm[5],Ns[1],Ns[2],Ns[3])
                # Only beams that satisfy every ray_sweep_3D prerequisite use the
                # characteristic (MOC) path; any other configuration falls back to
                # the standard SN sweep rather than tripping an assertion.
                if mixed_order_3d
                    # Mixed order: ray sweep at along-axis order fcs_ray_spatial_order.
                    # 𝒪_ray_3d drives the CSD branch and the moment projection (the
                    # written low-order entries map to the same slots of the full
                    # Nm[5] vector); Nm stays the solver's. The 𝚽E12 view keeps rows
                    # beyond Nm_ray_3d[4] untouched (zero), as in the 1D mixed path.
                    𝚽E12_view = is_CSD ? view(𝚽E12_beam, 1:Nm_ray_3d[4], :, :, :) : 𝚽E12_beam
                    _,_,_,_,_ = ray_sweep_3D(
                        𝚽l_view,
                        Qlout_beam,
                        Σtot[ig,:],mat,Ns,Δs,
                        Ωs,
                        zeros(Np),
                        Dn_beam,Np,
                        Mnx⁻_beam,Dnx⁻_beam,Mny⁻_beam,Dny⁻_beam,Mnz⁻_beam,Dnz⁻_beam,Np_surf,
                        𝒪_ray_3d,Nm,𝒞,ω,
                        sources_ig,is_adaptive,is_CSD,ΔEg,
                        𝚽E12_view,Sg⁻,Sg⁺,Sg,𝒲,isFC,
                        𝚽x12_in,𝚽y12_in,𝚽z12_in,boundary_conditions,1)
                else
                    sweep_fn = use_ray3d ? ray_sweep_3D : sn_sweep_3D
                    _,𝚽E12_beam,_,_,_ = sweep_fn(
                        𝚽l_view,
                        Qlout_beam,
                        Σtot[ig,:],mat,Ns,Δs,
                        Ωs,
                        zeros(Np),
                        Dn_beam,Np,
                        Mnx⁻_beam,Dnx⁻_beam,Mny⁻_beam,Dny⁻_beam,Mnz⁻_beam,Dnz⁻_beam,Np_surf,
                        𝒪_sweep,Nm,𝒞,ω,
                        sources_ig,is_adaptive,is_CSD,ΔEg,
                        𝚽E12_beam,Sg⁻,Sg⁺,Sg,𝒲,isFC,
                        𝚽x12_in,𝚽y12_in,𝚽z12_in,boundary_conditions,1)
                end
            end
        end

        if is_CSD
            if Ndims == 1
                for is in range(1,Nm[4]), ix in range(1,Ns[1]), p in range(1,Np)
                    𝚽cutoff_u_surf[p,is,ix,1,1] += Dn_beam[p] * 𝚽E12_beam[is,ix]
                end
            elseif Ndims == 2
                for is in range(1,Nm[4]), ix in range(1,Ns[1]), iy in range(1,Ns[2]), p in range(1,Np)
                    𝚽cutoff_u_surf[p,is,ix,iy,1] += Dn_beam[p] * 𝚽E12_beam[is,ix,iy]
                end
            else
                for is in range(1,Nm[4]), ix in range(1,Ns[1]), iy in range(1,Ns[2]), iz in range(1,Ns[3]), p in range(1,Np)
                    𝚽cutoff_u_surf[p,is,ix,iy,iz] += Dn_beam[p] * 𝚽E12_beam[is,ix,iy,iz]
                end
            end
        end
    end

    return φ_u_surf,𝚽cutoff_u_surf
end

"""
    _compute_first_collision_source(cross_sections::Cross_Sections,geometry::Geometry,
    solver::SN,φ_u::Array{Float64,6},sn_angle_discre::SN_Angular_Discretization,
    ℳ::Array{Float64},T::Array{Float64,2})

Compute the first collision source Q_FCS from the uncollided flux φ_u: the full BTE
scattering operator S·φ_u (in-group gi==ig plus out-of-group gi!=ig, via
`scattering_source` with `is_elastic=true`) plus (for BFP/FP) the Fokker-Planck
angular scattering operator.

# Input Argument(s)
- `cross_sections::Cross_Sections` : cross section informations.
- `geometry::Geometry` : geometry informations.
- `solver::SN` : discrete ordinates informations.
- `φ_u::Array{Float64,6}` : uncollided flux moments, shape `(Ng,Np,Nm[5],Ns[1],Ns[2],Ns[3])`.
- `sn_angle_discre::SN_Angular_Discretization` : precomputed angular basis.
- `ℳ::Array{Float64}` : Fokker-Planck scattering matrix.
- `T::Array{Float64,2}` : momentum transfer cross-sections `(Ng,Nmat)`.

# Output Argument(s)
- `Q_FCS::Array{Float64,6}` : first collision source moments, shape `(Ng,Np,Nm[5],Ns[1],Ns[2],Ns[3])`.

# Reference(s)
N/A

"""
function _compute_first_collision_source(cross_sections::Cross_Sections,geometry::Geometry,
                                         solver::SN,φ_u::Array{Float64,6},
                                         sn_angle_discre::SN_Angular_Discretization,
                                         ℳ::Array{Float64},T::Array{Float64,2})

    part = solver.get_particle()
    solver_type,_ = solver.get_solver_type()
    Ng = cross_sections.get_number_of_groups(part)
    Ns = geometry.get_number_of_voxels()
    mat = geometry.get_material_per_voxel()
    Nm_5 = size(φ_u,3)
    Np = sn_angle_discre.Np
    pl = sn_angle_discre.pl

    Q_FCS = zeros(size(φ_u))

    # BTE full scattering operator (is_elastic=true, includes gi==ig and gi!=ig).
    # Each ig writes a disjoint Q_FCS[ig,...] slice, so the ig loop is parallelized.
    # The inner scattering_source MUST use parallel=false: it runs inside this
    # @threads :static loop, and nested :static threading is illegal.
    if solver_type ∉ [4,5]
        Ls = maximum(pl)
        Σs = cross_sections.get_scattering(part,part,Ls)
        @threads :static for ig in range(1,Ng)
            scattering_source(
                @view(Q_FCS[ig,:,:,:,:,:]),
                φ_u,
                Σs[:,:,ig,:],
                mat, Np, pl, Nm_5, Ns, Ng, ig,
                true; parallel=false)
        end
    end

    # BFP/FP angular scattering: same ig parallelization, inner call parallel=false.
    if solver_type ∈ [2,4]
        @threads :static for ig in range(1,Ng)
            fokker_planck_source(
                Np,Nm_5,T[ig,:],
                @view(φ_u[ig,:,:,:,:,:]),
                @view(Q_FCS[ig,:,:,:,:,:]),
                Ns,mat,ℳ; parallel=false)
        end
    end

    return Q_FCS
end

"""
    _compute_fcs_precursor(cross_sections::Cross_Sections, geometry::Geometry,
    solver::SN, source::Source)

Precompute the FCS ingredients for a single particle before the coupled transport loop:
- uncollided flux `φ_u` driven by the original surface sources and point sources;
- first-collision source `Q_FCS` (full scattering operator applied to `φ_u`);
- a modified external source that combines the original volume source with `Q_FCS`
  and zeroes out the surface sources, surface source objects, and point sources.

Returns `nothing` when FCS is not applicable for this solver/source (BFP-EF,
non-void boundary, no surface source and no point source). In that case the caller
should fall back to the original source and standard `compute_flux`.

# Input Argument(s)
- `cross_sections::Cross_Sections` : cross section informations.
- `geometry::Geometry` : geometry informations.
- `solver::SN` : discrete ordinates informations.
- `source::Source` : original source informations.

# Output Argument(s)
- `result::Union{Nothing, Tuple{Flux_Per_Particle, Source}}` :
  - `Flux_Per_Particle` : uncollided flux `φ_u` (and cutoff flux if CSD).
  - `Source` : modified external source for the coupled collision solve.

# Reference(s)
N/A

"""
function _compute_fcs_precursor(cross_sections::Cross_Sections, geometry::Geometry,
                                solver::SN, source::Source)

    # No surface sources and no point sources: FCS path is equivalent to standard solve.
    if isempty(source.get_surface_source_objects()) && !has_point_sources(source)
        return nothing
    end

    # Cartesian geometry guard
    if geometry.get_type() != "cartesian"
        println(">>>FCS Point/Surface Path only supports cartesian geometry; falling back to standard source.")
        return nothing
    end

    solver_type, is_CSD = solver.get_solver_type()
    if solver_type == 6
        if has_point_sources(source)
            error("FCS point source is not supported for BFP-EF solver.")
        end
        println(">>>FCS on BFP-EF has limited effect; falling back to standard source.")
        return nothing
    end

    # Void boundary guard
    if any(!=(0), geometry.get_boundary_conditions())
        println(">>>FCS Point/Surface Path only supports void boundary; falling back to standard source.")
        return nothing
    end

    sn_angle_discre = build_angular_discretization(solver, geometry)
    part = solver.get_particle()
    Ng = cross_sections.get_number_of_groups(part)
    Ndims = geometry.get_dimension()

    # Fokker-Planck data (needed by both uncollided sweep and first collision source)
    if solver_type ∈ [2, 4]
        N = solver.get_quadrature_order()
        Nd = sn_angle_discre.Nd
        quadrature_type = solver.get_quadrature_type()
        Qdims = sn_angle_discre.Qdims
        fokker_planck_type = solver.get_angular_fokker_planck()
        ℳ, λ₀ = fokker_planck_scattering_matrix(
            N, Nd, quadrature_type, Ndims, fokker_planck_type,
            sn_angle_discre.Mn, sn_angle_discre.Dn,
            sn_angle_discre.pl, sn_angle_discre.Np, Qdims)
        T = cross_sections.get_momentum_transfer(part)
    else
        T = Array{Float64}(undef, 0, 0)
        ℳ = Array{Float64}(undef)
        λ₀ = 0.0
    end

    # Step 1: uncollided flux (surface path, augmented Σ_t for BFP/FP)
    _, 𝒪, Nm = solver.get_schemes(geometry, solver.get_is_full_coupling())
    Ns = geometry.get_number_of_voxels()
    φ_u = zeros(Ng, sn_angle_discre.Np, Nm[5], Ns[1], Ns[2], Ns[3])
    if is_CSD
        𝚽cutoff_u = zeros(sn_angle_discre.Np, Nm[5], Ns[1], Ns[2], Ns[3])
    end
    φ_u_surf, 𝚽cutoff_u_surf = _compute_uncollided_surface_flux(
        cross_sections, geometry, solver, source, sn_angle_discre; T=T, λ₀=λ₀, use_ray_sweep=solver.get_use_ray_sweep())
    φ_u .+= φ_u_surf
    if is_CSD
        𝚽cutoff_u .+= 𝚽cutoff_u_surf
    end

    # Step 1b: point-source uncollided flux (BTE path A, or BFP path B for CSD solvers)
    if has_point_sources(source)
        # FCS point sources follow the same 3D/Cartesian restriction as the standard
        # point-source path in sn_flux.jl. This is a deliberate restriction; the
        # underlying ray-tracing utilities assume a 3D Cartesian grid.
        @assert geometry.get_dimension() == 3 "FCS point source requires 3D geometry."
        @assert geometry.get_type() == "cartesian" "FCS point source requires cartesian grid."
        if is_CSD && solver_type ∉ (2, 3, 4)
            error("Point-source path B only supports BFP, BCSD, and FP solvers. CSD is not supported.")
        end
        cache_ps = build_point_source_trace_cache(source.point_sources, geometry, cross_sections, solver)
        if !is_CSD
            φ_u_point = _compute_uncollided_point_flux_bte(
                cache_ps, cross_sections, geometry, solver, source, sn_angle_discre)
            φ_u .+= φ_u_point
        else
            if !ENABLE_BFP_POINT_SOURCE[]
                error("BFP point-source Path B is disabled. Set Radiant.ENABLE_BFP_POINT_SOURCE[]=true to use Path B.")
            end
            φ_u_point, 𝚽cutoff_u_point = _compute_uncollided_point_flux_bfp(
                cache_ps, cross_sections, geometry, solver, source, sn_angle_discre; T=T, λ₀=λ₀)
            φ_u .+= φ_u_point
            𝚽cutoff_u .+= 𝚽cutoff_u_point
        end
    end

    # Step 2: first collision source (full scattering operator S)
    Q_FCS = _compute_first_collision_source(
        cross_sections, geometry, solver, φ_u, sn_angle_discre, ℳ, T)

    # Step 3: build modified external source (volume + Q_FCS, no surface/point source)
    modified_source = Source(part, cross_sections, geometry, solver)
    modified_source.add_volume_source(source.get_volume_sources() + Q_FCS)
    modified_source.normalization_factor = source.normalization_factor
    zero_surface_sources!(modified_source)
    zero_point_sources!(modified_source)
    zero_surface_source_objects!(modified_source)

    # Step 4: package φ_u as Flux_Per_Particle
    flux_u = Flux_Per_Particle(part)
    flux_u.add_flux(φ_u)
    if is_CSD
        flux_u.add_flux_cutoff(𝚽cutoff_u)
    end

    return (flux_u, modified_source)
end

"""
    _compute_flux_sn(cross_sections::Cross_Sections,geometry::Geometry,solver::SN,
    source::Source)

Wrapper around `compute_flux(::SN,...)` that temporarily disables the FCS flag, so the
FCS second-pass standard solve does not re-trigger the `compute_flux` entry guard and
recurse into `compute_flux_fcs`. The original flag value is restored before returning so
that the caller's FCS setting remains active for subsequent top-level calls.

# Input Argument(s)
- `cross_sections::Cross_Sections` : cross section informations.
- `geometry::Geometry` : geometry informations.
- `solver::SN` : discrete ordinates informations.
- `source::Source` : source informations.

# Output Argument(s)
- `flux::Flux_Per_Particle`: flux informations.

# Reference(s)
N/A

"""
function _compute_flux_sn(cross_sections::Cross_Sections,geometry::Geometry,solver::SN,source::Source;parallel::Bool=true)
    flag_original = solver.get_is_first_collision_source()
    solver.set_is_first_collision_source(false)
    try
        return compute_flux(cross_sections,geometry,solver,source;parallel=parallel)
    finally
        solver.set_is_first_collision_source(flag_original)
    end
end

"""
    compute_flux_fcs(cross_sections::Cross_Sections,geometry::Geometry,solver::SN,
    source::Source)

Solve the transport equation using the First Collision Source (FCS) method for a given
particle.

# Input Argument(s)
- `cross_sections::Cross_Sections` : cross section informations.
- `geometry::Geometry` : geometry informations.
- `solver::SN` : discrete ordinates informations.
- `source::Source` : source informations (must carry original `surface_source_objects`).

# Output Argument(s)
- `flux::Flux_Per_Particle`: flux informations.

# Reference(s)
N/A

"""
function compute_flux_fcs(cross_sections::Cross_Sections,geometry::Geometry,solver::SN,source::Source;parallel::Bool=true)

    precursor = _compute_fcs_precursor(cross_sections, geometry, solver, source)

    if isnothing(precursor)
        return _compute_flux_sn(cross_sections, geometry, solver, source;parallel=parallel)
    end

    flux_u, modified_source = precursor

    # Collision solve with FCS flag disabled
    flux_s = _compute_flux_sn(cross_sections, geometry, solver, modified_source;parallel=parallel)

    # Synthesis
    part = solver.get_particle()
    flux = Flux_Per_Particle(part)
    flux.set_uncollided_flux(flux_u.get_flux())
    flux.add_flux(flux_u.get_flux() + flux_s.get_flux())
    _, is_CSD = solver.get_solver_type()
    if is_CSD
        flux.add_flux_cutoff(flux_u.get_flux_cutoff() + flux_s.get_flux_cutoff())
    end
    flux.add_spectral_radius(flux_s.get_spectral_radius())

    return flux
end
