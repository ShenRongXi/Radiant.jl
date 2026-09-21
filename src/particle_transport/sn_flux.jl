"""
    compute_flux(cross_sections::Cross_Sections,geometry::Geometry,
    solver::SN,source::Source,electromagnetic_field::Electromagnetic_Field=Electromagnetic_Field();parallel::Bool=true)

Solve the transport equation using the discrete ordinates (SN) method for a given particle.  

# Input Argument(s)
- `cross_sections::Cross_Sections` : cross section informations.
- `geometry::Geometry` : geometry informations.
- `solver::SN` : Discrete ordinates informations.
- `source::Source` : source informations.
- `electromagnetic_field::Electromagnetic_Field` : external electromagnetic field (optional,
  defaults to no field).

# Output Argument(s)
- `flux::Flux_Per_Particle`: flux informations.

# Reference(s)
N/A

"""
function compute_flux(cross_sections::Cross_Sections,geometry::Geometry,solver::SN,source::Source,electromagnetic_field::Electromagnetic_Field=Electromagnetic_Field();parallel::Bool=true)

    # FCS guard: route to compute_flux_fcs when the FCS flag is enabled
    if solver.get_is_first_collision_source()
        return compute_flux_fcs(cross_sections,geometry,solver,source;parallel=parallel)
    end

    # Point-source uncollided-flux branch (standard SN path, BTE only). When the FCS flag
    # is off but the source carries internal point sources, compute their uncollided flux
    # analytically, fold it into the volume source as a first collision source, solve the
    # scattered flux, and return ψ_total = φ_u + ψ_s. See plan §2.4.1.
    if has_point_sources(source) && !solver.get_is_first_collision_source()
        @assert geometry.get_dimension() == 3 "Point-source path requires 3D geometry."
        @assert geometry.get_type() == "cartesian" "Point-source DDA requires cartesian grid."
        SN_type = solver.get_angular_boltzmann()
        @assert SN_type ∈ ("standard","galerkin-d") "Standard point-source path supports standard and galerkin-d angular discretization."
        _solver_type,_is_CSD = solver.get_solver_type()
        @assert !_is_CSD "Standard point-source path only supports BTE."
        _,𝒪p,Nmp = solver.get_schemes(geometry,solver.get_is_full_coupling())
        @assert all(𝒪p[1:3] .∈ Ref((1,2))) "Point-source path requires spatial order 1 or 2."
        @assert 𝒪p[4] == 1 "Standard point-source path requires energy order 1."

        sn_angle_discre = build_angular_discretization(solver,geometry)
        Np_p = sn_angle_discre.Np
        pl_p = sn_angle_discre.pl
        part_p = solver.get_particle()
        Ls_p = maximum(pl_p)
        Σs_p = cross_sections.get_scattering(part_p,part_p,Ls_p)
        mat_p = geometry.get_material_per_voxel()
        Ns_p = geometry.get_number_of_voxels()
        Ng_p = cross_sections.get_number_of_groups(part_p)

        cache_p = build_point_source_trace_cache(source.point_sources,geometry,cross_sections,solver)
        φ_u = _compute_uncollided_point_flux_bte(cache_p,cross_sections,geometry,solver,source,sn_angle_discre)

        Q_FCS = zeros(size(φ_u))
        for ig in range(1,Ng_p)
            scattering_source(@view(Q_FCS[ig,:,:,:,:,:]),φ_u,Σs_p[:,:,ig,:],
                              mat_p,Np_p,pl_p,size(φ_u,3),Ns_p,Ng_p,ig,true;parallel=false)
        end

        source_for_solver = deepcopy(source)
        source_for_solver.volume_sources .+= Q_FCS
        empty!(source_for_solver.point_sources)   # prevent infinite recursion
        flux_s = compute_flux(cross_sections,geometry,solver,source_for_solver;parallel=parallel)

        flux = Flux_Per_Particle(part_p)
        flux.add_flux(φ_u .+ flux_s.get_flux())
        flux.add_spectral_radius(flux_s.get_spectral_radius())
        return flux
    end

#----
# Geometry data
#----
Ndims = geometry.get_dimension()
geo_type = geometry.get_type()
if geo_type != "cartesian" error("Transport of particles in",geo_type," is unavailable.") end
Ns = geometry.get_number_of_voxels()
Δs = geometry.get_voxels_width()
mat = geometry.get_material_per_voxel()
boundary_conditions = geometry.get_boundary_conditions()

#----
# Preparation of angular discretisation
#----
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

#----
# Preparation of cross sections
#----
part = solver.get_particle()
solver_type,is_CSD = solver.get_solver_type()
Nmat = cross_sections.get_number_of_materials()
Ng = cross_sections.get_number_of_groups(part)
if is_CSD
    ΔE = cross_sections.get_energy_width(part)
    E = cross_sections.get_energies(part)
    Eb = cross_sections.get_energy_boundaries(part)
end
isFC = solver.get_is_full_coupling()
schemes,𝒪,Nm = solver.get_schemes(geometry,isFC)
ω,𝒞,is_adaptive,𝒲 = scheme_weights(𝒪,schemes,Ndims,is_CSD)

println(">>>Particle: $(get_type(part)) <<<")

# Total cross sections
Σtot = zeros(Ng,Nmat)
if solver_type ∈ [4,5] 
    Σtot = cross_sections.get_absorption(part)
else
    Σtot = cross_sections.get_total(part)
end

# Scattering cross sections (sized to the highest Legendre order actually carried by the
# angular basis: for Galerkin this is set by the quadrature, not by L; get_scattering
# zero-pads moments beyond the cross-section's Legendre order).
Ls = maximum(pl)
Σs = zeros(Nmat,Ng,Ng,Ls+1)
if solver_type ∉ [4,5]
    Σs = cross_sections.get_scattering(part,part,Ls)
end

# Stopping powers
if is_CSD
    S⁻ = zeros(Ng,Nmat); S⁺ = zeros(Ng,Nmat)
    Sb = cross_sections.get_boundary_stopping_powers(part)
    for n in range(1,Nmat)
        S⁻[:,n] = Sb[1:Ng,n] ; S⁺[:,n] = Sb[2:Ng+1,n]
    end
    S = zeros(Ng,Nmat,𝒪[4])
    for n in range(1,Nmat), ig in range(1,Ng)
        S[ig,n,1] = (S⁻[ig,n]+S⁺[ig,n])/2
        if (𝒪[4] > 1) S[ig,n,2] = (S⁻[ig,n]-S⁺[ig,n])/(2*sqrt(3)) end
    end
end

# Momentum transfer
if solver_type ∈ [2,4]
    T = zeros(Ng,Nmat)
    T = cross_sections.get_momentum_transfer(part)
    fokker_planck_type = solver.get_angular_fokker_planck()
    ℳ,λ₀ = fokker_planck_scattering_matrix(N,Nd,quadrature_type,Ndims,fokker_planck_type,Mn,Dn,pl,Np,Qdims)
    Σtot .+= T .* λ₀
end

# Elastic-free approximation
if solver_type == 6
    for n in range(1,Nmat), ig in range(1,Ng)
        Σtot[ig,n] -= Σs[n,ig,ig,1]
    end
end

# External electro-magnetic fields
is_EM,𝓔,𝓑 = electromagnetic_field.get_electromagnetic_field()
if is_EM
    if any(𝓔 .!= 0) error("Electric-field transport not yet implemented; only magnetic fields are supported.") end
    charge = part.get_charge()
    ℳ_EM,λ₀_EM = electromagnetic_scattering_matrix(𝓔,𝓑,charge,Ω,w,Ndims,Mn,Dn,pl,pm,Np,Ng,Eb,ΔE,Qdims)
    for ig in range(1,Ng) Σtot[ig,:] .+= λ₀_EM[ig] end
else
    ℳ_EM = zeros(Ng,Np,Np);
end

#----
# Acceleration solver
#----

𝒜 = solver.get_acceleration()
gmres_restart = solver.get_gmres_restart()
anderson_depth = solver.get_anderson_depth()

#----
# Fixed sources
#----

surface_sources = source.get_surface_sources()
volume_sources = source.get_volume_sources()
Np_source = Int64(min(Np_surf,length(surface_sources[1,:,1])))

#----
# Flux calculations
#----

ϵ_max = solver.get_convergence_criterion()
I_max = solver.get_maximum_iteration()

# Initialization flux
𝚽l = zeros(Ng,Np,Nm[5],Ns[1],Ns[2],Ns[3])
if is_CSD 𝚽cutoff = zeros(Np,Nm[5],Ns[1],Ns[2],Ns[3]) end

# All-group iteration
i_out = 1
is_outer_convergence = false
ϵ_out = Inf
is_outer_iteration = false
Ntot = 0
ρ_in = -ones(Ng) # In-group spectral radius (per energy group, last outer iteration)
if is_outer_iteration 𝚽l⁻ = zeros(Ng,Np,Nm[5],Ns[1],Ns[2],Ns[3]) end

while ~(is_outer_convergence)

    ρ_in = -ones(Ng) # In-group spectral radius
    if is_CSD
        𝚽E12 = zeros(Nd,Nm[4],Ns[1],Ns[2],Ns[3])
    else
        𝚽E12 = Array{Float64}(undef)
    end

    # Loop over energy group
    for ig in range(1,Ng)

        # Calculation of the Legendre components of the source (out-scattering)
        Qlout = zeros(Np,Nm[5],Ns[1],Ns[2],Ns[3])
        if solver_type ∉ [4,5] Qlout = scattering_source(Qlout,𝚽l,Σs[:,:,ig,:],mat,Np,pl,Nm[5],Ns,Ng,ig) end

        # Fixed volumic sources
        Qlout .+= volume_sources[ig,:,:,:,:,:]

        # Calculation of the group flux
        if is_CSD
            if (ig != 1) 𝚽E12 = 𝚽E12 .* ΔE[ig]/ΔE[ig-1] end
            Eg = E[ig]
            ΔEg = ΔE[ig]
            Sg⁻ = S⁻[ig,:]/ΔEg
            Sg⁺ = S⁺[ig,:]/ΔEg
            Sg = S[ig,:,:]/ΔEg
            if solver_type ∈ [2,4]
                Tg = T[ig,:]
            else
                Tg = Vector{Float64}()
                ℳ = Array{Float64}(undef)
            end
        else
            Eg = 0.0
            ΔEg = 0.0
            Sg⁻ = Vector{Float64}()
            Sg⁺ = Vector{Float64}()
            Sg = Vector{Float64}()
            Tg = Vector{Float64}()
            ℳ = Array{Float64}(undef)
        end
        𝚽l[ig,:,:,:,:,:],𝚽E12,ρ_in[ig],Ntot = sn_one_speed(𝚽l[ig,:,:,:,:,:],Qlout,Σtot[ig,:],Σs[:,ig,ig,:],mat,Ndims,Nd,ig,Ns,Δs,Ω,Mn,Dn,Np,pl,Mn_surf,Dn_surf,Np_surf,n_to_n⁺,𝒪,Nm,isFC,𝒞,ω,I_max,ϵ_max,surface_sources[ig,:,:],is_adaptive,is_CSD,solver_type,ΔEg,𝚽E12,Sg⁻,Sg⁺,Sg,Tg,ℳ,𝒜,Ntot,is_EM,ℳ_EM[ig,:,:],𝒲,boundary_conditions,Np_source,gmres_restart,anderson_depth)
    end

    # Verification of convergence in all energy groups
    if is_outer_iteration
        ϵ_out = norm(𝚽l .- 𝚽l⁻) / max(norm(𝚽l), 1e-16)
        𝚽l⁻ = 𝚽l
    end
    if (ϵ_out < ϵ_max || i_out >= I_max) || ~is_outer_iteration
        is_outer_convergence = true
        # Calculate the flux at the cutoff energy. The is loop uses Nm[4]
        # (not Nm[5]) because 𝚽E12 is sized (Nd, Nm[4], ...); tail elements
        # 𝚽cutoff[:, Nm[4]+1:Nm[5], ...] intentionally remain zero.
        if is_CSD
            Nxyz = Ns[1] * Ns[2] * Ns[3]
            if parallel && Threads.nthreads() > 1 && Nxyz >= PAR_MIN_NVOXELS
                @threads :static for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
                    ix, iy, iz = Tuple(I)
                    for is in range(1,Nm[4]), n in range(1,Nd), p in range(1,Np)
                        𝚽cutoff[p,is,ix,iy,iz] += Dn[p,n] * 𝚽E12[n,is,ix,iy,iz]
                    end
                end
            else
                for n in range(1,Nd), ix in range(1,Ns[1]), iy in range(1,Ns[2]), iz in range(1,Ns[3]), is in range(1,Nm[4]), p in range(1,Np)
                    𝚽cutoff[p,is,ix,iy,iz] += Dn[p,n] * 𝚽E12[n,is,ix,iy,iz]
                end
            end
        end
    else
        i_out += 1
    end
    
end

# Save flux
flux = Flux_Per_Particle(part)
flux.add_flux(𝚽l)
if is_CSD flux.add_flux_cutoff(𝚽cutoff) end
flux.add_spectral_radius(ρ_in)

return flux

end
