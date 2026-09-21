"""
    transport(cross_sections::Cross_Sections,geometry::Geometry,solvers::Solvers,
    sources::Fixed_Sources)

Solve transport calculations.

# Input Argument(s)
- `cross_sections::Cross_Sections`: cross section informations.
- `geometry::Geometry`: geometry informations.
- `solvers::Solvers`: solvers informations.
- `sources::Fixed_Sources`: sources informations.
- `electromagnetic_field::Electromagnetic_Field`: external electromagnetic field (optional,
  defaults to no field).

# Output Argument(s)
- `flux::Flux`: flux solution.

# Reference(s)
N/A

"""
function transport(cross_sections::Cross_Sections,geometry::Geometry,solvers::Solvers,sources::Fixed_Sources,electromagnetic_field::Electromagnetic_Field=Electromagnetic_Field())

#----
# Initialization
#----
Npart = solvers.get_number_of_particles()
flux = Flux()

#---
# One particles transport
#---
if Npart == 1

    # Initialization
    particle = solvers.get_particles()[1]
    method = solvers.get_method_by_index(1)
    fixed_source = sources.get_source(particle)

    if method isa SN && method.get_is_first_collision_source()
        precursor = _compute_fcs_precursor(cross_sections, geometry, method, fixed_source)
        if !isnothing(precursor)
            flux_u, modified_source = precursor
            flux_s = _compute_flux_sn(cross_sections, geometry, method, modified_source)
            flux_per_particle = Flux_Per_Particle(particle)
            flux_per_particle.set_uncollided_flux(flux_u.get_flux())  # store φ_u separately
            flux_per_particle.add_flux(flux_u.get_flux() + flux_s.get_flux())
            _, is_CSD = method.get_solver_type()
            if is_CSD
                flux_per_particle.add_flux_cutoff(flux_u.get_flux_cutoff() + flux_s.get_flux_cutoff())
            end
            flux_per_particle.add_spectral_radius(flux_s.get_spectral_radius())
            flux.add_flux(flux_per_particle)
        else
            particle_flux = _compute_flux_sn(cross_sections, geometry, method, fixed_source)
            flux.add_flux(particle_flux)
        end
    else
        particle_flux = compute_flux(cross_sections,geometry,method,fixed_source,electromagnetic_field)
        flux.add_flux(particle_flux)
    end

#----
# N-particles coupled transport
#----
else

    # Initialization
    particles = solvers.get_particles()
    fixed_source = Vector{Source}(undef,Npart)
    particle_sources = Vector{Source}(undef,Npart)
    method = Vector{Solver}(undef,Npart)
    for i in range(1,Npart)
        # Use index-based lookup so that multiple solvers can share the same tag.
        # NOTE: Fixed_Sources.build() still builds the source using the first solver
        # matching the tag. Multi-solver-per-tag is only safe when all solvers are
        # source-compatible (same angular/spatial discretization). This is a known gap.
        method[i] = solvers.get_method_by_index(i)
        fixed_source[i] = sources.get_source(particles[i])
        particle_sources[i] = Source(particles[i], cross_sections, geometry, method[i])
    end

    # Ordering the particles
    particle_index = zero(Npart)
    for i in range(1,Npart)
        if get_tag(particles[i]) ∈ get_tag.(sources.get_particles())
            i0 = i
            if i0 == 1
                particle_index = collect(1:Npart)
            else
                particle_index = append!(collect(i0:Npart),collect(1:i0-1))
            end
            break
        end
        if i == Npart error("No fixed sources are defined.") end
    end

    #----
    # FCS precomputation (before the coupled loop)
    #----
    # Use a small union container for per-particle FCS ingredients. Access is guarded
    # by `!isnothing` below, and the inner loop uses `sn_method = method[i]::SN` to
    # help the compiler narrow the union of SN/GN solvers. The runtime cost is minimal.
    fcs_data = Vector{Union{Nothing, Tuple{Flux_Per_Particle, Source}}}(undef, Npart)
    for i in range(1,Npart)
        if !(method[i] isa SN)
            println(">>>FCS is only supported for SN solvers; skipping FCS for particle $(get_tag(particles[i])) at index $i.")
            fcs_data[i] = nothing
            continue
        end
        sn_method = method[i]::SN
        if sn_method.get_is_first_collision_source()
            fcs_data[i] = _compute_fcs_precursor(
                cross_sections, geometry, sn_method, fixed_source[i])
        else
            fcs_data[i] = nothing
        end
    end

    # Replace fixed sources with Q_FCS-augmented sources, seed φ_u secondaries,
    # and inject φ_u into the total flux so convergence checks see the full solution.
    #
    # Alias note: when multiple solvers share the same particle tag, `fixed_source[i]`
    # initially points to the same Source object built by `Fixed_Sources.build()` using
    # the first matching solver. Replacing `fixed_source[i]` here only changes the local
    # vector slot for FCS-enabled solvers; non-FCS slots continue to reference that
    # original Source object. This is safe only when all solvers for that tag are
    # source-compatible (same angular/spatial discretization shape), as documented in
    # the "Known gap: shared-tag source compatibility" section.
    for i in range(1,Npart)
        if !isnothing(fcs_data[i])
            flux_u_i, modified_source_i = fcs_data[i]
            fixed_source[i] = modified_source_i
            # Store φ_u before merging into total flux
            flux_u_i.set_uncollided_flux(flux_u_i.get_flux())
            flux.add_flux(flux_u_i)
            for j in range(1,Npart)
                if i != j
                    particle_sources[j] += particle_source(
                        flux_u_i, cross_sections, geometry, method[i], method[j])
                end
            end
        end
    end

    # Coupled transport
    Ngen_max = solvers.get_maximum_number_of_generations()
    ϵ_gen = solvers.get_convergence_criterion()
    conv_type = solvers.get_convergence_type()
    for n in range(1,Ngen_max)

        flux⁻ = deepcopy(flux)

        # Loop over particles
        for p in range(1,Npart)

            i = particle_index[p]

            # Combinaison of fixed external and scattered particles sources 
            source = particle_sources[i]
            if n == 1 source += fixed_source[i] end

            # Transport (FCS flag disabled internally for SN; GN uses standard path)
            if method[i] isa SN
                particle_flux = _compute_flux_sn(cross_sections, geometry, method[i], source)
            else
                particle_flux = compute_flux(cross_sections,geometry,method[i],source,electromagnetic_field)
            end
            flux.add_flux(particle_flux)

            # Compute scattered particles sources
            if n == Ngen_max && p == Npart continue end
            for q in range(1,Npart)
                
                j = particle_index[q]
                if (n == Ngen_max) && (q ≤ p) continue end

                if i != j
                    particle_sources[j] += particle_source(particle_flux,cross_sections,geometry,method[i],method[j])
                else
                    particle_sources[i] = Source(particles[i],cross_sections,geometry,method[i])
                end
            end
        end

        # Verify convergence of the fluxes, energy deposition or charge deposition
        px = 0
        ϵmax = 0
        if n != 1
            if conv_type == "flux"
                for p in range(1,Npart)
                    ϵ = maximum(abs.(flux.get_flux(particles[p])[:,1,1,:,:,:] .- flux⁻.get_flux(particles[p])[:,1,1,:,:,:]) ./ max.(abs.(flux.get_flux(particles[p])[:,1,1,:,:,:]),eps()))
                    ϵmax = max(ϵmax,ϵ)
                    if (ϵ < ϵ_gen) px += 1 end
                end
            elseif conv_type == "energy-deposition"
                Edep = energy_deposition(cross_sections,geometry,solvers,sources,flux,flux.get_particles())
                Edep⁻ = energy_deposition(cross_sections,geometry,solvers,sources,flux⁻,flux.get_particles())
                ϵ = maximum(abs.(Edep .- Edep⁻) ./ max.(abs.(Edep),eps()))
                ϵmax = ϵ
                if (ϵ < ϵ_gen) px = Npart end
            elseif conv_type == "charge-deposition"
                Cdep = charge_deposition(cross_sections,geometry,solvers,sources,flux,flux.get_particles())
                Cdep⁻ = charge_deposition(cross_sections,geometry,solvers,sources,flux⁻,flux.get_particles())
                ϵ = maximum(abs.(Cdep .- Cdep⁻) ./ max.(abs.(Cdep),eps()))
                ϵmax = ϵ
                if (ϵ < ϵ_gen) px = Npart end
            else
                error("Unknown convergence type.")
            end
        end
        if px == Npart
            println(">>>Particle fluxes have converged after $n particle generations ( ϵ = ",@sprintf("%.4E",ϵmax)," )")
            break
        elseif n == Ngen_max
            println(">>>Particle fluxes have not converged after $n particle generations ( ϵ = ",@sprintf("%.4E",ϵmax)," )")
        end
    end
end

return flux
end