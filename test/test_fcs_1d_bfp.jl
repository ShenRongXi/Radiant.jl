using Radiant
using LinearAlgebra
using Test
using Plots

# Headless PNG output for CI
ENV["GKSwstype"] = "100"
gr(fmt=:png, dpi=300)
default(show=false)

"""
    scalar_flux_per_x(phi)

Extract the scalar flux (zeroth angular moment, first spatial/energy moment) per
voxel by summing over energy groups.
"""
function scalar_flux_per_x(phi::AbstractArray{Float64,6})
    Ng, _, _, Nx = size(phi)[1:4]
    flux_x = zeros(Nx)
    for ix in 1:Nx, ig in 1:Ng
        flux_x[ix] += phi[ig, 1, 1, ix, 1, 1]
    end
    return flux_x
end

"""
    reconstruct_uncollided_flux(cross_sections, geometry, solver, surface_source)

Reconstruct the uncollided surface flux for a BFP solver using the same angular
discretization and FP stabilization data as the FCS solve.
"""
function reconstruct_uncollided_flux(cs::Cross_Sections, geo::Geometry, m::SN, ss::Surface_Source)
    ad = Radiant.build_angular_discretization(m, geo)
    source = Source(m.get_particle(), cs, geo, m)
    source.add_source(ss)

    N = m.get_quadrature_order()
    Nd = ad.Nd
    quadrature_type = m.get_quadrature_type()
    Qdims = ad.Qdims
    fokker_planck_type = m.get_angular_fokker_planck()
    ℳ, λ₀ = Radiant.fokker_planck_scattering_matrix(
        N, Nd, quadrature_type, geo.get_dimension(), fokker_planck_type,
        ad.Mn, ad.Dn, ad.pl, ad.Np, Qdims)
    T = cs.get_momentum_transfer(m.get_particle())

    φ_u, _ = Radiant._compute_uncollided_surface_flux(
        cs, geo, m, source, ad; T=T, λ₀=λ₀)
    return φ_u
end

@testset "FCS 1D BFP benchmark: flux components and reference comparison" begin
    # Reproduce the 1D Gauss-Lobatto BFP benchmark from
    # Radiant-Examples/Paper_2025_AFP/benchmark1_1D_GaussLobatto.jl and verify that
    # the FCS-enabled solve reproduces the standard reference.  In addition, split
    # the FCS solution into uncollided and scattered scalar flux and plot the three
    # components versus depth.

    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)

    electron = Electron()
    photon = Photon()
    positron = Positron()

    L = 15
    cs = Cross_Sections()
    cs.set_materials(water)
    cs.set_particles([electron, photon, positron])
    cs.set_group_structure("log", 80, 10.0, 0.001)
    cs.set_interactions([Inelastic_Collision(), Elastic_Collision(), Bremsstrahlung(), Pair_Production(), Photoelectric(), Rayleigh(), Compton(), Auger(), Fluorescence(), Annihilation()])
    cs.set_legendre_order(L)

    xsec_file = joinpath(@__DIR__, "..", "..", "Radiant-Examples", "Paper_2025_AFP", "water_p$(L).xsec")
    if isfile(xsec_file)
        cs.set_source("fmac-m")
        cs.set_file(xsec_file)
    else
        cs.set_source("physics-models")
    end
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [40])
    geo.set_region_boundaries("x", [0.0, 5.0])
    geo.build(cs)

    function run_benchmark_case(; fcs::Bool=false)
        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")

        m = Discrete_Ordinates()
        m.set_particle(electron)
        m.set_solver_type("BFP")
        m.set_acceleration("livolant")
        m.set_quadrature("Gauss-Lobatto", 26, 1)
        m.set_legendre_order(25)
        m.set_angular_boltzmann("galerkin-d")
        m.set_angular_fokker_planck("finite-difference")
        m.set_convergence_criterion(1e-5)
        m.set_maximum_iteration(500)
        m.set_scheme("E", "DG", 2)
        m.set_scheme("x", "DG", 2)
        m.set_is_first_collision_source(fcs)

        solvers = Solvers()
        solvers.add_solver(m)

        sources = Fixed_Sources(cs, geo, solvers)
        sources.add_source(ss)

        cu = Computation_Unit()
        cu.set_cross_sections(cs)
        cu.set_geometry(geo)
        cu.set_solvers(solvers)
        cu.set_sources(sources)
        cu.run()

        x = cu.get_voxels_position("x")
        dose = cu.get_energy_deposition()
        flux = cu.flux.get_flux(electron)
        return x, dose, flux, m, ss
    end

    x_ref, dose_ref, flux_ref, _, _ = run_benchmark_case(; fcs=false)
    x, dose_fcs, flux_total, m_fcs, ss_fcs = run_benchmark_case(; fcs=true)

    @test x ≈ x_ref

    # Dose comparison
    @test isapprox(dose_fcs, dose_ref; rtol=5e-4)
    rel_err_dose = norm(dose_fcs .- dose_ref) / max(norm(dose_ref), 1e-16)
    @test rel_err_dose < 2e-4
    @test argmax(dose_fcs) == argmax(dose_ref)
    @test isapprox(maximum(dose_fcs), maximum(dose_ref); rtol=1e-4)

    # Split the FCS solution into uncollided and scattered components
    φ_u = reconstruct_uncollided_flux(cs, geo, m_fcs, ss_fcs)
    φ_s = flux_total .- φ_u

    ϕ_u = scalar_flux_per_x(φ_u)
    ϕ_s = scalar_flux_per_x(φ_s)
    ϕ_t = scalar_flux_per_x(flux_total)
    ϕ_ref = scalar_flux_per_x(flux_ref)

    # Total scalar flux must match the reference solve
    rel_err_scalar = norm(ϕ_t .- ϕ_ref) / max(norm(ϕ_ref), 1e-16)
    @test rel_err_scalar < 1e-4
    @test isapprox(ϕ_t, ϕ_ref; rtol=1e-4)
    @test argmax(ϕ_t) == argmax(ϕ_ref)

    # Components are physically non-negative up to DG moment-extraction noise
    @test minimum(ϕ_u) >= -1e-6 * maximum(ϕ_u)
    @test minimum(ϕ_s) >= -1e-6 * maximum(ϕ_s)

    # Conservation: total = uncollided + scattered
    @test isapprox(ϕ_t, ϕ_u .+ ϕ_s; rtol=1e-12)

    # Plot the three scalar-flux components versus depth
    y_min = max(1e-8, minimum(filter(>(0), [ϕ_u; ϕ_s; ϕ_t; ϕ_ref])))
    y_max = maximum([ϕ_u; ϕ_s; ϕ_t; ϕ_ref]) * 2.0
    p = plot(xlabel="Depth x (cm)", ylabel="Scalar flux (zeroth angular moment)",
             title="1D BFP FCS: uncollided / scattered / total scalar flux",
             legend=:topright, yscale=:log10, legendfontcolor=:black, legendfontsize=10,
             ylims=(y_min, y_max))
    plot!(p, x, ϕ_u, label="Uncollided φ_u", linewidth=2, linestyle=:dash)
    plot!(p, x, ϕ_s, label="Scattered φ_s", linewidth=2, linestyle=:dot)
    plot!(p, x, ϕ_t, label="Total φ_u + φ_s", linewidth=2, color=:black)
    plot!(p, x, ϕ_ref, label="Reference (standard SN)", linewidth=2, linestyle=:dashdot, color=:red)

    result_dir = joinpath(@__DIR__, "results")
    mkpath(result_dir)
    plot_path = joinpath(result_dir, "fcs_bfp_flux_moments.png")
    savefig(p, plot_path)
    @test isfile(plot_path)
    println("Saved plot to $plot_path")

    println("\nScalar flux statistics:")
    println("Max uncollided: ", maximum(ϕ_u), " at x = ", x[argmax(ϕ_u)])
    println("Max scattered:  ", maximum(ϕ_s), " at x = ", x[argmax(ϕ_s)])
    println("Max total:      ", maximum(ϕ_t), " at x = ", x[argmax(ϕ_t)])
    println("Integral uncollided / total = ", sum(ϕ_u) / sum(ϕ_t))
    println("Integral scattered  / total = ", sum(ϕ_s) / sum(ϕ_t))
    println("Relative L2 error total vs reference: ", rel_err_scalar)
end
