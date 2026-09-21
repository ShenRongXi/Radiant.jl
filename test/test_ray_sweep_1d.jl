using Radiant
using LinearAlgebra
using Test

"""
    setup_1d_problem_ray(Σa, Σs; Nx=80, Lx=5.0, N=4, L=3, solver_type="BTE", scheme_x_order=1, scheme_E_order=1)

Minimal helper for ray-sweep tests.  Similar to runtests.jl `setup_1d_problem`,
but allows choosing the solver type and spatial/energy scheme orders.
"""
function setup_1d_problem_ray(Σa, Σs; Nx=80, Lx=5.0, N=4, L=3, solver_type="BTE", scheme_x_order=1, scheme_E_order=1)
    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)

    electron = Electron()

    cs = Cross_Sections()
    cs.set_materials(water)
    cs.set_particles([electron])
    cs.set_source("custom")
    cs.set_absorption([Σa])
    cs.set_scattering([Σs])
    cs.set_legendre_order(L)
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [Nx])
    geo.set_region_boundaries("x", [0.0, Lx])
    geo.build(cs)

    m = SN()
    m.set_particle(electron)
    m.set_solver_type(solver_type)
    m.set_quadrature("gauss-legendre", N, 1)
    m.set_legendre_order(L)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DG", scheme_x_order)
    if uppercase(solver_type) ∈ ["BFP", "BCSD"]
        m.set_scheme("E", "DG", scheme_E_order)
    end

    return cs, geo, m, electron
end

@testset "ray_sweep_1D" begin

    @testset "BTE pure absorber: ray vs analytic exponential" begin
        cs, geo, m, electron = setup_1d_problem_ray(1.0, 0.0; Nx=40, Lx=4.0, N=8, L=5)
        ad = Radiant.build_angular_discretization(m, geo)

        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")

        source = Source(electron, cs, geo, m)
        source.add_source(ss)

        φ_u_ray, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad; use_ray_sweep=true)
        φ_u_sn,  _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad; use_ray_sweep=false)

        x = geo.get_voxels_position("x")
        Δx = geo.get_voxels_width()[1]
        Σt = 1.0
        for ix in eachindex(x)
            xl = x[ix] - Δx[ix] / 2
            xr = x[ix] + Δx[ix] / 2
            expected = (exp(-Σt * xl) - exp(-Σt * xr)) / (Σt * Δx[ix])
            @test isapprox(φ_u_ray[1, 1, 1, ix, 1, 1], expected; rtol=1e-12)
            @test φ_u_ray[1, 1, 1, ix, 1, 1] > 0.0
            @test φ_u_sn[1, 1, 1, ix, 1, 1] > 0.0
        end
    end

    @testset "BTE μ<0 direction symmetry" begin
        cs, geo, m, electron = setup_1d_problem_ray(1.0, 0.0; Nx=40, Lx=4.0, N=8, L=5)
        ad = Radiant.build_angular_discretization(m, geo)

        ss_plus = Surface_Source()
        ss_plus.set_particle(electron)
        ss_plus.set_intensity(1.0)
        ss_plus.set_energy_group(1)
        ss_plus.set_direction([-1.0, 0.0, 0.0])
        ss_plus.set_location("x+")

        source = Source(electron, cs, geo, m)
        source.add_source(ss_plus)

        φ_u_ray, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad; use_ray_sweep=true)
        x = geo.get_voxels_position("x")
        Δx = geo.get_voxels_width()[1]
        Lx = 4.0
        Σt = 1.0
        for ix in eachindex(x)
            xl = x[ix] - Δx[ix] / 2
            xr = x[ix] + Δx[ix] / 2
            expected = (exp(-Σt * (Lx - xr)) - exp(-Σt * (Lx - xl))) / (Σt * Δx[ix])
            @test isapprox(φ_u_ray[1, 1, 1, ix, 1, 1], expected; rtol=1e-12)
        end
    end

    @testset "Default path uses ray sweep (matches explicit true)" begin
        cs, geo, m, electron = setup_1d_problem_ray(1.0, 0.0; Nx=40, Lx=4.0, N=8, L=5)
        ad = Radiant.build_angular_discretization(m, geo)

        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")

        source = Source(electron, cs, geo, m)
        source.add_source(ss)

        φ_default, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)
        φ_explicit, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad; use_ray_sweep=true)
        @test φ_default ≈ φ_explicit
    end

    @testset "BTE 𝒪x=2 pure absorber: cell average and spatial moment" begin
        cs, geo, m, electron = setup_1d_problem_ray(1.0, 0.0; Nx=40, Lx=4.0, N=8, L=5, scheme_x_order=2)
        ad = Radiant.build_angular_discretization(m, geo)

        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")

        source = Source(electron, cs, geo, m)
        source.add_source(ss)

        φ_u_ray, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad; use_ray_sweep=true)

        x = geo.get_voxels_position("x")
        Δx = geo.get_voxels_width()[1]
        Σt = 1.0
        for ix in eachindex(x)
            xl = x[ix] - Δx[ix] / 2
            xr = x[ix] + Δx[ix] / 2
            # Cell-average scalar flux for unit incident intensity
            expected_avg = (exp(-Σt * xl) - exp(-Σt * xr)) / (Σt * Δx[ix])
            @test isapprox(φ_u_ray[1, 1, 1, ix, 1, 1], expected_avg; rtol=1e-12)
        end
    end

end
