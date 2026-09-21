# Verify that beam FCS uncollided flux has correct units [particles/(cm²·s)],
# not face-integrated units [particles/s].  Regression test for the
# _build_sources_ig face-area factor fix (方案A).
#
# Standalone: julia --project=. test/test_beam_fcs_unit.jl
if !isdefined(Main, :Radiant)
    using Radiant
end
using LinearAlgebra
using Test

# Lightweight single-group 3D vacuum problem.
function _setup_3d_vacuum(; Ns=(5,3,3), L=(1.0,0.6,0.6), N=4, Lg=2)
    water = Material("water"); water.set_density(1.0)
    water.add_element("H", 0.1119); water.add_element("O", 0.8881)
    electron = Electron()
    cs = Cross_Sections(); cs.set_materials(water); cs.set_particles([electron])
    cs.set_source("custom"); cs.set_absorption([0.0]); cs.set_scattering([0.0])
    cs.set_legendre_order(Lg); cs.build()
    geo = Geometry(); geo.set_type("cartesian"); geo.set_dimension(3)
    for f in ("x-","x+","y-","y+","z-","z+"); geo.set_boundary_conditions(f, "void"); end
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1); geo.set_number_of_regions("y", 1); geo.set_number_of_regions("z", 1)
    geo.set_voxels_per_region("x", [Ns[1]]); geo.set_voxels_per_region("y", [Ns[2]]); geo.set_voxels_per_region("z", [Ns[3]])
    geo.set_region_boundaries("x", [0.0, L[1]]); geo.set_region_boundaries("y", [0.0, L[2]]); geo.set_region_boundaries("z", [0.0, L[3]])
    geo.build(cs)
    m = SN(); m.set_particle(electron); m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre-chebychev", N, 3); m.set_legendre_order(Lg)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DD", 1); m.set_scheme("y", "DD", 1); m.set_scheme("z", "DD", 1)
    return cs, geo, m, electron
end

function _x_beam_source(electron, cs, geo, m)
    ss = Radiant.Surface_Source()
    ss.set_particle(electron); ss.set_intensity(1.0); ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0]); ss.set_location("x-")
    ss.set_boundaries("y", [0.0, 0.6]); ss.set_boundaries("z", [0.0, 0.6])
    source = Source(electron, cs, geo, m)
    source.add_source(ss)
    return source
end

@testset "beam FCS unit: uncollided flux = flux density [particles/(cm²·s)]" begin

    # ----- Test 1: Vacuum — beam intensity preserved cell-to-cell -----
    @testset "vacuum beam (Σt=0): φ_u = I₀ in every cell" begin
        cs, geo, m, electron = _setup_3d_vacuum()
        ad = Radiant.build_angular_discretization(m, geo)
        src = _x_beam_source(electron, cs, geo, m)

        # For vacuum (Σt=0), cell-average angular flux = I₀ everywhere.
        # The zeroth angular moment Y₀₀(Ω) = 1 in Radiant's unnormalized convention.
        # φ_u[1,1,1,ix,iy,iz] should equal I₀ = 1.0 (not I₀·Δy·Δz = 0.04).
        I0 = 1.0
        Ns = geo.get_number_of_voxels()

        for use_ray in (true, false)
            φ_u, _ = Radiant._compute_uncollided_surface_flux(
                cs, geo, m, src, ad; use_ray_sweep=use_ray)

            for iz in 1:Ns[3], iy in 1:Ns[2], ix in 1:Ns[1]
                val = φ_u[1, 1, 1, ix, iy, iz]
                @test isapprox(val, I0; rtol=1e-12)
            end
        end
    end

    # ----- Test 2: Attenuated beam — matches analytic exponential decay -----
    @testset "attenuated beam (Σt=1): φ_u(x) = I₀·exp(-Σt·x_c)" begin
        Σt = 1.0
        cs, geo, m, electron = _setup_3d_vacuum()
        # Override absorption to get attenuation
        cs.set_absorption([Σt]); cs.build()
        ad = Radiant.build_angular_discretization(m, geo)
        src = _x_beam_source(electron, cs, geo, m)

        I0 = 1.0
        Ns = geo.get_number_of_voxels()
        Δs = geo.get_voxels_width()
        Δx = Δs[1]
        x_edges = geo.get_voxels_boundaries("x")

        for use_ray in (true, false)
            φ_u, _ = Radiant._compute_uncollided_surface_flux(
                cs, geo, m, src, ad; use_ray_sweep=use_ray)

            for iz in 1:Ns[3], iy in 1:Ns[2], ix in 1:Ns[1]
                # Analytic cell-average: (1/Δx) ∫ I₀·exp(-Σt·x) dx
                analytic = I0 * (exp(-Σt * x_edges[ix]) - exp(-Σt * x_edges[ix+1])) / (Σt * Δx[ix])

                val = φ_u[1, 1, 1, ix, iy, iz]
                # DD (𝒪=1) introduces ~0.3% discretisation error for Σt·Δx=0.2.
                # 5e-3 is tight enough to catch the original ×Δy·Δz bug (factor 25×)
                # while allowing the expected spatial truncation error.
                @test isapprox(val, analytic; rtol=5e-3)
            end
        end
    end

end
