using Radiant
using LinearAlgebra
using Test

# Headless PNG output for CI (in case any plot routine is called)
ENV["GKSwstype"] = "100"

# Shared particle / material instances.  Cross-sections store data keyed by the
# particle objects, so the same instances must be passed to sources and solvers.
const water_ref   = Material("water")
const electron_ref = Electron()
const photon_ref   = Photon()
const positron_ref = Positron()

# -----------------------------------------------------------------------------
# Shared 1D water cross-sections and geometry for the three benchmark variants
# -----------------------------------------------------------------------------
function make_water_geometry(cs::Cross_Sections)
    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water_ref])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [40])
    geo.set_region_boundaries("x", [0.0, 5.0])
    geo.build(cs)
    return geo
end

function make_water_cross_sections(L::Int=15)
    water_ref.set_density(1.0)
    water_ref.add_element("H", 0.1119)
    water_ref.add_element("O", 0.8881)

    cs = Cross_Sections()
    cs.set_materials(water_ref)
    cs.set_particles([electron_ref, photon_ref, positron_ref])
    cs.set_group_structure("log", 80, 10.0, 0.001)
    cs.set_interactions([Inelastic_Collision(), Elastic_Collision(), Bremsstrahlung(), Pair_Production(), Photoelectric(), Rayleigh(), Compton(), Auger(), Fluorescence(), Annihilation()])
    cs.set_legendre_order(L)

    xsec_file = joinpath(@__DIR__, "..", "..", "FCS-Examples", "Paper_2025_AFP", "water_p$(L).xsec")
    if isfile(xsec_file)
        cs.set_source("fmac-m")
        cs.set_file(xsec_file)
    else
        cs.set_source("physics-models")
    end
    cs.build()
    return cs
end

# -----------------------------------------------------------------------------
# Run a single FCS-enabled BFP case, toggling the uncollided sweep kernel.
# -----------------------------------------------------------------------------
function run_fcs_bfp_case(cs::Cross_Sections, geo::Geometry,
                          quadrature_type::String, N::Int, Qdims::Int, L::Int;
                          use_ray_sweep::Bool=true)
    ss = Surface_Source()
    ss.set_particle(electron_ref)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

    m = Discrete_Ordinates()
    m.set_particle(electron_ref)
    m.set_solver_type("BFP")
    m.set_acceleration("livolant")
    m.set_quadrature(quadrature_type, N, Qdims)
    m.set_legendre_order(L)
    m.set_angular_boltzmann("galerkin-d")
    m.set_angular_fokker_planck("finite-difference")
    m.set_convergence_criterion(1e-5)
    m.set_maximum_iteration(500)
    m.set_scheme("E", "DG", 2)
    m.set_scheme("x", "DG", 2)
    m.set_is_first_collision_source(true)
    m.set_use_ray_sweep(use_ray_sweep)

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
    return x, dose
end

# -----------------------------------------------------------------------------
# Regression tests: final dose from ray sweep must match SN sweep to the
# discretization tolerance (the two kernels solve the same FCS equations).
# -----------------------------------------------------------------------------
const cs = make_water_cross_sections(15)
const geo = make_water_geometry(cs)

@testset "FCS ray vs SN sweep: Gauss-Lobatto (benchmark1)" begin
    x_ray, dose_ray = run_fcs_bfp_case(cs, geo, "Gauss-Lobatto", 12, 1, 15; use_ray_sweep=true)
    x_sn,  dose_sn  = run_fcs_bfp_case(cs, geo, "Gauss-Lobatto", 12, 1, 15; use_ray_sweep=false)

    @test x_ray ≈ x_sn
    rel_err = norm(dose_ray .- dose_sn) / max(norm(dose_sn), 1e-16)
    @test rel_err < 1e-3
    @test isapprox(dose_ray, dose_sn; rtol=1e-3)
end

@testset "FCS ray vs SN sweep: Carlson (benchmark2)" begin
    x_ray, dose_ray = run_fcs_bfp_case(cs, geo, "Carlson", 4, 3, 15; use_ray_sweep=true)
    x_sn,  dose_sn  = run_fcs_bfp_case(cs, geo, "Carlson", 4, 3, 15; use_ray_sweep=false)

    @test x_ray ≈ x_sn
    rel_err = norm(dose_ray .- dose_sn) / max(norm(dose_sn), 1e-16)
    @test rel_err < 1e-3
    @test isapprox(dose_ray, dose_sn; rtol=1e-3)
end

@testset "FCS ray vs SN sweep: Lebedev (benchmark2)" begin
    x_ray, dose_ray = run_fcs_bfp_case(cs, geo, "Lebedev", 9, 3, 15; use_ray_sweep=true)
    x_sn,  dose_sn  = run_fcs_bfp_case(cs, geo, "Lebedev", 9, 3, 15; use_ray_sweep=false)

    @test x_ray ≈ x_sn
    rel_err = norm(dose_ray .- dose_sn) / max(norm(dose_sn), 1e-16)
    @test rel_err < 1e-3
    @test isapprox(dose_ray, dose_sn; rtol=1e-3)
end
