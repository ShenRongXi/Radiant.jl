using Radiant
using LinearAlgebra
using Test

# -----------------------------------------------------------------------------
# API tests for fcs_ray_spatial_order
# -----------------------------------------------------------------------------
@testset "FCS mixed spatial order API" begin
    sn = SN()
    @test sn.get_fcs_ray_spatial_order() == 0

    sn.set_fcs_ray_spatial_order(1)
    @test sn.get_fcs_ray_spatial_order() == 1

    sn.set_fcs_ray_spatial_order(2)
    @test sn.get_fcs_ray_spatial_order() == 2

    sn.set_fcs_ray_spatial_order(0)
    @test sn.get_fcs_ray_spatial_order() == 0

    @test_throws ErrorException sn.set_fcs_ray_spatial_order(-1)
end

# -----------------------------------------------------------------------------
# Helper tests
# -----------------------------------------------------------------------------
@testset "FCS mixed spatial order helper" begin
    # non-FC
    Nm = Radiant._compute_nm_from_o([2, 1, 1, 2], false)
    @test Nm[5] == 3
    @test Nm[4] == 2

    # FC
    Nm_fc = Radiant._compute_nm_from_o([2, 1, 1, 2], true)
    @test Nm_fc[5] == 4
    @test Nm_fc[4] == 2

    # 1D BTE-like (Ox=2, other axes order 1)
    Nm_bte = Radiant._compute_nm_from_o([2, 1, 1, 1], false)
    @test Nm_bte[5] == 2
end

# -----------------------------------------------------------------------------
# Shared problem setup helpers
# -----------------------------------------------------------------------------
function _make_1d_bte_problem(; Ox=2, Nx=80, Lx=5.0, N=4, L=3, isFC=true)
    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)

    electron = Electron()

    cs = Cross_Sections()
    cs.set_materials(water)
    cs.set_particles([electron])
    cs.set_source("custom")
    cs.set_absorption([1.0])
    cs.set_scattering([0.0])
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
    m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre", N, 1)
    m.set_legendre_order(L)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DG", Ox)
    m.set_is_full_coupling(isFC)

    return cs, geo, m, electron
end

function _make_2d_bte_problem(; Nx=20, Ny=20, Lx=2.0, Ly=2.0, N=4, L=2)
    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)

    electron = Electron()

    cs = Cross_Sections()
    cs.set_materials(water)
    cs.set_particles([electron])
    cs.set_source("custom")
    cs.set_absorption([1.0])
    cs.set_scattering([0.0])
    cs.set_legendre_order(L)
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(2)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_boundary_conditions("y-", "void")
    geo.set_boundary_conditions("y+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_number_of_regions("y", 1)
    geo.set_voxels_per_region("x", [Nx])
    geo.set_voxels_per_region("y", [Ny])
    geo.set_region_boundaries("x", [0.0, Lx])
    geo.set_region_boundaries("y", [0.0, Ly])
    geo.build(cs)

    m = SN()
    m.set_particle(electron)
    m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre-chebychev", N, 2)
    m.set_legendre_order(L)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DD", 1)
    m.set_scheme("y", "DD", 1)

    return cs, geo, m, electron
end

function _make_3d_bte_problem(; Nx=10, Ny=10, Nz=10, Lx=1.5, Ly=1.5, Lz=1.5, N=4, L=2)
    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)

    electron = Electron()

    cs = Cross_Sections()
    cs.set_materials(water)
    cs.set_particles([electron])
    cs.set_source("custom")
    cs.set_absorption([1.0])
    cs.set_scattering([0.0])
    cs.set_legendre_order(L)
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(3)
    for face in ["x-", "x+", "y-", "y+", "z-", "z+"]
        geo.set_boundary_conditions(face, "void")
    end
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_number_of_regions("y", 1)
    geo.set_number_of_regions("z", 1)
    geo.set_voxels_per_region("x", [Nx])
    geo.set_voxels_per_region("y", [Ny])
    geo.set_voxels_per_region("z", [Nz])
    geo.set_region_boundaries("x", [0.0, Lx])
    geo.set_region_boundaries("y", [0.0, Ly])
    geo.set_region_boundaries("z", [0.0, Lz])
    geo.build(cs)

    m = SN()
    m.set_particle(electron)
    m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre-chebychev", N, 3)
    m.set_legendre_order(L)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DD", 1)
    m.set_scheme("y", "DD", 1)
    m.set_scheme("z", "DD", 1)

    return cs, geo, m, electron
end

# -----------------------------------------------------------------------------
# Scope tests: mixed-order only affects 1D ray-sweep path
# -----------------------------------------------------------------------------
@testset "FCS mixed spatial order scope" begin
    # 1D with use_ray_sweep=false: setting fcs_ray_spatial_order should not change result
    cs_1d, geo_1d, m_1d, electron = _make_1d_bte_problem(; Ox=2, Nx=40, Lx=2.0, N=4, L=3)
    ad_1d = Radiant.build_angular_discretization(m_1d, geo_1d)

    ss = Surface_Source()
    ss.set_particle(electron)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

    source = Source(electron, cs_1d, geo_1d, m_1d)
    source.add_source(ss)

    m_1d.set_use_ray_sweep(false)
    m_1d.set_fcs_ray_spatial_order(1)
    φ_u_ray_off_order1, _ = Radiant._compute_uncollided_surface_flux(cs_1d, geo_1d, m_1d, source, ad_1d; use_ray_sweep=false)

    m_1d.set_fcs_ray_spatial_order(0)
    φ_u_ray_off_order0, _ = Radiant._compute_uncollided_surface_flux(cs_1d, geo_1d, m_1d, source, ad_1d; use_ray_sweep=false)

    @test φ_u_ray_off_order1 ≈ φ_u_ray_off_order0

    # 2D: fcs_ray_spatial_order should not affect result
    cs_2d, geo_2d, m_2d, electron = _make_2d_bte_problem()
    ad_2d = Radiant.build_angular_discretization(m_2d, geo_2d)

    ss_2d = Surface_Source()
    ss_2d.set_particle(electron)
    ss_2d.set_intensity(1.0)
    ss_2d.set_energy_group(1)
    ss_2d.set_direction([1.0, 0.0, 0.0])
    ss_2d.set_location("x-")
    ss_2d.set_boundaries("y", [0.0, 2.0])

    source_2d = Source(electron, cs_2d, geo_2d, m_2d)
    source_2d.add_source(ss_2d)

    m_2d.set_fcs_ray_spatial_order(1)
    φ_u_2d_order1, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_2d, ad_2d)

    m_2d.set_fcs_ray_spatial_order(0)
    φ_u_2d_order0, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_2d, ad_2d)

    @test φ_u_2d_order1 ≈ φ_u_2d_order0

    # 3D: fcs_ray_spatial_order should not affect result
    cs_3d, geo_3d, m_3d, electron = _make_3d_bte_problem()
    ad_3d = Radiant.build_angular_discretization(m_3d, geo_3d)

    ss_3d = Surface_Source()
    ss_3d.set_particle(electron)
    ss_3d.set_intensity(1.0)
    ss_3d.set_energy_group(1)
    ss_3d.set_direction([1.0, 0.0, 0.0])
    ss_3d.set_location("x-")
    ss_3d.set_boundaries("y", [0.0, 1.5])
    ss_3d.set_boundaries("z", [0.0, 1.5])

    source_3d = Source(electron, cs_3d, geo_3d, m_3d)
    source_3d.add_source(ss_3d)

    m_3d.set_fcs_ray_spatial_order(1)
    φ_u_3d_order1, _ = Radiant._compute_uncollided_surface_flux(cs_3d, geo_3d, m_3d, source_3d, ad_3d)

    m_3d.set_fcs_ray_spatial_order(0)
    φ_u_3d_order0, _ = Radiant._compute_uncollided_surface_flux(cs_3d, geo_3d, m_3d, source_3d, ad_3d)

    @test φ_u_3d_order1 ≈ φ_u_3d_order0
end

# -----------------------------------------------------------------------------
# Degeneration tests
# -----------------------------------------------------------------------------
@testset "FCS mixed spatial order degeneration" begin
    for isFC in [false, true]
        # solver Ox=1: any fcs_ray_spatial_order >= 1 falls back to default
        cs, geo, m, electron = _make_1d_bte_problem(; Ox=1, Nx=40, Lx=2.0, N=4, L=3, isFC=isFC)
        ad = Radiant.build_angular_discretization(m, geo)

        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")

        source = Source(electron, cs, geo, m)
        source.add_source(ss)

        m.set_fcs_ray_spatial_order(0)
        φ_u_default, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)

        m.set_fcs_ray_spatial_order(1)
        φ_u_order1, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)

        m.set_fcs_ray_spatial_order(2)
        φ_u_order2, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)

        @test φ_u_default ≈ φ_u_order1
        @test φ_u_default ≈ φ_u_order2

        # solver Ox=2: fcs_ray_spatial_order=2 or 0 are equivalent
        cs2, geo2, m2, _ = _make_1d_bte_problem(; Ox=2, Nx=40, Lx=2.0, N=4, L=3, isFC=isFC)
        ad2 = Radiant.build_angular_discretization(m2, geo2)

        source2 = Source(electron, cs2, geo2, m2)
        source2.add_source(ss)

        m2.set_fcs_ray_spatial_order(0)
        φ_u2_default, _ = Radiant._compute_uncollided_surface_flux(cs2, geo2, m2, source2, ad2)

        m2.set_fcs_ray_spatial_order(2)
        φ_u2_order2, _ = Radiant._compute_uncollided_surface_flux(cs2, geo2, m2, source2, ad2)

        @test φ_u2_default ≈ φ_u2_order2
    end
end

# -----------------------------------------------------------------------------
# BTE pure absorber tests
# -----------------------------------------------------------------------------
@testset "FCS mixed spatial order BTE pure absorber" begin
    for isFC in [false, true]
        Σt = 1.0
        Nx = 80
        Lx = 5.0
        cs, geo, m, electron = _make_1d_bte_problem(; Ox=2, Nx=Nx, Lx=Lx, N=8, L=5, isFC=isFC)
        ad = Radiant.build_angular_discretization(m, geo)

        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")

        source = Source(electron, cs, geo, m)
        source.add_source(ss)

        # Mixed-order: Ox=1 for uncollided sweep, Ox=2 for solver
        m.set_fcs_ray_spatial_order(1)
        φ_u_mix, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)

        # High-order reference: Ox=2 for both
        m.set_fcs_ray_spatial_order(0)
        φ_u_high, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)

        x = geo.get_voxels_position("x")
        Δx = geo.get_voxels_width()[1]

        # Zeroth spatial moment should match the cell average of exp(-Σt*x)
        for ix in 1:Nx
            x_left = x[ix] - Δx[ix] / 2
            x_right = x[ix] + Δx[ix] / 2
            expected = (exp(-Σt * x_left) - exp(-Σt * x_right)) / (Σt * Δx[ix])
            @test isapprox(φ_u_mix[1, 1, 1, ix, 1, 1], expected; rtol=0.05)
            @test isapprox(φ_u_high[1, 1, 1, ix, 1, 1], expected; rtol=0.05)
        end

        # Zeroth moment of mixed-order should match high-order
        @test φ_u_mix[1, 1, 1, :, 1, 1] ≈ φ_u_high[1, 1, 1, :, 1, 1]

        # High-order spatial moments in mixed-order should be zero.
        # For BTE with Ox=2, Nm[5] == 2; the second moment is the linear spatial moment.
        @test all(φ_u_mix[1, 1, 2, :, 1, 1] .== 0.0)
    end
end

# -----------------------------------------------------------------------------
# BFP water benchmark tests
# -----------------------------------------------------------------------------
function _make_water_cross_sections(L::Int=15)
    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)

    electron = Electron()
    photon = Photon()
    positron = Positron()

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
    return cs, water, electron
end

function _make_water_geometry(water)
    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [80])
    geo.set_region_boundaries("x", [0.0, 5.0])
    return geo
end

function _run_fcs_bfp_case(cs, geo, water, electron; Ox=2, OE=2, fcs_ray_order=0, quadrature_type="Gauss-Lobatto", N=12)
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
    m.set_quadrature(quadrature_type, N, 1)
    m.set_legendre_order(15)
    m.set_angular_boltzmann("galerkin-d")
    m.set_angular_fokker_planck("finite-difference")
    m.set_convergence_criterion(1e-5)
    m.set_maximum_iteration(500)
    m.set_scheme("E", "DG", OE)
    m.set_scheme("x", "DG", Ox)
    m.set_is_first_collision_source(true)
    m.set_use_ray_sweep(true)
    m.set_fcs_ray_spatial_order(fcs_ray_order)

    solvers = Solvers()
    solvers.add_solver(m)

    # Geometry must be built with the cross-sections
    geo.build(cs)

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
    return x, dose, flux
end

@testset "FCS mixed spatial order BFP water benchmark" begin
    cs, water, electron = _make_water_cross_sections(15)
    geo = _make_water_geometry(water)

    x_high, dose_high, _ = _run_fcs_bfp_case(cs, geo, water, electron; Ox=2, OE=2, fcs_ray_order=0)

    # Re-build geometry for the mixed-order case to avoid shared-state issues
    geo_mix = _make_water_geometry(water)
    x_mix, dose_mix, _ = _run_fcs_bfp_case(cs, geo_mix, water, electron; Ox=2, OE=2, fcs_ray_order=1)

    @test x_high ≈ x_mix
    rel_err = norm(dose_high .- dose_mix) / max(norm(dose_high), 1e-16)
    @test rel_err < 1e-3
    @test isapprox(dose_high, dose_mix; rtol=1e-3)

    # Note: ray_sweep_1D currently only supports 𝒪E ≤ 2 in its cell kernel, so we
    # test the dimension-swap scenario via the mixed-order path itself:
    # solver Ox=2, OE=2 with fcs_ray_order=1 gives Ox_ray=1 ≠ OE=2.
end

# -----------------------------------------------------------------------------
# Illegal value handling
# -----------------------------------------------------------------------------
@testset "FCS mixed spatial order illegal values" begin
    cs, geo, m, electron = _make_1d_bte_problem(; Ox=4, Nx=20, Lx=2.0, N=4, L=3)
    ad = Radiant.build_angular_discretization(m, geo)

    ss = Surface_Source()
    ss.set_particle(electron)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

    source = Source(electron, cs, geo, m)
    source.add_source(ss)

    m.set_fcs_ray_spatial_order(3)
    @test_throws ErrorException Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)
end
