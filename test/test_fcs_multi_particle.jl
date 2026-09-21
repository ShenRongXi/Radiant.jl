using Radiant
using Test

@testset "Multi-particle FCS structure" begin
    # Two solvers associated with the same electron tag: one with FCS enabled, one without.
    # This exercises index-based method lookup and verifies transport() runs without error
    # and produces non-zero flux. It is a structural test, not a physical multi-particle
    # coupling test (the latter requires physics-models cross-sections).
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
    cs.set_legendre_order(2)
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [20])
    geo.set_region_boundaries("x", [0.0, 2.0])
    geo.build(cs)

    sn1 = SN()
    sn1.set_particle(electron)
    sn1.set_solver_type("BTE")
    sn1.set_quadrature("gauss-legendre", 4, 1)
    sn1.set_legendre_order(2)
    sn1.set_angular_boltzmann("standard")
    sn1.set_scheme("x", "DD", 1)
    sn1.set_is_first_collision_source(true)

    sn2 = SN()
    sn2.set_particle(electron)
    sn2.set_solver_type("BTE")
    sn2.set_quadrature("gauss-legendre", 4, 1)
    sn2.set_legendre_order(2)
    sn2.set_angular_boltzmann("standard")
    sn2.set_scheme("x", "DD", 1)
    sn2.set_is_first_collision_source(false)

    solvers = Solvers()
    solvers.set_maximum_number_of_generations(3)
    solvers.set_convergence_criterion(1e-4)
    solvers.add_solver(sn1)
    solvers.add_solver(sn2)

    ss = Surface_Source()
    ss.set_particle(electron)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

    fixed = Fixed_Sources(cs, geo, solvers)
    fixed.add_source(ss)
    fixed.build()

    flux = Radiant.transport(cs, geo, solvers, fixed)
    @test any(flux.get_flux(electron) .!= 0.0)
end

@testset "Single-particle FCS through transport.jl" begin
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
    cs.set_legendre_order(2)
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [20])
    geo.set_region_boundaries("x", [0.0, 2.0])
    geo.build(cs)

    sn = SN()
    sn.set_particle(electron)
    sn.set_solver_type("BTE")
    sn.set_quadrature("gauss-legendre", 4, 1)
    sn.set_legendre_order(2)
    sn.set_angular_boltzmann("standard")
    sn.set_scheme("x", "DD", 1)
    sn.set_is_first_collision_source(true)

    solvers = Solvers()
    solvers.add_solver(sn)

    ss = Surface_Source()
    ss.set_particle(electron)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

    fixed = Fixed_Sources(cs, geo, solvers)
    fixed.add_source(ss)
    fixed.build()

    flux_transport = Radiant.transport(cs, geo, solvers, fixed)
    flux_direct = Radiant.compute_flux_fcs(cs, geo, sn, fixed.get_source(electron))

    @test flux_transport.get_flux(electron) ≈ flux_direct.get_flux()
    @test flux_transport.get_uncollided_flux(electron) ≈ flux_direct.get_uncollided_flux()
end

@testset "Multi-particle FCS non-zero scattering regression" begin
    # Same two-solver setup but with non-zero scattering, so that Q_FCS is non-zero.
    # This verifies _compute_fcs_precursor injects Q_FCS into the modified source and
    # that transport() still runs and produces non-zero flux.
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
    cs.set_scattering([0.5])
    cs.set_legendre_order(2)
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [20])
    geo.set_region_boundaries("x", [0.0, 2.0])
    geo.build(cs)

    sn1 = SN()
    sn1.set_particle(electron)
    sn1.set_solver_type("BTE")
    sn1.set_quadrature("gauss-legendre", 4, 1)
    sn1.set_legendre_order(2)
    sn1.set_angular_boltzmann("standard")
    sn1.set_scheme("x", "DD", 1)
    sn1.set_is_first_collision_source(true)

    sn2 = SN()
    sn2.set_particle(electron)
    sn2.set_solver_type("BTE")
    sn2.set_quadrature("gauss-legendre", 4, 1)
    sn2.set_legendre_order(2)
    sn2.set_angular_boltzmann("standard")
    sn2.set_scheme("x", "DD", 1)
    sn2.set_is_first_collision_source(false)

    solvers = Solvers()
    solvers.set_maximum_number_of_generations(3)
    solvers.set_convergence_criterion(1e-4)
    solvers.add_solver(sn1)
    solvers.add_solver(sn2)

    ss = Surface_Source()
    ss.set_particle(electron)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

    fixed = Fixed_Sources(cs, geo, solvers)
    fixed.add_source(ss)
    fixed.build()

    precursor = Radiant._compute_fcs_precursor(cs, geo, sn1, fixed.get_source(electron))
    @test !isnothing(precursor)
    flux_u, modified_source = precursor
    @test any(modified_source.volume_sources .!= 0.0)

    flux = Radiant.transport(cs, geo, solvers, fixed)
    @test any(flux.get_flux(electron) .!= 0.0)
end

@testset "Multi-particle FCS with point source" begin
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
    cs.set_legendre_order(2)
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
    geo.set_voxels_per_region("x", [8])
    geo.set_voxels_per_region("y", [8])
    geo.set_voxels_per_region("z", [8])
    geo.set_region_boundaries("x", [0.0, 1.0])
    geo.set_region_boundaries("y", [0.0, 1.0])
    geo.set_region_boundaries("z", [0.0, 1.0])
    geo.build(cs)

    sn1 = SN()
    sn1.set_particle(electron)
    sn1.set_solver_type("BTE")
    sn1.set_quadrature("gauss-legendre-chebychev", 4, 3)
    sn1.set_legendre_order(2)
    sn1.set_angular_boltzmann("standard")
    sn1.set_scheme("x", "DD", 1)
    sn1.set_scheme("y", "DD", 1)
    sn1.set_scheme("z", "DD", 1)
    sn1.set_is_first_collision_source(true)

    sn2 = deepcopy(sn1)
    sn2.set_particle(electron)
    sn2.set_is_first_collision_source(false)

    solvers = Solvers()
    solvers.set_maximum_number_of_generations(2)
    solvers.set_convergence_criterion(1e-4)
    solvers.add_solver(sn1)
    solvers.add_solver(sn2)

    ps = Point_Source()
    ps.set_particle(electron)
    ps.set_intensity(1.0)
    ps.set_energy_group(1)
    ps.set_position([0.51, 0.52, 0.53])

    fixed = Fixed_Sources(cs, geo, solvers)
    fixed.add_source(ps)
    fixed.build()

    precursor = Radiant._compute_fcs_precursor(cs, geo, sn1, fixed.get_source(electron))
    @test !isnothing(precursor)
    flux_u, modified_source = precursor
    @test any(flux_u.get_flux() .!= 0.0)
    @test isempty(modified_source.point_sources)

    flux = Radiant.transport(cs, geo, solvers, fixed)
    @test any(flux.get_flux(electron) .!= 0.0)
end

@testset "Single-particle point-source FCS through transport.jl" begin
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
    cs.set_legendre_order(2)
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
    geo.set_voxels_per_region("x", [8])
    geo.set_voxels_per_region("y", [8])
    geo.set_voxels_per_region("z", [8])
    geo.set_region_boundaries("x", [0.0, 1.0])
    geo.set_region_boundaries("y", [0.0, 1.0])
    geo.set_region_boundaries("z", [0.0, 1.0])
    geo.build(cs)

    sn = SN()
    sn.set_particle(electron)
    sn.set_solver_type("BTE")
    sn.set_quadrature("gauss-legendre-chebychev", 4, 3)
    sn.set_legendre_order(2)
    sn.set_angular_boltzmann("standard")
    sn.set_scheme("x", "DD", 1)
    sn.set_scheme("y", "DD", 1)
    sn.set_scheme("z", "DD", 1)
    sn.set_is_first_collision_source(true)

    solvers = Solvers()
    solvers.add_solver(sn)

    ps = Point_Source()
    ps.set_particle(electron)
    ps.set_intensity(1.0)
    ps.set_energy_group(1)
    ps.set_position([0.51, 0.52, 0.53])

    fixed = Fixed_Sources(cs, geo, solvers)
    fixed.add_source(ps)
    fixed.build()

    flux_transport = Radiant.transport(cs, geo, solvers, fixed)
    flux_direct = Radiant.compute_flux_fcs(cs, geo, sn, fixed.get_source(electron))

    @test flux_transport.get_flux(electron) ≈ flux_direct.get_flux()
    @test flux_transport.get_uncollided_flux(electron) ≈ flux_direct.get_uncollided_flux()
end

@testset "compute_flux_fcs exposes retrievable uncollided flux" begin
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
    cs.set_legendre_order(2)
    cs.build()

    geo = Geometry()
    geo.set_type("cartesian")
    geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1)
    geo.set_voxels_per_region("x", [20])
    geo.set_region_boundaries("x", [0.0, 2.0])
    geo.build(cs)

    sn = SN()
    sn.set_particle(electron)
    sn.set_solver_type("BTE")
    sn.set_quadrature("gauss-legendre", 4, 1)
    sn.set_legendre_order(2)
    sn.set_angular_boltzmann("standard")
    sn.set_scheme("x", "DD", 1)
    sn.set_is_first_collision_source(true)

    solvers = Solvers()
    solvers.add_solver(sn)

    ss = Surface_Source()
    ss.set_particle(electron)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

    fixed = Fixed_Sources(cs, geo, solvers)
    fixed.add_source(ss)
    fixed.build()

    flux_direct = Radiant.compute_flux_fcs(cs, geo, sn, fixed.get_source(electron))
    phi_u = flux_direct.get_uncollided_flux()
    @test any(phi_u .!= 0.0)

    # The uncollided flux should equal the total flux when scattering is zero.
    @test flux_direct.get_flux() ≈ phi_u
end
