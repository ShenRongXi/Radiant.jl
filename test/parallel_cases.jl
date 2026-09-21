# Shared standard-SN problem definitions for the serial-vs-parallel regression.
#
# The same functions are used to (a) generate single-threaded serial reference fluxes and
# (b) recompute the flux inside the test suite (which may run multithreaded). Comparing the
# two validates that the direction-parallel sweep is numerically equivalent to serial.

using Radiant

function _pc_material()
    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)
    return water
end

# 1D standard SN with in-group scattering and an interior volume source.
function run_sn_1d_case()
    water = _pc_material()
    electron = Electron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron]); cs.set_source("custom")
    cs.set_absorption([0.4]); cs.set_scattering([0.6]); cs.set_legendre_order(3); cs.build()

    geo = Geometry()
    geo.set_type("cartesian"); geo.set_dimension(1)
    geo.set_boundary_conditions("x-", "void"); geo.set_boundary_conditions("x+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1); geo.set_voxels_per_region("x", [20])
    geo.set_region_boundaries("x", [0.0, 2.0]); geo.build(cs)

    m = SN()
    m.set_particle(electron); m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre", 8, 1); m.set_legendre_order(3)
    m.set_angular_boltzmann("standard"); m.set_scheme("x", "DD", 1)

    source = Source(electron, cs, geo, m)
    vs = Volume_Source(); vs.set_particle(electron); vs.set_intensity(1.0)
    vs.set_energy_group(1); vs.set_boundaries("x", [0.0, 2.0])
    source.add_source(vs)
    return Radiant.compute_flux(cs, geo, m, source).get_flux()
end

# 2D standard SN with in-group scattering and an interior volume source.
function run_sn_2d_case()
    water = _pc_material()
    electron = Electron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron]); cs.set_source("custom")
    cs.set_absorption([0.4]); cs.set_scattering([0.6]); cs.set_legendre_order(2); cs.build()

    geo = Geometry()
    geo.set_type("cartesian"); geo.set_dimension(2)
    geo.set_boundary_conditions("x-", "void"); geo.set_boundary_conditions("x+", "void")
    geo.set_boundary_conditions("y-", "void"); geo.set_boundary_conditions("y+", "void")
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1); geo.set_number_of_regions("y", 1)
    geo.set_voxels_per_region("x", [10]); geo.set_voxels_per_region("y", [10])
    geo.set_region_boundaries("x", [0.0, 2.0]); geo.set_region_boundaries("y", [0.0, 2.0])
    geo.build(cs)

    m = SN()
    m.set_particle(electron); m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre-chebychev", 8, 2); m.set_legendre_order(2)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DD", 1); m.set_scheme("y", "DD", 1)

    source = Source(electron, cs, geo, m)
    vs = Volume_Source(); vs.set_particle(electron); vs.set_intensity(1.0)
    vs.set_energy_group(1); vs.set_boundaries("x", [0.0, 2.0]); vs.set_boundaries("y", [0.0, 2.0])
    source.add_source(vs)
    return Radiant.compute_flux(cs, geo, m, source).get_flux()
end

# 3D standard SN with in-group scattering and an interior volume source.
function run_sn_3d_case()
    water = _pc_material()
    electron = Electron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron]); cs.set_source("custom")
    cs.set_absorption([0.4]); cs.set_scattering([0.6]); cs.set_legendre_order(2); cs.build()

    geo = Geometry()
    geo.set_type("cartesian"); geo.set_dimension(3)
    for f in ("x-","x+","y-","y+","z-","z+") geo.set_boundary_conditions(f, "void") end
    geo.set_material_per_region([water])
    geo.set_number_of_regions("x", 1); geo.set_number_of_regions("y", 1); geo.set_number_of_regions("z", 1)
    geo.set_voxels_per_region("x", [8]); geo.set_voxels_per_region("y", [8]); geo.set_voxels_per_region("z", [8])
    geo.set_region_boundaries("x", [0.0, 1.5]); geo.set_region_boundaries("y", [0.0, 1.5]); geo.set_region_boundaries("z", [0.0, 1.5])
    geo.build(cs)

    m = SN()
    m.set_particle(electron); m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre-chebychev", 4, 3); m.set_legendre_order(2)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DD", 1); m.set_scheme("y", "DD", 1); m.set_scheme("z", "DD", 1)

    source = Source(electron, cs, geo, m)
    vs = Volume_Source(); vs.set_particle(electron); vs.set_intensity(1.0)
    vs.set_energy_group(1)
    vs.set_boundaries("x", [0.0, 1.5]); vs.set_boundaries("y", [0.0, 1.5]); vs.set_boundaries("z", [0.0, 1.5])
    source.add_source(vs)
    return Radiant.compute_flux(cs, geo, m, source).get_flux()
end

# ---------------------------------------------------------------------------------------
# BFP (Boltzmann Fokker-Planck) cases. These are CSD problems (isCSD=true) using the
# angular Fokker-Planck operator, so they exercise two parallel paths the BTE cases do not:
# the per-direction energy-flux write 𝚽E12_temp[n,...] inside the parallel sweep, and the
# fokker_planck_source term. A small (4-group) water electron cross-section set is loaded
# from a committed fmac-m file so the build is fast and bit-reproducible; the same file is
# used by the reference generator and the test, so serial and parallel solves see identical
# cross-sections. Configuration mirrors the Paper_2025_AFP BFP benchmarks (Lebedev
# quadrature, galerkin-d basis, finite-difference FP).

const _BFP_XSEC = joinpath(@__DIR__, "data", "water_bfp_small.xsec")
# Cache the (cross_sections, electron, water) tuple so the three BFP cases share one build.
const _BFP_CACHE = Ref{Any}(nothing)

function _bfp_setup()
    if _BFP_CACHE[] === nothing
        water = _pc_material()
        electron = Electron()
        cs = Cross_Sections()
        cs.set_materials(water); cs.set_particles([electron])
        cs.set_group_structure("log", 4, 1.0, 0.05)
        cs.set_interactions([Inelastic_Collision(), Elastic_Collision()])
        cs.set_legendre_order(7)
        cs.set_source("fmac-m"); cs.set_file(_BFP_XSEC)
        cs.build()
        _BFP_CACHE[] = (cs, electron, water)
    end
    return _BFP_CACHE[]
end

function _run_bfp_case(ndims)
    cs, electron, water = _bfp_setup()
    axes = ("x","y","z")[1:ndims]
    faces = ("x-","x+","y-","y+","z-","z+")[1:2*ndims]

    geo = Geometry()
    geo.set_type("cartesian"); geo.set_dimension(ndims)
    for f in faces geo.set_boundary_conditions(f, "void") end
    geo.set_material_per_region([water])
    for ax in axes
        geo.set_number_of_regions(ax, 1); geo.set_voxels_per_region(ax, [5])
        geo.set_region_boundaries(ax, [0.0, 0.3])
    end
    geo.build(cs)

    m = SN()
    m.set_particle(electron); m.set_solver_type("BFP"); m.set_acceleration("livolant")
    m.set_quadrature("Lebedev", 7, 3); m.set_legendre_order(7)
    m.set_angular_boltzmann("galerkin-d"); m.set_angular_fokker_planck("finite-difference")
    m.set_convergence_criterion(1e-4); m.set_maximum_iteration(200)
    m.set_scheme("E", "DG", 1)
    for ax in axes m.set_scheme(ax, "DD", 1) end

    source = Source(electron, cs, geo, m)
    vs = Volume_Source(); vs.set_particle(electron); vs.set_intensity(1.0); vs.set_energy_group(1)
    for ax in axes vs.set_boundaries(ax, [0.0, 0.3]) end
    source.add_source(vs)
    return Radiant.compute_flux(cs, geo, m, source).get_flux()
end

run_bfp_1d_case() = _run_bfp_case(1)
run_bfp_2d_case() = _run_bfp_case(2)
run_bfp_3d_case() = _run_bfp_case(3)
