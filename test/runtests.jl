using Radiant
using LinearAlgebra
using Test

# Helper to set up a minimal 1D Cartesian problem with custom cross-sections.
function setup_1d_problem(Σa, Σs; Nx=80, Lx=5.0, N=4, L=3)
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
    m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre", N, 1)
    m.set_legendre_order(L)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x", "DD", 1)

    return cs, geo, m, electron
end

# Helper to set up a minimal 2D Cartesian problem with custom cross-sections.
function setup_2d_problem(Σa, Σs; Nx=40, Ny=40, Lx=2.0, Ly=2.0, N=8, L=3)
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

# Helper to set up a minimal 3D Cartesian problem with custom cross-sections.
function setup_3d_problem(Σa, Σs; Nx=20, Ny=20, Nz=20, Lx=1.5, Ly=1.5, Lz=1.5, N=4, L=2)
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
    geo.set_dimension(3)
    geo.set_boundary_conditions("x-", "void")
    geo.set_boundary_conditions("x+", "void")
    geo.set_boundary_conditions("y-", "void")
    geo.set_boundary_conditions("y+", "void")
    geo.set_boundary_conditions("z-", "void")
    geo.set_boundary_conditions("z+", "void")
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

@testset "Radiant.jl" begin

    @testset "FCS infrastructure" begin
        # SN FCS flag
        sn = SN()
        @test sn.get_is_first_collision_source() == false
        sn.set_is_first_collision_source(true)
        @test sn.get_is_first_collision_source() == true

        # SN_Angular_Discretization empty constructor
        ad = SN_Angular_Discretization()
        @test ad.Nd == 0
        @test ad.Np == 0
        @test ad.Np_surf == 0
        @test ad.Qdims == 0

        # zero_surface_sources! for 1D-style surface sources
        surface_sources = Array{Union{Array{Float64},Float64}}(undef, 2, 1, 2)
        surface_sources[1,1,1] = 1.5
        surface_sources[1,1,2] = 2.5
        surface_sources[2,1,1] = [3.0, 4.0]
        surface_sources[2,1,2] = [5.0, 6.0]
        Radiant.zero_surface_sources!(surface_sources)
        @test surface_sources[1,1,1] == 0.0
        @test surface_sources[1,1,2] == 0.0
        @test surface_sources[2,1,1] == [0.0, 0.0]
        @test surface_sources[2,1,2] == [0.0, 0.0]
    end

    @testset "Source zeroing helpers" begin
        cs, geo, m, electron = setup_1d_problem(1.0, 0.0; Nx=10, Lx=1.0, N=4, L=2)
        source = Source(electron, cs, geo, m)
        source.volume_sources .= 1.0
        source.surface_sources[1,1,1] = 2.0
        push!(source.point_sources, Point_Source())
        push!(source.surface_source_objects, Surface_Source())

        Radiant.zero_surface_sources!(source)
        @test all(x -> x == 0.0, source.surface_sources)
        Radiant.zero_point_sources!(source)
        @test isempty(source.point_sources)
        Radiant.zero_surface_source_objects!(source)
        @test isempty(source.surface_source_objects)
        # volume_sources 不受影响
        @test source.volume_sources[1,1,1,1,1,1] == 1.0
    end

    @testset "Source addition merges normalization factor" begin
        cs, geo, m, electron = setup_1d_problem(1.0, 0.0; Nx=10, Lx=1.0, N=4, L=2)
        s1 = Source(electron, cs, geo, m)
        s1.normalization_factor = 1.0
        s2 = Source(electron, cs, geo, m)
        s2.normalization_factor = 2.0
        s = s1 + s2
        @test s.normalization_factor ≈ 3.0
    end

    @testset "Source add_source accumulates normalization factor" begin
        cs, geo, m, electron = setup_1d_problem(1.0, 0.0; Nx=10, Lx=1.0, N=4, L=2)
        source = Source(electron, cs, geo, m)
        @test source.get_normalization_factor() == 0.0

        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")
        source.add_source(ss)

        @test source.get_normalization_factor() > 0.0
    end

    @testset "Solvers get_method_by_index" begin
        electron = Electron()

        sn1 = SN()
        sn1.set_particle(electron)
        sn1.set_is_first_collision_source(true)

        sn2 = SN()
        sn2.set_particle(electron)
        sn2.set_is_first_collision_source(false)

        solvers = Solvers()
        solvers.add_solver(sn1)
        solvers.add_solver(sn2)

        @test solvers.get_method_by_index(1) === sn1
        @test solvers.get_method_by_index(2) === sn2
        @test_throws ErrorException solvers.get_method_by_index(0)
        @test_throws ErrorException solvers.get_method_by_index(3)
    end

    @testset "FCS 1D uncollided surface flux" begin
        cs, geo, m, electron = setup_1d_problem(1.0, 0.0; Nx=100, Lx=5.0, N=8, L=5)

        ad = Radiant.build_angular_discretization(m, geo)
        @test ad.Nd == 8
        @test ad.Np == 6  # L+1 for standard SN
        @test ad.Qdims == 1

        # Beam basis coefficients for normally incident beam
        Dn_beam, M_beam = Radiant._beam_basis_coefficients([1.0, 0.0, 0.0], "X-", ad, "standard")
        @test length(Dn_beam) == ad.Np
        @test M_beam ≈ [1.0]

        # Surface source object
        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")

        source = Source(electron, cs, geo, m)
        source.add_source(ss)
        @test length(source.get_surface_source_objects()) == 1

        # Uncollided flux in pure absorber: φ ≈ I0 * exp(-Σt * x)
        φ_u, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source, ad)
        x = geo.get_voxels_position("x")
        Σt = 1.0
        for ix in eachindex(x)
            expected = exp(-Σt * x[ix])
            # scalar flux moment is [ig=1, p=1, is=1, ix]
            @test isapprox(φ_u[1, 1, 1, ix, 1, 1], expected; rtol=0.05)
        end

        # Multi-source superposition
        ss2 = Surface_Source()
        ss2.set_particle(electron)
        ss2.set_intensity(2.0)
        ss2.set_energy_group(1)
        ss2.set_direction([1.0, 0.0, 0.0])
        ss2.set_location("x-")
        source2 = Source(electron, cs, geo, m)
        source2.add_source(ss)
        source2.add_source(ss2)
        φ_u2, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, source2, ad)
        @test φ_u2 ≈ 3 * φ_u

        # Beam direction mismatch with location
        ss_bad = Surface_Source()
        ss_bad.set_particle(electron)
        ss_bad.set_intensity(1.0)
        ss_bad.set_energy_group(1)
        ss_bad.set_direction([-1.0, 0.0, 0.0])
        ss_bad.set_location("x-")
        source_bad = Source(electron, cs, geo, m)
        source_bad.add_source(ss_bad)
        @test_throws ErrorException Radiant._compute_uncollided_surface_flux(cs, geo, m, source_bad, ad)
    end

    @testset "FCS first collision source and full solve" begin
        # First collision source is non-zero with scattering
        cs_s, geo_s, m_s, electron = setup_1d_problem(0.5, 0.5; Nx=20, Lx=2.0, N=4, L=3)
        ad = Radiant.build_angular_discretization(m_s, geo_s)

        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")
        source = Source(electron, cs_s, geo_s, m_s)
        source.add_source(ss)

        φ_u, _ = Radiant._compute_uncollided_surface_flux(cs_s, geo_s, m_s, source, ad)
        Q_FCS = Radiant._compute_first_collision_source(cs_s, geo_s, m_s, φ_u, ad, Array{Float64}(undef), Array{Float64}(undef,0,0))
        @test any(Q_FCS .!= 0.0)

        # Standard and FCS solve give consistent scalar flux for a volume source
        # (no surface sources -> fallback path)
        cs_p, geo_p, m_p, _ = setup_1d_problem(0.5, 0.0; Nx=20, Lx=2.0, N=4, L=3)
        source_vol = Source(electron, cs_p, geo_p, m_p)
        vs = Volume_Source()
        vs.set_particle(electron)
        vs.set_intensity(1.0)
        vs.set_energy_group(1)
        vs.set_boundaries("x", [0.0, 2.0])
        source_vol.add_source(vs)

        flux_std_vol = Radiant.compute_flux(cs_p, geo_p, m_p, source_vol)
        m_p.set_is_first_collision_source(true)
        flux_fcs_vol = Radiant.compute_flux(cs_p, geo_p, m_p, source_vol)
        m_p.set_is_first_collision_source(false)
        @test flux_fcs_vol.get_flux() ≈ flux_std_vol.get_flux()

        # FCS flag is restored after a guarded call
        m_p.set_is_first_collision_source(true)
        _ = Radiant.compute_flux(cs_p, geo_p, m_p, source_vol)
        @test m_p.get_is_first_collision_source() == true
        m_p.set_is_first_collision_source(false)

        # Full FCS surface-source solve runs and, for pure absorber, φ_total ≈ φ_u
        cs_pa, geo_pa, m_pa, _ = setup_1d_problem(1.0, 0.0; Nx=20, Lx=2.0, N=4, L=3)
        source_pa = Source(electron, cs_pa, geo_pa, m_pa)
        source_pa.add_source(ss)

        m_pa.set_is_first_collision_source(true)
        flux_fcs_pa = Radiant.compute_flux(cs_pa, geo_pa, m_pa, source_pa)
        m_pa.set_is_first_collision_source(false)

        ad_pa = Radiant.build_angular_discretization(m_pa, geo_pa)
        φ_u_pa, _ = Radiant._compute_uncollided_surface_flux(cs_pa, geo_pa, m_pa, source_pa, ad_pa)
        @test flux_fcs_pa.get_flux() ≈ φ_u_pa
    end

    @testset "_compute_fcs_precursor structure" begin
        # Use non-zero scattering so that Q_FCS is non-zero and the modified source differs.
        cs, geo, m, electron = setup_1d_problem(0.5, 0.5; Nx=20, Lx=2.0, N=4, L=3)
        ss = Surface_Source()
        ss.set_particle(electron)
        ss.set_intensity(1.0)
        ss.set_energy_group(1)
        ss.set_direction([1.0, 0.0, 0.0])
        ss.set_location("x-")
        source = Source(electron, cs, geo, m)
        source.add_source(ss)
        m.set_is_first_collision_source(true)

        precursor = Radiant._compute_fcs_precursor(cs, geo, m, source)
        @test !isnothing(precursor)
        flux_u, modified_source = precursor
        @test flux_u isa Radiant.Flux_Per_Particle
        @test modified_source isa Source
        @test Radiant.get_tag(flux_u.particle) == Radiant.get_tag(electron)
        @test Radiant.get_tag(modified_source.particle) == Radiant.get_tag(electron)
        # surface sources and point sources are cleared in the modified source
        @test all(x -> x == 0.0, modified_source.surface_sources)
        @test isempty(modified_source.point_sources)
        @test isempty(modified_source.surface_source_objects)
        # first collision source is non-zero when scattering is present
        @test any(modified_source.volume_sources .!= source.volume_sources)
        # normalization factor is preserved
        @test modified_source.normalization_factor == source.normalization_factor
    end

    @testset "FCS 2D/3D uncollided surface flux" begin
        # 2D X- beam at 45 degrees: exponential decay along centerline
        cs_2d, geo_2d, m_2d, electron = setup_2d_problem(1.0, 0.0; Nx=40, Ny=40, Lx=2.0, Ly=2.0, N=8, L=3)
        ad_2d = Radiant.build_angular_discretization(m_2d, geo_2d)
        @test ad_2d.Qdims == 2
        @test ad_2d.Np == div((3+1)*(3+2),2)

        ss_2d = Surface_Source()
        ss_2d.set_particle(electron)
        ss_2d.set_intensity(1.0)
        ss_2d.set_energy_group(1)
        μ = η = sqrt(2)/2
        ss_2d.set_direction([μ, η, 0.0])
        ss_2d.set_location("x-")
        ss_2d.set_boundaries("y", [0.0, 2.0])
        source_2d = Source(electron, cs_2d, geo_2d, m_2d)
        source_2d.add_source(ss_2d)

        φ_u_2d, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_2d, ad_2d)
        φ0_2d = φ_u_2d[1,1,1,:,:,1]
        x = geo_2d.get_voxels_position("x")
        y = geo_2d.get_voxels_position("y")
        ix0 = 5
        iy0 = argmin(abs.(y .- x[ix0]))
        s0 = x[ix0] / μ
        val0 = φ0_2d[ix0, iy0]
        for Δix in [5, 10, 15]
            ix = ix0 + Δix
            iy = argmin(abs.(y .- x[ix]))
            s = x[ix] / μ
            @test φ0_2d[ix, iy] ≈ val0 * exp(-(s - s0)) rtol=0.1
        end

        # 2D boundary filtering: half face gives roughly half total
        ss_full = Surface_Source()
        ss_full.set_particle(electron)
        ss_full.set_intensity(1.0)
        ss_full.set_energy_group(1)
        ss_full.set_direction([1.0, 0.0, 0.0])
        ss_full.set_location("x-")
        ss_full.set_boundaries("y", [0.0, 2.0])
        source_full = Source(electron, cs_2d, geo_2d, m_2d)
        source_full.add_source(ss_full)
        φ_full, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_full, ad_2d)
        φ0_full = φ_full[1,1,1,:,:,1]

        ss_half = Surface_Source()
        ss_half.set_particle(electron)
        ss_half.set_intensity(1.0)
        ss_half.set_energy_group(1)
        ss_half.set_direction([1.0, 0.0, 0.0])
        ss_half.set_location("x-")
        ss_half.set_boundaries("y", [0.0, 1.0])
        source_half = Source(electron, cs_2d, geo_2d, m_2d)
        source_half.add_source(ss_half)
        φ_half, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_half, ad_2d)
        φ0_half = φ_half[1,1,1,:,:,1]
        Δx = geo_2d.get_voxels_width()[1]
        Δy = geo_2d.get_voxels_width()[2]
        total_full = sum(φ0_full[i,j] * Δx[i] * Δy[j] for i in eachindex(Δx), j in eachindex(Δy))
        total_half = sum(φ0_half[i,j] * Δx[i] * Δy[j] for i in eachindex(Δx), j in eachindex(Δy))
        @test total_half ≈ 0.5 * total_full rtol=0.05

        # 2D multi-source superposition (same direction)
        ss_single = Surface_Source()
        ss_single.set_particle(electron)
        ss_single.set_intensity(1.0)
        ss_single.set_energy_group(1)
        ss_single.set_direction([1.0, 0.0, 0.0])
        ss_single.set_location("x-")
        ss_single.set_boundaries("y", [0.0, 2.0])
        source_single = Source(electron, cs_2d, geo_2d, m_2d)
        source_single.add_source(ss_single)
        φ_single, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_single, ad_2d)

        ss_double = Surface_Source()
        ss_double.set_particle(electron)
        ss_double.set_intensity(2.0)
        ss_double.set_energy_group(1)
        ss_double.set_direction([1.0, 0.0, 0.0])
        ss_double.set_location("x-")
        ss_double.set_boundaries("y", [0.0, 2.0])
        source_double = Source(electron, cs_2d, geo_2d, m_2d)
        source_double.add_source(ss_single)
        source_double.add_source(ss_double)
        φ_double, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_double, ad_2d)
        @test φ_double[1,1,1,:,:,1] ≈ 3 * φ_single[1,1,1,:,:,1] rtol=0.05

        # 2D non-X face (Y-) runs and is non-zero
        ss_y = Surface_Source()
        ss_y.set_particle(electron)
        ss_y.set_intensity(1.0)
        ss_y.set_energy_group(1)
        ss_y.set_direction([0.0, 1.0, 0.0])
        ss_y.set_location("y-")
        ss_y.set_boundaries("x", [0.0, 2.0])
        source_y = Source(electron, cs_2d, geo_2d, m_2d)
        source_y.add_source(ss_y)
        φ_y, _ = Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_y, ad_2d)
        @test any(φ_y .!= 0.0)

        # 2D beam direction mismatch on Y- face
        ss_y_bad = Surface_Source()
        ss_y_bad.set_particle(electron)
        ss_y_bad.set_intensity(1.0)
        ss_y_bad.set_energy_group(1)
        ss_y_bad.set_direction([0.0, -1.0, 0.0])
        ss_y_bad.set_location("y-")
        ss_y_bad.set_boundaries("x", [0.0, 2.0])
        source_y_bad = Source(electron, cs_2d, geo_2d, m_2d)
        source_y_bad.add_source(ss_y_bad)
        @test_throws ErrorException Radiant._compute_uncollided_surface_flux(cs_2d, geo_2d, m_2d, source_y_bad, ad_2d)

        # 2D full FCS solve for pure absorber equals uncollided flux
        m_2d.set_is_first_collision_source(true)
        flux_fcs_2d = Radiant.compute_flux(cs_2d, geo_2d, m_2d, source_full)
        m_2d.set_is_first_collision_source(false)
        @test flux_fcs_2d.get_flux() ≈ φ_full

        # 2D FCS fallback for volume source matches standard solve
        cs_v2, geo_v2, m_v2, _ = setup_2d_problem(0.5, 0.0; Nx=20, Ny=20, Lx=1.5, Ly=1.5, N=4, L=2)
        vs_2d = Volume_Source()
        vs_2d.set_particle(electron)
        vs_2d.set_intensity(1.0)
        vs_2d.set_energy_group(1)
        vs_2d.set_boundaries("x", [0.0, 1.5])
        vs_2d.set_boundaries("y", [0.0, 1.5])
        source_v2 = Source(electron, cs_v2, geo_v2, m_v2)
        source_v2.add_source(vs_2d)
        flux_std_v2 = Radiant.compute_flux(cs_v2, geo_v2, m_v2, source_v2)
        m_v2.set_is_first_collision_source(true)
        flux_fcs_v2 = Radiant.compute_flux(cs_v2, geo_v2, m_v2, source_v2)
        m_v2.set_is_first_collision_source(false)
        @test flux_fcs_v2.get_flux() ≈ flux_std_v2.get_flux()

        # 3D X- beam: exponential decay along centerline
        cs_3d, geo_3d, m_3d, electron = setup_3d_problem(1.0, 0.0; Nx=20, Ny=20, Nz=20, Lx=1.5, Ly=1.5, Lz=1.5, N=4, L=2)
        ad_3d = Radiant.build_angular_discretization(m_3d, geo_3d)
        @test ad_3d.Qdims == 3
        @test ad_3d.Np == (2+1)^2

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

        φ_u_3d, _ = Radiant._compute_uncollided_surface_flux(cs_3d, geo_3d, m_3d, source_3d, ad_3d)
        φ0_3d = φ_u_3d[1,1,1,:,:,:]
        x3 = geo_3d.get_voxels_position("x")
        ix3_0 = 3
        jy = div(size(φ0_3d,2),2) + 1
        kz = div(size(φ0_3d,3),2) + 1
        val3_0 = φ0_3d[ix3_0, jy, kz]
        for Δix in [3, 6]
            ix = ix3_0 + Δix
            @test φ0_3d[ix, jy, kz] ≈ val3_0 * exp(-(x3[ix] - x3[ix3_0])) rtol=0.15
        end

        # 3D multi-source superposition (same direction)
        ss_single3 = Surface_Source()
        ss_single3.set_particle(electron)
        ss_single3.set_intensity(1.0)
        ss_single3.set_energy_group(1)
        ss_single3.set_direction([1.0, 0.0, 0.0])
        ss_single3.set_location("x-")
        ss_single3.set_boundaries("y", [0.0, 1.5])
        ss_single3.set_boundaries("z", [0.0, 1.5])
        source_single3 = Source(electron, cs_3d, geo_3d, m_3d)
        source_single3.add_source(ss_single3)
        φ_single3, _ = Radiant._compute_uncollided_surface_flux(cs_3d, geo_3d, m_3d, source_single3, ad_3d)

        ss_double3 = Surface_Source()
        ss_double3.set_particle(electron)
        ss_double3.set_intensity(2.0)
        ss_double3.set_energy_group(1)
        ss_double3.set_direction([1.0, 0.0, 0.0])
        ss_double3.set_location("x-")
        ss_double3.set_boundaries("y", [0.0, 1.5])
        ss_double3.set_boundaries("z", [0.0, 1.5])
        source_double3 = Source(electron, cs_3d, geo_3d, m_3d)
        source_double3.add_source(ss_single3)
        source_double3.add_source(ss_double3)
        φ_double3, _ = Radiant._compute_uncollided_surface_flux(cs_3d, geo_3d, m_3d, source_double3, ad_3d)
        @test φ_double3[1,1,1,:,:,:] ≈ 3 * φ_single3[1,1,1,:,:,:] rtol=0.05

        # 3D non-X faces run and produce non-zero flux
        ss_y3 = Surface_Source()
        ss_y3.set_particle(electron)
        ss_y3.set_intensity(1.0)
        ss_y3.set_energy_group(1)
        ss_y3.set_direction([0.0, 1.0, 0.0])
        ss_y3.set_location("y-")
        ss_y3.set_boundaries("x", [0.0, 1.5])
        ss_y3.set_boundaries("z", [0.0, 1.5])
        source_y3 = Source(electron, cs_3d, geo_3d, m_3d)
        source_y3.add_source(ss_y3)
        φ_y3, _ = Radiant._compute_uncollided_surface_flux(cs_3d, geo_3d, m_3d, source_y3, ad_3d)
        @test any(φ_y3 .!= 0.0)

        ss_z3 = Surface_Source()
        ss_z3.set_particle(electron)
        ss_z3.set_intensity(1.0)
        ss_z3.set_energy_group(1)
        ss_z3.set_direction([0.0, 0.0, 1.0])
        ss_z3.set_location("z-")
        ss_z3.set_boundaries("x", [0.0, 1.5])
        ss_z3.set_boundaries("y", [0.0, 1.5])
        source_z3 = Source(electron, cs_3d, geo_3d, m_3d)
        source_z3.add_source(ss_z3)
        φ_z3, _ = Radiant._compute_uncollided_surface_flux(cs_3d, geo_3d, m_3d, source_z3, ad_3d)
        @test any(φ_z3 .!= 0.0)

        # 3D full FCS solve for pure absorber equals uncollided flux
        m_3d.set_is_first_collision_source(true)
        flux_fcs_3d = Radiant.compute_flux(cs_3d, geo_3d, m_3d, source_3d)
        m_3d.set_is_first_collision_source(false)
        @test flux_fcs_3d.get_flux() ≈ φ_u_3d

        # 3D FCS fallback for volume source matches standard solve
        cs_v3, geo_v3, m_v3, _ = setup_3d_problem(0.5, 0.0; Nx=10, Ny=10, Nz=10, Lx=1.0, Ly=1.0, Lz=1.0, N=4, L=2)
        vs_3d = Volume_Source()
        vs_3d.set_particle(electron)
        vs_3d.set_intensity(1.0)
        vs_3d.set_energy_group(1)
        vs_3d.set_boundaries("x", [0.0, 1.0])
        vs_3d.set_boundaries("y", [0.0, 1.0])
        vs_3d.set_boundaries("z", [0.0, 1.0])
        source_v3 = Source(electron, cs_v3, geo_v3, m_v3)
        source_v3.add_source(vs_3d)
        flux_std_v3 = Radiant.compute_flux(cs_v3, geo_v3, m_v3, source_v3)
        m_v3.set_is_first_collision_source(true)
        flux_fcs_v3 = Radiant.compute_flux(cs_v3, geo_v3, m_v3, source_v3)
        m_v3.set_is_first_collision_source(false)
        @test flux_fcs_v3.get_flux() ≈ flux_std_v3.get_flux()
    end

end
nothing

include("test_ray_moc_kernel.jl")
include("test_ray_sweep_1d.jl")
include("test_ray_sweep_3d.jl")
include("test_fcs_1d_bfp.jl")
include("test_fcs_ray_vs_sn.jl")
include("test_fcs_ray_vs_sn_3d.jl")
include("test_fcs_mixed_spatial_order.jl")
include("test_uncollided_flux_sn.jl")
include("test_fcs_point_source_linear_source.jl")
include("test_fcs_point_source_dE_rescale.jl")
include("test_fcs_multi_particle.jl")
include("test_beam_fcs_unit.jl")
include("test_parallel.jl")
