# Parallel-correctness tests for the SN solver.
#
# These tests verify that the `parallel=true` code paths produce results that are
# numerically identical (bit-for-bit / within float round-off) to an independent serial
# reference. They are most meaningful when run with JULIA_NUM_THREADS > 1, but are designed
# to pass with any thread count (the parallel branches are guarded by `nthreads() > 1`).

using Radiant
using LinearAlgebra
using Test
using JLD2

include("parallel_cases.jl")

# Combined relative/absolute criterion (plan §9): the parallel direction sweep must be
# numerically equivalent to the committed serial reference.
function _compare_serial_parallel(serial, parallel; rel_tol=1e-9, abs_tol=1e-12)
    rel = norm(parallel .- serial) / max(norm(serial), 1e-16)
    abs_ = norm(parallel .- serial)
    return rel < rel_tol || abs_ < abs_tol
end

# Self-contained minimal 1D Cartesian problem (so this file can run standalone, independent
# of helpers defined in runtests.jl).
function _setup_1d_problem_pp(Σa, Σs; Nx=30, Lx=3.0, N=8, L=3)
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

# Independent serial reference for in-group scattering source.  Loop order matches
# the CartesianIndices + is→p order used by the voxel-parallel implementation.
function _ref_scattering_ingroup(𝚽l, Σs, mat, P, pl, Nm, Ns)
    Ql = zeros(P, Nm, Ns[1], Ns[2], Ns[3])
    for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
        ix, iy, iz = Tuple(I)
        for is in 1:Nm, p in 1:P
            Ql[p,is,ix,iy,iz] += Σs[mat[ix,iy,iz], pl[p]+1] * 𝚽l[p,is,ix,iy,iz]
        end
    end
    return Ql
end

# Independent serial reference for out-of-group scattering source.  Loop order matches
# the CartesianIndices + gi→is→p order used by the voxel-parallel implementation.
function _ref_scattering_outgroup(𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, gf, is_elastic)
    Ql = zeros(P, Nm, Ns[1], Ns[2], Ns[3])
    for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
        ix, iy, iz = Tuple(I)
        for gi in 1:Ngi
            if gi != gf || is_elastic
                for is in 1:Nm, p in 1:P
                    Ql[p,is,ix,iy,iz] += Σs[mat[ix,iy,iz], gi, pl[p]+1] * 𝚽l[gi,p,is,ix,iy,iz]
                end
            end
        end
    end
    return Ql
end

# Independent serial reference for particle (secondary) sources.
function _ref_particle_sources(𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, Ngf)
    Ql = zeros(Ngf, P, Nm, Ns[1], Ns[2], Ns[3])
    for gf in 1:Ngf, ix in 1:Ns[1], iy in 1:Ns[2], iz in 1:Ns[3]
        for gi in 1:Ngi, is in 1:Nm, p in 1:P
            Ql[gf,p,is,ix,iy,iz] += Σs[mat[ix,iy,iz], gi, gf, pl[p]+1] * 𝚽l[gi,p,is,ix,iy,iz]
        end
    end
    return Ql
end

@testset "Parallel source terms" begin

    # Small but non-trivial problem dimensions
    Nx, Ny, Nz = 10, 3, 2
    Ns = [Nx, Ny, Nz]
    Nmat = 2
    P = 4
    pl = [0, 1, 2, 3]
    Lmax = maximum(pl)
    Nm = 2
    Ngi = 3
    Ngf = 3

    # Deterministic material map and flux (no Random dependency)
    mat = Array{Int64}(undef, Nx, Ny, Nz)
    for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        mat[ix,iy,iz] = (ix + iy + iz) % Nmat + 1
    end

    @testset "in-group scattering_source" begin
        Σs = zeros(Nmat, Lmax+1)
        for m in 1:Nmat, l in 1:Lmax+1
            Σs[m,l] = 0.1 * m + 0.01 * l
        end
        𝚽l = zeros(P, Nm, Nx, Ny, Nz)
        for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, p in 1:P, is in 1:Nm
            𝚽l[p,is,ix,iy,iz] = sin(ix) + 0.1*iy + 0.01*iz + p + is
        end

        expected = _ref_scattering_ingroup(𝚽l, Σs, mat, P, pl, Nm, Ns)

        Ql_s = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.scattering_source(Ql_s, 𝚽l, Σs, mat, P, pl, Nm, Ns; parallel=false)
        @test Ql_s ≈ expected

        Ql_p = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.scattering_source(Ql_p, 𝚽l, Σs, mat, P, pl, Nm, Ns; parallel=true)
        @test Ql_p ≈ expected
    end

    @testset "out-of-group scattering_source" begin
        Σs = zeros(Nmat, Ngi, Lmax+1)
        for m in 1:Nmat, gi in 1:Ngi, l in 1:Lmax+1
            Σs[m,gi,l] = 0.05 * m + 0.02 * gi + 0.01 * l
        end
        𝚽l = zeros(Ngi, P, Nm, Nx, Ny, Nz)
        for gi in 1:Ngi, ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, p in 1:P, is in 1:Nm
            𝚽l[gi,p,is,ix,iy,iz] = cos(ix) + 0.1*gi + 0.01*iy + p + is
        end
        gf = 2

        for is_elastic in (false, true)
            expected = _ref_scattering_outgroup(𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, gf, is_elastic)

            Ql_s = zeros(P, Nm, Nx, Ny, Nz)
            Radiant.scattering_source(Ql_s, 𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, gf, is_elastic; parallel=false)
            @test Ql_s ≈ expected

            Ql_p = zeros(P, Nm, Nx, Ny, Nz)
            Radiant.scattering_source(Ql_p, 𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, gf, is_elastic; parallel=true)
            @test Ql_p ≈ expected
        end
    end

    @testset "particle_sources" begin
        Σs = zeros(Nmat, Ngi, Ngf, Lmax+1)
        for m in 1:Nmat, gi in 1:Ngi, gf in 1:Ngf, l in 1:Lmax+1
            Σs[m,gi,gf,l] = 0.03 * m + 0.02 * gi + 0.01 * gf + 0.005 * l
        end
        𝚽l = zeros(Ngi, P, Nm, Nx, Ny, Nz)
        for gi in 1:Ngi, ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, p in 1:P, is in 1:Nm
            𝚽l[gi,p,is,ix,iy,iz] = sin(ix*gi) + 0.1*iy + p + is
        end

        expected = _ref_particle_sources(𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, Ngf)

        Ql_s = zeros(Ngf, P, Nm, Nx, Ny, Nz)
        Radiant.particle_sources(Ql_s, 𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, Ngf; parallel=false)
        @test Ql_s ≈ expected

        Ql_p = zeros(Ngf, P, Nm, Nx, Ny, Nz)
        Radiant.particle_sources(Ql_p, 𝚽l, Σs, mat, P, pl, Nm, Ns, Ngi, Ngf; parallel=true)
        @test Ql_p ≈ expected
    end

    @testset "fokker_planck_source" begin
        T = [0.7, 1.3]                       # momentum transfer per material
        ℳ = zeros(P, P)
        for n in 1:P, m in 1:P
            ℳ[n,m] = 0.2 * n - 0.1 * m + 0.05
        end
        𝚽l = zeros(P, Nm, Nx, Ny, Nz)
        for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, is in 1:Nm, m in 1:P
            𝚽l[m,is,ix,iy,iz] = cos(ix) + 0.3*is + 0.1*iy + m
        end
        mat3 = mat  # already Array{Int64,3}

        # Independent serial reference — loop order matches the CartesianIndices +
        # is→m→n order used by the voxel-parallel implementation.
        expected = zeros(P, Nm, Nx, Ny, Nz)
        for I in CartesianIndices((Nx, Ny, Nz))
            ix, iy, iz = Tuple(I)
            for is in 1:Nm, m in 1:P, n in 1:P
                expected[n,is,ix,iy,iz] += T[mat3[ix,iy,iz]] * ℳ[n,m] * 𝚽l[m,is,ix,iy,iz]
            end
        end

        Ql_s = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.fokker_planck_source(P, Nm, T, 𝚽l, Ql_s, Ns, mat3, ℳ; parallel=false)
        @test Ql_s ≈ expected

        Ql_p = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.fokker_planck_source(P, Nm, T, 𝚽l, Ql_p, Ns, mat3, ℳ; parallel=true)
        @test Ql_p ≈ expected
    end

end

# Build a 2-group BTE Cross_Sections by hand (the custom path is hardwired to Ng=1).
# Returns (cs, geo, m, electron) where cs carries inter-group scattering Σs[gi->gf].
function setup_multigroup_fcs_problem(Σs_mat; Nx=20, Lx=2.0, N=4, L=3)
    water = Material("water")
    water.set_density(1.0)
    water.add_element("H", 0.1119)
    water.add_element("O", 0.8881)

    electron = Electron()

    cs = Cross_Sections()
    cs.set_materials(water)
    cs.set_particles([electron])
    cs.set_source("custom")
    cs.set_absorption([0.5])
    cs.set_scattering([0.5])
    cs.set_legendre_order(L)
    cs.build()

    # Overwrite the single-group multigroup data with a 2-group scattering matrix.
    Ng = 2
    Nmat = cs.get_number_of_materials()
    mcs_array = Array{Radiant.Multigroup_Cross_Sections}(undef, 1, Nmat)
    for n in 1:Nmat
        mcs = Radiant.Multigroup_Cross_Sections(Ng)
        # Σsl shape (Ngi, Ngf, L+1); here only the isotropic moment is populated.
        Σsl = zeros(Ng, Ng, L+1)
        for gi in 1:Ng, gf in 1:Ng
            Σsl[gi,gf,1] = Σs_mat[gi,gf]
        end
        mcs.set_scattering(Σsl)
        mcs.set_total([1.0, 1.0])
        mcs.set_absorption([0.5, 0.5])
        mcs_array[1,n] = mcs
    end
    cs.set_multigroup_cross_sections(mcs_array)
    cs.set_number_of_groups([Ng])

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

@testset "FCS-1 multigroup first collision source" begin
    # 2-group scattering with non-zero inter-group coupling (the path uncovered by the
    # existing Ng=1 tests). Σs_mat[gi,gf] = scattering from group gi into group gf.
    Σs_mat = [0.30 0.20;
              0.10 0.40]
    cs, geo, m, electron = setup_multigroup_fcs_problem(Σs_mat; Nx=20, Lx=2.0, N=4, L=3)

    ad = Radiant.build_angular_discretization(m, geo)
    Ng = cs.get_number_of_groups(electron)
    @test Ng == 2

    Ns = geo.get_number_of_voxels()
    _, _, Nm = m.get_schemes(geo, m.get_is_full_coupling())
    Np = ad.Np
    Nm5 = Nm[5]

    # Synthetic, deterministic uncollided flux φ_u of shape (Ng, Np, Nm5, Ns...).
    φ_u = zeros(Ng, Np, Nm5, Ns[1], Ns[2], Ns[3])
    for ig in 1:Ng, p in 1:Np, is in 1:Nm5, ix in 1:Ns[1]
        φ_u[ig,p,is,ix,1,1] = sin(0.3*ix + ig) + 0.1*p + 0.05*is + ig
    end

    # Parallelized first collision source (the function under test).
    Q_FCS = Radiant._compute_first_collision_source(
        cs, geo, m, φ_u, ad, Array{Float64}(undef), Array{Float64}(undef,0,0))

    # Independent serial reference: Q_FCS[ig] = Σ_gi Σs[mat,gi,ig,l(p)] * φ_u[gi].
    mat = geo.get_material_per_voxel()
    pl = ad.pl
    Ls = maximum(pl)
    Σs_full = cs.get_scattering(electron, electron, Ls)   # (Nmat, Ngi, Ngf, Ls+1)
    Q_ref = zeros(size(φ_u))
    for ig in 1:Ng
        for gi in 1:Ng
            for ix in 1:Ns[1], iy in 1:Ns[2], iz in 1:Ns[3], p in 1:Np, is in 1:Nm5
                Q_ref[ig,p,is,ix,iy,iz] += Σs_full[mat[ix,iy,iz],gi,ig,pl[p]+1] * φ_u[gi,p,is,ix,iy,iz]
            end
        end
    end

    @test Q_FCS ≈ Q_ref
    # Inter-group coupling must actually contribute (guards against a trivially-passing test).
    @test any(Q_FCS .!= 0.0)
end

@testset "Parallel post-processing" begin
    # Build and solve a small 1D problem, then compare the serial and parallel code paths
    # of the post-processing routines (flux extraction, energy/charge deposition) on the
    # same converged flux object.
    cs, geo, m, electron = _setup_1d_problem_pp(0.5, 0.5; Nx=30, Lx=3.0, N=8, L=3)

    ss = Surface_Source()
    ss.set_particle(electron)
    ss.set_intensity(1.0)
    ss.set_energy_group(1)
    ss.set_direction([1.0, 0.0, 0.0])
    ss.set_location("x-")

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
    flux_obj = cu.flux

    @testset "flux extraction" begin
        F_s = Radiant.flux(cs, geo, flux_obj, electron; parallel=false)
        F_p = Radiant.flux(cs, geo, flux_obj, electron; parallel=true)
        @test F_p ≈ F_s
    end

    @testset "energy_deposition" begin
        D_s = Radiant.energy_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=false)
        D_p = Radiant.energy_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=true)
        @test D_p ≈ D_s
    end

    @testset "charge_deposition" begin
        C_s = Radiant.charge_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=false)
        C_p = Radiant.charge_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=true)
        @test C_p ≈ C_s
    end
end

@testset "P0 direction-parallel sweep vs serial reference" begin
    # Recompute each standard-SN case (under whatever thread count this suite runs with) and
    # compare against the committed single-threaded serial reference. Under multiple threads
    # this exercises the direction-parallel sweep in sn_inner_pass!.
    ref_dir = joinpath(@__DIR__, "data")

    @testset "1D" begin
        s = load(joinpath(ref_dir, "sn_1d_serial_reference.jld2"))["flux"]
        @test _compare_serial_parallel(s, run_sn_1d_case())
    end
    @testset "2D" begin
        s = load(joinpath(ref_dir, "sn_2d_serial_reference.jld2"))["flux"]
        @test _compare_serial_parallel(s, run_sn_2d_case())
    end
    @testset "3D" begin
        s = load(joinpath(ref_dir, "sn_3d_serial_reference.jld2"))["flux"]
        @test _compare_serial_parallel(s, run_sn_3d_case())
    end
end

@testset "BFP direction-parallel sweep vs serial reference" begin
    # BFP cases are CSD + Fokker-Planck, exercising the per-direction energy-flux write
    # 𝚽E12_temp[n,...] inside the parallel sweep that the BTE cases do not. Compared against
    # the committed single-threaded serial reference.
    ref_dir = joinpath(@__DIR__, "data")

    @testset "1D" begin
        s = load(joinpath(ref_dir, "bfp_1d_serial_reference.jld2"))["flux"]
        @test _compare_serial_parallel(s, run_bfp_1d_case())
    end
    @testset "2D" begin
        s = load(joinpath(ref_dir, "bfp_2d_serial_reference.jld2"))["flux"]
        @test _compare_serial_parallel(s, run_bfp_2d_case())
    end
    @testset "3D" begin
        s = load(joinpath(ref_dir, "bfp_3d_serial_reference.jld2"))["flux"]
        @test _compare_serial_parallel(s, run_bfp_3d_case())
    end
end
