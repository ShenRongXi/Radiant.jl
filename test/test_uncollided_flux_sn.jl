# Tests for the point-source uncollided-flux infrastructure (SN solver).
#
# This file is included from runtests.jl, but is also runnable standalone:
#   julia --project=. test/test_uncollided_flux_sn.jl
if !isdefined(Main, :Radiant)
    using Radiant
end
using LinearAlgebra
using Test

@testset "Uncollided flux SN - Task 0 (DDA + projection core)" begin

    # --- Lagrange -> Legendre projection -------------------------------------
    @testset "projection: constant function" begin
        # All 8 vertices equal => only the (1,1,1) moment is non-zero and equals c.
        c = 3.0
        vals = fill(c, 8)
        m = Radiant._project_lagrange_to_legendre(vals, 2, 2, 2)
        @test size(m) == (2, 2, 2)
        @test m[1, 1, 1] ≈ c
        for ix in 1:2, iy in 1:2, iz in 1:2
            (ix, iy, iz) == (1, 1, 1) && continue
            @test isapprox(m[ix, iy, iz], 0.0; atol=1e-12)
        end
    end

    @testset "projection: linear-in-xi function (normalized basis)" begin
        # f(ξ,η,ζ) = ξ : vertex value is +1 when a==1 (ξ=+1), -1 when a==0 (ξ=-1).
        vals = zeros(8)
        for a in 0:1, b in 0:1, c in 0:1
            idx = 1 + a + 2b + 4c
            vals[idx] = (a == 1) ? 1.0 : -1.0
        end
        m = Radiant._project_lagrange_to_legendre(vals, 2, 2, 2)
        # Normalized first moment of f(ξ)=ξ is 1/√3.
        @test m[2, 1, 1] ≈ 1 / sqrt(3)
        @test isapprox(m[1, 1, 1], 0.0; atol=1e-12)
        # Unnormalized coefficient s/8 must equal 1 (distinguishes the two conventions).
        @test m[2, 1, 1] * sqrt(3) ≈ 1.0
        # All other moments vanish.
        for ix in 1:2, iy in 1:2, iz in 1:2
            (ix, iy, iz) in ((1, 1, 1), (2, 1, 1)) && continue
            @test isapprox(m[ix, iy, iz], 0.0; atol=1e-12)
        end
    end

    @testset "projection: linear in eta and zeta" begin
        vals_y = zeros(8); vals_z = zeros(8)
        for a in 0:1, b in 0:1, c in 0:1
            idx = 1 + a + 2b + 4c
            vals_y[idx] = (b == 1) ? 1.0 : -1.0
            vals_z[idx] = (c == 1) ? 1.0 : -1.0
        end
        my = Radiant._project_lagrange_to_legendre(vals_y, 2, 2, 2)
        mz = Radiant._project_lagrange_to_legendre(vals_z, 2, 2, 2)
        @test my[1, 2, 1] ≈ 1 / sqrt(3)
        @test mz[1, 1, 2] ≈ 1 / sqrt(3)
    end

    @testset "projection: order 1 reduces to 8-vertex average" begin
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        m = Radiant._project_lagrange_to_legendre(vals, 1, 1, 1)
        @test size(m) == (1, 1, 1)
        @test m[1, 1, 1] ≈ sum(vals) / 8
    end

    # --- moment_index --------------------------------------------------------
    @testset "moment_index: fully coupled (BTE, energy order 1)" begin
        𝒪 = [2, 2, 2, 1]
        @test Radiant.moment_index(1, 1, 1, 1, 𝒪, true) == 1
        @test Radiant.moment_index(2, 1, 1, 1, 𝒪, true) == 2
        @test Radiant.moment_index(1, 2, 1, 1, 𝒪, true) == 3
        @test Radiant.moment_index(1, 1, 2, 1, 𝒪, true) == 5
    end

    @testset "moment_index: restricted (one high-order axis)" begin
        𝒪 = [2, 2, 2, 1]
        @test Radiant.moment_index(1, 1, 1, 1, 𝒪, false) == 1
        @test Radiant.moment_index(2, 1, 1, 1, 𝒪, false) == 2
        @test Radiant.moment_index(1, 2, 1, 1, 𝒪, false) == 3
        @test Radiant.moment_index(1, 1, 2, 1, 𝒪, false) == 4
    end

    # --- find_cell -----------------------------------------------------------
    @testset "find_cell: interior points" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        @test Radiant.find_cell(xe, ye, ze, [0.5, 0.5, 0.5]) == CartesianIndex(1, 1, 1)
        @test Radiant.find_cell(xe, ye, ze, [2.5, 0.5, 0.5]) == CartesianIndex(3, 1, 1)
        @test Radiant.find_cell(xe, ye, ze, [1.5, 2.5, 0.5]) == CartesianIndex(2, 3, 1)
    end

    @testset "find_cell: boundary point snaps into the grid" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        # Point exactly on lower x boundary snaps to first cell.
        @test Radiant.find_cell(xe, ye, ze, [0.0, 0.5, 0.5]) == CartesianIndex(1, 1, 1)
        # Point exactly on upper x boundary snaps to last cell.
        @test Radiant.find_cell(xe, ye, ze, [3.0, 0.5, 0.5]) == CartesianIndex(3, 1, 1)
    end

    # --- _ray_entry_to_box ---------------------------------------------------
    @testset "_ray_entry_to_box: face entry along x" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        entry = Radiant._ray_entry_to_box([-1.0, 1.5, 1.5], [2.0, 1.5, 1.5], xe, ye, ze)
        @test entry ≈ [0.0, 1.5, 1.5]
    end

    @testset "_ray_entry_to_box: oblique entry" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        entry = Radiant._ray_entry_to_box([-1.0, 0.5, 1.5], [1.0, 1.5, 1.5], xe, ye, ze)
        @test entry ≈ [0.0, 1.0, 1.5]
    end

    @testset "_ray_entry_to_box: negative direction from high side" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        entry = Radiant._ray_entry_to_box([4.0, 1.5, 1.5], [0.5, 1.5, 1.5], xe, ye, ze)
        @test entry ≈ [3.0, 1.5, 1.5]
    end

    @testset "_ray_entry_to_box: ray parallel to a slab and outside it misses" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        # src 在 x 方向盒内、y 方向盒外，且射线沿 x 平行于 y-slab：无交。
        @test Radiant._ray_entry_to_box([1.0, -1.0, 1.5], [2.0, -1.0, 1.5], xe, ye, ze) === nothing
    end

    @testset "_point_in_box: inside / outside / boundary" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        @test Radiant._point_in_box([1.5, 1.5, 1.5], xe, ye, ze) == true
        @test Radiant._point_in_box([0.0, 1.5, 1.5], xe, ye, ze) == true   # 闭盒
        @test Radiant._point_in_box([-0.1, 1.5, 1.5], xe, ye, ze) == false
        @test Radiant._point_in_box([3.1, 1.5, 1.5], xe, ye, ze) == false
    end

    # --- dda_optical_depth ---------------------------------------------------
    @testset "dda_optical_depth: straight ray along x" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        τ = Radiant.dda_optical_depth([0.1, 0.5, 0.5], [2.9, 0.5, 0.5], xe, ye, ze, mat, σ)
        @test τ ≈ 2.0 * 2.8
    end

    @testset "dda_optical_depth: reverse (negative) direction" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        τ = Radiant.dda_optical_depth([2.9, 0.5, 0.5], [0.1, 0.5, 0.5], xe, ye, ze, mat, σ)
        @test τ ≈ 2.0 * 2.8
    end

    @testset "dda_optical_depth: heterogeneous materials along x" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        mat[2, :, :] .= 2   # middle x-slab is material 2
        σ = [1.0, 10.0]
        # path: cell1 (0.1->1.0, len .9, σ1) + cell2 (1->2, len 1, σ10) + cell3 (2->2.9, .9, σ1)
        τ = Radiant.dda_optical_depth([0.1, 0.5, 0.5], [2.9, 0.5, 0.5], xe, ye, ze, mat, σ)
        @test τ ≈ 1.0 * 0.9 + 10.0 * 1.0 + 1.0 * 0.9
    end

    # --- degenerate corner/edge tracing (zero-step fallback) -----------------
    @testset "dda: main diagonal through cell corners does not crash" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        # Ray passes exactly through interior corners (1,1,1),(2,2,2).
        τ = Radiant.dda_optical_depth([0.1, 0.1, 0.1], [2.9, 2.9, 2.9], xe, ye, ze, mat, σ)
        @test τ ≈ 2.0 * norm([2.8, 2.8, 2.8])
    end

    @testset "dda: diagonal through cell edges (xy-plane) does not crash" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        τ = Radiant.dda_optical_depth([0.1, 0.1, 0.5], [2.9, 2.9, 0.5], xe, ye, ze, mat, σ)
        @test τ ≈ 2.0 * norm([2.8, 2.8, 0.0])
    end

    @testset "dda: segment lengths sum to the source-target distance" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        src = [0.1, 0.3, 0.7]; tgt = [2.9, 1.8, 2.2]
        cells, segments = Radiant.dda_trace_cells(src, tgt, xe, ye, ze, mat)
        @test length(cells) == length(segments)
        @test sum(segments) ≈ norm(tgt .- src)
    end

    @testset "dda_trace_cells: source equal to target throws" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        @test_throws AssertionError Radiant.dda_trace_cells([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], xe, ye, ze, mat)
    end

    @testset "dda_optical_depth: external source, vacuum outside" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        # 源在 x<0 真空区；域内路径为 0 -> 2.9，τ 只计域内段。
        τ = Radiant.dda_optical_depth([-0.5, 0.5, 0.5], [2.9, 0.5, 0.5], xe, ye, ze, mat, σ)
        @test τ ≈ 2.0 * 2.9
    end

    @testset "dda_optical_depth: external source, oblique ray" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        src = [-1.0, 0.5, 1.5]; tgt = [1.0, 1.5, 1.5]
        # 进入点为 (0, 1.0, 1.5)，域内路径长 = norm(tgt - entry)。
        s_inside = norm(tgt .- [0.0, 1.0, 1.5])
        τ = Radiant.dda_optical_depth(src, tgt, xe, ye, ze, mat, σ)
        @test τ ≈ 2.0 * s_inside
    end

    @testset "dda_optical_depth: internal source unchanged" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        τ = Radiant.dda_optical_depth([0.1, 0.5, 0.5], [2.9, 0.5, 0.5], xe, ye, ze, mat, σ)
        @test τ ≈ 2.0 * 2.8
    end

    @testset "dda_trace_cells: ray entering exactly at target returns empty ray" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        # 源 (-1,1.5,1.5) 到顶点 (0,0,0)：t_enter = 1，恰在目标点进入网格。
        cells, segments = Radiant.dda_trace_cells([-1.0, 1.5, 1.5], [0.0, 0.0, 0.0],
                                                  xe, ye, ze, mat)
        @test isempty(cells)
        @test isempty(segments)
    end

    @testset "dda_optical_depth: zero-length in-grid path gives exactly zero tau" begin
        xe = collect(0.0:1.0:3.0); ye = copy(xe); ze = copy(xe)
        mat = ones(Int64, 3, 3, 3)
        σ = [2.0]
        # 角点顶点与面上内部顶点两种 entry == tgt 情形，τ 必须精确为 0
        #（不能是 _snap_to_interior 产生的 ~1e-9 伪段）。
        @test Radiant.dda_optical_depth([-1.0, 1.5, 1.5], [0.0, 0.0, 0.0], xe, ye, ze, mat, σ) == 0.0
        @test Radiant.dda_optical_depth([-1.0, 1.5, 1.5], [0.0, 1.0, 2.0], xe, ye, ze, mat, σ) == 0.0
    end

end

@testset "Uncollided flux SN - Task 1 (AngularDistribution & Point_Source)" begin

    # --- AngularDistribution -------------------------------------------------
    @testset "AngularDistribution: default is isotropic (0.5)" begin
        h = Radiant.AngularDistribution()
        @test h.edges == [-1.0, 1.0]
        @test h.values == [0.5]
        @test Radiant.evaluate(h, -1.0) ≈ 0.5
        @test Radiant.evaluate(h, 0.0) ≈ 0.5
        @test Radiant.evaluate(h, 1.0) ≈ 0.5     # upper endpoint included
        # ∫ f dμ = 1 for the default distribution
        @test sum(h.values .* diff(h.edges)) ≈ 1.0
    end

    @testset "AngularDistribution: set_bins! and evaluate by bin" begin
        h = Radiant.AngularDistribution()
        Radiant.set_bins!(h, [-1.0, 0.0, 1.0], [0.25, 0.75])
        @test Radiant.evaluate(h, -0.5) ≈ 0.25
        @test Radiant.evaluate(h, 0.5) ≈ 0.75
        @test_throws AssertionError Radiant.set_bins!(h, [-1.0, 0.0, 1.0], [0.25])         # length mismatch
        @test_throws AssertionError Radiant.set_bins!(h, [-1.0, 1.0, 0.5], [0.5, 0.5])     # non-monotonic
        @test_throws AssertionError Radiant.set_bins!(h, [-0.5, 1.0], [1.0])               # endpoints not ±1
    end

    @testset "AngularDistribution: normalize!" begin
        h = Radiant.AngularDistribution()
        Radiant.set_bins!(h, [-1.0, 1.0], [2.0])
        Radiant.normalize!(h)
        @test sum(h.values .* diff(h.edges)) ≈ 1.0
    end

    # --- Point_Source --------------------------------------------------------
    @testset "Point_Source: defaults" begin
        ps = Radiant.Point_Source()
        @test ps.position == zeros(3)
        @test ps.intensity == 1.0
        @test ps.reference_direction == [1.0, 0.0, 0.0]
        @test ps.is_build == false
    end

    @testset "Point_Source: setters/getters via method notation" begin
        electron = Electron()
        ps = Radiant.Point_Source()
        ps.set_particle(electron)
        ps.set_intensity(5.0)
        ps.set_position([0.3, 0.4, 0.5])
        @test ps.get_intensity() ≈ 5.0
        @test ps.get_normalization_factor() ≈ 5.0
        @test ps.position == [0.3, 0.4, 0.5]
    end

    @testset "Point_Source: reference direction is normalized" begin
        ps = Radiant.Point_Source()
        ps.set_reference_direction([3.0, 0.0, 0.0])
        @test ps.reference_direction ≈ [1.0, 0.0, 0.0]
        ps.set_reference_direction([0.0, 3.0, 4.0])
        @test ps.reference_direction ≈ [0.0, 0.6, 0.8]
    end

    @testset "Point_Source: energy group / spectrum mutual exclusion" begin
        ps = Radiant.Point_Source()
        ps.set_energy_group(2)
        @test ps.energy_group == 2
        @test isempty(ps.energy_spectrum)
        ps.set_energy_spectrum([0.2, 0.8])
        @test ps.energy_spectrum == [0.2, 0.8]
        @test ismissing(ps.energy_group)
        ps.set_energy_group(1)
        @test ps.energy_group == 1
        @test isempty(ps.energy_spectrum)
    end

    @testset "Point_Source: build from energy group makes one-hot spectrum" begin
        ps = Radiant.Point_Source()
        ps.set_energy_group(2)
        ps.build(3)
        @test ps.energy_spectrum == [0.0, 1.0, 0.0]
        @test ps.is_build == true
    end

    @testset "Point_Source: build validates and normalizes spectrum" begin
        ps = Radiant.Point_Source()
        ps.set_energy_spectrum([0.3, 0.3])      # sums to 0.6 -> normalized
        ps.build(2)
        @test sum(ps.energy_spectrum) ≈ 1.0
        @test ps.energy_spectrum ≈ [0.5, 0.5]

        ps_bad = Radiant.Point_Source()
        ps_bad.set_energy_spectrum([0.5, 0.5])
        @test_throws AssertionError ps_bad.build(3)   # wrong length

        ps_none = Radiant.Point_Source()
        @test_throws ErrorException ps_none.build(2)  # neither group nor spectrum

        ps_g = Radiant.Point_Source()
        ps_g.set_energy_group(5)
        @test_throws AssertionError ps_g.build(2)     # group out of range
    end

end

# Minimal 3D Cartesian setup with integer-positioned vertices (edges 0,1,2,3).
function _setup_point_3d(; Σa=1.0, Σs=0.0, Nx=3, Ny=3, Nz=3, L=2)
    water = Material("water"); water.set_density(1.0)
    water.add_element("H", 0.1119); water.add_element("O", 0.8881)
    electron = Electron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron]); cs.set_source("custom")
    cs.set_absorption([Σa]); cs.set_scattering([Σs]); cs.set_legendre_order(L); cs.build()
    geo = Geometry()
    geo.set_type("cartesian"); geo.set_dimension(3)
    for b in ("x-","x+","y-","y+","z-","z+"); geo.set_boundary_conditions(b, "void"); end
    geo.set_material_per_region([water])
    for ax in ("x","y","z"); geo.set_number_of_regions(ax, 1); end
    geo.set_voxels_per_region("x", [Nx]); geo.set_voxels_per_region("y", [Ny]); geo.set_voxels_per_region("z", [Nz])
    geo.set_region_boundaries("x", [0.0, Float64(Nx)])
    geo.set_region_boundaries("y", [0.0, Float64(Ny)])
    geo.set_region_boundaries("z", [0.0, Float64(Nz)])
    geo.build(cs)
    m = SN(); m.set_particle(electron); m.set_solver_type("BTE")
    m.set_quadrature("gauss-legendre-chebychev", 4, 3); m.set_legendre_order(L)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x","DD",1); m.set_scheme("y","DD",1); m.set_scheme("z","DD",1)
    return cs, geo, m, electron
end

@testset "Uncollided flux SN - Task 3 (BTE path A: basis vector & tau cache)" begin

    @testset "_sn_basis_vector matches Dn/w at quadrature directions (standard)" begin
        cs, geo, m, electron = _setup_point_3d()
        m.set_angular_boltzmann("standard")
        ad = Radiant.build_angular_discretization(m, geo)
        Ω = ad.Ω; w = ad.w; Dn = ad.Dn; Nd = ad.Nd
        for n in 1:Nd
            Ωn = [Ω[1][n], Ω[2][n], Ω[3][n]]
            Y = Radiant._sn_basis_vector(Ωn, ad)
            @test length(Y) == ad.Np
            @test Y ≈ Dn[:, n] ./ w[n]
        end
    end

    @testset "_sn_basis_vector matches Dn/w at quadrature directions (galerkin-d)" begin
        cs, geo, m, electron = _setup_point_3d()
        m.set_angular_boltzmann("galerkin-d")
        ad = Radiant.build_angular_discretization(m, geo)
        Ω = ad.Ω; w = ad.w; Dn = ad.Dn; Nd = ad.Nd
        for n in 1:Nd
            Ωn = [Ω[1][n], Ω[2][n], Ω[3][n]]
            Y = Radiant._sn_basis_vector(Ωn, ad)
            @test length(Y) == ad.Np
            @test Y ≈ Dn[:, n] ./ w[n]
        end
    end

    @testset "BTE point flux: shape and analytic vertex average (O=1)" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)   # Ng=1, spatial order 1
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        ad = Radiant.build_angular_discretization(m, geo)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        φ_u = Radiant._compute_uncollided_point_flux_bte(cache, cs, geo, m, src, ad)

        Ns = geo.get_number_of_voxels()
        Np = ad.Np
        @test size(φ_u) == (1, Np, 1, Ns[1], Ns[2], Ns[3])

        xe = geo.get_voxels_boundaries("x")
        ye = geo.get_voxels_boundaries("y")
        ze = geo.get_voxels_boundaries("z")
        mat = geo.get_material_per_voxel()
        σ = [1.0]; q = 2.0
        # Independently reconstruct each cell's O=1 moment as the 8-vertex average of the
        # documented kernel Y·(1/r²)·q·(1/2π)·f(μ)·exp(-τ), with f(μ)=0.5 (isotropic).
        for cell in (CartesianIndex(3, 3, 3), CartesianIndex(2, 2, 2), CartesianIndex(1, 1, 1))
            ic, jc, kc = cell[1], cell[2], cell[3]
            acc = zeros(Np)
            for a in 0:1, b in 0:1, c in 0:1
                v = [xe[ic+a], ye[jc+b], ze[kc+c]]
                dvec = v .- ps.position
                r = norm(dvec); Ω = dvec ./ r
                τ = Radiant.dda_optical_depth(ps.position, v, xe, ye, ze, mat, σ)
                Y = Radiant._sn_basis_vector(Ω, ad)
                acc .+= Y .* (1 / r^2) .* q .* (1 / (2π)) .* 0.5 .* exp(-τ)
            end
            acc ./= 8
            for p in 1:Np
                @test φ_u[1, p, 1, ic, jc, kc] ≈ acc[p]
            end
        end
    end

    @testset "BTE point flux: high spatial order (O=2) moment assembly" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)
        m.set_scheme("x", "DD", 2); m.set_scheme("y", "DD", 2); m.set_scheme("z", "DD", 2)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.5); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        ad = Radiant.build_angular_discretization(m, geo)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        φ_u = Radiant._compute_uncollided_point_flux_bte(cache, cs, geo, m, src, ad)

        _, 𝒪, Nm = m.get_schemes(geo, m.get_is_full_coupling())
        isFC = m.get_is_full_coupling()
        @test 𝒪[1] == 2 && 𝒪[2] == 2 && 𝒪[3] == 2
        Np = ad.Np
        xe = geo.get_voxels_boundaries("x")
        ye = geo.get_voxels_boundaries("y")
        ze = geo.get_voxels_boundaries("z")
        mat = geo.get_material_per_voxel()
        σ = [1.0]; q = 1.5

        # Independently project the 8 vertex kernel values onto the normalized Legendre
        # moments and check every (p, ix,iy,iz) slot lands at its moment_index in φ_u.
        cell = CartesianIndex(2, 2, 3)
        ic, jc, kc = cell[1], cell[2], cell[3]
        vert = [zeros(Np) for _ in 1:8]
        for a in 0:1, b in 0:1, c in 0:1
            idx = 1 + a + 2b + 4c
            v = [xe[ic+a], ye[jc+b], ze[kc+c]]
            dvec = v .- ps.position
            r = norm(dvec); Ω = dvec ./ r
            τ = Radiant.dda_optical_depth(ps.position, v, xe, ye, ze, mat, σ)
            Y = Radiant._sn_basis_vector(Ω, ad)
            vert[idx] .= Y .* (1 / r^2) .* q .* (1 / (2π)) .* 0.5 .* exp(-τ)
        end
        any_high_nonzero = false
        for p in 1:Np
            vals = [vert[idx][p] for idx in 1:8]
            moments = Radiant._project_lagrange_to_legendre(vals, 𝒪[1], 𝒪[2], 𝒪[3])
            for ix in 1:𝒪[1], iy in 1:𝒪[2], iz in 1:𝒪[3]
                if !isFC && ((ix > 1) + (iy > 1) + (iz > 1) > 1)
                    continue
                end
                is = Radiant.moment_index(ix, iy, iz, 1, 𝒪, isFC)
                @test φ_u[1, p, is, ic, jc, kc] ≈ moments[ix, iy, iz]
                if (ix, iy, iz) != (1, 1, 1) && abs(moments[ix, iy, iz]) > 1e-12
                    any_high_nonzero = true
                end
            end
        end
        @test any_high_nonzero   # 1/r² variation must populate high-order spatial moments
    end

end

@testset "Uncollided flux SN - Task 2 (Source / Fixed_Sources integration)" begin

    @testset "Source: add_source(::Point_Source) and has_point_sources" begin
        cs, geo, m, electron = _setup_point_3d()
        src = Source(electron, cs, geo, m)
        @test src.has_point_sources() == false
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0); ps.set_position([0.3, 0.4, 0.5])
        src.add_source(ps)
        @test src.has_point_sources() == true
        @test length(src.point_sources) == 1
        # add_source(::Point_Source) must not touch normalization_factor
        @test src.get_normalization_factor() == 0
    end

    @testset "Source: Base.:+ merges point sources only" begin
        cs, geo, m, electron = _setup_point_3d()
        s1 = Source(electron, cs, geo, m)
        s2 = Source(electron, cs, geo, m)
        ps1 = Radiant.Point_Source(); ps1.set_particle(electron); ps1.set_position([0.3,0.4,0.5])
        ps2 = Radiant.Point_Source(); ps2.set_particle(electron); ps2.set_position([1.3,1.4,1.5])
        s1.add_source(ps1); s2.add_source(ps2)
        s3 = s1 + s2
        @test length(s3.point_sources) == 2
    end

    @testset "Fixed_Sources.build: valid interior point source" begin
        cs, geo, m, electron = _setup_point_3d()
        solvers = Solvers(); solvers.add_solver(m)
        fs = Fixed_Sources(cs, geo, solvers)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_energy_group(1); ps.set_position([0.3, 0.4, 0.5])
        fs.add_source(ps)
        fs.build()
        built = fs.get_source(electron)
        @test built.has_point_sources() == true
        @test built.point_sources[1].is_build == true
    end

    @testset "Fixed_Sources.build: external point source builds (void BCs)" begin
        cs, geo, m, electron = _setup_point_3d()
        solvers = Solvers(); solvers.add_solver(m)
        fs = Fixed_Sources(cs, geo, solvers)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_energy_group(1); ps.set_position([-0.1, 0.5, 0.5])
        fs.add_source(ps)
        fs.build()
        built = fs.get_source(electron)
        @test built.has_point_sources() == true
        @test built.point_sources[1].position == [-0.1, 0.5, 0.5]
    end

    @testset "Fixed_Sources.build: external point source requires void BCs" begin
        cs, geo, m, electron = _setup_point_3d()
        # 重新搭一个 x+ 为 reflective 的几何（其余面 void）
        geo_r = Geometry()
        geo_r.set_type("cartesian"); geo_r.set_dimension(3)
        for b in ("x-","y-","y+","z-","z+"); geo_r.set_boundary_conditions(b, "void"); end
        geo_r.set_boundary_conditions("x+", "reflective")
        geo_r.set_material_per_region(geo.material_per_region)
        for ax in ("x","y","z"); geo_r.set_number_of_regions(ax, 1); end
        geo_r.set_voxels_per_region("x", [3]); geo_r.set_voxels_per_region("y", [3]); geo_r.set_voxels_per_region("z", [3])
        geo_r.set_region_boundaries("x", [0.0, 3.0]); geo_r.set_region_boundaries("y", [0.0, 3.0]); geo_r.set_region_boundaries("z", [0.0, 3.0])
        geo_r.build(cs)
        solvers = Solvers(); solvers.add_solver(m)
        fs = Fixed_Sources(cs, geo_r, solvers)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_energy_group(1); ps.set_position([-0.1, 0.5, 0.5])
        fs.add_source(ps)
        @test_throws AssertionError fs.build()
    end

    @testset "Fixed_Sources.build: point source on a grid vertex throws" begin
        cs, geo, m, electron = _setup_point_3d()
        solvers = Solvers(); solvers.add_solver(m)
        fs = Fixed_Sources(cs, geo, solvers)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_energy_group(1); ps.set_position([1.0, 1.0, 1.0])
        fs.add_source(ps)
        @test_throws AssertionError fs.build()
    end

    @testset "Fixed_Sources.build: point source too close to boundary throws" begin
        cs, geo, m, electron = _setup_point_3d()
        solvers = Solvers(); solvers.add_solver(m)
        fs = Fixed_Sources(cs, geo, solvers)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_energy_group(1)
        ps.set_position([1e-12, 0.5, 0.5])   # within boundary tolerance of x=0
        fs.add_source(ps)
        @test_throws AssertionError fs.build()
    end

end

@testset "Uncollided flux SN - Task 4 (standard SN path, BTE)" begin

    @testset "pure absorber: total flux equals uncollided point flux" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        vol_before = copy(src.get_volume_sources())

        m.set_is_first_collision_source(false)
        flux = Radiant.compute_flux(cs, geo, m, src)

        ad = Radiant.build_angular_discretization(m, geo)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        φ_u = Radiant._compute_uncollided_point_flux_bte(cache, cs, geo, m, src, ad)

        # Pure absorber: no scattering -> Q_FCS = 0 -> scattered flux is zero -> total = φ_u.
        @test flux.get_flux() ≈ φ_u
        # Original source must not be polluted (the path deepcopies before adding Q_FCS).
        @test src.get_volume_sources() == vol_before
        # The point source must still be present (not consumed) on the original source.
        @test src.has_point_sources() == true
        # Spectral radius is reported.
        @test length(flux.get_spectral_radius()) == 1
    end

    @testset "with scattering: scattered flux is non-zero (Q_FCS applied)" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.5)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)

        m.set_is_first_collision_source(false)
        flux = Radiant.compute_flux(cs, geo, m, src)

        ad = Radiant.build_angular_discretization(m, geo)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        φ_u = Radiant._compute_uncollided_point_flux_bte(cache, cs, geo, m, src, ad)

        # The first-collision (scattered) component is φ-total minus the uncollided part.
        scattered = flux.get_flux() .- φ_u
        @test maximum(abs.(scattered)) > 0      # scattering of φ_u produced a non-zero source
        @test all(isfinite, flux.get_flux())
    end

    @testset "energy order > 1 rejected on standard point-source path" begin
        # Standard SN point-source path requires energy order 1 (BTE). A CSD solver
        # (energy order 2) must not silently run the BTE point-source branch.
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)
        m.set_solver_type("BFP")
        m.set_scheme("E", "DG", 2)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        m.set_is_first_collision_source(false)
        @test_throws AssertionError Radiant.compute_flux(cs, geo, m, src)
    end

end

@testset "Uncollided flux SN - Task 5 (FCS path, BTE)" begin

    @testset "BTE point-source FCS: flag true == flag false (consistency)" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.3)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)

        src_false = Source(electron, cs, geo, m); src_false.add_source(ps)
        m.set_is_first_collision_source(false)
        flux_false = Radiant.compute_flux(cs, geo, m, src_false)

        src_true = Source(electron, cs, geo, m); src_true.add_source(ps)
        m.set_is_first_collision_source(true)
        flux_true = Radiant.compute_flux(cs, geo, m, src_true)
        m.set_is_first_collision_source(false)

        # FCS path (no surface source, point source only) must equal the standard path.
        @test flux_true.get_flux() ≈ flux_false.get_flux()
    end

    @testset "BTE point-source FCS matches uncollided + scattered split" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)   # pure absorber
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.5); ps.set_position([0.6, 0.7, 0.8])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        m.set_is_first_collision_source(true)
        flux = Radiant.compute_flux(cs, geo, m, src)
        m.set_is_first_collision_source(false)

        ad = Radiant.build_angular_discretization(m, geo)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        φ_u = Radiant._compute_uncollided_point_flux_bte(cache, cs, geo, m, src, ad)
        # Pure absorber: scattered flux vanishes -> total FCS flux equals uncollided flux.
        @test flux.get_flux() ≈ φ_u
    end

    @testset "BFP-EF(6) with point source throws explicit error" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)
        m.set_solver_type("BFP-EF")
        m.set_scheme("E", "DG", 2)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        m.set_is_first_collision_source(true)
        @test_throws "FCS point source is not supported for BFP-EF solver." Radiant.compute_flux(cs, geo, m, src)
        m.set_is_first_collision_source(false)
    end

end

# Minimal 3D BFP setup on a custom (single-group) cross-section, used to unit-test the
# path-B ray kernels with injected Σtot/S/ΔE coefficients (no physics data required).
function _setup_bfp_unit(; σ=1.0, Nx=3, Ny=3, Nz=3)
    water = Material("water"); water.set_density(1.0)
    water.add_element("H", 0.1119); water.add_element("O", 0.8881)
    electron = Electron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron]); cs.set_source("custom")
    cs.set_absorption([σ]); cs.set_scattering([0.0]); cs.set_legendre_order(2); cs.build()
    geo = Geometry()
    geo.set_type("cartesian"); geo.set_dimension(3)
    for b in ("x-","x+","y-","y+","z-","z+"); geo.set_boundary_conditions(b, "void"); end
    geo.set_material_per_region([water])
    for ax in ("x","y","z"); geo.set_number_of_regions(ax, 1); end
    geo.set_voxels_per_region("x", [Nx]); geo.set_voxels_per_region("y", [Ny]); geo.set_voxels_per_region("z", [Nz])
    geo.set_region_boundaries("x", [0.0, Float64(Nx)])
    geo.set_region_boundaries("y", [0.0, Float64(Ny)])
    geo.set_region_boundaries("z", [0.0, Float64(Nz)])
    geo.build(cs)
    m = SN(); m.set_particle(electron); m.set_solver_type("BFP")
    m.set_quadrature("gauss-legendre-chebychev", 4, 3); m.set_legendre_order(2)
    m.set_angular_boltzmann("standard")
    m.set_scheme("x","DD",1); m.set_scheme("y","DD",1); m.set_scheme("z","DD",1)
    m.set_scheme("E","DG",2)
    m.set_angular_fokker_planck("finite-difference")
    return cs, geo, m, electron
end

@testset "Uncollided flux SN - Task 6 (BFP path B)" begin

    @testset "get_scheme_type getter" begin
        cs, geo, m, electron = _setup_bfp_unit()
        @test m.get_scheme_type("E") == "DG"
        @test m.get_scheme_type("x") == "DD"
    end

    @testset "_ray_cell_csd_step! wraps _ray_cell_kernel_1D" begin
        Σt = 2.0; β1 = 0.3; β2 = 0.1; ds = 0.7; ΦE_in = [1.5, 0.4]; phi_prev = 0.9
        A = Radiant._build_A_matrix(Σt, β1, β2)
        q_eff = Radiant._build_boundary_source(β1, β2, phi_prev; scheme_E="DG")
        out_ref, int_ref = Radiant._ray_cell_kernel_1D(A, ds, ΦE_in, q_eff)
        out, intg = Radiant._ray_cell_csd_step!(Σt, β1, β2, ds, ΦE_in, phi_prev)
        @test out ≈ out_ref
        @test intg ≈ int_ref
    end

    @testset "ray_contribution_to_vertex!: zero stopping power reduces to BTE analytic" begin
        cs, geo, m, electron = _setup_bfp_unit(σ=1.0)
        ad = Radiant.build_angular_discretization(m, geo)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)

        Ns = geo.get_number_of_voxels(); Np = ad.Np
        _, 𝒪, _ = m.get_schemes(geo, m.get_is_full_coupling())
        𝒪E = 𝒪[4]
        @test 𝒪E == 2
        Σtot = fill(1.0, 1, 1); S = zeros(1, 1, 𝒪E); ΔE = [1.0]
        phi_vertex = zeros(1, Np, 𝒪E, Ns[1]+1, Ns[2]+1, Ns[3]+1)

        xe = geo.get_voxels_boundaries("x")
        ye = geo.get_voxels_boundaries("y")
        ze = geo.get_voxels_boundaries("z")
        i, j, k = 3, 3, 4
        target = [xe[i], ye[j], ze[k]]
        Radiant.ray_contribution_to_vertex!(phi_vertex, cache, ps, target, 1, i, j, k,
                                            geo, cs, m, ad, Σtot, S, ΔE, 1)

        dvec = target .- ps.position; r = norm(dvec); Ω = dvec ./ r
        Y = Radiant._sn_basis_vector(Ω, ad)
        τ = cache.tau[1, 1, i, j, k]
        geo_f = 1.0 / r^2
        for p in 1:Np
            # With S=0 the CSD step is pure attenuation -> exact BTE analytic kernel.
            @test phi_vertex[1, p, 1, i, j, k] ≈ Y[p] * (1/(2π)) * 2.0 * 0.5 * exp(-τ) * geo_f
            @test isapprox(phi_vertex[1, p, 2, i, j, k], 0.0; atol=1e-12)
        end
        # cache.rays must be populated for this (ipoint, vertex) and cover the full distance.
        cells_v, segs_v = cache.rays[1, i, j, k]
        @test sum(segs_v) ≈ r
    end

    @testset "ray_contribution_to_vertex!: geo factor uses true vertex distance s_total" begin
        # The endpoint geometric factor must use the full source->vertex distance s_total
        # (= Σ segments), not a segment midpoint. With S=0 the value is exactly
        # Y·exp(-σ·s_total)·q/(2π)·f(μ)·(1/s_total²); a midpoint factor would change 1/s².
        cs, geo, m, electron = _setup_bfp_unit(σ=0.7)
        ad = Radiant.build_angular_discretization(m, geo)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        Ns = geo.get_number_of_voxels(); Np = ad.Np
        _, 𝒪, _ = m.get_schemes(geo, m.get_is_full_coupling())
        Σtot = fill(0.7, 1, 1); S = zeros(1, 1, 𝒪[4]); ΔE = [1.0]
        phi_vertex = zeros(1, Np, 𝒪[4], Ns[1]+1, Ns[2]+1, Ns[3]+1)
        xe = geo.get_voxels_boundaries("x"); ye = geo.get_voxels_boundaries("y"); ze = geo.get_voxels_boundaries("z")
        # Vertex spanning multiple cells so s_total differs clearly from any single segment.
        i, j, k = 4, 4, 4
        target = [xe[i], ye[j], ze[k]]
        Radiant.ray_contribution_to_vertex!(phi_vertex, cache, ps, target, 1, i, j, k,
                                            geo, cs, m, ad, Σtot, S, ΔE, 1)
        r = norm(target .- ps.position)
        Ω = (target .- ps.position) ./ r
        Y = Radiant._sn_basis_vector(Ω, ad)
        cells_v, segs_v = cache.rays[1, i, j, k]
        @test length(segs_v) >= 3              # ray actually spans several cells
        for p in 1:Np
            @test phi_vertex[1, p, 1, i, j, k] ≈ Y[p] * (1/(2π)) * 1.0 * 0.5 * exp(-0.7 * r) * (1 / r^2)
        end
    end

    @testset "ray_contribution_to_vertex!: external source geo factor uses true distance" begin
        # S=0 -> CSD 步是纯衰减，顶点值闭式为 Y·exp(-σ·s_in)·q·f(μ)/(2π)·(1/r²)，
        # 其中 s_in 是域内路径长（衰减只在域内发生），r 是真实源-顶点距离
        #（1/s² 几何扩散从发射点起算，真空段也计入）。
        cs, geo, m, electron = _setup_bfp_unit(σ=0.7)
        ad = Radiant.build_angular_discretization(m, geo)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([-1.0, 1.5, 1.5])
        ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)   # 直接构造，绕过 Fixed_Sources.build 校验
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        Ns = geo.get_number_of_voxels(); Np = ad.Np
        _, 𝒪, _ = m.get_schemes(geo, m.get_is_full_coupling())
        Σtot = fill(0.7, 1, 1); S = zeros(1, 1, 𝒪[4]); ΔE = [1.0]
        phi_vertex = zeros(1, Np, 𝒪[4], Ns[1]+1, Ns[2]+1, Ns[3]+1)
        xe = geo.get_voxels_boundaries("x"); ye = geo.get_voxels_boundaries("y"); ze = geo.get_voxels_boundaries("z")
        # 顶点 (3,2,2) -> 坐标 (2,1,1)：dvec = (3,-0.5,-0.5)，t_enter = 1/3，
        # 进入点 (0,4/3,4/3) 不在任何网格线上（避免角/棱 tie-breaking 依赖）。
        i, j, k = 3, 2, 2
        target = [xe[i], ye[j], ze[k]]
        Radiant.ray_contribution_to_vertex!(phi_vertex, cache, ps, target, 1, i, j, k,
                                            geo, cs, m, ad, Σtot, S, ΔE, 1)
        r = norm(target .- ps.position)              # = √9.5 ≈ 3.0822，真实距离
        Ω = (target .- ps.position) ./ r
        Y = Radiant._sn_basis_vector(Ω, ad)
        cells_v, segs_v = cache.rays[1, i, j, k]
        s_in = sum(segs_v)                           # 域内路径长 ≈ 2.0548
        @test s_in ≈ r - sqrt(9.5) / 3               # 真空段 = √9.5/3 ≈ 1.0274
        @test s_in < r
        for p in 1:Np
            @test phi_vertex[1, p, 1, i, j, k] ≈ Y[p] * (1/(2π)) * 1.0 * 0.5 * exp(-0.7 * s_in) * (1 / r^2)
        end
    end

end

# 3D BFP setup on physics-model cross-sections (8 groups), used for path-B assembler tests.
function _setup_bfp_phys_3d(; Ng=8)
    water = Material("water"); water.set_density(1.0)
    water.add_element("H", 0.1119); water.add_element("O", 0.8881)
    electron = Electron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron])
    cs.set_group_structure("log", Ng, 1.0, 0.01)
    cs.set_interactions([Inelastic_Collision(), Elastic_Collision()])
    cs.set_legendre_order(7); cs.set_source("physics-models"); cs.build()
    geo = Geometry(); geo.set_type("cartesian"); geo.set_dimension(3)
    for b in ("x-","x+","y-","y+","z-","z+"); geo.set_boundary_conditions(b, "void"); end
    geo.set_material_per_region([water])
    for ax in ("x","y","z"); geo.set_number_of_regions(ax, 1); end
    geo.set_voxels_per_region("x", [3]); geo.set_voxels_per_region("y", [3]); geo.set_voxels_per_region("z", [3])
    geo.set_region_boundaries("x", [0.0, 3.0])
    geo.set_region_boundaries("y", [0.0, 3.0])
    geo.set_region_boundaries("z", [0.0, 3.0])
    geo.build(cs)
    m = SN(); m.set_particle(electron); m.set_solver_type("BFP")
    m.set_quadrature("gauss-legendre-chebychev", 4, 3); m.set_legendre_order(7)
    m.set_angular_boltzmann("galerkin-d"); m.set_angular_fokker_planck("finite-difference")
    m.set_scheme("x","DD",1); m.set_scheme("y","DD",1); m.set_scheme("z","DD",1); m.set_scheme("E","DG",2)
    return cs, geo, m, electron
end

function _bfp_fp_data(cs, geo, m)
    ad = Radiant.build_angular_discretization(m, geo)
    N = m.get_quadrature_order(); Nd = ad.Nd
    qt = m.get_quadrature_type(); Qdims = ad.Qdims
    fpt = m.get_angular_fokker_planck()
    ℳ, λ₀ = Radiant.fokker_planck_scattering_matrix(
        N, Nd, qt, geo.get_dimension(), fpt, ad.Mn, ad.Dn, ad.pl, ad.Np, Qdims)
    T = cs.get_momentum_transfer(m.get_particle())
    return ad, ℳ, T, λ₀
end

@testset "Uncollided flux SN - Task 6.1 (BFP path B assembler, physics)" begin
    cs, geo, m, electron = _setup_bfp_phys_3d(Ng=8)
    ad, ℳ, T, λ₀ = _bfp_fp_data(cs, geo, m)
    Ng = cs.get_number_of_groups(electron)
    Ns = geo.get_number_of_voxels()
    _, 𝒪, Nm = m.get_schemes(geo, m.get_is_full_coupling())

    ps = Radiant.Point_Source()
    ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([0.3, 0.4, 0.5])
    ps.set_energy_group(1); ps.build(Ng)
    src = Source(electron, cs, geo, m); src.add_source(ps)
    cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
    φ_u, 𝚽cutoff = Radiant._compute_uncollided_point_flux_bfp(
        cache, cs, geo, m, src, ad; T=T, λ₀=λ₀)

    @testset "shapes match the FCS moment layout" begin
        @test size(φ_u) == (Ng, ad.Np, Nm[5], Ns[1], Ns[2], Ns[3])
        @test size(𝚽cutoff) == (ad.Np, Nm[5], Ns[1], Ns[2], Ns[3])
    end

    @testset "flux is finite and non-zero" begin
        @test all(isfinite, φ_u)
        @test maximum(abs.(φ_u)) > 0
    end

    @testset "CSD down-coupling reaches the lowest group" begin
        # Only group 1 emits; lower groups must receive flux through CSD energy coupling.
        @test maximum(abs.(φ_u[Ng, :, :, :, :, :])) > 0
    end

    @testset "cutoff fills only the cell-average spatial moment (TODO limitation)" begin
        @test all(isfinite, 𝚽cutoff)
        if Nm[5] > 1
            @test all(𝚽cutoff[:, 2:end, :, :, :] .== 0)
        end
    end

    @testset "BFP/FP path requires non-empty T matrix" begin
        @test_throws AssertionError Radiant._compute_uncollided_point_flux_bfp(
            cache, cs, geo, m, src, ad; T=Array{Float64}(undef, 0, 0), λ₀=λ₀)
    end
end

@testset "Uncollided flux SN - Task 6.2 (BFP point source FCS integration)" begin

    @testset "CSD(5) point source is explicitly unsupported" begin
        cs, geo, m, electron = _setup_bfp_phys_3d(Ng=6)
        m.set_solver_type("CSD")
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(6)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        m.set_is_first_collision_source(true)
        @test_throws "CSD is not supported" Radiant.compute_flux(cs, geo, m, src)
        m.set_is_first_collision_source(false)
    end

    @testset "BFP point source end-to-end FCS solve is finite" begin
        cs, geo, m, electron = _setup_bfp_phys_3d(Ng=6)
        m.set_acceleration("livolant")
        m.set_convergence_criterion(1e-4); m.set_maximum_iteration(100)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([0.6, 0.7, 0.8])
        ps.set_energy_group(1); ps.build(6)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        m.set_is_first_collision_source(true)
        flux = Radiant.compute_flux(cs, geo, m, src)
        m.set_is_first_collision_source(false)
        @test all(isfinite, flux.get_flux())
        @test maximum(abs.(flux.get_flux())) > 0
        @test all(isfinite, flux.get_flux_cutoff())
    end

    @testset "disabled switch errors for CSD solvers" begin
        cs, geo, m, electron = _setup_bfp_phys_3d(Ng=6)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([0.3, 0.4, 0.5])
        ps.set_energy_group(1); ps.build(6)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        m.set_is_first_collision_source(true)
        Radiant.ENABLE_BFP_POINT_SOURCE[] = false
        try
            @test_throws "Path B is disabled" Radiant.compute_flux(cs, geo, m, src)
        finally
            Radiant.ENABLE_BFP_POINT_SOURCE[] = true   # restore default
        end
        m.set_is_first_collision_source(false)
    end
end

@testset "Uncollided flux SN - Task 6.3 (BFP point source through grid degeneracies)" begin
    cs, geo, m, electron = _setup_bfp_phys_3d(Ng=8)
    ad, ℳ, T, λ₀ = _bfp_fp_data(cs, geo, m)
    Ng = cs.get_number_of_groups(electron)

    # Place the source on two grid interfaces (x=1 and y=2).  Several source→vertex
    # rays then pass exactly through grid corners/edges/faces, producing zero-length
    # DDA segments.  Before the fix these caused 0/0 -> NaN in the segment-average
    # flux, which the CSD down-scatter term propagated to all lower energy groups.
    ps = Radiant.Point_Source()
    ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position([1.0, 2.0, 1.5])
    ps.set_energy_group(1); ps.build(Ng)
    src = Source(electron, cs, geo, m); src.add_source(ps)
    cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
    φ_u, 𝚽cutoff = Radiant._compute_uncollided_point_flux_bfp(
        cache, cs, geo, m, src, ad; T=T, λ₀=λ₀)

    # Sanity check: zero-length segments were actually generated.
    has_zero_seg = false
    for k in 1:size(cache.rays, 4), j in 1:size(cache.rays, 3), i in 1:size(cache.rays, 2)
        cells, segs = cache.rays[1, i, j, k]
        if any(iszero.(segs))
            has_zero_seg = true
            break
        end
    end
    @test has_zero_seg

    # Main regression assertions: no NaN/Inf, and CSD coupling reaches the lowest group.
    @test all(isfinite, φ_u)
    @test all(isfinite, 𝚽cutoff)
    @test maximum(abs.(φ_u[Ng, :, :, :, :, :])) > 0
end

@testset "Uncollided flux SN - Task 7 (regressions & deprecations)" begin

    @testset "map_moments handles energy order > 1 (𝒪f[4] index fix)" begin
        # Previously map_moments indexed 𝒪f[4][1], erroring for energy order 2. The fix
        # uses 𝒪f[4]; this exercises a full FC map with energy moments.
        𝒪 = [2, 2, 2, 2]
        mp = Radiant.map_moments(𝒪, 𝒪, true, true)
        @test length(mp) == prod(𝒪)
        @test sort(mp) == collect(1:prod(𝒪))   # FC->FC identity is a permutation
    end

    @testset "no-arg get_voxels_boundaries warns and differs from axis version" begin
        cs, geo, m, electron = _setup_point_3d()
        # Non-uniform-ish check: the deprecated no-arg version returns widths, not boundaries.
        widths = (@test_logs (:warn,) match_mode=:any geo.get_voxels_boundaries())
        edges_x = geo.get_voxels_boundaries("x")
        @test widths[1] != edges_x          # widths (length Nx) differ from boundaries (length Nx+1)
        @test length(edges_x) == length(widths[1]) + 1
    end

    @testset "Point_Source and AngularDistribution are exported" begin
        @test isdefined(Radiant, :Point_Source)
        @test isdefined(Radiant, :AngularDistribution)
        @test Point_Source isa Type            # usable unqualified after export
        @test AngularDistribution isa Type
    end
end



@testset "Uncollided flux SN - external point source (BTE)" begin

    @testset "tau cache: only the in-grid path contributes" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0)
        ps.set_position([-1.0, 1.5, 1.5]); ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        x_edges = geo.get_voxels_boundaries("x")
        y_edges = geo.get_voxels_boundaries("y")
        z_edges = geo.get_voxels_boundaries("z")
        for k in 1:4, j in 1:4, i in 1:4
            rv = [x_edges[i], y_edges[j], z_edges[k]]
            dvec = rv .- ps.position
            t_enter = (0.0 - ps.position[1]) / dvec[1]      # 从 x<0 进入 x=0 面
            entry = ps.position .+ t_enter .* dvec
            τ_exact = 1.0 * norm(rv .- entry)               # Σt = Σa = 1.0
            # 顶点分三类：i == 1 的 16 个顶点 t_enter = 1（entry == rv，τ = 0，
            # 依赖 Task 2 的零长度特判）；i == 3 且 j 或 k ∈ {1,4} 的 12 个顶点
            # 进入点落在入口面内部网格线上（4 个为线交点），行进依赖零步
            # tie-breaking（见设计说明）；其余 36 个为普通面进入。
            # atol must exceed the DDA interior-snap tolerance (1e-9 * Δmin = 1e-9
            # here): the march start at the grid entry is snapped inward by
            # _snap_to_interior, shortening the in-grid path by up to ~1e-9.
            @test cache.tau[1, 1, i, j, k] ≈ τ_exact atol=1e-8
        end
    end

    @testset "pure absorber: total flux equals external uncollided point flux" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.0)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_intensity(2.0)
        ps.set_position([-1.0, 1.5, 1.5]); ps.set_energy_group(1); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)

        m.set_is_first_collision_source(false)
        flux = Radiant.compute_flux(cs, geo, m, src)

        ad = Radiant.build_angular_discretization(m, geo)
        cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
        φ_u = Radiant._compute_uncollided_point_flux_bte(cache, cs, geo, m, src, ad)

        @test flux.get_flux() ≈ φ_u
        @test all(isfinite, φ_u)
        # φ_u holds Np angular moments per cell; higher moments of a directional
        # point-source uncollided flux are signed. Only the scalar-flux moment
        # (p = 1) must be non-negative.
        @test all(>=(0), φ_u[:, 1, :, :, :, :])
    end

    @testset "Fixed_Sources.build + compute_flux end-to-end with external source" begin
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.5)
        solvers = Solvers(); solvers.add_solver(m)
        fs = Fixed_Sources(cs, geo, solvers)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_energy_group(1)
        ps.set_position([-1.0, 1.5, 1.5])
        fs.add_source(ps)
        fs.build()
        src = fs.get_source(electron)
        flux = Radiant.compute_flux(cs, geo, m, src)
        @test all(isfinite, flux.get_flux())
    end

    @testset "FCS path with external point source (smoke)" begin
        # 覆盖 compute_flux_fcs → _compute_fcs_precursor 的外部点源分支
        #（sn_flux_fcs.jl:660-683，非 CSD 时复用路径 A 的同一套函数），
        # 为 CLAUDE.md 中"外部点源同样适用于 FCS 路径"的声明提供 e2e 兜底。
        cs, geo, m, electron = _setup_point_3d(Σa=1.0, Σs=0.5)
        ps = Radiant.Point_Source()
        ps.set_particle(electron); ps.set_energy_group(1)
        ps.set_position([-1.0, 1.5, 1.5]); ps.build(1)
        src = Source(electron, cs, geo, m); src.add_source(ps)
        m.set_is_first_collision_source(true)
        flux = Radiant.compute_flux(cs, geo, m, src)
        @test all(isfinite, flux.get_flux())
    end
end
