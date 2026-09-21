using Radiant
using LinearAlgebra
using Test

# -----------------------------------------------------------------------------
# Shared helpers: 3D BFP point-source problem on physics-model cross-sections
# (mirrors _setup_bfp_phys_3d in test_uncollided_flux_sn.jl, with selectable
# spatial scheme so both flat (𝒪x=1) and linear (𝒪x=2) ray paths are exercised).
# -----------------------------------------------------------------------------

function _ls_setup_bfp_3d(; Ng=8, Nv=3, Ldom=3.0, spatial=("DD", 1))
    water = Material("water"); water.set_density(1.0)
    water.add_element("H", 0.1119); water.add_element("O", 0.8881)
    electron = Electron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron])
    cs.set_group_structure("log", Ng, 1.0, 0.01)
    cs.set_interactions([Inelastic_Collision(), Elastic_Collision()])
    cs.set_legendre_order(7); cs.set_source("physics-models"); cs.build()
    geo = Geometry(); geo.set_type("cartesian"); geo.set_dimension(3)
    for b in ("x-", "x+", "y-", "y+", "z-", "z+"); geo.set_boundary_conditions(b, "void"); end
    geo.set_material_per_region([water])
    for ax in ("x", "y", "z"); geo.set_number_of_regions(ax, 1); end
    geo.set_voxels_per_region("x", [Nv]); geo.set_voxels_per_region("y", [Nv]); geo.set_voxels_per_region("z", [Nv])
    geo.set_region_boundaries("x", [0.0, Ldom]); geo.set_region_boundaries("y", [0.0, Ldom]); geo.set_region_boundaries("z", [0.0, Ldom])
    geo.build(cs)
    m = SN(); m.set_particle(electron); m.set_solver_type("BFP")
    m.set_quadrature("gauss-legendre-chebychev", 4, 3); m.set_legendre_order(7)
    m.set_angular_boltzmann("galerkin-d"); m.set_angular_fokker_planck("finite-difference")
    for ax in ("x", "y", "z"); m.set_scheme(ax, spatial[1], spatial[2]); end
    m.set_scheme("E", "DG", 2)
    return cs, geo, m, electron
end

function _ls_fp_data(cs, geo, m)
    ad = Radiant.build_angular_discretization(m, geo)
    ℳ, λ₀ = Radiant.fokker_planck_scattering_matrix(
        m.get_quadrature_order(), ad.Nd, m.get_quadrature_type(), geo.get_dimension(),
        m.get_angular_fokker_planck(), ad.Mn, ad.Dn, ad.pl, ad.Np, ad.Qdims)
    T = cs.get_momentum_transfer(m.get_particle())
    return ad, ℳ, T, λ₀
end

"""Compute the BFP point-source uncollided flux φ_u for a unit group-1 point source."""
function _ls_phi_u(cs, geo, m, electron, pos; Ng=8)
    ad, ℳ, T, λ₀ = _ls_fp_data(cs, geo, m)
    ps = Radiant.Point_Source()
    ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position(pos)
    ps.set_energy_group(1); ps.build(Ng)
    src = Source(electron, cs, geo, m); src.add_source(ps)
    cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
    φ_u, _ = Radiant._compute_uncollided_point_flux_bfp(cache, cs, geo, m, src, ad; T=T, λ₀=λ₀)
    return φ_u
end

_rel_l2(a, b) = norm(a .- b) / max(norm(b), 1e-300)

"""Run f(; ) with FCS_POINT_SOURCE_CSD_SUBSTEPS temporarily set to n; always restores."""
function _ls_with_substeps(f, n)
    old = Radiant.FCS_POINT_SOURCE_CSD_SUBSTEPS[]
    Radiant.FCS_POINT_SOURCE_CSD_SUBSTEPS[] = n
    try
        return f()
    finally
        Radiant.FCS_POINT_SOURCE_CSD_SUBSTEPS[] = old
    end
end

# =============================================================================
# Module 1: CSD intra-segment sub-stepping knob (verification reference)
# =============================================================================
@testset "BFP point source: CSD sub-stepping knob" begin
    @test isdefined(Radiant, :FCS_POINT_SOURCE_CSD_SUBSTEPS)
    @test Radiant.FCS_POINT_SOURCE_CSD_SUBSTEPS[] == 1   # default: no sub-stepping

    cs, geo, m, electron = _ls_setup_bfp_3d(; Ng=8, Nv=3, spatial=("DD", 1))
    pos = [0.3, 0.4, 0.5]
    φ1 = _ls_phi_u(cs, geo, m, electron, pos)
    φ8 = _ls_with_substeps(8) do
        _ls_phi_u(cs, geo, m, electron, pos)
    end
    φ32 = _ls_with_substeps(32) do
        _ls_phi_u(cs, geo, m, electron, pos)
    end
    φ64 = _ls_with_substeps(64) do
        _ls_phi_u(cs, geo, m, electron, pos)
    end

    @test all(isfinite, φ64)
    # Sub-stepping changes the flat-source result: the flat-source bias is a real
    # discretization error, not round-off noise.
    @test _rel_l2(φ1, φ64) > 1e-6
    # Self-convergence: 32 sub-steps is much closer to 64 than 8 is (~O(Δs²)).
    @test _rel_l2(φ32, φ64) < 0.3 * _rel_l2(φ8, φ64)
end

# =============================================================================
# Module 2: fcs_ray_spatial_order switch (flat vs linear CSD interface source)
# =============================================================================
@testset "BFP point source: fcs_ray_spatial_order switch semantics" begin
    pos = [0.3, 0.4, 0.5]

    # Invalid order must error (same message as the surface-beam path).
    cs, geo, m, electron = _ls_setup_bfp_3d(; Ng=6, Nv=3, spatial=("DG", 2))
    m.set_fcs_ray_spatial_order(3)
    @test_throws "fcs_ray_spatial_order must be 0 (inherit solver order), 1 or 2." _ls_phi_u(cs, geo, m, electron, pos; Ng=6)

    # 0 = inherit: DG-2 solver default ≡ explicit 2; explicit 1 (flat) differs.
    m.set_fcs_ray_spatial_order(0)
    φ_default = _ls_phi_u(cs, geo, m, electron, pos; Ng=6)
    m.set_fcs_ray_spatial_order(2)
    φ_ox2 = _ls_phi_u(cs, geo, m, electron, pos; Ng=6)
    m.set_fcs_ray_spatial_order(1)
    φ_flat = _ls_phi_u(cs, geo, m, electron, pos; Ng=6)
    @test φ_default == φ_ox2
    @test _rel_l2(φ_flat, φ_ox2) > 1e-6

    # Inherit on a DD-1 solver ≡ explicit 1 (preserves pre-fix behavior).
    cs1, geo1, m1, e1 = _ls_setup_bfp_3d(; Ng=6, Nv=3, spatial=("DD", 1))
    φ_d1_default = _ls_phi_u(cs1, geo1, m1, e1, pos; Ng=6)
    m1.set_fcs_ray_spatial_order(1)
    @test φ_d1_default == _ls_phi_u(cs1, geo1, m1, e1, pos; Ng=6)
    # Inherit takes the maximum over axes for non-uniform spatial orders.
    cs2, geo2, m2, e2 = _ls_setup_bfp_3d(; Ng=6, Nv=3, spatial=("DG", 2))
    m2.set_scheme("y", "DD", 1)
    m2.set_scheme("z", "DD", 1)
    φ_mixed_default = _ls_phi_u(cs2, geo2, m2, e2, pos; Ng=6)
    m2.set_fcs_ray_spatial_order(2)
    @test φ_mixed_default == _ls_phi_u(cs2, geo2, m2, e2, pos; Ng=6)
end

# Realistic 40-group water cross-sections (fmac-m file, same data as benchmark3).
# The physics-models setup used elsewhere in this file has λ₀T so large that every
# cell is optically thick for the uncollided ray (Σ̃t·Δs ~ 15), a regime in which NO
# low-order source representation works; the accuracy tests below therefore require
# the realistic data and are skipped (with a notice) when the file is unavailable.
const _LS_XSEC = joinpath(@__DIR__, "..", "..", "Radiant-Examples", "Paper_2025_AFP", "water_p15_g40.xsec")

function _ls_setup_bfp_3d_fmac(; Ng=40, Nv=6, Ldom=0.9)
    water = Material("water"); water.set_density(1.0)
    water.add_element("H", 0.1119); water.add_element("O", 0.8881)
    electron = Electron(); photon = Photon(); positron = Positron()
    cs = Cross_Sections()
    cs.set_materials(water); cs.set_particles([electron, photon, positron])
    cs.set_group_structure("log", Ng, 10.0, 0.001)
    cs.set_interactions([Inelastic_Collision(), Elastic_Collision(), Bremsstrahlung(),
        Pair_Production(), Photoelectric(), Rayleigh(), Compton(), Auger(),
        Fluorescence(), Annihilation()])
    cs.set_legendre_order(15)
    cs.set_source("fmac-m"); cs.set_file(_LS_XSEC); cs.build()
    geo = Geometry(); geo.set_type("cartesian"); geo.set_dimension(3)
    for b in ("x-", "x+", "y-", "y+", "z-", "z+"); geo.set_boundary_conditions(b, "void"); end
    geo.set_material_per_region([water])
    for ax in ("x", "y", "z"); geo.set_number_of_regions(ax, 1); end
    geo.set_voxels_per_region("x", [Nv]); geo.set_voxels_per_region("y", [Nv]); geo.set_voxels_per_region("z", [Nv])
    geo.set_region_boundaries("x", [0.0, Ldom]); geo.set_region_boundaries("y", [0.0, Ldom]); geo.set_region_boundaries("z", [0.0, Ldom])
    geo.build(cs)
    m = SN(); m.set_particle(electron); m.set_solver_type("BFP")
    m.set_quadrature("gauss-legendre-chebychev", 4, 3); m.set_legendre_order(15)
    m.set_angular_boltzmann("galerkin-d"); m.set_angular_fokker_planck("finite-difference")
    for ax in ("x", "y", "z"); m.set_scheme(ax, "DG", 2); end
    m.set_scheme("E", "DG", 2)
    return cs, geo, m, electron
end

@testset "BFP point source: linear interface source matches subdivided reference" begin
    if !isfile(_LS_XSEC)
        println("  [skip] $(_LS_XSEC) not found; accuracy tests require the realistic 40-group data.")
    else
    pos = [0.08, 0.09, 0.10]
    cs, geo, m, electron = _ls_setup_bfp_3d_fmac(; Ng=40, Nv=6, Ldom=0.9)

    m.set_fcs_ray_spatial_order(1)
    φ_flat = _ls_phi_u(cs, geo, m, electron, pos; Ng=40)
    m.set_fcs_ray_spatial_order(2)
    φ_ox2 = _ls_phi_u(cs, geo, m, electron, pos; Ng=40)
    # Reference: flat scheme with 32 CSD sub-steps (converged; its residual vs the
    # 64-sub-step solution is ~3e-6, negligible against the tolerances below).
    m.set_fcs_ray_spatial_order(1)
    φ_ref = _ls_with_substeps(32) do
        _ls_phi_u(cs, geo, m, electron, pos; Ng=40)
    end
    # Cross-check: linear scheme with 32 sub-steps converges to the same limit.
    m.set_fcs_ray_spatial_order(2)
    φ_ox2_32 = _ls_with_substeps(32) do
        _ls_phi_u(cs, geo, m, electron, pos; Ng=40)
    end

    @test all(isfinite, φ_ox2)
    @test all(isfinite, φ_ox2_32)
    # The flat-scheme bias is real on this benchmark-like (Δ = 0.15 cm) grid ...
    @test _rel_l2(φ_flat, φ_ref) > 1e-3
    # ... and the linear interface source removes at least 90% of it.
    @test _rel_l2(φ_ox2, φ_ref) < 0.1 * _rel_l2(φ_flat, φ_ref)
    # Both schemes converge to the same limit under sub-stepping.
    @test _rel_l2(φ_ox2_32, φ_ref) < 1e-3

    # The flat-scheme bias is a distance-growing overestimate: the cell-average
    # scalar-flux bias vs the reference stays positive and grows from mid-domain
    # to the corner cell farthest from the source.
    bias_of(φ) = dropdims(sum(@view(φ[:, 1, 1, :, :, :]); dims=1); dims=1) ./
                 dropdims(sum(@view(φ_ref[:, 1, 1, :, :, :]); dims=1); dims=1) .- 1.0
    b = bias_of(φ_flat)
    @test b[6, 6, 6] > 0.0
    @test b[6, 6, 6] > b[3, 3, 3]
    # The linear scheme's bias is uniformly tiny in comparison.
    b2 = bias_of(φ_ox2)
    @test abs(b2[6, 6, 6]) < 0.1 * abs(b[6, 6, 6])
    end
end

@testset "BFP point source: end-to-end FCS solve with linear source (DG-2)" begin
    cs, geo, m, electron = _ls_setup_bfp_3d(; Ng=6, Nv=3, spatial=("DG", 2))
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
