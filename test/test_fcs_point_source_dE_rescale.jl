using Radiant
using LinearAlgebra
using Test

# Regression tests for the path-B (BFP point source) ΔE group-transition rescale
# fix (fix_point_fcs_dE.md):
#   1. The CSD inter-group coupling state must be rescaled by ΔE[g]/ΔE[g-1] at
#      group transitions (same convention as sn_flux.jl / sn_flux_fcs.jl).
#      Verified by comparing a distant on-axis point source against a perpendicular
#      beam on the centerline: identical physics, so group ratios must agree.
#   2. Multiple point sources must accumulate (not overwrite) at shared vertices.

# -----------------------------------------------------------------------------
# Helpers (self-contained copies of the _ls_* pattern in
# test_fcs_point_source_linear_source.jl so this file runs standalone)
# -----------------------------------------------------------------------------

const _DE_XSEC = joinpath(@__DIR__, "..", "..", "Radiant-Examples", "Paper_2025_AFP", "water_p15_g40.xsec")

function _de_setup_fmac(; Ng=40, Nv=6, Ldom=0.9)
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
    cs.set_source("fmac-m"); cs.set_file(_DE_XSEC); cs.build()
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

function _de_setup_phys_3d(; Ng=8, Nv=3, Ldom=3.0)
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
    for ax in ("x", "y", "z"); m.set_scheme(ax, "DG", 2); end
    m.set_scheme("E", "DG", 2)
    return cs, geo, m, electron
end

function _de_fp_data(cs, geo, m)
    ad = Radiant.build_angular_discretization(m, geo)
    ℳ, λ₀ = Radiant.fokker_planck_scattering_matrix(
        m.get_quadrature_order(), ad.Nd, m.get_quadrature_type(), geo.get_dimension(),
        m.get_angular_fokker_planck(), ad.Mn, ad.Dn, ad.pl, ad.Np, ad.Qdims)
    T = cs.get_momentum_transfer(m.get_particle())
    return ad, ℳ, T, λ₀
end

function _de_make_point_source(electron, pos, Ng)
    ps = Radiant.Point_Source()
    ps.set_particle(electron); ps.set_intensity(1.0); ps.set_position(pos)
    ps.set_energy_group(1); ps.build(Ng)
    return ps
end

function _de_phi_u_point(cs, geo, m, electron, positions::Vector{<:Vector{Float64}}; Ng)
    ad, ℳ, T, λ₀ = _de_fp_data(cs, geo, m)
    src = Source(electron, cs, geo, m)
    for pos in positions
        src.add_source(_de_make_point_source(electron, pos, Ng))
    end
    cache = Radiant.build_point_source_trace_cache(src.point_sources, geo, cs, m)
    φ_u, _ = Radiant._compute_uncollided_point_flux_bfp(cache, cs, geo, m, src, ad; T=T, λ₀=λ₀)
    return φ_u
end

function _de_phi_u_beam(cs, geo, m, electron, dir, loc, Ldom)
    ad, ℳ, T, λ₀ = _de_fp_data(cs, geo, m)
    ss = Surface_Source()
    ss.set_particle(electron); ss.set_intensity(1.0)
    ss.set_energy_group(1); ss.set_direction(dir); ss.set_location(loc)
    # Full-face beam: both transverse axes must be bounded explicitly.
    for ax in ("x", "y", "z")
        ss.set_boundaries(ax, [0.0, Ldom])
    end
    src = Source(electron, cs, geo, m); src.add_source(ss)
    φ_u, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; T=T, λ₀=λ₀)
    return φ_u
end

_rel_l2(a, b) = norm(a .- b) / max(norm(b), 1e-300)

# =============================================================================
# 1. ΔE group-transition rescale: distant on-axis point source ≡ perpendicular
#    beam on the centerline (same direction, same optical path).
# =============================================================================
@testset "BFP point source: ΔE rescale — point vs beam centerline group ratios" begin
    if !isfile(_DE_XSEC)
        println("  [skip] $(_DE_XSEC) not found; accuracy test requires the realistic 40-group data.")
    else
        Ng = 40; Nv = 6; Ldom = 0.9
        cs, geo, m, electron = _de_setup_fmac(; Ng=Ng, Nv=Nv, Ldom=Ldom)

        # External on-axis point source at 200 cm; centerline rays are axis-parallel.
        φ_pt = _de_phi_u_point(cs, geo, m, electron, [[Ldom + 200.0, Ldom/2, Ldom/2]]; Ng=Ng)
        # Perpendicular beam entering through x+ toward -x (same direction as the
        # centerline point-source rays).
        φ_bm = _de_phi_u_beam(cs, geo, m, electron, [-1.0, 0.0, 0.0], "x+", Ldom)

        # Centerline column: with Nv=6 the source axis y=z=0.45 lies on a grid
        # plane, so the symmetric cell pair (3,3)/(4,4) straddles it.
        for ix in 1:Nv, (jc, kc) in ((3, 3), (4, 4))
            r21_pt = φ_pt[2, 1, 1, ix, jc, kc] / φ_pt[1, 1, 1, ix, jc, kc]
            r21_bm = φ_bm[2, 1, 1, ix, jc, kc] / φ_bm[1, 1, 1, ix, jc, kc]
            # Missing ΔE rescale inflates the point-path ratio by ΔE[g-1]/ΔE[g]
            # ≈ 1.259 per cascade step — far outside this tolerance.
            @test isapprox(r21_pt, r21_bm; rtol=0.12)
            r31_pt = φ_pt[3, 1, 1, ix, jc, kc] / φ_pt[1, 1, 1, ix, jc, kc]
            r31_bm = φ_bm[3, 1, 1, ix, jc, kc] / φ_bm[1, 1, 1, ix, jc, kc]
            if r31_pt > 0 && r31_bm > 0   # skip DG sign-artifact cells
                # Two cascade steps: the bug inflates this ratio by ≈ 1.259².
                @test isapprox(r31_pt, r31_bm; rtol=0.20)
            end
        end
    end
end

# =============================================================================
# 2. Multiple point sources must accumulate at shared vertices (not overwrite).
# =============================================================================
@testset "BFP point source: multiple point sources superpose exactly" begin
    Ng = 8
    cs, geo, m, electron = _de_setup_phys_3d(; Ng=Ng)
    pos1 = [0.3, 0.4, 0.5]
    pos2 = [2.6, 2.5, 2.4]

    φ1 = _de_phi_u_point(cs, geo, m, electron, [pos1]; Ng=Ng)
    φ2 = _de_phi_u_point(cs, geo, m, electron, [pos2]; Ng=Ng)
    φ12 = _de_phi_u_point(cs, geo, m, electron, [pos1, pos2]; Ng=Ng)

    # Superposition is exact: each (source, vertex) march is independent.
    @test _rel_l2(φ12, φ1 .+ φ2) < 1e-10
end
