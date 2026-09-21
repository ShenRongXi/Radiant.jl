using Radiant
using LinearAlgebra
using Test

# Lightweight single-group custom 3D problem (fast: avoids physics-model build).
function _setup_3d_custom(Σa, Σs; Ns=(30,4,4), L=(1.5,1.5,1.5), N=4, Lg=2)
    water = Material("water"); water.set_density(1.0)
    water.add_element("H", 0.1119); water.add_element("O", 0.8881)
    electron = Electron()
    cs = Cross_Sections(); cs.set_materials(water); cs.set_particles([electron])
    cs.set_source("custom"); cs.set_absorption([Σa]); cs.set_scattering([Σs])
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

function _axis_beam_source(electron, cs, geo, m, Ωs, loc)
    ss = Surface_Source()
    ss.set_particle(electron); ss.set_intensity(1.0); ss.set_energy_group(1)
    ss.set_direction(Ωs); ss.set_location(loc)
    if loc ∈ ("x-","x+")
        ss.set_boundaries("y", [0.0, 1.5]); ss.set_boundaries("z", [0.0, 1.5])
    elseif loc ∈ ("y-","y+")
        ss.set_boundaries("x", [0.0, 1.5]); ss.set_boundaries("z", [0.0, 1.5])
    else
        ss.set_boundaries("x", [0.0, 1.5]); ss.set_boundaries("y", [0.0, 1.5])
    end
    source = Source(electron, cs, geo, m)
    source.add_source(ss)
    return source
end

@testset "ray_sweep_3D FCS integration" begin

    # ----- Tier-2a: fallback predicate (exhaustive, cheap) -----
    @testset "_ray3d_applicable predicate" begin
        ax = [([1.0,0.0,0.0],1), ([-1.0,0.0,0.0],1),
              ([0.0,1.0,0.0],2), ([0.0,0.0,-1.0],3)]
        # Applicable: axis-aligned, transverse==1, along-axis∈{1,2}, CSD⇒𝒪E==2
        for (Ω, d) in ax
            𝒪1 = [1,1,1,1]
            𝒪2 = [1,1,1,1]; 𝒪2[d] = 2
            @test Radiant._ray3d_applicable(true, Ω, 𝒪1, false)
            @test Radiant._ray3d_applicable(true, Ω, 𝒪2, false)
            𝒪csd = [1,1,1,2]; 𝒪csd[d] = 2
            @test Radiant._ray3d_applicable(true, Ω, 𝒪csd, true)
        end
        # Not applicable
        @test !Radiant._ray3d_applicable(false, [1.0,0.0,0.0], [1,1,1,1], false)  # ray off
        @test !Radiant._ray3d_applicable(true, [0.8,0.6,0.0], [1,1,1,1], false)   # oblique
        @test !Radiant._ray3d_applicable(true, [1.0,0.0,0.0], [2,2,1,2], false)   # transverse>1
        @test !Radiant._ray3d_applicable(true, [1.0,0.0,0.0], [3,1,1,1], false)   # along-axis>2
        @test !Radiant._ray3d_applicable(true, [1.0,0.0,0.0], [2,1,1,3], true)    # CSD 𝒪E≠2
    end

    # ----- Tier-2b: ray vs SN flux integral (axis-aligned, six faces) -----
    # ray_sweep_3D is the exact uncollided flux; SN (DD) converges to it. On a fine
    # grid (Σt·Δ ≈ 0.05) the integrated flux agrees to ~1e-2. Integral-level sanity
    # check, not a strict cell-wise equivalence.
    @testset "ray vs SN uncollided flux integral (six faces)" begin
        Σt = 1.0
        # Cube fine on ALL axes (Σt·Δ ≈ 0.083) so every sweep direction is well
        # resolved — otherwise a coarse sweep axis inflates the SN truncation error.
        cs, geo, m, electron = _setup_3d_custom(Σt, 0.0; Ns=(12,12,12), L=(1.0,1.0,1.0))
        ad = Radiant.build_angular_discretization(m, geo)
        faces = [("x-",[1.0,0.0,0.0]), ("x+",[-1.0,0.0,0.0]),
                 ("y-",[0.0,1.0,0.0]), ("y+",[0.0,-1.0,0.0]),
                 ("z-",[0.0,0.0,1.0]), ("z+",[0.0,0.0,-1.0])]
        for (loc, Ωs) in faces
            src = _axis_beam_source(electron, cs, geo, m, Ωs, loc)
            φ_ray, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=true)
            φ_sn,  _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=false)
            # Scalar 0th moment [ig=1, p=1, is=1, :, :, :] = uncollided scalar flux.
            S_ray = sum(@view φ_ray[1,1,1,:,:,:])
            S_sn  = sum(@view φ_sn[1,1,1,:,:,:])
            @test isapprox(S_ray, S_sn; rtol=1e-2)
            @test S_ray > 0.0
        end
    end

    # ----- Tier-2c: fallback routes to SN end-to-end (oblique) -----
    @testset "oblique beam falls back to SN (bit-identical)" begin
        Σt = 1.0
        cs, geo, m, electron = _setup_3d_custom(Σt, 0.0; Ns=(20,4,4))
        ad = Radiant.build_angular_discretization(m, geo)
        # [0.8,0.6,0.0] is a unit vector but oblique ⇒ ray not applicable ⇒ SN path.
        src = _axis_beam_source(electron, cs, geo, m, [0.8,0.6,0.0], "x-")
        φ_true,  _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=true)
        φ_false, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=false)
        @test φ_true == φ_false   # same sn_sweep_3D code path
    end

end

# Helper: create a DG-1 reference solver copying angular settings from `m`.
function _make_dg1_ref_solver(m::SN)
    # NOTE: get_solver_type() returns (Int, Bool), but set_solver_type()
    # expects a String.  Access the raw field m.solver_type directly.
    m_ref = SN()
    m_ref.set_particle(m.get_particle())
    m_ref.set_solver_type(m.solver_type)
    m_ref.set_quadrature(m.get_quadrature_type(), m.get_quadrature_order(), m.get_quadrature_dimension())
    m_ref.set_legendre_order(m.get_legendre_order())
    m_ref.set_angular_boltzmann(m.get_angular_boltzmann())
    m_ref.set_scheme("x", "DG", 1)
    m_ref.set_scheme("y", "DG", 1)
    m_ref.set_scheme("z", "DG", 1)
    if haskey(m.scheme_type, "E")
        m_ref.set_scheme("E", m.scheme_type["E"], get(m.scheme_order, "E", 1))
    end
    m_ref.set_use_ray_sweep(true)
    m_ref.set_force_ray_sweep(false)
    return m_ref
end

@testset "force_ray_sweep" begin

    # ----- Test 1: force ray sweep with DG-2 transverse (path verification) -----
    @testset "force ray sweep with DG-2 transverse (path verification)" begin
        Σt = 1.0
        cs, geo, m, electron = _setup_3d_custom(Σt, 0.0; Ns=(12,12,12), L=(1.0,1.0,1.0))
        m.set_scheme("x", "DG", 2)
        m.set_scheme("y", "DG", 2)
        m.set_scheme("z", "DG", 2)
        ad = Radiant.build_angular_discretization(m, geo)

        # DG-1 ray_sweep reference.
        # NOTE: `ad` is built from `m` (DG-2), but angular discretization does
        # not depend on spatial orders, so it is safe to reuse for `m_ref`.
        m_ref = _make_dg1_ref_solver(m)
        src = _axis_beam_source(electron, cs, geo, m_ref, [1.0,0.0,0.0], "x-")
        φ_ref, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m_ref, src, ad; use_ray_sweep=true)

        # force=false: _ray3d_applicable returns false → sn_sweep_3D
        m.set_force_ray_sweep(false)
        φ_noforce, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=true)

        # force=true: should use ray_sweep_3D with transverse clamped to 1
        m.set_force_ray_sweep(true)
        φ_force, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=true)

        # Path verification:
        # If force works: φ_force uses ray_sweep_3D (transverse clamped=1),
        #   its zeroth spatial moment should match DG-1 ray_sweep reference.
        # If force is ignored: φ_force falls back to sn_sweep_3D,
        #   which differs from ray_sweep by more than tight tolerance.
        # Extract scalar flux (zeroth spatial moment is=1) for comparison
        # since the two solvers have different Nm[5] (8 vs 1).
        φ_force_scalar = φ_force[1, :, 1, :, :, :]
        φ_ref_scalar   = φ_ref[1, :, 1, :, :, :]
        @test φ_force_scalar ≈ φ_ref_scalar
        @test sum(φ_ref) > 0.0
    end

    # ----- Test 2: force_ray_sweep with oblique beam falls back to SN -----
    @testset "force ray sweep with oblique beam falls back to SN" begin
        cs, geo, m, electron = _setup_3d_custom(1.0, 0.0; Ns=(10,10,10), L=(1.0,1.0,1.0))
        m.set_scheme("y", "DG", 2); m.set_scheme("z", "DG", 2)
        m.set_force_ray_sweep(true)
        ad = Radiant.build_angular_discretization(m, geo)

        # Oblique beam — force should NOT activate
        # (_is_axis_aligned([0.8,0.6,0.0]) → false)
        src = _axis_beam_source(electron, cs, geo, m, [0.8,0.6,0.0], "x-")
        φ_oblique, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad)

        # Same with force=false should be bit-identical (both fall to sn_sweep_3D)
        m.set_force_ray_sweep(false)
        φ_oblique_noforce, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad)
        @test φ_oblique == φ_oblique_noforce
    end

    # ----- Test 3: force_ray_sweep respects use_ray_sweep=false -----
    @testset "force_ray_sweep respects use_ray_sweep=false" begin
        cs, geo, m, electron = _setup_3d_custom(1.0, 0.0; Ns=(10,10,10), L=(1.0,1.0,1.0))
        m.set_scheme("y", "DG", 2); m.set_scheme("z", "DG", 2)
        m.set_use_ray_sweep(false)       # user explicitly disables ray_sweep
        m.set_force_ray_sweep(true)      # force should NOT override
        ad = Radiant.build_angular_discretization(m, geo)

        src = _axis_beam_source(electron, cs, geo, m, [1.0,0.0,0.0], "x-")
        # NOTE: must pass use_ray_sweep=false explicitly — the kwarg defaults to true
        # and the function does NOT read solver.get_use_ray_sweep() internally.
        φ, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=false)

        # force=false with same setup should be identical
        m.set_force_ray_sweep(false)
        φ_noforce, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=false)
        @test φ == φ_noforce
    end

    # ----- Test 4: default force_ray_sweep=false preserves existing behavior -----
    @testset "force_ray_sweep default (false) preserves existing behavior" begin
        cs, geo, m, electron = _setup_3d_custom(1.0, 0.0; Ns=(10,10,10), L=(1.0,1.0,1.0))
        m.set_scheme("x", "DG", 2); m.set_scheme("y", "DG", 2); m.set_scheme("z", "DG", 2)
        # force_ray_sweep not set → default false
        ad = Radiant.build_angular_discretization(m, geo)

        src = _axis_beam_source(electron, cs, geo, m, [1.0,0.0,0.0], "x-")
        φ_default, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=true)

        # Explicit false should match
        m.set_force_ray_sweep(false)
        φ_explicit, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=true)
        @test φ_default == φ_explicit
    end

    # ----- Test 5: force_ray_sweep combined with fcs_ray_spatial_order -----
    @testset "force_ray_sweep combined with fcs_ray_spatial_order (DG-2 all axes)" begin
        # Verify that force (transverse clamp to 1) and mixed_order (along-axis
        # reduction) compose correctly.  The result with (force=true, ray_order=1)
        # should match a DG-1 ray_sweep reference (transverse=1, along-axis=1).
        Σt = 1.0
        cs, geo, m, electron = _setup_3d_custom(Σt, 0.0; Ns=(12,12,12), L=(1.0,1.0,1.0))
        m.set_scheme("x", "DG", 2)
        m.set_scheme("y", "DG", 2)
        m.set_scheme("z", "DG", 2)
        m.set_force_ray_sweep(true)
        m.set_fcs_ray_spatial_order(1)
        ad = Radiant.build_angular_discretization(m, geo)

        # DG-1 all-axes ray_sweep reference (reuses _make_dg1_ref_solver)
        m_ref = _make_dg1_ref_solver(m)
        m_ref.set_fcs_ray_spatial_order(0)  # inherit solver's 𝒪=[1,1,1,1]
        src = _axis_beam_source(electron, cs, geo, m_ref, [1.0,0.0,0.0], "x-")
        # NOTE: `ad` from DG-2 solver `m` is safe to reuse — angular discretization
        # does not depend on spatial orders.
        φ_ref, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m_ref, src, ad; use_ray_sweep=true)

        # force + mixed_order: 𝒪_sweep = [1,1,1,𝒪E] (transverse→1, along-axis→1)
        φ_combo, _ = Radiant._compute_uncollided_surface_flux(cs, geo, m, src, ad; use_ray_sweep=true)

        # Zeroth spatial moments should match DG-1 reference
        φ_combo_scalar = φ_combo[1, :, 1, :, :, :]
        φ_ref_scalar   = φ_ref[1, :, 1, :, :, :]
        @test φ_combo_scalar ≈ φ_ref_scalar
        @test sum(φ_ref) > 0.0

        # Higher-order transverse moments must be identically zero.
        # φ_combo shape: (Ng, Np, Nm[5], Nx, Ny, Nz)
        #   dim 1 = Ng (energy groups)
        #   dim 2 = Np (angular moments)
        #   dim 3 = Nm[5] (spatial moments)  ← check this
        #   dim 4-6 = Nx, Ny, Nz
        energy_order = haskey(m.scheme_order, "E") ? m.scheme_order["E"] : 1
        Nm_ray = Radiant._compute_nm_from_o([1, 1, 1, energy_order], m.get_is_full_coupling())
        if Nm_ray[5] < size(φ_combo, 3)              # dim 3 = Nm[5]
            high_order_slice = @view φ_combo[1, :, Nm_ray[5]+1:end, :, :, :]
            @test all(iszero, high_order_slice)
        end
    end

end
