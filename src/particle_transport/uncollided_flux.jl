"""
Point-source uncollided-flux infrastructure for the SN solver.

This file provides the geometry-agnostic building blocks used by the BTE (path A)
and BFP (path B) point-source first-collision-source paths:

  * `find_cell`                    : locate the cell containing a point.
  * `_ray_entry_to_box`            : entry point of an external source ray into the grid.
  * `dda_trace_cells`              : 3D digital differential analyzer voxel march.
  * `dda_optical_depth`            : optical depth along a source -> target ray.
  * `_project_lagrange_to_legendre`: 8 vertex values -> normalized tensor Legendre moments.
  * `moment_index`                 : (ix,iy,iz,iE) -> flat moment index (matches `map_moments`).

External point sources (position outside the grid bounding box) are supported:
the region outside the domain is vacuum, the DDA march starts at the ray's entry
point into the grid (a ray entering at the target vertex itself yields an empty
ray and tau = 0), and the 1/r^2 factor uses the true source-vertex distance.

Note: the upstream package does not depend on StaticArrays, so positions are plain
`AbstractVector{<:Real}` (length 3) rather than `SVector{3}`.
"""

# ---------------------------------------------------------------------------
# Point-source ray-trace cache
# ---------------------------------------------------------------------------

"""
    ENABLE_BFP_POINT_SOURCE

Master switch for the BFP point-source FCS path (path B). Enabled by default; kept as a
`Ref{Bool}` so it can be toggled at runtime (`Radiant.ENABLE_BFP_POINT_SOURCE[] = false`)
for debugging or comparison without recompilation.
"""
const ENABLE_BFP_POINT_SOURCE = Ref(true)

"""
    FCS_POINT_SOURCE_CSD_SUBSTEPS

Number of CSD sub-steps per DDA segment in the BFP point-source path (path B). Default 1
(no sub-stepping). Values > 1 subdivide every nonzero ray segment for the CSD energy
stepping only (material is constant within a segment, so no extra ray tracing is needed).
Used as a convergence reference for the flat/linear inter-group coupling schemes; kept as
a `Ref{Int}` so it can be toggled at runtime (`Radiant.FCS_POINT_SOURCE_CSD_SUBSTEPS[] = 64`)
for verification without recompilation.
"""
const FCS_POINT_SOURCE_CSD_SUBSTEPS = Ref(1)

"""
    PointSourceTraceCache

Cached ray-tracing results for point-source uncollided flux. `tau[ipoint, ig, i, j, k]`
holds the optical depth from point source `ipoint` to grid vertex `(i,j,k)` in group `ig`
(used by the BTE path A). `rays[ipoint, i, j, k]` holds the `(cells, segments)` of that
ray (geometry- and energy-independent, filled by the BFP path B). A 4D array is used
(not a `Dict`) so per-vertex parallel writes never collide, and the `ipoint` dimension
keeps multiple sources from overwriting one another.
"""
struct PointSourceTraceCache
    tau::Array{Float64,5}
    rays::Array{Tuple{Vector{CartesianIndex{3}},Vector{Float64}},4}
end

# ---------------------------------------------------------------------------
# Cell location
# ---------------------------------------------------------------------------

"""
    _snap_to_interior(r_d, edges_d, tol)

If a coordinate `r_d` lies (within `tol`) on the lower/upper boundary of the grid
along one axis, nudge it a tiny distance into the interior so that `find_cell`
stays inside the grid and `dda_trace_cells` does not produce a zero-length initial
segment. This is a local safeguard of the present ray-tracing procedure.
"""
function _snap_to_interior(r_d::Real, edges_d::AbstractVector{<:Real}, tol::Real)
    if abs(r_d - edges_d[1]) < tol
        Δ = edges_d[2] - edges_d[1]
        return edges_d[1] + 1e-9 * Δ
    elseif abs(r_d - edges_d[end]) < tol
        Δ = edges_d[end] - edges_d[end-1]
        return edges_d[end] - 1e-9 * Δ
    else
        return float(r_d)
    end
end

"""
    find_cell(x_edges, y_edges, z_edges, r; tol=nothing)

Return the `CartesianIndex` of the cell containing point `r = (x,y,z)`. Points on a
boundary are snapped into the grid interior first. The default tolerance matches the
`Fixed_Sources.build()` boundary tolerance, `1e-9 * min(Δx_min, Δy_min, Δz_min)`.
"""
function find_cell(x_edges::AbstractVector{<:Real}, y_edges::AbstractVector{<:Real},
                   z_edges::AbstractVector{<:Real}, r::AbstractVector{<:Real};
                   tol::Union{Nothing,Real}=nothing)
    if isnothing(tol)
        Δmin = min(minimum(diff(x_edges)), minimum(diff(y_edges)), minimum(diff(z_edges)))
        tol = 1e-9 * Δmin
    end
    rx = _snap_to_interior(r[1], x_edges, tol)
    ry = _snap_to_interior(r[2], y_edges, tol)
    rz = _snap_to_interior(r[3], z_edges, tol)
    ix = clamp(searchsortedlast(x_edges, rx), 1, length(x_edges) - 1)
    iy = clamp(searchsortedlast(y_edges, ry), 1, length(y_edges) - 1)
    iz = clamp(searchsortedlast(z_edges, rz), 1, length(z_edges) - 1)
    return CartesianIndex(ix, iy, iz)
end

"""
    _point_in_box(r, x_edges, y_edges, z_edges)

Return `true` when `r = (x,y,z)` lies inside the closed bounding box of the grid.
"""
function _point_in_box(r::AbstractVector{<:Real}, x_edges::AbstractVector{<:Real},
                       y_edges::AbstractVector{<:Real}, z_edges::AbstractVector{<:Real})
    return (x_edges[1] <= r[1] <= x_edges[end]) &&
           (y_edges[1] <= r[2] <= y_edges[end]) &&
           (z_edges[1] <= r[3] <= z_edges[end])
end

"""
    _ray_entry_to_box(src, tgt, x_edges, y_edges, z_edges)

Entry point of the segment `src -> tgt` into the grid bounding box (slab method),
for `src` outside the box. Returns `nothing` when the segment does not intersect
the box. Callers must check `_point_in_box(src, ...)` first; for an interior `src`
the entry point is `src` itself and this function is not needed.
"""
function _ray_entry_to_box(src::AbstractVector{<:Real}, tgt::AbstractVector{<:Real},
                           x_edges::AbstractVector{<:Real}, y_edges::AbstractVector{<:Real},
                           z_edges::AbstractVector{<:Real})
    d = Float64[tgt[1] - src[1], tgt[2] - src[2], tgt[3] - src[3]]
    t_enter = 0.0
    t_exit  = 1.0          # only the segment src -> tgt matters (tgt is inside the box)
    for (s_d, d_d, lo, hi) in ((src[1], d[1], x_edges[1], x_edges[end]),
                               (src[2], d[2], y_edges[1], y_edges[end]),
                               (src[3], d[3], z_edges[1], z_edges[end]))
        if iszero(d_d)
            (s_d < lo || s_d > hi) && return nothing   # parallel and outside the slab
        else
            t1 = (lo - s_d) / d_d
            t2 = (hi - s_d) / d_d
            t_enter = max(t_enter, min(t1, t2))
            t_exit  = min(t_exit,  max(t1, t2))
        end
    end
    t_enter > t_exit && return nothing
    return Float64[src[1] + t_enter * d[1],
                   src[2] + t_enter * d[2],
                   src[3] + t_enter * d[3]]
end

# ---------------------------------------------------------------------------
# 3D DDA voxel march
# ---------------------------------------------------------------------------

"""
    dda_trace_cells(src, tgt, x_edges, y_edges, z_edges, mat_ids; tol=nothing)

March from `src` to `tgt` through a Cartesian grid, returning `(cells, segments)`:
the list of traversed cells and the path length within each. Handles zero direction
components, negative directions, and the degenerate case where the current point lies
exactly on a cell interface (corner/edge crossing) via an x->y->z tie-breaking fallback.
If `src` lies outside the grid bounding box (external point source, vacuum outside),
the march starts at the ray's entry point into the grid instead; a ray that enters
the grid at (within tol of) `tgt` itself returns empty `cells`/`segments`.
"""
function dda_trace_cells(src::AbstractVector{<:Real}, tgt::AbstractVector{<:Real},
                         x_edges::AbstractVector{<:Real}, y_edges::AbstractVector{<:Real},
                         z_edges::AbstractVector{<:Real}, mat_ids;
                         tol::Union{Nothing,Real}=nothing)
    # `all(src .!= tgt)` would reject legal vertices coplanar with the source; only a
    # full coincidence is forbidden.
    @assert src != tgt "DDA source and target must differ."

    Ω = (tgt .- src) ./ norm(tgt .- src)

    if isnothing(tol)
        Δmin = min(minimum(diff(x_edges)), minimum(diff(y_edges)), minimum(diff(z_edges)))
        tol = max(1e-12, 1e-9 * Δmin)
    end

    # External point source: vacuum outside the domain contributes no optical
    # depth, so advance the march start to the ray's entry point into the grid.
    # The 1/r^2 geometric factor is unaffected: callers use the true source
    # position for distances and directions.
    if !_point_in_box(src, x_edges, y_edges, z_edges)
        entry = _ray_entry_to_box(src, tgt, x_edges, y_edges, z_edges)
        isnothing(entry) && error("DDA ray from external source does not intersect the grid.")
        src = entry
        # Ray enters the grid exactly at (within tol of) the target: the in-grid
        # path has zero length. Return an empty ray (tau = 0) instead of marching
        # the spurious ~1e-9 segment that _snap_to_interior would create.
        # NOTE: path B derives its 1/s^2 factor from sum(segments); an empty ray
        # there is only correct once s_total uses the true source-vertex distance
        # (see ray_contribution_to_vertex!). Do not ship one without the other.
        norm(tgt .- src) <= tol && return CartesianIndex{3}[], Float64[]
    end

    ijk = find_cell(x_edges, y_edges, z_edges, src; tol=tol)

    r = Float64[
        _snap_to_interior(src[1], x_edges, tol),
        _snap_to_interior(src[2], y_edges, tol),
        _snap_to_interior(src[3], z_edges, tol),
    ]

    function dist_to_next(i, edges, omg, r_d)
        if abs(omg) < eps()
            return Inf
        elseif omg > 0
            return max(edges[i+1] - r_d, 0.0) / omg
        else
            return max(r_d - edges[i], 0.0) / abs(omg)
        end
    end

    cells = Vector{CartesianIndex{3}}()
    segments = Vector{Float64}()
    done = false

    while !done
        push!(cells, ijk)

        dx = dist_to_next(ijk[1], x_edges, Ω[1], r[1])
        dy = dist_to_next(ijk[2], y_edges, Ω[2], r[2])
        dz = dist_to_next(ijk[3], z_edges, Ω[3], r[3])

        d = min(dx, dy, dz)
        s_to_tgt = norm(tgt .- r)
        if d >= s_to_tgt - tol
            d = s_to_tgt
            done = true
        end

        if d <= tol && !done
            # Degenerate: current point sits on one or more interfaces and the
            # recomputed distance is ~0. Advance the cell index for every axis that is
            # both sitting on its next interface (dist <= tol) and moving, with a
            # zero-length segment, so the march turns the corner/edge instead of
            # stranding an axis. Fall back to error only if nothing can advance.
            di = (dx <= tol && abs(Ω[1]) > eps()) ? Int(sign(Ω[1])) : 0
            dj = (dy <= tol && abs(Ω[2]) > eps()) ? Int(sign(Ω[2])) : 0
            dk = (dz <= tol && abs(Ω[3]) > eps()) ? Int(sign(Ω[3])) : 0
            if di == 0 && dj == 0 && dk == 0
                error("DDA step length ($d) below tolerance ($tol) and no axis on an interface could advance; point source may be too close to a cell boundary or vertex.")
            end
            ijk = CartesianIndex(ijk[1] + di, ijk[2] + dj, ijk[3] + dk)
            push!(segments, 0.0)
            continue
        end

        push!(segments, d)
        r = r .+ d .* Ω

        if !done
            # Step every axis whose interface is reached within tol simultaneously.
            # At a true corner/edge crossing this advances all tied axes at once
            # (the collapsed face-adjacent cells are measure-zero in path length).
            step_x = d >= dx - tol
            step_y = d >= dy - tol
            step_z = d >= dz - tol
            di = step_x ? Int(sign(Ω[1])) : 0
            dj = step_y ? Int(sign(Ω[2])) : 0
            dk = step_z ? Int(sign(Ω[3])) : 0
            ijk = CartesianIndex(ijk[1] + di, ijk[2] + dj, ijk[3] + dk)
        end
    end

    return cells, segments
end

"""
    dda_optical_depth(src, tgt, x_edges, y_edges, z_edges, mat_ids, sigma_eff)

Optical depth `∫ σ ds` from `src` to `tgt`, where `sigma_eff[material_id]` gives the
effective total cross-section per material.
"""
function dda_optical_depth(src::AbstractVector{<:Real}, tgt::AbstractVector{<:Real},
                           x_edges::AbstractVector{<:Real}, y_edges::AbstractVector{<:Real},
                           z_edges::AbstractVector{<:Real}, mat_ids,
                           sigma_eff::AbstractVector{<:Real})
    cells, segments = dda_trace_cells(src, tgt, x_edges, y_edges, z_edges, mat_ids)
    τ = 0.0
    for (c, ds) in zip(cells, segments)
        τ += sigma_eff[mat_ids[c]] * ds
    end
    return τ
end

# ---------------------------------------------------------------------------
# Lagrange (vertex) -> normalized tensor Legendre projection
# ---------------------------------------------------------------------------

"""
    _project_lagrange_to_legendre(vertex_values, 𝒪x, 𝒪y, 𝒪z)

Project the 8 cell-vertex values (ordered `idx = 1 + a + 2b + 4c`, with `a,b,c ∈ {0,1}`
selecting `ξ,η,ζ = ∓1`) onto the normalized tensor-product Legendre basis used by the
solver. For `𝒪=1` this reduces to the 8-vertex average; the normalization factor
`√(2i-1)` converts the standard Legendre coefficient `s/8` to Radiant's `P̃_l = √(2l+1) P_l`.
"""
function _project_lagrange_to_legendre(vertex_values::AbstractVector{<:Real},
                                       𝒪x::Integer, 𝒪y::Integer, 𝒪z::Integer)
    moments = zeros(𝒪x, 𝒪y, 𝒪z)
    for ix in 1:𝒪x, iy in 1:𝒪y, iz in 1:𝒪z
        s = 0.0
        for a in 0:1, b in 0:1, c in 0:1
            idx = 1 + a + 2b + 4c
            sign_x = (ix == 1) ? 1.0 : ((a == 1) ? 1.0 : -1.0)
            sign_y = (iy == 1) ? 1.0 : ((b == 1) ? 1.0 : -1.0)
            sign_z = (iz == 1) ? 1.0 : ((c == 1) ? 1.0 : -1.0)
            s += sign_x * sign_y * sign_z * vertex_values[idx]
        end
        norm_x = sqrt(2 * ix - 1)
        norm_y = sqrt(2 * iy - 1)
        norm_z = sqrt(2 * iz - 1)
        moments[ix, iy, iz] = s / (8.0 * norm_x * norm_y * norm_z)
    end
    return moments
end

# ---------------------------------------------------------------------------
# Moment index (must match map_moments.jl conventions)
# ---------------------------------------------------------------------------

"""
    moment_index(ix, iy, iz, iE, 𝒪, isFC)

Flat moment index for spatial orders `(ix,iy,iz)` and energy order `iE`, matching the
fully-coupled / restricted indexing used in `map_moments.jl`.
"""
function moment_index(ix::Integer, iy::Integer, iz::Integer, iE::Integer,
                      𝒪::AbstractVector{<:Integer}, isFC::Bool)
    if isFC
        return 𝒪[2]*𝒪[1]*𝒪[4]*(iz-1) + 𝒪[1]*𝒪[4]*(iy-1) + 𝒪[4]*(ix-1) + iE
    else
        if (ix > 1 && (iy > 1 || iz > 1 || iE > 1)) ||
           (iy > 1 && (ix > 1 || iz > 1 || iE > 1)) ||
           (iz > 1 && (ix > 1 || iy > 1 || iE > 1)) ||
           (iE > 1 && (ix > 1 || iy > 1 || iz > 1))
            error("restricted scheme only allows one high-order axis.")
        end
        i = 1 + (iE-1) + (ix-1) + (iy-1) + (iz-1)
        if ix > 1; i += 𝒪[4]-1; end
        if iy > 1; i += 𝒪[4]-1 + 𝒪[1]-1; end
        if iz > 1; i += 𝒪[4]-1 + 𝒪[1]-1 + 𝒪[2]-1; end
        return i
    end
end

# ---------------------------------------------------------------------------
# Direction -> SN angular basis vector
# ---------------------------------------------------------------------------

"""
    _sn_basis_vector(Ω, sn_angle_discre)

Evaluate the solver angular basis at direction `Ω = (μ,η,ξ)`, returning a length-`Np`
vector. Reuses the same `real_spherical_harmonics_up_to_L` + `pl/pm` remapping as
`_beam_basis_coefficients`, so it matches the solver basis ordering for both `"standard"`
and `"galerkin-d"`. At a quadrature direction `Ω[:,n]` it equals `Dn[:,n] ./ w[n]`.
"""
function _sn_basis_vector(Ω::AbstractVector{<:Real},
                          sn_angle_discre::SN_Angular_Discretization)
    μs, ηs, ξs = Ω[1], Ω[2], Ω[3]
    ϕs = atan(ξs, ηs)
    Lmax = maximum(sn_angle_discre.pl)
    if sn_angle_discre.Qdims == 1
        Pls = legendre_polynomials_up_to_L(Lmax, μs)
        return [Pls[sn_angle_discre.pl[p]+1] for p in 1:sn_angle_discre.Np]
    else
        Ylms = real_spherical_harmonics_up_to_L(Lmax, μs, ϕs)
        return [Ylms[sn_angle_discre.pl[p]+1][sn_angle_discre.pm[p]+sn_angle_discre.pl[p]+1]
                for p in 1:sn_angle_discre.Np]
    end
end

# ---------------------------------------------------------------------------
# BTE path A: tau cache + vertex value computation + spatial projection
# ---------------------------------------------------------------------------

"""
    build_point_source_trace_cache(point_sources, geometry, cross_sections, solver)

Pre-allocate a `PointSourceTraceCache` and fill `tau[ipoint, ig, i, j, k]` with the optical
depth from each point source to each grid vertex (BTE uses total cross-section; FP/CSD use
absorption, matching §0.5/§4.9). The `rays` field is allocated but only populated by the
BFP path B.
"""
function build_point_source_trace_cache(point_sources::Vector{Point_Source},
                                        geometry::Geometry,
                                        cross_sections::Cross_Sections,
                                        solver::SN)
    x_edges = geometry.get_voxels_boundaries("x")
    y_edges = geometry.get_voxels_boundaries("y")
    z_edges = geometry.get_voxels_boundaries("z")
    mat_ids = geometry.get_material_per_voxel()
    Ns = geometry.get_number_of_voxels()
    Nx, Ny, Nz = Ns[1], Ns[2], Ns[3]
    Npoints = length(point_sources)

    part = solver.get_particle()
    Ng = cross_sections.get_number_of_groups(part)
    solver_type, _ = solver.get_solver_type()
    Σt = (solver_type ∈ (4, 5)) ? cross_sections.get_absorption(part) : cross_sections.get_total(part)

    tau = zeros(Npoints, Ng, Nx + 1, Ny + 1, Nz + 1)
    rays = Array{Tuple{Vector{CartesianIndex{3}},Vector{Float64}},4}(undef,
                                                                     Npoints, Nx + 1, Ny + 1, Nz + 1)
    for p in 1:Npoints
        src = point_sources[p].position
        for k in 1:Nz+1, j in 1:Ny+1, i in 1:Nx+1
            tgt = [x_edges[i], y_edges[j], z_edges[k]]
            for g in 1:Ng
                tau[p, g, i, j, k] = dda_optical_depth(src, tgt, x_edges, y_edges, z_edges,
                                                       mat_ids, @view(Σt[g, :]))
            end
        end
    end
    return PointSourceTraceCache(tau, rays)
end

"""
    uncollided_volume_source_sn!(φ_u, g, cache, sources, geometry, sn_angle_discre, solver, adjoint)

Accumulate the BTE point-source uncollided flux moments for energy group `g` into `φ_u`.
Each grid vertex receives the analytic uncollided angular flux projected onto the SN basis
(`Y·(1/r²)·q·(1/2π)·f(μ)·exp(-τ)`); each cell then projects its 8 vertex values onto the
normalized tensor Legendre spatial moments. The vertex/point loops run serially because
multiple sources accumulate into the same vertex.
"""
function uncollided_volume_source_sn!(φ_u::Array{Float64,6}, g::Integer,
                                      cache::PointSourceTraceCache,
                                      sources::Vector{Point_Source}, geometry::Geometry,
                                      sn_angle_discre::SN_Angular_Discretization,
                                      solver::SN, adjoint::Bool)
    Ns = geometry.get_number_of_voxels()
    Nx, Ny, Nz = Ns[1], Ns[2], Ns[3]
    Np = sn_angle_discre.Np
    _, 𝒪, _ = solver.get_schemes(geometry, solver.get_is_full_coupling())
    isFC = solver.get_is_full_coupling()
    x_edges = geometry.get_voxels_boundaries("x")
    y_edges = geometry.get_voxels_boundaries("y")
    z_edges = geometry.get_voxels_boundaries("z")
    ε = 1e-6 * minimum(vcat(geometry.get_voxels_width()...))
    Npoints = length(sources)

    phi_vertex = zeros(Np, Nx + 1, Ny + 1, Nz + 1)
    for k in 1:Nz+1, j in 1:Ny+1, i in 1:Nx+1
        rv = [x_edges[i], y_edges[j], z_edges[k]]
        for p in 1:Npoints
            ps = sources[p]
            dvec = rv .- ps.position
            dist = max(norm(dvec), ε)
            dinv = 1.0 / dist
            omega = adjoint ? -dvec .* dinv : dvec .* dinv
            fmu = evaluate(ps.angular_distribution, dot(omega, ps.reference_direction))
            fmu <= 0 && continue
            qg = ps.intensity * ps.energy_spectrum[g]
            Y = _sn_basis_vector(omega, sn_angle_discre)
            phi_vertex[:, i, j, k] .+= Y .* (dinv^2) .* qg .* (1 / (2π)) .* fmu .* exp(-cache.tau[p, g, i, j, k])
        end
    end

    for kc in 1:Nz, jc in 1:Ny, ic in 1:Nx
        vertex_vals = Vector{Vector{Float64}}(undef, 8)
        for a in 0:1, b in 0:1, c in 0:1
            idx = 1 + a + 2b + 4c
            vertex_vals[idx] = phi_vertex[:, ic + a, jc + b, kc + c]
        end
        for p in 1:Np
            vals = [vertex_vals[idx][p] for idx in 1:8]
            moments = _project_lagrange_to_legendre(vals, 𝒪[1], 𝒪[2], 𝒪[3])
            for ix in 1:𝒪[1], iy in 1:𝒪[2], iz in 1:𝒪[3]
                if !isFC
                    n_high = (ix > 1) + (iy > 1) + (iz > 1)
                    n_high > 1 && continue
                end
                is = moment_index(ix, iy, iz, 1, 𝒪, isFC)
                φ_u[g, p, is, ic, jc, kc] += moments[ix, iy, iz]
            end
        end
    end
    return φ_u
end

"""
    _compute_uncollided_point_flux_bte(cache, cross_sections, geometry, solver, source, sn_angle_discre)

Compute the BTE point-source uncollided flux moments `φ_u`, shape
`(Ng, Np, Nm[5], Nx, Ny, Nz)`, by accumulating per-group vertex values and projecting onto
the cell spatial moments (path A).
"""
function _compute_uncollided_point_flux_bte(cache::PointSourceTraceCache,
                                            cross_sections::Cross_Sections,
                                            geometry::Geometry, solver::SN, source::Source,
                                            sn_angle_discre::SN_Angular_Discretization)
    part = solver.get_particle()
    Ng = cross_sections.get_number_of_groups(part)
    Ns = geometry.get_number_of_voxels()
    _, 𝒪, Nm = solver.get_schemes(geometry, solver.get_is_full_coupling())
    @assert all(𝒪[1:3] .∈ Ref((1, 2))) "Point-source path A supports spatial order 1 or 2."
    Np = sn_angle_discre.Np
    φ_u = zeros(Ng, Np, Nm[5], Ns[1], Ns[2], Ns[3])
    for g in 1:Ng
        uncollided_volume_source_sn!(φ_u, g, cache, source.point_sources, geometry,
                                     sn_angle_discre, solver, false)
    end
    return φ_u
end

# ---------------------------------------------------------------------------
# BFP path B: CSD ray stepping + vertex value computation + spatial projection
# ---------------------------------------------------------------------------

"""
    _ray_cell_csd_step!(Σt_eff, β1, β2, ds, ΦE_in, phi_prev_out)

One CSD per-segment energy step, reusing the 1D MOC kernels (`_build_A_matrix`,
`_build_boundary_source`, `_ray_cell_kernel_1D`). The CSD inter-group source
`phi_prev_out` (upstream group's lower energy-edge outgoing flux) is treated as a constant
flat source within the segment (Layer 3 approximation). The `1/r²` geometric factor is NOT
applied here; it is multiplied in only when writing the vertex value.
"""
function _ray_cell_csd_step!(Σt_eff::Float64, β1::Float64, β2::Float64,
                             ds::Float64, ΦE_in::Vector{Float64},
                             phi_prev_out::Float64)
    A = _build_A_matrix(Σt_eff, β1, β2)
    q_eff = _build_boundary_source(β1, β2, phi_prev_out; scheme_E="DG")
    ΦE_out, ΦE_int = _ray_cell_kernel_1D(A, ds, ΦE_in, q_eff)
    return ΦE_out, ΦE_int
end

"""
    _ray_cell_csd_step_ox2!(Σt_eff, β1, β2, ds, ΦE_in, phiE_prev)

One CSD per-segment energy step with the linear (polynomial) interface source — the same
treatment as the `ray_sweep_1D` 𝒪x=2 branch (`ray_sweep_1D.jl:126-138`). `phiE_prev`
holds the upstream group's lower energy-edge flux spatial moments `[average, slope]`
within the segment (its within-segment linear profile), replacing the flat segment-average
source of `_ray_cell_csd_step!`. Returns the outgoing energy coefficients `ΦE_out` and
this group's own lower energy-edge flux spatial moments (for the next group in the
cascade). The `1/r²` geometric factor is NOT applied here.

Zero-length segments take the exact ds→0 limit (`ΦE_in` passes through unchanged) via a
short-circuit: besides being correct, this avoids the unused NaN/Inf intermediates that
`ray_moc_Lmp_Ox2` would otherwise form at Δs = 0 (`F/Δs`, `Ainv/Δs`).
"""
function _ray_cell_csd_step_ox2!(Σt_eff::Float64, β1::Float64, β2::Float64,
                                 ds::Float64, ΦE_in::Vector{Float64},
                                 phiE_prev::Vector{Float64})
    if ds == 0.0
        return copy(ΦE_in), [ΦE_in[1] - sqrt(3.0) * ΦE_in[2], 0.0]
    end
    A = _build_A_matrix(Σt_eff, β1, β2)
    E, F, Ainv, _ = ray_moc_matrices_2x2(A, ds)
    B = A * ds
    beta_up = β1 + sqrt(3.0) * β2
    ΦE_out = ray_outgoing_flux_Ox2(E, F, Ainv, B, ds, ΦE_in, phiE_prev, beta_up, 1.0)
    M = ray_moments_Ox2(E, F, Ainv, B, ds, ΦE_in, phiE_prev, beta_up, 1.0)
    phiE_new = [M[1, 1] - sqrt(3.0) * M[2, 1], M[1, 2] - sqrt(3.0) * M[2, 2]]
    return ΦE_out, phiE_new
end

"""
    ray_contribution_to_vertex!(phi_vertex, cache, source, target, ipoint, i, j, k,
                                geometry, cross_sections, solver, sn_angle_discre,
                                Σtot, S, ΔE, ray_order; s_min=nothing)

Compute the BFP point-source uncollided energy moments at grid vertex `(i,j,k)` for one
point source, marching CSD groups from high to low along the source→vertex ray. The vertex
value is the ray-endpoint outgoing energy moment projected onto the SN basis and scaled by
the `1/s²` geometric factor evaluated at the true vertex distance `s_total`. Writes
`phi_vertex[g, p, :, i, j, k]` and caches the ray geometry in `cache.rays[ipoint, i, j, k]`.
`ray_order` selects the CSD inter-group coupling scheme: 1 = flat segment-average source
(legacy), 2 = linear (polynomial) interface source.
"""
function ray_contribution_to_vertex!(phi_vertex::Array{Float64,6},
                                     cache::PointSourceTraceCache,
                                     source::Point_Source, target,
                                     ipoint::Int64, i::Int64, j::Int64, k::Int64,
                                     geometry::Geometry, cross_sections::Cross_Sections,
                                     solver::SN, sn_angle_discre::SN_Angular_Discretization,
                                     Σtot, S, ΔE, ray_order::Int64;
                                     s_min::Union{Float64,Nothing}=nothing)
    x_edges = geometry.get_voxels_boundaries("x")
    y_edges = geometry.get_voxels_boundaries("y")
    z_edges = geometry.get_voxels_boundaries("z")
    mat_ids = geometry.get_material_per_voxel()
    Ng = cross_sections.get_number_of_groups(source.get_particle())
    Np = sn_angle_discre.Np

    @assert all(ΔE .> 0.0) "Path B assumes group index 1 is the highest energy (ΔE > 0)."

    cells, segments = dda_trace_cells(source.position, target, x_edges, y_edges, z_edges, mat_ids)
    cache.rays[ipoint, i, j, k] = (cells, segments)

    # CSD intra-segment sub-stepping: split every nonzero DDA segment into nsub equal
    # sub-segments. Material is constant within a segment, so no extra ray tracing is
    # needed; nsub = 1 (default) leaves the original segment list untouched.
    nsub = FCS_POINT_SOURCE_CSD_SUBSTEPS[]
    if nsub > 1
        wcells = CartesianIndex{3}[]
        wds = Float64[]
        for (c, ds) in zip(cells, segments)
            if ds > 0
                for _ in 1:nsub
                    push!(wcells, c)
                    push!(wds, ds / nsub)
                end
            else
                push!(wcells, c)
                push!(wds, ds)
            end
        end
    else
        wcells = cells
        wds = segments
    end
    nseg = length(wds)

    Ω = (target .- source.position) ./ norm(target .- source.position)
    fmu = evaluate(source.angular_distribution, dot(Ω, source.reference_direction))
    Y = _sn_basis_vector(Ω, sn_angle_discre)

    _, 𝒪, _ = solver.get_schemes(geometry, solver.get_is_full_coupling())
    @assert solver.get_scheme_type("E") == "DG" "Path B requires the DG energy scheme."
    @assert 𝒪[4] == 2 "Path B requires DG energy order 2."
    @assert all(𝒪[1:3] .∈ Ref((1, 2))) "Path B supports spatial order 1 or 2."
    𝒪E = 𝒪[4]

    Δx_min = minimum(diff(x_edges))
    s_min_actual = isnothing(s_min) ? 1e-6 * Δx_min : s_min
    # True source -> vertex distance. For an external source the DDA march covers
    # only the in-grid part (vacuum outside), so sum(segments) would understate s;
    # for a zero-length in-grid path (ray enters at the vertex itself) it would
    # give s_total = 0 and a geo factor of 1/s_min^2 ~ 1e12. The 1/s^2 factor must
    # use the full geometric distance. For interior sources sum(segments) and
    # norm(...) agree up to roundoff and the ~1e-9*Δ _snap_to_interior offset
    # (the sign of the difference is not guaranteed), so the s_min floor below
    # behaves as before; do not rely on norm(...) >= sum(segments) as an invariant.
    s_total = norm(target .- source.position)
    geo_factor_last = 1.0 / max(s_total^2, s_min_actual^2)

    q_spectrum = source.energy_spectrum
    # Upstream-group coupling state per work segment, scheme dependent:
    # - ray_order == 1 (flat):   segment-average energy moments (Layer 2/3 legacy);
    # - ray_order == 2 (linear): lower energy-edge flux spatial moments [avg, slope].
    prev_avg = (ray_order == 2) ? Vector{Vector{Float64}}() : [zeros(𝒪E) for _ in 1:nseg]
    prev_phiE = (ray_order == 2) ? [zeros(2) for _ in 1:nseg] : Vector{Vector{Float64}}()

    for g in 1:Ng
        # ΔE-weighted convention: the CSD coupling state produced by group g-1 is
        # weighted by ΔE[g-1]; rescale it to ΔE[g] before use (same convention as
        # sn_flux.jl, gn_flux.jl and sn_flux_fcs.jl).
        if g > 1
            κΔ = ΔE[g] / ΔE[g-1]
            if ray_order == 2
                for v in prev_phiE; v .*= κΔ; end
            else
                for v in prev_avg;  v .*= κΔ; end
            end
        end
        if q_spectrum[g] > 0
            ΦE_in = [source.intensity * q_spectrum[g] * fmu / (2π), 0.0]
        else
            ΦE_in = zeros(𝒪E)
        end
        ΦE_out = copy(ΦE_in)
        for (iseg, (c, ds)) in enumerate(zip(wcells, wds))
            mid = mat_ids[c]
            Σt_eff = Σtot[g, mid]
            β1 = S[g, mid, 1] / ΔE[g]
            β2 = (𝒪E > 1) ? S[g, mid, 2] / ΔE[g] : 0.0
            if ray_order == 2
                # Linear interface source: the upstream group's lower energy-edge flux
                # enters with its within-segment linear profile (same treatment as the
                # ray_sweep_1D 𝒪x=2 branch).
                phiE_prev = (g > 1) ? prev_phiE[iseg] : zeros(2)
                ΦE_out, prev_phiE[iseg] = _ray_cell_csd_step_ox2!(Σt_eff, β1, β2, ds, ΦE_in, phiE_prev)
            else
                # Upstream group's lower energy-edge outgoing flux (segment-average, Layer 2).
                phi_prev_out = (g > 1) ? (prev_avg[iseg][1] - sqrt(3.0) * prev_avg[iseg][2]) : 0.0
                ΦE_out, ΦE_int = _ray_cell_csd_step!(Σt_eff, β1, β2, ds, ΦE_in, phi_prev_out)
                # Zero-length segments are produced by dda_trace_cells when a ray passes
                # exactly through a grid corner/edge/face.  For ds=0 the kernel gives
                # ΦE_int=0 and ΦE_out=ΦE_in, so the spatial average is the point flux.
                if ds > 0
                    prev_avg[iseg] = ΦE_int ./ ds          # segment-average, without geo_factor
                else
                    prev_avg[iseg] = copy(ΦE_in)
                end
            end
            ΦE_in = ΦE_out
        end
        # Vertex value: ray-endpoint outgoing moment × angular basis × 1/s² geometric
        # factor.  Accumulate (not assign): multiple point sources share vertices.
        for p in 1:Np
            phi_vertex[g, p, :, i, j, k] .+= Y[p] .* ΦE_out .* geo_factor_last
        end
    end
    return phi_vertex
end

"""
    _compute_uncollided_point_flux_bfp(cache, cross_sections, geometry, solver, source,
                                       sn_angle_discre; T, λ₀)

Compute the BFP point-source uncollided flux moments `φ_u` (shape
`(Ng, Np, Nm[5], Nx, Ny, Nz)`) and cutoff moments `𝚽cutoff_u` (shape
`(Np, Nm[5], Nx, Ny, Nz)`) via path B: per-vertex CSD ray marching followed by a
Lagrange→Legendre spatial projection. The cutoff is the last group's lower energy-edge
flux averaged over each cell's 8 vertices (cell-average spatial moment only).
"""
function _compute_uncollided_point_flux_bfp(cache::PointSourceTraceCache,
                                            cross_sections::Cross_Sections,
                                            geometry::Geometry, solver::SN, source::Source,
                                            sn_angle_discre::SN_Angular_Discretization;
                                            T::Array{Float64,2}=Array{Float64}(undef, 0, 0),
                                            λ₀::Float64=0.0)
    part = solver.get_particle()
    solver_type, _ = solver.get_solver_type()
    Ns = geometry.get_number_of_voxels()
    Nx, Ny, Nz = Ns[1], Ns[2], Ns[3]
    Nmat = cross_sections.get_number_of_materials()
    Ng = cross_sections.get_number_of_groups(part)
    _, 𝒪, Nm = solver.get_schemes(geometry, solver.get_is_full_coupling())
    isFC = solver.get_is_full_coupling()
    𝒪E = 𝒪[4]
    Np = sn_angle_discre.Np
    @assert 𝒪E == 2 "Path B requires DG energy order 2."
    @assert all(𝒪[1:3] .∈ Ref((1, 2))) "Path B supports spatial order 1 or 2."
    # Only BFP/FP carry an angular Fokker-Planck operator (non-empty T); BCSD(3) does not.
    if solver_type ∈ (2, 4)
        @assert size(T, 1) > 0 && size(T, 2) > 0 "BFP/FP point-source uncollided flux requires non-empty T matrix."
    end

    # Spatial order of the CSD inter-group source on rays: fcs_ray_spatial_order 1 = flat
    # segment-average source (legacy), 2 = linear (polynomial) interface source;
    # 0 = inherit the solver's spatial order (rays have no distinguished axis, so the
    # maximum over the three axes is used). Mirrors the surface-beam validation.
    ray_order = solver.get_fcs_ray_spatial_order()
    if ray_order != 0 && ray_order ∉ (1, 2)
        error("fcs_ray_spatial_order must be 0 (inherit solver order), 1 or 2.")
    end
    if ray_order == 0
        ray_order = maximum(𝒪[1:3])
    end

    # Σtot / S / ΔE constructed to match sn_flux_fcs.jl:304-327 (BCSD's T is empty, §4.9).
    Σtot = (solver_type ∈ (4, 5)) ? cross_sections.get_absorption(part) : cross_sections.get_total(part)
    if solver_type ∈ (2, 4)
        Σtot = Σtot .+ T .* λ₀
    end
    ΔE = cross_sections.get_energy_width(part)
    Sb = cross_sections.get_boundary_stopping_powers(part)
    S⁻ = zeros(Ng, Nmat); S⁺ = zeros(Ng, Nmat)
    for n in 1:Nmat
        S⁻[:, n] = Sb[1:Ng, n]; S⁺[:, n] = Sb[2:Ng+1, n]
    end
    S = zeros(Ng, Nmat, 𝒪E)
    for n in 1:Nmat, ig in 1:Ng
        S[ig, n, 1] = (S⁻[ig, n] + S⁺[ig, n]) / 2
        if 𝒪E > 1
            S[ig, n, 2] = (S⁻[ig, n] - S⁺[ig, n]) / (2 * sqrt(3))
        end
    end

    x_edges = geometry.get_voxels_boundaries("x")
    y_edges = geometry.get_voxels_boundaries("y")
    z_edges = geometry.get_voxels_boundaries("z")
    Npoints = length(source.point_sources)

    phi_vertex = zeros(Ng, Np, 𝒪E, Nx + 1, Ny + 1, Nz + 1)
    for p in 1:Npoints
        ps = source.point_sources[p]
        for k in 1:Nz+1, j in 1:Ny+1, i in 1:Nx+1
            target = [x_edges[i], y_edges[j], z_edges[k]]
            ray_contribution_to_vertex!(phi_vertex, cache, ps, target, p, i, j, k,
                                        geometry, cross_sections, solver, sn_angle_discre,
                                        Σtot, S, ΔE, ray_order)
        end
    end

    φ_u = zeros(Ng, Np, Nm[5], Nx, Ny, Nz)
    𝚽cutoff_u = zeros(Np, Nm[5], Nx, Ny, Nz)
    for kc in 1:Nz, jc in 1:Ny, ic in 1:Nx
        for g in 1:Ng, p in 1:Np, iE in 1:𝒪E
            vals = Vector{Float64}(undef, 8)
            for a in 0:1, b in 0:1, c in 0:1
                idx = 1 + a + 2b + 4c
                vals[idx] = phi_vertex[g, p, iE, ic + a, jc + b, kc + c]
            end
            moments = _project_lagrange_to_legendre(vals, 𝒪[1], 𝒪[2], 𝒪[3])
            for ix in 1:𝒪[1], iy in 1:𝒪[2], iz in 1:𝒪[3]
                if !isFC
                    n_high = (ix > 1) + (iy > 1) + (iz > 1) + (iE > 1)
                    n_high > 1 && continue
                end
                is = moment_index(ix, iy, iz, iE, 𝒪, isFC)
                φ_u[g, p, is, ic, jc, kc] += moments[ix, iy, iz]
            end
        end
        # Cutoff flux: last group's lower energy-edge (e=-1/2) flux, cell-averaged over the
        # 8 vertices. TODO: only the cell-average spatial moment (is=1) is filled; high-order
        # spatial cutoff moments remain zero.
        for p in 1:Np
            cutoff_acc = 0.0
            for a in 0:1, b in 0:1, c in 0:1
                iv = ic + a; jv = jc + b; kv = kc + c
                cutoff_acc += phi_vertex[Ng, p, 1, iv, jv, kv] - sqrt(3.0) * phi_vertex[Ng, p, 2, iv, jv, kv]
            end
            𝚽cutoff_u[p, 1, ic, jc, kc] += cutoff_acc / 8
        end
    end
    return φ_u, 𝚽cutoff_u
end
