"""
    Fixed_Sources

Structure used to define a collection of fixed sources and their properties.

# Mandatory field(s)
- `sources_list::Vector{Source}` : list of Volume_Source or Surface_Source sources.
- `cross_sections` : Cross_Sections structure.
- `geometry` : Geometry structure.
- `solvers` : Solvers structure.

# Optional field(s) - with default values
- N/A

"""
mutable struct Fixed_Sources

    # Variable(s)
    number_of_particles        ::Int64
    particles                  ::Vector{Particle}
    normalization_factor       ::Float64
    sources_names              ::Vector{String}
    sources_list               ::Vector{Source}
    cross_sections             ::Cross_Sections
    geometry                   ::Geometry
    solvers                    ::Solvers
    source_collection          ::Vector{Union{Surface_Source,Volume_Source,Point_Source}}
    is_build                   ::Bool

    # Constructor(s)
    function Fixed_Sources(cross_sections,geometry,solvers)

        this = new()

        this.number_of_particles = 0
        this.normalization_factor = 0
        this.particles = Vector{Particle}()
        this.sources_names = Vector{String}()
        this.sources_list = Vector{Source}()
        this.cross_sections = cross_sections
        this.geometry = geometry
        this.solvers = solvers
        this.source_collection = Vector{Union{Surface_Source,Volume_Source,Point_Source}}()
        this.is_build = false

        return this
    end
end

# Method(s)
"""
    add_source(this::Fixed_Sources,fixed_source::Union{Surface_Source,Volume_Source,Point_Source})

To add a surface, volume or point source to the Fixed_Sources structure.

# Input Argument(s)
- `this::Fixed_Sources` : collection of fixed sources.
- `fixed_source::Union{Surface_Source,Volume_Source,Point_Source}` : volume, surface or point source.

# Output Argument(s)
N/A

# Examples
```jldoctest
julia> vs = Volume_Source()
julia> ... # Define the volume source properties
julia> fs = Fixed_Sources()
julia> fs.add_source(vs)
```
"""
function add_source(this::Fixed_Sources,fixed_source::Union{Surface_Source,Volume_Source,Point_Source})
    push!(this.source_collection,deepcopy(fixed_source))
end

"""
    build(this::Fixed_Sources)

To build the fixed sources structure.

# Input Argument(s)
- `this::Fixed_Sources` : fixed sources.

# Output Argument(s)
N/A

# Examples
```jldoctest
julia> fs = Fixed_Sources()
julia> ... # Defining the fixed sources properties
julia> fs.build()
```
"""
function build(this::Fixed_Sources)
    for fixed_source in this.source_collection
        particle = fixed_source.get_particle()
        method = this.solvers.get_method(particle)
        if fixed_source isa Point_Source
            cs_index = findfirst(x -> get_tag(x) == get_tag(particle), this.cross_sections.particles)
            if isnothing(cs_index) error(string("No cross sections available for ",particle," particle.")) end
            Ng = this.cross_sections.number_of_groups[cs_index]
            fixed_source.build(Ng)
            _validate_point_source_position(fixed_source, this.geometry)
        end
        if get_tag(particle) ∈ get_tag.(this.particles)
            source = Source(particle,this.cross_sections,this.geometry,method)
            source.add_source(fixed_source)
            index = findfirst(x -> get_tag(x) == get_tag(particle),this.particles)
            this.sources_list[index] += source
        else
            this.number_of_particles += 1
            push!(this.particles,particle)
            source = Source(particle,this.cross_sections,this.geometry,method)
            source.add_source(fixed_source)
            push!(this.sources_list,source)
        end
        this.normalization_factor += fixed_source.get_normalization_factor()
    end
end

"""
    _is_near_vertex(pos, x_edges, y_edges, z_edges, tol)

Return `true` when `pos` coincides (within `tol`) with a grid vertex, i.e. it is within
`tol` of an edge plane along all three axes simultaneously.
"""
function _is_near_vertex(pos, x_edges, y_edges, z_edges, tol)
    return any(abs(pos[1] - xv) < tol for xv in x_edges) &&
           any(abs(pos[2] - yv) < tol for yv in y_edges) &&
           any(abs(pos[3] - zv) < tol for zv in z_edges)
end

"""
    _validate_point_source_position(ps::Point_Source, geometry::Geometry)

Validate a point-source position for a 3D Cartesian grid. Two admissible cases:

  * strictly inside the grid (outside the `1e-9 * min(Δx_min, Δy_min, Δz_min)`
    boundary tolerance band on every axis): must not coincide with a grid vertex;
  * strictly outside the grid (beyond the tolerance band on at least one axis):
    the region outside the domain is treated as vacuum, which is only consistent
    with void (non-re-entrant) boundary conditions on all six faces.

Positions inside the tolerance band are ambiguous and rejected — unless another
axis is strictly outside, in which case the source is external and the band
semantics do not apply (the ray-box entry computation does not depend on the
other coordinates). Note this check runs only through `Fixed_Sources.build`;
sources added directly via `Source.add_source` are not validated.
"""
function _validate_point_source_position(ps::Point_Source, geometry::Geometry)
    @assert geometry.get_dimension() == 3 "Point sources require a 3D geometry."
    x_edges = geometry.get_voxels_boundaries("x")
    y_edges = geometry.get_voxels_boundaries("y")
    z_edges = geometry.get_voxels_boundaries("z")
    Δx_min = minimum(diff(x_edges))
    Δy_min = minimum(diff(y_edges))
    Δz_min = minimum(diff(z_edges))
    tol = 1e-9 * min(Δx_min, Δy_min, Δz_min)
    pos = ps.position
    inside_x = x_edges[1] + tol < pos[1] < x_edges[end] - tol
    inside_y = y_edges[1] + tol < pos[2] < y_edges[end] - tol
    inside_z = z_edges[1] + tol < pos[3] < z_edges[end] - tol
    if inside_x && inside_y && inside_z
        @assert !_is_near_vertex(pos, x_edges, y_edges, z_edges, tol) "Point source must not coincide with a grid vertex."
    else
        strictly_outside = (pos[1] < x_edges[1] - tol || pos[1] > x_edges[end] + tol ||
                            pos[2] < y_edges[1] - tol || pos[2] > y_edges[end] + tol ||
                            pos[3] < z_edges[1] - tol || pos[3] > z_edges[end] + tol)
        @assert strictly_outside "Point source too close to the boundary (within tolerance band); place it strictly inside or strictly outside the geometry."
        # Vacuum outside the domain is only consistent with void boundaries. This
        # is a hard failure, unlike the FCS path's soft fallback for non-void BCs
        # (sn_flux_fcs.jl): an external source with re-entrant boundaries is a
        # modeling error, not a solver capability limitation.
        bcs = geometry.get_boundary_conditions()
        @assert all(bcs .== 0) "External point sources assume vacuum outside the domain and require void boundary conditions on all faces."
    end
end

"""
    get_source(this::Fixed_Sources,particle::Particle)

Get the sources for a given particle.

# Input Argument(s)
- `this::Fixed_Sources` : collection of fixed sources.
- `particle::Particle` : particle.

# Output Argument(s)
- `source::Source` : sources for the given particle.

"""
function get_source(this::Fixed_Sources,particle::Particle)
    index = findfirst(x -> get_tag(x) == get_tag(particle),this.particles)
    method = this.solvers.get_method(particle)
    if isnothing(index)
        source = Source(particle,this.cross_sections,this.geometry,method)
        return source
    else
        return this.sources_list[index]
    end
end

"""
    get_particles(this::Fixed_Sources)

Get the particle list.

# Input Argument(s)
- `this::Fixed_Sources` : collection of fixed sources.

# Output Argument(s)
- `particles::Vector{Particle}` : particle list.

"""
function get_particles(this::Fixed_Sources)
    return this.particles
end

"""
    get_normalization_factor(this::Fixed_Sources)

Get the normalization factor.

# Input Argument(s)
- `this::Fixed_Sources` : collection of fixed sources.

# Output Argument(s)
- `normalization_factor::Float64` : normalization factor.

"""
function get_normalization_factor(this::Fixed_Sources)
    return this.normalization_factor
end