"""
    Point_Source

Structure describing an internal point source for the SN solver.

# Field(s)
- `name::Union{Missing,String}` : optional identifier.
- `particle::Union{Missing,Particle}` : emitted particle.
- `position::Vector{Float64}` : `[x, y, z]` in cm; either strictly inside the geometry
  (not coinciding with a grid vertex) or strictly outside it. For an external position
  the region outside the domain is treated as vacuum: it contributes no optical depth,
  while the 1/r² geometric attenuation uses the true source-detector distance. External
  sources require void boundary conditions on all faces. Position and boundary-condition
  validation runs only through `Fixed_Sources.build`; sources added directly via
  `Source.add_source` are not validated.
- `intensity::Float64` : source strength (total emitted particles).
- `energy_group::Union{Missing,Int64}` : emission energy group (mutually exclusive with
  `energy_spectrum`).
- `energy_spectrum::Vector{Float64}` : emission spectrum over groups (mutually exclusive
  with `energy_group`).
- `angular_distribution::AngularDistribution` : cosine distribution about
  `reference_direction` (defaults to isotropic).
- `reference_direction::Vector{Float64}` : unit reference direction for `μ`.
- `is_build::Bool` : whether `build` has been called.
"""
mutable struct Point_Source
    name::Union{Missing,String}
    particle::Union{Missing,Particle}
    position::Vector{Float64}
    intensity::Float64
    energy_group::Union{Missing,Int64}
    energy_spectrum::Vector{Float64}
    angular_distribution::AngularDistribution
    reference_direction::Vector{Float64}
    is_build::Bool

    function Point_Source()
        this = new()
        this.name = missing
        this.particle = missing
        this.position = zeros(3)
        this.intensity = 1.0
        this.energy_group = missing
        this.energy_spectrum = Vector{Float64}()
        this.angular_distribution = AngularDistribution()
        this.reference_direction = [1.0, 0.0, 0.0]
        this.is_build = false
        return this
    end
end

# --- setters ---------------------------------------------------------------

function set_particle(this::Point_Source, particle::Particle)
    this.particle = particle
    return this
end

function set_intensity(this::Point_Source, intensity::Real)
    if intensity ≤ 0 error("The intensity should be greater than 0.") end
    this.intensity = intensity
    return this
end

function set_position(this::Point_Source, position::Vector{Float64})
    if length(position) != 3 error("Point source position must have 3 components.") end
    this.position = position
    return this
end

"""
    set_energy_group(this::Point_Source, g::Int64)

Set the emission energy group, clearing the mutually-exclusive `energy_spectrum`.
"""
function set_energy_group(this::Point_Source, g::Int64)
    this.energy_group = g
    this.energy_spectrum = Vector{Float64}()
    return this
end

"""
    set_energy_spectrum(this::Point_Source, spec::Vector{Float64})

Set the emission spectrum, clearing the mutually-exclusive `energy_group`. Validation and
normalization are deferred to `build`.
"""
function set_energy_spectrum(this::Point_Source, spec::Vector{Float64})
    this.energy_spectrum = spec
    this.energy_group = missing
    return this
end

function set_angular_distribution(this::Point_Source, h::AngularDistribution)
    this.angular_distribution = h
    return this
end

"""
    set_reference_direction(this::Point_Source, dir::Vector{Float64})

Set and normalize the reference direction used to compute `μ = Ω · n`.
"""
function set_reference_direction(this::Point_Source, dir::Vector{Float64})
    n = norm(dir)
    @assert n > 0.0 "Reference direction must be non-zero."
    this.reference_direction = dir / n
    return this
end

# --- getters ---------------------------------------------------------------

get_particle(this::Point_Source) = this.particle
get_intensity(this::Point_Source) = this.intensity
get_normalization_factor(this::Point_Source) = this.intensity

# --- build -----------------------------------------------------------------

"""
    build(this::Point_Source, Ng::Int64)

Finalize the source: resolve `energy_spectrum` from either the explicit spectrum
(validated and normalized) or the energy group (one-hot), and mark `is_build = true`.
"""
function build(this::Point_Source, Ng::Int64)
    if !isempty(this.energy_spectrum)
        @assert length(this.energy_spectrum) == Ng "Point source energy spectrum length must equal Ng."
        @assert all(this.energy_spectrum .>= 0.0) "Point source energy spectrum must be non-negative."
        s = sum(this.energy_spectrum)
        if !(s ≈ 1.0)
            @warn "Point source energy spectrum sum = $s; normalizing."
            this.energy_spectrum ./= s
        end
    elseif !ismissing(this.energy_group)
        @assert 1 <= this.energy_group <= Ng "Point source energy group out of range."
        spec = zeros(Ng)
        spec[this.energy_group] = 1.0
        this.energy_spectrum = spec
    else
        error("Point source must set either energy_group or energy_spectrum.")
    end
    this.is_build = true
    return this
end
