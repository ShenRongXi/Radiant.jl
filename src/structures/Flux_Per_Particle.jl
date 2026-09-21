"""
    Flux_Per_Particle

Structure used to contain flux information per particle.
Flux arrays are accumulated in-place across generations (not stored as a list).
`total_flux` and `total_flux_cutoff` start as `missing` and are initialized on
the first call to `add_flux`.

!!! warning "Thread safety"
    `add_flux` performs in-place accumulation (`.+=`) and is **not thread-safe**.
    It must be called sequentially. All current call sites in `transport.jl` are
    serial loops.
"""
mutable struct Flux_Per_Particle

    # Variable(s)
    particle                        ::Particle
    total_flux                      ::Union{Missing, Array{Float64,6}}
    total_flux_cutoff               ::Union{Missing, Array{Float64,5}}
    uncollided_flux                 ::Union{Missing, Array{Float64,6}}
    spectral_radius                 ::Vector{Vector{Float64}}

    # Constructor(s)
    function Flux_Per_Particle(particle::Particle)

        this = new()
        this.particle = particle
        this.total_flux = missing
        this.total_flux_cutoff = missing
        this.uncollided_flux = missing
        this.spectral_radius = Vector{Vector{Float64}}()

        return this
    end
end

# Method(s)
"""
    add_flux(this::Flux_Per_Particle,flux::Array{Float64,6})

Add flux solution to the accumulated total.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.
- `flux::Array{Float64,6}` : flux solution.

# Output Argument(s)
N/A

"""
function add_flux(this::Flux_Per_Particle,flux::Array{Float64,6})
    if ismissing(this.total_flux)
        this.total_flux = copy(flux)       # first generation: copy
    else
        this.total_flux .+= flux           # subsequent generations: accumulate in-place
    end
end

"""
    add_flux(this::Flux_Per_Particle,flux_per_particle::Flux_Per_Particle)

Merge flux solutions from another Flux_Per_Particle structure into this one.

!!! warning "Ownership transfer"
    `Flux.add_flux` (in Flux.jl) pushes the incoming `Flux_Per_Particle` object
    directly into its container on first encounter, then in-place modifies that
    container entry on subsequent generations. Callers should not reuse a
    `Flux_Per_Particle` after passing it to `Flux.add_flux`.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions (target, modified in-place).
- `flux_per_particle::Flux_Per_Particle` : structure to contain flux solutions (source, not modified).

# Output Argument(s)
N/A

"""
function add_flux(this::Flux_Per_Particle,flux_per_particle::Flux_Per_Particle)
    if get_tag(this.particle) != get_tag(flux_per_particle.particle) error("Flux particle don't fit.") end
    if !ismissing(flux_per_particle.total_flux)
        if ismissing(this.total_flux)
            this.total_flux = copy(flux_per_particle.total_flux)
        else
            this.total_flux .+= flux_per_particle.total_flux
        end
    end
    if !ismissing(flux_per_particle.total_flux_cutoff)
        if ismissing(this.total_flux_cutoff)
            this.total_flux_cutoff = copy(flux_per_particle.total_flux_cutoff)
        else
            this.total_flux_cutoff .+= flux_per_particle.total_flux_cutoff
        end
    end
    if !ismissing(flux_per_particle.uncollided_flux)
        if ismissing(this.uncollided_flux)
            this.uncollided_flux = copy(flux_per_particle.uncollided_flux)
        end
        # Keep the first generation uncollided flux; do NOT accumulate
    end
    # NOTE: spectral_radius is always merged regardless of whether flux data exists.
    # This is intentional — spectral_radius metadata should be transferred even when
    # the source Flux_Per_Particle has no flux arrays (e.g., early-generation particles).
    append!(this.spectral_radius, flux_per_particle.spectral_radius)
end

"""
    add_flux_cutoff(this::Flux_Per_Particle,flux_cutoff::Array{Float64,5})

Add flux at cutoff solution to the accumulated total.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.
- `flux_cutoff::Array{Float64,5}` : flux at cutoff solution.

# Output Argument(s)
N/A

"""
function add_flux_cutoff(this::Flux_Per_Particle,flux_cutoff::Array{Float64,5})
    if ismissing(this.total_flux_cutoff)
        this.total_flux_cutoff = copy(flux_cutoff)
    else
        this.total_flux_cutoff .+= flux_cutoff
    end
end

"""
    add_spectral_radius(this::Flux_Per_Particle,spectral_radius::Vector{Float64})

Add the per-energy-group in-group spectral-radius estimate of one generation to the list.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.
- `spectral_radius::Vector{Float64}` : estimated in-group spectral radius per energy group.

# Output Argument(s)
N/A

"""
function add_spectral_radius(this::Flux_Per_Particle,spectral_radius::Vector{Float64})
    push!(this.spectral_radius,spectral_radius)
end

"""
    get_spectral_radius(this::Flux_Per_Particle)

Get the per-energy-group in-group spectral-radius estimate of the last generation solved for the
particle (see the convergence-acceleration section of the documentation for its definition).

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.

# Output Argument(s)
- `spectral_radius::Vector{Float64}` : estimated in-group spectral radius per energy group.

"""
function get_spectral_radius(this::Flux_Per_Particle)
    if length(this.spectral_radius) == 0 error("No spectral-radius data available.") end
    return this.spectral_radius[end]
end

"""
    get_flux(this::Flux_Per_Particle)

Get the total accumulated flux solution for the particle.

!!! warning "Return value is an internal reference"
    The returned array is the internal accumulation buffer, NOT a copy.
    Callers must NOT mutate the returned array in-place (`.+=`, slice assignment, etc.).
    If an independent copy is needed, use `copy(get_flux(...))`.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.

# Output Argument(s)
- `total_flux::Array{Float64,6}` : accumulated flux solution (internal reference, read-only).

"""
function get_flux(this::Flux_Per_Particle)
    if ismissing(this.total_flux)
        error("No flux data available.")
    end
    return this.total_flux     # return internal reference; caller must not mutate!
end

"""
    get_flux_cutoff(this::Flux_Per_Particle)

Get the total accumulated flux solution at cutoff for the particle.

!!! warning "Return value is an internal reference"
    The returned array is the internal accumulation buffer, NOT a copy.
    Callers must NOT mutate the returned array in-place (`.+=`, slice assignment, etc.).
    If an independent copy is needed, use `copy(get_flux_cutoff(...))`.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.

# Output Argument(s)
- `total_flux_cutoff::Array{Float64,5}` : accumulated flux solution at cutoff (internal reference, read-only).

"""
function get_flux_cutoff(this::Flux_Per_Particle)
    if ismissing(this.total_flux_cutoff)
        error("No flux cutoff data available.")
    end
    return this.total_flux_cutoff  # return internal reference; caller must not mutate!
end

"""
    set_uncollided_flux(this::Flux_Per_Particle,flux::Array{Float64,6})

Store the uncollided (first-collision) flux for the particle.
This is the φ_u component computed by the FCS precursor, stored separately
from the total flux so it can be retrieved without re-computation.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.
- `flux::Array{Float64,6}` : uncollided flux array.
"""
function set_uncollided_flux(this::Flux_Per_Particle, flux::Array{Float64,6})
    this.uncollided_flux = copy(flux)
end

"""
    get_uncollided_flux(this::Flux_Per_Particle)

Get the uncollided (first-collision) flux for the particle.
Returns the φ_u component that was stored by the FCS path during transport.

!!! warning "Return value is an internal reference"
    The returned array is the internal buffer, NOT a copy.
    If an independent copy is needed, use `copy(get_uncollided_flux(...))`.

# Input Argument(s)
- `this::Flux_Per_Particle` : structure to contain flux solutions.

# Output Argument(s)
- `uncollided_flux::Array{Float64,6}` : uncollided flux (internal reference, read-only).
"""
function get_uncollided_flux(this::Flux_Per_Particle)
    if ismissing(this.uncollided_flux)
        error("No uncollided flux data available (FCS may not have been used).")
    end
    return this.uncollided_flux  # return internal reference; caller must not mutate!
end
