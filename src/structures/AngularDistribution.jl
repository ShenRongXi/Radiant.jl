"""
    AngularDistribution(edges, values)

Cosine-distribution histogram describing a point source's angular distribution relative
to its `reference_direction`.

The default `edges=[-1.0, 1.0]`, `values=[0.5]` satisfies `∫_{-1}^{1} f(μ) dμ = 1`. In the
point-source kernel `evaluate(h, μ) = 0.5` is multiplied by `1/(2π)`, giving `1/(4π)`,
i.e. an isotropic point source `q/(4π r²)`. User-supplied `values` must be normalized
(use `normalize!`).
"""
mutable struct AngularDistribution
    edges::Vector{Float64}    # length nbins+1, monotonic, spanning [-1, 1]
    values::Vector{Float64}   # length nbins
    function AngularDistribution()
        return new([-1.0, 1.0], [0.5])
    end
end

"""
    set_bins!(h::AngularDistribution, edges::Vector{Float64}, values::Vector{Float64})

Set the histogram bins. `edges` must be strictly increasing and span `[-1, 1]`; `values`
must be non-negative and have length `length(edges) - 1`.
"""
function set_bins!(h::AngularDistribution, edges::Vector{Float64}, values::Vector{Float64})
    @assert length(edges) == length(values) + 1
    @assert all(diff(edges) .> 0)
    @assert edges[1] ≈ -1.0 && edges[end] ≈ 1.0
    @assert all(values .>= 0)
    h.edges = edges
    h.values = values
    return h
end

"""
    normalize!(h::AngularDistribution)

Scale `values` so that `∫_{-1}^{1} f(μ) dμ = 1`.
"""
function normalize!(h::AngularDistribution)
    total = sum(h.values .* diff(h.edges))
    total > 0 && (h.values ./= total)
    return h
end

"""
    evaluate(h::AngularDistribution, μ::Float64)

Evaluate `f(μ)`. The upper endpoint `μ = edges[end]` is included in the last bin.
"""
function evaluate(h::AngularDistribution, μ::Float64)
    nbins = length(h.values)
    for i in 1:nbins
        if μ >= h.edges[i] && (i == nbins || μ < h.edges[i+1])
            return h.values[i]
        end
    end
    return 0.0
end
