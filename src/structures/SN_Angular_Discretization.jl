"""
    SN_Angular_Discretization

Structure used to store the angular discretization data associated with a SN solver
and a geometry. Built once by `build_angular_discretization` and reused by both the
standard `compute_flux` path and the FCS path.

# Field(s)
- `Ω::Vector{Vector{Float64}}` : director cosines.
- `w::Vector{Float64}` : quadrature weights.
- `Nd::Int64` : number of discrete ordinates.
- `Np::Int64` : number of full-range angular interpolation basis.
- `Mn::Array{Float64,2}` : moment-to-discrete matrix.
- `Dn::Array{Float64,2}` : discrete-to-moment matrix.
- `pl::Vector{Int64}` : legendre order associated with each interpolation basis.
- `pm::Vector{Int64}` : spherical harmonics order associated with each interpolation basis.
- `Np_surf::Int64` : number of half-range angular interpolation basis for surfaces.
- `Mn_surf::Vector{Array{Float64}}` : moment-to-discrete matrix for each surface.
- `Dn_surf::Vector{Array{Float64}}` : discrete-to-moment matrix for each surface.
- `n⁺_to_n::Vector{Vector{Int64}}` : mapping from half-range to full-range indices.
- `n_to_n⁺::Vector{Vector{Int64}}` : mapping from full-range to half-range indices.
- `sn_type::String` : angular Boltzmann discretization ("standard" / "galerkin-m" / "galerkin-d").
- `Qdims::Int64` : quadrature dimension (1 / 2 / 3).

"""
mutable struct SN_Angular_Discretization

    # Variable(s)
    Ω                          ::Vector{Vector{Float64}}
    w                          ::Vector{Float64}
    Nd                         ::Int64
    Np                         ::Int64
    Mn                         ::Array{Float64,2}
    Dn                         ::Array{Float64,2}
    pl                         ::Vector{Int64}
    pm                         ::Vector{Int64}
    Np_surf                    ::Int64
    Mn_surf                    ::Vector{Array{Float64}}
    Dn_surf                    ::Vector{Array{Float64}}
    n⁺_to_n                    ::Vector{Vector{Int64}}
    n_to_n⁺                    ::Vector{Vector{Int64}}
    sn_type                    ::String
    Qdims                      ::Int64

    # Constructor(s)
    function SN_Angular_Discretization()
        this = new()
        this.Ω = Vector{Vector{Float64}}()
        this.w = Vector{Float64}()
        this.Nd = 0
        this.Np = 0
        this.Mn = Array{Float64}(undef,0,0)
        this.Dn = Array{Float64}(undef,0,0)
        this.pl = Vector{Int64}()
        this.pm = Vector{Int64}()
        this.Np_surf = 0
        this.Mn_surf = Vector{Array{Float64}}()
        this.Dn_surf = Vector{Array{Float64}}()
        this.n⁺_to_n = Vector{Vector{Int64}}()
        this.n_to_n⁺ = Vector{Vector{Int64}}()
        this.sn_type = ""
        this.Qdims = 0
        return this
    end

    function SN_Angular_Discretization(
        Ω::Vector{Vector{Float64}},
        w::Vector{Float64},
        Nd::Int64,
        Np::Int64,
        Mn::Array{Float64,2},
        Dn::Array{Float64,2},
        pl::Vector{Int64},
        pm::Vector{Int64},
        Np_surf::Int64,
        Mn_surf::Vector{Array{Float64}},
        Dn_surf::Vector{Array{Float64}},
        n⁺_to_n::Vector{Vector{Int64}},
        n_to_n⁺::Vector{Vector{Int64}},
        sn_type::String,
        Qdims::Int64)

        this = new()
        this.Ω = Ω
        this.w = w
        this.Nd = Nd
        this.Np = Np
        this.Mn = Mn
        this.Dn = Dn
        this.pl = pl
        this.pm = pm
        this.Np_surf = Np_surf
        this.Mn_surf = Mn_surf
        this.Dn_surf = Dn_surf
        this.n⁺_to_n = n⁺_to_n
        this.n_to_n⁺ = n_to_n⁺
        this.sn_type = sn_type
        this.Qdims = Qdims
        return this
    end
end
