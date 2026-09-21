"""
    scattering_source(Ql::Array{Float64},𝚽l::Array{Float64},Σs::Array{Float64},
    mat::Array{Int64},P::Int64,pl::Vector{Int64},Nm::Int64,Ns::Vector{Int64})

Compute the elastic (in-group) scattering source.

# Input Argument(s)
- `Ql::Array{Float64}`: Legendre components of the in-cell source.
- `𝚽l::Array{Float64}`: Legendre components of the in-cell flux.
- `ndims::Int64`: dimension of the geometry.
- `Σs::Array{Float64}`: Legendre moments of the scattering differential cross-sections.
- `mat::Array{Int64}`: material identifier per voxel.
- `P::Int64`: number of angular interpolation basis.
- `pl::Vector{Int64}`: legendre order associated with each interpolation basis. 
- `Nm::Int64`: number of spatial and/or energy moments.
- `Ns::Vector{Int64}`: number of voxels per axis.

# Output Argument(s)
- `Ql::Array{Float64}`: Legendre components of the in-cell source.

# Reference(s)
N/A

"""
# Serial core of the in-group scattering source for a single voxel.
@inline function _scattering_source_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,I::CartesianIndex{3})
    ix, iy, iz = Tuple(I)
    for is in range(1,Nm), p in range(1,P)
        Ql[p,is,ix,iy,iz] += Σs[mat[ix,iy,iz],pl[p]+1] * 𝚽l[p,is,ix,iy,iz]
    end
    return Ql
end

function scattering_source(Ql::AbstractArray{Float64},𝚽l::AbstractArray{Float64},Σs::AbstractArray{Float64},mat::Array{Int64},P::Int64,pl::Vector{Int64},Nm::Int64,Ns::Vector{Int64};parallel::Bool=true)
    Nxyz = Ns[1] * Ns[2] * Ns[3]
    if parallel && Threads.nthreads() > 1 && Nxyz >= PAR_MIN_NVOXELS
        @threads :static for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
            _scattering_source_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,I)
        end
    else
        for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
            _scattering_source_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,I)
        end
    end
    return Ql
end

"""
    scattering_source(Ql::Array{Float64},𝚽l::Array{Float64},Σs::Array{Float64},
    mat::Array{Int64},P::Int64,pl::Vector{Int64},Nm::Int64,Ns::Vector{Int64},Ngi::Int64,
    gf::Int64)

Compute the inelastic (out-of-group) scattering source.

# Input Argument(s)
- `Ql::Array{Float64}`: Legendre components of the in-cell source.
- `𝚽l::Array{Float64}`: Legendre components of the in-cell flux.
- `ndims::Int64`: dimension of the geometry.
- `Σs::Array{Float64}`: Legendre moments of the scattering differential cross-sections.
- `mat::Array{Int64}`: material identifier per voxel.
- `P::Int64`: number of angular interpolation basis.
- `pl::Vector{Int64}`: legendre order associated with each interpolation basis. 
- `Nm::Int64`: number of spatial and/or energy moments.
- `Ns::Vector{Int64}`: number of voxels per axis.
- `Ngi::Int64`: number of energy groups.
- `gf::Int64`: group in which the particles scatter.

# Output Argument(s)
- `Ql::Array{Float64}`: Legendre components of the in-cell source.

# Reference(s)
N/A

"""
# Serial core of the out-of-group scattering source for a single voxel. The `gi` loop
# stays inside the voxel to preserve the short-circuit when `gi == gf && !is_elastic`.
@inline function _scattering_source_out_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,Ngi,gf,is_elastic,I::CartesianIndex{3})
    ix, iy, iz = Tuple(I)
    for gi in range(1,Ngi)
        if gi != gf || is_elastic
            for is in range(1,Nm), p in range(1,P)
                Ql[p,is,ix,iy,iz] += Σs[mat[ix,iy,iz],gi,pl[p]+1] * 𝚽l[gi,p,is,ix,iy,iz]
            end
        end
    end
    return Ql
end

function scattering_source(Ql::AbstractArray{Float64},𝚽l::AbstractArray{Float64},Σs::AbstractArray{Float64},mat::Array{Int64},P::Int64,pl::Vector{Int64},Nm::Int64,Ns::Vector{Int64},Ngi::Int64,gf::Int64,is_elastic::Bool=false;parallel::Bool=true)
    Nxyz = Ns[1] * Ns[2] * Ns[3]
    if parallel && Threads.nthreads() > 1 && Nxyz >= PAR_MIN_NVOXELS
        @threads :static for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
            _scattering_source_out_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,Ngi,gf,is_elastic,I)
        end
    else
        for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
            _scattering_source_out_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,Ngi,gf,is_elastic,I)
        end
    end
    return Ql
end

"""
    particle_source(Ql::Array{Float64},𝚽l::Array{Float64},Σs::Array{Float64},
    mat::Array{Int64},P::Int64,pl::Vector{Int64},Nm::Int64,Ns::Vector{Int64},Ngi::Int64,
    Ngf::Int64)

Compute the source produced by a secondary particle.

# Input Argument(s)
- `Ql::Array{Float64}`: Legendre components of the in-cell source.
- `𝚽l::Array{Float64}`: Legendre components of the in-cell flux.
- `ndims::Int64`: dimension of the geometry.
- `Σs::Array{Float64}`: Legendre moments of the scattering differential cross-sections.
- `mat::Array{Int64}`: material identifier per voxel.
- `P::Int64`: number of angular interpolation basis.
- `pl::Vector{Int64}`: legendre order associated with each interpolation basis. 
- `Nm::Int64`: number of spatial and/or energy moments.
- `Ns::Vector{Int64}`: number of voxels per axis.
- `Ngi::Int64`: number of energy groups for the incoming particle.
- `Ngf::Int64`: number of energy groups for the outgoing particle.

# Output Argument(s)
- `Ql::Array{Float64}`: Legendre components of the in-cell source.

# Reference(s)
N/A

"""
# Serial core of the secondary-particle source over a range of outgoing groups gf.
function _particle_sources_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,Ns,Ngi,gf_range)
    for gf in gf_range, ix in range(1,Ns[1]), iy in range(1,Ns[2]), iz in range(1,Ns[3])
        for gi in range(1,Ngi), is in range(1,Nm), p in range(1,P)
            Ql[gf,p,is,ix,iy,iz] += Σs[mat[ix,iy,iz],gi,gf,pl[p]+1] * 𝚽l[gi,p,is,ix,iy,iz]
        end
    end
    return Ql
end

function particle_sources(Ql::AbstractArray{Float64},𝚽l::AbstractArray{Float64},Σs::AbstractArray{Float64},mat::Array{Int64},P::Int64,pl::Vector{Int64},Nm::Int64,Ns::Vector{Int64},Ngi::Int64,Ngf::Int64;parallel::Bool=true)
    if parallel && Threads.nthreads() > 1
        @threads :static for gf in range(1,Ngf)
            _particle_sources_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,Ns,Ngi,gf:gf)
        end
    else
        _particle_sources_impl!(Ql,𝚽l,Σs,mat,P,pl,Nm,Ns,Ngi,range(1,Ngf))
    end
    return Ql
end