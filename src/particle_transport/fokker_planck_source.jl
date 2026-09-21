"""
    FP_source(N::Int64,P::Int64,Nm::Int64,T::Vector{Float64},𝚽l::Array{Float64},
    Ql::Array{Float64},Ns::Vector{Int64},mat::Array{Int64,3},ℳ::Array{Float64,2},
    Mn::Array{Float64,2},Dn::Array{Float64,2})

Calculate the angular Fokker-Planck source term in Cartesian geometry.

# Input Argument(s)
- `P::Int64`: number of angular interpolation basis.
- `Nm::Int64`: total number of spatial and energy moments.
- `T::Vector{Float64}`: restricted momentum transfer.
- `𝚽l::Array{Float64}`: Legendre components of the in-cell flux.
- `Ql::Array{Float64}`: Legendre components of the in-cell source.
- `Ns::Vector{Int64}`: number of voxels per axis.    
- `mat::Array{Int64,3}`: material identifier per voxel.
- `ℳ::Array{Float64,2}`: Fokker-Planck scattering matrix.

# Output Argument(s)
- `Ql::Array{Float64}`: Legendre components of the in-cell source.

# Reference(s)
- Morel (1988) : A Hybrid Collocation-Galerkin-Sn Method for Solving the Boltzmann
  Transport Equation.

"""
function fokker_planck_source(P::Int64,Nm::Int64,T::Vector{Float64},𝚽l::AbstractArray{Float64},Ql::AbstractArray{Float64},Ns::Vector{Int64},mat::Array{Int64,3},ℳ::Array{Float64,2};parallel::Bool=true)
    Nxyz = Ns[1] * Ns[2] * Ns[3]
    if parallel && Threads.nthreads() > 1 && Nxyz >= PAR_MIN_NVOXELS
        @threads :static for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
            _fokker_planck_source_impl!(Ql,𝚽l,T,ℳ,mat,P,Nm,I)
        end
    else
        for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
            _fokker_planck_source_impl!(Ql,𝚽l,T,ℳ,mat,P,Nm,I)
        end
    end
    return Ql
end

# Serial core of the angular Fokker-Planck source for a single voxel. Inner loop order
# is→m→n (n innermost) so Ql[n,is,...] writes are contiguous in Julia column-major order.
@inline function _fokker_planck_source_impl!(Ql,𝚽l,T,ℳ,mat,P,Nm,I::CartesianIndex{3})
    ix, iy, iz = Tuple(I)
    for is in range(1,Nm), m in range(1,P), n in range(1,P)
        Ql[n,is,ix,iy,iz] += T[mat[ix,iy,iz]] * ℳ[n,m] * 𝚽l[m,is,ix,iy,iz]
    end
    return Ql
end