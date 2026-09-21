"""
    flux(cross_sections::Cross_Sections,geometry::Geometry,flux::Flux,particle::Particle)

Calculate and extract the flux solution for a given particle.

# Input Argument(s)
- `cross_sections::Cross_Sections`: cross section informations.
- `geometry::Geometry`: geometry informations.
- `flux::Flux`: flux informations.
- `particle::Particle`: particle.

# Output Argument(s)
- `F::Array{Float64}`: integrated flux [particle per cm² per s] per group and per voxel.

# Reference(s)
N/A

"""
function flux(cross_sections::Cross_Sections,geometry::Geometry,flux::Flux,particle::Particle;parallel::Bool=true)

#----
# Extract geometry data
#----
Ndims = geometry.get_dimension()
Ns = geometry.get_number_of_voxels()

#----
# Print the flux solution
#----
Ng = cross_sections.get_number_of_groups(particle)
F = zeros(Ng,Ns[1],Ns[2],Ns[3])
𝚽l = flux.get_flux(particle)
if parallel && Threads.nthreads() > 1 && Ns[1]*Ns[2]*Ns[3] >= PAR_MIN_NVOXELS
    @threads :static for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
        ix, iy, iz = Tuple(I)
        for ig in range(1,Ng)
            F[ig,ix,iy,iz] = 𝚽l[ig,1,1,ix,iy,iz]
        end
    end
else
    for I in CartesianIndices((Ns[1], Ns[2], Ns[3]))
        ix, iy, iz = Tuple(I)
        for ig in range(1,Ng)
            F[ig,ix,iy,iz] = 𝚽l[ig,1,1,ix,iy,iz]
        end
    end
end

if Ndims == 1
    return F[:,:,1,1]
elseif Ndims == 2
    return F[:,:,:,1]
elseif Ndims == 3
    return F
end

end