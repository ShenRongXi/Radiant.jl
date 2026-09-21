"""
    map_moments(𝒪i::Vector{Int64},𝒪f::Vector{Int64})

Produce a map between different space/energy moments configuration of the angular flux.

# Input Argument(s)
- `𝒪i::Vector{Int64}`: incoming moment orders in space and energy.
- `𝒪f::Vector{Int64}`: outgoing moment orders in space and energy.
- `isFC_in::Bool` : boolean indicating if the high-order incoming moments are fully coupled
  or not.
- `isFC_out::Bool` : boolean indicating if the high-order outgoing moments are fully coupled
  or not.

# Output Argument(s)
- `map::Vector{Int64}`: vector of correspondance between the incoming and outgoing moments.

# Reference(s)
N/A

"""
function map_moments(𝒪i::Vector{Int64},𝒪f::Vector{Int64},isFC_in::Bool,isFC_out::Bool)

Nmi = prod(𝒪i)
Nmf = prod(𝒪f)

M = zeros(max(𝒪i[1],𝒪f[1]),max(𝒪i[2],𝒪f[2]),max(𝒪i[3],𝒪f[3]),max(𝒪i[4],𝒪f[4]))
map = zeros(Int64,Nmf)

for ix in range(1,𝒪i[1]), iy in range(1,𝒪i[2]), iz in range(1,𝒪i[3]), iE in range(1,𝒪i[4])
    if isFC_in
        i = 𝒪i[2]*𝒪i[1]*𝒪i[4]*(iz-1)+𝒪i[1]*𝒪i[4]*(iy-1)+𝒪i[4]*(ix-1)+iE
    else
        if (ix > 1 && (iy > 1 || iz > 1 || iE > 1)) || (iy > 1 && (ix > 1 || iz > 1 || iE > 1)) || (iz > 1 && (ix > 1 || iy > 1 || iE > 1)) || (iE > 1 && (ix > 1 || iy > 1 || iz > 1)) continue end
        i = 1 + (iE-1) + (ix-1) + (iy-1) + (iz-1)
        if ix > 1 i += 𝒪i[4]-1 end
        if iy > 1 i += 𝒪i[4]-1 + 𝒪i[1]-1 end
        if iz > 1 i += 𝒪i[4]-1 + 𝒪i[1]-1 + 𝒪i[2]-1 end
    end
    M[ix,iy,iz,iE] = i
end

for ix in range(1,𝒪f[1]), iy in range(1,𝒪f[2]), iz in range(1,𝒪f[3]), iE in range(1,𝒪f[4])
    if isFC_out
        i = 𝒪f[2]*𝒪f[1]*𝒪f[4]*(iz-1)+𝒪f[1]*𝒪f[4]*(iy-1)+𝒪f[4]*(ix-1)+iE
    else
        if (ix > 1 && (iy > 1 || iz > 1 || iE > 1)) || (iy > 1 && (ix > 1 || iz > 1 || iE > 1)) || (iz > 1 && (ix > 1 || iy > 1 || iE > 1)) || (iE > 1 && (ix > 1 || iy > 1 || iz > 1)) continue end
        i = 1 + (iE-1) + (ix-1) + (iy-1) + (iz-1)
        if ix > 1 i += 𝒪f[4]-1 end
        if iy > 1 i += 𝒪f[4]-1 + 𝒪f[1]-1 end
        if iz > 1 i += 𝒪f[4]-1 + 𝒪f[1]-1 + 𝒪f[2]-1 end
    end
    if ix <= 𝒪i[1] && iy <= 𝒪i[2] && iz <= 𝒪i[3] && iE <= 𝒪i[4]
        map[i] = M[ix,iy,iz,iE]
    end
end
return map
end