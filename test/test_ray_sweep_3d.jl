using Test
using Radiant
using LinearAlgebra

# ---------------------------------------------------------------------------
# Test helpers for direct-call (tier-1, analytic) tests.
# Builds all arguments for Radiant.ray_sweep_3D for a single-material, uniform-Δ,
# void-boundary, single-basis problem. `sources` is in flux units (per transverse
# cell = I0, NOT area-scaled) so the analytic comparison is exact.
# ---------------------------------------------------------------------------
function _build_minimal_3D_inputs(; Σt_val, Ns, Δ, I0, 𝒪, loc, isCSD, β1=0.0, β2=0.0, ΔE=1.0)
    Nx, Ny, Nz = Ns
    Nm = Radiant._compute_nm_from_o(𝒪, false)          # non-FC convention
    mat = ones(Int, Nx, Ny, Nz)
    Δs  = [fill(Δ, Nx), fill(Δ, Ny), fill(Δ, Nz)]
    Σt  = [Σt_val]
    Ω   = Dict("x-"=>[1.0,0.0,0.0], "x+"=>[-1.0,0.0,0.0],
               "y-"=>[0.0,1.0,0.0], "y+"=>[0.0,-1.0,0.0],
               "z-"=>[0.0,0.0,1.0], "z+"=>[0.0,0.0,-1.0])[loc]

    P = 1; Np_surf = 1; Np_source = 1
    Mn = [1.0]; Dn = [1.0]
    Mnx⁻ = [1.0]; Mny⁻ = [1.0]; Mnz⁻ = [1.0]
    Dnx⁻ = [0.0]; Dny⁻ = [0.0]; Dnz⁻ = [0.0]            # exit-face flux not checked here

    𝚽l = zeros(P, Nm[5], Nx, Ny, Nz)
    Ql = zeros(P, Nm[5], Nx, Ny, Nz)                    # unused by MOC path

    # All 6 faces initialized; incident face set to flux unit I0 (uniform).
    faceidx = Dict("x-"=>1,"x+"=>2,"y-"=>3,"y+"=>4,"z-"=>5,"z+"=>6)[loc]
    sources = Matrix{Union{Float64,Array{Float64}}}(undef, 1, 6)
    for f in 1:6
        sz = f ∈ (1,2) ? (Ny,Nz) : f ∈ (3,4) ? (Nx,Nz) : (Nx,Ny)
        sources[1,f] = zeros(sz)
    end
    sources[1, faceidx] .= I0

    if isCSD
        @assert 𝒪[4] == 2
        S = reshape([β1, β2], 1, 2)
        S⁻ = [0.0]; S⁺ = [0.0]
        𝚽E12 = zeros(Nm[4], Nx, Ny, Nz)
    else
        S = zeros(1, max(𝒪[4],1)); S⁻ = [0.0]; S⁺ = [0.0]
        𝚽E12 = Array{Float64}(undef)                    # unused
    end

    # Placeholders (unused by MOC path)
    C = zeros(maximum(𝒪)); ω = Vector{Array{Float64}}(undef,4); for i in 1:4; ω[i]=zeros(1); end
    𝒲 = zeros(𝒪[4],𝒪[4],𝒪[4]); isAdapt = false; isFC = false

    𝚽x12⁻ = zeros(Np_surf,Nm[1],2,Ny,Nz)
    𝚽y12⁻ = zeros(Np_surf,Nm[2],2,Nx,Nz)
    𝚽z12⁻ = zeros(Np_surf,Nm[3],2,Nx,Ny)
    bc = zeros(Int, 6)

    return (; 𝚽l, Ql, Σt, mat, Ns, Δs, Ω, Mn, Dn, P, Mnx⁻, Dnx⁻, Mny⁻, Dny⁻, Mnz⁻, Dnz⁻,
            Np_surf, 𝒪, Nm, C, ω, sources, isAdapt, isCSD, ΔE, 𝚽E12, S⁻, S⁺, S, 𝒲, isFC,
            𝚽x12⁻, 𝚽y12⁻, 𝚽z12⁻, bc, Np_source)
end

function _run_ray_sweep_3D(a)
    return Radiant.ray_sweep_3D(a.𝚽l, a.Ql, a.Σt, a.mat, a.Ns, a.Δs, a.Ω, a.Mn, a.Dn, a.P,
        a.Mnx⁻, a.Dnx⁻, a.Mny⁻, a.Dny⁻, a.Mnz⁻, a.Dnz⁻, a.Np_surf, a.𝒪, a.Nm, a.C, a.ω,
        a.sources, a.isAdapt, a.isCSD, a.ΔE, a.𝚽E12, a.S⁻, a.S⁺, a.S, a.𝒲, a.isFC,
        a.𝚽x12⁻, a.𝚽y12⁻, a.𝚽z12⁻, a.bc, a.Np_source)
end

@testset "ray_sweep_3D" begin

    @testset "axis detection" begin
        @test Radiant._ray_sweep_axis([1.0,0.0,0.0]) == (1, 1.0)
        @test Radiant._ray_sweep_axis([-1.0,0.0,0.0]) == (1, -1.0)
        @test Radiant._ray_sweep_axis([0.0,1.0,0.0]) == (2, 1.0)
        @test Radiant._ray_sweep_axis([0.0,0.0,-1.0]) == (3, -1.0)
        @test_throws ErrorException Radiant._ray_sweep_axis([0.7,0.7,0.0])
        @test Radiant._is_axis_aligned([1.0,0.0,0.0])
        @test !Radiant._is_axis_aligned([0.8,0.6,0.0])   # unit vector but oblique
    end

    @testset "project_M_to_Phi3D_n (FC, x-axis)" begin
        𝒪 = [2,1,1,3]; Nm5 = prod(𝒪)
        M = reshape(collect(1.0:6.0), 3, 2)
        𝚽n = Radiant.project_M_to_Phi3D_n(M, 𝒪, true, 1, Nm5)
        for jx in 1:2, k in 1:3
            @test 𝚽n[𝒪[4]*(jx-1)+k] == M[k,jx]
        end
    end

    @testset "project_M_to_Phi3D_n (non-FC, x & y axes)" begin
        𝒪x = [2,1,1,3]; Nm5x = 1 + sum(𝒪x .- 1)
        M = reshape(collect(1.0:6.0), 3, 2)
        𝚽x = Radiant.project_M_to_Phi3D_n(M, 𝒪x, false, 1, Nm5x)
        @test 𝚽x[1]==M[1,1]; @test 𝚽x[2]==M[2,1]; @test 𝚽x[3]==M[3,1]
        @test 𝚽x[1 + (𝒪x[4]-1) + 1] == M[1,2]
        𝒪y = [1,2,1,3]; Nm5y = 1 + sum(𝒪y .- 1)
        𝚽y = Radiant.project_M_to_Phi3D_n(M, 𝒪y, false, 2, Nm5y)
        @test 𝚽y[1]==M[1,1]
        @test 𝚽y[1 + (𝒪y[4]-1) + (𝒪y[1]-1) + 1] == M[1,2]
    end

    @testset "BTE Ox=1 vs analytic exponential (x-)" begin
        Σt = 0.7; Ns = [5,2,3]; Δ = 0.3; I0 = 2.0
        a = _build_minimal_3D_inputs(; Σt_val=Σt, Ns, Δ, I0, 𝒪=[1,1,1,1], loc="x-", isCSD=false)
        𝚽l, = _run_ray_sweep_3D(a)
        for iy in 1:Ns[2], iz in 1:Ns[3], ix in 1:Ns[1]
            x0=(ix-1)*Δ; x1=ix*Δ
            Manal = I0*(exp(-Σt*x0) - exp(-Σt*x1))/(Σt*Δ)
            @test isapprox(𝚽l[1,1,ix,iy,iz], Manal; rtol=1e-12)
        end
    end

    @testset "BTE Ox=2 vs analytic moments (all axes)" begin
        # Within-cell flux Φ(τ)=phi_in·exp(-a τ), a=Σt·Δ, τ=σ/Δ∈[0,1].
        # M[1,1] = phi_in·∫₀¹ exp(-aτ)dτ
        # M[1,2] = phi_in·√3·sx·∫₀¹ (2τ-1) exp(-aτ)dτ        (P̃₂=√3·sx·(2τ-1))
        Σt = 0.9; Δ = 0.25; I0 = 1.7; a = Σt*Δ
        ∫0 = (1 - exp(-a))/a
        ∫1 = (1 - exp(-a)*(1+a))/a^2
        Mc(phi) = phi * ∫0
        Ml(phi, sx) = phi * sqrt(3.0) * sx * (2*∫1 - ∫0)
        for (loc, axis, sx) in [("x-",1,1.0), ("y-",2,1.0), ("z-",3,1.0)]
            Ns = [4,4,4]
            𝒪 = [1,1,1,1]; 𝒪[axis] = 2
            a_in = _build_minimal_3D_inputs(; Σt_val=Σt, Ns, Δ, I0, 𝒪, loc, isCSD=false)
            𝚽l, = _run_ray_sweep_3D(a_in)
            Nd = Ns[axis]
            for i in 1:Nd
                phi_in = I0*exp(-Σt*(i-1)*Δ)
                idx = ntuple(ax -> ax==axis ? i : 1, 3)  # 沿轴 i，横向取 1
                @test isapprox(𝚽l[1,1,idx...], Mc(phi_in); rtol=1e-12)
                @test isapprox(𝚽l[1,2,idx...], Ml(phi_in, sx); rtol=1e-12)
            end
        end
    end

    # Independent oracle for CSD: build the 2×2 reaction matrix by hand and use
    # LinearAlgebra.exp (different algorithm than the kernel's Cayley-Hamilton).
    # Single energy group ⇒ no downscatter source ⇒ q_eff = 0 ⇒ pure 2×2 transport.
    _A_csd(Σt, β1, β2) = begin
        β_up = β1 + sqrt(3.0)*β2; β_down = β1 - sqrt(3.0)*β2
        [Σt+β_down            -sqrt(3.0)*β_down;
         sqrt(3.0)*β_up        Σt+3.0*β_down+2.0*sqrt(3.0)*β2]
    end

    @testset "CSD Ox=1 vs LinearAlgebra.exp oracle (all axes)" begin
        Σt = 0.8; β1 = 0.3; β2 = 0.05; Δ = 0.2; I0 = 1.3
        A = _A_csd(Σt, β1, β2)
        E = exp(-A*Δ); F = A \ (Matrix{Float64}(I,2,2) - E)
        for (loc, axis) in [("x-",1), ("y-",2), ("z-",3)]
            Ns = [4,2,2]
            𝒪 = [1,1,1,2]                       # CSD: 𝒪E=2
            a_in = _build_minimal_3D_inputs(; Σt_val=Σt, Ns, Δ, I0, 𝒪, loc, isCSD=true, β1, β2)
            𝚽l, 𝚽E12out = _run_ray_sweep_3D(a_in)
            Nd = Ns[axis]
            ΦE = [I0, 0.0]                       # entrance to first cell
            for i in 1:Nd
                ΦEint = F*ΦE
                M = ΦEint ./ Δ                   # M[k,1]
                idx = ntuple(ax -> ax==axis ? i : 1, 3)
                @test isapprox(𝚽l[1,1,idx...], M[1]; rtol=1e-10)
                @test isapprox(𝚽l[1,2,idx...], M[2]; rtol=1e-10)
                @test isapprox(𝚽E12out[1,idx...], M[1]-sqrt(3.0)*M[2]; rtol=1e-10)
                ΦE = E*ΦE                        # entrance to next cell
            end
        end
    end

    @testset "CSD Ox=2 vs LinearAlgebra oracle (all axes)" begin
        Σt = 0.8; β1 = 0.3; β2 = 0.05; Δ = 0.2; I0 = 1.3; sx = 1.0
        A = _A_csd(Σt, β1, β2)
        B = A*Δ; E = exp(-B); Binv = inv(B)
        Id = Matrix{Float64}(I,2,2)
        J0 = Binv*(Id - E); J1 = Binv*(J0 - E)
        for (loc, axis) in [("x-",1), ("y-",2), ("z-",3)]
            Ns = [4,2,2]
            𝒪 = [1,1,1,2]; 𝒪[axis] = 2          # 𝒪_d=2, 𝒪E=2
            a_in = _build_minimal_3D_inputs(; Σt_val=Σt, Ns, Δ, I0, 𝒪, loc, isCSD=true, β1, β2)
            𝚽l, 𝚽E12out = _run_ray_sweep_3D(a_in)
            Nd = Ns[axis]
            ΦE = [I0, 0.0]
            for i in 1:Nd
                M1 = J0*ΦE                          # jx=1: M[:,1]
                G2 = (-sqrt(3.0)*sx)*J0 + (2*sqrt(3.0)*sx)*J1
                M2 = G2*ΦE                          # jx=2: M[:,2]
                idx = ntuple(ax -> ax==axis ? i : 1, 3)
                # non-FC 𝒪=[...2...,2]: is 1->M[1,1], 2->M[2,1], 3->M[1,2]
                @test isapprox(𝚽l[1,1,idx...], M1[1]; rtol=1e-9)
                @test isapprox(𝚽l[1,2,idx...], M1[2]; rtol=1e-9)
                @test isapprox(𝚽l[1,3,idx...], M2[1]; rtol=1e-9)
                @test isapprox(𝚽E12out[1,idx...], M1[1]-sqrt(3.0)*M1[2]; rtol=1e-9)
                @test isapprox(𝚽E12out[2,idx...], M2[1]-sqrt(3.0)*M2[2]; rtol=1e-9)
                ΦE = E*ΦE
            end
        end
    end

end
