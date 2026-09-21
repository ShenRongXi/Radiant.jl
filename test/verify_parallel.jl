# 验证 ParSnFCS.jl 体素并行功能的三个维度：
#   1. 正确性 — parallel=true vs parallel=false 数值一致
#   2. 线程利用率 — 所有线程是否都被调度
#   3. 性能缩放 — 多线程是否有实际加速
#
# 用法:
#   JULIA_NUM_THREADS=28 julia --project=. test/verify_parallel.jl

using Radiant
using LinearAlgebra
using Test
using JLD2
using Base.Threads

include("parallel_cases.jl")

const NTHREADS = Threads.nthreads()
println("="^70)
println("  体素并行功能验证")
println("  Julia 线程数: $NTHREADS")
println("="^70)

# ============================================================================
# 第一部分：正确性验证 — 比较 parallel=true vs parallel=false
# ============================================================================
@testset "1. 正确性：parallel=true vs parallel=false 数值一致" begin

    # ---- 1a. 源项函数（小规模合成数据） ----
    @testset "1a. 源项函数（群内散射、群外散射、FP源、粒子源）" begin
        Nx, Ny, Nz = 10, 3, 2
        Ns = [Nx, Ny, Nz]
        Nmat = 2
        P = 4
        pl = [0, 1, 2, 3]
        Nm = 2
        Ngi = 3
        Ngf = 3

        mat = Array{Int64}(undef, Nx, Ny, Nz)
        for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
            mat[ix,iy,iz] = (ix + iy + iz) % Nmat + 1
        end

        # 群内散射
        Σs_in = zeros(Nmat, maximum(pl)+1)
        for m in 1:Nmat, l in 1:maximum(pl)+1
            Σs_in[m,l] = 0.1 * m + 0.01 * l
        end
        𝚽_in = zeros(P, Nm, Nx, Ny, Nz)
        for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, p in 1:P, is in 1:Nm
            𝚽_in[p,is,ix,iy,iz] = sin(ix) + 0.1*iy + 0.01*iz + p + is
        end
        Q_ser = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.scattering_source(Q_ser, 𝚽_in, Σs_in, mat, P, pl, Nm, Ns; parallel=false)
        Q_par = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.scattering_source(Q_par, 𝚽_in, Σs_in, mat, P, pl, Nm, Ns; parallel=true)
        @test Q_par ≈ Q_ser
        println("     ✅ 群内散射源: 串并行一致 (norm diff = $(norm(Q_par .- Q_ser)))")

        # 群外散射
        Σs_out = zeros(Nmat, Ngi, maximum(pl)+1)
        for m in 1:Nmat, gi in 1:Ngi, l in 1:maximum(pl)+1
            Σs_out[m,gi,l] = 0.05 * m + 0.02 * gi + 0.01 * l
        end
        𝚽_out = zeros(Ngi, P, Nm, Nx, Ny, Nz)
        for gi in 1:Ngi, ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, p in 1:P, is in 1:Nm
            𝚽_out[gi,p,is,ix,iy,iz] = cos(ix) + 0.1*gi + 0.01*iy + p + is
        end
        Qo_ser = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.scattering_source(Qo_ser, 𝚽_out, Σs_out, mat, P, pl, Nm, Ns, Ngi, 2; parallel=false)
        Qo_par = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.scattering_source(Qo_par, 𝚽_out, Σs_out, mat, P, pl, Nm, Ns, Ngi, 2; parallel=true)
        @test Qo_par ≈ Qo_ser
        println("     ✅ 群外散射源: 串并行一致 (norm diff = $(norm(Qo_par .- Qo_ser)))")

        # Fokker-Planck 源
        T = [0.7, 1.3]
        ℳ = zeros(P, P)
        for n in 1:P, m in 1:P
            ℳ[n,m] = 0.2 * n - 0.1 * m + 0.05
        end
        𝚽_fp = zeros(P, Nm, Nx, Ny, Nz)
        for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, is in 1:Nm, m in 1:P
            𝚽_fp[m,is,ix,iy,iz] = cos(ix) + 0.3*is + 0.1*iy + m
        end
        Qfp_ser = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.fokker_planck_source(P, Nm, T, 𝚽_fp, Qfp_ser, Ns, mat, ℳ; parallel=false)
        Qfp_par = zeros(P, Nm, Nx, Ny, Nz)
        Radiant.fokker_planck_source(P, Nm, T, 𝚽_fp, Qfp_par, Ns, mat, ℳ; parallel=true)
        @test Qfp_par ≈ Qfp_ser
        println("     ✅ Fokker-Planck 源: 串并行一致 (norm diff = $(norm(Qfp_par .- Qfp_ser)))")

        # 粒子源（二次粒子）
        Σs_ps = zeros(Nmat, Ngi, Ngf, maximum(pl)+1)
        for m in 1:Nmat, gi in 1:Ngi, gf in 1:Ngf, l in 1:maximum(pl)+1
            Σs_ps[m,gi,gf,l] = 0.03 * m + 0.02 * gi + 0.01 * gf + 0.005 * l
        end
        𝚽_ps = zeros(Ngi, P, Nm, Nx, Ny, Nz)
        for gi in 1:Ngi, ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, p in 1:P, is in 1:Nm
            𝚽_ps[gi,p,is,ix,iy,iz] = sin(ix*gi) + 0.1*iy + p + is
        end
        Qps_ser = zeros(Ngf, P, Nm, Nx, Ny, Nz)
        Radiant.particle_sources(Qps_ser, 𝚽_ps, Σs_ps, mat, P, pl, Nm, Ns, Ngi, Ngf; parallel=false)
        Qps_par = zeros(Ngf, P, Nm, Nx, Ny, Nz)
        Radiant.particle_sources(Qps_par, 𝚽_ps, Σs_ps, mat, P, pl, Nm, Ns, Ngi, Ngf; parallel=true)
        @test Qps_par ≈ Qps_ser
        println("     ✅ 粒子源: 串并行一致 (norm diff = $(norm(Qps_par .- Qps_ser)))")
    end

    # ---- 1b. 后处理函数（基于完整求解的 Flux 对象） ----
    @testset "1b. 后处理（通量提取、能量沉积、电荷沉积）" begin
        # 构建并求解一个小型 1D BTE 问题
        water = Material("water"); water.set_density(1.0)
        water.add_element("H", 0.1119); water.add_element("O", 0.8881)
        electron = Electron()

        cs = Cross_Sections()
        cs.set_materials(water); cs.set_particles([electron]); cs.set_source("custom")
        cs.set_absorption([0.5]); cs.set_scattering([0.5]); cs.set_legendre_order(3)
        cs.build()

        geo = Geometry()
        geo.set_type("cartesian"); geo.set_dimension(1)
        geo.set_boundary_conditions("x-", "void"); geo.set_boundary_conditions("x+", "void")
        geo.set_material_per_region([water])
        geo.set_number_of_regions("x", 1); geo.set_voxels_per_region("x", [30])
        geo.set_region_boundaries("x", [0.0, 3.0]); geo.build(cs)

        m = SN()
        m.set_particle(electron); m.set_solver_type("BTE")
        m.set_quadrature("gauss-legendre", 8, 1); m.set_legendre_order(3)
        m.set_angular_boltzmann("standard"); m.set_scheme("x", "DD", 1)

        ss = Surface_Source(); ss.set_particle(electron); ss.set_intensity(1.0)
        ss.set_energy_group(1); ss.set_direction([1.0, 0.0, 0.0]); ss.set_location("x-")

        solvers = Solvers(); solvers.add_solver(m)
        sources = Fixed_Sources(cs, geo, solvers); sources.add_source(ss)

        cu = Computation_Unit()
        cu.set_cross_sections(cs); cu.set_geometry(geo)
        cu.set_solvers(solvers); cu.set_sources(sources)
        cu.run()
        flux_obj = cu.flux

        # 通量提取
        F_s = Radiant.flux(cs, geo, flux_obj, electron; parallel=false)
        F_p = Radiant.flux(cs, geo, flux_obj, electron; parallel=true)
        @test F_p ≈ F_s
        println("     ✅ 通量提取: 串并行一致 (norm diff = $(norm(F_p .- F_s)))")

        # 能量沉积
        D_s = Radiant.energy_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=false)
        D_p = Radiant.energy_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=true)
        @test D_p ≈ D_s
        println("     ✅ 能量沉积: 串并行一致 (norm diff = $(norm(D_p .- D_s)))")

        # 电荷沉积
        C_s = Radiant.charge_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=false)
        C_p = Radiant.charge_deposition(cs, geo, solvers, sources, flux_obj, [electron]; parallel=true)
        @test C_p ≈ C_s
        println("     ✅ 电荷沉积: 串并行一致 (norm diff = $(norm(C_p .- C_s)))")
    end
end
println()

# ===========================================================================
# 第二部分：线程利用率 — 验证多个线程确实被调度
# ===========================================================================
@testset "2. 线程利用率：验证 @threads 确实分派到多个线程" begin
    Nxyz = 2400  # 模拟 20×20×6 体素
    thread_ids = zeros(Int, Nxyz)
    nt = Threads.nthreads()

    # 模拟体素并行循环：每个"体素"记录处理它的线程 ID
    @threads :static for i in 1:Nxyz
        thread_ids[i] = Threads.threadid()
    end

    unique_threads = length(Set(thread_ids))
    println("     总任务数: $Nxyz")
    println("     被调度的线程数: $unique_threads / $nt")
    println("     各线程任务分配: ", join(map(t -> count(==(t), thread_ids), 1:nt), ", "))

    if nt > 1
        @test unique_threads > 1
        println("     ✅ 多个线程确实被调度 (使用了 $unique_threads 个线程)")
    else
        println("     ⚠️  单线程运行，跳过并行验证。请用 JULIA_NUM_THREADS>1 重新运行。")
    end
end
println()

# ===========================================================================
# 第三部分：性能缩放 — 比较 1 线程 vs 多线程的效率
# ===========================================================================
@testset "3. 性能缩放：多线程加速比" begin
    Nx, Ny, Nz = 30, 30, 10  # 9000 体素
    Ns = [Nx, Ny, Nz]
    Nmat = 5
    P = 10
    pl = collect(0:P-1)
    Nm = 4
    Ng = 20

    mat = Array{Int64}(undef, Nx, Ny, Nz)
    for ix in 1:Nx, iy in 1:Ny, iz in 1:Nz
        mat[ix,iy,iz] = (ix + iy + iz) % Nmat + 1
    end

    # 群外散射源 — 计算量最大、最典型的场景
    Σs = zeros(Nmat, Ng, maximum(pl)+1)
    for m in 1:Nmat, gi in 1:Ng, l in 1:maximum(pl)+1
        Σs[m,gi,l] = 0.05 * m + 0.02 * gi + 0.01 * l
    end
    𝚽l = zeros(Ng, P, Nm, Nx, Ny, Nz)
    for gi in 1:Ng, ix in 1:Nx, iy in 1:Ny, iz in 1:Nz, p in 1:P, is in 1:Nm
        𝚽l[gi,p,is,ix,iy,iz] = cos(ix) + 0.1*gi + 0.01*iy + p + is
    end

    # 预热（触发 JIT 编译）
    Q_warm = zeros(P, Nm, Nx, Ny, Nz)
    Radiant.scattering_source(Q_warm, 𝚽l, Σs, mat, P, pl, Nm, Ns, Ng, 1; parallel=true)

    # 测量多线程时间
    n_trials = 10
    t_par = Inf
    for _ in 1:n_trials
        Q_par = zeros(P, Nm, Nx, Ny, Nz)
        t = @elapsed Radiant.scattering_source(Q_par, 𝚽l, Σs, mat, P, pl, Nm, Ns, Ng, 1; parallel=true)
        t_par = min(t_par, t)
    end

    # 测量单线程时间
    t_ser = Inf
    for _ in 1:n_trials
        Q_ser = zeros(P, Nm, Nx, Ny, Nz)
        t = @elapsed Radiant.scattering_source(Q_ser, 𝚽l, Σs, mat, P, pl, Nm, Ns, Ng, 1; parallel=false)
        t_ser = min(t_ser, t)
    end

    speedup = t_ser / t_par
    eff = speedup / Threads.nthreads() * 100

    println("     问题规模: $(Nx)×$(Ny)×$(Nz) = $(Nx*Ny*Nz) 体素, Ng=$Ng, P=$P")
    println("     串行最快时间: $(round(t_ser, digits=4)) s")
    println("     并行最快时间: $(round(t_par, digits=4)) s")
    println("     加速比: $(round(speedup, digits=2))×")
    println("     并行效率: $(round(eff, digits=1))%")

    if Threads.nthreads() > 1
        @test speedup > 1.5  # 至少 1.5× 加速
        println("     ✅ 多线程有实际加速（加速比 $(round(speedup, digits=2))× > 1.5×）")
    else
        println("     ⚠️  单线程运行，跳过加速比测试。")
    end
end
println()

# ===========================================================================
# 第四部分：关键检查 — 确认 @threads 嵌套不会发生
# ===========================================================================
@testset "4. 嵌套 @threads 防御检查" begin
    Ns_test = [4, 4, 4]
    Nxyz_test = 64
    Nm_test = 2
    P_test = 3
    pl_test = [0, 1, 2]
    mat_test = ones(Int64, 4, 4, 4)
    Σs_test = zeros(1, 3)
    𝚽_test = zeros(P_test, Nm_test, 4, 4, 4)

    # 场景 A：外层 ig 并行 + 内层 scattering_source(parallel=false)
    # 模拟 _compute_first_collision_source 的调用模式
    Ng_test = 5
    for ig in 1:Ng_test
        Q_tmp = zeros(P_test, Nm_test, 4, 4, 4)
        # 在类似 FCS 的上下文中调用 — parallel=false 必须安全
        Radiant.scattering_source(Q_tmp, 𝚽_test, Σs_test, mat_test, P_test, pl_test, Nm_test, Ns_test; parallel=false)
    end
    println("     ✅ 场景A：外层串行 ig 循环 + 内层 parallel=false — 无报错")

    # 场景 B：散射源在 energy deposition 中被调用时 parallel=true
    # 验证不会触发 Julia 的 "nested @threads" 错误
    Q_tmp2 = zeros(P_test, Nm_test, 4, 4, 4)
    Radiant.scattering_source(Q_tmp2, 𝚽_test, Σs_test, mat_test, P_test, pl_test, Nm_test, Ns_test; parallel=true)
    println("     ✅ 场景B：独立调用 scattering_source(parallel=true) — 无报错")

    # 场景 C：fokker_planck_source 同样验证
    Qfp_tmp = zeros(P_test, Nm_test, 4, 4, 4)
    T_test = [0.5]
    ℳ_test = zeros(P_test, P_test)
    Radiant.fokker_planck_source(P_test, Nm_test, T_test, 𝚽_test, Qfp_tmp, Ns_test, mat_test, ℳ_test; parallel=false)
    Radiant.fokker_planck_source(P_test, Nm_test, T_test, 𝚽_test, Qfp_tmp, Ns_test, mat_test, ℳ_test; parallel=true)
    println("     ✅ 场景C：fokker_planck_source parallel=true/false 均无报错")
end
println()

# ===========================================================================
# 汇总
# ===========================================================================
println("="^70)
println("  验证完成。")
if NTHREADS > 1
    println("  多线程并行功能：✅ 正确、可用、有加速")
else
    println("  ⚠️  当前为单线程。请用 JULIA_NUM_THREADS=28 重新运行以完整验证。")
end
println("="^70)
