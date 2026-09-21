# Generate single-threaded (serial) reference fluxes for the SN parallel regression.
# Run once with a single Julia thread:
#     JULIA_NUM_THREADS=1 julia --project=. test/generate_parallel_references.jl
# The resulting test/data/*.jld2 files are committed and loaded by test_parallel.jl.

using JLD2

include("parallel_cases.jl")

if Threads.nthreads() != 1
    error("Generate references with JULIA_NUM_THREADS=1 so they capture the serial path.")
end

data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

println("Generating 1D serial reference...")
jldsave(joinpath(data_dir, "sn_1d_serial_reference.jld2"); flux = run_sn_1d_case())
println("Generating 2D serial reference...")
jldsave(joinpath(data_dir, "sn_2d_serial_reference.jld2"); flux = run_sn_2d_case())
println("Generating 3D serial reference...")
jldsave(joinpath(data_dir, "sn_3d_serial_reference.jld2"); flux = run_sn_3d_case())

println("Generating 1D BFP serial reference...")
jldsave(joinpath(data_dir, "bfp_1d_serial_reference.jld2"); flux = run_bfp_1d_case())
println("Generating 2D BFP serial reference...")
jldsave(joinpath(data_dir, "bfp_2d_serial_reference.jld2"); flux = run_bfp_2d_case())
println("Generating 3D BFP serial reference...")
jldsave(joinpath(data_dir, "bfp_3d_serial_reference.jld2"); flux = run_bfp_3d_case())
println("Done. References written to ", data_dir)
