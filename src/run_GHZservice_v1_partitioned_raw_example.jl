using DataFrames
using JLD2
using StatsBase
const code = "Steane713"
const error_model = "depolarizing"
##
include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
##
start = time()
result = run_single_configuration(
    F_link = 0.97,
    link_success_prob = 1e-3,
    attempt_time = 1e-6,
    T_coherence = Inf,
    Δt_CNOTgate = 100e-6,
    gate_fidelity = 0.9997,
    Δt_readout = 1e-3,
    readout_fidelity = 1.0,
    Δt_rotation_shuttle = 100e-6,
    cutoff = Inf,

    seed = 1234,
    target_samples = 2000,
    max_wallclock = 60.0,
)

df_raw = result.raw_events

println("Runtime: $(time() - start) seconds")
##

if code == "Steane713"
    required_count = 2
else
    required_count = 1
end

# All stabilizer/generator indices expected
all_generators = collect(eachindex(GENERATORS))

# Work in timestep order
sort!(df_raw, :timesteps)

counts = Dict(generator => 0 for generator in all_generators)
timesteps_markers = Float64[]
for row_index in 1:nrow(df_raw)
    counts[df_raw.generator_idx[row_index]] += 1
    if all(counts[generator] >= required_count for generator in all_generators)
        push!(timesteps_markers, df_raw.timesteps[row_index])

        for generator in all_generators
            counts[generator] = 0
        end
    end
end

##
pmem = 0.04
t_max = t_cycle(pmem; T_coh = 1.0)
@info t_max
@info mean(diff(timesteps_markers))
##

include(joinpath(@__DIR__, "GHZfromtrace.jl"))
bellpair_times = Matrix{Float64}(df_raw[:, [:bellpair_1, :bellpair_2, :bellpair_3, :bellpair_4]])

start = time()
df_raw.fidelity = [
    simulate_piecemaker_trace(
        collect(row);
        T_coherence = 1.0,
        F_link = 1.0,
        t_rotation_shuttle = 0.0,
        t_CNOT = 0.1,
        F_CNOT = 1.0,
        t_readout = 0.0,
        F_readout = 1.0,
    )
    for row in eachrow(bellpair_times)
]
println("Runtime: $(time() - start) seconds")
