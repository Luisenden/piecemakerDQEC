using DataFrames
using JLD2
using StatsBase
const code = "Steane713"
const error_model = "depolarizing"
##
include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
include(joinpath(@__DIR__, "GHZfidelity_closedform.jl"))
##
start = time()
result = run_single_configuration(
    F_link = 1.0, # (dummy)
    link_success_prob = 1e-3,
    attempt_time = 1e-6,
    T_coherence = Inf, # (dummy)
    Δt_CNOTgate = 100e-6,
    gate_fidelity = 1.0, # (dummy)
    Δt_readout = 1e-3,
    readout_fidelity = 1.0, # (dummy)
    Δt_rotation_shuttle = 100e-6,
    cutoff = Inf,

    seed = 1234,
    target_samples = 2000,
    max_wallclock = 60.0,
)

df_raw = result.raw_events

println("Runtime: $(time() - start) seconds")
##

# Completions per generator that make one QEC round: Steane checks share X/Z supports.
required_count = code == "Steane713" ? 2 : 1

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
t_cycle = mean(diff(timesteps_markers))
##

# Exact per-GHZ fidelity from the logged timing trace (b_j, c_j, g_j, piecemaker readout,
# consumption). All times come from the event simulation itself, so queueing behind other
# GHZ attempts is included. The noise parameters are applied here; the event simulation
# itself runs noise-free (T_coherence = Inf, perfect pairs).
start = time()
df_raw.fidelity = ghz_fidelities_from_log(df_raw;
    T_coherence = 1.0,          # τ (depolarizing) or T2 (dephasing), same units as the simulation (s)
    memory = :depolarizing,     # or :dephasing
    F_link = 0.97,
    F_CNOT = 0.9997,
    F_readout = 1.0,
)
println("Closed-form fidelities: $(time() - start) seconds for $(nrow(df_raw)) GHZ states")
@info "mean GHZ fidelity" mean(df_raw.fidelity)
