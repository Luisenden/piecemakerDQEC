# Parameter-sweep / HPC driver for GHZservice_v1_partitioned_sim_raw.jl
using DataFrames
using JLD2

# Command-line arguments retain the meaning of the original script.
const code = length(ARGS) >= 1 ? ARGS[1] : "Steane7"
const error_model = length(ARGS) >= 2 ? ARGS[2] : "depolarizing"
target_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5000
global_idx = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
output_path = length(ARGS) >= 5 ? ARGS[5] : "./"
max_wallclock = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 60.0
seed = 1234

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))

# Local-operation parameter grid. One PBS array index selects one tuple.
attempt_times = [0.1e-6, 0.5e-6, 1e-6, 1e-5]
T_coherences = [0.5, 1.0, 2.0, 10.0, 20.0]
CNOTgate_times = [1e-6, 10e-6, 100e-6, 250e-6]
CNOTgate_fidelities = [0.999, 0.9995, 0.9997, 0.9999, 0.99999]
readout_times = [0.1e-3, 1e-3, 2e-3]
readout_fidelities = [0.999, 0.9999, 1.0]
rotation_shuttle_times = [10e-6, 50e-6, 100e-6]

parameter_combinations = [
    (attempt_time, T_coherence, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, Δt_rotation_shuttle)
    for attempt_time in attempt_times
    for T_coherence in T_coherences
    for gate_fidelity in CNOTgate_fidelities
    for Δt_CNOTgate in CNOTgate_times
    for Δt_readout in readout_times
    for readout_fidelity in readout_fidelities
    for Δt_rotation_shuttle in rotation_shuttle_times
]

@assert 1 <= global_idx <= length(parameter_combinations) "global_idx=$(global_idx) is outside 1:$(length(parameter_combinations))"

(
    attempt_time,
    T_coherence,
    Δt_CNOTgate,
    gate_fidelity,
    Δt_readout,
    readout_fidelity,
    Δt_rotation_shuttle,
) = parameter_combinations[global_idx]

link_success_probs = [10.0^(-x) for x in 1.0:5.0]
F_links = [1.0 - 2.5^(-x) for x in 3.0:10.0]
cutoffs = [
    get_cutoff(T_coherence, 0.01),
    get_cutoff(T_coherence, 0.05),
    Inf,
]

# Event data stay raw. To avoid repeating all physical parameters in every event
# row, each event carries only run_id; df_run_metadata maps run_id -> parameters.
raw_dfs = DataFrame[]
run_metadata_rows = NamedTuple[]
run_id = 0

for link_success_prob in link_success_probs
    for F_link in F_links
        for cutoff in cutoffs
            run_id += 1

            result = run_single_configuration(
                F_link = F_link,
                link_success_prob = link_success_prob,
                attempt_time = attempt_time,
                T_coherence = T_coherence,
                Δt_CNOTgate = Δt_CNOTgate,
                gate_fidelity = gate_fidelity,
                Δt_readout = Δt_readout,
                readout_fidelity = readout_fidelity,
                Δt_rotation_shuttle = Δt_rotation_shuttle,
                cutoff = cutoff,
                seed = seed,
                target_samples = target_samples,
                max_wallclock = max_wallclock,
            )

            events = result.raw_events
            insertcols!(events, 1, :run_id => fill(run_id, nrow(events)))
            push!(raw_dfs, events)

            push!(run_metadata_rows, (
                run_id = run_id,
                code = code,
                error_model = error_model,
                seed = seed,
                global_idx = global_idx,
                F_link = F_link,
                link_success_prob = link_success_prob,
                attempt_time = attempt_time,
                T_coherence = T_coherence,
                tCNOT = Δt_CNOTgate,
                gate_fidelity = gate_fidelity,
                tReadout = Δt_readout,
                readout_fidelity = readout_fidelity,
                tRotationShuttle = Δt_rotation_shuttle,
                cutoff = cutoff,
                runtime = result.runtime,
                wallclock_time = result.wallclock_time,
                converged = result.converged,
                n_raw_events = nrow(events),
            ))

            @info "Completed run_id=$(run_id), F_link=$(F_link), link_success_prob=$(link_success_prob), cutoff=$(cutoff), raw events=$(nrow(events))"
        end
    end
end

df_raw = isempty(raw_dfs) ? DataFrame() : vcat(raw_dfs...)
df_run_metadata = DataFrame(run_metadata_rows)

mkpath(output_path)
output_file = joinpath(output_path, "raw_ghz_service_v1_$(code)_$(error_model)_$(global_idx).jld2")

# No aggregate generator/data-qubit/node summaries are produced in this version.
@save output_file df_raw df_run_metadata
@info "Saved raw GHZ events to $(output_file)"
