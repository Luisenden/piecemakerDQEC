const code = "Steane713"
const error_model = "depolarizing"

index = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
k_max = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
target_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2000
max_wallclock = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 60.0
output_path = length(ARGS) >= 5 ? ARGS[5] : "./"

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))

## Step 1: find the break-even memory budget (p_mem^max, τ_max) for the QEC code and gate fidelity.
const gate_fidelity_qec = 0.9997
p_mem_max = find_pmem_max(qec_parity_checks(code), gate_fidelity_qec)
# as alternative use the stricter p_mem_max = pmem_peak(pmem_values, eps_star_values).
τ_max = t_cycle(p_mem_max; T_coh = T_COH_DATA)

@load joinpath(@__DIR__, "step2_pruned_timing_configurations_$(code).jld2") members
member = members[index]
##
sim_kwargs = (seed = 1234, target_samples = target_samples, max_wallclock = max_wallclock)
timing_cache = Dict{Any, Any}()
start = time()
U_tc = cutoff_range!(timing_cache, member, τ_max; k_max = k_max, patience = 1, sim_kwargs = sim_kwargs)
step3_table = summary_table(U_tc)
println("Step 3: $(nrow(step3_table)) timing-cutoff pairs in U_{t,c} ($(round(time() - start; digits = 1)) s)")
##
jldsave(joinpath(output_path, "U_t_member$(index)_timing_cutoff_pairs_$(code).jld2");
    p_mem_max, τ_max, T_COH_DATA, sim_kwargs,
    step3_table)