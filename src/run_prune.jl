
const code = length(ARGS) >= 1 ? ARGS[1] : ""
const gate_fidelity_qec = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
const target_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
const max_wallclock = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 60.0
const output_path = length(ARGS) >= 5 ? ARGS[5] : "./"
const seed = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1234
code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
## Step 1: find the break-even memory budget (p_mem^max, τ_max) for the QEC code and gate fidelity.
p_mem_max = find_pmem_max(qec_parity_checks(code), gate_fidelity_qec; seed=seed)

τ_max = t_cycle(p_mem_max; T_coh = T_COH_DATA)
@info "Break-even memory budget" p_mem_max τ_max
## Step 2: prune the timing configurations

# Candidate values per timing parameter (any order; sorted by order_timing_grid).
# A single value fixes that parameter.
const timing_grid = order_timing_grid((
    attempt_time        = [0.1e-6, 0.5e-6, 1e-6, 10e-6],          # t_att
    Δt_CNOTgate         = [1e-6, 10e-6, 100e-6, 250e-6],          # t_CNOT
    Δt_readout          = [0.1e-3, 1e-3, 2e-3],                   # t_ro
    Δt_rotation_shuttle = [10e-6, 50e-6, 100e-6],                 # t_buff
    link_success_prob   = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],         # p_link
))
sim_kwargs = (seed = seed, target_samples = target_samples, max_wallclock = max_wallclock)
timing_cache = Dict{Any, Any}()

start = time()
step2 = prune_timing_configurations!(timing_cache, timing_grid, τ_max; sim_kwargs = sim_kwargs)
println("Step 2: $(length(step2.members)) of $(prod(length, timing_grid)) timing configurations survive " *
        "($(step2.n_simulations) simulations, $(round(time() - start; digits = 1)) s)")
##
members = step2.members
sim_kwargs = step2.sim_kwargs
run_info = (; code, gate_fidelity_qec, sim_kwargs...)
error_budget = (; p_mem_max, τ_max)
T_coh_data = step2.T_coh_data
@save joinpath(@__DIR__, "pruned_timing_configurations_$(code).jld2") members run_info error_budget T_coh_data