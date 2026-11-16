
const code = length(ARGS) >= 1 ? ARGS[1] : ""
const gate_fidelity_qec = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
const target_rounds = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
const max_wallclock = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 60.0
const singlezone = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false 
const output_path = length(ARGS) >= 6 ? ARGS[6] : "./"
const seed = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 1234  # one fusion zone vs. as many as needed
const zone = singlezone ? "singlezone" : "multizone"                    # prefix of every output file
code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
include(joinpath(@__DIR__, "search_grid.jl"))   # TIMING_GRID

# the error budget depends only on the code, not on the node partitioning (BASE_CODE strips "_partitioned")
@load joinpath(@__DIR__, "error_budget_$(BASE_CODE)_Fg$(round(Int, 1e4 * gate_fidelity_qec)).jld2") p_mem_max τ_max
## Step 2: prune the timing configurations

# Candidate values per timing parameter (search_grid.jl; sorted by order_timing_grid).
# To fix a parameter, override its axis with a single value via merge.
const timing_grid = order_timing_grid(TIMING_GRID)
sim_kwargs = (seed = seed, target_rounds = target_rounds, max_wallclock = max_wallclock, singlezone = singlezone)
timing_cache = Dict{Any, Any}()

start = time()
step2 = prune_timing_configurations!(timing_cache, timing_grid, τ_max; sim_kwargs = sim_kwargs)
println("Step 2: $(length(step2.members)) of $(prod(length, timing_grid)) timing configurations survive " *
        "($(step2.n_simulations) simulations, $(round(time() - start; digits = 1)) s)")

# cycle-time estimate τ̂ and its batch-means standard error τ_se for every simulated configuration
evaluations = summary_table(collect(values(timing_cache)))
##
members = step2.members
sim_kwargs = step2.sim_kwargs
run_info = (; code, gate_fidelity_qec, sim_kwargs...)
error_budget = (; p_mem_max, τ_max)
T_coh_data = step2.T_coh_data
@save joinpath(@__DIR__, "$(zone)_pruned_timing_configurations_$(code)_Fg$(round(Int, 1e4 * gate_fidelity_qec)).jld2") members evaluations run_info error_budget T_coh_data