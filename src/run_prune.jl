const code = "Steane713"
const error_model = "depolarizing"

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
## Step 1: find the break-even memory budget (p_mem^max, τ_max) for the QEC code and gate fidelity.
const gate_fidelity_qec = 0.9997
const p_mem_max = 0.08660254037844387#find_pmem_max(qec_parity_checks(code), gate_fidelity_qec)
# Alternatives: take the value from the Step-1 curve directly (e.g. p_mem_max = 0.04), or use
# the stricter p_mem_max = pmem_peak(pmem_values, eps_star_values).
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
sim_kwargs = (seed = 1234, target_samples = 2000, max_wallclock = 60.0)
timing_cache = Dict{Any, Any}()

start = time()
step2 = prune_timing_configurations!(timing_cache, timing_grid, τ_max; sim_kwargs = sim_kwargs)
println("Step 2: $(length(step2.members)) of $(prod(length, timing_grid)) timing configurations survive " *
        "($(step2.n_simulations) simulations, $(round(time() - start; digits = 1)) s)")
##
members = step2.members
@save joinpath(@__DIR__, "step2_pruned_timing_configurations_$(code).jld2") members
