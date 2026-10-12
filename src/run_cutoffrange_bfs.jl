# Steps 3 + 4 in one run: sweep the cutoffs of one member of U_{t,∞} and, as soon as a pair
# (h_t, c) is found feasible, run the fidelity BFS on its trace. The trace is dropped right
# after, so only the summary per (h_t, c) and the BFS results are kept (no re-simulation as in
# run_findcutoffrange.jl → run_bfs.jl).

using JLD2

const code = length(ARGS) >= 1 ? ARGS[1] : ""
const error_model = length(ARGS) >= 2 ? ARGS[2] : "depolarizing"
index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
k_max = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 50
input_path = length(ARGS) >= 5 ? ARGS[5] : "./"
output_path = length(ARGS) >= 6 ? ARGS[6] : "./"

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

# Steps 3+4 reuse the gate fidelity and noise model Step 2 (run_prune.jl) pruned with.
@load joinpath(input_path, "pruned_timing_configurations_$(code).jld2") members run_info
const gate_fidelity_qec = run_info.gate_fidelity_qec
const memory_model = Symbol(error_model)
const seed = run_info.seed

const target_samples = run_info.target_samples
const max_wallclock = run_info.max_wallclock

@info "Settings" code gate_fidelity_qec error_model target_samples max_wallclock

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
include(joinpath(@__DIR__, "GHZfidelity_closedform.jl"))

p_mem_max, τ_max = run_info.error_budget.p_mem_max, run_info.error_budget.τ_max
@info "Break-even memory budget" p_mem_max τ_max

# Bracket of the ε* binary search; kept here so the guards below stay in sync with the call.
const LOGEPS_MIN, LOGEPS_MAX, LOGEPS_TOL = -6.0, -1.0, 0.05

# The timing grid the members of U_{t,∞} were drawn from (Step 2). Only saved alongside the
# results, so Step 5 can map a timing value back to its ladder index.
const timing_grid = order_timing_grid((
    attempt_time        = [0.1e-6, 0.5e-6, 1e-6, 10e-6],          # t_att
    Δt_CNOTgate         = [1e-6, 10e-6, 100e-6, 250e-6],          # t_CNOT
    Δt_readout          = [0.1e-3, 1e-3, 2e-3],                   # t_ro
    Δt_rotation_shuttle = [10e-6, 50e-6, 100e-6],                 # t_buff
    link_success_prob   = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],         # p_link
))

# Candidate values per fidelity parameter (any order; sorted most → least demanding by
# order_fidelity_grid, so index 1 is the corner h_f^⊤ the BFS starts from and walks down from).
# A single value fixes that parameter.
const fidelity_grid = order_fidelity_grid((
    F_link      = [1.0 - 2.5^(-x) for x in 3.0:10.0],           # F_link
    F_CNOT      = [0.999, 0.9995, 0.9997, 0.9999, 0.99999],     # F_CNOT
    F_readout   = [0.999, 0.9999, 1.0],                         # F_ro
    T_coherence = [0.5, 1.0, 2.0, 10.0, 20.0],                  # T_coh,comm
))

member = members[index]
sim_kwargs = (seed = seed, target_samples = target_samples, max_wallclock = max_wallclock)

@info "Steps 3+4: member $(index), $(prod(length, fidelity_grid)) fidelity configurations" p_mem_max τ_max sim_kwargs

decoder = TableDecoder(qec_parity_checks(code), error_weight = 4)   # as in find_pmem_max
eps_cache = Dict{Float64, Any}()       # ε* depends on p_mem only, so evaluate it once per value

const EPS_CACHE_RTOL = 0.01

"""Key of `eps_cache` for `p`: the closest stored p_mem within EPS_CACHE_RTOL, else `p` itself."""
function eps_cache_key(cache, p)
    key, best = p, Inf
    for q in keys(cache)
        d = abs(p - q)
        if d <= EPS_CACHE_RTOL * q && d < best
            key, best = q, d
        end
    end
    return key
end

rows = NamedTuple[]
log_rows = NamedTuple[]

"""
Step 4 for one feasible pair (h_t, c): ε*(p_mem) and the BFS over H_f, on the trace that Step 3
has just simulated. `ev` is an entry of U_{t,c} (see `cutoff_range!`), trace included.
"""
function run_bfs!(ev)
    t_row = time()
    h_t = ev.h_t
    n_ghz = nrow(ev.raw_events)

    logrow(status; eps_star = NaN, logeps = NaN, eps_status = :none, p_mem_eps = NaN,
           n_visited = 0, n_feasible = 0, n_minimal = 0) = push!(log_rows, (
        cutoff = ev.cutoff, k = ev.k, p_kept = ev.p_kept,
        τ = ev.τ, p_mem = ev.p_mem, p_mem_eps = p_mem_eps,
        n_ghz = n_ghz, status = status,
        eps_star = eps_star, logeps = logeps, eps_status = eps_status,
        n_visited = n_visited, n_feasible = n_feasible, n_minimal = n_minimal,
        seconds = time() - t_row,
    ))

    ## Determine the largest tolerable GHZ infidelity ε*(p_mem).
    p_eps = eps_cache_key(eps_cache, ev.p_mem)
    res = get!(eps_cache, p_eps) do
        classify_point(decoder, p_eps, gate_fidelity_qec, 1.0).status == :interval_above &&
            return (eps_ghz = NaN, logeps = NaN, status = :below_break_even_floor)
        find_break_even(decoder, p_eps, gate_fidelity_qec, LOGEPS_MIN, LOGEPS_MAX, LOGEPS_TOL)
    end
    if res.status == :below_break_even_floor
        @warn "p_L > p_mem even for a perfect GHZ state: this (h_t, c) is too slow to break even; BFS skipped" h_t cutoff=ev.cutoff ev.p_mem p_eps
        logrow(:below_break_even_floor; eps_status = res.status, p_mem_eps = p_eps)
        return
    end
    res.logeps <= LOGEPS_MIN + LOGEPS_TOL &&
        @warn "ε* is clipped by the lower bracket; lower LOGEPS_MIN" h_t cutoff=ev.cutoff res.eps_ghz
    res.logeps >= LOGEPS_MAX - LOGEPS_TOL &&
        @warn "ε* is clipped by the upper bracket; widen LOGEPS_MAX" h_t cutoff=ev.cutoff res.eps_ghz

    ## Step 4 of the write-up: BFS over H_f.
    bfs = feasible_fidelity_bfs(fidelity_grid, ev.raw_events, res.eps_ghz; memory = memory_model)
    for idx in bfs.minimal
        h_f = fidelity_point(fidelity_grid, idx)
        push!(rows, merge(h_t, h_f, (
            cutoff = ev.cutoff, k = ev.k, p_kept = ev.p_kept, τ = ev.τ, p_mem = ev.p_mem,
            eps_star = res.eps_ghz,
            eps_hat = ghz_infidelity(ev.raw_events, h_f; memory = memory_model),
        )))
    end
    logrow(:ok; eps_star = res.eps_ghz, logeps = res.logeps, eps_status = res.status, p_mem_eps = p_eps,
           n_visited = bfs.n_visited, n_feasible = length(bfs.feasible), n_minimal = length(bfs.minimal))

    println("      BFS ",
            @sprintf("c = %-10s k = %2d | τ = %8.3g ms  p_mem = %.4f  ε* = %.3g | ",
                     isfinite(ev.cutoff) ? @sprintf("%.3g ms", 1e3 * ev.cutoff) : "Inf",
                     ev.k, 1e3 * ev.τ, ev.p_mem, res.eps_ghz),
            bfs.n_visited, " visited, ", length(bfs.feasible), " feasible, ",
            length(bfs.minimal), " minimal",
            @sprintf(" | %d GHZ, %.1f s", n_ghz, time() - t_row))
    flush(stdout)
end
##
timing_cache = Dict{Any, Any}()
start = time()
U_tc = cutoff_range!(timing_cache, member, τ_max; k_max = k_max, patience = 1, sim_kwargs = sim_kwargs,
                     on_feasible = run_bfs!, keep_traces = false)
step3_table = summary_table(U_tc)
step4_table = isempty(rows) ? DataFrame() : DataFrame(rows)
step4_log = isempty(log_rows) ? DataFrame() : DataFrame(log_rows)
println("Steps 3+4: $(nrow(step3_table)) timing-cutoff pairs in U_{t,c}, $(nrow(step4_table)) minimal " *
        "feasible (h_t, c, h_f) ($(round(time() - start; digits = 1)) s)")
##
jldsave(joinpath(output_path, "U_t_member$(index)_timing_cutoff_pairs_$(code).jld2");
    p_mem_max, τ_max, T_COH_DATA, sim_kwargs,
    step3_table)
jldsave(joinpath(output_path, "bfs_member$(index)_minimal_fidelity_$(code).jld2");
    p_mem_max, τ_max, T_COH_DATA, sim_kwargs, gate_fidelity_qec,
    timing_grid, fidelity_grid, step4_table, step4_log)
