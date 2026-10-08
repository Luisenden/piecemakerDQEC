const code = "Steane713"
const error_model = "depolarizing"

index       = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 231
input_path  = length(ARGS) >= 2 ? ARGS[2] :
    "...4_backup_project_piecemakerDQEC/output_v1_cutoff/"
output_path = length(ARGS) >= 3 ? ARGS[3] : "./"
max_wall    = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : nothing   # override max_wallclock

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
include(joinpath(@__DIR__, "GHZfidelity_closedform.jl"))

const memory_model = Symbol(error_model)
const gate_fidelity_qec = 0.9997      # QEC-layer two-qubit gate fidelity, as in Steps 1-3

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

## Load one member of U_{t,∞} with its timing-cutoff pairs from Step 3.
@load joinpath(input_path, "U_t_member$(index)_timing_cutoff_pairs_$(code).jld2") step3_table sim_kwargs p_mem_max τ_max
max_wall === nothing || (sim_kwargs = merge(sim_kwargs, (max_wallclock = max_wall,)))

@info "Step 4: member $(index), $(nrow(step3_table)) timing-cutoff pairs, " *
      "$(prod(length, fidelity_grid)) fidelity configurations" p_mem_max τ_max sim_kwargs

decoder = TableDecoder(qec_parity_checks(code), error_weight = 4)   # as in find_pmem_max
eps_cache = Dict{Float64, Any}()       # ε* depends on p_mem only, so evaluate it once per value
timing_cache = Dict{Any, Any}()


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

# h_t in the field order of Steps 2-3, so the cache key matches `timing_point`.
timing_from_row(r) = NamedTuple{keys(TIMING_DIRECTIONS)}(
    ntuple(i -> Float64(r[keys(TIMING_DIRECTIONS)[i]]), length(TIMING_DIRECTIONS)))

rows = NamedTuple[]
log_rows = NamedTuple[]
t_start = time()

for (i, r) in enumerate(eachrow(step3_table))
    t_row = time()
    h_t = timing_from_row(r)

    # Step 3 saved only the summary, so the trace T(h_t, c) is regenerated here. Same seed and
    # sim_kwargs as Step 3, hence the same event stream.
    ev = evaluate_timing!(timing_cache, h_t, r.cutoff; sim_kwargs = sim_kwargs)
    empty!(timing_cache)

    logrow(status; eps_star = NaN, logeps = NaN, eps_status = :none, p_mem_eps = NaN,
           n_visited = 0, n_feasible = 0, n_minimal = 0) = push!(log_rows, (
        cutoff = ev.cutoff, k = r.k, p_kept = r.p_kept,
        τ = ev.τ, p_mem = ev.p_mem, τ_step3 = r.τ, p_mem_step3 = r.p_mem,
        p_mem_eps = p_mem_eps,
        n_ghz = nrow(ev.raw_events), status = status,
        eps_star = eps_star, logeps = logeps, eps_status = eps_status,
        n_visited = n_visited, n_feasible = n_feasible, n_minimal = n_minimal,
        seconds = time() - t_row,
    ))

    # The re-run is cut by max_wallclock as well as by target_samples, so a faster or slower
    # machine logs a different number of GHZ states and τ̂ moves a little.
    if isfinite(r.p_mem) && abs(ev.p_mem - r.p_mem) > 0.01 * r.p_mem
        @warn "p_mem of the re-run differs from Step 3 by more than 1 %" h_t cutoff=ev.cutoff ev.p_mem r.p_mem
    end
    if !cycle_time_feasible(ev, τ_max)
        @warn "re-run of (h_t, c) is no longer within τ_max; skipped" h_t cutoff=ev.cutoff ev.τ τ_max
        logrow(:cycle_time_infeasible)
        continue
    end

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
        continue
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
            cutoff = ev.cutoff, k = r.k, p_kept = r.p_kept, τ = ev.τ, p_mem = ev.p_mem,
            eps_star = res.eps_ghz,
            eps_hat = ghz_infidelity(ev.raw_events, h_f; memory = memory_model),
        )))
    end
    logrow(:ok; eps_star = res.eps_ghz, logeps = res.logeps, eps_status = res.status, p_mem_eps = p_eps,
           n_visited = bfs.n_visited, n_feasible = length(bfs.feasible), n_minimal = length(bfs.minimal))

    println("[", lpad(i, ndigits(nrow(step3_table))), "/", nrow(step3_table), "] ",
            @sprintf("c = %-10s k = %2d | τ = %8.3g ms  p_mem = %.4f  ε* = %.3g | ",
                     isfinite(ev.cutoff) ? @sprintf("%.3g ms", 1e3 * ev.cutoff) : "Inf",
                     r.k, 1e3 * ev.τ, ev.p_mem, res.eps_ghz),
            bfs.n_visited, " visited, ", length(bfs.feasible), " feasible, ",
            length(bfs.minimal), " minimal",
            @sprintf(" | %d GHZ, %.1f s", nrow(ev.raw_events), time() - t_row))
    flush(stdout)
end

step4_table = isempty(rows) ? DataFrame() : DataFrame(rows)
step4_log = isempty(log_rows) ? DataFrame() : DataFrame(log_rows)
println("Step 4: $(nrow(step4_table)) minimal feasible (h_t, c, h_f) over " *
        "$(nrow(step3_table)) timing-cutoff pairs ($(round(time() - t_start; digits = 1)) s)")
##
jldsave(joinpath(output_path, "step4_member$(index)_minimal_fidelity_$(code).jld2");
    p_mem_max, τ_max, T_COH_DATA, sim_kwargs, gate_fidelity_qec,
    timing_grid, fidelity_grid, step4_table, step4_log)
