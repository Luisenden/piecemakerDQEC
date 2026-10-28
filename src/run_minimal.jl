# Step 5: the minimal feasible set over the joint timing × fidelity grid.
#
# Steps 3+4 (run_cutoffrange_bfs.jl) ran one BFS per member of U_{t,∞} and saved, per member, the
# minimal feasible fidelity configurations for each of that member's timing-cutoff pairs. Every
# row of a step4_table therefore already carries a full hardware point (h_t, h_f) together with
# its cutoff c. This script pools those rows over all members and keeps the least demanding ones.
#
# Why pooling the per-member minima is enough: if (h_t, h_f) is globally minimal, then h_f must
# already be minimal within its own member's feasible set — otherwise some h_f' ≥ h_f for the
# same h_t would dominate it. The global frontier is thus a subset of the union of the per-member
# frontiers, and no feasible point is lost by never looking at the non-minimal interior.
#
# Both grids are ordered so that index 1 is the most demanding value and a larger index is more
# relaxed (see order_timing_grid / order_fidelity_grid). The joint index vector (i_t..., i_f...)
# therefore inherits that orientation on all 9 axes, and "minimal feasible" = componentwise
# maximal = the Pareto front of least demanding hardware.
#
# The cutoff c is a protocol knob, not hardware, so by default the frontier is taken over
# (h_t, h_f) alone and the cutoff with the most slack against ε* is reported for each point.
# keep_cutoff = true instead lists every cutoff that attains a frontier point.
#
# Usage:
#   julia --project=src src/run_minimal.jl <code> [input_path] [output_path] [keep_cutoff] [singlezone]
#
#   singlezone = true pools the single-fusion-zone files (singlezone_* prefix); the default
#   false pools the parallel ones (multizone_* prefix).

using JLD2
using DataFrames
using Printf
using CSV

const code = length(ARGS) >= 1 ? ARGS[1] : ""
input_path = length(ARGS) >= 2 ? ARGS[2] : "./"
output_path = length(ARGS) >= 3 ? ARGS[3] : "./"
keep_cutoff = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true
singlezone = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false
zone = singlezone ? "singlezone" : "multizone"

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

## Load: the bfs and U_t file of each member, merged on the member index.

function load_member(index)
    @load joinpath(input_path, "$(zone)_bfs_member$(index)_minimal_fidelity_$(code).jld2") p_mem_max τ_max T_coh_data sim_kwargs gate_fidelity_qec timing_grid fidelity_grid step4_table step4_log
    @load joinpath(input_path, "$(zone)_U_t_member$(index)_timing_cutoff_pairs_$(code).jld2") step3_table
    return (; index, p_mem_max, τ_max, T_coh_data, sim_kwargs, gate_fidelity_qec,
              timing_grid, fidelity_grid, step4_table, step4_log, step3_table)
end

bfs_re = Regex("^$(zone)_bfs_member(\\d+)_minimal_fidelity_$(code)\\.jld2\$")
indices = sort([parse(Int, m.captures[1])
                for m in (match(bfs_re, f) for f in readdir(input_path)) if m !== nothing])
isempty(indices) && error("no $(zone)_bfs_member<index>_minimal_fidelity_$(code).jld2 found in $(input_path)")

data = [load_member(index) for index in indices]

# One frontier over different grids would be meaningless, since the Pareto comparison is on
# indices. All members come from the same pruned file, so the grids and budget must agree exactly.
(; p_mem_max, τ_max, T_coh_data, sim_kwargs, gate_fidelity_qec, timing_grid, fidelity_grid) = data[1]
for d in data
    (d.timing_grid, d.fidelity_grid, d.τ_max) == (timing_grid, fidelity_grid, τ_max) ||
        error("member $(d.index): grids or τ_max differ from member $(data[1].index)")
    get(d.sim_kwargs, :singlezone, false) == singlezone ||
        error("member $(d.index): sim_kwargs.singlezone does not match the requested singlezone = $(singlezone)")
end

tnames, fnames = keys(timing_grid), keys(fidelity_grid)
anames = (tnames..., fnames...)
length(anames) == 9 ||
    error("expected 5 timing + 4 fidelity axes, got $(length(tnames)) + $(length(fnames))")
idx_names = [Symbol("idx_", name) for name in anames]

@info "Step 5: pooling Step-4 results" code input_path n_members = length(data) p_mem_max τ_max
t_start = time()

## Pool: map every Step-4 row onto the joint index lattice.

"""Ladder index of `v` on `axis`, i.e. the position the value was drawn from."""
function axis_index(axis, v)
    i = findfirst(a -> isapprox(a, v; rtol = 1e-9), axis)
    i === nothing && error("value $(v) is not on the saved grid $(axis)")
    return i
end

function index_rows(d)
    t = copy(d.step4_table)
    t[!, :member] .= d.index
    for (name, axis) in pairs(merge(timing_grid, fidelity_grid))
        t[!, Symbol("idx_", name)] = [axis_index(axis, v) for v in t[!, name]]
    end
    return t
end

with_rows = [d for d in data if nrow(d.step4_table) > 0]
isempty(with_rows) && error("every Step-4 table was empty: no feasible (h_t, c, h_f) anywhere")
pooled = reduce(vcat, [index_rows(d) for d in with_rows]; cols = :union)
pooled.joint = collect(zip((pooled[!, n] for n in idx_names)...))

## The frontier

"""`q` is less demanding than `p` on every axis and differs somewhere."""
dominates(q, p) = all(q .>= p) && q != p

"""
The componentwise-maximal (least demanding) elements of `points`. A dominator has a strictly
larger index sum, so visiting the points in order of decreasing sum means every point only has
to be tested against the front accepted so far.
"""
function pareto_maxima(points)
    front = eltype(points)[]
    for p in sort(unique(points); by = p -> (-sum(p), p))
        any(q -> dominates(q, p), front) || push!(front, p)
    end
    return front
end

front = pareto_maxima(pooled.joint)
on_front = Set(front)
frontier = pooled[in.(pooled.joint, Ref(on_front)), :]

# A joint index fixes h_t, hence the member it came from.
nrow(unique(frontier, [:joint, :member])) == length(front) ||
    @warn "a frontier point is reported by several members: duplicate timing configurations"

frontier.margin = frontier.eps_star .- frontier.eps_hat
frontier.margin_rel = frontier.margin ./ frontier.eps_star
frontier.index_sum = sum.(frontier.joint)
transform!(groupby(frontier, :joint), nrow => :n_cutoffs)

# Collapse over the cutoff: keep the row with the most slack against ε*, ties on the faster cycle.
if !keep_cutoff
    sort!(frontier, [:margin, :τ]; rev = [true, false])
    unique!(frontier, :joint)
end
# Least demanding overall first, so the top of the table is the cheapest hardware.
sort!(frontier, [:index_sum, :joint, :cutoff]; rev = [true, false, false])
step5_table = select(frontier, Not(:joint))

step5_coverage = DataFrame(member = [d.index for d in data],
                           n_step3 = [nrow(d.step3_table) for d in data],
                           n_step4 = [nrow(d.step4_table) for d in data])

step4_log = reduce(vcat, [d.step4_log for d in data]; cols = :union)
step5_status = hasproperty(step4_log, :status) ?
    sort(combine(groupby(step4_log, :status), nrow => :n), :n; rev = true) : DataFrame()

## Reporting

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
fmt_axis(name, v) = name in (:F_link, :F_CNOT, :F_readout) ? @sprintf("%.6g", v) :
                    name === :T_coherence ? @sprintf("%g s", v) :
                    name === :link_success_prob ? @sprintf("%g", v) : fmt_t(v)

n_empty = count(==(0), step5_coverage.n_step4)

println()
println("Step 5: ", length(front), " minimal feasible (h_t, h_f) of ", length(unique(pooled.joint)),
        " distinct pooled points (", nrow(pooled), " Step-4 rows over ", length(with_rows), " members)")
println("        grid: ", prod(length, timing_grid), " timing × ", prod(length, fidelity_grid),
        " fidelity = ", prod(length, timing_grid) * prod(length, fidelity_grid), " joint points")
@printf("        p_mem_max = %.4g, τ_max = %.4g ms\n", p_mem_max, 1e3τ_max)
n_empty > 0 && println("        ", n_empty, " member(s) had no feasible fidelity point at all")
isempty(step5_status) || println("        Step-4 (h_t, c) outcomes: ",
    join(["$(r.status): $(r.n)" for r in eachrow(step5_status)], ", "))

# Per axis, the most relaxed value that any frontier point gets away with. No single point need
# reach all of them at once, so this is an envelope, not a configuration.
println()
println("Least demanding value reached on the frontier, per axis:")
for (d, name) in enumerate(anames)
    axis = d <= length(tnames) ? timing_grid[name] : fidelity_grid[name]
    idxs = [p[d] for p in front]
    @printf("  %-20s %-14s (index %d/%d)   most demanding still needed: %s\n",
            name, fmt_axis(name, axis[maximum(idxs)]), maximum(idxs), length(axis),
            fmt_axis(name, axis[minimum(idxs)]))
end

println()
println("Frontier (least demanding first", keep_cutoff ? "" : ", best cutoff per point", "):")
for (i, r) in enumerate(eachrow(step5_table))
    i > 25 && (println("  … and ", nrow(step5_table) - 25, " more"); break)
    @printf("  [%3d] %s | %s | c = %-10s τ = %7.3g ms  p_mem = %.4f  ε̂ = %.3g ≤ ε* = %.3g\n",
            i,
            join([fmt_axis(n, r[n]) for n in tnames], " "),
            join([fmt_axis(n, r[n]) for n in fnames], " "),
            isfinite(r.cutoff) ? @sprintf("%.3g ms", 1e3r.cutoff) : "Inf",
            1e3r.τ, r.p_mem, r.eps_hat, r.eps_star)
end
println()

## Save

jldsave(joinpath(output_path, "$(zone)_minimal_requirements_$(code).jld2");
    p_mem_max, τ_max, T_coh_data, sim_kwargs, gate_fidelity_qec,
    timing_grid, fidelity_grid, keep_cutoff,
    step5_table, step5_coverage, step5_status,
    n_pooled = nrow(pooled), n_frontier = length(front))
CSV.write(joinpath(output_path, "$(zone)_minimal_requirements_$(code).csv"), step5_table)
println("Step 5: wrote $(nrow(step5_table)) frontier rows to $(output_path) ",
        @sprintf("(%.1f s)", time() - t_start))
