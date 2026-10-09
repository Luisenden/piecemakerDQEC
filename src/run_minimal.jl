# Step 5: the minimal feasible set over the joint timing × fidelity grid.
#
# Step 4 (run_bfs.jl) ran one BFS per member of U_{t,∞} and saved, per member, the minimal
# feasible fidelity configurations for each of that member's timing-cutoff pairs. Every row of
# a step4_table therefore already carries a full hardware point (h_t, h_f) together with its
# cutoff c. This script pools those rows over all members and keeps the least demanding ones.
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
# (h_t, h_f) alone and the cutoff that attains each point is reported alongside. --keep-cutoff
# instead lists every cutoff that attains a frontier point.
#
# This is pure post-processing: it reads the saved grids out of the .jld2 files rather than
# re-deriving them, so it deliberately does not include the simulation stack.
#
# Usage:
#   julia --project=src src/run_minimal.jl <bfs_path> [timing_path] [output_path] [flags]
#
#   bfs_path     folder holding bfs_member<index>_minimal_fidelity_<code>.jld2   (Step 4)
#   timing_path  folder holding U_t_member<index>_timing_cutoff_pairs_<code>.jld2 (Step 3),
#                joined to the Step-4 files on <index>; defaults to bfs_path
#   output_path  where step5_minimal_requirements_<code>.{jld2,csv} are written; defaults to ./
#
#   --code=NAME     code tag in the file names (default Steane713)
#   --keep-cutoff   keep one row per (h_t, h_f, c) instead of one per (h_t, h_f)
#   --no-csv        skip the CSV next to the .jld2

using JLD2
using DataFrames
using Printf
using CSV

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

hasflag(name) = ("--$(name)" in flags)
function flagval(name, default)
    i = findfirst(f -> startswith(f, "--$(name)="), flags)
    return i === nothing ? default : String(split(flags[i], "="; limit = 2)[2])
end

let unknown = filter(f -> !(f in ("--keep-cutoff", "--no-csv")) && !startswith(f, "--code="), flags)
    isempty(unknown) || throw(ArgumentError("unknown flag(s): $(join(unknown, ", "))"))
end

const code  = flagval("code", "Steane713")
bfs_path    = length(pos) >= 1 ? pos[1] : "./"
timing_path = length(pos) >= 2 ? pos[2] : bfs_path
output_path = length(pos) >= 3 ? pos[3] : "./"
keep_cutoff = hasflag("keep-cutoff")
write_csv   = !hasflag("no-csv")

# Grid values round-trip through JLD2 as the Float64 they were written as, so matching a stored
# value back to its ladder index is exact; the tolerance only guards against a grid that was
# re-typed between runs.
const GRID_RTOL = 1e-9

## ---------------------------------------------------------------- file discovery and loading

bfs_re()    = Regex("^bfs_member(\\d+)_minimal_fidelity_" * code * "\\.jld2\$")
timing_re() = Regex("^U_t_member(\\d+)_timing_cutoff_pairs_" * code * "\\.jld2\$")

"""Map member index → path for every file in `dir` whose name matches `re`."""
function index_files(dir::AbstractString, re::Regex)
    isdir(dir) || throw(ArgumentError("not a directory: $(dir)"))
    out = Dict{Int, String}()
    for f in readdir(dir)
        m = match(re, f)
        m === nothing && continue
        out[parse(Int, m.captures[1])] = joinpath(dir, f)
    end
    return out
end

getkey_or(f, key, default) = haskey(f, key) ? f[key] : default

load_bfs(path) = jldopen(path, "r") do f
    (step4_table   = getkey_or(f, "step4_table", DataFrame()),
     step4_log     = getkey_or(f, "step4_log", DataFrame()),
     timing_grid   = f["timing_grid"],
     fidelity_grid = f["fidelity_grid"],
     p_mem_max     = getkey_or(f, "p_mem_max", NaN),
     τ_max         = getkey_or(f, "τ_max", NaN))
end

load_timing(path) = jldopen(path, "r") do f
    (step3_table = getkey_or(f, "step3_table", DataFrame()),
     p_mem_max   = getkey_or(f, "p_mem_max", NaN),
     τ_max       = getkey_or(f, "τ_max", NaN))
end

"""Ladder index of `v` on `axis`, i.e. the position the value was drawn from."""
function axis_index(axis::AbstractVector{<:Real}, v::Real, name::Symbol, member::Int)
    best, bestd = 0, Inf
    for (i, a) in enumerate(axis)
        d = abs(a - v)
        d < bestd && ((best, bestd) = (i, d))
    end
    ok = bestd == 0 || bestd <= GRID_RTOL * max(abs(v), abs(axis[best]))
    ok || error("member $(member): value $(v) of $(name) is not on the saved grid $(axis)")
    return best
end

"""Same axes, same length, same values — grids pooled into one frontier must agree."""
function grids_agree(a::NamedTuple, b::NamedTuple)
    keys(a) == keys(b) || return false
    return all(keys(a)) do k
        length(a[k]) == length(b[k]) && all(isapprox.(a[k], b[k]; rtol = GRID_RTOL))
    end
end

## ------------------------------------------------------------------------- the Pareto filter

"""`q` is less demanding than `p` on every axis and differs somewhere."""
function dominates(q::NTuple{N, Int}, p::NTuple{N, Int}) where {N}
    strict = false
    for d in 1:N
        q[d] < p[d] && return false
        q[d] != p[d] && (strict = true)
    end
    return strict
end

"""
    pareto_maxima(points) -> Vector

The componentwise-maximal (least demanding) elements of `points`, as `least_demanding` in
utils_pseudothreshold.jl returns but without its O(|points|²) scan over the whole set: a
dominator has a strictly larger index sum, so visiting the points in order of decreasing sum
means every point only has to be tested against the front accepted so far.
"""
function pareto_maxima(points::Vector{NTuple{N, Int}}) where {N}
    isempty(points) && return NTuple{N, Int}[]
    front = NTuple{N, Int}[]
    for p in sort(unique(points); by = p -> (-sum(p), p))
        any(q -> dominates(q, p), front) || push!(front, p)
    end
    return front
end

## ------------------------------------------------------------------------------ pool Step 4

"""
    pool_step4(bfs_files, timing_files) -> NamedTuple

Read every Step-4 file, map each of its rows back onto the joint index lattice, and return the
pooled points together with the grids they are indexed against and a per-member coverage report.
The Step-3 files are joined in on the member index purely to see which members never produced a
Step-4 result — those are holes in the frontier that should be reported.
"""
function pool_step4(bfs_files::Dict{Int, String}, timing_files::Dict{Int, String})
    timing_grid   = nothing      # taken from the first Step-4 file, then checked against the rest
    fidelity_grid = nothing
    p_mem_max, τ_max = NaN, NaN

    rows     = NamedTuple[]
    coverage = NamedTuple[]
    status_counts = Dict{Symbol, Int}()

    for index in sort(collect(union(keys(bfs_files), keys(timing_files))))
        bfs_file    = get(bfs_files, index, nothing)
        timing_file = get(timing_files, index, nothing)

        n_step3 = 0
        if timing_file !== nothing
            tm = load_timing(timing_file)
            n_step3 = nrow(tm.step3_table)
            isnan(τ_max) && ((p_mem_max, τ_max) = (tm.p_mem_max, tm.τ_max))
        end

        if bfs_file === nothing
            @warn "member $(index) has a Step-3 file but no Step-4 file; its timing " *
                  "configuration cannot contribute to the frontier" timing_file
            push!(coverage, (member = index, has_timing = true, has_bfs = false,
                             n_step3 = n_step3, n_step4 = 0, n_joint = 0, status = :missing_bfs))
            continue
        end

        bf = load_bfs(bfs_file)

        # One frontier over heterogeneous grids would be meaningless: an index would stand for a
        # different value in each file, and the pooled Pareto comparison is on indices.
        if timing_grid === nothing
            timing_grid, fidelity_grid = bf.timing_grid, bf.fidelity_grid
            isnan(τ_max) && ((p_mem_max, τ_max) = (bf.p_mem_max, bf.τ_max))
        else
            grids_agree(timing_grid, bf.timing_grid) ||
                error("member $(index): timing_grid differs from the earlier Step-4 files")
            grids_agree(fidelity_grid, bf.fidelity_grid) ||
                error("member $(index): fidelity_grid differs from the earlier Step-4 files")
        end
        isfinite(bf.τ_max) && isfinite(τ_max) && !isapprox(bf.τ_max, τ_max; rtol = 1e-6) &&
            @warn "member $(index): τ_max differs from the earlier files" bf.τ_max τ_max

        for r in eachrow(bf.step4_log)
            s = Symbol(r.status)
            status_counts[s] = get(status_counts, s, 0) + 1
        end

        if nrow(bf.step4_table) == 0
            # Either every (h_t, c) was dropped before the BFS, or h_f^⊤ itself missed ε*.
            push!(coverage, (member = index, has_timing = timing_file !== nothing, has_bfs = true,
                             n_step3 = n_step3, n_step4 = 0, n_joint = 0,
                             status = :no_feasible_point))
            continue
        end

        tnames, fnames = keys(timing_grid), keys(fidelity_grid)
        anames = (tnames..., fnames...)
        for name in anames
            hasproperty(bf.step4_table, name) ||
                error("member $(index): step4_table has no column $(name)")
        end

        n_before = length(rows)
        for r in eachrow(bf.step4_table)
            it = ntuple(d -> axis_index(timing_grid[tnames[d]], r[tnames[d]], tnames[d], index),
                        length(tnames))
            jf = ntuple(d -> axis_index(fidelity_grid[fnames[d]], r[fnames[d]], fnames[d], index),
                        length(fnames))
            push!(rows, (
                member = index, joint = (it..., jf...),
                values = NamedTuple{anames}(ntuple(d -> Float64(r[anames[d]]), length(anames))),
                cutoff = Float64(r.cutoff),
                k = hasproperty(bf.step4_table, :k) ? Int(r.k) : 0,
                p_kept = hasproperty(bf.step4_table, :p_kept) ? Float64(r.p_kept) : NaN,
                τ = Float64(r.τ), p_mem = Float64(r.p_mem),
                eps_star = Float64(r.eps_star), eps_hat = Float64(r.eps_hat)))
        end
        push!(coverage, (member = index, has_timing = timing_file !== nothing, has_bfs = true,
                         n_step3 = n_step3, n_step4 = nrow(bf.step4_table),
                         n_joint = length(rows) - n_before, status = :ok))
    end

    isempty(rows) && error("every Step-4 file was empty: no feasible (h_t, c, h_f) anywhere")
    return (rows = rows, coverage = coverage, status_counts = status_counts,
            timing_grid = timing_grid, fidelity_grid = fidelity_grid,
            p_mem_max = p_mem_max, τ_max = τ_max)
end

bfs_files    = index_files(bfs_path, bfs_re())
timing_files = index_files(timing_path, timing_re())

isempty(bfs_files) &&
    error("no bfs_member<index>_minimal_fidelity_$(code).jld2 found in $(bfs_path)")

@info "Step 5: pooling Step-4 results" code bfs_path timing_path n_bfs =
    length(bfs_files) n_timing = length(timing_files)

t_start = time()
pooled = pool_step4(bfs_files, timing_files)

# Plain globals, not consts: the file is meant to be re-runnable cell by cell in the REPL.
rows          = pooled.rows
coverage      = pooled.coverage
status_counts = pooled.status_counts
timing_grid   = pooled.timing_grid
fidelity_grid = pooled.fidelity_grid
p_mem_max     = pooled.p_mem_max
τ_max         = pooled.τ_max

tnames = keys(timing_grid)
fnames = keys(fidelity_grid)
anames = (tnames..., fnames...)
length(anames) == 9 ||
    error("expected 5 timing + 4 fidelity axes, got $(length(tnames)) + $(length(fnames))")

points = NTuple{9, Int}[r.joint for r in rows]

## --------------------------------------------------------------------------- the frontier

front = pareto_maxima(points)
on_front = Set(front)

# A joint index fixes h_t, hence the member it came from; anything else means two Step-4 files
# describe the same timing configuration.
members_of = Dict{NTuple{9, Int}, Set{Int}}()
for r in rows
    r.joint in on_front || continue
    push!(get!(members_of, r.joint, Set{Int}()), r.member)
end
for (joint, ms) in members_of
    length(ms) == 1 ||
        @warn "one frontier point is reported by several members; the Step-4 folder has " *
              "duplicate timing configurations" joint members=sort(collect(ms))
end

# Collapse over the cutoff: keep the row with the most slack against ε*, breaking ties on the
# faster cycle time, and record how many cutoffs reach the same hardware point.
slack(r) = r.eps_star - r.eps_hat
best = Dict{NTuple{9, Int}, Any}()
n_cutoffs = Dict{NTuple{9, Int}, Int}()
for r in rows
    r.joint in on_front || continue
    n_cutoffs[r.joint] = get(n_cutoffs, r.joint, 0) + 1
    cur = get(best, r.joint, nothing)
    if cur === nothing || (slack(r), -r.τ) > (slack(cur), -cur.τ)
        best[r.joint] = r
    end
end

kept = keep_cutoff ? [r for r in rows if r.joint in on_front] : [best[j] for j in front]
# Least demanding overall first, so the top of the table is the cheapest hardware.
sort!(kept; by = r -> (-sum(r.joint), r.joint, r.cutoff))

step5_table = DataFrame([
    merge(r.values, (
        member = r.member, cutoff = r.cutoff, k = r.k, p_kept = r.p_kept,
        τ = r.τ, p_mem = r.p_mem,
        eps_star = r.eps_star, eps_hat = r.eps_hat,
        margin = r.eps_star - r.eps_hat,
        margin_rel = (r.eps_star - r.eps_hat) / r.eps_star,
        n_cutoffs = n_cutoffs[r.joint],
        index_sum = sum(r.joint),
    ), NamedTuple{ntuple(d -> Symbol("idx_", anames[d]), 9)}(r.joint))
    for r in kept
])

step5_coverage = DataFrame(coverage)
step5_status = DataFrame(status = collect(keys(status_counts)),
                         n = collect(values(status_counts)))
isempty(step5_status) || sort!(step5_status, :n; rev = true)

## ------------------------------------------------------------------------------- reporting

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
fmt_axis(name, v) = name in (:F_link, :F_CNOT, :F_readout) ? @sprintf("%.6g", v) :
                    name === :T_coherence ? @sprintf("%g s", v) :
                    name === :link_success_prob ? @sprintf("%g", v) : fmt_t(v)

n_missing = count(r -> r.status === :missing_bfs, coverage)
n_empty   = count(r -> r.status === :no_feasible_point, coverage)

println()
println("Step 5: ", length(front), " minimal feasible (h_t, h_f) of ", length(unique(points)),
        " distinct pooled points (", length(points), " Step-4 rows over ",
        count(r -> r.status === :ok, coverage), " members)")
println("        grid: ", prod(length, timing_grid), " timing × ", prod(length, fidelity_grid),
        " fidelity = ", prod(length, timing_grid) * prod(length, fidelity_grid), " joint points")
isfinite(p_mem_max) && @printf("        p_mem_max = %.4g, τ_max = %.4g ms\n", p_mem_max, 1e3τ_max)
n_missing > 0 && println("        ", n_missing, " member(s) have Step-3 but no Step-4 file — ",
                         "the frontier may be incomplete")
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

## ------------------------------------------------------------------------------------ save

out_jld2 = joinpath(output_path, "step5_minimal_requirements_$(code).jld2")
jldsave(out_jld2;
    step5_table, step5_coverage, step5_status,
    timing_grid, fidelity_grid, p_mem_max, τ_max,
    keep_cutoff, n_pooled = length(points), n_frontier = length(front))

if write_csv
    out_csv = joinpath(output_path, "step5_minimal_requirements_$(code).csv")
    CSV.write(out_csv, step5_table)
    println("Step 5: wrote $(nrow(step5_table)) frontier rows to\n  $(out_jld2)\n  $(out_csv)")
else
    println("Step 5: wrote $(nrow(step5_table)) frontier rows to\n  $(out_jld2)")
end
@printf("        (%.1f s)\n", time() - t_start)
