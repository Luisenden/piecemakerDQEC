# Step 6: how far each minimal feasible solution is from the baseline hardware.
#
# Step 5 (run_minimal.jl) returned the Pareto front of least demanding hardware points
# (h_t, h_f) that still reach ε̂ ≤ ε*. Each of those points is a *target*; this script asks what
# it would cost to get there from `BASELINE`, the hardware we assume is available today.
#
# For every axis the improvement factor IF is the factor by which the baseline value has to be
# pushed to reach the solution's value:
#
#   IF > 1  -> the parameter has to be improved by that factor
#   IF = 1  -> the solution sits exactly at the baseline
#   IF < 1  -> the baseline already suffices; the solution *relaxes* that parameter
#
# The factor is defined per kind of parameter, so that "a factor of 2" means the same kind of
# effort on every axis (see `improvement_factors`):
#
#   :time      t_baseline / t          a 2× faster gate is a factor of 2
#   :ratio     T / T_baseline          a 2× longer coherence time is a factor of 2
#   :logprob   log(p_baseline)/log(p)  halving the number of orders of magnitude to 1 is a 2
#   :fidelity  same, on the depolarizing parameter (4F−1)/3, i.e. ~ the infidelity ratio
#
# No credit is given for a relaxed parameter when the per-solution effort is aggregated: the
# aggregates (IF_max, IF_total, IF_n_improved) are taken over max(IF, 1), because a parameter
# that may be worse than baseline does not pay for one that must be better.
#
# Step 5 writes one row per (h_t, h_f, c) when it was run with --keep-cutoff. Improvement
# factors depend on the hardware only, so the cutoff is collapsed away by default (keeping the
# row with the largest margin, as Step 5 does) — otherwise a hardware point reached by many
# cutoffs would be counted many times in every statistic below.
#
# Usage:
#   julia --project=src src/run_improvements.jl <step5_path> [output_path] [flags]
#
#   step5_path   step5_minimal_requirements_<code>.jld2, or the folder holding it (Step 5)
#   output_path  where step6_improvement_factors_<code>.{jld2,csv} are written; defaults to ./
#
#   --code=NAME            code tag in the file names (default Steane713)
#   --keep-cutoff          one row per (h_t, h_f, c) instead of one per (h_t, h_f)
#   --perfect-factor=X     IF of a parameter that is already perfect (F = 1), default 10
#   --top=N                number of solutions listed in the report (default 25)
#   --no-csv               skip the CSV next to the .jld2
#
# The figure is drawn separately, from the file this script writes:
#   julia --project=src src/plot_improvements.jl <output_path>

using JLD2
using DataFrames
using Statistics
using Printf
using CSV

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

hasflag(name) = ("--$(name)" in flags)
function flagval(name, default)
    i = findfirst(f -> startswith(f, "--$(name)="), flags)
    return i === nothing ? default : String(split(flags[i], "="; limit = 2)[2])
end

let known = ("code", "perfect-factor", "top")
    unknown = filter(flags) do f
        f in ("--keep-cutoff", "--no-csv") && return false
        return !any(k -> startswith(f, "--$(k)="), known)
    end
    isempty(unknown) || throw(ArgumentError("unknown flag(s): $(join(unknown, ", "))"))
end

const code  = flagval("code", "Steane713")
step5_path  = length(pos) >= 1 ? pos[1] : "./"
output_path = length(pos) >= 2 ? pos[2] : "./"
keep_cutoff = hasflag("keep-cutoff")
write_csv   = !hasflag("no-csv")
n_top       = parse(Int, flagval("top", "25"))

# A perfect parameter (F_readout = 1) has an infinite improvement factor under the :fidelity
# definition. Infinities would swallow every aggregate and every plot axis, so they are
# replaced by this stand-in — an arbitrary but explicit "needs a lot" marker.
const PERFECT_FACTOR = parse(Float64, flagval("perfect-factor", "10.0"))

## ------------------------------------------------------------------------ the baseline point

# Hardware assumed available today. Values are per axis of the Step-4/5 grids and need not lie
# on those grids — the baseline is a reference point, not a candidate solution.
const BASELINE = (
    attempt_time        = 10e-6,      # t_att
    Δt_CNOTgate         = 100e-6,     # t_CNOT
    Δt_readout          = 1e-3,       # t_ro
    Δt_rotation_shuttle = 100e-6,     # t_buff
    link_success_prob   = 1e-4,       # p_link
    F_link              = 0.97,       # F_link
    F_CNOT              = 0.9997,     # F_CNOT
    F_readout           = 0.9999,     # F_ro
    T_coherence         = 1.0,        # T_coh,comm
)

# How an axis is turned into an improvement factor; see the header.
const IF_KIND = (
    attempt_time        = :time,
    Δt_CNOTgate         = :time,
    Δt_readout          = :time,
    Δt_rotation_shuttle = :time,
    link_success_prob   = :logprob,
    F_link              = :fidelity,
    F_CNOT              = :fidelity,
    F_readout           = :fidelity,
    T_coherence         = :ratio,
)

# Report order: the axes grouped as the grids are (5 timing, then 4 fidelity).
const AXES = keys(BASELINE)

const AXIS_LABEL = (
    attempt_time        = "t_att",
    Δt_CNOTgate         = "t_CNOT",
    Δt_readout          = "t_ro",
    Δt_rotation_shuttle = "t_buff",
    link_success_prob   = "p_link",
    F_link              = "F_link",
    F_CNOT              = "F_CNOT",
    F_readout           = "F_ro",
    T_coherence         = "T_coh",
)

## ------------------------------------------------------------------- the improvement factors

"""Depolarizing parameter of a state/gate of fidelity `F`; (4F−1)/3 for one qubit."""
depolarizing_probability(F) = (4F - 1) / 3

"""Factor on a duration: the baseline has to get `b / v` times faster to reach `v`."""
time_factor(v, b) = b / v

"""Factor on a quantity that is simply scaled up, such as a coherence time."""
ratio_factor(v, b) = v / b

"""
Factor on a probability that is read in orders of magnitude: the ratio of the exponents,
log(b)/log(v). Reaching p = 1 exactly costs `PERFECT_FACTOR` rather than Inf.
"""
logprob_factor(v, b) = v >= 1 ? PERFECT_FACTOR : log(b) / log(v)

"""Factor on a fidelity, taken on the depolarizing parameter (≈ the infidelity ratio)."""
fidelity_factor(v, b) =
    logprob_factor(depolarizing_probability(v), depolarizing_probability(b))

"""
    improvement_factors(point; baseline = BASELINE) -> NamedTuple

The raw improvement factor per axis of `point` (anything indexable by the axis names: a
DataFrameRow or a NamedTuple). Factors below 1 mean the solution relaxes that axis.
"""
function improvement_factors(point; baseline::NamedTuple = BASELINE)
    factors = map(AXES) do name
        v, b, kind = point[name], baseline[name], IF_KIND[name]
        kind === :time     ? time_factor(v, b)     :
        kind === :ratio    ? ratio_factor(v, b)    :
        kind === :logprob  ? logprob_factor(v, b)  :
        kind === :fidelity ? fidelity_factor(v, b) :
        error("unknown improvement-factor kind $(kind) for $(name)")
    end
    return NamedTuple{AXES}(factors)
end

"""
    improvement_summary(point; baseline = BASELINE) -> NamedTuple

`improvement_factors` prefixed with `IF_`, plus the per-solution aggregates over the axes that
actually have to improve:

  `IF_max`         the single hardest requirement — the factor that gates the whole solution
  `IF_bottleneck`  which axis that is, or "none" if the baseline already reaches the point
  `IF_n_improved`  how many of the 9 axes have to improve at all
  `IF_total`       ∏ max(IF, 1), the combined effort, and `IF_log10_total` its log10
  `IF_relaxed`     how many axes the solution leaves below baseline (free slack)
"""
function improvement_summary(point; baseline::NamedTuple = BASELINE)
    f = improvement_factors(point; baseline = baseline)

    factors = Float64[f[name] for name in AXES]
    needed  = max.(factors, 1.0)                 # no credit for a relaxed parameter
    worst, i = findmax(needed)

    return merge(
        NamedTuple{ntuple(d -> Symbol("IF_", AXES[d]), length(AXES))}(Tuple(factors)),
        (IF_max         = worst,
         IF_bottleneck  = worst > 1.0 ? String(AXES[i]) : "none",
         IF_n_improved  = count(>(1.0), factors),
         IF_total       = prod(needed),
         IF_log10_total = sum(log10, needed),
         IF_relaxed     = count(<(1.0), factors)),
    )
end

## -------------------------------------------------------------------------- load the Step-5 file

in_jld2 = isdir(step5_path) ?
    joinpath(step5_path, "step5_minimal_requirements_$(code).jld2") : step5_path
isfile(in_jld2) || error("no Step-5 file at $(in_jld2)")

step5 = jldopen(in_jld2, "r") do f
    (step5_table = f["step5_table"],
     timing_grid = f["timing_grid"], fidelity_grid = f["fidelity_grid"],
     p_mem_max = f["p_mem_max"], τ_max = f["τ_max"],
     keep_cutoff = haskey(f, "keep_cutoff") ? f["keep_cutoff"] : false,
     n_frontier = haskey(f, "n_frontier") ? f["n_frontier"] : 0)
end

step5_table = step5.step5_table
nrow(step5_table) == 0 && error("the Step-5 table is empty: no frontier to compare")

for name in AXES
    hasproperty(step5_table, name) ||
        error("the Step-5 table has no column $(name); it was written for a different grid")
end

t_start = time()
@info "Step 6: improvement factors against the baseline" code in_jld2 n_rows =
    nrow(step5_table) step5_kept_cutoff = step5.keep_cutoff

## ------------------------------------------------------- collapse the cutoff, keep the hardware

idx_cols = [c for c in (Symbol("idx_", n) for n in AXES) if hasproperty(step5_table, c)]
# The joint index tuple identifies a hardware point exactly; fall back on the values if an older
# Step-5 file did not store the indices.
point_cols = length(idx_cols) == length(AXES) ? idx_cols : collect(AXES)

frontier = if keep_cutoff || !hasproperty(step5_table, :cutoff)
    copy(step5_table)
else
    # Same tie-break as Step 5: most slack against ε*, then the faster cycle time.
    sorted = sort(step5_table, [order(:margin, rev = true), :τ])
    collapsed = combine(groupby(sorted, point_cols; sort = false), first)
    select!(collapsed, names(step5_table))      # groupby/first reorders the columns
    collapsed
end

n_points = nrow(unique(select(step5_table, point_cols)))
n_points == nrow(frontier) || keep_cutoff ||
    @warn "collapsing the cutoff did not reduce to one row per hardware point" n_points n_rows =
        nrow(frontier)

## ------------------------------------------------------------------------- the factors per row

transform!(frontier,
    AsTable(collect(AXES)) => ByRow(r -> improvement_summary(r)) => AsTable)

# Easiest first: the smallest bottleneck, then the smallest overall effort. This is the order
# the frontier should be read in — the top row is the cheapest way to reach the target.
sort!(frontier, [:IF_max, :IF_total])

if_cols = [Symbol("IF_", n) for n in AXES]
step6_table = frontier

# Per axis across the frontier: how often it has to improve at all, and by how much. The
# minimum is the one that matters most — "there is a solution that only needs this much".
step6_axis = DataFrame(
    axis         = collect(String.(AXES)),
    label        = collect(String[AXIS_LABEL[n] for n in AXES]),
    kind         = String[String(IF_KIND[n]) for n in AXES],
    baseline     = collect(Float64[BASELINE[n] for n in AXES]),
    IF_min       = [minimum(step6_table[!, c]) for c in if_cols],
    IF_median    = [median(step6_table[!, c]) for c in if_cols],
    IF_max       = [maximum(step6_table[!, c]) for c in if_cols],
    n_improved   = [count(>(1.0), step6_table[!, c]) for c in if_cols],
    n_relaxed    = [count(<(1.0), step6_table[!, c]) for c in if_cols],
    n_bottleneck = [count(==(String(n)), step6_table.IF_bottleneck) for n in AXES],
)

## ------------------------------------------------------------------------------- reporting

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
fmt_axis(name, v) = name in (:F_link, :F_CNOT, :F_readout) ? @sprintf("%.6g", v) :
                    name === :T_coherence ? @sprintf("%g s", v) :
                    name === :link_success_prob ? @sprintf("%g", v) : fmt_t(v)
fmt_if(x) = x >= 100 ? @sprintf("%.0f", x) : x >= 10 ? @sprintf("%.1f", x) : @sprintf("%.2f", x)

n_reachable = count(==("none"), step6_table.IF_bottleneck)

println()
println("Step 6: ", nrow(step6_table), " minimal feasible solution(s)",
        keep_cutoff ? " (one per (h_t, h_f, c))" : " (one per (h_t, h_f))",
        " from ", nrow(step5_table), " Step-5 row(s)")
println("        baseline: ",
        join([string(AXIS_LABEL[n], " = ", fmt_axis(n, BASELINE[n])) for n in AXES], ", "))
isfinite(step5.p_mem_max) &&
    @printf("        target:   p_mem_max = %.4g, τ_max = %.4g ms\n", step5.p_mem_max, 1e3step5.τ_max)
if n_reachable > 0
    println("        ", n_reachable, " solution(s) need no improvement at all — the baseline ",
            "already reaches the target")
end

println()
println("Per axis over the whole frontier (IF > 1 = must improve):")
@printf("  %-22s %-9s %-12s %9s %9s %9s %8s %8s %8s\n",
        "axis", "label", "baseline", "IF_min", "IF_med", "IF_max",
        "n_impr", "n_relax", "n_bottl")
for r in eachrow(step6_axis)
    name = Symbol(r.axis)
    @printf("  %-22s %-9s %-12s %9s %9s %9s %8d %8d %8d\n",
            r.axis, r.label, fmt_axis(name, r.baseline),
            fmt_if(r.IF_min), fmt_if(r.IF_median), fmt_if(r.IF_max),
            r.n_improved, r.n_relaxed, r.n_bottleneck)
end

let never = step6_axis[step6_axis.n_improved .== 0, :label],
    always = step6_axis[step6_axis.n_improved .== nrow(step6_table), :label]
    isempty(never) ||
        println("  → never needs improving: ", join(never, ", "))
    isempty(always) ||
        println("  → every solution needs it: ", join(always, ", "))
end

println()
println("Cheapest solutions first (hardest single requirement, then combined effort):")
for (i, r) in enumerate(eachrow(step6_table))
    i > n_top && (println("  … and ", nrow(step6_table) - n_top, " more"); break)
    @printf("  [%3d] IF_max = %-7s (%-19s) n_improved = %d/%d  ∏IF = %-9s τ = %7.3g ms\n",
            i, fmt_if(r.IF_max), r.IF_bottleneck, r.IF_n_improved, length(AXES),
            fmt_if(r.IF_total), 1e3r.τ)
    @printf("        %s\n",
            join([string(AXIS_LABEL[n], " ", fmt_axis(n, r[n]), " (×", fmt_if(r[Symbol("IF_", n)]), ")")
                  for n in AXES], "  "))
end

let best = step6_table[1, :]
    println()
    println("Nearest solution to the baseline: IF_max = ", fmt_if(best.IF_max),
            " on ", best.IF_bottleneck, ", ", best.IF_n_improved, " axis/axes to improve, ",
            best.IF_relaxed, " relaxed")
end
println()

## ------------------------------------------------------------------------------------ save

out_jld2 = joinpath(output_path, "step6_improvement_factors_$(code).jld2")
jldsave(out_jld2;
    step6_table, step6_axis,
    baseline = BASELINE, if_kind = IF_KIND, perfect_factor = PERFECT_FACTOR,
    timing_grid = step5.timing_grid, fidelity_grid = step5.fidelity_grid,
    p_mem_max = step5.p_mem_max, τ_max = step5.τ_max,
    keep_cutoff, source = in_jld2)

if write_csv
    out_csv = joinpath(output_path, "step6_improvement_factors_$(code).csv")
    CSV.write(out_csv, step6_table)
    println("Step 6: wrote $(nrow(step6_table)) rows to\n  $(out_jld2)\n  $(out_csv)")
else
    println("Step 6: wrote $(nrow(step6_table)) rows to\n  $(out_jld2)")
end
@printf("        (%.1f s)\n", time() - t_start)

# The figures are drawn from the .jld2 above: src/plot_improvements.jl (the polar improvement
# histogram), src/plot_substitution.jl (which axes trade against each other), and
# src/plot_bottleneck.jl (which axis binds, against the cutoff tier).
