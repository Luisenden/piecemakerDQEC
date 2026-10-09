# Figure for Step 6: which hardware axes trade against each other on the frontier.
#
# The polar histogram (plot_improvements.jl) shows how far each axis has to move, one axis at a
# time. It cannot show whether two axes move *together*: whether a solution that spends less on
# T_coh has to spend more on t_att, or whether the two requirements are independent. That is the
# geometry of the frontier, and it is what this figure reports.
#
# For every pair of axes the cell holds Spearman's ρ between their improvement factors, taken
# over the rows of the Step-6 table:
#
#   ρ < 0   substitutes — the solutions that relax one axis demand more of the other; the
#           requirement can be moved between them, so neither is fixed on its own
#   ρ > 0   complements — the two tighten together; they are one requirement in two columns,
#           and buying one without the other buys nothing
#   ρ ≈ 0   independent — the axis can be chosen without regard to the other
#
# Spearman rather than Pearson because only the ordering of the factors is meaningful (they are
# ratios spanning decades, read on a log axis). Being rank based it is invariant under any
# monotone transform, so ρ on the factors and ρ on their log10 are the same number and the
# factors are used as they stand.
#
# Caveat, and the reason for `--within`: the frontier is not a sample from a population. It is
# the set of minimal solutions, so ρ describes the shape of that set and nothing else — it is a
# descriptive statistic here, not an inference, and no p-value is reported. An axis that takes
# one value across the whole table has no ordering to correlate and is dropped.
#
# `--within=AXIS` conditions on a discrete axis before correlating: ρ is computed inside each
# level of that axis and the levels are pooled (Fisher-z, weighted by n−3). Use it when one axis
# dominates the table and induces correlations between all the others simply by splitting it
# into blocks — F_link does exactly this on the Steane713 frontier, where it takes three values
# and is the bottleneck of nearly every solution.
#
# Usage:
#   julia --project=src src/plot_substitution.jl <step6_path> [output_path] [flags]
#
#   step6_path   step6_improvement_factors_<code>.jld2, or the folder holding it (Step 6)
#   output_path  where step6_substitution_<code>.pdf is written; defaults to ./
#
#   --code=NAME       code tag in the file names (default Steane713)
#   --within=AXIS     correlate inside each level of AXIS and pool the levels (see above)
#   --min-group=N     levels with fewer than N rows are skipped under --within (default 4)
#   --full            draw the whole matrix instead of the lower triangle
#   --csv             also write the matrix next to the figure
#   --out=FILE        write here instead of <output_path>/step6_substitution_<code>.pdf

using JLD2
using DataFrames
using Statistics
using StatsBase
using Printf
using LaTeXStrings
using Measures
using Colors
using Plots
using CSV
gr()

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

hasflag(name) = ("--$(name)" in flags)
function flagval(name, default)
    i = findfirst(f -> startswith(f, "--$(name)="), flags)
    return i === nothing ? default : String(split(flags[i], "="; limit = 2)[2])
end

let known = ("code", "within", "min-group", "out")
    unknown = filter(flags) do f
        f in ("--full", "--csv") && return false
        return !any(k -> startswith(f, "--$(k)="), known)
    end
    isempty(unknown) || throw(ArgumentError("unknown flag(s): $(join(unknown, ", "))"))
end

const code  = flagval("code", "Steane713")
step6_path  = length(pos) >= 1 ? pos[1] : "./"
output_path = length(pos) >= 2 ? pos[2] : "./"
lower_only  = !hasflag("full")
write_csv   = hasflag("csv")
min_group   = parse(Int, flagval("min-group", "4"))
min_group >= 3 || throw(ArgumentError("--min-group must be at least 3 (Fisher-z needs n > 3)"))

## -------------------------------------------------------------------------- load the Step-6 file

in_jld2 = isdir(step6_path) ?
    joinpath(step6_path, "step6_improvement_factors_$(code).jld2") : step6_path
isfile(in_jld2) || error("no Step-6 file at $(in_jld2)")

step6 = jldopen(in_jld2, "r") do f
    (step6_table = f["step6_table"], baseline = f["baseline"])
end

step6_table = step6.step6_table
nrow(step6_table) == 0 && error("the Step-6 table is empty: nothing to correlate")

# The baseline's axis names are the axes, in the order Step 6 reports them.
const AXES = keys(step6.baseline)

for name in AXES
    hasproperty(step6_table, Symbol("IF_", name)) ||
        error("the Step-6 table has no column IF_$(name); it was written for a different grid")
end

const AXIS_LATEX = (
    attempt_time        = L"t_{\mathrm{att}}",
    Δt_CNOTgate         = L"t_{\mathrm{CNOT}}",
    Δt_readout          = L"t_{\mathrm{ro}}",
    Δt_rotation_shuttle = L"t_{\mathrm{buff}}",
    link_success_prob   = L"p_{\mathrm{link}}",
    F_link              = L"F_{\mathrm{link}}",
    F_CNOT              = L"F_{\mathrm{CNOT}}",
    F_readout           = L"F_{\mathrm{ro}}",
    T_coherence         = L"T_{\mathrm{coh}}",
)

within_axis = let s = flagval("within", "")
    if isempty(s)
        nothing
    else
        sym = Symbol(s)
        sym in AXES || throw(ArgumentError("--within=$(s) is not one of $(join(String.(AXES), ", "))"))
        hasproperty(step6_table, sym) ||
            error("the Step-6 table has no column $(s) to condition on")
        sym
    end
end

## ------------------------------------------------------------------------ the rank correlation

"""Spearman's ρ of `x` and `y`, or `NaN` when either is constant (the ranks are then tied)."""
function spearman_or_nan(x::AbstractVector, y::AbstractVector)
    length(x) >= 2 || return NaN
    (minimum(x) == maximum(x) || minimum(y) == maximum(y)) && return NaN
    return corspearman(collect(x), collect(y))
end

"""
    pooled_spearman(blocks_x, blocks_y; min_group) -> Float64

ρ inside each block, pooled across blocks by averaging the Fisher-z transforms with weights
`n − 3` and transforming back. Blocks shorter than `min_group`, and blocks in which either
column is constant, carry no information about the association and are skipped; `NaN` means no
block survived.

The transform is the usual normalising one for a Pearson ρ. For a Spearman ρ it is an
approximation (its variance carries an extra factor ≈ 1.06), which is harmless here: that
factor is common to every block, so it cancels out of a weighted *average*.
"""
function pooled_spearman(blocks_x, blocks_y; min_group::Int = 4)
    zsum, wsum, used = 0.0, 0.0, 0
    for (x, y) in zip(blocks_x, blocks_y)
        length(x) >= max(min_group, 4) || continue
        ρ = spearman_or_nan(x, y)
        isnan(ρ) && continue
        w = length(x) - 3
        zsum += w * atanh(clamp(ρ, -0.999999, 0.999999))
        wsum += w
        used += 1
    end
    return (ρ = wsum == 0 ? NaN : tanh(zsum / wsum), n_blocks = used)
end

"""
    substitution_matrix(df, axes_; within, min_group) -> (ρ, n_blocks)

Symmetric matrix of the pairwise rank correlation between the `IF_<axis>` columns. With
`within` set, each entry is the pooled within-level ρ of that axis.
"""
function substitution_matrix(df, axes_; within::Union{Nothing, Symbol} = nothing,
                             min_group::Int = 4)
    n = length(axes_)
    columns = [Float64.(df[!, Symbol("IF_", name)]) for name in axes_]

    blocks = if within === nothing
        nothing
    else
        [[Float64.(g[!, Symbol("IF_", name)]) for name in axes_]
         for g in groupby(df, within; sort = true)]
    end

    ρ = fill(NaN, n, n)
    n_blocks = zeros(Int, n, n)

    for i in 1:n, j in i:n
        r, nb = if within === nothing
            (i == j ? (minimum(columns[i]) == maximum(columns[i]) ? NaN : 1.0) :
                      spearman_or_nan(columns[i], columns[j]), 1)
        elseif i == j
            # The diagonal is 1 wherever the axis varies inside at least one usable level.
            p = pooled_spearman((b[i] for b in blocks), (b[i] for b in blocks);
                                min_group = min_group)
            (isnan(p.ρ) ? NaN : 1.0, p.n_blocks)
        else
            p = pooled_spearman((b[i] for b in blocks), (b[j] for b in blocks);
                                min_group = min_group)
            (p.ρ, p.n_blocks)
        end
        ρ[i, j] = ρ[j, i] = r
        n_blocks[i, j] = n_blocks[j, i] = nb
    end

    return (ρ = ρ, n_blocks = n_blocks)
end

## --------------------------------------------------------------- drop the axes with no ordering

# An axis that is constant over the rows being correlated has no ranking to correlate: its whole
# row and column would be blank. Reporting that in the log is more useful than a stripe of empty
# cells in the figure.
full = substitution_matrix(step6_table, AXES; within = within_axis, min_group = min_group)

const KEPT = [i for i in 1:length(AXES) if !isnan(full.ρ[i, i])]
const DROPPED = [AXES[i] for i in 1:length(AXES) if isnan(full.ρ[i, i])]

length(KEPT) >= 2 ||
    error("fewer than two axes vary" * (within_axis === nothing ? "" :
          " within the levels of $(within_axis)") * ": nothing to correlate")

axes_kept = Tuple(AXES[i] for i in KEPT)
ρ = full.ρ[KEPT, KEPT]
n_blocks = full.n_blocks[KEPT, KEPT]

## ------------------------------------------------------------------------------- the heat map

"""
    substitution_heatmap(ρ; kwargs...) -> plot

Matrix of rank correlations as a diverging heat map: blue for the pairs that substitute for one
another (ρ < 0), red for the pairs that tighten together (ρ > 0), neutral at 0. Every cell also
carries its number, so the reading never rests on colour alone; a pair whose ρ is undefined is
left open and marked.

  `labels`      axis labels, in matrix order
  `lower_only`  draw the lower triangle only (the matrix is symmetric)
"""
function substitution_heatmap(
    ρ;
    labels,
    lower_only = true,
    # Diverging: two poles that read as opposite, with a neutral — never a hue — at the middle.
    poles = (colorant"#2a78d6", colorant"#f0efec", colorant"#d03b3b"),
    ink = colorant"#0b0b0b",
    muted = colorant"#8a8984",
)
    n = size(ρ, 1)
    gradient = cgrad([poles...])
    gap = 0.03                               # hairline of surface between cells

    p = plot(
        aspect_ratio = :equal,
        legend = false,
        grid = false,
        framestyle = :none,
        size = (660, 560),
        margin = 2mm,
    )

    # Row 1 at the top, so the matrix reads like a matrix rather than like a plot.
    rowy(i) = n + 1 - i

    for i in 1:n, j in 1:n
        lower_only && j > i && continue

        x0, x1 = j - 0.5 + gap, j + 0.5 - gap
        y0, y1 = rowy(i) - 0.5 + gap, rowy(i) + 0.5 - gap
        cell = Shape([x0, x1, x1, x0], [y0, y0, y1, y1])

        r = ρ[i, j]

        if isnan(r)
            # Open, not grey: at ρ = 0 the gradient is itself grey, and "undefined" must not
            # look like "uncorrelated".
            plot!(p, cell; seriestype = :shape, fillcolor = :white, linecolor = muted,
                  linestyle = :dash, linewidth = 0.8, label = false)
            annotate!(p, j, rowy(i), text("n/a", 7, "Computer Modern", muted))
            continue
        end

        # The diagonal carries no information; keep it visible but recessive.
        fill = i == j ? colorant"#f0efec" : gradient[(clamp(r, -1, 1) + 1) / 2]
        plot!(p, cell; seriestype = :shape, fillcolor = fill, linecolor = :white,
              linewidth = 0.6, label = false)

        txt = i == j ? "1" : @sprintf("%.2f", r)
        # White ink only where the fill is dark enough to swallow black.
        color = (i != j && abs(r) > 0.62) ? colorant"#ffffff" : ink
        annotate!(p, j, rowy(i), text(txt, 8, "Computer Modern", color))
    end

    # Row labels at the left, column labels under the bottom row — which is the one row that has
    # a cell in every column, in the triangular layout as much as in the full one.
    for k in 1:n
        annotate!(p, 0.35, rowy(k), text(labels[k], 11, "Computer Modern", :right))
        annotate!(p, k, rowy(n) - 0.62, text(labels[k], 11, "Computer Modern", :center, :top))
    end

    ## colour bar, via a dummy series
    scatter!(
        p,
        [NaN, NaN],
        [NaN, NaN];
        marker_z = [-1.0, 1.0],
        c = cgrad([poles...]),
        clims = (-1, 1),
        colorbar = true,
        colorbar_title = L"\mathrm{Spearman}\ \rho",
        colorbar_ticks = ([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"]),
        label = false,
        position = :right,
    )

    xlims!(p, -1.1, n + 0.8)
    ylims!(p, 0.0, n + 1.0)

    return p
end

## ------------------------------------------------------------------------------------ draw

fs = 12
prev_defaults = (
    fontfamily = Plots.default(:fontfamily),
    tickfontsize = Plots.default(:tickfontsize),
    guidefontsize = Plots.default(:guidefontsize),
    legendfontsize = Plots.default(:legendfontsize),
    titlefontsize = Plots.default(:titlefontsize),
)
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs,
        legendfontsize = fs, titlefontsize = fs)

fig = substitution_heatmap(
    ρ;
    labels = [AXIS_LATEX[name] for name in axes_kept],
    lower_only = lower_only,
)

suffix = within_axis === nothing ? "" : "_within_$(within_axis)"
out_pdf = let s = flagval("out", "")
    isempty(s) ? joinpath(output_path, "step6_substitution_$(code)$(suffix).pdf") : s
end
savefig(fig, out_pdf)
default(; prev_defaults...)

if write_csv
    out_csv = replace(out_pdf, r"\.pdf$" => ".csv")
    out_csv == out_pdf && (out_csv = out_pdf * ".csv")
    table = DataFrame(axis = collect(String.(axes_kept)))
    for (j, name) in enumerate(axes_kept)
        table[!, Symbol(name)] = ρ[:, j]
    end
    CSV.write(out_csv, table)
end

## -------------------------------------------------------------------------------- the report

offdiag = [ρ[i, j] for i in 1:size(ρ, 1), j in 1:size(ρ, 2) if i < j]
usable = filter(!isnan, offdiag)

println("Step 6 figure: substitution matrix over ", nrow(step6_table), " solution(s), ",
        length(axes_kept), " axes, ", length(usable), " pair(s)")
within_axis === nothing ||
    println("        correlated within the ", length(unique(step6_table[!, within_axis])),
            " level(s) of ", within_axis, ", pooled by Fisher-z (weights n−3, min ",
            min_group, " rows per level)")
isempty(DROPPED) ||
    println("        dropped (constant, no ordering to correlate): ",
            join(String.(DROPPED), ", "))

if !isempty(usable)
    pairs = [(ρ[i, j], axes_kept[i], axes_kept[j])
             for i in 1:size(ρ, 1), j in 1:size(ρ, 2) if i < j && !isnan(ρ[i, j])]
    sort!(pairs, by = first)
    show_n = min(3, length(pairs))
    println("        strongest substitutes (ρ < 0):")
    for (r, a, b) in pairs[1:show_n]
        @printf("          %-22s %-22s ρ = %+.2f\n", a, b, r)
    end
    println("        strongest complements (ρ > 0):")
    for (r, a, b) in reverse(pairs[end - show_n + 1:end])
        @printf("          %-22s %-22s ρ = %+.2f\n", a, b, r)
    end
end

println("        read\n  ", in_jld2, "\n        wrote\n  ", out_pdf)
write_csv && println("  ", replace(out_pdf, r"\.pdf$" => ".csv"))
