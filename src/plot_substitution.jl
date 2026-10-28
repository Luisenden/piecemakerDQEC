# Figure for Step 5: which hardware axes trade against each other on the frontier. Each cell of
# the matrix holds Spearman's ρ between two axes over the minimal feasible solutions:
#
#   ρ < 0   substitutes: the solutions that relax one axis demand more of the other
#   ρ > 0   complements: the two tighten together, one requirement in two columns
#   ρ ≈ 0   independent: the axis can be chosen without regard to the other
#
# The values are the ladder indices idx_<axis> of the Step-5 table as run_minimal.jl wrote them,
# never recomputed. Index 1 is the most demanding value on every axis, so all nine axes share one
# orientation. ρ is rank based, hence the same on the indices as on the values or on any
# improvement factor that grows with demand. With keep_cutoff = true a hardware point appears once
# per cutoff; it is counted once here, since the cutoff is not a hardware axis.
#
# The frontier is the set of minimal solutions, not a sample, so ρ describes the shape of that set
# and no p-value is reported. An axis that takes one value on the whole frontier has no ordering
# to correlate and is dropped.
#
# Usage:
#   julia --project=src src/plot_substitution.jl <code> [input_path] [output_path] [--within=AXIS] [--full] [--singlezone]
#
#   --within=AXIS   ρ inside each value of AXIS (levels with fewer than 4 points are skipped),
#                   pooled by Fisher-z with weights n - 3. Use it when one axis splits the frontier
#                   into blocks and so induces correlations between all the others, as F_link does.
#                   writes <zone>_minimal_requirements_<code>_substitution_within_<AXIS>.pdf.
#   --full          draw the whole matrix instead of the lower triangle.
#   --singlezone    read/write the single-fusion-zone files (<zone> = singlezone); default multizone.
#   writes <zone>_minimal_requirements_<code>_substitution.pdf.

using JLD2
using DataFrames
using StatsBase
using Printf
using LaTeXStrings
using Measures
using Plots
gr()

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

const code = length(pos) >= 1 ? pos[1] : ""
input_path = length(pos) >= 2 ? pos[2] : "./"
output_path = length(pos) >= 3 ? pos[3] : "./"

within_flag = let i = findfirst(startswith("--within="), flags)
    i === nothing ? "" : String(split(flags[i], "="; limit = 2)[2])
end
lower_only = !("--full" in flags)
known(f) = startswith(f, "--within=") || f in ("--full", "--singlezone")
all(known, flags) ||
    error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --within=AXIS, --full, --singlezone")
const zone = "--singlezone" in flags ? "singlezone" : "multizone"

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

@load joinpath(input_path, "$(zone)_minimal_requirements_$(code).jld2") step5_table timing_grid fidelity_grid
nrow(step5_table) == 0 && error("the Step-5 table is empty: nothing to correlate")

const anames = (keys(timing_grid)..., keys(fidelity_grid)...)
const idx = Dict(name => Symbol("idx_", name) for name in anames)

# One row per hardware point: the cutoff is a protocol knob and would only repeat points.
points = unique(step5_table, collect(values(idx)))

within = within_flag == "" ? nothing : Symbol(within_flag)
within === nothing || within in anames ||
    error("--within=$(within_flag) is not a hardware axis; the axes are $(join(anames, ", "))")

const AXIS_LATEX = Dict(
    :attempt_time => L"t_{\mathrm{att}}", :Δt_CNOTgate => L"t_{\mathrm{CNOT}}",
    :Δt_readout => L"t_{\mathrm{readout}}", :Δt_rotation_shuttle => L"t_{\mathrm{rot/shuttle}}",
    :link_success_prob => L"p_{\mathrm{link}}", :F_link => L"F_{\mathrm{link}}",
    :F_CNOT => L"F_{\mathrm{CNOT}}", :F_readout => L"F_{\mathrm{readout}}",
    :T_coherence => L"T_{\mathrm{coh}}")
axis_label(name) = get(AXIS_LATEX, name, String(name))

## Correlate

"""Spearman's ρ of `x` and `y`, or `NaN` when either is constant (the ranks are then all tied)."""
function spearman(x, y)
    length(x) >= 2 && !allequal(x) && !allequal(y) || return NaN
    return corspearman(Float64.(x), Float64.(y))
end

"""
ρ inside each block, pooled by averaging the Fisher-z transforms with weights `n - 3`. Blocks of
fewer than 4 points, or in which either column is constant, are skipped; `NaN` if none is left.
For a Spearman ρ the transform is approximate, but the error is a common factor of every block's
variance and cancels from a weighted average.
"""
function pooled_spearman(blocks, a, b)
    zsum, wsum = 0.0, 0.0
    for g in blocks
        nrow(g) >= 4 || continue
        r = spearman(g[!, a], g[!, b])
        isnan(r) && continue
        zsum += (nrow(g) - 3) * atanh(clamp(r, -0.999999, 0.999999))
        wsum += nrow(g) - 3
    end
    return wsum == 0 ? NaN : tanh(zsum / wsum)
end

correlate(a, b) = within === nothing ? spearman(points[!, a], points[!, b]) :
                  pooled_spearman(groupby(points, idx[within]), a, b)

# The diagonal is 1 wherever the axis varies (within a usable level), else the axis is dropped.
varies = [!isnan(correlate(idx[n], idx[n])) for n in anames]
kept, dropped = anames[varies], anames[.!varies]
length(kept) >= 2 || error("fewer than two axes vary" *
                           (within === nothing ? "" : " within the values of $(within)") *
                           " on the frontier: nothing to correlate")

n = length(kept)
ρ = [i == j ? 1.0 : correlate(idx[kept[i]], idx[kept[j]]) for i in 1:n, j in 1:n]

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs, legendfontsize = fs - 2)

# Diverging, blue for substitutes and red for complements, with a neutral at ρ = 0. The value is
# printed on every cell, so the reading never rests on colour alone.
const POLES = cgrad(["#2a78d6", "#f0efec", "#d03b3b"])
const NEUTRAL = "#f0efec"

# Each cell is its own filled square rather than a heatmap image, which PDF viewers smooth over.
# Row 1 is at the top, so it reads like a matrix.
cell(i, j; d = 0.47) = Shape([j - d, j + d, j + d, j - d], [n + 1 - i - d, n + 1 - i - d, n + 1 - i + d, n + 1 - i + d])
cell_text(i, j) = isnan(ρ[i, j]) ? text("n/a", fs - 3, "Computer Modern", :gray55) :
                  text(i == j ? "1" : @sprintf("%.2f", ρ[i, j]), fs - 3, "Computer Modern",
                       i != j && abs(ρ[i, j]) > 0.62 ? :white : :black)

labels = axis_label.(kept)
fig = plot(;
    xticks = (1:n, labels),
    yticks = (1:n, reverse(labels)),
    xrotation = n > 5 ? 45 : 0,
    xlims = (0.5, n + 0.5), ylims = (0.5, n + 0.5),
    aspect_ratio = :equal,
    framestyle = :grid,                 # tick labels without a box, which a triangle would leave half empty
    tick_direction = :none,
    grid = false,
    legend = false,
    left_margin = 4mm, bottom_margin = n > 5 ? 8mm : 4mm, right_margin = 4mm,
    size = (max(480, 60n + 220), max(420, 60n + 140)),
)
for i in 1:n, j in 1:n
    lower_only && j > i && continue
    # Open, not grey: at ρ = 0 the gradient is itself neutral, and "undefined" must not look
    # like "uncorrelated".
    fill = isnan(ρ[i, j]) ? :white : i == j ? NEUTRAL : POLES[(clamp(ρ[i, j], -1, 1) + 1) / 2]
    plot!(fig, cell(i, j); fillcolor = fill, linecolor = isnan(ρ[i, j]) ? :gray55 : :white,
          linestyle = isnan(ρ[i, j]) ? :dash : :solid, linewidth = 1)
end
annotate!(fig, [(j, n + 1 - i, cell_text(i, j)) for i in 1:n for j in 1:n if !(lower_only && j > i)])

# The colour bar, via an empty series carrying the colour range.
scatter!(fig, [NaN], [NaN]; marker_z = [0.0], c = POLES, clims = (-1, 1), colorbar = true,
         colorbar_title = L"\mathrm{Spearman}\ \rho", label = false)

suffix = within === nothing ? "" : "_within_$(within)"
out_pdf = joinpath(output_path, "$(zone)_minimal_requirements_$(code)_substitution$(suffix).pdf")
savefig(fig, out_pdf)

pairs = sort([(ρ[i, j], kept[i], kept[j]) for i in 1:n for j in 1:i - 1 if !isnan(ρ[i, j])]; by = first)
println("Step 5 substitution: $(nrow(points)) frontier points over $(n) axes",
        within === nothing ? "" : ", within the values of $(within)",
        isempty(dropped) ? "" : " ($(join(dropped, ", ")) constant, dropped)")
isempty(pairs) || println("  strongest substitute: $(pairs[1][2]) × $(pairs[1][3]), ρ = ",
                          @sprintf("%+.2f", pairs[1][1]),
                          "\n  strongest complement: $(pairs[end][2]) × $(pairs[end][3]), ρ = ",
                          @sprintf("%+.2f", pairs[end][1]))
println("  wrote $(out_pdf)")
