# Figure for Step 5: how the minimal feasible solutions distribute over two columns of the
# Step-5 table. Each cell of the heatmap counts the table rows that take that pair of values, and
# the count is printed on the cell.
#
# The rows are read from the Step-5 table as run_minimal.jl wrote them, never recomputed.
# Values run from small to large: left to right on x, bottom to top on y. Only values that at
# least one row takes are drawn, so no row or column of the heatmap is all zeros. c = ∞ comes
# last and is labelled ∞.
#
# Usage:
#   julia --project=src src/plot_minimal_counts.jl <code> <x_column> <y_column> [input_path] [output_path] [singlezone]
#
#   e.g. julia --project=src src/plot_minimal_counts.jl BB12_2_3 F_link T_coherence
#   singlezone = true reads/writes the single-fusion-zone files (<zone> = singlezone); default multizone.
#   writes <zone>_minimal_requirements_<code>_counts_<x_column>_<y_column>.pdf.

using JLD2
using DataFrames
using Printf
using LaTeXStrings
using Measures
using Plots
gr()

const code = length(ARGS) >= 1 ? ARGS[1] : ""
x_col = length(ARGS) >= 2 ? Symbol(ARGS[2]) : Symbol("")
y_col = length(ARGS) >= 3 ? Symbol(ARGS[3]) : Symbol("")
input_path = length(ARGS) >= 4 ? ARGS[4] : "./"
output_path = length(ARGS) >= 5 ? ARGS[5] : "./"
const zone = length(ARGS) >= 6 && parse(Bool, ARGS[6]) ? "singlezone" : "multizone"

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")
(x_col == Symbol("") || y_col == Symbol("")) && error("Missing the two columns to count over, e.g. F_link T_coherence")

@load joinpath(input_path, "$(zone)_minimal_requirements_$(code).jld2") step5_table timing_grid fidelity_grid
nrow(step5_table) == 0 && error("the Step-5 table is empty: nothing to plot")
for col in (x_col, y_col)
    hasproperty(step5_table, col) || error("$(col) is not a column of the Step-5 table; " *
                                           "it has $(join(propertynames(step5_table), ", "))")
end
x_col == y_col && error("the two columns must differ")

## Categories

const grids = merge(timing_grid, fidelity_grid)

"""The values drawn along `col`, ascending: the saved ladder for a hardware axis, else the values that occur."""
function categories(col)
    col in keys(grids) && return sort(collect(grids[col]))
    vals = sort(unique(step5_table[!, col]))
    length(vals) <= 30 || error("$(col) takes $(length(vals)) distinct values, too many to count " *
                                "cells over; pick a discrete column")
    return vals
end

"""Position of `v` among `cats`, tolerant of float round-off on the saved grids."""
function category_index(cats, v)
    i = findfirst(a -> a == v || (a isa Real && isapprox(a, v; rtol = 1e-9)), cats)
    i === nothing && error("value $(v) is not among the categories $(cats)")
    return i
end

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
function fmt_value(col, v)
    v isa Real && !isfinite(v) && return "∞"
    col in (:F_link, :F_CNOT, :F_readout) && return @sprintf("%.6g", v)
    col === :T_coherence && return @sprintf("%g s", v)
    col in keys(timing_grid) && col !== :link_success_prob && return fmt_t(v)
    col === :cutoff && return @sprintf("%.3g ms", 1e3v)
    v isa Integer && return string(v)
    return @sprintf("%.4g", v)
end

const AXIS_LATEX = Dict(
    :attempt_time => L"t_{\mathrm{att}}", :Δt_CNOTgate => L"\Delta t_{\mathrm{CNOT}}",
    :Δt_readout => L"\Delta t_{\mathrm{readout}}", :Δt_rotation_shuttle => L"\Delta t_{\mathrm{rot/shuttle}}",
    :link_success_prob => L"p_{\mathrm{link}}", :F_link => L"F_{\mathrm{link}}",
    :F_CNOT => L"F_{\mathrm{CNOT}}", :F_readout => L"F_{\mathrm{readout}}",
    :T_coherence => L"T_{\mathrm{coh}}", :cutoff => L"c", :k => L"k",
    :p_kept => L"p_{\mathrm{kept}}", :member => L"\mathrm{member}", :n_cutoffs => L"n_{c}")
axis_label(col) = get(AXIS_LATEX, col, String(col))

x_cats, y_cats = categories(x_col), categories(y_col)

## Count

counts = zeros(Int, length(y_cats), length(x_cats))    # heatmap wants rows along y
for r in eachrow(step5_table)
    counts[category_index(y_cats, r[y_col]), category_index(x_cats, r[x_col])] += 1
end

# A value no row takes would only add a line of zeros, so it is left out.
keep_y, keep_x = vec(any(>(0), counts; dims = 2)), vec(any(>(0), counts; dims = 1))
counts = counts[keep_y, keep_x]
x_cats, y_cats = x_cats[keep_x], y_cats[keep_y]

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs + 2)

# One hue from white (no rows) to the series blue of plot_minimal.jl. The count is printed on
# every cell, so the colour only has to carry the ordering and no colour bar is needed.
const SHADE = cgrad(["#ffffff", "#2a78d6"])
cmax = max(maximum(counts), 1)
cell_text(n) = text(string(n), fs - 1, "Computer Modern",
                    n == 0 ? :gray70 : n > cmax / 2 ? :white : :black)

# Each cell is its own filled square rather than a heatmap image, which PDF viewers smooth over.
# A white outline keeps neighbouring equal counts apart.
cell(i, j; d = 0.5) = Shape([i - d, i + d, i + d, i - d], [j - d, j - d, j + d, j + d])

nx, ny = length(x_cats), length(y_cats)
fig = plot(;
    xticks = (1:nx, [fmt_value(x_col, v) for v in x_cats]),
    yticks = (1:ny, [fmt_value(y_col, v) for v in y_cats]),
    xrotation = nx > 5 ? 45 : 0,
    xlabel = axis_label(x_col),
    ylabel = axis_label(y_col),
    xlims = (0.5, nx + 0.5), ylims = (0.5, ny + 0.5),
    framestyle = :box,
    grid = false,
    legend = false,
    left_margin = 4mm, bottom_margin = nx > 5 ? 8mm : 4mm, right_margin = 4mm,
    size = (max(420, 80nx + 160), max(320, 50ny + 140)),
)
for i in 1:nx, j in 1:ny
    plot!(fig, cell(i, j); fillcolor = SHADE[counts[j, i] / cmax], linecolor = :white, linewidth = 2)
end
annotate!(fig, [(i, j, cell_text(counts[j, i])) for i in 1:nx for j in 1:ny])

out_pdf = joinpath(output_path, "$(zone)_minimal_requirements_$(code)_counts_$(x_col)_$(y_col).pdf")
savefig(fig, out_pdf)
println("Step 5 counts: $(nrow(step5_table)) frontier rows over $(x_col) × $(y_col) ",
        "($(nx) × $(ny) cells, $(count(>(0), counts)) occupied), wrote $(out_pdf)")
