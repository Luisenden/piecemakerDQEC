# Figure for Step 5: a handful of selected minimal feasible solutions, read across all nine
# hardware axes at once.
#
# Each axis is one row of the figure (y). Along x, each axis gets one integer slot per distinct
# value the Step-5 table takes on it, ordered from least demanding (x = 1) to most demanding, so
# a point further right asks more of the hardware. Axes take different numbers of values; slots
# past the last value of an axis are shaded grey. The value of every slot is printed in it.
#
# Every selected row of the table is one series: a marker on each axis. Series are offset
# slightly within an axis row, so rows that share a value stay visible. The legend gives each
# series' ε̂, τ̂ and cutoff c.
#
# The rows are read from the Step-5 table as run_minimal.jl wrote them, never recomputed. Row
# numbers are positions in that table, i.e. the [i] that run_minimal.jl prints.
#
# Usage:
#   julia --project=src src/plot_minimal_rows.jl <code> [input_path] [output_path] [--rows=1,2,3] [--singlezone]
#
#   --rows defaults to the first five rows (the least demanding by index sum).
#   --singlezone reads/writes the single-fusion-zone files (<zone> = singlezone); default multizone.
#   writes <zone>_minimal_requirements_<code>_rows_<r1>_<r2>_….pdf.

using JLD2
using DataFrames
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

rows_flag = let i = findfirst(startswith("--rows="), flags)
    i === nothing ? "" : String(split(flags[i], "="; limit = 2)[2])
end
known(f) = startswith(f, "--rows=") || f == "--singlezone"
all(known, flags) ||
    error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --rows=1,2,3, --singlezone")
const zone = "--singlezone" in flags ? "singlezone" : "multizone"

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

@load joinpath(input_path, "$(zone)_minimal_requirements_$(code).jld2") step5_table timing_grid fidelity_grid
nrow(step5_table) == 0 && error("the Step-5 table is empty: nothing to plot")

const SERIES_COLORS  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const SERIES_MARKERS = [:circle, :rect, :utriangle, :diamond,
                        :dtriangle, :pentagon, :hexagon, :star5]

rows = rows_flag == "" ? collect(1:min(5, nrow(step5_table))) :
       [parse(Int, strip(s)) for s in split(rows_flag, ",")]
isempty(rows) && error("--rows selects no row")
allunique(rows) || error("--rows lists a row twice: $(rows)")
all(r -> 1 <= r <= nrow(step5_table), rows) ||
    error("--rows must lie in 1:$(nrow(step5_table)), got $(rows)")
length(rows) <= length(SERIES_COLORS) ||
    error("$(length(rows)) rows selected, more than the $(length(SERIES_COLORS)) that stay distinguishable")

## Slots

const grids = merge(timing_grid, fidelity_grid)
const anames = collect(keys(grids))

"""Ladder index of `v` on `axis`; index 1 is the most demanding value."""
function axis_index(axis, v)
    i = findfirst(a -> isapprox(a, v; rtol = 1e-9), axis)
    i === nothing && error("value $(v) is not on the saved grid $(axis)")
    return i
end

# Per axis, the ladder indices the whole table takes, largest (least demanding) first. The slots
# come from the whole table, not only the selected rows, so the x position of a value does not
# depend on which rows are drawn.
slots = Dict(name => sort(unique(axis_index(grids[name], v) for v in step5_table[!, name]); rev = true)
             for name in anames)
n_slots = [length(slots[name]) for name in anames]
nx, ny = maximum(n_slots), length(anames)

slot_of(name, v) = findfirst(==(axis_index(grids[name], v)), slots[name])

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
fmt_axis(name, v) = name in (:F_link, :F_CNOT, :F_readout) ? @sprintf("%.6g", v) :
                    name === :T_coherence ? @sprintf("%g s", v) :
                    name === :link_success_prob ? @sprintf("%g", v) : fmt_t(v)

const AXIS_LATEX = Dict(
    :attempt_time => L"t_{\mathrm{att}}", :Δt_CNOTgate => L"\Delta t_{\mathrm{CNOT}}",
    :Δt_readout => L"\Delta t_{\mathrm{readout}}", :Δt_rotation_shuttle => L"\Delta t_{\mathrm{rot/shuttle}}",
    :link_success_prob => L"p_{\mathrm{link}}", :F_link => L"F_{\mathrm{link}}",
    :F_CNOT => L"F_{\mathrm{CNOT}}", :F_readout => L"F_{\mathrm{readout}}",
    :T_coherence => L"T_{\mathrm{coh}}")
axis_label(name) = get(AXIS_LATEX, name, String(name))

"""`%.3g` of `x` as LaTeX, with any exponent written as `\\times 10^{k}`."""
function latex_g(x)
    m = match(r"^(-?[\d.]+)e([+-]\d+)$", @sprintf("%.3g", x))
    m === nothing && return @sprintf("%.3g", x)
    return "$(m[1]) \\times 10^{$(parse(Int, m[2]))}"
end

# Legend entry of a series: the ε̂, τ̂ and cutoff c the table holds for its row.
latex_ms(x) = isfinite(x) ? "$(latex_g(1e3 * x))\\,\\mathrm{ms}" : "\\infty"
series_label(row) = latexstring("\\hat{\\varepsilon} = $(latex_g(row.eps_hat)),\\ ",
                                "\\hat{\\tau} = $(latex_ms(row.τ)),\\ c = $(latex_ms(row.cutoff)),\\ k=$(row.k)")

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs, legendfontsize = fs - 2)

# Axes run top to bottom in grid order, timing first, so y grows downwards.
fig = plot(;
    xticks = 1:nx,
    yticks = (1:ny, axis_label.(anames)),
    xlabel = "value index (least → most demanding)",
    xlims = (0.5, nx + 0.5), ylims = (0.5, ny + 0.5),
    yflip = true,
    framestyle = :box,
    grid = false,
    legend = :outertop,
    left_margin = 4mm, bottom_margin = 4mm, right_margin = 2mm,
    size = (max(660, 90nx + 380), max(360, 52ny + 120)),
)

# Slots an axis does not have.
for (j, n) in enumerate(n_slots)
    n < nx && plot!(fig, Shape([n + 0.5, nx + 0.5, nx + 0.5, n + 0.5], [j - 0.5, j - 0.5, j + 0.5, j + 0.5]);
                    fillcolor = :gray85, linecolor = :gray85, label = false)
end
# Recessive cell boundaries, and a firmer line between the timing and the fidelity axes.
vline!(fig, 1.5:1:nx - 0.5; color = :gray80, linewidth = 0.5, label = false)
hline!(fig, 1.5:1:ny - 0.5; color = :gray80, linewidth = 0.5, label = false)
hline!(fig, [length(timing_grid) + 0.5]; color = :gray40, linewidth = 1, label = false)

# The value of every slot, small and grey along the bottom edge of its cell.
annotate!(fig, [(i, j + 0.32, text(fmt_axis(name, grids[name][slots[name][i]]), fs - 4, "Computer Modern", :gray45))
                for (j, name) in enumerate(anames) for i in 1:n_slots[j]])

# Series sit between the top of the cell and the value text, spread evenly in the selection order.
ns = length(rows)
dodge(k) = ns == 1 ? -0.08 : -0.32 + 0.48 * (k - 1) / (ns - 1)

for (k, r) in enumerate(rows)
    row = step5_table[r, :]
    xs = [slot_of(name, row[name]) for name in anames]
    ys = (1:ny) .+ dodge(k)
    scatter!(fig, xs, ys;
        color = SERIES_COLORS[k],
        markershape = SERIES_MARKERS[k],
        markersize = 5,
        markerstrokecolor = :white,     # keeps overlapping markers apart
        markerstrokewidth = 0.8,
        label = series_label(row),
    )
end

out_pdf = joinpath(output_path, "$(zone)_minimal_requirements_$(code)_rows_$(join(rows, "_")).pdf")
savefig(fig, out_pdf)
println("Step 5 rows: $(ns) of $(nrow(step5_table)) frontier rows over $(ny) axes ",
        "($(join(["$(n)=$(c)" for (n, c) in zip(anames, n_slots)], ", ")) slots), wrote $(out_pdf)")
