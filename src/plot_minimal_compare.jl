# Figure for Step 5: the partitioned and non-partitioned minimal feasible solutions of one code,
# side by side. For each value of one column (x) a pair of bars counts how many solutions of
# either set take that value.
#
# Both Step-5 tables are read as run_minimal.jl wrote them, never recomputed:
# <zone>_minimal_requirements_<code>.jld2 and <zone>_minimal_requirements_<code>_partitioned.jld2.
#
# A configuration that is feasible at several cutoffs appears once per cutoff in the table. To
# avoid counting it more than once, rows with the same nine hardware axis values are merged into
# one distinct hardware configuration whenever the x column depends on the hardware alone (one of
# the nine axes, or rate). For any other column, or with --all-rows, every table row is counted.
#
# Besides the table columns, x can be the derived column
#   rate = link_success_prob / attempt_time,
# the nominal Bell-pair generation rate per communication link, r = p_link / t_att.
#
# Values run from small to large, left to right. Only values that at least one row of either set
# takes are drawn. c = ∞ comes last and is labelled ∞.
#
# Usage:
#   julia --project=src src/plot_minimal_compare.jl <code> <x_column> [input_path] [output_path] [--all-rows] [--singlezone]
#
#   e.g. julia --project=src src/plot_minimal_compare.jl BB12_2_3 rate
#   --singlezone reads/writes the single-fusion-zone files (<zone> = singlezone); default multizone.
#   writes <zone>_minimal_requirements_<code>_compare_<x_column>[_allrows].pdf.

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
x_col = length(pos) >= 2 ? Symbol(pos[2]) : Symbol("")
input_path = length(pos) >= 3 ? pos[3] : "./"
output_path = length(pos) >= 4 ? pos[4] : "./"

known(f) = f in ("--all-rows", "--singlezone")
all(known, flags) ||
    error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --all-rows, --singlezone")
all_rows = "--all-rows" in flags
const zone = "--singlezone" in flags ? "singlezone" : "multizone"

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")
x_col == Symbol("") && error("Missing the column to count over, e.g. rate, F_link or T_coherence")

## Load

"""The Step-5 table of `name`, with the derived rate column, and the saved grids."""
function load_step5(name)
    file = joinpath(input_path, "$(zone)_minimal_requirements_$(name).jld2")
    isfile(file) || error("$(file) not found; run run_minimal.jl for $(name) first")
    @load file step5_table timing_grid fidelity_grid
    nrow(step5_table) == 0 && error("the Step-5 table of $(name) is empty: nothing to plot")
    # Rounded so that equal rates reached from different grid points (1e-4/1e-7 = 1e-3/1e-6)
    # fall into one bar despite float round-off.
    step5_table.rate = round.(step5_table.link_success_prob ./ step5_table.attempt_time; sigdigits = 6)
    return step5_table, timing_grid, fidelity_grid
end

const SETS = [(label = "Partitioned", name = "$(code)_partitioned"),
              (label = "Non-partitioned", name = code)]
loaded = [load_step5(s.name) for s in SETS]
timing_grid, fidelity_grid = loaded[1][2], loaded[1][3]
all(l -> l[2] == timing_grid && l[3] == fidelity_grid, loaded) ||
    @warn "the two sets were run on different grids; values are matched by value, not by grid index"

const anames = vcat(collect(keys(timing_grid)), collect(keys(fidelity_grid)))
for (s, (t, _, _)) in zip(SETS, loaded)
    hasproperty(t, x_col) || error("$(x_col) is not a column of the $(s.name) Step-5 table; " *
                                   "it has $(join(propertynames(t), ", "))")
end

# Merging rows into hardware configurations only keeps x well defined if x is fixed by the hardware.
hardware_only = x_col in anames || x_col === :rate
dedup = hardware_only && !all_rows
!hardware_only && !all_rows &&
    println("$(x_col) is not fixed by the hardware alone: counting every table row")
tables = [dedup ? unique(t, anames) : t for (t, _, _) in loaded]

## Categories

"""Position of `v` among `cats`, tolerant of float round-off."""
function category_index(cats, v)
    i = findfirst(a -> a == v || (a isa Real && isapprox(a, v; rtol = 1e-9)), cats)
    i === nothing && error("value $(v) is not among the categories $(cats)")
    return i
end

cats = sort(unique(vcat([t[!, x_col] for t in tables]...)))
cats = cats[[findfirst(a -> isapprox(a, v; rtol = 1e-9), cats) == i for (i, v) in enumerate(cats)]]
length(cats) <= 30 || error("$(x_col) takes $(length(cats)) distinct values, too many for bars; " *
                            "pick a discrete column")

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
function fmt_si(x)
    for (f, s) in ((1e9, "G"), (1e6, "M"), (1e3, "k"))
        x >= f && return @sprintf("%g%s", x / f, s)
    end
    return @sprintf("%g", x)
end
function fmt_value(col, v)
    v isa Real && !isfinite(v) && return "∞"
    col in (:F_link, :F_CNOT, :F_readout) && return @sprintf("%.6g", v)
    col === :T_coherence && return @sprintf("%g s", v)
    col === :rate && return fmt_si(v)
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
    :p_kept => L"p_{\mathrm{kept}}", :member => L"\mathrm{member}", :n_cutoffs => L"n_{c}",
    :rate => L"r = p_{\mathrm{link}} / t_{\mathrm{att}} \ [\mathrm{s}^{-1}]")
axis_label(col) = get(AXIS_LATEX, col, String(col))

## Count

counts = zeros(Int, length(cats), length(SETS))
for (j, t) in enumerate(tables), v in t[!, x_col]
    counts[category_index(cats, v), j] += 1
end

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs + 2, legendfontsize = fs)

# Orange and blue, the first two series colours of plot_minimal_rows.jl, stay apart under the
# common colour-vision deficiencies.
const SET_COLORS = ["#eb6834", "#2a78d6"]

nx, nset = length(cats), length(SETS)
w = 0.8 / nset                       # bar width; a pair fills 80 % of its slot
ymax = maximum(counts)

fig = plot(;
    xticks = (1:nx, [fmt_value(x_col, v) for v in cats]),
    xrotation = nx > 6 ? 45 : 0,
    xlabel = axis_label(x_col),
    ylabel = dedup ? "distinct hardware configurations" : "frontier rows",
    xlims = (0.4, nx + 0.6), ylims = (0, 1.15ymax),
    framestyle = :axes,
    grid = :y, gridalpha = 0.15, gridstyle = :dash,
    legend = :topleft, foreground_color_legend = nothing,
    left_margin = 4mm, bottom_margin = nx > 6 ? 8mm : 4mm, right_margin = 4mm,
    size = (max(480, 90nx + 160), 380),
)
for (j, s) in enumerate(SETS)
    # The 2 % trimmed off each side leaves a thin gap between the two bars of a pair.
    xs = (1:nx) .+ (j - (nset + 1) / 2) * w
    bar!(fig, xs, counts[:, j]; bar_width = 0.96w, fillcolor = SET_COLORS[j], linewidth = 0,
         label = "$(s.label) ($(nrow(tables[j])))")
    annotate!(fig, [(x, n, text(string(n), fs - 2, "Computer Modern", :gray30, :bottom))
                    for (x, n) in zip(xs, counts[:, j]) if n > 0])
end

suffix = dedup || !hardware_only ? "" : "_allrows"
out_pdf = joinpath(output_path, "$(zone)_minimal_requirements_$(code)_compare_$(x_col)$(suffix).pdf")
savefig(fig, out_pdf)
println("Step 5 compare over $(x_col): ",
        join(["$(s.label) $(nrow(t))" for (s, t) in zip(SETS, tables)], ", "),
        dedup ? " distinct hardware configurations" : " frontier rows",
        " over $(nx) values, wrote $(out_pdf)")
