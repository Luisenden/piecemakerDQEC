# Figure for Step 5: ε̂ and the cycle time τ of every minimal feasible solution against the
# cutoff c, in two panels over one shared c axis.
#
# The values are read from the Step-5 table as run_minimal.jl wrote them, never recomputed.
# c is drawn on a log axis. c = ∞ (no cutoff) has no place on a numeric axis, so it is drawn in
# its own column right of a dashed break, labelled ∞.
#
# Usage:
#   julia --project=src src/plot_minimal.jl <code> [input_path] [output_path] [lines] [--group=COLUMN] [--singlezone]
#
#   lines = true connects the points in order of c; the default draws markers only.
#   --group=COLUMN draws one series per value of COLUMN, e.g. k (cutoff tier) or member, and
#   writes <zone>_minimal_requirements_<code>_by_<COLUMN>.pdf.
#   --singlezone reads/writes the single-fusion-zone files (<zone> = singlezone); default multizone.

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
lines = length(pos) >= 4 ? parse(Bool, pos[4]) : false

group = let i = findfirst(startswith("--group="), flags)
    i === nothing ? "" : String(split(flags[i], "="; limit = 2)[2])
end
known(f) = startswith(f, "--group=") || f == "--singlezone"
all(known, flags) ||
    error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --group=COLUMN, --singlezone")
const zone = "--singlezone" in flags ? "singlezone" : "multizone"

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

@load joinpath(input_path, "$(zone)_minimal_requirements_$(code).jld2") step5_table
nrow(step5_table) == 0 && error("the Step-5 table is empty: nothing to plot")

## Series

# Fixed order, so a value keeps its colour from figure to figure. The marker shape is a second
# encoding: past three series the hues alone are not safe for colour-blind readers.
const SERIES_COLORS  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const SERIES_MARKERS = [:circle, :rect, :utriangle, :diamond,
                        :dtriangle, :pentagon, :hexagon, :star5]

const GROUP_LATEX = Dict(:k => "k", :member => "\\mathrm{member}", :p_kept => "p_{\\mathrm{kept}}",
                         :n_cutoffs => "n_{c}", :cutoff => "c")

group_value(v) = v isa Real && !isfinite(v) ? "\\infty" :
                 v isa Integer ? string(v) : @sprintf("%.4g", v)

series = if group == ""
    [(label = "", df = step5_table)]
else
    col = Symbol(group)
    hasproperty(step5_table, col) || error("--group=$(group) is not a column of the Step-5 table; " *
                                           "it has $(join(propertynames(step5_table), ", "))")
    sym = get(GROUP_LATEX, col, "\\mathrm{" * replace(group, "_" => "\\_") * "}")
    [(label = latexstring(sym, " = ", group_value(first(g[!, col]))), df = DataFrame(g))
     for g in groupby(step5_table, col; sort = true)]
end
length(series) <= length(SERIES_COLORS) ||
    error("--group=$(group) gives $(length(series)) series, more than the $(length(SERIES_COLORS)) " *
          "that stay distinguishable; pick a column with fewer values")

## Axes

# Finite cutoffs on a log ms axis; c = ∞ in a column a quarter of the span right of them. Spans,
# padding and the ∞ column are all measured in decades, so they look the same on the log axis.
finite_c = 1e3 .* filter(isfinite, step5_table.cutoff)
all(>(0), finite_c) || error("a finite cutoff c <= 0 has no place on a log axis")
has_inf = any(!isfinite, step5_table.cutoff)
l_lo, l_hi = isempty(finite_c) ? (0.0, 1.0) : log10.(extrema(finite_c))
l_lo == l_hi && ((l_lo, l_hi) = (l_lo - 0.5, l_hi + 0.5))
l_span = l_hi - l_lo
l_inf = l_hi + 0.25l_span
c_lo, c_hi, c_inf = exp10.((l_lo, l_hi, l_inf))

# Decades if the range holds two or more; else 1-2-5 steps; else linear ticks inside the range.
in_c(t) = c_lo <= t <= c_hi
c_ticks = let decades = filter(in_c, [exp10(k) for k in floor(Int, l_lo):ceil(Int, l_hi)])
    steps = filter(in_c, [m * exp10(k) for k in floor(Int, l_lo):ceil(Int, l_hi) for m in (1, 2, 5)])
    length(decades) >= 2 ? decades :
    length(steps) >= 2   ? steps :
    filter(in_c, Plots.PlotUtils.optimize_ticks(c_lo, c_hi)[1])
end
c_labels = [@sprintf("%g", t) for t in c_ticks]
c_xticks = has_inf ? (vcat(c_ticks, c_inf), vcat(c_labels, "∞")) : (c_ticks, c_labels)
c_x(c) = isfinite(c) ? 1e3c : c_inf

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs, legendfontsize = fs - 2)

# Explicit, since the panels start empty and the ∞ break line would otherwise pull them out to 0.
function padded(v)
    lo, hi = extrema(v)
    pad = hi > lo ? 0.04 * (hi - lo) : 0.1 * abs(hi)
    (lo - pad, hi + pad)
end
common = (framestyle = :box, gridalpha = 0.15, left_margin = 4mm, right_margin = 3mm,
          xscale = :log,
          xlims = exp10.((l_lo - 0.08l_span, (has_inf ? l_inf : l_hi) + 0.08l_span)))

p_eps = plot(; common...,
    ylabel = L"\hat{\varepsilon}",
    yguidefontsize = fs + 4,            # a lone accented glyph reads small at the body size
    ylims = padded(step5_table.eps_hat),
    xticks = (c_xticks[1], fill("", length(c_xticks[1]))),  # c is labelled once, on the bottom panel
    bottom_margin = -2mm,
    legend = false,
)

p_τ = plot(; common...,
    xlabel = L"c\;[\mathrm{ms}]",
    ylabel = L"\tau\;[\mathrm{ms}]",
    xticks = c_xticks,
    ylims = padded(1e3 .* step5_table.τ),
    top_margin = -2mm,
    # Inside the panel: an outer legend would narrow this panel only and misalign the c axes.
    # Top left, since τ grows with c and :best does not avoid the ∞ column.
    legend = length(series) > 1 ? :topleft : false,
)
for p in (p_eps, p_τ)
    has_inf && vline!(p, [exp10(l_hi + 0.125l_span)]; color = :gray, linestyle = :dash, linewidth = 0.8,
                      label = false)
end

for (i, s) in enumerate(series)
    style = (
        seriestype = lines ? :path : :scatter,
        color = SERIES_COLORS[i],
        linewidth = 2,
        markershape = SERIES_MARKERS[i],
        markersize = 3,
        markerstrokecolor = :white,     # keeps overlapping markers apart
        markerstrokewidth = 0.8,
    )
    df = sort(s.df, [:cutoff, :eps_hat])
    plot!(p_eps, c_x.(df.cutoff), df.eps_hat; style..., label = false)

    dfτ = sort(s.df, [:cutoff, :τ])
    plot!(p_τ, c_x.(dfτ.cutoff), 1e3 .* dfτ.τ; style..., label = s.label)
end

fig = plot(p_eps, p_τ; layout = grid(2, 1), link = :x, size = (700, 560))

suffix = group == "" ? "" : "_by_$(group)"
out_pdf = joinpath(output_path, "$(zone)_minimal_requirements_$(code)$(suffix).pdf")
savefig(fig, out_pdf)
println("Step 5 figure: ε̂ and τ against c for $(nrow(step5_table)) frontier rows ",
        "($(count(!isfinite, step5_table.cutoff)) with c = ∞) in $(length(series)) series, ",
        "wrote $(out_pdf)")
