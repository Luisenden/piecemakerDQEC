# Figure: how the Pareto front of every code is spread over the ladder of each hardware axis.
#
# Reads the pooled (singlezone + multizone) fronts that plot_minimal_boxplot.jl writes as
# combined_minimal_requirements_<code>.csv. Those files hold one row per front point and per
# cutoff/layout it is feasible with; here every hardware point counts once (unique over the nine
# axis values), so n is the size of the Pareto set.
#
# One panel per hardware axis. Along x the values of that axis in the full search space (SEARCH_GRID
# in search_grid.jl), equally spaced in ladder order, least demanding on the left. Per level one bar
# per code, its height the fraction of that code's Pareto set sitting at level k on this axis, so
# the bars of one code sum to 1 in every panel.
#
# Every front point is feasible with some finite cutoff; some are also feasible without one
# (c = ∞). --cutoff splits each bar into the points that need a finite cutoff (shaded, at the
# bottom, so their share reads off the axis) and those also feasible at c = ∞ (solid, on top).
#
# Usage:
#   julia --project=src src/plot_pareto_levels.jl [input_path] [output_path]
#         [--codes=Steane713,BB12_2_3_partitioned] [--cutoff]
#
#   Codes without a CSV in input_path are skipped with a warning.
#   writes <output_path>/pareto_level_distribution_<code1>_<code2>_….pdf
#   (…_cutoff.pdf with --cutoff)

using CSV
using DataFrames
using Printf
using LaTeXStrings
using Measures
using Plots
gr()

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

input_path  = length(pos) >= 1 ? pos[1] : "./"
output_path = length(pos) >= 2 ? pos[2] : "./"

known(f) = startswith(f, "--codes=") || f == "--cutoff"
all(known, flags) ||
    error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --codes=a,b,c and --cutoff")
split_cutoff = "--cutoff" in flags

codes = let i = findfirst(startswith("--codes="), flags)
    i === nothing ? ["Steane713", "BB12_2_3_partitioned"] :
        String.(strip.(split(split(flags[i], "="; limit = 2)[2], ",")))
end
allunique(codes) || error("--codes lists a code twice: $(codes)")

# Short names as in plot_minimal_boxplot.jl: the partitioned BB12 is the default and plain "BB12".
const CODE_LABEL = Dict(
    "Steane713"            => "Steane",
    "BB12_2_3_partitioned" => "BB12",
    "BB12_2_3"             => "BB12 non-partitioned",
    "GB26_2_5"             => "GB26",
)

# Same palette and fixed per-code colours as plot_minimal_codes.jl and plot_minimal_boxplot.jl,
# so a code looks the same in every figure; other codes take the next unused colour.
const SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const CODE_STYLE = Dict("Steane713" => 1, "BB12_2_3_partitioned" => 2, "BB12_2_3" => 3, "GB26_2_5" => 4)

# Panel order: row-wise, timing axes first.
const anames = [:attempt_time, :Δt_CNOTgate, :Δt_readout,
                :Δt_rotation_shuttle, :link_success_prob, :F_link,
                :F_CNOT, :F_readout, :T_coherence]

include(joinpath(@__DIR__, "search_grid.jl"))   # SEARCH_GRID, search_slot

## Load: one row per hardware point

# needs_cutoff: none of the point's rows has c = ∞, i.e. it is feasible only with a finite cutoff.
tables = Dict{String, DataFrame}()
for c in codes
    file = joinpath(input_path, "combined_minimal_requirements_$(c).csv")
    isfile(file) || (@warn "no $(file), $(c) skipped"; continue)
    t = CSV.read(file, DataFrame)
    nrow(t) == 0 && (@warn "$(file) is empty, $(c) skipped"; continue)
    tables[c] = combine(groupby(t, anames), :cutoff => (x -> all(isfinite, x)) => :needs_cutoff)
end
codes = filter(in(keys(tables)), codes)
isempty(codes) && error("nothing to plot: no combined_minimal_requirements_<code>.csv in $(input_path)")
length(codes) <= length(SERIES_COLORS) ||
    error("$(length(codes)) codes, more than the $(length(SERIES_COLORS)) that stay distinguishable")

color_of = let free = setdiff(eachindex(SERIES_COLORS), values(CODE_STYLE)), s = Dict{String, String}()
    for c in codes
        s[c] = SERIES_COLORS[haskey(CODE_STYLE, c) ? CODE_STYLE[c] : popfirst!(free)]
    end
    s
end

"""
Fraction of the Pareto set of `code` at each ladder level of axis `name`, as `(all, cut)`:
all points, and those of them that need a finite cutoff. Both are shares of the whole set.
"""
function level_fractions(code, name)
    t = tables[code]
    slots = search_slot.(name, t[!, name])
    nk, n = length(SEARCH_GRID[name]), nrow(t)
    return (all = [count(==(k), slots) / n for k in 1:nk],
            cut = [count(slots .== k .&& t.needs_cutoff) / n for k in 1:nk])
end

fractions = Dict((c, n) => level_fractions(c, n) for c in codes for n in anames)

for c in codes
    @printf("%-22s n = %3d (%3d need a finite cutoff)   mean level per axis: %s\n",
            c, nrow(tables[c]), count(tables[c].needs_cutoff),
            join([@sprintf("%s %.2f", n, sum((1:length(f)) .* f)) for n in anames
                  for f in (fractions[(c, n)].all,)], ", "))
end

## Value labels on the x axis

# GR's Computer Modern has no Unicode superscripts, so values are LaTeX tick labels.

"""`x` as LaTeX `10^{e}` or `m \\cdot 10^{e}`, `0` for zero."""
function sci(x)
    x == 0 && return "0"
    e = floor(Int, log10(x) + 1e-9)
    m = round(x / 10.0^e; digits = 2)
    return m == 1 ? "10^{$(e)}" : "$(@sprintf("%g", m)) \\cdot 10^{$(e)}"
end

# Per axis: the panel title and how a value of the ladder is printed (in the title's units).
const AXIS = Dict(
    :attempt_time        => (title = L"t_{\mathrm{att}}\ [\mu\mathrm{s}]",  fmt = v -> @sprintf("%g", 1e6v)),
    :Δt_CNOTgate         => (title = L"t_{\mathrm{CNOT}}\ [\mu\mathrm{s}]", fmt = v -> @sprintf("%g", 1e6v)),
    :Δt_readout          => (title = L"t_{\mathrm{ro}}\ [\mathrm{ms}]",     fmt = v -> @sprintf("%g", 1e3v)),
    :Δt_rotation_shuttle => (title = L"t_{\mathrm{buff}}\ [\mu\mathrm{s}]", fmt = v -> @sprintf("%g", 1e6v)),
    :link_success_prob   => (title = L"p_{\mathrm{link}}",                  fmt = sci),
    :F_link              => (title = L"1 - F_{\mathrm{link}} = 2.5^{-x};\ \mathrm{value:}\ x",
                             fmt = v -> @sprintf("%g", round(-log(1 - v) / log(2.5); digits = 2))),
    # In units of 10⁻⁴: as m·10^e the ladder's ticks are too wide to stand side by side.
    :F_CNOT              => (title = L"1 - F_{\mathrm{CNOT}}\ [10^{-4}]",
                             fmt = v -> @sprintf("%g", round(1e4 * (1 - v); sigdigits = 3))),
    :F_readout           => (title = L"1 - F_{\mathrm{ro}}",                fmt = v -> sci(1 - v)),
    :T_coherence         => (title = L"T_{\mathrm{coh,comm}}\ [\mathrm{s}]", fmt = v -> @sprintf("%g", v)),
)

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs, legendfontsize = fs,
        legendtitlefontsize = fs, titlefontsize = fs, plot_titlefontsize = fs,
        colorbar_tickfontsize = fs, colorbar_titlefontsize = fs, annotationfontsize = fs)

# Opacity of the shaded (finite-cutoff-only) part of a bar under --cutoff.
const SHADE = 0.4

ncodes = length(codes)
group_width = 0.8
bar_w = group_width / ncodes
offset(j) = (j - (ncodes + 1) / 2) * bar_w

ncols = 3
nrows = cld(length(anames), ncols)

function panel(i, name)
    nk = length(SEARCH_GRID[name])
    left, bottom = mod1(i, ncols) == 1, cld(i, ncols) == nrows
    fig = plot(;
        title = AXIS[name].title,
        # Levels sit at equal spacing in ladder order; the tick names the physical value.
        xticks = (1:nk, [latexstring(AXIS[name].fmt(v)) for v in SEARCH_GRID[name]]),
        xlabel = bottom && mod1(i, ncols) == cld(ncols, 2) ?
                 L"\mathrm{Value\ (from\ least\ to\ most\ demanding)}" : "",
        xlims = (0.5, nk + 0.5), ylims = (0, 1.05),
        yticks = (0:0.5:1, left ? ["0.0", "0.5", "1.0"] : fill("", 3)),
        ylabel = left ? L"\mathrm{Fraction\ of}\ \mathcal{P}" : "",
        framestyle = :box, grid = :y, gridalpha = 0.2, gridlinewidth = 0.6,
        legend = false,
    )
    for (j, c) in enumerate(codes)
        f = fractions[(c, name)]
        # GR draws a hairline even at linewidth 0, so it takes the fill's colour (and opacity).
        bar_style = (bar_width = bar_w, color = color_of[c], linecolor = color_of[c], linewidth = 0,
                     label = false)
        if !split_cutoff
            ks = findall(>(0), f.all)    # empty levels get no bar, not a sliver on the axis
            isempty(ks) || bar!(fig, ks .+ offset(j), f.all[ks]; bar_style...)
            continue
        end
        # Shaded at the bottom: needs a finite cutoff. Solid on top: also feasible at c = ∞.
        ks = findall(>(0), f.cut)
        isempty(ks) || bar!(fig, ks .+ offset(j), f.cut[ks]; bar_style..., fillalpha = SHADE, linealpha = SHADE)
        ks = findall(f.all .> f.cut)
        isempty(ks) || bar!(fig, ks .+ offset(j), f.all[ks]; fillrange = f.cut[ks], bar_style...)
    end
    return fig
end

panels = [panel(i, n) for (i, n) in enumerate(anames)]

# A frameless top panel carries the one legend for the whole figure; at most four codes per row,
# so the labels stay within the figure width.
# With --cutoff two grey entries follow the codes and say what solid and shaded mean.
nentries = ncodes + (split_cutoff ? 2 : 0)
legend_cols = min(nentries, 4)
legend_h = 0.025 * cld(nentries, legend_cols)
legend_panel = plot(; framestyle = :none, legend = :top, legend_column = legend_cols,
                    foreground_color_legend = nothing, background_color_legend = nothing)
for c in codes
    bar!(legend_panel, [NaN], [NaN]; color = color_of[c], linewidth = 0,
         label = "$(get(CODE_LABEL, c, c)) (n = $(nrow(tables[c])))")
end
if split_cutoff
    bar!(legend_panel, [NaN], [NaN]; color = :gray45, linecolor = :gray45, linewidth = 0,
         label = L"\mathrm{cutoff\ policy}\ c = \infty")
    bar!(legend_panel, [NaN], [NaN]; color = :gray45, linecolor = :gray45, linewidth = 0,
         fillalpha = SHADE, linealpha = SHADE, label = L"c < \infty")
end

fig = plot(legend_panel, plot(panels...; layout = (nrows, ncols));
    layout = grid(2, 1; heights = [legend_h, 1 - legend_h]),
    size = (1000, 920 + round(Int, 920 * legend_h)),
    left_margin = 4mm, bottom_margin = 6mm, right_margin = 2mm,
)

out_pdf = joinpath(output_path,
                   "pareto_level_distribution_$(join(codes, "_"))$(split_cutoff ? "_cutoff" : "").pdf")
savefig(fig, out_pdf)
println("wrote $(out_pdf)")
