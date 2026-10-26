# Figure for Step 5: at the hardware configuration (h_t, h_f) of one selected row, the memory
# error p_mem (or another column, see --left) and the GHZ infidelity ε̂ against the nominal
# quantile p_k = 1/k of the cutoff c_k = Q̂_{T_GHZ}(1/k), over one shared p axis: the --left
# column on the left y axis, ε̂ on the right.
#
# Every row of the Step-5 table with the same nine axis values as the selected row is the same
# configuration at a different cutoff, so together they give one point per cutoff policy.
#
# The values are read from the Step-5 table as run_minimal.jl wrote them, never recomputed. Only
# cutoffs at which (h_t, h_f) is on the frontier appear; with keep_cutoff = false run_minimal.jl
# kept a single cutoff per point, so there is only one. c = ∞ (k = 0) has no quantile and is left out.
#
# Usage:
#   julia --project=src src/plot_minimal_p.jl <code> [input_path] [output_path] --row=i [--left=COLUMN]
#
#   --row=i          row i of the Step-5 table (the [i] run_minimal.jl prints)
#   --left=COLUMN    column on the left y axis, default p_mem; e.g. tau (= τ), cutoff, eps_star, margin
#   writes minimal_requirements_<code>_vs_p_row<i>[_<COLUMN>].pdf

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

flag_value(name) = let i = findfirst(startswith("--$(name)="), flags)
    i === nothing ? "" : String(split(flags[i], "="; limit = 2)[2])
end
row_flag = flag_value("row")
left_flag = flag_value("left")
known(f) = startswith(f, "--row=") || startswith(f, "--left=")
all(known, flags) ||
    error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --row=i, --left=COLUMN")

code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")
row_flag == "" && error("Missing --row=i: the row whose configuration to plot")

@load joinpath(input_path, "minimal_requirements_$(code).jld2") step5_table timing_grid fidelity_grid keep_cutoff
nrow(step5_table) == 0 && error("the Step-5 table is empty: nothing to plot")

tnames, fnames = collect(keys(timing_grid)), collect(keys(fidelity_grid))

# The left y axis: symbol, unit and the factor that brings the stored value into that unit.
# Columns not listed are plotted as stored, under their own name.
const LEFT_AXIS = Dict(
    :p_mem => ("p_{\\mathrm{mem}}", "\\%", 100), :τ => ("\\hat{\\tau}", "\\mathrm{ms}", 1e3),
    :cutoff => ("c", "\\mathrm{ms}", 1e3), :eps_star => ("\\varepsilon^*", "\\%", 100),
    :margin => ("\\varepsilon^* - \\hat{\\varepsilon}", "\\%", 100),
    :margin_rel => ("(\\varepsilon^* - \\hat{\\varepsilon})/\\varepsilon^*", "", 1))
left = Symbol(left_flag == "" ? "p_mem" : left_flag == "tau" ? "τ" : left_flag)
hasproperty(step5_table, left) && eltype(step5_table[!, left]) <: Real ||
    error("--left=$(left_flag) is not a numeric column of the Step-5 table; it has $(join(propertynames(step5_table), ", "))")
left_sym, left_unit, left_scale = get(LEFT_AXIS, left, ("\\mathrm{" * replace(String(left), "_" => "\\_") * "}", "", 1))
left_label = latexstring(left_sym, left_unit == "" ? "" : "\\;[" * left_unit * "]")

## Selection

row = parse(Int, row_flag)
1 <= row <= nrow(step5_table) || error("--row must lie in 1:$(nrow(step5_table)), got $(row)")
ref = step5_table[row, :]

sel = filter(r -> all(r[n] == ref[n] for n in (tnames..., fnames...)), step5_table)

keep_cutoff || @warn "run_minimal.jl ran with keep_cutoff = false: one cutoff per frontier point, so only one p_k is shown"

n_inf = count(==(0), sel.k)
sel = sel[sel.k .> 0, :]
nrow(sel) == 0 && error("row $(row) reaches the frontier only at c = ∞: no p_k to plot against")
allunique(sel.k) || error("the configuration of row $(row) has several rows at the same cutoff")
sort!(sel, :p_kept)

## Draw

const C_LEFT = "#2a78d6"
const C_EPS  = "#eb6834"

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs, legendfontsize = fs - 2)

# Explicit, so the shared p axis keeps a margin around its points. Both y axes also leave a band
# free at the top for the legends, so they never cover a point, wherever the curves run.
function padded(v; headroom = 0.0)
    lo, hi = extrema(v)
    span = hi > lo ? hi - lo : max(0.2 * abs(hi), 1e-12)
    (lo - 0.05span, hi + (0.05 + headroom) * span)
end
headroom = 0.22
xlims = padded(sel.p_kept)
y_left = left_scale .* sel[!, left]
ylims_left = padded(y_left; headroom)
ylims_eps = padded(100 .* sel.eps_hat; headroom)
marker = (markersize = 4, markerstrokecolor = :white, markerstrokewidth = 0.8)

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
fmt_axis(name, v) = name in (:F_link, :F_CNOT, :F_readout) ? @sprintf("%.6g", v) :
                    name === :T_coherence ? @sprintf("%g s", v) :
                    name === :link_success_prob ? @sprintf("%g", v) : fmt_t(v)
config_line(names) = join(["$(n) = $(fmt_axis(n, ref[n]))" for n in names], ",  ")

# The same symbols as in the other Step-5 figures.
const AXIS_LATEX = Dict(
    :attempt_time => "t_{\\mathrm{att}}", :Δt_CNOTgate => "\\Delta t_{\\mathrm{CNOT}}",
    :Δt_readout => "\\Delta t_{\\mathrm{readout}}", :Δt_rotation_shuttle => "\\Delta t_{\\mathrm{rot/shuttle}}",
    :link_success_prob => "p_{\\mathrm{link}}", :F_link => "F_{\\mathrm{link}}",
    :F_CNOT => "F_{\\mathrm{CNOT}}", :F_readout => "F_{\\mathrm{readout}}",
    :T_coherence => "T_{\\mathrm{coh}}")
latex_t(x) = x >= 1e-3 ? @sprintf("%g\\,\\mathrm{ms}", 1e3x) : @sprintf("%g\\,\\mu\\mathrm{s}", 1e6x)
# %.10g, since the F_link ladder holds values like 0.9998951424 that %.6g would round.
latex_value(name, v) = name in (:F_link, :F_CNOT, :F_readout, :link_success_prob) ? @sprintf("%.10g", v) :
                       name === :T_coherence ? @sprintf("%g\\,\\mathrm{s}", v) : latex_t(v)
latex_line(names) = latexstring(join(["$(get(AXIS_LATEX, n, "\\mathrm{$(n)}")) = $(latex_value(n, ref[n]))"
                                      for n in names], ",\\;\\; "))

fig = plot(sel.p_kept, y_left;
    framestyle = :box, gridalpha = 0.15, size = (760, 480),
    left_margin = 4mm, right_margin = 6mm, bottom_margin = 3mm,
    # GR's math text has no line break, so h_t goes in the figure title and h_f in the axis title.
    plot_title = latex_line(tnames), plot_titlefontsize = fs - 4, plot_titlevspan = 0.04,
    title = latex_line(fnames), titlefontsize = fs - 4,
    xlims,
    xlabel = L"\mathrm{cutoff\ policy}\ p_k = 1/k \mathrm{(surviving\ fraction\ of\ GHZ\ states\ under\ no\ cutoff)}",
    ylabel = left_label,
    ylims = ylims_left,
    color = C_LEFT, linewidth = 2, markershape = :circle, marker...,
    label = latexstring(left_sym, "\\;\\;"),  # GR sizes the box for the source text, not the glyphs
    legend = :topleft,
)

# twinx shares the p axis; each axis keeps its own legend, in opposite top corners.
ax_eps = twinx(fig)
plot!(ax_eps, sel.p_kept, 100 .* sel.eps_hat;
    xlims,
    # Transparent and gridless, else the twin paints over the left grid and legend.
    grid = false, background_color_subplot = RGBA(0, 0, 0, 0), background_color_inside = RGBA(0, 0, 0, 0),
    ylabel = L"\mathrm{GHZ\ infidelity}\ \hat{\varepsilon}\;[\%]",
    ylims = ylims_eps,
    color = C_EPS, linewidth = 2, linestyle = :dash, markershape = :rect, marker...,
    label = L"\hat{\varepsilon}\;\;\;",
    legend = :topright,
)
# twinx drops the left grid and turns the :box frame into :axes; take back the grid (the left
# ticks set it, as in the reference figure) and close the box at the top by hand.
fig.subplots[1][:yaxis][:grid] = true
hline!(fig, [last(ylims_left)]; color = :black, linewidth = 1, label = false)

suffix = left === :p_mem ? "" : "_" * replace(String(left), "τ" => "tau")
out_pdf = joinpath(output_path, "minimal_requirements_$(code)_vs_p_row$(row)$(suffix).pdf")
savefig(fig, out_pdf)

println("Step 5 figure: $(left) and ε̂ against p_k = 1/k for the configuration of row $(row), at ",
        nrow(sel), " cutoff(s)", n_inf > 0 ? " (c = ∞ left out)" : "")
println("  h_t: ", config_line(tnames))
println("  h_f: ", config_line(fnames))
println("  k:   ", join(sel.k, ", "))
println("  wrote $(out_pdf)")
