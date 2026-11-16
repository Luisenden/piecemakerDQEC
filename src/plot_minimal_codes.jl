# Figure: the least demanding hardware point of every code, read across all nine hardware axes.
#
# Reads the pooled (singlezone + multizone) fronts that plot_minimal_boxplot.jl writes as
# combined_minimal_requirements_<code>.csv. Per code, the front point with the smallest index sum
# Σ_i k_i is drawn, k_i = value index of axis i in the full search space (search_grid.jl), i.e. its
# x position in the figure.
# Ties on Σk are broken as in plot_minimal_boxplot.jl: most slack against ε* (eps_star − eps_hat),
# then shorter τ̂. The number of tied points is printed.
#
# Layout as in plot_minimal_rows.jl: one row per axis (y), and along x one integer slot per value
# of the full search space (SEARCH_GRID in search_grid.jl), ordered from
# least demanding (x = 1) to most demanding. Slots past the last value of an axis are shaded grey.
#
# Every code is one series (colour + marker); the legend gives its Σk, ε̂, τ̂ and cutoff c.
#
# Usage:
#   julia --project=src src/plot_minimal_codes.jl [input_path] [output_path]
#         [--codes=Steane713,BB12_2_3_partitioned,BB12_2_3,GB26_2_5]
#
#   Codes without a CSV in input_path are skipped with a warning.
#   writes <output_path>/combined_minimal_requirements_least_demanding_<code1>_<code2>_….pdf

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

known(f) = startswith(f, "--codes=")
all(known, flags) || error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --codes=a,b,c")

codes = let i = findfirst(startswith("--codes="), flags)
    i === nothing ? ["Steane713", "BB12_2_3_partitioned", "BB12_2_3", "GB26_2_5"] :
        String.(strip.(split(split(flags[i], "="; limit = 2)[2], ",")))
end
allunique(codes) || error("--codes lists a code twice: $(codes)")

const CODE_LATEX = Dict(
    "Steane713"            => "\\mathrm{Steane}\\ [[7,1,3]]",
    "BB12_2_3_partitioned" => "\\mathrm{BB12}\\ [[12,2,3]]",
    "BB12_2_3"             => "\\mathrm{BB12}\\ [[12,2,3]]\\ \\mathrm{non{-}partitioned}",
    "GB26_2_5"             => "\\mathrm{GB26}\\ [[26,2,5]]",
)
code_latex(c) = get(CODE_LATEX, c, "\\mathrm{$(replace(c, "_" => "\\_"))}")

const SERIES_COLORS  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const SERIES_MARKERS = [:circle, :rect, :utriangle, :diamond,
                        :dtriangle, :pentagon, :hexagon, :star5]
# Fixed style per known code, so a code looks the same whichever codes are drawn; other codes
# take the next unused style.
const CODE_STYLE = Dict("Steane713" => 1, "BB12_2_3_partitioned" => 2, "BB12_2_3" => 3, "GB26_2_5" => 4)

const TIMING = [:attempt_time, :link_success_prob, :Δt_CNOTgate, :Δt_readout, :Δt_rotation_shuttle]
const FIDELITY = [:F_link, :F_CNOT, :F_readout, :T_coherence]
const anames = [TIMING; FIDELITY]
idx_col(name) = Symbol("idx_", name)

include(joinpath(@__DIR__, "search_grid.jl"))   # SEARCH_GRID, search_slot, search_index_sum

slot_of(name, row) = search_slot(name, row[name])
sigma_k(row) = search_index_sum(name -> row[name], anames)

## Load

tables = Dict{String, DataFrame}()
for c in codes
    file = joinpath(input_path, "combined_minimal_requirements_$(c).csv")
    isfile(file) || (@warn "no $(file), $(c) skipped"; continue)
    t = CSV.read(file, DataFrame)
    nrow(t) == 0 && (@warn "$(file) is empty, $(c) skipped"; continue)
    tables[c] = t
end
codes = filter(in(keys(tables)), codes)
isempty(codes) && error("nothing to plot: no combined_minimal_requirements_<code>.csv in $(input_path)")
length(codes) <= length(SERIES_COLORS) ||
    error("$(length(codes)) codes, more than the $(length(SERIES_COLORS)) that stay distinguishable")

style_of = let free = setdiff(eachindex(SERIES_COLORS), values(CODE_STYLE)), s = Dict{String, Int}()
    for c in codes
        s[c] = haskey(CODE_STYLE, c) ? CODE_STYLE[c] : popfirst!(free)
    end
    s
end

## Least demanding point per code

picked = Dict{String, DataFrameRow}()
for c in codes
    t = tables[c]
    t.sigma_k = [sigma_k(r) for r in eachrow(t)]
    best = t[t.sigma_k .== minimum(t.sigma_k), :]
    n_tied = nrow(unique(best, [idx_col(n) for n in anames]))
    sort!(best, [:margin, :τ]; rev = [true, false])
    picked[c] = best[1, :]
    @printf("%-22s Σk = %d  (%d front point(s) tied, %d row(s) incl. cutoffs/layouts), drawn: %s, ε̂ = %.3g, τ̂ = %.3g ms\n",
            c, picked[c].sigma_k, n_tied, nrow(best), picked[c].zone, picked[c].eps_hat, 1e3 * picked[c].τ)
end

## Slots: the full search space, least demanding first

n_slots = [length(SEARCH_GRID[name]) for name in anames]
nx, ny = maximum(n_slots), length(anames)

fmt_t(x) = x >= 1e-3 ? @sprintf("%g ms", 1e3x) : @sprintf("%g µs", 1e6x)
fmt_axis(name, v) = name in (:F_link, :F_CNOT, :F_readout) ? @sprintf("%.6g", v) :
                    name === :T_coherence ? @sprintf("%g s", v) :
                    name === :link_success_prob ? @sprintf("%g", v) : fmt_t(v)

const AXIS_LATEX = Dict(
    :attempt_time => L"t_{\mathrm{att}}", :link_success_prob => L"p_{\mathrm{link}}",
    :Δt_CNOTgate => L"t_{\mathrm{CNOT}}", :Δt_readout => L"t_{\mathrm{ro}}",
    :Δt_rotation_shuttle => L"t_{\mathrm{buff}}", :F_link => L"F_{\mathrm{link}}",
    :F_CNOT => L"F_{\mathrm{CNOT}}", :F_readout => L"F_{\mathrm{ro}}",
    :T_coherence => L"T_{\mathrm{coh,comm}}")

"""`%.3g` of `x` as LaTeX, with any exponent written as `\\times 10^{k}`."""
function latex_g(x)
    m = match(r"^(-?[\d.]+)e([+-]\d+)$", @sprintf("%.3g", x))
    m === nothing && return @sprintf("%.3g", x)
    return "$(m[1]) \\times 10^{$(parse(Int, m[2]))}"
end
latex_ms(x) = isfinite(x) ? "$(latex_g(1e3 * x))\\,\\mathrm{ms}" : "\\infty"

series_label(c, row) = latexstring(code_latex(c), ":\\ \\Sigma k = $(sigma_k(row)),\\ ",
    "\\hat{\\varepsilon} = $(latex_g(row.eps_hat)),\\ \\hat{\\tau} = $(latex_ms(row.τ)),\\ ",
    "c = $(latex_ms(row.cutoff))")

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs, legendfontsize = fs,
        legendtitlefontsize = fs, titlefontsize = fs, plot_titlefontsize = fs,
        colorbar_tickfontsize = fs, colorbar_titlefontsize = fs, annotationfontsize = fs)

fig = plot(;
    xticks = 1:nx,
    yticks = (1:ny, [AXIS_LATEX[n] for n in anames]),
    xlabel = "value index (least → most demanding)",
    xlims = (0.5, nx + 0.5), ylims = (0.5, ny + 0.5),
    yflip = true,
    framestyle = :box,
    grid = false,
    legend = :outertop,
    left_margin = 4mm, bottom_margin = 4mm, right_margin = 2mm,
    size = (max(760, 90nx + 380), max(360, 52ny + 120 + 18length(codes))),
)

for (j, n) in enumerate(n_slots)
    n < nx && plot!(fig, Shape([n + 0.5, nx + 0.5, nx + 0.5, n + 0.5], [j - 0.5, j - 0.5, j + 0.5, j + 0.5]);
                    fillcolor = :gray85, linecolor = :gray85, label = false)
end
vline!(fig, 1.5:1:nx - 0.5; color = :gray80, linewidth = 0.5, label = false)
hline!(fig, 1.5:1:ny - 0.5; color = :gray80, linewidth = 0.5, label = false)
hline!(fig, [length(TIMING) + 0.5]; color = :gray40, linewidth = 1, label = false)

annotate!(fig, [(i, j + 0.32, text(fmt_axis(name, SEARCH_GRID[name][i]),fs, "Computer Modern", :gray45))
                for (j, name) in enumerate(anames) for i in 1:n_slots[j]])

# Codes sit between the top of the cell and the value text, spread evenly in the given order.
ns = length(codes)
dodge(k) = ns == 1 ? -0.08 : -0.34 + 0.42 * (k - 1) / (ns - 1)

for (k, c) in enumerate(codes)
    row = picked[c]
    scatter!(fig, [slot_of(name, row) for name in anames], (1:ny) .+ dodge(k);
        color = SERIES_COLORS[style_of[c]],
        markershape = SERIES_MARKERS[style_of[c]],
        markersize = 5,
        markerstrokecolor = :white,
        markerstrokewidth = 0.8,
        label = series_label(c, row),
    )
end

out_pdf = joinpath(output_path, "combined_minimal_requirements_least_demanding_$(join(codes, "_")).pdf")
savefig(fig, out_pdf)
println("wrote $(out_pdf)")
