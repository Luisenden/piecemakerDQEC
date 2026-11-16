# Figure: the least demanding hardware per code, once the fusion-zone layout is also free to choose,
# summarised as box plots of its demand and/or its performance.
#
# run_minimal.jl writes one Step-5 frontier per code and per layout (singlezone_* / multizone_*).
# Here the two layouts of a code are pooled and the Pareto front is taken again over the union:
# a hardware point (h_t, h_f) is kept if no point feasible in either layout is at least as relaxed
# on every axis and more relaxed on one. Like the cutoff, the layout is treated as a design knob,
# not hardware. Pooling the two per-layout fronts is enough for the same reason pooling the
# per-member fronts is in run_minimal.jl: a globally minimal point is minimal in its own layout.
#
# Metrics (one panel each, --metrics=...):
#   ladder_sum  demand Σ_i k_i: k_i is the value index of axis i (1 = most relaxed value) in the
#               full search space (SEARCH_GRID in search_grid.jl), not in the grid a code
#               was run on, so the same physical value has the same k_i in every code and every
#               figure. Higher = more demanding.
#   tau         cycle time τ̂
#   pmem        memory error p̂_mem
#   eps         GHZ infidelity ε̂
#   tau_pmem    τ̂ on the left axis and p̂_mem on the right. p̂_mem is not simulated but derived,
#               p̂_mem = 3/4 (1 − exp(−τ̂ / T_coh,data)) with one fixed T_coh,data, so the two
#               share one box plot; the right-axis ticks sit at the τ̂ of each p̂_mem value.
#   cutoff      the cutoff time c, on a log axis since it spans several decades across codes (the
#               box, quartiles and whiskers are taken on log10 c). Points without a cutoff (c = ∞)
#               sit in a separate band at the top and are left out of the box.
# A front point can be reached with several cutoffs and in both layouts. For tau/pmem/eps/cutoff it
# is represented by the one with the most slack against ε* (ties: shorter τ̂), as run_minimal.jl
# does with keep_cutoff = false, so every panel has exactly one value per front point.
#
# One box per code (median, quartiles, whiskers at 1.5 IQR), with every front point drawn on top,
# jittered, in the code's colour (fixed per code, as in plot_minimal_codes.jl: Steane blue, BB12
# orange, BB12 non-partitioned green, GB26 yellow), and shaped by the layout(s) it is feasible in.
#
# Usage:
#   julia --project=src src/plot_minimal_boxplot.jl [input_path] [output_path]
#         [--codes=Steane713,BB12_2_3,BB12_2_3_partitioned] [--metrics=ladder_sum] [--no-zones]
#
#   --metrics   comma-separated subset of ladder_sum, tau, pmem, eps, tau_pmem, cutoff (default ladder_sum),
#               drawn side by side in the order given, e.g. --metrics=tau_pmem,eps
#   --no-zones  draws all front points as circles, without the layout legend.
#
#   writes <output_path>/pareto_<metrics>_boxplot_combined[_nozones].pdf and, per code,
#   combined_minimal_requirements_<code>.csv (the pooled front, one row per point and layout/cutoff).

using JLD2
using DataFrames
using CSV
using Printf
using Random
using Statistics
using LaTeXStrings
using Measures
using StatsPlots
gr()

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

input_path  = length(pos) >= 1 ? pos[1] :
    "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_minimal"
output_path = length(pos) >= 2 ? pos[2] : "./"

known(f) = startswith(f, "--codes=") || startswith(f, "--metrics=") || f == "--no-zones"
all(known, flags) ||
    error("unknown flag(s): $(join(filter(!known, flags), ", ")); only --codes=a,b,c, --metrics=a,b, --no-zones")
const color_by_zone = !("--no-zones" in flags)

"""The comma-separated list of `--<name>=...`, or `default` if the flag is absent."""
function list_flag(name, default)
    i = findfirst(startswith("--$(name)="), flags)
    i === nothing && return default
    return String.(strip.(split(split(flags[i], "="; limit = 2)[2], ",")))
end
codes   = list_flag("codes", ["Steane713", "BB12_2_3", "BB12_2_3_partitioned"])
metrics = list_flag("metrics", ["ladder_sum"])

const CODE_LABEL = Dict(
    "Steane713"            => "Steane [[7,1,3]]",
    "BB12_2_3"             => "BB12 [[12,2,3]]\nnon-partitioned",
    "BB12_2_3_partitioned" => "BB12 [[12,2,3]]",
    "GB26_2_5"             => "GB26 [[26,2,5]]",
)
const CODE_SHORT = Dict(
    "Steane713"            => "Steane",
    "BB12_2_3"             => "BB12\nnon-partitioned",
    "BB12_2_3_partitioned" => "BB12",
    "GB26_2_5"             => "GB26",
)

# Per metric: the front column it reads, the scale applied for display, whether it is drawn on a
# log axis, and the axis label.
const METRICS = Dict(
    "ladder_sum" => (col = :demand,  scale = 1,   integer = true,  log = false,
                     label = L"\mathrm{index\ sum}\ \Sigma_i\, k_i\ \mathrm{(higher = more\ demanding)}"),
    "tau"        => (col = :τ,       scale = 1e3, integer = false, log = false,
                     label = L"\mathrm{cycle\ time}\ \hat{\tau}\ \mathrm{[ms]}"),
    "pmem"       => (col = :p_mem,   scale = 1,   integer = false, log = false,
                     label = L"\mathrm{memory\ error}\ \hat{p}_{\mathrm{mem}}"),
    "eps"        => (col = :eps_hat, scale = 1e3, integer = false, log = false,
                     label = L"\mathrm{GHZ\ infidelity}\ \hat{\epsilon}\ [10^{-3}]"),
    "tau_pmem"   => (col = :τ,       scale = 1e3, integer = false, log = false,
                     label = L"\mathrm{cycle\ time}\ \hat{\tau}\ \mathrm{[ms]}"),
    "cutoff"     => (col = :cutoff,  scale = 1e3, integer = false, log = true,
                     label = L"\mathrm{cutoff}\ c\ \mathrm{[ms]}"),
)
isempty(metrics) && error("--metrics selects nothing")
all(in(keys(METRICS)), metrics) ||
    error("unknown metric(s) $(join(filter(!in(keys(METRICS)), metrics), ", ")); choose from $(join(sort(collect(keys(METRICS))), ", "))")
allunique(metrics) || error("--metrics lists a metric twice: $(metrics)")

const ZONES = ("singlezone", "multizone")

## Load and pool the two layouts of every code

"""The Step-5 table of `zone` for `code`, or `nothing` if run_minimal.jl never wrote it."""
function load_zone(code, zone)
    file = joinpath(input_path, "$(zone)_minimal_requirements_$(code).jld2")
    isfile(file) || return nothing
    @load file step5_table timing_grid fidelity_grid T_coh_data
    t = copy(step5_table)
    t[!, :zone] .= zone
    return (; table = t, grids = merge(timing_grid, fidelity_grid), T_coh_data)
end

loaded = Dict{String, Any}()
for code in codes
    parts = filter(!isnothing, [load_zone(code, z) for z in ZONES])
    isempty(parts) && (@warn "no singlezone/multizone file for $(code) in $(input_path), skipped"; continue)
    length(parts) < length(ZONES) &&
        @warn "$(code): only $(parts[1].table.zone[1]) found, the front is that layout's alone"
    # The front compares indices, so both layouts must sit on the same ladders.
    all(p -> p.grids == parts[1].grids, parts) ||
        error("$(code): singlezone and multizone were run on different grids")
    loaded[code] = (; table = reduce(vcat, [p.table for p in parts]; cols = :union),
                      grids = parts[1].grids, T_coh_data = [p.T_coh_data for p in parts])
end
codes = filter(in(keys(loaded)), codes)
isempty(codes) && error("nothing to plot: no minimal-requirements file for any requested code")

# The p̂_mem axis of tau_pmem is one map from τ̂, so it needs one data-qubit coherence time.
const T_COH_DATA = let Ts = unique(reduce(vcat, [loaded[c].T_coh_data for c in codes]))
    "tau_pmem" in metrics && length(Ts) > 1 &&
        error("tau_pmem needs one T_coh_data, but the files use $(Ts): p̂_mem is then not a function of τ̂ alone")
    first(Ts)
end
pmem_of_tau(τ) = 3 / 4 * (1 - exp(-τ / T_COH_DATA))
tau_of_pmem(p) = -T_COH_DATA * log(1 - 4p / 3)

const anames = collect(keys(loaded[codes[1]].grids))
const idx_names = [Symbol("idx_", n) for n in anames]
for c in codes
    collect(keys(loaded[c].grids)) == anames || error("$(c): axes differ from $(codes[1])")
end

include(joinpath(@__DIR__, "search_grid.jl"))   # SEARCH_GRID, search_index_sum

## Pareto front over the pooled layouts (same orientation as run_minimal.jl: larger index = more relaxed)

dominates(q, p) = all(q .>= p) && q != p

function pareto_maxima(points)
    front = eltype(points)[]
    for p in sort(unique(points); by = p -> (-sum(p), p))
        any(q -> dominates(q, p), front) || push!(front, p)
    end
    return front
end

"""Index sum Σk of a hardware point (ladder indices `joint` on `grids`), on the full search space."""
demand(joint, grids) = search_index_sum(name -> grids[name][joint[findfirst(==(name), anames)]], anames)

"""Layout label of a front point from the set of layouts it appears in."""
origin_of(zs) = (s = Set(zs); length(s) == 2 ? "both" : only(s) == "singlezone" ? "singlezone only" : "multizone only")

fronts = Dict{String, DataFrame}()
for code in codes
    t = loaded[code].table
    t.joint = collect(zip((t[!, n] for n in idx_names)...))
    on_front = Set(pareto_maxima(t.joint))
    rows = t[in.(t.joint, Ref(on_front)), :]
    rows.margin = rows.eps_star .- rows.eps_hat
    rows.demand = demand.(rows.joint, Ref(loaded[code].grids))

    # One entry per hardware point: the cutoff/layout with the most slack, labelled by every
    # layout the point is feasible in.
    origins = combine(groupby(rows, :joint), :zone => origin_of => :origin)
    pts = unique(sort(rows, [:margin, :τ]; rev = [true, false]), :joint)
    pts = innerjoin(select(pts, Not(:zone)), origins; on = :joint)
    fronts[code] = pts

    CSV.write(joinpath(output_path, "combined_minimal_requirements_$(code).csv"), select(rows, Not(:joint)))

    n_zone = Dict(z => length(unique(t.joint[t.zone .== z])) for z in ZONES)
    @printf("%-22s singlezone %3d + multizone %3d points -> combined front %3d  (%s)\n",
            code, get(n_zone, "singlezone", 0), get(n_zone, "multizone", 0), nrow(pts),
            join(["$(o): $(count(==(o), pts.origin))" for o in ("singlezone only", "multizone only", "both")], ", "))
    @printf("%-22s median  Σk = %.1f   τ̂ = %.3g ms   p̂_mem = %.3g   ε̂ = %.3g\n",
            "", median(pts.demand), 1e3median(pts.τ), median(pts.p_mem), median(pts.eps_hat))
    fin_c = filter(isfinite, pts.cutoff)
    @printf("%-22s cutoff  finite %3d (median %s ms, range %s ms)   c = ∞ %3d\n", "", length(fin_c),
            isempty(fin_c) ? "–" : @sprintf("%.3g", 1e3median(fin_c)),
            isempty(fin_c) ? "–" : @sprintf("%.3g–%.3g", (1e3 .* extrema(fin_c))...),
            count(isinf, pts.cutoff))
end

## Draw

fs = 12
default(fontfamily = "Computer Modern", tickfontsize = fs, guidefontsize = fs, legendfontsize = fs,
        legendtitlefontsize = fs, titlefontsize = fs, plot_titlefontsize = fs,
        colorbar_tickfontsize = fs, colorbar_titlefontsize = fs, annotationfontsize = fs)

# The layout(s) a front point is feasible in is shown by its marker shape.
const ORIGIN_SHAPE = Dict("singlezone only" => :circle, "multizone only" => :rect, "both" => :diamond)
const ORIGINS = ["singlezone only", "multizone only", "both"]
# One colour per code, for its box and its points: the same palette and fixed per-code colours as
# plot_minimal_codes.jl and plot_pareto_levels.jl, so a code looks the same in every figure;
# other codes take the next unused colour.
const SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const CODE_STYLE = Dict("Steane713" => 1, "BB12_2_3_partitioned" => 2, "BB12_2_3" => 3, "GB26_2_5" => 4)
length(codes) <= length(SERIES_COLORS) ||
    error("$(length(codes)) codes, more than the $(length(SERIES_COLORS)) that stay distinguishable")
const color_of = let free = setdiff(eachindex(SERIES_COLORS), values(CODE_STYLE)), s = Dict{String, String}()
    for c in codes
        s[c] = SERIES_COLORS[haskey(CODE_STYLE, c) ? CODE_STYLE[c] : popfirst!(free)]
    end
    s
end

multi = length(metrics) > 1
code_tick(c) = multi ? CODE_SHORT[c] : get(CODE_LABEL, c, c)

marker_size(shape) = shape === :circle ? 4 : 3.5

"""One panel: box per code of `metric`, front points on top."""
function panel(metric; show_legend)
    m = METRICS[metric]
    # Log metrics are drawn as log10 on a linear axis, so the box statistics are taken on the log.
    tr(v) = m.log ? log10(v) : v
    vals = Dict(c => tr.(m.scale .* fronts[c][!, m.col]) for c in codes)
    all_v = reduce(vcat, values(vals))
    fin_v = filter(isfinite, all_v)
    isempty(fin_v) && error("$(metric): no finite value to plot")
    lo, hi = extrema(fin_v)
    # A little room around the points; integer metrics keep integer steps.
    pad   = m.integer ? 1.5 : max(0.06 * (hi - lo), m.log ? 0.05 : 1e-9)
    # Infinite values (c = ∞) get their own band above a dashed line at the top of the panel.
    has_inf = any(isinf, all_v)
    y_inf   = hi + 3pad
    ylims   = has_inf ? (lo - pad, y_inf + 1.5pad) : (lo - pad, hi + pad)
    shown(v) = isinf(v) ? y_inf : v

    fig = plot(;
        xticks = (1:length(codes), code_tick.(codes)),
        xlims = (0.4, length(codes) + 0.6),
        ylims,
        # A narrow panel has no room for the long demand label.
        ylabel = multi && metric == "ladder_sum" ? L"\mathrm{index\ sum}\ \Sigma_i\, k_i" : m.label,
        framestyle = :box,
        grid = :y, gridalpha = 0.15, gridlinewidth = 0.6,
        # Demand rises left to right, performance metrics fall: put the legend in the empty corner.
        # The cutoff panel has its ∞ band along the top and Steane far above the lowest values.
        legend = show_legend ? (m.integer ? :topleft : m.log ? :bottomleft : :topright) : false,
    )
    m.integer && plot!(fig; yticks = ceil(Int, ylims[1]):floor(Int, ylims[2]))
    if m.log || has_inf
        # Log axis: ticks at 1, 2, 5 × 10^k (only decades if that gives too many); plus ∞.
        if m.log
            top = has_inf ? hi + 1.5pad : ylims[2]
            cand(ms) = sort([log10(f * 10.0^k) for k in floor(Int, ylims[1]):ceil(Int, top) for f in ms])
            ts = filter(t -> ylims[1] <= t <= top, cand([1, 2, 5]))
            length(ts) > 8 && (ts = filter(t -> ylims[1] <= t <= top, cand([1])))
            labels = [@sprintf("%g", 10.0^t) for t in ts]
        else
            ts, labels = Float64[], String[]   # keep the automatic ticks below
        end
        if has_inf
            plot!(fig, [0.4, length(codes) + 0.6], fill(hi + 1.5pad, 2);
                color = :gray50, linestyle = :dash, linewidth = 0.8, label = false)
        end
        m.log && plot!(fig; yticks = (vcat(ts, has_inf ? [y_inf] : Float64[]),
                                      vcat(labels, has_inf ? [L"\infty"] : String[])))
    end

    for (x, c) in enumerate(codes)
        fv = filter(isfinite, vals[c])
        isempty(fv) && continue
        boxplot!(fig, fill(x, length(fv)), fv;
            bar_width = 0.55, whisker_range = 1.5, outliers = false,
            fillcolor = color_of[c], fillalpha = 0.15, linecolor = color_of[c], linewidth = 1.2, label = false)
    end

    # Same seed in every panel, so a point sits at the same x in all of them.
    rng = MersenneTwister(1)
    groups = color_by_zone ? ORIGINS : [nothing]
    for origin in groups
        shape = origin === nothing ? :circle : ORIGIN_SHAPE[origin]
        for (x, c) in enumerate(codes)
            sel = origin === nothing ? trues(length(vals[c])) : fronts[c].origin .== origin
            d = vals[c][sel]
            isempty(d) && continue
            xs = x .+ 0.36 .* (rand(rng, length(d)) .- 0.5)
            # Integer metrics stack on whole numbers, so they get a little vertical jitter too.
            ys = m.integer ? d .+ 0.16 .* (rand(rng, length(d)) .- 0.5) : shown.(d)
            scatter!(fig, xs, ys;
                color = color_of[c], markershape = shape, markersize = marker_size(shape),
                markeralpha = 0.75, markerstrokecolor = :white, markerstrokewidth = 0.5, label = false)
        end
        # Legend: one neutral grey entry per layout, since colour now stands for the code.
        origin === nothing || scatter!(fig, [NaN], [NaN];
            color = :gray40, markershape = shape, markersize = marker_size(shape),
            markerstrokecolor = :white, markerstrokewidth = 0.5, label = origin)
    end

    if metric == "tau_pmem"
        # Right axis: round p̂_mem values, each placed at the τ̂ [ms] it corresponds to.
        p_lo, p_hi = pmem_of_tau.(max.(ylims, 0) ./ m.scale)
        step = something(findfirst(s -> (p_hi - p_lo) / s <= 7, [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]), 6)
        step = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05][step]
        ps = filter(p -> p_lo <= p <= p_hi, step .* (ceil(p_lo / step):floor(p_hi / step)))
        xlims = (0.4, length(codes) + 0.6)
        plot!(twinx(fig);
            ylims, xlims, xticks = false,
            yticks = (m.scale .* tau_of_pmem.(ps), [@sprintf("%g", p) for p in ps]),
            ylabel = L"\mathrm{memory\ error}\ \hat{p}_{\mathrm{mem}}",
            grid = false, legend = false)
        # twinx switches off the τ̂ grid and turns the box frame into bare axes: restore both.
        main = fig.subplots[1]
        main[:yaxis][:grid] = true
        # Half the line is clipped at the plot edge, so twice the frame width.
        plot!(main, [xlims...], fill(ylims[2], 2); color = :black, linewidth = 2, label = false)
    end
    return fig
end

# One legend per figure, preferably on a performance panel: their top-right corner is empty.
legend_panel = something(findfirst(m -> !METRICS[m].integer, metrics), 1)
panels = [panel(metric; show_legend = color_by_zone && i == legend_panel) for (i, metric) in enumerate(metrics)]
# The twin-axis panel needs extra width for its right-hand ticks and label.
widths = [m == "tau_pmem" ? 1.3 : 1.0 for m in metrics]
fig = plot(panels...;
    layout = multi ? grid(1, length(panels); widths = widths ./ sum(widths)) : (1, 1),
    size = multi ? (round(Int, 360 * sum(widths)), 440) : (680, 440),
    left_margin = multi ? 8mm : 5mm, bottom_margin = 6mm,
    right_margin = "tau_pmem" in metrics ? 8mm : 2mm,
)

# tau_pmem is written tau-pmem, so tau_pmem,eps and tau,pmem,eps get different files.
out_pdf = joinpath(output_path, "pareto_$(join(replace.(metrics, "tau_pmem" => "tau-pmem"), "_"))_boxplot_combined$(color_by_zone ? "" : "_nozones").pdf")
savefig(fig, out_pdf)
println("wrote $(out_pdf)")
