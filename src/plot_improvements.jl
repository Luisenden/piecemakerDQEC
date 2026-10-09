# Figure for Step 6: the polar improvement histogram of a Step-6 table.
#
# Step 6 (run_improvements.jl) wrote one row per minimal feasible solution with the improvement
# factor of every hardware axis (the `IF_<axis>` columns). This script only draws them; it reads
# the factors out of the file and never recomputes them, so the figure always shows exactly what
# Step 6 reported.
#
# A spider plot shows one solution at a time, which is no use for a 204-point frontier. The
# polar histogram keeps the spider's angular layout — one sector per axis, radius = log10(IF) —
# but replaces the per-solution trace with a count: how many of the frontier's solutions land
# in each (axis, decade) cell. Reading it:
#
#   the solid IF = 1 ring   inside it the baseline already suffices, outside it the axis
#                           has to improve; a sector dark only outside the ring is a
#                           requirement no solution escapes
#   a sector's radial mass  how much that axis has to improve, and how much the frontier
#                           disagrees about it — a tight ring is a hard constraint, a smear
#                           is an axis that can be traded against the others
#   comparing sectors       which axis the whole frontier is pushed furthest out on
#
# Usage:
#   julia --project=src src/plot_improvements.jl <step6_path> [output_path] [flags]
#
#   step6_path   step6_improvement_factors_<code>.jld2, or the folder holding it (Step 6)
#   output_path  where step6_improvement_histogram_<code>.pdf is written; defaults to ./
#
#   --code=NAME       code tag in the file names (default Steane713)
#   --split=COLUMN    one panel per value of COLUMN, drawn side by side on a common radial range
#                     and a common colour scale. Two uses:
#
#                     --split=<axis>  when one hardware axis dominates the figure. On the
#                        Steane713 frontier F_link takes three values and is the bottleneck of
#                        nearly every solution, so its sector is a three-spike pattern that
#                        compresses the colour scale of the other eight. --split=F_link puts each
#                        of its values in its own panel and lets the rest of the figure use the
#                        full scale. The axis itself leaves the wheel: it is constant inside a
#                        panel, so its sector would be a single cell holding every solution of
#                        that panel — no information, and it would set the top of the colour
#                        scale on its own. Each panel shows the remaining axes *given* that value.
#
#                     --split=p_kept  (or k, or cutoff) to read the wheel against the protocol
#                        knob. The cutoff is not a hardware axis, so the wheel keeps all nine
#                        sectors and each panel shows the hardware the frontier demands *given*
#                        that cutoff tier. A sector that moves outward from panel to panel is an
#                        axis no cutoff choice buys you out of; one that moves inward is an axis
#                        the protocol pays for. p_kept = 1/k is the scale-free coordinate — the
#                        share of GHZ attempts the cutoff keeps — and is comparable across
#                        hardware in a way the cutoff in ms is not, since a slower machine has a
#                        longer generation-time tail and the same ms is a different quantile.
#                        This needs a Step-6 file built with --keep-cutoff: without it there is
#                        one cutoff per hardware point and it is whichever one Step 5's tie-break
#                        picked, so the panels would show that rule, not the physics.
#
#                     A split figure always colours by share (see --normalize): panels hold
#                     different numbers of solutions, so a raw count would make the biggest panel
#                     look like the strongest requirement. Each panel's title states how many
#                     solutions its share is of.
#   --split-bins=N    group the split column into N equal-count bins instead of one panel per
#                     distinct value. The cutoff takes a different value in nearly every row, so
#                     splitting on it needs this. Bin edges fall on real data values and each
#                     panel's title names the range it actually holds. A non-finite value always
#                     gets a panel of its own at the end — c = ∞, no cutoff at all, is a
#                     different protocol rather than a point on the range.
#   --range=LO,HI     radial range as factors, e.g. 1e-2,1e3 (default: fit the data)
#   --nbins=N         radial bins (default 15)
#   --normalize       colour by share of solutions instead of count (implied by --split)
#   --no-clip         drop values outside an explicit --range instead of folding them into the
#                     edge bins (they then vanish from the picture)
#   --out=FILE        write here instead of <output_path>/step6_improvement_histogram_<code>.pdf

using JLD2
using DataFrames
using Statistics
using Printf
using LaTeXStrings
using Measures
using Plots
gr()

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

hasflag(name) = ("--$(name)" in flags)
function flagval(name, default)
    i = findfirst(f -> startswith(f, "--$(name)="), flags)
    return i === nothing ? default : String(split(flags[i], "="; limit = 2)[2])
end

let known = ("code", "split", "split-bins", "range", "nbins", "out")
    unknown = filter(flags) do f
        f in ("--normalize", "--no-clip") && return false
        return !any(k -> startswith(f, "--$(k)="), known)
    end
    isempty(unknown) || throw(ArgumentError("unknown flag(s): $(join(unknown, ", "))"))
end

const code  = flagval("code", "Steane713")
step6_path  = length(pos) >= 1 ? pos[1] : "./"
output_path = length(pos) >= 2 ? pos[2] : "./"
nbins       = parse(Int, flagval("nbins", "15"))
normalize   = hasflag("normalize")
clip        = !hasflag("no-clip")
split_name  = flagval("split", "")      # validated against the table once the file is loaded
split_bins  = parse(Int, flagval("split-bins", "0"))   # 0 = one panel per distinct value

# Past this the wheels are too small to read and the colour bar's column squeezes them further.
# Reachable only by accident — splitting on a continuous column without --split-bins — so the
# message points at the flag that fixes it rather than silently writing an unusable figure.
const MAX_PANELS = 8

# `nothing` lets the histogram fit its radial range to the data; an explicit range makes
# figures from different runs directly comparable.
plot_range = let s = flagval("range", "")
    isempty(s) ? nothing : let v = parse.(Float64, split(s, ","))
        length(v) == 2 || throw(ArgumentError("--range needs LO,HI, got $(s)"))
        (v[1], v[2])
    end
end

## -------------------------------------------------------------------------- load the Step-6 file

in_jld2 = isdir(step6_path) ?
    joinpath(step6_path, "step6_improvement_factors_$(code).jld2") : step6_path
isfile(in_jld2) || error("no Step-6 file at $(in_jld2)")

step6 = jldopen(in_jld2, "r") do f
    (step6_table = f["step6_table"], baseline = f["baseline"],
     keep_cutoff = haskey(f, "keep_cutoff") ? f["keep_cutoff"] : false)
end

step6_table = step6.step6_table
nrow(step6_table) == 0 && error("the Step-6 table is empty: nothing to plot")

# The baseline's axis names are the axes, in the order Step 6 reports them.
const AXES = keys(step6.baseline)

for name in AXES
    hasproperty(step6_table, Symbol("IF_", name)) ||
        error("the Step-6 table has no column IF_$(name); it was written for a different grid")
end

# The split is on the column's *value*, not on an improvement factor, so a panel is one tier of
# the frontier — "the solutions whose F_link is 0.99934464", or "…whose cutoff keeps a quarter of
# the attempts" — and the wheel inside it reads as the requirements given that tier.
split_col = if isempty(split_name)
    nothing
else
    sym = Symbol(split_name)
    hasproperty(step6_table, sym) || throw(ArgumentError(
        "--split=$(split_name) is not a column of the Step-6 table; the hardware axes are " *
        join(String.(AXES), ", ") * ", and the protocol/outcome columns carried over from " *
        "Step 5 include " * join(String.(intersect(propertynames(step6_table),
            (:cutoff, :k, :p_kept, :n_cutoffs, :τ, :p_mem, :margin, :member))), ", ")))
    sym
end

# A hardware axis is constant inside its own panel, so it leaves the wheel (see --split in the
# header). Anything else — the cutoff above all — is not one of the nine sectors, so the wheel
# stays whole and each panel shows all nine axes conditioned on that tier.
split_is_axis = split_col !== nothing && split_col in AXES

# Step 6 collapses the cutoff unless it was run with --keep-cutoff, keeping the largest-margin row
# per hardware point. The surviving `cutoff` is then an artefact of that tie-break, not a protocol
# choice the frontier offers, and panels cut on it would show the tie-break rule.
if split_col in (:cutoff, :k, :p_kept) && !step6.keep_cutoff
    @warn "this Step-6 file collapsed the cutoff: there is one row per hardware point and its " *
          "cutoff is whichever one Step 5's tie-break kept, so the panels do not show what the " *
          "cutoff buys. Re-run Step 5 and Step 6 with --keep-cutoff." split_col
end

# Panels hold different numbers of solutions, so a count would say more about a panel's size than
# about its requirements; the share makes the panels comparable, and the panel title carries the
# number it is a share of.
split_col === nothing || (normalize = true)

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

# Symbols for the columns that are not hardware axes but are worth splitting on — the protocol
# knob and the per-solution outcomes Step 5 carried through.
const COLUMN_LATEX = Dict(
    :cutoff     => "c",
    :k          => "k",
    :p_kept     => "p_{\\mathrm{kept}}",
    :n_cutoffs  => "n_{c}",
    :τ          => "\\tau",
    :p_mem      => "p_{\\mathrm{mem}}",
    :eps_hat    => "\\hat{\\varepsilon}",
    :eps_star   => "\\varepsilon^{*}",
    :margin     => "\\varepsilon^{*} - \\hat{\\varepsilon}",
    :margin_rel => "(\\varepsilon^{*} - \\hat{\\varepsilon})/\\varepsilon^{*}",
    :member     => "\\mathrm{member}",
)

const COLORMAP = cgrad([:white, :lavender, :cornflowerblue, :mediumblue, :navy])

"""`v` in LaTeX: a decimal stays as it is, 1.0e-5 becomes `10^{-5}` rather than `1e-05`."""
function latex_number(v)
    s = @sprintf("%.10g", v)
    occursin('e', s) || return s
    mantissa, exponent = split(s, 'e')
    e = parse(Int, exponent)
    return mantissa == "1" ? "10^{$(e)}" : "$(mantissa) \\times 10^{$(e)}"
end

"""Math-mode symbol of `col`, without the delimiters, for building a longer `latexstring`."""
column_symbol(col::Symbol) =
    col in AXES          ? strip(String(AXIS_LATEX[col]), '$') :
    haskey(COLUMN_LATEX, col) ? COLUMN_LATEX[col] :
    "\\mathrm{" * replace(String(col), "_" => "\\_") * "}"

"""
One value of `col` in LaTeX. Durations are stored in seconds and read in milliseconds; a
non-finite cutoff is `∞`, the protocol that never discards.
"""
function column_value_latex(col::Symbol, v)
    v isa Real && !isfinite(v) && return v < 0 ? "-\\infty" : "\\infty"
    v isa Real || return "\\mathrm{" * replace(string(v), "_" => "\\_") * "}"
    col in (:cutoff, :τ) &&
        return latex_number(round(1e3 * v; sigdigits = 3)) * "\\,\\mathrm{ms}"
    col in (:k, :n_cutoffs, :member) && return string(Int(v))
    # Quantile bin edges land on raw data values, which carry more digits than a title can hold.
    return latex_number(col in AXES ? v : round(v; sigdigits = 3))
end

"""
Heading of one panel of a split figure: the column's symbol, the value or range that panel
holds, and the number of solutions in it — the panel's colours are shares of exactly that
number, so without it a dark cell in a 4-solution panel reads like one in a 200-solution panel.
"""
function panel_title(col::Symbol, panel_df)
    values = unique(panel_df[!, col])
    body = if length(values) == 1
        string(column_symbol(col), " = ", column_value_latex(col, only(values)))
    else
        string(column_symbol(col), " \\in [", column_value_latex(col, minimum(values)),
               ",\\, ", column_value_latex(col, maximum(values)), "]")
    end
    return latexstring(body, "\\;(n = ", nrow(panel_df), ")")
end

"""Same heading in plain text, for the run's report on stdout."""
function panel_label(col::Symbol, panel_df)
    values = unique(panel_df[!, col])
    fmt(v) = v isa Real ? @sprintf("%.6g", v) : string(v)
    return length(values) == 1 ?
        string(col, " = ", fmt(only(values))) :
        string(col, " ∈ [", fmt(minimum(values)), ", ", fmt(maximum(values)), "]")
end

"""
    split_groups(df, col; nbins = 0) -> Vector{DataFrame}

The rows of each panel of a `--split` figure, in the order the panels are drawn.

  `nbins = 0`  one panel per distinct value of `col` — right for a hardware axis, whose grid
               holds a handful of levels
  `nbins > 0`  `nbins` equal-count bins over the finite values, for a column like the cutoff
               that takes a different value in nearly every row. Equal-count rather than
               equal-width: the quantile cutoffs c_k = Q̂(1/k) bunch up at the loose end, and
               equal-width bins there would be one crowded panel beside several near-empty ones.
               The edges are quantiles of the data, so every panel names a range the table
               actually contains.

A non-finite value is never binned with the rest: c = ∞ is the protocol that discards nothing,
a different thing from a very loose cutoff rather than the far end of the same range. It gets
the last panel, which is also where it belongs on a loosening scale.
"""
function split_groups(df, col; nbins::Int = 0)
    v = df[!, col]
    numeric = eltype(v) <: Real

    finite_rows  = numeric ? findall(isfinite, v) : collect(1:nrow(df))
    special_rows = numeric ? findall(!isfinite, v) : Int[]

    panels = DataFrame[]
    finite = df[finite_rows, :]

    if !numeric || nbins <= 0 || length(unique(v[finite_rows])) <= nbins
        for g in groupby(finite, col; sort = true)
            push!(panels, DataFrame(g))
        end
    else
        values = Float64.(finite[!, col])
        # Lumpy data can put several quantiles on the same value; those collapse into one edge,
        # and the figure ends up with fewer panels than asked for rather than empty ones.
        edges = unique([quantile(values, i / nbins) for i in 0:nbins])

        if length(edges) < 2
            push!(panels, finite)
        else
            for b in 1:(length(edges) - 1)
                rows = [i for (i, x) in enumerate(values)
                        if clamp(searchsortedlast(edges, x), 1, length(edges) - 1) == b]
                isempty(rows) || push!(panels, finite[rows, :])
            end
        end
    end

    isempty(special_rows) || push!(panels, df[special_rows, :])
    return panels
end

## ------------------------------------------------------------------------------- the histogram

"""
    improvement_bins(df; kwargs...) -> (counts, log_edges, n_solutions, n_clipped)

Bin the improvement factors of every row of `df` (a Step-6 table, read from its `IF_<axis>`
columns — the factors are not recomputed here) into `nbins` radial bins linear in log10(IF),
one histogram per axis. `counts` is `nbins × length(axes_)`.

  `axes_`              which axes to bin, in angular order (default all nine)
  `nbins`              radial bins spanning `improvement_range`
  `improvement_range`  `(lo, hi)` as factors, e.g. `(1e-2, 1e3)`; `nothing` fits the data
  `clip`               fold values outside an explicit range into the edge bins rather than
                       dropping them, so no solution silently disappears from the picture

Binning is separate from drawing so that several panels can share one radial range and one
colour scale — see `--split`, where comparing panels only means something if they do.
"""
function improvement_bins(
    df;
    axes_ = AXES,
    nbins = 15,
    improvement_range = nothing,
    clip = true,
)
    ## ---------------------------------------------------- 1. factors, rows = solutions
    values = [Float64(df[i, Symbol("IF_", name)])
              for i in 1:nrow(df), name in axes_]

    all(values .> 0) || error("improvement factors must be > 0 to go on a log radial axis")

    logvalues = log10.(values)
    nsolutions, nparams = size(values)

    ## ---------------------------------------------------- 2. common logarithmic radial bins
    min_exp, max_exp = if improvement_range === nothing
        floor(minimum(logvalues)), ceil(maximum(logvalues))
    else
        log10(improvement_range[1]), log10(improvement_range[2])
    end
    min_exp < max_exp || error("improvement_range must be increasing")

    log_edges = collect(range(min_exp, max_exp; length = nbins + 1))

    ## ---------------------------------------------------- 3. one histogram per axis
    counts = zeros(Int, nbins, nparams)
    n_clipped = 0

    for j in 1:nparams, x in logvalues[:, j]
        if x < min_exp || x > max_exp
            # Dropping these would make a sector look emptier than the frontier is.
            clip || continue
            n_clipped += 1
            x = clamp(x, min_exp, max_exp)
        end
        b = x == max_exp ? nbins : searchsortedlast(log_edges, x)
        counts[clamp(b, 1, nbins), j] += 1
    end

    return (counts = counts, log_edges = log_edges,
            n_solutions = nsolutions, n_clipped = n_clipped)
end

"""
    draw_polar_histogram(counts, log_edges; n_solutions, kwargs...) -> plot

Draw binned improvement factors as a polar histogram: one angular sector per axis, radius
linear in log10(IF), cell colour = number of solutions in that cell.

  `zmax`       top of the colour scale; `nothing` takes it from `counts`. Pass the same value to
               every panel of a split figure, or their colours are not comparable.
  `colorbar`   draw the scale (once per figure is enough)
  `pad_right`  empty space kept to the right of the wheel, in radial units. The colour bar needs
               it; the panels that do not draw one still have to reserve it, or the aspect ratio
               shrinks the panel that does and the panels end up at different sizes.
  `title`      panel heading, e.g. the value the figure was split on
"""
function draw_polar_histogram(
    counts,
    log_edges;
    n_solutions,
    axes_ = AXES,
    colormap = COLORMAP,
    normalize = false,
    zmax = nothing,
    colorbar = true,
    pad_right = colorbar ? 2.0 : 0.1,
    title = nothing,
    labels = [AXIS_LATEX[name] for name in axes_],
)
    nbins, nparams = size(counts)
    min_exp, max_exp = first(log_edges), last(log_edges)

    zvalues = normalize ? counts ./ n_solutions : Float64.(counts)
    zmax = something(zmax, maximum(zvalues))

    ## ---------------------------------------------------- 4. annular sector as a Shape
    function annular_sector(r_inner, r_outer, θ1, θ2; npoints = 30)
        θ_outer = collect(range(θ1, θ2; length = npoints))
        θ_inner = reverse(θ_outer)
        Shape(
            vcat(r_outer .* cos.(θ_outer), r_inner .* cos.(θ_inner)),
            vcat(r_outer .* sin.(θ_outer), r_inner .* sin.(θ_inner)),
        )
    end

    ## ---------------------------------------------------- 5. radius is linear in log10(IF)
    # The offset leaves a hole in the centre; without it the innermost cells degenerate to
    # slivers and the radial tick labels collide.
    radial_offset = 1.0
    log_to_radius(x) = radial_offset + x - min_exp

    radial_edges = log_to_radius.(log_edges)
    θ_edges = collect(range(0, 2π; length = nparams + 1))
    angular_gap = 0.015                      # hairline between sectors
    rmax = maximum(radial_edges)

    ## ---------------------------------------------------- 6. base plot
    p = plot(
        aspect_ratio = :equal,
        axis = false,
        grid = false,          # the cartesian grid would show through behind the polar one
        legend = false,
        size = (600, 400),
        margin = 0mm,
        rightmargin = 0mm,
        title = title === nothing ? "" : title,
    )

    gradient = cgrad(colormap)

    ## ---------------------------------------------------- 7. the cells
    for j in 1:nparams
        θ1 = θ_edges[j] + angular_gap
        θ2 = θ_edges[j + 1] - angular_gap

        for b in 1:nbins
            z = zvalues[b, j]
            plot!(
                p,
                annular_sector(radial_edges[b], radial_edges[b + 1], θ1, θ2);
                seriestype = :shape,
                fillcolor = gradient[zmax == 0 ? 0.0 : z / zmax],
                linecolor = :white,
                linewidth = 0.6,
                label = false,
            )
        end
    end

    ## ---------------------------------------------------- 8. angular separators
    for θ in θ_edges[1:end-1]
        plot!(
            p,
            [radial_offset * cos(θ), rmax * cos(θ)],
            [radial_offset * sin(θ), rmax * sin(θ)];
            color = :black,
            linewidth = 0.5,
            alpha = 0.5,
            label = false,
        )
    end

    ## ---------------------------------------------------- 9. radial grid at powers of ten
    # After the separators, so the tick labels and their backing patches end up on top of
    # everything else rather than being cut by a separator running through them.
    θcircle = range(0, 2π; length = 300)

    # Put the radial ticks on the sector boundary closest to vertical, so the labels run up a
    # gap between two sectors rather than across one sector's cells.
    θlabel = θ_edges[argmin(abs.(θ_edges[1:end-1] .- π / 2))]

    for exponent in ceil(Int, min_exp):floor(Int, max_exp)
        r = log_to_radius(exponent)

        # IF = 1 is the baseline itself: the ring that separates "relaxed" from "must improve".
        plot!(
            p,
            r .* cos.(θcircle),
            r .* sin.(θcircle);
            linewidth = exponent == 0 ? 1.5 : 0.7,
            linestyle = exponent == 0 ? :solid : :dot,
            color = :black,
            alpha = 0.5,
            label = false,
        )

        x, y = r * cos(θlabel), r * sin(θlabel)

        # A label over a dark cell is unreadable, so clear the patch underneath it first.
        plot!(
            p,
            Shape(x .+ [-0.42, 0.42, 0.42, -0.42], y .+ [-0.22, -0.22, 0.22, 0.22]);
            seriestype = :shape,
            fillcolor = :white,
            linecolor = :white,
            label = false,
        )
        annotate!(p, x, y, text(latexstring("10^{", exponent, "}"), 10, "Computer Modern"))
    end

    ## ---------------------------------------------------- 10. axis labels
    label_radius = rmax + 0.45

    for j in 1:nparams
        θ = (θ_edges[j] + θ_edges[j + 1]) / 2
        x, y = label_radius * cos(θ), label_radius * sin(θ)

        halign = cos(θ) >  0.2 ? :left :
                 cos(θ) < -0.2 ? :right : :center

        annotate!(p, x, y, text(labels[j], 13, "Computer Modern", halign))
    end

    ## ---------------------------------------------------- 11. colour bar, via a dummy series
    # One bar per figure: in a split figure the panels share `zmax`, so repeating it per panel
    # would only repeat the same scale and eat the width the panels need.
    if colorbar
        scatter!(
            p,
            [NaN, NaN],
            [NaN, NaN];
            marker_z = [0.0, zmax],
            c = colormap,
            clims = (0, zmax),
            colorbar = true,
            colorbar_title = normalize ? "Share of solutions" : "Number of solutions",
            label = false,
            position = :right,
        )
    end

    xlims!(p, -label_radius - 0.1, label_radius + pad_right)
    ylims!(p, -label_radius - 0.5, label_radius + 0.5)

    return p
end

"""
    colorbar_panel(zmax; kwargs...) -> plot

A subplot that holds nothing but the shared colour scale. A bar drawn on one of the wheels
would eat that wheel's drawing area, leaving it visibly smaller than its neighbours — panels of
different sizes read as panels of different data, which is exactly what a split figure must not
suggest. Given its own column, every wheel keeps the same size.
"""
function colorbar_panel(zmax; colormap = COLORMAP, title = "Number of solutions")
    # The only series below is two NaNs, which leaves the panel with no data window and makes GR
    # reject its viewport; the explicit limits give it one. Nothing is drawn inside them.
    p = plot(framestyle = :none, legend = false, grid = false, margin = 0mm,
             xlims = (0, 1), ylims = (0, 1))
    scatter!(
        p,
        [NaN, NaN],
        [NaN, NaN];
        marker_z = [0.0, zmax],
        c = colormap,
        clims = (0, zmax),
        colorbar = true,
        colorbar_title = title,
        label = false,
    )
    return p
end

"""
    polar_improvement_histogram(df; kwargs...) -> (plot, counts, log_edges, …)

Bin and draw in one step: the single-panel figure. Takes the keywords of both
`improvement_bins` and `draw_polar_histogram`.
"""
function polar_improvement_histogram(
    df;
    axes_ = AXES,
    nbins = 15,
    improvement_range = nothing,
    clip = true,
    kwargs...,
)
    b = improvement_bins(df; axes_ = axes_, nbins = nbins,
                         improvement_range = improvement_range, clip = clip)
    p = draw_polar_histogram(b.counts, b.log_edges;
                             n_solutions = b.n_solutions, axes_ = axes_, kwargs...)
    return (plot = p, counts = b.counts, log_edges = b.log_edges,
            n_solutions = b.n_solutions, n_clipped = b.n_clipped)
end

## ------------------------------------------------------------------------------------ draw

# Keep the figure in the paper's font without permanently restyling a caller's session.
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

panels = if split_col === nothing
    [(title = nothing, label = "all", df = step6_table)]
else
    groups = split_groups(step6_table, split_col; nbins = split_bins)

    length(groups) <= MAX_PANELS || throw(ArgumentError(
        "--split=$(split_col) gives $(length(groups)) panels, more than the $(MAX_PANELS) that " *
        "still fit side by side" *
        (split_bins > 0 ? "; lower --split-bins" :
         "; the column is not a short ladder of levels, so group it with --split-bins=N " *
         "(3 or 4 reads well)")))

    [(title = panel_title(split_col, g), label = panel_label(split_col, g), df = g)
     for g in groups]
end

# A hardware axis is constant inside its own panel: its sector would be one cell holding the
# whole panel, which says nothing and would set `zmax` by itself. Drop it, and read each panel as
# the other axes conditioned on that value. A column that is not an axis — the cutoff — owns no
# sector, so the wheel keeps all nine.
panel_axes = split_is_axis ? Tuple(n for n in AXES if n !== split_col) : AXES

# Panels can only be compared if they are drawn on the same radial range: fitting each one to
# its own data would put the same improvement factor at a different radius in each panel.
shared_range = plot_range
if shared_range === nothing && length(panels) > 1
    all_log = log10.([Float64(step6_table[i, Symbol("IF_", n)])
                      for i in 1:nrow(step6_table), n in panel_axes])
    shared_range = (10.0^floor(minimum(all_log)), 10.0^ceil(maximum(all_log)))
end

binned = [improvement_bins(pan.df; axes_ = panel_axes, nbins = nbins,
                           improvement_range = shared_range, clip = clip)
          for pan in panels]

# …and on the same colour scale, for the same reason.
zmax = maximum(maximum(normalize ? b.counts ./ b.n_solutions : Float64.(b.counts))
               for b in binned)

multi = length(panels) > 1

drawn = [draw_polar_histogram(
             b.counts, b.log_edges;
             n_solutions = b.n_solutions,
             axes_ = panel_axes,
             normalize = normalize,
             zmax = zmax,                       # one scale for the whole figure
             colorbar = !multi,                 # which a column of its own carries
             title = pan.title,
         )
         for (b, pan) in zip(binned, panels)]

fig = if !multi
    drawn[1]
else
    n_panels = length(drawn)
    # GR draws the bar at a width proportional to the *whole figure*, so its column needs a
    # minimum share of the total rather than a minimum number of pixels: squeeze the share and
    # the viewport comes out negative ("Rectangle definition is invalid"). Measured threshold is
    # near 0.12; 0.16 holds for every panel count, and the figure grows to keep the wheels the
    # same size as the figure gains panels.
    panel_px = 460
    bar_width = 0.16
    total_px = round(Int, panel_px * n_panels / (1 - bar_width))

    plot(drawn...,
         colorbar_panel(zmax;
                        title = normalize ? "Share of solutions" : "Number of solutions");
         layout = grid(1, n_panels + 1;
                       widths = [fill((1 - bar_width) / n_panels, n_panels)..., bar_width]),
         size = (total_px, 460))
end

suffix = split_col === nothing ? "" : "_by_$(split_col)"
out_pdf = let s = flagval("out", "")
    isempty(s) ? joinpath(output_path, "step6_improvement_histogram_$(code)$(suffix).pdf") : s
end
savefig(fig, out_pdf)
default(; prev_defaults...)

edges = binned[1].log_edges
n_clipped = sum(b.n_clipped for b in binned)

println("Step 6 figure: polar improvement histogram over ", nrow(step6_table), " solution(s), ",
        length(edges) - 1, " radial bins over 10^",
        @sprintf("%g", first(edges)), "…10^", @sprintf("%g", last(edges)))
if split_col !== nothing
    println("        split on ", split_col,
            split_bins > 0 ? " into $(length(panels)) equal-count bin(s)" :
                             " into $(length(panels)) panel(s)",
            " of ", length(panel_axes), " axes",
            split_is_axis ? " ($(split_col) is constant in a panel and left out)" :
                            " ($(split_col) is not a hardware axis, so the wheel keeps all of them)",
            ", common radial range and colour scale (zmax = ", @sprintf("%g", zmax), "):")
    for (b, pan) in zip(binned, panels)
        println("          ", pan.label, "  —  ", b.n_solutions, " solution(s)")
    end
end
n_clipped > 0 && println("        ", n_clipped, " of ",
    nrow(step6_table) * length(panel_axes), " axis values fell outside that range and were ",
    "folded into the edge bins")
println("        read\n  ", in_jld2, "\n        wrote\n  ", out_pdf)
