# Figure for Step 6: which hardware axis is the binding requirement, against the cutoff tier.
#
# The polar histogram (plot_improvements.jl) shows how far every axis has to move; the
# substitution matrix (plot_substitution.jl) shows which axes trade against each other. Neither
# answers the one-line question: *which* axis actually gates a solution, and does that change
# with the cutoff?
#
# Step 6 already answers it per solution. `IF_bottleneck` names the axis with the largest
# improvement factor — the single hardest requirement, the one that has to be met however
# generous the rest of the machine is ("none" when the baseline already reaches the target, so
# nothing has to improve at all). This figure is the count of that column: one bar per cutoff
# tier k, split into coloured segments by which axis is the bottleneck. Reading it:
#
#   a segment's height    how many of the frontier's solutions that axis gates at that k
#   a bar's total height   how many solutions exist at that k, which is not constant — a tight
#                          cutoff admits fewer hardware points. Use --normalize to read each bar
#                          as shares instead, which is the right view for comparing tiers.
#   a colour that grows    an axis the cutoff makes *more* binding, i.e. one the protocol cannot
#   with k                 buy you out of; a colour that shrinks is one the cutoff pays for
#
# --split=COLUMN colours the segments by another column's value instead. The useful case is a
# hardware axis: --split=F_link gives, per tier, how many of the frontier's solutions ask for
# each F_link on the grid, so the bar shows the distribution of the required link fidelity and
# how the cutoff shifts it. A split on a value is an ordered quantity, not nine unrelated
# identities, so those segments take one blue ramp, light (lowest value) to dark (highest),
# rather than the categorical hues.
#
# k is the cutoff tier of c_k = Q̂_{T_GHZ}(1/k): k = 1 keeps every GHZ attempt and larger k
# discards more, so the bars run from the loosest protocol on the left to the tightest on the
# right. k = 0 marks c = ∞, no cutoff at all, and is labelled ∞ rather than 0.
#
# Only the tiers the table actually contains get a bar, and they are drawn equally spaced: the
# feasible k are sparse and uneven, so true numeric spacing would be mostly empty axis. The x
# axis is therefore an order, not a scale — worth remembering on a --by column that is really
# continuous (p_kept, cutoff), where the bars are near-unique values and only an evenly spaced
# subset of them is labelled, see --max-xticks.
#
# Usage:
#   julia --project=src src/plot_bottleneck.jl <step6_path> [output_path] [flags]
#
#   step6_path   step6_improvement_factors_<code>.jld2, or the folder holding it (Step 6)
#   output_path  where step6_bottleneck_by_k_<code>.pdf is written; defaults to ./
#
#   --code=NAME     code tag in the file names (default Steane713)
#   --by=COLUMN     bar on this column instead of k; any integer-valued or few-valued column of
#                   the Step-6 table (`member`, `n_cutoffs`, a hardware axis, …)
#   --split=COLUMN  colour the segments by this column instead of IF_bottleneck; any few-valued
#                   column, typically a hardware axis (`F_link`, `T_coherence`, …). The figure
#                   is then named after it: step6_F_link_by_k_<code>.pdf
#   --normalize     each bar to its own height, so the segments are shares of that tier rather
#                   than counts. The tiers hold very different numbers of solutions, so this is
#                   the view to compare them on; the counts are the view of where the frontier is.
#   --no-none       drop the solutions that need no improvement at all instead of giving them a
#                   segment of their own (on --split=COLUMN it still drops those rows, they just
#                   have no segment of their own to begin with)
#   --csv           also write the count table next to the figure
#   --max-xticks=N  label N of the bars instead of as many as the axis is judged to hold. A
#                   column with a value per solution (p_kept, cutoff) has far more bars than
#                   labels that fit, so only an evenly spaced subset is labelled; this is the
#                   override when the estimate is wrong for your figure size.
#   --out=FILE      write here instead of <output_path>/step6_bottleneck_by_k_<code>.pdf
#
# Needs a Step-6 file built with --keep-cutoff to say anything about the cutoff: without it
# Step 6 keeps one row per hardware point and its k is whichever one Step 5's tie-break picked,
# so the bars would show that rule rather than the physics. The script warns rather than
# refuses, since --by on a hardware axis is unaffected.

using JLD2
using DataFrames
using Printf
using LaTeXStrings
using Measures
using Plots
using CSV
gr()

flags = filter(a -> startswith(a, "--"), ARGS)
pos   = filter(a -> !startswith(a, "--"), ARGS)

hasflag(name) = ("--$(name)" in flags)
function flagval(name, default)
    i = findfirst(f -> startswith(f, "--$(name)="), flags)
    return i === nothing ? default : String(split(flags[i], "="; limit = 2)[2])
end

let known = ("code", "by", "split", "max-xticks", "out")
    unknown = filter(flags) do f
        f in ("--normalize", "--no-none", "--csv") && return false
        return !any(k -> startswith(f, "--$(k)="), known)
    end
    isempty(unknown) || throw(ArgumentError("unknown flag(s): $(join(unknown, ", "))"))
end

const code  = flagval("code", "Steane713")
step6_path  = length(pos) >= 1 ? pos[1] : "./"
output_path = length(pos) >= 2 ? pos[2] : "./"
normalize   = hasflag("normalize")
drop_none   = hasflag("no-none")
write_csv   = hasflag("csv")
by_name     = flagval("by", "k")
split_name  = flagval("split", "IF_bottleneck")

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

# The baseline's axis names are the axes, in the order Step 6 reports them — which is also the
# order the segments stack in, so the figure and the Step-6 report read the same way round.
const AXES = keys(step6.baseline)

"""The column `--by`/`--split` names, checked against the table so a typo is not an empty plot."""
function named_column(flag, name)
    col = Symbol(name)
    hasproperty(step6_table, col) || throw(ArgumentError(
        "--$(flag)=$(name) is not a column of the Step-6 table; the hardware axes are " *
        join(String.(AXES), ", ") * ", and the protocol/outcome columns carried over from Step 5 " *
        "include " * join(String.(intersect(propertynames(step6_table),
            (:cutoff, :k, :p_kept, :n_cutoffs, :member))), ", ")))
    return col
end

# The default split and --no-none both read IF_bottleneck; a split on a hardware axis does not,
# so an older Step-6 file is still worth plotting that way.
if (split_name == "IF_bottleneck" || drop_none) && !hasproperty(step6_table, :IF_bottleneck)
    error("the Step-6 table has no IF_bottleneck column; it was written by an older " *
          "run_improvements.jl. Give --split=COLUMN (a hardware axis) to plot it anyway.")
end

by_col    = named_column("by", by_name)
split_col = named_column("split", split_name)

by_col === split_col && throw(ArgumentError(
    "--by and --split are both $(by_name): every bar would be one segment"))

if by_col in (:k, :cutoff, :p_kept) && !step6.keep_cutoff
    @warn "this Step-6 file collapsed the cutoff: there is one row per hardware point and its " *
          "cutoff is whichever one Step 5's tie-break kept, so the bars do not show what the " *
          "cutoff buys. Re-run Step 5 and Step 6 with --keep-cutoff." by_col
end

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

const COLUMN_LATEX = Dict(
    :k         => L"k",
    :cutoff    => L"c",
    :p_kept    => L"p_{\mathrm{kept}}",
    :n_cutoffs => L"n_{c}",
    :member    => L"\mathrm{member}",
)

column_label(col::Symbol) =
    col in AXES ? AXIS_LATEX[col] :
    get(COLUMN_LATEX, col, latexstring("\\mathrm{", replace(String(col), "_" => "\\_"), "}"))

## ---------------------------------------------------------------------------------- colours
#
# Categorical hues in a fixed order, from the validated eight-slot palette; the order itself is
# the colourblind-safety mechanism, so slots are taken from the front and never reshuffled. The
# categories present take slots 1…m contiguously, which keeps every stack that can be drawn on
# the palette's validated *adjacent* pairlist — a bottleneck axis therefore has no colour of its
# own across files with different axis sets, and the legend, not the hue, is what names it.
#
# Validated (OKLab ΔE×100, light surface #fcfcfb, adjacent pairs, Machado et al. severity 1.0):
# every drawable stack [none, slots 1…m, other], m = 1…8, clears worst-pair CVD ΔE ≥ 9.1
# (target 8) and worst-pair normal-vision ΔE ≥ 17.2 (floor 15). The two neutrals sit below the
# chroma floor on purpose: "no requirement" and "other" are not identities and must not read as
# one of the nine axes.
const SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const NONE_COLOR  = "#898781"      # baseline already suffices — nothing to improve
const OTHER_COLOR = "#52514e"      # the fold-in, only if more than 8 axes are ever the bottleneck
const SURFACE     = "#fcfcfb"      # the gap between segments is the surface showing through

# A split on a value (--split=F_link) is one ordered quantity, so it takes one hue light→dark
# and not nine identities: the ramp itself says which segment is the larger value, which the
# categorical hues cannot. These are the blue ramp's steps 250…700, the band whose every step
# clears 2:1 against the light surface — the ordinal rule, so the lightest segment is still a
# segment and not a hole in the bar. `ramp_color` reads it at 0 (lightest) … 1 (darkest),
# interpolating between steps so any number of values spreads over the whole band.
const ORDINAL_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                      "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

function ramp_color(t::Real)
    hex2rgb(h) = (parse(Int, h[2:3], base = 16), parse(Int, h[4:5], base = 16),
                  parse(Int, h[6:7], base = 16))
    u = clamp(t, 0.0, 1.0) * (length(ORDINAL_RAMP) - 1)
    lo = floor(Int, u)
    hi = min(lo + 1, length(ORDINAL_RAMP) - 1)
    a, b, w = hex2rgb(ORDINAL_RAMP[lo + 1]), hex2rgb(ORDINAL_RAMP[hi + 1]), u - lo
    return "#" * join(string(round(Int, a[d] + w * (b[d] - a[d])), base = 16, pad = 2)
                      for d in 1:3)
end

## ------------------------------------------------------------------- how a value is written

"""
`v` as a typeset label for the x-axis or the legend: the tier `k = 0` is c = ∞, and a float
that is whole loses its `.0`.
"""
function value_label(col::Symbol, v)
    v isa Real && !isfinite(v) && return L"\infty"
    col === :k && v == 0 && return L"\infty"
    v isa Integer && return latexstring(string(v))
    v isa Real && isinteger(v) && return latexstring(string(Int(v)))
    v isa Real && return latexstring(@sprintf("%.4g", v))
    return latexstring("\\mathrm{", replace(string(v), "_" => "\\_"), "}")
end

"""`v` as plain text, for the printed table, the CSV header and the segment names."""
plain_value(col::Symbol, v) =
    col === :k && v == 0                           ? "Inf" :
    v isa Real && !isfinite(v)                     ? (v < 0 ? "-Inf" : "Inf") :
    v isa Real && !isa(v, Integer) && !isinteger(v) ? @sprintf("%.4g", v) : string(v)

## ------------------------------------------------------------------------------ the counts

table = drop_none ? step6_table[step6_table.IF_bottleneck .!= "none", :] : step6_table
nrow(table) == 0 && error("no rows left to plot" * (drop_none ? " after --no-none" : ""))

# Bars left to right: ascending tier. k = 0 is c = ∞, the protocol that discards nothing, and
# sorting it first puts it at the loose end where it belongs.
bar_values = sort(unique(table[!, by_col]))

# The segments, top of the stack first — which is also the order the legend lists them in, see
# the drawing loop. Each one carries its plain name (for the report and the CSV), its legend
# symbol, its colour, and the test for the split values it holds; "other" is the only segment
# that ever holds more than one.
cat_names  = String[]
labels     = LaTeXString[]
colors     = String[]
cat_holds  = Function[]

if split_col === :IF_bottleneck
    # An identity, not a quantity: "none" at the top, then the axes in Step 6's own order, so
    # the figure and the Step-6 report read the same way round. Only the categories that
    # actually occur get a segment — an axis that never gates a solution would otherwise take a
    # slot and a legend entry to say nothing.
    present = Set(table.IF_bottleneck)
    axis_cats = [String(n) for n in AXES if String(n) in present]

    # Nine axes against eight validated slots: if every one of them is ever the bottleneck, the
    # least frequent folds into "other" rather than getting an unvalidated ninth hue.
    folded = String[]
    if length(axis_cats) > length(SERIES_COLORS)
        counts_of(c) = count(==(c), table.IF_bottleneck)
        keep = Set(sort(axis_cats; by = counts_of, rev = true)[1:length(SERIES_COLORS)])
        folded = [c for c in axis_cats if !(c in keep)]
        axis_cats = [c for c in axis_cats if c in keep]
        @warn "more bottleneck axes than validated colour slots; the least frequent are " *
              "folded into one \"other\" segment" n_axes = length(axis_cats) + length(folded) folded
    end

    show_none = ("none" in present) && !drop_none
    cat_names = vcat(show_none ? ["none"] : String[],
                     axis_cats,
                     isempty(folded) ? String[] : ["other"])

    cat_label(nm) = nm == "none"  ? L"\mathrm{none}"  :
                    nm == "other" ? L"\mathrm{other}" : AXIS_LATEX[Symbol(nm)]
    cat_color(nm, i) = nm == "none"  ? NONE_COLOR :
                       nm == "other" ? OTHER_COLOR :
                       SERIES_COLORS[i - (show_none ? 1 : 0)]

    labels    = LaTeXString[cat_label(nm) for nm in cat_names]
    colors    = String[cat_color(nm, i) for (i, nm) in enumerate(cat_names)]
    cat_holds = Function[nm == "other" ? (b -> b in folded) : (b -> b == nm) for nm in cat_names]
else
    # A quantity: one segment per distinct value, largest at the top of the stack and darkest on
    # the ramp, so the stack reads like a vertical colour bar and the legend reads down it. Every
    # value that occurs gets a segment, however few solutions hold it — the point of this split
    # is the shape of the distribution, and dropping its tail would misstate it.
    split_values = sort(unique(table[!, split_col]); rev = true)
    m = length(split_values)
    m <= length(ORDINAL_RAMP) || @warn "more distinct values than the ordinal ramp has steps; " *
        "neighbouring segments will be close in colour" column = split_col n_values = m

    cat_names = String[plain_value(split_col, v) for v in split_values]
    labels    = LaTeXString[value_label(split_col, v) for v in split_values]
    colors    = String[ramp_color(m == 1 ? 1.0 : (m - i) / (m - 1)) for i in 1:m]
    cat_holds = Function[(x -> isequal(x, v)) for v in split_values]
end

counts = [count(r -> r[by_col] == v && cat_holds[j](r[split_col]), eachrow(table))
          for v in bar_values, j in 1:length(cat_names)]
totals = vec(sum(counts; dims = 2))

# Share of the bar rather than of the table: the question --normalize answers is "given this
# cutoff, what gates the solutions", and that is a share of the tier.
heights = normalize ? counts ./ totals : Float64.(counts)

count_table = DataFrame(by_col => bar_values, :n_solutions => totals)
# `n_0.98`, `n_F_link`, … — the segment under the name the report prints it by.
for (j, nm) in enumerate(cat_names)
    count_table[!, Symbol("n_", nm)] = counts[:, j]
end

## ------------------------------------------------------------------------------------ draw

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

x = 1:length(bar_values)

# A column that takes a value per solution rather than a handful of tiers — p_kept, cutoff, a
# fine grid axis — gives tens of bars, and one label under each of them is an unreadable smear.
# Label an evenly spaced subset instead: first and last always, the rest spread between them,
# the unlabelled bars keeping no tick of their own. The bars are equally spaced by construction,
# so a reader interpolates between two labels; what is lost is only the value of a bar that has
# no label, and the --csv table is the place to read those off anyway.
#
# The budget is how many labels fit: the plot area is roughly 450 px wide once the legend and
# the y-axis have taken theirs, and a character of Computer Modern at `fs` is about 0.55·fs px.
tick_text = [value_label(by_col, v) for v in bar_values]
tick_budget = let s = flagval("max-xticks", "")
    n = if isempty(s)
        wide = maximum(length(plain_value(by_col, v)) for v in bar_values)
        floor(Int, 450 / (0.55 * fs * wide + 10))
    else
        v = tryparse(Int, s)
        v === nothing && throw(ArgumentError("--max-xticks=$(s) is not an integer"))
        v
    end
    clamp(n, 2, length(bar_values))
end
# `range` with a length always lands on both ends; `unique` drops the repeats a short budget
# makes, so the labels stay evenly spaced rather than bunching up.
tick_at = unique(round.(Int, range(1, length(bar_values), length = tick_budget)))

# Stacked by drawing each segment's *cumulative* height back to front, so the series are added
# top segment first. groupedbar would put series 1 at the bottom while the legend lists it
# first, i.e. legend top ↔ stack bottom; this way the legend reads down the stack. Each bar
# carries a surface-coloured outline, which is what leaves the gap between the segments.
nseg = length(cat_names)
cumulative = [sum(heights[i, j:nseg]) for i in 1:length(bar_values), j in 1:nseg]

p = plot(
    size = (640, 420),
    legend = :outertopright,
    legendtitle = split_col === :IF_bottleneck ? L"\mathrm{bottleneck}" : column_label(split_col),
    # A hairline y-grid only, so a segment's height can be read off; GR draws it before the
    # series, so the bars cover it and it never cuts across a segment.
    grid = :y,
    gridstyle = :solid,
    gridlinewidth = 0.6,
    foreground_color_grid = RGB(0.882, 0.878, 0.851),      # #e1e0d9
    gridalpha = 1.0,
    framestyle = :axes,
    foreground_color_axis = RGB(0.765, 0.761, 0.718),      # #c3c2b7, recessive baseline
    foreground_color_border = RGB(0.765, 0.761, 0.718),
    xlabel = column_label(by_col),
    ylabel = normalize ? L"\mathrm{share\ of\ solutions}" : L"\mathrm{solutions}",
    xticks = (x[tick_at], tick_text[tick_at]),
    xlims = (0.4, length(bar_values) + 0.6),
    ylims = (0, normalize ? 1.0 : 1.04 * maximum(totals)),
    bottommargin = 3mm,
    leftmargin = 3mm,
)

for j in 1:nseg
    bar!(p, x, cumulative[:, j];
         bar_width = 0.72,
         fillcolor = colors[j],
         linecolor = SURFACE,            # the 2 px surface gap between segments
         linewidth = 1.2,
         label = labels[j])
end

base = "step6_$(split_col === :IF_bottleneck ? "bottleneck" : split_col)_by_$(by_col)_$(code)"
out_pdf = let s = flagval("out", "")
    isempty(s) ? joinpath(output_path, base * ".pdf") : s
end
savefig(p, out_pdf)
default(; prev_defaults...)

## ---------------------------------------------------------------------------------- report

println()
println("Step 6 figure: ",
        split_col === :IF_bottleneck ? "binding requirement" : String(split_col),
        " over ", by_col, " for ", nrow(table), " solution(s)",
        drop_none ? " (--no-none: the ones needing no improvement are dropped)" : "")

@printf("  %-10s %8s  %s\n", String(by_col), "n",
        (split_col === :IF_bottleneck ? "bottleneck" : String(split_col)) * " (count)")
for (i, v) in enumerate(bar_values)
    parts = [string(nm, "=", counts[i, j]) for (j, nm) in enumerate(cat_names)
             if counts[i, j] > 0]
    @printf("  %-10s %8d  %s\n", plain_value(by_col, v), totals[i], join(parts, ", "))
end

if write_csv
    # --out can name any extension GR writes; swapping ".pdf" blindly would make the CSV land on
    # top of a figure written as .png.
    out_csv = (isempty(splitext(out_pdf)[2]) ? out_pdf : splitext(out_pdf)[1]) * ".csv"
    CSV.write(out_csv, count_table)
    println("        read\n  ", in_jld2, "\n        wrote\n  ", out_pdf, "\n  ", out_csv)
else
    println("        read\n  ", in_jld2, "\n        wrote\n  ", out_pdf)
end
