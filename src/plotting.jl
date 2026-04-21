using JLD2
using DataFrames, StatsPlots, Statistics
using Plots
using Plots.PlotMeasures: mm
using LaTeXStrings

## load jld2 files from output directory
path = "./output_10++/"
files = readdir(path)
dataframes = DataFrame[]
for file in files
    if endswith(file, "jld2")
        filename = path * file
        @load filename logs
        push!(dataframes, logs)
    end
end
df = vcat(dataframes...)

show(describe(df), allrows=true, allcols=true)
##
df = df[df[!, :discarded] .== false, :]
df.infidel_log = 1 .- df.GHZfidel

# mean/std for each (T_coherence, purify)
# summ = combine(groupby(df, [:F_link, :purify, :T_coherence]),
#     :infidel_log => mean => :μ,
#     :infidel_log => std  => :σ,
#     nrow => :nlogs
# )
# summ[!, :se] = summ.σ ./ sqrt.(summ.nlogs)

# mean/std for each (T_coherence, purify)
summ = combine(groupby(df, [:F_link, :purify, :T_coherence]),
    :GHZfidel => mean => :μ,
    :GHZfidel => std  => :σ,
    nrow => :nlogs
)
summ[!, :se] = summ.σ ./ sqrt.(summ.nlogs)

summ = summ[(summ.F_link .> 0.5) .& (summ.F_link .< 1.0), :]
# make subplot per T_coherence, with purified vs unpurified as different series
Ts = sort(unique(summ.T_coherence))
p = plot(
    layout = (1, length(Ts)),
    link = :y,
    size = (350 * length(Ts), 400),
    margin = 10mm
)

for (i, T) in enumerate(Ts)
    summ_T = summ[summ.T_coherence .== T, :]
    @info summ_T
    @df summ_T scatter!(
        p[i],
        :F_link, :nlogs,
        group = :purify,
        # yerror = :se,
        # yscale = :log10,
        xlabel = "Bell pair fidelity "* L"$F_{link}$",
        ylabel = i == 1 ? "State count" : "",
        title = L"$T_{depol} =$" * "$(T) s",
        legendtitle = "Purify",
        legend = i == length(Ts) ? :outerbottomright : false, 
        # ylims = (1e-2, 1e-0),
        xlims = (0.8, 1.0),
        minorgrid = true
    )
end

display(p)
savefig(p, "GHZ_purification_nlogs.pdf")

## plot the number of discarded states per second for each (T_coherence, purify)
df_range = df[(df.F_link .> 0.5) .& (df.F_link .<= 0.9), :]
grouped_coherence = combine(groupby(df_range, [:F_link, :purify, :T_coherence]),
    :discarded => mean => :mean_discarded,
    :discarded => std => :std_discarded,
    nrow => :nlogs
)
grouped_coherence_plot = grouped_coherence[(grouped_coherence.purify .== true), :]
@df grouped_coherence_plot scatter(
    :F_link, :mean_discarded,
    yerror = :std_discarded ./ sqrt.(grouped_coherence_plot.nlogs),
    group = :T_coherence,
    xlabel = "Link Fidelity",
    title = "Average share of abborted states\n due to purification failure",
    titlefontsize = 10,
    ylabel = "Average Share of Abborted States",
    legendtitle = "Coherence Time (s)",
    legend = :outertopright
)   


##
df_range = df[(df.F_link .> 0.5) .& (df.F_link .<= 0.9), :]

grouped_coherence = combine(groupby(df_range, [:F_link, :purify, :T_coherence]),
:discarded => mean => :mean_discarded,
:discarded => std => :std_discarded,
nrow => :nlogs
)
grouped_coherence_plot = grouped_coherence[(grouped_coherence.purify .== true), :]
@df grouped_coherence_plot scatter(
    :F_link, :mean_discarded,
    yerror = :std_discarded ./ sqrt.(grouped_coherence_plot.nlogs),
    group = :T_coherence,
    xlabel = "Link Fidelity",
    title = "Average share of abborted states\n due to purification failure",
    titlefontsize = 10,
    ylabel = "Average Share of Abborted States",
    legendtitle = "Coherence Time (s)",
    legend = :outertopright
)

grouped_discarded = combine(groupby(df_range, [:discarded, :purify, :T_coherence, :F_link,]),
nrow => :nlogs
)
grouped_discarded_nondiscarded = grouped_discarded[grouped_discarded.discarded .== false, :]
@df grouped_discarded_nondiscarded scatter(
    :F_link, :nlogs,
    group = (:purify, :T_coherence),
    markershape = [:circle :circle :diamond :diamond],
    seriescolor = [:blue :orange :blue :orange],
    xlabel = "Link Fidelity",
    title = "Number of protocol completions per second\n (only non-discarded states)",
    titlefontsize = 10,
    ylabel = "Number of protocol completions (Hz)",
    legendtitle = "Coherence Time (s)",
    legend = :outertopright
)

## plotting time evulation of protocol
filename1 = "/Users/localadmin/Documents/github/piecemakerDQEC/GHZservice_purification_trial_logs_true_0.0001_100.0_0.9_runtime_10.0.jld2"
filename2 = "/Users/localadmin/Documents/github/piecemakerDQEC/GHZservice_purification_trial_logs_false_0.0001_100.0_0.9_runtime_10.0.jld2"
logs_purify_true = DataFrame()
logs_purify_false = DataFrame()
@load filename1 logs
logs_purify_true = copy(logs)
@load filename2 logs
logs_purify_false = copy(logs)
alllogs = vcat(logs_purify_true, logs_purify_false)
alllogs_non_discarded = alllogs[alllogs.discarded .== false, :]
@df alllogs_non_discarded plot(
    :timesteps, :GHZfidel,
    group = :purify,
    xlabel = "Time (s)",
    ylabel = "GHZ Fidelity",
    title = "GHZ Fidelity over time for different numbers of clients serviced\n (only non-discarded states)",
    titlefontsize = 10,
    legend_title = "Purified",
    legend = :outertopright
)