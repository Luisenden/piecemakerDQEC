using JLD2
using DataFrames, StatsPlots, Statistics
using Plots
using Plots.PlotMeasures: mm
using LaTeXStrings

##
@load "ghz_service_v1_dephase_data.jld2" df
df_dephase = df[df.num_clients .== 4, :]

##
@load "ghz_service_v1_depol_data.jld2" df
df_depol = df[df.num_clients .== 4, :]

##
df = vcat(df_dephase, df_depol)

groupby_columns = [:error_model, :attempt_time, :link_success_prob, :T_coherence, :F_link, :runtime]
grouped_df = combine(groupby(df_full_ghz, groupby_columns),
    :GHZfidel => mean => :mean_GHZfidel,
    :GHZfidel => std => :std_GHZfidel,
    nrow => :nlogs
)
grouped_df[!, :rate] = grouped_df.nlogs ./ grouped_df.runtime

grouped_df_above_fidelity_target = grouped_df[grouped_df.mean_GHZfidel .>= 0.99, :]

##
@df grouped_df_above_fidelity_target boxplot([:attempt_time])

## load jld2 files from output directory
@load "GHZservice_purification_compare_logs_false_1.0e-6_0.0001_Inf_runtime5.0.jld2" logsv21
@load "GHZservice_purification_compare_logs_true_1.0e-6_0.0001_Inf_runtime5.0.jld2" logsv21true
##
path = "./output_05152026++/"
files = readdir(path)
dataframes = DataFrame[]
for file in files
    if startswith(file, "GHZservice_purification_singleslot")
        filename = path * file
        @load filename logs
        logs[!, :purify] .= -1
        #push!(dataframes, logs)
    elseif startswith(file, "GHZservice_purification_twoslot")
        filename = path * file
        @load filename logs
        push!(dataframes, logs)
    end
end
dfv22 = vcat(dataframes...)


##
path = "./output_10++/"
files = readdir(path)
dataframes = DataFrame[]
for file in files
    if startswith(file, "GHZservice_purification_trial_logs_true")
        filename = path * file
        @load filename logs
        push!(dataframes, logs)
    end
end
dfv21true = vcat(dataframes...)
##
summ1false = combine(groupby(logsv21, [:F_link, :purify, :T_coherence]),
    :GHZfidel => mean => :μ,
    :GHZfidel => std  => :σ,
    nrow => :nlogs
)
summ1false[!, :se] = summ1false.σ ./ sqrt.(summ1false.nlogs)

summ1true = combine(groupby(dfv21true, [:F_link, :purify, :T_coherence]),
    :GHZfidel => mean => :μ,
    :GHZfidel => std  => :σ,
    nrow => :nlogs
)
summ1true[!, :se] = summ1true.σ ./ sqrt.(summ1true.nlogs)

summ2 = combine(groupby(dfv22, [:F_link, :purify, :T_coherence]),
    :GHZfidel => mean => :μ,
    :GHZfidel => std  => :σ,
    nrow => :nlogs
)
summ2[!, :se] = summ2.σ ./ sqrt.(summ2.nlogs)

df_single_slot = vcat(summ1false, summ1true, summ2)
##

Ts = [0.1]
p = plot(
    layout = (1, length(Ts)),
    link = :y,
    size = (400 * length(Ts), 500),
    margin = 10mm
)
marker_shapes = [:circle, :utriangle, :diamond]
line_styles   = [:solid, :dash, :dot]
for (i, T) in enumerate(Ts)
    for (j, df) in enumerate([summ1false, summ1true, summ2])
        summ_T = df[df.T_coherence .== T, :]
        @df summ_T plot!(
            p[i],
            :F_link, :nlogs,#:μ,#
            #yerror = :se,
            group = :purify,
            linestyle = line_styles[j],
            linewidth = 2,
            markershape = marker_shapes[j],
            markersize = 5,
            xlabel = "Bell pair fidelity "* L"$F_{link}$",
            ylabel = i == 1 ? "GHZ Fidelity" : "",
            title = L"$T_{coherence} =$" * "$(T) s",
            titlefontsize = 10,
            legendtitle = "Purify",
            minorgrid = true,
            # ylims = (0.7, 1.0),
            # xlims = (0.95, 1.0)
        )
    end
end

display(p)
##

#summ = summ[(summ.F_link .> 0.5) .& (summ.F_link .< 1.0), :]
# make subplot per T_coherence, with purified vs unpurified as different series
Ts = sort(unique(summ.T_coherence))
p = plot(
    layout = (2, length(Ts)),
    link = :y,
    size = (350 * length(Ts), 800),
    margin = 10mm
)
for (j, error_model) in enumerate(unique(summ.error_model))
    summ_error_model = summ[summ.error_model .== error_model, :]
    for (i, T) in enumerate(Ts)
        summ_T = summ_error_model[summ_error_model.T_coherence .== T, :]
        @info summ_T
        @df summ_T scatter!(
            p[j, i],
            :F_link, :μ,#
            yerror = :se,
            group = :purify,
            # yerror = :se,
            # yscale = :log10,
            xlabel = "Bell pair fidelity "* L"$F_{link}$",
            ylabel = i == 1 ? "GHZ Fidelity  (Noise model: $(error_model))" : "",
            title = L"$T_{coherence} =$" * "$(T) s",
            titlefontsize = 10,
            legendtitle = "Purify",
            legend = i == length(Ts) ? :outerbottomright : false, 
            # ylims = (1e-2, 1e-0),
            minorgrid = true
        )
    end
end

display(p)
savefig(p, "GHZ_purification_nlogs++.pdf")

##
# mean/std for each (T_coherence, purify)
# summ = combine(groupby(df, [:F_link, :purify, :T_coherence]),
#     :infidel_log => mean => :μ,
#     :infidel_log => std  => :σ,
#     nrow => :nlogs
# )
# summ[!, :se] = summ.σ ./ sqrt.(summ.nlogs)

# mean/std for each (T_coherence, purify)
summ = combine(groupby(df, [:F_link, :purify, :error_model, :T_coherence]),
    :max_timestep => mean => :μ,
    :max_timestep => std  => :σ,
    nrow => :nlogs
)
summ[!, :se] = summ.σ ./ sqrt.(summ.nlogs)

#summ = summ[(summ.F_link .> 0.5) .& (summ.F_link .< 1.0), :]
# make subplot per T_coherence, with purified vs unpurified as different series
Ts = sort(unique(summ.T_coherence))
p = plot(
    layout = (2, length(Ts)),
    link = :y,
    size = (350 * length(Ts), 800),
    margin = 10mm
)
for (j, error_model) in enumerate(unique(summ.error_model))
    summ_error_model = summ[summ.error_model .== error_model, :]
    for (i, T) in enumerate(Ts)
        summ_T = summ_error_model[summ_error_model.T_coherence .== T, :]
        @info summ_T
        @df summ_T scatter!(
            p[j, i],
            :F_link, :μ,#
            yerror = :se,
            group = :purify,
            # yerror = :se,
            # yscale = :log10,
            xlabel = "Bell pair fidelity "* L"$F_{link}$",
            ylabel = i == 1 ? "GHZ Fidelity  (Noise model: $(error_model))" : "",
            title = L"$T_{coherence} =$" * "$(T) s",
            titlefontsize = 10,
            legendtitle = "Purify",
            legend = i == length(Ts) ? :outerbottomright : false, 
            # ylims = (1e-2, 1e-0),
            minorgrid = true
        )
    end
end

display(p)
savefig(p, "GHZ_purification_nlogs_max_timesteps.pdf")


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