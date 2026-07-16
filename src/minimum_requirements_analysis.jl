using JLD2
using Statistics
using DataFrames
using CSV
using StatsPlots

##

folder = "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1_csv"
files = readdir(folder)
files = ["ghz_service_v1_Steane7_depolarizing_$(i).csv" for i in 1:60]
##
columns =  [:attempt_time, :T_coherence, :error_model, :cutoff, :tRotationShuttle, :tCNOT, :gate_fidelity, 
            :tReadout, :readout_fidelity, :runtime, :wallclock_time, :nlogs]

dfs = DataFrame[]
for file in files[5:end]
    path = joinpath(folder, file)
    df = CSV.read(path, DataFrame)
    df = df[df.client_1 .& df.client_2 .& df.client_3 .& df.client_4, :]
    @info "Loaded file"
    isempty(df) && continue
    @info "Loaded file: $file with $(nrow(df)) rows."
    summary = combine(groupby(df, [columns;[:cutoff, :link_success_prob, :F_link]]), # add remaining parameters in order to have the values in the summary table!
        :GHZfidel => mean => :mean_GHZfidel, 
        :GHZfidel => std => :std_GHZfidel,
        :GHZfidel => x -> std(x)/sqrt(length(x)) => :sem_GHZfidel,
        :timesteps => x -> length(x)/maximum(x) => :mean_rate,
        :timesteps => x -> mean([first(x); diff(x)]) => :mean_generation_time,
        :timesteps => x -> std([first(x); diff(x)]) => :std_generation_time,
        :timesteps => x -> std([first(x); diff(x)])/sqrt(length(x)) => :sem_generation_time,)

    push!(dfs, summary)
end
df_total = vcat(dfs...)
#CSV.write(folder * "/SUMMARY_ghz_service_v1_steane7_depolarizing.csv", df_total)



##

df_out[!, :depolarizing] = df_out[!, :error_model] .== "depolarizing"
df_out[!, :dephasing] = df_out[!, :error_model] .== "dephasing"
select!(df_out, Not(:error_model))

for i in 1:4
    df_out[!, Symbol("client_$i")] = in.(i, df_out[!, :clients_serviced])
end

for col in names(df_out, Int64)
    df_out[!, col] = Int32.(df_out[!, col])
end

for col in names(df_out, Float64)
    df_out[!, col] = Float32.(df_out[!, col])
end

select!(df_out, Not(:clients_serviced))

describe(df_out)
CSV.write("/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1/ghz_service_v1_Steane7_depolarizing_23.csv", df_out)
##
@df df_out histogram(:nlogs, bins=50, xlabel="Number of Logs", ylabel="Count", title="Histogram of Mean GHZ Fidelity")
## count states per clients serviced
df = df[df.num_clients .== 4, :]

df.clients_serviced = sort.(df.clients_serviced)
df_count = combine(groupby(df, :clients_serviced), nrow => :count)
df_count[!, :absdiff] .= (sum(abs.(df_count[:,2] .- df_count[:,2]'), dims=2) ./ length(df_count[:,2])) ./ df_count[:,2]

##
@load "ghz_service_v1_depolarizing_final.jld2" df
df_depol = df[df.num_clients .== 4, :]

df = vcat(df_dephase, df_depol)
##

groupby_columns = [:error_model, :attempt_time, :link_success_prob, :T_coherence, :F_link, :runtime, :cutoff]
grouped_df = combine(groupby(df, groupby_columns),
    :GHZfidel => mean => :mean_GHZfidel,
    :GHZfidel => std => :std_GHZfidel,
    nrow => :nlogs
)
grouped_df[!, :rate] = grouped_df.nlogs ./ grouped_df.runtime

##
target_rate = 100.0 # Hz
target_fidelity = 0.99
grouped_df_above_fidelity_target = grouped_df[grouped_df.mean_GHZfidel .>= target_fidelity, :]
grouped_df_above_rate_target = grouped_df[grouped_df.rate .>= target_rate, :]
grouped_df_above_both_targets = grouped_df[(grouped_df.mean_GHZfidel .>= target_fidelity) .&& (grouped_df.rate .>= target_rate), :]


## pareto front analysis

objectives = [
    (:attempt_time, :max),
    (:link_success_prob, :min),
    (:T_coherence, :min),
    (:F_link, :min),
    (:cutoff, :max),
]

function better_or_equal(a, b, sense)
    if sense == :max
        return a >= b
    elseif sense == :min
        return a <= b
    else
        error("sense must be :min or :max")
    end
end

function strictly_better(a, b, sense)
    if sense == :max
        return a > b
    elseif sense == :min
        return a < b
    else
        error("sense must be :min or :max")
    end
end

function pareto_front(df, objectives)
    n = nrow(df)
    dominated = falses(n)

    for i in 1:n
        for j in 1:n
            i == j && continue

            j_better_or_equal_all = all(
                better_or_equal(df[j, col], df[i, col], sense)
                for (col, sense) in objectives
            )

            j_strictly_better_somewhere = any(
                strictly_better(df[j, col], df[i, col], sense)
                for (col, sense) in objectives
            )

            if j_better_or_equal_all && j_strictly_better_somewhere
                dominated[i] = true
                break
            end
        end
    end

    return df[.!dominated, :]
end

pareto_df = pareto_front(grouped_df_above_both_targets, objectives)


##
using DataFrames, Plots, Printf, LaTeXStrings
using Plots.PlotMeasures
using StatsPlots

##
params = [:attempt_time, :link_success_prob, :T_coherence, :F_link, :cutoff]

function fmtval(x)
    if x isa AbstractFloat
        isinf(x) && return "Inf"
        isnan(x) && return "NaN"
        s = @sprintf("%.4g", x)
        return replace(s, "e-0" => "e-", "e+0" => "e+")
    else
        return string(x)
    end
end

function sorted_unique(vals)
    vals = collect(skipmissing(vals))

    if all(v -> v isa Number, vals)
        return sort(unique(vals))
    else
        return sort(unique(vals), by=string)
    end
end

function count_matrix_for_param(d, p, vals, models; groupcol=:error_model)
    counts = zeros(Int, length(vals), length(models))

    for (midx, model) in enumerate(models)
        subvals = collect(skipmissing(d[d[!, groupcol] .== model, p]))

        for (vidx, v) in enumerate(vals)
            counts[vidx, midx] = count(x -> isequal(x, v), subvals)
        end
    end

    return counts
end

function plot_value_counts_many(
    dfs;
    labels = ["df$i" for i in eachindex(dfs)],
    groupcol = :error_model,
    params = [:attempt_time, :link_success_prob, :T_coherence, :F_link],
    map_params_to_latx_symbols = Dict(
        :attempt_time => L"\Delta t_\mathrm{attempt}",
        :link_success_prob => L"p_\mathrm{link}",
        :T_coherence => L"T_\mathrm{coherence}",
        :F_link => L"F_\mathrm{link}",
        :cutoff => L"\mathrm{cutoff}",
    ),
)

    # all error models across all dfs
    models = unique(vcat([collect(skipmissing(d[!, groupcol])) for d in dfs]...))
    model_labels = string.(models)

    # global unique x-values per parameter across all dfs
    global_vals = Dict(
        p => sorted_unique(vcat([collect(skipmissing(d[!, p])) for d in dfs]...))
        for p in params
    )

    subplots = Plots.Plot[]

    for (l, p) in enumerate(params)
        vals = global_vals[p]
        val_labels = fmtval.(vals)

        for (k, d) in enumerate(dfs)
            counts = count_matrix_for_param(d, p, vals, models; groupcol=groupcol)

        plt = groupedbar(
            1:length(vals),
            counts;
            label = reshape(model_labels, 1, :),
            bar_width = 0.75,
            title = l == 1 ? string(labels[k]) : "",
            ylabel = k == 1 ? string(map_params_to_latx_symbols[p]) : "",
            xticks = (1:length(vals), val_labels),
            xrotation = 35,
            legend = l == 1 && k == length(dfs) ? :outerright : false,
            grid = true,
            bottom_margin = 8mm,
            left_margin = k == 1 ? 12mm : 4mm,
            right_margin = 4mm,
        )

            push!(subplots, plt)
        end
    end

    plot(
        subplots...,
        layout = (length(params), length(dfs)),
        size = (380 * length(dfs), 260 * length(params)),
        plot_title = "Counting feasible solutions per parameter",
    )
end

plot_value_counts_many(
    [grouped_df_above_fidelity_target, grouped_df_above_rate_target, grouped_df_above_both_targets, pareto_df],;
    labels = [
        latexstring("F_{\\mathrm{GHZ}} \\geq ", target_fidelity),
        latexstring("\\mathrm{Delivery\\ rate} \\geq ", target_rate, "\\,\\mathrm{Hz}"),
        L"\mathrm{both\ targets}",
        L"\mathrm{Pareto\ front}",
    ],
)

savefig("minimum_requirements_analysis.pdf")

##
grouped_df_nocutoff = grouped_df#[grouped_df.cutoff .== Inf, :]
grouped_df_depol = grouped_df_nocutoff[grouped_df_nocutoff.error_model .== "depolarizing", :]
grouped_df_attempt_time_1ms = grouped_df_depol[grouped_df_depol.attempt_time .== 0.001, :]

grouped_df.link_success_prob  = round.(grouped_df.link_success_prob, digits=7)
@df grouped_df scatter(
    :rate, 1 .- :mean_GHZfidel,
    yerror = :std_GHZfidel/sqrt.(grouped_df.nlogs),
    group = :link_success_prob,
    yscale = :log10,
    xscale = :log10,
    ylabel = L"GHZ state infidelity $(1-F_{\mathrm{GHZ}})$",
    xlabel = "Delivery rate (Hz)",
    title = "Simulated data points",
    legendtitle = "Link success prob.",
    minorgrid = true,
    legend = :outerright,
)

## spider web plots
using Random, Measures, Plots; gr()

Random.seed!(1789)

baseline_target_values = Dict(
    :attempt_time => [0.5, 2.0],
    :link_success_prob => [3.1, 4.0],
    :T_coherence => [0.1, 2.0],
    :F_link => [0.2, 2.0],
    :Δt_measure => [0.1, 2.0]
)

improvement_factors = []
labels = []
for (key, val) in baseline_target_values
    push!(improvement_factors, abs(log(val[1])/log(val[2])))
    @info key, log(val[1])/log(val[2])
    push!(labels, string(key))
end
push!(improvement_factors, improvement_factors[1]) # periodicity for polar plot

improvement_factors_1 = improvement_factors[1:end-1] .* rand(length(improvement_factors)-1) # randomize the second plot for demonstration
push!(improvement_factors_1, improvement_factors_1[1]) # periodicity for polar plot

n = length(labels)
θ = LinRange(0, 2pi, n+1)
z = 1.15*exp.(im*2π*(0:n-1)/n)

plot(θ, improvement_factors, proj=:polar, ms=3, m=:o, c=:blue, fill=(true,:blues), fa=0.4, xaxis=false, margin=5mm, label="Steane 7") #lims=(0,1)
annotate!(real.(z), imag.(z), text.(labels,12,"Computer Modern"))
plot!(θ, improvement_factors_1, proj=:polar, ms=3, m=:o, c=:red, fill=(true,:reds), fa=0.4, label="BB 12")

