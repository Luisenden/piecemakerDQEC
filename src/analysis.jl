
##
using DataFrames
using StatsPlots 
using CSV
using LaTeXStrings

include("utils_pseudothreshold.jl")

function extract_pL(gen_time, F_GHZ; nsamples=100_000)
    p_mem_val = p_mem(gen_time * 2)  # generation time per stabilizer generator takes twice as long

    setup = CShorSyndromeECCSetup(p_mem_val, 1.0, F_GHZ)
    decoder = TableDecoder(code)

    r = cevaluate_decoder_pL(decoder, setup, nsamples)

    return (
        pL = r,
        p_mem = p_mem_val,
    )
end

default(
    fontfamily = "Computer Modern",
    labelfontsize = 14,
    tickfontsize = 14,
    linewidth = 2,
    markersize = 4,
    grid = true,
    minorgrid = true,
    framestyle = :box,
    dpi = 300,            # relevant for PNG output; PDFs are vector anyway
    size = (600, 400),
)


##

@load "summary_ghz_service_v1_Steane7_depolarizing_current_adapted.jld2" df_out


##
transform!(
    df_out,
    [:mean_generation_time, :mean_GHZfidel] =>
        ByRow((gen_time, F_GHZ) ->
            extract_pL(gen_time, F_GHZ; nsamples=1000_000)
        ) =>
        AsTable
)

##
df_out[!, :mean_generation_time] = df_out.mean_generation_time * 2
using PrettyTables
pretty_table(df_out[:,[:generator_idx, :mean_GHZfidel, :mean_generation_time, :cutoff, :pL, :p_mem]]; backend = :latex, formatters = [fmt__round(4)])

##


##
df_out_current = df_out[ df_out.generator_idx .> 0, :]


##
n_cutoffs = length(unique(round.(df_out_current.cutoff; digits=2)))
group_shapes = reshape([:circle, :rect, :utriangle, :diamond, :star5, :hexagon, :dtriangle, :pentagon][1:n_cutoffs], 1, :)

p1 = @df df_out_current groupbar(:generator_idx, :mean_GHZfidel,
    group = round.(:cutoff; digits=2),
    xlabel = "",
    ylabel = L"Mean GHZ Fidelity $\overline{F}_{\mathrm{GHZ}}$",
    legendtitle = L" $t_{\mathrm{cut}}$ (s)",
    alpha = 0.8,
    ylim = (0.8, 0.9),
    yerror = :sem_GHZfidel,
    xticks = (sort(unique(df_out_current.generator_idx)), fill("", length(unique(df_out_current.generator_idx)))),
    xminorgrid = false,
    markershape = group_shapes,
    legend = :bottomright
)

p2 = @df df_out_current scatter(:generator_idx, :mean_generation_time,
    group = round.(:cutoff; digits=2),
    xlabel = "Stabilizer generator",
    ylabel = L"Mean inter-completion time $\overline{\Delta t}_{\mathrm{GHZ}}$ (s)",
    alpha = 0.8,
    ylim = (0,0.2),
    yerror = :sem_generation_time,
    xticks = (sort(unique(df_out_current.generator_idx)), [L"G_%$i" for i in sort(unique(df_out_current.generator_idx))]),
    xminorgrid = false,
    markershape = group_shapes,
    legend=false
)

plot(p1, p2, layout = (2, 1), size = (400, 800),
    left_margin = 8Plots.mm,
    right_margin = 8Plots.mm,
    bottom_margin = 6Plots.mm,
    top_margin = 4Plots.mm,
)

savefig("mean_GHZfidelity_generation_time.pdf")

##

# 1) Count rows per unique clients_serviced group
g = combine(groupby(alllogs, :clients_serviced), nrow => :count)
sort!(g, :count, rev=true)

# 2) Make a readable label for plotting
g.group = ["[" * join(v, ",") * "]" for v in g.clients_serviced]

# 3) Plot counts per group (categorical "histogram")
@df g bar(:group, :count,
    xlabel = "clients_serviced group",
    ylabel = "count",
    legend = false,
    xrotation = 45,
)
##
using StatsPlots
using LaTeXStrings

successful_logs = alllogs[.!alllogs.discarded, :]

combined_logs = combine(groupby(successful_logs, [:F_link, :T_coherence, :cutoff, :runtime, :link_success_prob]),
    :GHZfidel => mean => :mean_fidelity,
    :GHZfidel => (x -> std(x) / sqrt(length(x))) => :std_error,
    :GHZfidel => std => :std_dev,
    nrow => :rate
)
combined_logs.rate .= combined_logs.rate ./ combined_logs.runtime  # Convert count to rate (Hz)
combined_logs[!, "Infidelity"] = 1 .- combined_logs.mean_fidelity

##
pl = @df combined_logs plot(:T_coherence, :Infidelity, yerr = :std_dev, group = :F_link,  xscale = :log10, yscale = :log10, xlabel ="Coherence time (s)", ylabel= L"1-F", 
legendtitle = L"F_{Bell}")
savefig(pl, "mean_GHZfidelity.pdf")
##
for F_link in unique(combined_logs.F_link)
    subset = combined_logs[combined_logs.F_link .== F_link, :]
    
    # Create subplot for each T_coherence
    T_coh_values = sort(unique(subset.T_coherence))
    
    plots = []
    for T_coh in T_coh_values
        data = subset[subset.T_coherence .== T_coh, :]
        
        # Main plot with fidelity
        p = @df data plot(:link_success_prob, :mean_fidelity,
            yerror = :std_error,
            group = :cutoff,
            xlabel = L"p_{\mathrm{link}}",
            ylabel = L"\mathrm{Mean\ GHZ\ Fidelity}",
            xscale = :log10,
            title = L"T_{coh} = %$(T_coh) s",
            legend = :bottomright,
            ylim = (0.0, 1.0),
            labels = reshape([isinf(t) ? L"\infty" : L"%$(t*1000)ms" for t in sort(unique(data.cutoff))], 1, :)
        )
        
        # Add twin axis for count
        p2 = twinx(p)
        @df data plot!(p2, :link_success_prob, :rate,
            group = :cutoff,
            yscale = :log10,
            xscale = :log10,
            ylabel = L"\mathrm{Rate\ (Hz)}",
            linestyle = :dash,
            alpha = 0.5,
            legend = false,
            color = reshape([1:length(unique(data.cutoff))...], 1, :))  # Match colors
        
        push!(plots, p)
    end
    
    final_plot = plot(plots..., 
        layout = (1, 3),
        size = (1400, 450),  # Slightly larger to accommodate twin axes
        plot_title = L"F_{link} = %$(F_link), T_{CNOT} = 500 \ \mu s",
        left_margin = 12Plots.mm,   # Extra space for left y-axis
        right_margin = 12Plots.mm,  # Extra space for right y-axis (twin)
        bottom_margin = 8Plots.mm,
        top_margin = 4Plots.mm,
        plot_titlevspan = 0.08,
        plot_titlefontsize = 14)
    
    savefig(final_plot, "mean_GHZfidelity_Flink$(F_link)_Δt_CNOT500.pdf")
end


##
using Statistics
using StatsBase

# Calculate variance explained by each parameter
function calculate_eta_squared(df::DataFrame, param::Symbol, response::Symbol)
    # Group by parameter and calculate means
    group_means = combine(groupby(df, param), response => mean => :group_mean)
    df_with_means = leftjoin(df, group_means, on=param)
    
    # Total sum of squares
    grand_mean = mean(df[!, response])
    SST = sum((df[!, response] .- grand_mean).^2)
    
    # Between-group sum of squares
    SSB = sum((df_with_means.group_mean .- grand_mean).^2)
    
    # Eta-squared (proportion of variance explained)
    η² = SSB / SST
    return η²
end

# Convert categorical variables to numeric for correlation
successful_logs_numeric = copy(successful_logs_clean)
successful_logs_numeric.cutoff_numeric = replace(successful_logs_numeric.cutoff, Inf => 1000.0)  # Replace Inf with large number

# Calculate correlations
using StatsBase
correlations = DataFrame(
    parameter = [:F_link, :link_success_prob, :T_coherence, :cutoff_numeric],
    correlation = [cor(successful_logs_numeric[!, p], successful_logs_numeric.GHZfidel) 
                   for p in [:F_link, :link_success_prob, :T_coherence, :cutoff_numeric]],
    abs_correlation = [abs(cor(successful_logs_numeric[!, p], successful_logs_numeric.GHZfidel)) 
                       for p in [:F_link, :link_success_prob, :T_coherence, :cutoff_numeric]]
)

sort!(correlations, :abs_correlation, rev=true)
@debug "Correlations with GHZ Fidelity:"
@debug correlations

# Analyze each parameter
successful_logs_clean = successful_logs_Δt_CNOT500[.!isnan.(successful_logs_Δt_CNOT500.GHZfidel), :]

params = [:F_link, :link_success_prob, :T_coherence, :cutoff]
effects = DataFrame(
    parameter = params,
    eta_squared = [calculate_eta_squared(successful_logs_clean, p, :GHZfidel) for p in params]
)

# Sort by effect size
sort!(effects, :eta_squared, rev=true)
@debug "Variance explained by each parameter:"
@debug effects

# Visualize
@df effects bar(string.(:parameter), :eta_squared,
    xlabel = "Parameter",
    ylabel = L"\eta^2 \mathrm{\ (Variance\ Explained)}",
    title = "Effect Size of Parameters on GHZ Fidelity",
    legend = false,
    ylim = (0, 1))
savefig("parameter_effects_eta_squared.pdf")

# Calculate the range of mean fidelities for each parameter
function calculate_fidelity_range(df::DataFrame, param::Symbol)
    means = combine(groupby(df, param), :GHZfidel => mean => :mean_fidelity)
    return maximum(means.mean_fidelity) - minimum(means.mean_fidelity)
end

ranges = DataFrame(
    parameter = params,
    fidelity_range = [calculate_fidelity_range(successful_logs_clean, p) for p in params]
)

sort!(ranges, :fidelity_range, rev=true)
@debug "Range of mean fidelities for each parameter:"
@debug ranges

@df ranges bar(string.(:parameter), :fidelity_range,
    xlabel = "Parameter",
    ylabel = L"\Delta \mathrm{Mean\ Fidelity}",
    title = "Parameter Impact on GHZ Fidelity",
    legend = false)
savefig("parameter_effects_range.pdf")


##
##
# Test homogeneity of variance assumption

using HypothesisTests

# Function to check variance across groups for each parameter
function check_homogeneity(df::DataFrame, param::Symbol, response::Symbol)
    groups = groupby(df, param)
    
    # Calculate variance for each group
    variances = combine(groups, response => var => :variance, nrow => :count)
    
    # Levene's test (more robust than Bartlett's test)
    # Manual implementation since HypothesisTests.jl might not have it
    group_data = [group[!, response] for group in groups]
    
    @debug "Variance by $(param):"
    @debug variances
    
    # Check ratio of max to min variance (rule of thumb: should be < 3-4)
    max_var = maximum(variances.variance)
    min_var = minimum(variances.variance)
    ratio = max_var / min_var
    
    println("\nVariance ratio (max/min) for $(param): $(round(ratio, digits=2))")
    if ratio > 3
        println("⚠️  WARNING: Variance assumption may be violated (ratio > 3)")
    else
        println("✓ Variance assumption seems reasonable (ratio < 3)")
    end
    
    return variances, ratio
end

# Check each parameter
@debug "=== Checking Homogeneity of Variance ==="

for param in [:F_link, :link_success_prob, :T_coherence, :cutoff]
    println("\n" * "="^50)
    variances, ratio = check_homogeneity(successful_logs_clean, param, :GHZfidel)
end

##
# Filter data for eta-squared analysis
successful_logs_filtered = successful_logs_clean[
    (successful_logs_clean.cutoff .!= 0.001) .&
    (.!(successful_logs_clean.T_coherence .∈ Ref([0.1, 1.0]))) .&
    (.!(successful_logs_clean.link_success_prob .∈ Ref([0.01, 0.1]))),
    :]

@debug "Filtered data: $(nrow(successful_logs_filtered)) rows (from $(nrow(successful_logs_clean)) original)"
@debug "Remaining parameter values:"
@debug "  F_link: $(sort(unique(successful_logs_filtered.F_link)))"
@debug "  cutoff: $(sort(unique(successful_logs_filtered.cutoff)))"
@debug "  T_coherence: $(sort(unique(successful_logs_filtered.T_coherence)))"
@debug "  link_success_prob: $(sort(unique(successful_logs_filtered.link_success_prob)))"

# Only analyze parameters that have more than one unique value
params_to_analyze = Symbol[]
for param in [:F_link, :link_success_prob, :T_coherence, :cutoff]
    n_unique = length(unique(successful_logs_filtered[!, param]))
    if n_unique > 1
        push!(params_to_analyze, param)
        @debug "  $(param): $(n_unique) unique values ✓"
    else
        @warn "  $(param): Only $(n_unique) unique value - SKIPPING η² analysis"
    end
end

if isempty(params_to_analyze)
    @error "No parameters with multiple values to analyze!"
else
    # Analyze each parameter with filtered data
    effects_filtered = DataFrame(
        parameter = params_to_analyze,
        eta_squared = [calculate_eta_squared(successful_logs_filtered, p, :GHZfidel) for p in params_to_analyze]
    )
    
    # Sort by effect size
    sort!(effects_filtered, :eta_squared, rev=true)
    @debug "Variance explained by each parameter (filtered data):"
    @debug effects_filtered
    
    # Visualize
    @df effects_filtered bar(string.(:parameter), :eta_squared,
        xlabel = "Parameter",
        ylabel = L"\eta^2 \mathrm{\ (Variance\ Explained)}",
        title = "Effect Size of Parameters on GHZ Fidelity (Filtered)",
        legend = false,
        ylim = (0, 1))
    savefig("parameter_effects_eta_squared_filtered.pdf")
    
    # Calculate ranges with filtered data
    ranges_filtered = DataFrame(
        parameter = params_to_analyze,
        fidelity_range = [calculate_fidelity_range(successful_logs_filtered, p) for p in params_to_analyze]
    )
    
    sort!(ranges_filtered, :fidelity_range, rev=true)
    @debug "Range of mean fidelities for each parameter (filtered data):"
    @debug ranges_filtered
    
    @df ranges_filtered bar(string.(:parameter), :fidelity_range,
        xlabel = "Parameter",
        ylabel = L"\Delta \mathrm{Mean\ Fidelity}",
        title = "Parameter Impact on GHZ Fidelity (Filtered)",
        legend = false)
    savefig("parameter_effects_range_filtered.pdf")
    
    # Check homogeneity with filtered data
    @debug "=== Checking Homogeneity of Variance (Filtered Data) ==="
    
    for param in params_to_analyze
        println("\n" * "="^50)
        variances, ratio = check_homogeneity(successful_logs_filtered, param, :GHZfidel)
    end
    
    # Compare filtered vs unfiltered results (only for parameters in both)
    common_params = intersect(params, params_to_analyze)
    if !isempty(common_params)
        comparison = DataFrame(
            parameter = common_params,
            eta_squared_all = [effects[effects.parameter .== p, :eta_squared][1] for p in common_params],
            eta_squared_filtered = [effects_filtered[effects_filtered.parameter .== p, :eta_squared][1] for p in common_params],
        )
        comparison.difference = comparison.eta_squared_filtered .- comparison.eta_squared_all
        
        sort!(comparison, :eta_squared_filtered, rev=true)
        @debug "Comparison: All data vs Filtered data:"
        @debug comparison
    end
end

##
# Analysis: Which parameter combinations achieve F_GHZ ≥ 0.95?

threshold_fidelity = 0.95

# Get mean fidelity for each parameter combination
param_combinations = combine(groupby(successful_logs_Δt_CNOT500, 
    [:F_link, :link_success_prob, :T_coherence, :cutoff]),
    :GHZfidel => mean => :mean_fidelity,
    :GHZfidel => std => :std_fidelity,
    nrow => :count
)

# Filter for combinations that meet the threshold
high_fidelity_params = param_combinations[param_combinations.mean_fidelity .>= threshold_fidelity, :]
sort!(high_fidelity_params, :mean_fidelity, rev=true)

@debug "=== Parameter Combinations Achieving F_GHZ ≥ $(threshold_fidelity) ==="
@debug "Found $(nrow(high_fidelity_params)) out of $(nrow(param_combinations)) combinations"
@debug high_fidelity_params

# Analyze minimum required values for each parameter
@debug "\n=== Minimum Parameter Requirements for F_GHZ ≥ $(threshold_fidelity) ==="

println("\nF_link:")
f_link_analysis = combine(groupby(high_fidelity_params, :F_link), nrow => :n_combinations)
sort!(f_link_analysis, :F_link)
println(f_link_analysis)
println("Minimum F_link required: $(minimum(high_fidelity_params.F_link))")

println("\nT_coherence:")
t_coh_analysis = combine(groupby(high_fidelity_params, :T_coherence), nrow => :n_combinations)
sort!(t_coh_analysis, :T_coherence)
println(t_coh_analysis)
println("Minimum T_coherence required: $(minimum(high_fidelity_params.T_coherence)) s")

println("\nlink_success_prob:")
p_link_analysis = combine(groupby(high_fidelity_params, :link_success_prob), nrow => :n_combinations)
sort!(p_link_analysis, :link_success_prob)
println(p_link_analysis)
println("Note: All link success probabilities can achieve ≥ $(threshold_fidelity) with right parameters")

println("\ncutoff:")
cutoff_analysis = combine(groupby(high_fidelity_params, :cutoff), nrow => :n_combinations)
high_fidelity_params.cutoff_str = [isinf(c) ? "∞" : string(c) for c in high_fidelity_params.cutoff]
cutoff_analysis.cutoff_str = [isinf(c) ? "∞" : string(c) for c in cutoff_analysis.cutoff]
sort!(cutoff_analysis, :cutoff)
println(cutoff_analysis)

# Create a summary table showing necessary AND sufficient conditions
@debug "\n=== Summary: Necessary Conditions ==="
println("To achieve F_GHZ ≥ $(threshold_fidelity), you MUST have:")
println("  • F_link ≥ $(minimum(high_fidelity_params.F_link))")
println("  • T_coherence ≥ $(minimum(high_fidelity_params.T_coherence)) s")

# Visualize: Heatmap showing which combinations work
using StatsPlots

# Create a pivot table for visualization
for cutoff_val in sort(unique(param_combinations.cutoff))
    for t_coh in sort(unique(param_combinations.T_coherence))
        data = param_combinations[
            (param_combinations.cutoff .== cutoff_val) .& 
            (param_combinations.T_coherence .== t_coh), :]
        
        if nrow(data) > 0
            # Create pivot: F_link vs link_success_prob
            pivot_data = unk(data, :link_success_prob, :F_link, :mean_fidelity)
            
            # Convert to matrix for heatmap
            f_link_vals = sort(unique(data.F_link))
            p_link_vals = sort(unique(data.link_success_prob))
            
            matrix = zeros(length(p_link_vals), length(f_link_vals))
            for (i, p) in enumerate(p_link_vals)
                for (j, f) in enumerate(f_link_vals)
                    subset = data[(data.link_success_prob .== p) .& (data.F_link .== f), :]
                    if nrow(subset) > 0
                        matrix[i, j] = subset.mean_fidelity[1]
                    else
                        matrix[i, j] = NaN
                    end
                end
            end
            
            cutoff_str = isinf(cutoff_val) ? "∞" : "$(cutoff_val*1000)ms"
            
            p = heatmap(f_link_vals, p_link_vals,
                matrix,
                xlabel = L"F_{link}",
                ylabel = L"p_{link}",
                title = "T_coh=$(t_coh)s, cutoff=$(cutoff_str)",
                c = :RdYlGn,
                clim = (0.5, 1.0),
                colorbar_title = "Mean F_GHZ",
                yscale = :log10)
            
            # Add threshold line
            hline!([threshold_fidelity], color=:black, linewidth=2, linestyle=:dash, label="Threshold")
            
            savefig(p, "fidelity_heatmap_Tcoh$(t_coh)_cutoff$(cutoff_str).pdf")
        end
    end
end

# Create a decision tree style summary
@debug "\n=== Decision Guide ==="
println("\nIf you want F_GHZ ≥ $(threshold_fidelity):")
println("\n1. ESSENTIAL: Choose F_link ≥ $(minimum(high_fidelity_params.F_link))")
println("   Best: F_link ≥ $(quantile(high_fidelity_params.F_link, 0.5))")

println("\n2. ESSENTIAL: Choose T_coherence ≥ $(minimum(high_fidelity_params.T_coherence)) s")
println("   Best: T_coherence ≥ $(quantile(high_fidelity_params.T_coherence, 0.5)) s")

println("\n3. Cutoff time:")
cutoff_freq = combine(groupby(high_fidelity_params, :cutoff), nrow => :count)
sort!(cutoff_freq, :count, rev=true)
println("   Most reliable: cutoff = $(isinf(cutoff_freq.cutoff[1]) ? "∞ (no cutoff)" : "$(cutoff_freq.cutoff[1]*1000)ms")")

println("\n4. Link success probability:")
println("   Any value works if conditions 1-3 are met")
println("   But higher p_link → higher rate")

# Create a barplot showing "success rate" for each parameter value
for param in [:F_link, :T_coherence, :cutoff, :link_success_prob]
    all_values = unique(param_combinations[!, param])
    success_counts = Int[]
    total_counts = Int[]
    
    for val in all_values
        subset_all = param_combinations[param_combinations[!, param] .== val, :]
        subset_success = high_fidelity_params[high_fidelity_params[!, param] .== val, :]
        push!(total_counts, nrow(subset_all))
        push!(success_counts, nrow(subset_success))
    end
    
    success_rate = success_counts ./ total_counts
    
    if param == :cutoff
        labels = [isinf(v) ? "∞" : "$(v*1000)ms" for v in all_values]
    else
        labels = string.(all_values)
    end
    
    p = bar(labels, success_rate,
        xlabel = string(param),
        ylabel = "Success Rate",
        title = "Fraction of combinations achieving F_GHZ ≥ $(threshold_fidelity)",
        legend = false,
        ylim = (0, 1),
        xrotation = 45)
    
    annotate!(p, [(i, success_rate[i] + 0.05, text("$(success_counts[i])/$(total_counts[i])", 8)) 
                  for i in 1:length(all_values)])
    
    savefig(p, "success_rate_$(param).pdf")
end

# Summary statistics table
summary_table = DataFrame(
    Parameter = ["F_link", "T_coherence", "cutoff", "link_success_prob"],
    Minimum_Required = [
        minimum(high_fidelity_params.F_link),
        minimum(high_fidelity_params.T_coherence),
        minimum(filter(!isinf, high_fidelity_params.cutoff)),
        minimum(high_fidelity_params.link_success_prob)
    ],
    Recommended = [
        quantile(high_fidelity_params.F_link, 0.75),
        quantile(high_fidelity_params.T_coherence, 0.75),
        quantile(filter(!isinf, high_fidelity_params.cutoff), 0.75),
        quantile(high_fidelity_params.link_success_prob, 0.75)
    ]
)

@debug "\n=== Summary Table ==="
@debug summary_table

# Save results
using CSV
CSV.write("high_fidelity_parameter_combinations.csv", high_fidelity_params)
CSV.write("parameter_requirements_summary.csv", summary_table)
@debug "\nResults saved to CSV files"

##

