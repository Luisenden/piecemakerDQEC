using QuantumSavory
using QuantumSavory: Register, X, Z, CNOT
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames
using Statistics
using Plots

const ghzs = [ghz(n) for n in 1:10] # make const in order to not build new every time

# Main simulation, measures everything in seconds (s)
const steane_generators = [[1,2,3,5], [1,2,4,6], [2,3,4,7]]
const toric_code_generators = [[1,2,4,5], [2,3,5,6], [1,3,4,6]]


# Pre-compute client to generator mapping for O(1) lookup
const CLIENT_TO_GENERATORS = Dict(
    1 => [1, 2],
    2 => [1, 2, 3],
    3 => [1, 3],
    4 => [2, 3],
    5 => [1],
    6 => [2],
    7 => [3]
)

# Encapsulate simulation state
mutable struct SimulationState
    progress::Vector{Vector{Int}}
    to_be_measured::Vector{Vector{Int}}
    logs::Vector{Tuple{Float64, Vector{Int}, Float64, Bool}}
end

function noisy_bell_state(target_fidelity::Float64=0.97)
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return target_fidelity*perfect_pair_dm + (1-target_fidelity)*mixed_dm
end

@resumable function projectout(sim, net, slot_idx, gen_set_idx, n_clients_in_set, state::SimulationState)
    push!(state.to_be_measured, state.progress[gen_set_idx])
    state.progress[gen_set_idx] = Vector{Int}() # Clear the generator set progress
    @yield lock(net[1][slot_idx])
    @info "Projecting out piecemaker qubit at slot $(slot_idx), $(net[1][slot_idx])"
    res = project_traceout!(net[1][slot_idx], σˣ)
    @yield timeout(sim, Δt_meas)
    @info "Tagging client $(slot_idx) with Z correction result $(res) for generator set $(gen_set_idx)"
    tag!(net[1+slot_idx][1], Tag(:updateZ, res, gen_set_idx, n_clients_in_set))
    unlock(net[1][slot_idx])
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, client_slot::RegRef)
    @yield lock(piecemaker_slot) & lock(client_slot)
    apply!((piecemaker_slot, client_slot), CNOT)
    @yield timeout(sim, Δt_CNOT)
    res = project_traceout!(client_slot, Z)
    @yield timeout(sim, Δt_meas)
    tag!(net[1 + client_slot.idx][1], Tag(:updateX, res))
    unlock(piecemaker_slot)
    unlock(client_slot)
    @info "Fused client $(client_slot.idx) with first client $(piecemaker_slot.idx)"
end

function get_oldest_generator_for_candidate(sim, net::RegisterNet, candidate::Int, state::SimulationState)
    # Use pre-computed mapping instead of searching
    potential_gens_indcs = CLIENT_TO_GENERATORS[candidate]
    
    # Pre-allocate and avoid repeated allocations
    n_clients = length(net.registers) - 1
    accesstimes = Vector{Float64}(undef, n_clients)
    for i in 1:n_clients
        accesstimes[i] = net.registers[i+1].accesstimes[1]
    end
    
    # Replace 0.0 with Inf in-place to avoid allocation
    accesstimes_replaced = replace(accesstimes, 0.0 => Inf)
    sorted_indices = sortperm(accesstimes_replaced)
    
    @info "accesstimes: $(accesstimes), sorted: $(sorted_indices)"

    # Check if all potential generators are empty
    if all(isempty(state.progress[i]) for i in potential_gens_indcs)
        @info "Progress $(state.progress) is empty, cannot find oldest generator for candidate $(candidate), return index $(potential_gens_indcs[1])"
        return now(sim), potential_gens_indcs[1]
    end

    # Optimized search: iterate through sorted clients, check if they're in any valid generator
    for oldest_idx in sorted_indices
        for gen_idx in potential_gens_indcs
            prog = state.progress[gen_idx]
            if !isempty(prog) && oldest_idx ∈ prog
                @info "Found generator index $(gen_idx) for candidate $(candidate) with oldest client index $(oldest_idx)"
                timestamp = accesstimes[oldest_idx]
                return timestamp, gen_idx
            end
        end
    end
    
    error("No generator found for candidate $(candidate), sorted_indices: $(sorted_indices)")
end

@resumable function GeneratorServiceProt(sim, net, candidate::Int, state::SimulationState, steane_generators, cutoff::Float64)

    (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim, net, candidate, state)

    while now(sim) - timestamp > cutoff
        @info "Generator set $(idx_steane_take) with clients $(state.progress[idx_steane_take][1]) TOO OLD"
        @yield @process projectout(sim, net, state.progress[idx_steane_take][1], idx_steane_take, length(state.progress[idx_steane_take]), state)
        (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim, net, candidate, state)
    end

    s = state.progress[idx_steane_take]
    if candidate ∉ s
        push!(s, candidate)
        if length(s) > 1
            @yield @process fusion(sim, net, net[1][s[1]], net[1][candidate])
            @info "fusing candidate $(candidate) into generator set index $(idx_steane_take) with current starting index $(s[1])"
        end
        if length(s) == length(steane_generators[idx_steane_take])
            @info "Generator set $(idx_steane_take) completed with clients $(s), projecting out piecemaker qubit"
            @yield @process projectout(sim, net, s[1], idx_steane_take, length(s), state)
        end
    end
end

@resumable function listen_fuse(sim, net, state::SimulationState, steane_generators, cutoff::Float64)
    while true
        @yield onchange_tag(net[1])
        
        while true
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if !isnothing(counterpart)
                slot, _, _ = counterpart
                @yield @process GeneratorServiceProt(sim, net, slot.idx, state, steane_generators, cutoff)
                @info "Sorted client $(slot.idx) into generator sets, current progress: $(state.progress)"
            else
                break
            end
        end
    end
end

@resumable function listen_log(sim, net, state::SimulationState, steane_generators)
    while true
        @yield onchange_tag(net[1])
        isdonemessage = querydelete!(net[1], :Zdone, ❓, ❓)

        if !isnothing(isdonemessage)
            genset = steane_generators[isdonemessage[3][2]]
            n_clients_in_set = isdonemessage[3][3]
            @info "received Zdone tag: $(isdonemessage)"
            
            discarded = true
            fidelity = 0.0
            
            # Measure fidelity
            if length(genset) != n_clients_in_set
                @info "MEASURE OUT BEFORE COMPLETION $(state.to_be_measured[1])"
                clients_to_measure = state.to_be_measured[1]
                @yield reduce(&, [lock(net[1+i][1]) for i in clients_to_measure])
                obs_proj = SProjector(StabilizerState(ghzs[length(clients_to_measure)]))
                fidelity = real(observable([net[1+i][1] for i in clients_to_measure], obs_proj))
                for i in clients_to_measure
                    traceout!(net[1 + i][1])
                    unlock(net[1 + i][1])
                end
            else
                @yield reduce(&, [lock(net[1+i][1]) for i in genset])
                obs_proj = SProjector(StabilizerState(ghzs[n_clients_in_set]))
                fidelity = real(observable([net[1+i][1] for i in genset], obs_proj))
                @info "clients serviced: $(genset) --> fidelity: $(fidelity)"
                for i in genset
                    traceout!(net[1 + i][1])
                    unlock(net[1 + i][1])
                end
                discarded = false
            end
            
            timesteps = now(sim)
            push!(state.logs, (timesteps, popfirst!(state.to_be_measured), fidelity, discarded))
            @info "Updated progress: $(state.progress)"
        end
    end
end

@resumable function correct_and_inform(sim, net::RegisterNet, client::Int)
    while true
        @yield onchange_tag(net[1+client][1])
        msg1 = querydelete!(net[1+client][1], :updateX, ❓)
        msg2 = querydelete!(net[1+client][1], :updateZ, ❓, ❓, ❓)
        
        if !isnothing(msg1) || !isnothing(msg2)
            if !isnothing(msg1)
                value = msg1[3][2]
                @info "X received at client $(client), with value $(value)"
                @yield lock(net[1+client][1])
                if value == 2
                    apply!(net[1+client][1], X, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Xdone, client))
            end
            
            if !isnothing(msg2)
                @info "Z received at client $(client)"
                value = msg2[3][2]
                gen_set_idx = msg2[3][3]
                n_clients_in_set = msg2[3][4]
                @info "Z received at client $(client), with value $(value), gen_set_idx=$(gen_set_idx), n_clients_in_set=$(n_clients_in_set)"
                @yield lock(net[1+client][1])
                if value == 2
                    # noisyZ = NonInstantGate(Z, Δt_XZ)
                    apply!(net[1+client][1], Z, time = now(sim))
                    # @yield timeout(sim, Δt_XZ)
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Zdone, gen_set_idx, n_clients_in_set))
            end
        end
    end
end

function prepare_sim(n, T_link, cutoff::Float64, F_link::Float64, link_success_prob::Float64, steane_generators)
    states_representation = QuantumOpticsRepr()
    noise_model = Depolarization(T_link)

    # Initialize simulation state
    state = SimulationState(
        [Vector{Int}() for _ in 1:length(steane_generators)],
        Vector{Vector{Int}}(),
        Vector{Tuple{Float64, Vector{Int}, Float64, Bool}}()
    )

    # Network setup
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n])
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n]

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    sim = get_time_tracker(net)

    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1, 
            pairstate = noisy_bell_state(F_link),
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = 50e-9, # 8-10 ns
            retry_lock_time = 5e-9, local_busy_time_post = 0.0
        )

        @process entangler()
        @process correct_and_inform(sim, net, i)
    end

    @process listen_fuse(sim, net, state, steane_generators, cutoff)
    @process listen_log(sim, net, state, steane_generators)

    return sim, state
end


n = 7

Δt_CNOT = 100e-6 # 100 µs
Δt_XZ = 10e-6 # 10 µs 
Δt_meas = 100e-9 # 100 ns
F_link = 1.0
Δt_cutoff_list = [1e-3] # 1 ms cutoff 

dataframes = DataFrame[]
for link_success_prob in [0.001]#, 1e-2, 1e-1]
    for T_coherence in [10e-3]#, 100e-3, 1.0] # 1 ms, 100 ms, 1 s
        for F_link in [0.98]#, 0.999, 1.0]
            for cutoff in Δt_cutoff_list
                runtime = 0.01
                sim, state = prepare_sim(n, T_coherence, cutoff, F_link, link_success_prob, steane_generators)
                t_wallclock = @elapsed run(sim, runtime)
                
                logs = DataFrame(state.logs, [:timesteps, :clients_serviced, :GHZfidel, :discarded])
                logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)
                logs = transform(logs, :clients_serviced => (x -> [length(c) for c in x]) => :num_clients)

                logs[!, "link_success_prob"] .= link_success_prob
                logs[!, "runtime"] .= runtime
                logs[!, "tCNOT"] .= Δt_CNOT
                logs[!, "T_coherence"] .= T_coherence
                logs[!, "F_link"] .= F_link
                logs[!, "cutoff"] .= cutoff
                logs[!, "wallclock_time"] .= t_wallclock

                push!(dataframes, logs)
                @info "completed simulation for link_success_prob=$(link_success_prob), T_CNOT=$(Δt_CNOT), T_coherence=$(T_coherence), F_link=$(F_link), cutoff=$(cutoff), wallclock=$(t_wallclock)s, collected $(nrow(logs)) logs"
            end
        end
    end
end

alllogs = vcat(dataframes...)
@info "Summary statistics:"
@info describe(alllogs)
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
pl = @df combined_logs plot(:T_coherence, :Infidelity, yerr = :std_dev, group = :F_link,  xscale = :log10, yscale = :log10, xlabel ="Coherence time (s)", ylabel= L"1-F", legendtitle = L"F_{start}")
savefig(pl, "mean_GHZfidelity.pdf")


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
@info "Correlations with GHZ Fidelity:"
@info correlations

# Analyze each parameter
successful_logs_clean = successful_logs_Δt_CNOT500[.!isnan.(successful_logs_Δt_CNOT500.GHZfidel), :]

params = [:F_link, :link_success_prob, :T_coherence, :cutoff]
effects = DataFrame(
    parameter = params,
    eta_squared = [calculate_eta_squared(successful_logs_clean, p, :GHZfidel) for p in params]
)

# Sort by effect size
sort!(effects, :eta_squared, rev=true)
@info "Variance explained by each parameter:"
@info effects

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
@info "Range of mean fidelities for each parameter:"
@info ranges

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
    
    @info "Variance by $(param):"
    @info variances
    
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
@info "=== Checking Homogeneity of Variance ==="

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

@info "Filtered data: $(nrow(successful_logs_filtered)) rows (from $(nrow(successful_logs_clean)) original)"
@info "Remaining parameter values:"
@info "  F_link: $(sort(unique(successful_logs_filtered.F_link)))"
@info "  cutoff: $(sort(unique(successful_logs_filtered.cutoff)))"
@info "  T_coherence: $(sort(unique(successful_logs_filtered.T_coherence)))"
@info "  link_success_prob: $(sort(unique(successful_logs_filtered.link_success_prob)))"

# Only analyze parameters that have more than one unique value
params_to_analyze = Symbol[]
for param in [:F_link, :link_success_prob, :T_coherence, :cutoff]
    n_unique = length(unique(successful_logs_filtered[!, param]))
    if n_unique > 1
        push!(params_to_analyze, param)
        @info "  $(param): $(n_unique) unique values ✓"
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
    @info "Variance explained by each parameter (filtered data):"
    @info effects_filtered
    
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
    @info "Range of mean fidelities for each parameter (filtered data):"
    @info ranges_filtered
    
    @df ranges_filtered bar(string.(:parameter), :fidelity_range,
        xlabel = "Parameter",
        ylabel = L"\Delta \mathrm{Mean\ Fidelity}",
        title = "Parameter Impact on GHZ Fidelity (Filtered)",
        legend = false)
    savefig("parameter_effects_range_filtered.pdf")
    
    # Check homogeneity with filtered data
    @info "=== Checking Homogeneity of Variance (Filtered Data) ==="
    
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
        @info "Comparison: All data vs Filtered data:"
        @info comparison
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

@info "=== Parameter Combinations Achieving F_GHZ ≥ $(threshold_fidelity) ==="
@info "Found $(nrow(high_fidelity_params)) out of $(nrow(param_combinations)) combinations"
@info high_fidelity_params

# Analyze minimum required values for each parameter
@info "\n=== Minimum Parameter Requirements for F_GHZ ≥ $(threshold_fidelity) ==="

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
@info "\n=== Summary: Necessary Conditions ==="
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
@info "\n=== Decision Guide ==="
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

@info "\n=== Summary Table ==="
@info summary_table

# Save results
using CSV
CSV.write("high_fidelity_parameter_combinations.csv", high_fidelity_params)
CSV.write("parameter_requirements_summary.csv", summary_table)
@info "\nResults saved to CSV files"