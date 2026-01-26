using QuantumSavory
using QuantumSavory.ProtocolZoo
using ResumableFunctions
using ConcurrentSim
using Graphs
using DataFrames
using StatsBase
using StatsPlots
using Random

function clear_states!(net, n)
    for i in 1:n
        traceout!(net[1][i])
        traceout!(net[1 + i][1])
    end
end

@resumable function tm_entangler(sim, net, n, link_success_prob,t_detect)
    if t_detect <= 0.5
        attempt_time = 1.0
    else
        attempt_time = 2.0 * t_detect
    end
    
    entanglers = EntanglerProt[]
    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1,
            success_prob = link_success_prob, rounds = 1, attempts = -1, attempt_time = attempt_time,
            local_busy_time_pre = 0.0, local_busy_time_post = 0.0, retry_lock_time = 0.001,
        )
        push!(entanglers, entangler)
    end
    
    for ent in entanglers
        @process ent()
        @yield timeout(sim, t_detect) # initial delay between attempts
    end
end

@resumable function BellPairTracker(sim, net, n, link_success_prob, logging::DataFrame)
    count = 0
    first_arrival_time = 0.0
    last_arrival_time = 0.0
    while true
        @yield onchange_tag(net[1])
        current_time_stamp = now(sim)
        while true
            msg = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if isnothing(msg)
                break
            end
            slot, _, _ = msg
            @debug "New Bell pair established between switch and client $(slot.idx)"    
            count += 1
            if count == 1
                first_arrival_time = current_time_stamp
            end
            if count > n-1
                @debug "All Bell pairs established at $(now(sim)). Clear states and reset counter"
                last_arrival_time = current_time_stamp
                time_diff = last_arrival_time - first_arrival_time
                @debug "Time difference between first and last entangling event: $(time_diff)"
                push!(logging, (n = n, p = link_success_prob, first_arrival = first_arrival_time, last_arrival = last_arrival_time, time_diff = time_diff))
                count = 0
                #clear_states!(net, n)
                return
            end
        end
    end
end

function prepare_sim_parallel(n, link_success_prob, logging)

    states_representation = QuantumOpticsRepr()
    
    T_link = 10
    noise_model = Depolarization(T_link) # noise model applied to the memory qubits

    # Network setup
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n])
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n] # client qubits

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    # Discrete event simulation
    sim = get_time_tracker(net)

    @process BellPairTracker(sim, net, n, link_success_prob, logging)

    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1,
            success_prob = link_success_prob, rounds = 1, attempts = -1, attempt_time = 1.0,
            local_busy_time_pre = 0.0, local_busy_time_post = 0.0, retry_lock_time = 0.001,
        )
        @process entangler()
    end
    return sim, logging
end

function prepare_sim_tm(n, link_success_prob, logging, t_detect)

    states_representation = QuantumOpticsRepr()
    
    T_link = 10
    noise_model = Depolarization(T_link) # noise model applied to the memory qubits

    # Network setup
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n])
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n] # client qubits

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    # Discrete event simulation
    sim = get_time_tracker(net)

    @process BellPairTracker(sim, net, n, link_success_prob, logging)
    @process tm_entangler(sim, net, n, link_success_prob, t_detect)

    return sim, logging
end

## sim sequential attempts
n = 2
t_detect = 1.0

ntrials = 100
dfs = DataFrame[]
for p in range(0.05, stop=1.0, length=20)
    for _ in 1:ntrials
        logging = DataFrame(n = Int[], p = Float64[], first_arrival = Float64[], last_arrival = Float64[], time_diff = Float64[])
        sim, logging = prepare_sim_tm(n, p, logging, t_detect)
        timed = @elapsed run(sim)
        push!(dfs, logging)
        @info "Simulation completed in $(timed) seconds for p=$(p)"
    end
end
df_results = vcat(dfs...)

## sim parallel
n = 2
ntrials = 100
dfs = DataFrame[]
for p in range(0.05, stop=1.0, length=20)
    for _ in 1:ntrials
        logging = DataFrame(n = Int[], p = Float64[], first_arrival = Float64[], last_arrival = Float64[], time_diff = Float64[])
        sim, logging = prepare_sim_parallel(n, p, logging)
        timed = @elapsed run(sim)
        push!(dfs, logging)
        @info "Simulation completed in $(timed) seconds for p=$(p)"
    end
end
df_results_parallel = vcat(dfs...)


##
using Statistics
df_grouped = combine(groupby(df_results, [:n, :p]), 
    :time_diff => mean => :avg_time_diff,
    :time_diff => sem => :se_time_diff,
    :time_diff => length => :num_samples,
    :first_arrival => mean => :first_avg,
    :last_arrival => mean => :last_avg
)
show(df_grouped, allrows=true)
##
using Statistics
df_grouped_parallel = combine(groupby(df_results_parallel, [:n, :p]), 
    :time_diff => mean => :avg_time_diff,
    :time_diff => sem => :se_time_diff,
    :time_diff => length => :num_samples,
    :first_arrival => mean => :first_avg,
    :last_arrival => mean => :last_avg
)
show(df_grouped_parallel, allrows=true)
##

pal = palette(:default)
color1 = pal[1]
color2 = pal[2]

p1 = scatter(df_grouped.p, df_grouped.avg_time_diff, yerror = df_grouped.se_time_diff, label="simulated sequential time bins", xlabel = "Link Success Probability", ylabel = "E[|τ₁ - τ₂|]", title = "Entangling events n=2", legend=:topright, color=color1)
scatter!(p1, df_grouped_parallel.p, df_grouped_parallel.avg_time_diff, yerror = df_grouped_parallel.se_time_diff, label="simulated parallel time bins", color=color2)
x = range(0.01, stop=1.0, length=100)
plot!(p1, x, (2.0 .- x) ./ x, label = " (2 - p) / p sequential time bins", lw=2, ylims=(0,25), color=color1)
plot!(p1, x, 1.0 ./ x .+ 1.0 ./ (x .- 2), label = "1/p + 1/(p-2) parallel time bins", lw=2, color=color2)
plot!(p1, x, (n-1) ./ x, label = "(n-1)/p sequential", lw=2, color=:gray, ylims=(0,25))