using QuantumSavory
using QuantumSavory.ProtocolZoo
using ResumableFunctions
using ConcurrentSim
using Graphs
using DataFrames
using StatsBase
using StatsPlots
using Random

const perfect_pair = (Z1⊗Z1 + Z2⊗Z2) / sqrt(2)
keep_trying = true

@resumable function keep_entangling(sim, net, link_success_prob, α)
    while true
        length(idcs_to_retry) == 0 && break
        @yield @process noskip_entangler(sim, net, link_success_prob, α) # launch first entanglement attempt sequentially on all nodes
        @info "Launched entanglement attempts for slots $(idcs_to_retry) at $(now(sim)). Wait for entangling events to arrive and check which slots need to retry."
    end
    return
end

@resumable function noskip_entangler(sim, net, link_success_prob, α) 
    if length(idcs_to_retry) * α <= 1.0
        attempt_time = 1.0
    else
        attempt_time = length(idcs_to_retry) * α
    end
    
    entanglers = EntanglerProt[]
    for i in idcs_to_retry
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1 + i, chooseA = 1, nodeB = 1, chooseB = i,
            success_prob = link_success_prob, rounds = 1, attempts = 1, attempt_time = attempt_time,
            local_busy_time_pre = 0.0, local_busy_time_post = 0.0, retry_lock_time = 0.001,
        )
        push!(entanglers, entangler)
    end
    
    for ent in entanglers
        @process ent()
        @yield timeout(sim, α) # initial delay between attempts
    end
end

@resumable function BellPairNoiseProt(sim, net, n, link_success_prob, logging::DataFrame)

    fidelities = Float64[]
    arrived_pairs = RegRef[]

    while length(arrived_pairs) ≤ n
        keeptrying = true
        
        @yield onchange_tag(net[1])
        keep_trying = false
        while true
            msg = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if isnothing(msg)
                @info "No new Bell pair established at $(now(sim)). Wait for next entanglement attempt to complete and check again."
                break
            end
            slot, _, _ = msg  
            pop!(idcs_to_retry, slot.idx)  
            push!(arrived_pairs, slot)
            @info "New Bell pair established between switch and client message: $(msg)."
            if length(arrived_pairs) > n-1
                @info "All Bell pairs established at $(now(sim)). Calculate fidelities, clear states and reset tracker."
                for slot in arrived_pairs
                    fidel = real(observable([slot, net[1+slot.idx][1]], projector(perfect_pair)))
                    push!(fidelities, fidel)
                end
                push!(logging, (n = n, p = link_success_prob, fidelities = fidelities))
                @info "RETURN"
                return
            end
        end
        @info "Slots that need to retry entanglement attempt: $(idcs_to_retry). $(arrived_pairs) have been successful so far."
    end
end

function prepare_sim(n, link_success_prob, logging, α, λ)

    states_representation = QuantumOpticsRepr()
    
    T2time = -1/log(λ)
    noise_model = T2Dephasing(T2time) # noise model applied to the memory qubits

    # Network setup
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n])
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n] # client qubits

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    # Discrete event simulation
    sim = get_time_tracker(net)

    @process keep_entangling(sim, net, link_success_prob, α)
    @process BellPairNoiseProt(sim, net, n, link_success_prob, logging)
    return sim, logging
end

## sim non skipped time bins
n = 7
α = 1.0 # sequential share
λ = 0.9 # noise parameter for T2 dephasing

n = 7
idcs_to_retry = Set(1:n)

ntrials = 1
dfs = DataFrame[]
for p in [0.2]#range(0.05, stop=1.0, length=2)
    for _ in 1:ntrials
        logging = DataFrame(n = Int[], p = Float64[], fidelities = Vector{Float64}[])
        sim, logging = prepare_sim(n, p, logging, α, λ)
        timed = @elapsed run(sim)
        push!(dfs, logging)
        @info "Simulation completed in $(timed) seconds for p=$(p)"
    end
end
df_results = vcat(dfs...)

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

pal = palette(:default)
color1 = pal[1]
color2 = pal[2]

p1 = scatter(df_grouped.p, df_grouped.avg_time_diff, yerror = df_grouped.se_time_diff, 
            label="simulated overlapping time bins, α=$(α)", xlabel = "Link Success Probability", 
            ylabel = "E[|τ₁ - τₙ|]", title = "Entangling events n=$(n)", legend=:topright, color=:red, marker =:x, 
            )#xlims = (0.1,0.7))
# scatter!(p1, df_grouped_parallel.p, df_grouped_parallel.avg_time_diff, yerror = df_grouped_parallel.se_time_diff, label="simulated parallel time bins", color=color2
x = range(0.01, stop=1.0, length=100)
# plot!(p1, x, (2.0 .- x) ./ x, label = " (2 - p) / p sequential time bins", lw=1, color=color1, ls=:dot)
# plot!(p1, x, 1.0 ./ x .+ 1.0 ./ (x .- 2), label = "1/p + 1/(p-2) parallel time bins", lw=1, color=color2, ls=:dot)
plot!(p1, x, (n-1) ./ x, label = "(n-1)/p sequential", lw=1, color=:black, ls=:dot, ylims=(0,25))