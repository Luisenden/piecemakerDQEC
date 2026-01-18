using QuantumSavory
using QuantumSavory: Register, X, Z, Y, I, CNOT, PauliNoiseCPTP
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames
using Statistics
using Plots
using CSV

function gate_noise_model(p_g::Float64)
    @assert 0.0 ≤ p_g ≤ 1.0
    cumprob = 1.0
    r = rand()
    for (q, op) in [(p_g/3, X), (p_g/3, Y), (p_g/3, Z)]
        cumprob -= q
        if r > cumprob
            return op
        end
    end
    return I
end

function noisy_bell_state(target_fidelity::Float64=0.97)
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return target_fidelity*perfect_pair_dm + (1-target_fidelity)*mixed_dm
end

@resumable function projectout(sim, net, slot_idx, gate_noise::Float64)
    @yield lock(net[1][slot_idx]) & lock(net[1+slot_idx][1])
    @debug "Projecting out piecemaker qubit at slot $(slot_idx), $(net[1][slot_idx])"
    @yield timeout(sim, t_meas)
    res = project_traceout!(net[1][slot_idx], X)
    @debug "Tagging client $(slot_idx) with Z correction result $(res)"
    tag!(net[1+slot_idx][1], Tag(:updateZ, res)) # communicate change to latest node
    unlock(net[1][slot_idx])
    unlock(net[1+slot_idx][1])
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, client_slot::RegRef, gate_noise::Float64)
    @yield lock(piecemaker_slot) & lock(client_slot) & lock(net[1 + client_slot.idx][1])
    @yield timeout(sim, t_CNOT)
    apply!((piecemaker_slot, client_slot), CNOT)
    op1 = gate_noise_model(gate_noise)
    op2 = gate_noise_model(gate_noise)
    apply!(piecemaker_slot, op1)
    apply!(client_slot, op2)
    res = project_traceout!(client_slot, Z)
    @yield timeout(sim, t_meas)
    tag!(net[1 + client_slot.idx][1], Tag(:updateX, res))
    unlock(piecemaker_slot)
    unlock(client_slot)
    unlock(net[1 + client_slot.idx][1])
    @debug "Fused client $(client_slot.idx) with first client $(piecemaker_slot.idx)"
end

@resumable function listen_fuse(sim, net, gate_noise::Float64)
    active_clients = Int[]
    accesstime = 0
    while true
        @yield onchange_tag(net[1])
        while true # there could be multiple new clients to fuse
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)

            if !isnothing(counterpart)
                slot, _, _ = counterpart

                if isempty(active_clients)
                    push!(active_clients, slot.idx)
                    accesstime = net.registers[1].accesstimes[slot.idx]
                else
                    @yield @process fusion(sim, net, net[1][active_clients[1]], net[1][slot.idx], gate_noise)
                    push!(active_clients, slot.idx)
                end

                if length(active_clients) == n
                    @debug "Projecting out after $(length(active_clients)) clients have fused"
                    @yield @process projectout(sim, net, active_clients[1], gate_noise)
                    empty!(active_clients)
                    continue
                end

                if now(sim) - accesstime > t_cutoff
                    @debug "Tracing out existing fused clients after cutoff time reached"
                    @yield lock(net[1][active_clients[1]])
                    traceout!(net[1][active_clients[1]])
                    unlock(net[1][active_clients[1]])
                    for clientidx in active_clients
                        @yield lock(net[1 + clientidx][1])
                        traceout!(net[1 + clientidx][1])
                        unlock(net[1 + clientidx][1])
                    end
                    empty!(active_clients)
                end

            else
                break
            end
        end
    end
end

@resumable function listen_log(sim, net, statelogs)
    while true
        @yield onchange_tag(net[1])

        isdonemessage = querydelete!(net[1], :Zdone)


        if !isnothing(isdonemessage)
            @debug "received Zdone tag: $(isdonemessage)"
            
            # Measure fidelity
            @yield reduce(&, [lock(net[1+i][1]) for i in 1:n])
            fidelity = real(observable([net[1+i][1] for i in 1:n], obs_proj))

            for i in 1:n
                traceout!(net[1 + i][1])
                unlock(net[1 + i][1])
            end
            
            timesteps = now(sim)
            push!(statelogs, (timesteps, fidelity))
            @debug "Updated logs: $(statelogs)"
        end
    end
end

@resumable function correct_and_inform(sim, net::RegisterNet, client::Int, gate_noise)
    while true
        @yield onchange_tag(net[1+client][1])
        msg1 = querydelete!(net[1+client][1], :updateX, ❓)
        msg2 = querydelete!(net[1+client][1], :updateZ, ❓)
        
        if !isnothing(msg1) || !isnothing(msg2)
            if !isnothing(msg1)
                value = msg1[3][2]
                @debug "X received at client $(client), with value $(value)"
                @yield lock(net[1+client][1])
                if value == 2
                    apply!(net[1+client][1], X)
                    op = gate_noise_model(gate_noise)
                    apply!(net[1+client][1], op)
                end
                tag!(net[1][1], Tag(:Xdone, client))
                unlock(net[1+client][1])
            end
            
            if !isnothing(msg2)
                @debug "Z received at client $(client)"
                value = msg2[3][2]
                @debug "Z received at client $(client), with value $(value))"
                @yield lock(net[1+client][1])
                if value == 2
                    apply!(net[1+client][1], Z)
                    op = gate_noise_model(gate_noise)
                    apply!(net[1+client][1], op)
                end
                tag!(net[1][1], Tag(:Zdone))
                unlock(net[1+client][1])
            end
        end
    end
end

function prepare_sim(n::Int, T_coherence::Float64, gate_noise::Float64, F_link::Float64, link_success_prob::Float64)

    statelogs = Tuple{Float64, Float64}[]

    states_representation = QuantumOpticsRepr()

    noise_model = Depolarization(T_coherence)

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
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = 1e-6,
            retry_lock_time = 1e-7, local_busy_time_post = 0.0
        )

        @process entangler()
        @process correct_and_inform(sim, net, i, gate_noise)
    end

    @process listen_fuse(sim, net, gate_noise)
    @process listen_log(sim, net, statelogs)

    return sim, statelogs
end
##

n = 4
const obs_proj = SProjector(StabilizerState(ghz(n)))

t_CNOT = 100e-6#1e-3 
t_XZ = 10e-6#1e-3 
t_meas = 10e-6#1e-5 # 10 microseconds
t_cutoff = 4e-3 # 4 milliseconds

T_coherence = 10.0 # 10 seconds

dataframes = DataFrame[]
for link_success_prob in [1e-1]#[2e-4, 1e-3, 1e-2, 1e-1]
    for gate_noise in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        for F_link in [1.0]
            runtime = 0.1
            sim, statelogs = prepare_sim(n, T_coherence, gate_noise, F_link, link_success_prob)
            t_wallclock = @elapsed run(sim, runtime)
            
            logs = DataFrame(statelogs, [:timesteps, :GHZfidel])
            logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)

            logs[!, "link_success_prob"] .= link_success_prob
            logs[!, "runtime"] .= runtime
            logs[!, "t_CNOT"] .= t_CNOT
            logs[!, "t_meas"] .= t_meas
            logs[!, "t_XZ"] .= t_XZ
            logs[!, "gate_noise"] .= gate_noise
            # logs[!, "memory_depolar_prob"] .= memory_depolar_prob
            # logs[!, "T_coherence"] .= 1 / (-log(1 - memory_depolar_prob))
            logs[!, "T_coherence"] .= T_coherence
            logs[!, "F_link"] .= F_link
            logs[!, "wallclock_time"] .= t_wallclock

            #save intermediate results
            CSV.write("GHZservice_n4_results_$(link_success_prob)_$(gate_noise)_$(F_link).csv", logs)
            push!(dataframes, logs)
            @info "completed simulation for link_success_prob=$(link_success_prob), T_CNOT=$(t_CNOT), gate_noise=$(gate_noise), F_link=$(F_link), wallclock=$(t_wallclock)s, collected $(nrow(logs)) logs"
        end
    end
end

alllogs = vcat(dataframes...)
@debug alllogs


##
using DataFrames, Statistics
using StatsPlots   # loads Plots + grouping niceties
using LaTeXStrings

#read data from csv files
alllogs = DataFrame[]
for link_success_prob in [1e-1]
    for gate_noise in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        for F_link in [0.999, 0.99, 0.98, 1.0]
            df = CSV.read("output/GHZservice_n4_results_$(link_success_prob)_$(gate_noise)_$(F_link).csv", DataFrame)
            push!(alllogs, df)
        end
    end
end
alllogs = vcat(alllogs...)

# Aggregate
gdf = combine(groupby(alllogs, [:gate_noise, :F_link]),
    :GHZfidel   => mean => :avg_GHZfidel,
    :time_diff  => (t -> 1/mean(t)) => :avg_gen_rate,   # mean of (1/time_diff)
)

gdf[!, :infidelity] = 1 .- gdf.avg_GHZfidel

# Plot
default(titlefontsize=16, guidefontsize=16, tickfontsize=12, legendfontsize=12, legendtitlefontsize=14)
p1 = @df gdf plot(
    :gate_noise, :infidelity,
    group = :F_link,
    xlabel = L"Physical error $p$",
    xscale = :log10,

    ylabel = L"1 - F",
    yscale = :log10,
    marker = :circle,
    lw = 2,
    legend = :best,
    legendtitle = L"F_{link}",
)

plot(p1)
savefig("GHZservice_n4_results.png")