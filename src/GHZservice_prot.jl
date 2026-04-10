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
    pending_measure::Dict{Int, Vector{Int}}
    next_measure_id::Int
    logs::Vector{Tuple{Float64, Vector{Int}, Float64, Bool}}
end

function noisy_bell_state(target_fidelity::Float64=0.97)
    λ = (4*target_fidelity - 1)/3
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return λ*perfect_pair_dm + (1-λ)*mixed_dm
end

@resumable function projectout(sim, net, slot_idx, gen_set_idx, n_clients_in_set, state::SimulationState)
    # Make a copy of the clients in the generator set before clearing it, to avoid mutation issues
    clients = copy(state.progress[gen_set_idx])

    # generate unique id
    state.next_measure_id += 1
    meas_id = state.next_measure_id

    # store it keyed
    state.pending_measure[meas_id] = clients

    # clear generator set progress
    state.progress[gen_set_idx] = Int[]

    @yield lock(net[1][slot_idx])
    @debug "Projecting out piecemaker qubit at slot $(slot_idx)"
    res = project_traceout!(net[1][slot_idx], σˣ)
    @yield timeout(sim, Δt_meas)

    tag!(net[1+slot_idx][1], Tag(:updateZ, res, gen_set_idx, n_clients_in_set, meas_id))

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
    @debug "Fused client $(client_slot.idx) with first client $(piecemaker_slot.idx)"
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
    
    @debug "accesstimes: $(accesstimes), sorted: $(sorted_indices)"

    # Check if all potential generators are empty
    if all(isempty(state.progress[i]) for i in potential_gens_indcs)
        @debug "Progress $(state.progress) is empty, cannot find oldest generator for candidate $(candidate), return index $(potential_gens_indcs[1])"
        return now(sim), potential_gens_indcs[1]
    end

    # Optimized search: iterate through sorted clients, check if they're in any valid generator
    for oldest_idx in sorted_indices
        for gen_idx in potential_gens_indcs
            prog = state.progress[gen_idx]
            if !isempty(prog) && oldest_idx ∈ prog
                @debug "Found generator index $(gen_idx) for candidate $(candidate) with oldest client index $(oldest_idx)"
                timestamp = accesstimes[oldest_idx]
                return timestamp, gen_idx
            end
        end
    end
    
    error("No generator found for candidate $(candidate), sorted_indices: $(sorted_indices)")
end

@resumable function GeneratorServiceProt(sim, net, candidate::Int, state::SimulationState, steane_generators, idx_generator_take::Int)

    s = state.progress[idx_generator_take]
    if candidate ∉ s
        push!(s, candidate)
        if length(s) > 1
            @process fusion(sim, net, net[1][s[1]], net[1][candidate])
            @debug "fusing candidate $(candidate) into generator set index $(idx_generator_take) with current starting index $(s[1])"
        end
        if length(s) == length(steane_generators[idx_generator_take])
            @debug "Progress: $(state.progress). Generator set $(idx_generator_take) completed with clients $(s), projecting out piecemaker qubit"
            @process projectout(sim, net, s[1], idx_generator_take, length(s), state)
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
                candidate = slot.idx
                (timestamp, idx_generator_take) = get_oldest_generator_for_candidate(sim, net, candidate, state)
                while now(sim) - timestamp > cutoff
                    @debug "Generator set $(idx_generator_take) with clients $(state.progress[idx_generator_take]) TOO OLD (timestamp: $(timestamp), now: $(now(sim)), cutoff: $(cutoff))"
                    @yield @process projectout(sim, net, state.progress[idx_generator_take][1], idx_generator_take, length(state.progress[idx_generator_take]), state)
                    (timestamp, idx_generator_take) = get_oldest_generator_for_candidate(sim, net, candidate, state)
                end
                @process GeneratorServiceProt(sim, net, candidate, state, steane_generators, idx_generator_take)
            else
                break
            end
        end
    end
end

@resumable function listen_log(sim, net, state::SimulationState, steane_generators)
    discarded = true
    while true
        @yield onchange_tag(net[1])
        isdonemessage = querydelete!(net[1], :Zdone, ❓, ❓, ❓)

        if !isnothing(isdonemessage)
            genset = steane_generators[isdonemessage[3][2]]
            n_clients_in_set = isdonemessage[3][3]
            meas_id = isdonemessage[3][4]
            @debug "received Zdone tag: $(isdonemessage)"
            
            discarded = true
            fidelity = 0.0
            
            # Measure fidelity
            if length(genset) != n_clients_in_set
                @debug "MEASURE OUT BEFORE COMPLETION $(state.pending_measure[meas_id]), progress: $(state.progress)"
                clients_to_measure = pop!(state.pending_measure, meas_id, nothing)

                if clients_to_measure === nothing
                    @warn "Missing pending_measure for meas_id=$(meas_id). Dropping this log entry."
                    continue
                end

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
                @debug "clients serviced: $(genset) --> fidelity: $(fidelity)"
                for i in genset
                    traceout!(net[1 + i][1])
                    unlock(net[1 + i][1])
                end
                discarded = false
            end
            
            timesteps = now(sim)
            push!(state.logs, (timesteps, (length(genset) != n_clients_in_set ? clients_to_measure : copy(genset)), fidelity, discarded))
            @debug "Updated progress: $(state.progress)"
        end
    end
end

@resumable function correct_and_inform(sim, net::RegisterNet, client::Int)
    while true
        @yield onchange_tag(net[1+client][1])
        msg1 = querydelete!(net[1+client][1], :updateX, ❓)
        msg2 = querydelete!(net[1+client][1], :updateZ, ❓, ❓, ❓, ❓)
        
        if !isnothing(msg1) || !isnothing(msg2)
            if !isnothing(msg1)
                value = msg1[3][2]
                @debug "X received at client $(client), with value $(value)"
                @yield lock(net[1+client][1])
                if value == 2
                    apply!(net[1+client][1], X, time = now(sim))
                    @yield timeout(sim, Δt_XZ)
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Xdone, client))
            end
            
            if !isnothing(msg2)
                @debug "Z received at client $(client)"
                value = msg2[3][2]
                gen_set_idx = msg2[3][3]
                n_clients_in_set = msg2[3][4]
                meas_id = msg2[3][5]

                @debug "Z received at client $(client), with value $(value), gen_set_idx=$(gen_set_idx), n_clients_in_set=$(n_clients_in_set)"
                @yield lock(net[1+client][1])
                if value == 2
                    apply!(net[1+client][1], Z, time = now(sim))
                    @yield timeout(sim, Δt_XZ)
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Zdone, gen_set_idx, n_clients_in_set, meas_id))
            end
        end
    end
end

function prepare_sim(n, T_link, cutoff::Float64, F_link::Float64, link_success_prob::Float64, steane_generators, attempt_t)
    states_representation = QuantumOpticsRepr()
    noise_model = Depolarization(T_link)

    # Initialize simulation state
    state = SimulationState(
        [Int[] for _ in 1:length(steane_generators)],
        Dict{Int, Vector{Int}}(),
        -1,
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
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = attempt_t, # 8-10 ns
            retry_lock_time = 5e-9, local_busy_time_post = 0.0
        )

        @process entangler()
        @process correct_and_inform(sim, net, i)
    end

    @process listen_fuse(sim, net, state, steane_generators, cutoff)
    @process listen_log(sim, net, state, steane_generators)

    return sim, state
end
##
n = 7

Δt_CNOT = 100e-6 # 100 µs
Δt_XZ = 10e-6 # 10 µs 
Δt_meas = 100e-9 # 100 ns
Δt_cutoff_list = [Inf] # 1 ms cutoff

dataframes = DataFrame[]
for attempt_time in [1e-6] # 10 ns, 100 ns, 1 µs #1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    for link_success_prob in [0.01]#, 1e-2, 1e-1]
        for T_coherence in [Inf]#[10e-3]#, 100e-3, 1.0] # 1 ms, 100 ms, 1 s
            for F_link in [1.0]#, 0.999, 1.0]
                for cutoff in Δt_cutoff_list
                    runtime = 0.005
                    sim, state = prepare_sim(n, T_coherence, cutoff, F_link, link_success_prob, steane_generators, attempt_time)
                    t_wallclock = @elapsed run(sim, runtime)
                    
                    logs = DataFrame(state.logs, [:timesteps, :clients_serviced, :GHZfidel, :discarded])
                    logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)
                    logs = transform(logs, :clients_serviced => (x -> [length(c) for c in x]) => :num_clients)

                    logs[!, "link_success_prob"] .= link_success_prob
                    logs[!, "attempt_time"] .= attempt_time
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
end
alllogs = vcat(dataframes...;)