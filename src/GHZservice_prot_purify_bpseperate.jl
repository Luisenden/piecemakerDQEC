using QuantumSavory
using QuantumSavory: Register, X, Z, CNOT
using QuantumSavory.ProtocolZoo
using QuantumSavory.CircuitZoo: Purify2to1Node
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames
using Statistics
using Plots
using JLD2
using DataFrames, StatsPlots, Statistics

const ghzs = [ghz(n) for n in 1:10] # make const in order to not build new every time

# Main simulation, measures everything in seconds (s)
const steane_generators = [
    [1, 2, 3, 5],
    [1, 2, 4, 6],
    [2, 3, 4, 7],
]

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

# Costum tag to send measurement information to logger
# struct OpenMeasurement
#     gen_set_idx::Int
#     clients::Vector{Int}
#     discarded::Bool
#     num_purifications::Int
#     orphan_purif_pairs::Int
#     meas_id::Int
# end
# Base.show(io::IO, x::OpenMeasurement) =
#     print(io, "OpenMeasurement(gen_set_idx=$(x.gen_set_idx), meas=$(x.meas_id), clients=$(x.clients), discarded=$(x.discarded), num_purifications=$(x.num_purifications))")

# Costum tag to track association of clients to generator sets
# struct HomeGeneratorSet
#     gen_set_idx::Int
#     creation_time::Float64
# end

# Snapshot of a state that is about to be measured/logged
struct PendingMeasurement
    clients::Vector{Int}
    num_purifications::Int
end

# Encapsulate simulation state
mutable struct SimulationState
    progress::Vector{Vector{Int}}
    purif_counts::Vector{Int}
    orphan_purif_pairs::Int
    pending_measure::Dict{Int, PendingMeasurement}
    next_measure_id::Int
    logs::Vector{Tuple{Float64, Vector{Int}, Float64, Bool, Int, Int}}
end

function noisy_bell_state(target_fidelity::Float64=0.97)
    λ = (4 * target_fidelity - 1) / 3
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return λ * perfect_pair_dm + (1 - λ) * mixed_dm
end

@resumable function projectout(sim, net, slot_idx, gen_set_idx, state::SimulationState)
    # Snapshot clients and counters before clearing progress
    clients = copy(state.progress[gen_set_idx])
    num_purifications = state.purif_counts[gen_set_idx]

    # Generate unique id
    state.next_measure_id += 1
    meas_id = state.next_measure_id

    # Store pending measurement info keyed by meas_id
    state.pending_measure[meas_id] = PendingMeasurement(
        clients,
        num_purifications
    )

    # Clear generator set progress and reset counters
    state.progress[gen_set_idx] = Int[]
    state.purif_counts[gen_set_idx] = 0

    @yield lock(net[1][slot_idx])
    @debug "Projecting out piecemaker qubit at slot $(slot_idx)"
    res = project_traceout!(net[1][slot_idx], σˣ)
    res == 2 && apply!(net[1+slot_idx][1], Z)
    #@yield timeout(sim, Δt_meas)
    unlock(net[1][slot_idx])
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, client_slot::RegRef)
    @yield lock(piecemaker_slot) & lock(client_slot)
    apply!((piecemaker_slot, client_slot), CNOT)
    @yield timeout(sim, Δt_CNOT)
    res = project_traceout!(client_slot, Z)
    #@yield timeout(sim, Δt_meas)

    res == 2 && apply!(net[1+client_slot.idx][1], X) # TODO: correction gate is now faster than light, add timeout!
    unlock(piecemaker_slot)
    unlock(client_slot)
    @debug "Fused client $(client_slot.idx) with first client $(piecemaker_slot.idx)"
end

function get_oldest_generator_for_candidate(sim, net::RegisterNet, candidate::Int, state::SimulationState)
    potential_gens_indcs = CLIENT_TO_GENERATORS[candidate]

    n_clients = length(net.registers) - 1
    accesstimes = Vector{Float64}(undef, n_clients)
    for i in 1:n_clients
        accesstimes[i] = net.registers[i + 1].accesstimes[1]
    end

    accesstimes_replaced = replace(accesstimes, 0.0 => Inf)
    sorted_indices = sortperm(accesstimes_replaced)

    @debug "accesstimes: $(accesstimes), sorted: $(sorted_indices)"

    if all(isempty(state.progress[i]) for i in potential_gens_indcs)
        @debug "Progress $(state.progress) is empty, cannot find oldest generator for candidate $(candidate), return index $(potential_gens_indcs[1])"
        return now(sim), potential_gens_indcs[1]
    end

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
    target_len = length(steane_generators[idx_generator_take])

    if candidate ∉ s
        push!(s, candidate)
        if length(s) > 1
            @yield @process fusion(sim, net, net[1][s[1]], net[1][candidate])
            @yield @process entangler_for_purification(sim, net, candidate, attempt_time, link_success_prob)
            @debug "fusing candidate $(candidate) into generator set index $(idx_generator_take) with current starting index $(s[1])"
        end
        if length(s) == target_len
            # Wait for all X corrections corresponding to fused non-pivot clients
            # The first client in the snapshot is the pivot / piecemaker.
            @debug "Generator set index $(idx_generator_take) with clients $(s) is ready for measurement, waiting for Xdone tags from clients before projecting out"

            # Delete remaining counterpart tags for the clients being measured, as this indicates the end of the states lifetime
            for i in s
                counterpart = querydelete!(net[1 + i][1], EntanglementCounterpart, ❓, ❓; filo = false)
                if !isnothing(counterpart)
                    @debug "For client $(i), found and DELETED TAG counterpart tag: $(counterpart)"
                else
                    @warn "For client $(i), did NOT find expected counterpart tag."
                end
            end
            @debug "Progress: $(state). Generator set $(idx_generator_take) completed with clients $(s), projecting out piecemaker qubit"
            @yield @process projectout(sim, net, s[1], idx_generator_take, state)

            tag!(net[1][1], Tag(:Zdone, idx_generator_take, length(s), state.next_measure_id))
        end
    else
        error("Candidate $(candidate) already in progress for generator set index $(idx_generator_take), current clients in set: $(s)")
    end
    return
end

@resumable function PurifyProt(sim, net, candidate::Int, state::SimulationState)
    gen_idx = findfirst(i -> candidate ∈ state.progress[i], 1:length(state.progress))

    if isnothing(gen_idx) || !isassigned(net[1 + candidate][1]) # this can happen if the purification request arrives after the generator set has been completed and projected out, in which case we just count it as an orphan purification pair
        @yield lock(net[1][candidate]) & lock(net[1 + candidate][2])
        project_traceout!(net[1][candidate], σˣ)
        project_traceout!(net[1 + candidate][2], σˣ)
        unlock(net[1][candidate])
        unlock(net[1 + candidate][2])
        return
    end

    s = state.progress[gen_idx]

    if candidate == s[1]
        state.orphan_purif_pairs += 1
        @warn "Candidate $(candidate) is the first client in its generator set, cannot purify, skipped"
        @yield lock(net[1][candidate]) & lock(net[1 + candidate][2])
        project_traceout!(net[1][candidate], σˣ)
        project_traceout!(net[1 + candidate][2], σˣ)
        unlock(net[1][candidate])
        unlock(net[1 + candidate][2])
        return
    end

    @yield lock(net[1][candidate]) & lock(net[1][s[1]])
    res1 = Purify2to1Node(:Z)(net[1][s[1]], net[1][candidate])
    unlock(net[1][candidate])
    unlock(net[1][s[1]])

    @yield lock(net[1 + candidate][1]) & lock(net[1 + candidate][2])
    res2 = Purify2to1Node(:Z)(net[1 + candidate][1], net[1 + candidate][2])
    unlock(net[1 + candidate][1])
    unlock(net[1 + candidate][2])

    if res1 == 2 && res1 == res2 # we only accept states that correspond to the (+,+) eigenspaces of the ZZ stabilizers
        state.purif_counts[gen_idx] += 1
        @info "Purification successful for candidate $(candidate); count now $(state.purif_counts[gen_idx])"
    else
        @info "Purification failed for candidate $(candidate), projecting out with clients $(s)"
        @yield @process projectout(sim, net, state.progress[gen_idx][1], gen_idx, state)
    end
end

@resumable function listen_fuse(sim, net, state::SimulationState, steane_generators, cutoff::Float64)
    while true
        @yield onchange_tag(net[1])

        while true
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓; filo = false)
            @debug "Current progress: $(state.progress)"
            if !isnothing(counterpart)
                @debug "Received EntanglementCounterpart tag: $(counterpart)"
                @debug "current state: $(state)"
                slot, _, taglog = counterpart
                candidate = slot.idx
                @debug slot.idx
                if taglog[3] == 1 # from tag "Entangled to client X.1"
                    (timestamp, idx_generator_take) = get_oldest_generator_for_candidate(sim, net, candidate, state)
                    while now(sim) - timestamp > cutoff
                        @debug "Generator set $(idx_generator_take) with clients $(state.progress[idx_generator_take]) TOO OLD (timestamp: $(timestamp), now: $(now(sim)), cutoff: $(cutoff))"
                        @yield @process projectout(sim, net, state.progress[idx_generator_take][1], idx_generator_take, state)
                        (timestamp, idx_generator_take) = get_oldest_generator_for_candidate(sim, net, candidate, state)
                    end
                    @yield @process GeneratorServiceProt(sim, net, candidate, state, steane_generators, idx_generator_take)
                elseif taglog[3] == 2 # from tag "Entangled to client X.2"
                    @info "Received purification request for candidate $(candidate)"
                    @yield @process PurifyProt(sim, net, candidate, state)
                end
            else
                break
            end
        end
    end
end


@resumable function listen_log(sim, net, state::SimulationState, steane_generators)
    while true
        @yield onchange_tag(net[1])
        isdonemessage = querydelete!(net[1], :Zdone, ❓, ❓, ❓; filo = false)

        if !isnothing(isdonemessage)
            gen_set_idx = isdonemessage[3][2]
            genset = steane_generators[gen_set_idx]
            n_clients_in_set = isdonemessage[3][3]
            meas_id = isdonemessage[3][4]
            @debug "received Zdone tag: $(isdonemessage)"

            pending = pop!(state.pending_measure, meas_id, nothing)
            if pending === nothing
                @warn "Missing pending_measure for meas_id=$(meas_id). Dropping this log entry."
                continue
            end

            clients_to_measure = pending.clients
            num_purifications = pending.num_purifications
            discarded = (length(genset) != n_clients_in_set)
            @debug "Logging measurement $(pending) for generator set index $(gen_set_idx) with clients $(clients_to_measure)"

            obs_proj = SProjector(StabilizerState(ghzs[4]))
            @yield reduce(&, [lock(net[1 + i][1]) for i in clients_to_measure])
            fidelity = real(observable([net[1 + i][1] for i in clients_to_measure], obs_proj))

            if discarded
                @debug "MEASURE OUT BEFORE COMPLETION $(clients_to_measure), progress: $(state.progress)"
            else
                @debug "clients serviced: $(clients_to_measure) --> fidelity: $(fidelity)"
            end

            for i in clients_to_measure
                traceout!(net[1 + i][1])
                unlock(net[1 + i][1])
            end

            timesteps = now(sim)
            push!(state.logs, (
                timesteps,
                copy(clients_to_measure),
                fidelity,
                discarded,
                num_purifications,
                state.orphan_purif_pairs,
            ))

            state.orphan_purif_pairs = 0

            @debug "Updated progress: $(state.progress)"
        end
    end
end

@resumable function entangler_for_purification(sim, net, client, attempt_t, link_success_prob)
    entangler = EntanglerProt(
        sim=sim, net=net, nodeA=1, chooseA=client,
        nodeB=1 + client, chooseB=2,
        randomize=false,
        success_prob = link_success_prob, rounds = 1, attempts = -1, attempt_time = attempt_t,
        retry_lock_time = attempt_t/2, local_busy_time_post = 0.0
    )
    @yield @process entangler()
end

function prepare_sim(n::Int, T_link::Float64, cutoff::Float64, F_link::Float64, link_success_prob::Float64, steane_generators, attempt_t, purify::Bool)
    states_representation = QuantumOpticsRepr()
    @debug "Preparing simulation with parameters: n=$(n), T_link=$(T_link), cutoff=$(cutoff), F_link=$(F_link), link_success_prob=$(link_success_prob), attempt_time=$(attempt_t), purify=$(purify)"
    noise_model = Depolarization(T_link)


    # Initialize simulation state
    state = SimulationState(
        [Int[] for _ in 1:length(steane_generators)],
        zeros(Int, length(steane_generators)),
        0, # orphan_purif_pairs
        Dict{Int, PendingMeasurement}(),
        -1,
        Tuple{Float64, Vector{Int}, Float64, Bool, Int, Int, Int}[]
    )

    # Network setup
    switch = Register(
        [Qubit() for _ in 1:n],
        [states_representation for _ in 1:n],
        [noise_model for _ in 1:n]
    )
    clients = [
        Register(
            [Qubit(), Qubit()],
            [states_representation, states_representation],
            [noise_model, nothing]
        ) for _ in 1:n
    ]

    graph = star_graph(n + 1)
    net = RegisterNet(graph, [switch, clients...])

    sim = get_time_tracker(net)

    for i in 1:n
        # Entangler for generation of bell pairs for GHZ state generation
        entangler = EntanglerProt(
            sim=sim, net=net, nodeA=1, chooseA=i,
            nodeB=1+i, chooseB=1,
            randomize=false,
            pairstate=noisy_bell_state(F_link),
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = attempt_t,
            retry_lock_time = attempt_t/2, local_busy_time_post = 0.0
        )

        @process entangler()
    end
    @process listen_fuse(sim, net, state, steane_generators, cutoff)
    @process listen_log(sim, net, state, steane_generators)

    return sim, state
end

##
n = 7

Δt_CNOT = 100e-6  # 100 µs

attempt_time = 1.0e-6 # 1 µs
link_success_prob = 0.0001
cutoff = Inf

runtime = 1.0
purify = true

dataframes = DataFrame[]

for F_link in [1.0-2.5^(-x) for x in  2:6][1]
    for T_coherence in [0.1, 0.01][1]
        sim, state = prepare_sim(n, T_coherence, cutoff, F_link, link_success_prob, steane_generators, attempt_time, purify)
        t_wallclock = @elapsed run(sim, runtime)

        log = DataFrame(
            state.logs,
            [
                :timesteps,
                :clients_serviced,
                :GHZfidel,
                :discarded,
                :num_purifications,
                :orphan_purif_pairs
            ]
        )

        log[!, "attempt_time"] .= attempt_time
        log[!, "link_success_prob"] .= link_success_prob
        log[!, "cutoff"] .= cutoff
        log[!, "runtime"] .= runtime
        log[!, "purify"] .= purify
        log[!, "wallclock_time"] .= t_wallclock
        log[!, "T_coherence"] .= T_coherence
        log[!, "F_link"] .= F_link

        push!(dataframes, log)
        @info "completed simulation for F_link=$(F_link), T_coherence=$(T_coherence)"
    end
end
logsv21true = vcat(dataframes...)
##
@save "GHZservice_purification_compare_logs_true_$(attempt_time)_$(link_success_prob)_$(cutoff)_runtime$(runtime).jld2" logsv21true
##

Δt_XZ = 10e-6     # 10 µs
Δt_meas = 100e-9  # 100 ns
dataframes = DataFrame[]
for purify in [false]
    for attempt_time in [10^(-x) for x in 5.0:8.0] # 10ns to 10μs
        for link_success_prob in [10^(-x) for x in 1.0:5.0] # 0.1 to 0.00001
            for T_coherence in [1/10^x for x in -1.0:3.0] # 0.1s to 0.00001s
                for F_link in [1.0-0.01^(x) for x in 1.0:6.0] #[1.0- 2.5^(-x) for x in 1.0:6.0]
                    runtime = attempt_time/link_success_prob * 1000
                    for cutoff in [Inf] # cutoff at 10%, 20% and 50% of expected runtime

                        sim, state = prepare_sim(n, T_coherence, cutoff, F_link, link_success_prob, steane_generators, attempt_time, purify)
                        t_wallclock = @elapsed run(sim, runtime)

                        logs = DataFrame(
                            state.logs,
                            [
                                :timesteps,
                                :clients_serviced,
                                :GHZfidel,
                                :discarded,
                                :num_purifications,
                                :orphan_purif_pairs
                            ]
                        )
                        logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)
                        logs = transform(logs, :clients_serviced => (x -> [length(c) for c in x]) => :num_clients)

                        logs[!, "attempt_time"] .= attempt_time
                        logs[!, "link_success_prob"] .= link_success_prob
                        logs[!, "T_coherence"] .= T_coherence
                        logs[!, "F_link"] .= F_link
                        logs[!, "runtime"] .= runtime
                        logs[!, "tCNOT"] .= Δt_CNOT
                        logs[!, "cutoff"] .= cutoff
                        logs[!, "wallclock_time"] .= t_wallclock
                        logs[!, "purify"] .= purify
                        
                        @save "GHZservice_purification_trial_logs_purify$(purify)_attempttime$(attempt_time)_linksuccessprob$(link_success_prob)_Tcoherence$(T_coherence)_Flink$(F_link)_runtime$(runtime).jld2" logs
                        push!(dataframes, logs)
                        @info "completed simulation for link_success_prob=$(link_success_prob), T_CNOT=$(Δt_CNOT), T_coherence=$(T_coherence), F_link=$(F_link), cutoff=$(cutoff), wallclock=$(t_wallclock)s, collected $(nrow(logs)) logs"
                    end
                end
            end
        end
    end
end

alllogs = vcat(dataframes...)

##
summ = combine(groupby(alllogs, [:F_link, :purify, :T_coherence]),
    :GHZfidel => mean => :μ,
    :GHZfidel => std  => :σ,
    nrow => :nlogs
)
summ[!, :se] = summ.σ ./ sqrt.(summ.nlogs)
