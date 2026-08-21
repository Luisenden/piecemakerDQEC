using QuantumSavory
using QuantumSavory: Register, X, Z, Y, CNOT
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using DataFrames
using Statistics
using JLD2
using DataFrames
using Random
##
seed = 1234

code = length(ARGS) >= 1 ? ARGS[1] : "Steane7"
error_model = length(ARGS) >= 2 ? ARGS[2] : "depolarizing"
target_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3000
global_idx = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
output_path = length(ARGS) >= 5 ? ARGS[5] : "./"

const codes = Dict(

    "Steane7" => (7, [
        [1, 2, 3, 5],
        [1, 2, 4, 6],
        [2, 3, 4, 7],
    ]),

    "GB16_2_4" => (16, [
        [1, 6, 9, 10],
        [2, 7, 10, 11],
        [3, 8, 11, 12],
        [1, 4, 12, 13],
        [2, 5, 13, 14],
        [3, 6, 14, 15],
        [4, 7, 15, 16],

        [1, 8, 9, 12],
        [1, 2, 10, 13],
        [2, 3, 11, 14],
        [3, 4, 12, 15],
        [4, 5, 13, 16],
        [5, 6, 9, 14],
        [6, 7, 10, 15],
    ]),

    "GB26_2_5" => (26, [
        [1, 4, 16, 18],
        [2, 5, 17, 19],
        [3, 6, 18, 20],
        [4, 7, 19, 21],
        [5, 8, 20, 22],
        [6, 9, 21, 23],
        [7, 10, 22, 24],
        [8, 11, 23, 25],
        [9, 12, 24, 26],
        [10, 13, 14, 25],
        [1, 11, 15, 26],
        [2, 12, 14, 16],

        [10, 12, 14, 25],
        [11, 13, 15, 26],
        [1, 12, 16, 26],
        [2, 13, 14, 17],
        [1, 3, 15, 18],
        [2, 4, 16, 19],
        [3, 5, 17, 20],
        [4, 6, 18, 21],
        [5, 7, 19, 22],
        [6, 8, 20, 23],
        [7, 9, 21, 24],
        [8, 10, 22, 25],
    ]),
)

const paulis = (nothing, X, Y, Z)
@inline function sample_depol2q(gate_fidelity::Float64)
    λ = 4/3 * (1-gate_fidelity)
    rand() < (1-λ) && return (nothing, nothing)
    return (rand(paulis), rand(paulis))
end

const ghzs = [ghz(k) for k in 1:4] # make const in order to not build new every time

const CLIENT_TO_GENERATORS = Dict(
    i => findall(g -> i in g, codes[code][2]) for i in 1:codes[code][1]
)

const CLIENT_TO_CLIENTINCOMMON = Dict(
    i => sort(setdiff(unique(vcat((collect(codes[code][2][g]) for g in CLIENT_TO_GENERATORS[i])...)), [i]))
    for i in 1:codes[code][1]
)

const LogRow = Tuple{Float64, Int, Float64}

function noisy_bell_state(target_fidelity::Float64=0.97)
    λ = (4 * target_fidelity - 1) / 3
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return λ * perfect_pair_dm + (1 - λ) * mixed_dm
end

struct SwitchSlotInfo
    switchslot_idx::Int
    clientslot_idx::Int
    time_of_creation::Float64
end

function next_attempt_id!(counter)
    counter[] += 1
    return counter[]
end

@resumable function consumer(sim, net, pm_slot::RegRef, gen_set_index::Int, iscutoff::Bool, log_data::Vector{LogRow}, Δt_readout::Float64, readout_fidelity::Float64)

    msgs = queryall(net[1], :PartOfGenSet, gen_set_index, pm_slot.idx, ❓; filo = false)
    
    !iscutoff && @assert length(msgs) == 4 "Expected to find all slots of complete generator set for piecemaker slot $(pm_slot.idx) but received: $(msgs)"

    pm_msg = query(net[1][pm_slot.idx], :isPiecemaker, ❓; assigned = true, filo = false)
    @assert !isnothing(pm_msg) "Expected piecemaker slot $(pm_slot.idx) to be tagged with isPiecemaker but it is not!"
    
    # check that the four switch slots are exactly the correct generator
    switchslots_idcs = sort([msg.slot.idx for msg in msgs])
    @debug "Generator set indices for generator set $(gen_set_index): $(switchslots_idcs)"

    
    !iscutoff && @assert switchslots_idcs == codes[code][2][gen_set_index] "The generator set is not correct."
    @assert allunique(switchslots_idcs) "All switch slots tagged as part of the generator set need to be unique!"
    @assert allequal([msg.tag[3] for msg in msgs]) "Not all slots tagged as part of the generator set have the same piecemaker slot idx!"

    tagids = [msg.tag[4] for msg in msgs]
    # choose arbitrary client slot idx among the 4 tagged as part of the generator set to apply correction if needed   
    clientslot_idcs_msgs = [client_msg for switchslotmsg in msgs for client_msg in queryall(net[1][switchslotmsg.slot.idx], SwitchSlotInfo, ❓, ❓, ❓; filo=false) if client_msg.id in tagids]
    clientslot_idcs = [client_msg.tag[3] for client_msg in clientslot_idcs_msgs]
    @debug "SWICH SLOT INDICES: $([msg.slot.idx for msg in msgs]), CLIENT SLOT INDICES: $clientslot_idcs"
    
    # zip can truncate: If clientslot_idcs has length 3 or 5 because a SwitchSlotInfo tag is missing or duplicated, zip silently truncates. That could produce a smaller/wrong GHZ fidelity calculation without immediately failing.
    @assert length(clientslot_idcs_msgs) == length(msgs)
    @assert length(clientslot_idcs) == length(msgs)
    clientslots_to_measure = [net[1+switchslot_idx][clientslot_idx] for (switchslot_idx, clientslot_idx) in zip([msg.slot.idx for msg in msgs], clientslot_idcs)]
    @debug "Measuring client qubits at slots $(clientslot_idcs) to consume GHZ state for generator set $(gen_set_index)"

    @yield reduce(&, lock.(clientslots_to_measure))
    @yield lock(pm_slot)
    @debug "Projecting out piecemaker qubit at slot $(pm_slot.idx) to consume GHZ state for generator set $(gen_set_index)"
    res = project_traceout!(pm_slot, σˣ)
    if rand() > readout_fidelity
        res = 3 - res
    end
    res == 2 && apply!(first(clientslots_to_measure), Z)
    !iscutoff && @yield timeout(sim, Δt_readout)
    obs_proj = SProjector(StabilizerState(ghzs[length(clientslots_to_measure)]))
    fidelity = real(observable(clientslots_to_measure, obs_proj))
    time_of_consumption = now(sim)
    
    for clientslot in clientslots_to_measure
        project_traceout!(clientslot, σˣ)
    end
    switchslots_idcs = [msg.slot.idx for msg in msgs]

    # untag
    for msg in msgs
        untag!(net[1], msg.id)
    end
    for msg in clientslot_idcs_msgs
        @debug "Deleting tag $(msg.tag) with id $(msg.id) at slot $(msg.slot)"
        untag!(net[1], msg.id)
        unlock(net[1 + msg.slot.idx][msg.tag[3]])
    end
    untag!(net[1], pm_msg.id)

    unlock(pm_slot)

    # we can do  some logging here if we want to track consumption times etc.
    !iscutoff && push!(log_data, (
        time_of_consumption,
        gen_set_index,
        fidelity,
    ))
end


function drain_entanglement_msgs!(net)
    msgs = Any[]

    while true
        msg = querydelete!(
            net[1],
            EntanglementCounterpart,
            ❓,
            ❓,
            ❓;
            assigned = true,
            filo = false,
        )

        isnothing(msg) && break

        remote_node = msg.tag[2]
        remote_slot = msg.tag[3]
        pair_id = msg.tag[4]

        reciprocal_msg = querydelete!(
            net[remote_node][remote_slot],
            EntanglementCounterpart,
            1,
            msg.slot.idx,
            pair_id;
            filo = false,
        )

        @assert !isnothing(reciprocal_msg) """
        Missing reciprocal EntanglementCounterpart:
        switch slot = $(msg.slot.idx)
        client node = $(remote_node)
        client slot = $(remote_slot)
        pair_id = $(pair_id)
        """

        push!(msgs, msg)
    end

    return msgs
end

@resumable function discard_bell_msgs(sim, net, msgs, pause_until)
    
    @yield timeout(sim, max(0.0, pause_until - now(sim))) # we wait until global pause is over and delete (that prevents the switch from being flooded with messages as the slots are blocked)
    for msg in msgs
        switch_slot = msg.slot
        client_slot = net[1 + msg.slot.idx][msg.tag[3]]
        @yield lock(switch_slot) & lock(client_slot)

        project_traceout!(switch_slot, σˣ)
        project_traceout!(client_slot, σˣ)
        id = tag!(switch_slot, Tag(:discarded))
        unlock(switch_slot)
        unlock(client_slot)
        untag!(net[1], id)
    end
end


@resumable function switch_worker(
    sim,
    net,
    pending_batches::Vector{Vector{Any}}, # batches of messages from bell pairs that arrived at the same time
    worker_running::Ref{Bool},
    attempt_counter::Ref{Int},
    Δt_CNOTgate::Float64,
    gate_fidelity::Float64,
    Δt_readout::Float64,
    readout_fidelity::Float64,
    cutoff::Float64,
    Δt_rotation_shuttle::Float64,
    global_pause::Ref{Float64},
    log_data::Vector{LogRow},
)
    while !isempty(pending_batches)
        msgs = popfirst!(pending_batches)

        # This is the physical switch pause for accepting/rotating/shuttling this batch.
        # The listener remains alive during this timeout and can discard newly arriving Bell pairs.
        global_pause[] = now(sim) + length(msgs) * Δt_rotation_shuttle

        for msg in msgs
            clientslot_idx = msg.tag[3]
            tagid = tag!(msg.slot, Tag(SwitchSlotInfo, msg.slot.idx, clientslot_idx, now(sim))) # mark time of creation
            @yield timeout(sim, Δt_rotation_shuttle) # this is the physical switch pause for accepting/rotating/shuttling this batch
            # get corresponding slot idx at client
            # tag with switch slot info
            @yield @process SwitchSlotProt(sim, net, msg.slot.idx, clientslot_idx, Int(tagid), attempt_counter, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, log_data)
            @yield @process CutoffDiscardProt(sim, net, cutoff, log_data, Δt_readout, readout_fidelity)
        end
    end

    worker_running[] = false
end


@resumable function switch_listener(
    sim,
    net,
    attempt_counter::Ref{Int},
    Δt_CNOTgate::Float64,
    gate_fidelity::Float64,
    Δt_readout::Float64,
    readout_fidelity::Float64,
    cutoff::Float64,
    Δt_rotation_shuttle::Float64,
    global_pause::Ref{Float64},
    log_data::Vector{LogRow},
)
    pending_batches = Vector{Any}[]
    worker_running = Ref(false)

    while true
        @yield onchange(net[1], Tag)

        msgs = drain_entanglement_msgs!(net)
        isempty(msgs) && continue

        batch_time = now(sim)

        if batch_time < global_pause[]
            @process discard_bell_msgs(
                sim,
                net,
                msgs,
                global_pause[],
            )
            continue
        end

        push!(pending_batches, msgs) # qeue up the bell pairs that have arrived

        if !worker_running[]
            worker_running[] = true
            @process switch_worker(
                sim,
                net,
                pending_batches,
                worker_running,
                attempt_counter,
                Δt_CNOTgate,
                gate_fidelity,
                Δt_readout,
                readout_fidelity,
                cutoff,
                Δt_rotation_shuttle,
                global_pause,
                log_data,
            )
        end
    end
end


@resumable function CutoffDiscardProt(sim, net, cutoff::Float64, log_data::Vector{LogRow}, Δt_readout::Float64, readout_fidelity::Float64)
    # this process is triggered by the switch_listener and checks if there are any switch slots that have been stored for longer than the cutoff time.

    msgs = queryall(net[1], :isPiecemaker, ❓; assigned = true, filo = false)
    @debug "CutoffDiscardProt: messages found: $(length(msgs))"
    isempty(msgs) && return # nothing to discard

    for msg in msgs
        time_of_creation = query(msg.slot, SwitchSlotInfo, ❓, ❓, ❓; assigned = true, filo = false).tag[4]
        if now(sim) - time_of_creation > cutoff
            gensetinfo_msg = query(msg.slot, :PartOfGenSet, ❓, ❓, ❓; filo = false)
            pm_slot_idx = gensetinfo_msg.tag[3]
            genset_idx = gensetinfo_msg.tag[2]
            @yield @process consumer(sim, net, net[1][pm_slot_idx], genset_idx, true, log_data, Δt_readout, readout_fidelity) # if the cutoff time has been exceeded, we consume the GHZ state that was being built with the corresponding piecemaker slot to free up the switch slots
            break
        end
    end
end

@resumable function SwitchSlotProt(sim, net, switchslot_idx::Int, clientslot_idx::Int, tagid::Int, attempt_counter::Ref{Int}, Δt_CNOTgate::Float64, gate_fidelity::Float64, Δt_readout::Float64, readout_fidelity::Float64, log_data::Vector{LogRow})
    # this is a sequential protocol, meaning that it occupies the switch slots for its duration until fusion is complete
    # first it checks for all possible GHZ attempts to fuse with (using tag PartOfGenSet)
    # if the answer is (an)other slot(s) it calls fusion for the oldest piecemaker slot
    # if non of the current GHZ attempts are suitable for fusion, it becomes a new piecemaker and starts a new GHZ attempt 
    # and receives tag isPiecemaker; in addition it receives a generator set tag PartOfGenSet unformly at random from the generator sets that contain the current switch slot idx

    @assert isnothing(query(net[1][switchslot_idx], :PartOfGenSet, ❓, ❓, ❓; assigned = true, filo = false)) "This slot is already part of a GHZ attempt, it should not have received a new Bell pair."

    # we query for all potential piecemaker slots among clients that have a generator set in common
    pmslot_msgs = []
    for otherswitchslot_idx in CLIENT_TO_CLIENTINCOMMON[switchslot_idx]
        pmslot_msg = query(net[1][otherswitchslot_idx], :isPiecemaker, ❓; assigned = true, filo = false)
        if !isnothing(pmslot_msg)
            @debug "Found potential piecemaker slot at switch slot idx $(otherswitchslot_idx) with info $(pmslot_msg)"
            gensetinfo_msg = query(pmslot_msg.slot, :PartOfGenSet, ❓, pmslot_msg.slot.idx, ❓; assigned = true, filo = false)
            @debug "checking if current switch slot idx $(switchslot_idx) and potential piecemaker slot idx $(otherswitchslot_idx) are part of the same generator $(CLIENT_TO_GENERATORS[gensetinfo_msg.tag[2]])"
            (switchslot_idx ∈ codes[code][2][gensetinfo_msg.tag[2]]) && push!(pmslot_msgs, pmslot_msg) # only consider it as potential piecemaker slot if the current switch slot is part of the same generator set as the piecemaker slot
        end
    end
    
    if isempty(pmslot_msgs)
        # there is no piecemaker slot in any of the potentially common generator sets, this slot needs therefore to become a piecemaker and start a new GHZ attempt
        @yield lock(net[1][switchslot_idx])
        attempt_id = next_attempt_id!(attempt_counter)
        tag!(net[1][switchslot_idx], Tag(:isPiecemaker, attempt_id))
        gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
        @debug "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
        unlock(net[1][switchslot_idx])
        
    else
        # in this case there is at least one suitable piecemaker slot to fuse with, we choose the oldest one (i.e., the one with the smallest attempt_id)
        min_counter = Inf
        pmslot_to_fuse_with = nothing
        gen_set_idx = nothing
        for pmslot_msg in pmslot_msgs
            msg_pm_info = query(pmslot_msg.slot, :isPiecemaker, ❓; assigned = true, filo = false)
            msg_gensetinfo = query(pmslot_msg.slot, :PartOfGenSet, ❓, pmslot_msg.slot.idx, ❓; assigned = true, filo = false)
            counter = msg_pm_info.tag[2]
            # we take the piecemaker slot with the smallest attempt_id (i.e., the one that has been waiting the longest for fusion)
            if counter < min_counter 
                pmslot_to_fuse_with = pmslot_msg.slot
                gen_set_idx = msg_gensetinfo.tag[2]
                min_counter = counter
            end
        end

        if isnothing(pmslot_to_fuse_with) 
            # in this case all potential piecemaker slots in common used in a GHZ state for an unsuited, meaning that it needs to start a new generator set in which it is itself the piecemaker
            @debug "in this case all potential piecemaker slots are in an unsuitable generator set, meaning that $(switchslot_idx) needs to start a new generator set in which it is itself the piecemaker"
            @yield lock(net[1][switchslot_idx])
            attempt_id = next_attempt_id!(attempt_counter)
            tag!(net[1][switchslot_idx], Tag(:isPiecemaker, attempt_id))
            gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
            @debug "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
            unlock(net[1][switchslot_idx])
        else
            # in this case we found a piecemaker slot within a suitable generator set and fuse with it
            @yield @process fusion(sim, net, pmslot_to_fuse_with, net[1][switchslot_idx], net[1+switchslot_idx][clientslot_idx], gen_set_idx, tagid, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity)

            # check if after fusion the GHZ state is complete with 4 clients (i.e., the current slot was the last to complete the generator set)
            msgs = queryall(net[1], :PartOfGenSet, gen_set_idx, pmslot_to_fuse_with.idx, ❓; filo = false)
            if length(msgs) == 4 
                @yield @process consumer(sim, net, pmslot_to_fuse_with, gen_set_idx, false, log_data, Δt_readout, readout_fidelity) # if the fused state is complete, run the consume listener to consume it;
            end
        end
    end
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, clientswitch_slot::RegRef, client_slot::RegRef, gen_set_idx::Int, tagid::Int, Δt_CNOTgate::Float64, gate_fidelity::Float64, Δt_readout::Float64, readout_fidelity::Float64)
    @yield lock(piecemaker_slot) & lock(clientswitch_slot) & lock(client_slot)
    apply!((piecemaker_slot, clientswitch_slot), CNOT)
    @yield timeout(sim, Δt_CNOTgate)
    noisygate = sample_depol2q(gate_fidelity)
    !isnothing(noisygate[1]) && apply!(piecemaker_slot, noisygate[1])
    !isnothing(noisygate[2]) && apply!(clientswitch_slot, noisygate[2])

    res = project_traceout!(clientswitch_slot, σᶻ)
    if rand() > readout_fidelity
        res = 3 - res
    end
    @yield timeout(sim, Δt_readout)
    res == 2 && apply!(client_slot, X) # TODO: correction gate is now faster than light, add timeout!
    tag!(clientswitch_slot, Tag(:PartOfGenSet, gen_set_idx, piecemaker_slot.idx, tagid))
    unlock(piecemaker_slot)
    unlock(clientswitch_slot)
    unlock(client_slot)
    @debug "Fused client $(clientswitch_slot.idx) with first client $(piecemaker_slot.idx)"
end

@resumable function naive_entangler(
    sim,
    net,
    n,
    F_link,
    link_success_prob,
    attempt_t,
    Δt_rotation_shuttle,
)
    for i in 1:n
        entangler = EntanglerProt(
            sim = sim,
            net = net,
            nodeA = 1,
            nodeB = 1 + i,

            chooseslotA = i,
            chooseslotB = 1,

            pairstate = DepolarizedBellPair(; F = F_link),

            success_prob = link_success_prob,
            rounds = -1,
            attempts = -1,
            attempt_time = attempt_t,
            retry_lock_time = nothing,
        )

        @process entangler()
    end
end

function prepare_sim(n, T_link::Float64, F_link::Float64, link_success_prob::Float64, attempt_t::Float64, Δt_CNOTgate::Float64, gate_fidelity::Float64, Δt_readout::Float64, readout_fidelity::Float64,
    error_model::String, cutoff::Float64, Δt_rotation_shuttle::Float64, log_data::Vector{LogRow})


    states_representation = QuantumOpticsRepr()
    @debug "Preparing simulation with parameters: n=$(n), T_link=$(T_link), cutoff=$(cutoff), F_link=$(F_link), link_success_prob=$(link_success_prob), attempt_time=$(attempt_t)"
    
    noise_model = error_model == "dephasing" ? T2Dephasing(T_link) : Depolarization(T_link)

    # Network setup
    switch = Register(
        [Qubit() for _ in 1:n],
        [states_representation for _ in 1:n],
        [noise_model for _ in 1:n]
    )
    clients = [
        Register(
            [Qubit()],
            [states_representation],
            [noise_model]
        ) for _ in 1:n
    ]

    graph = star_graph(n + 1)
    net = RegisterNet(graph, [switch, clients...])

    sim = get_time_tracker(net)

    attempt_counter = Ref(0)
    global_pause = Ref(0.0)

    @process switch_listener(sim, net, attempt_counter, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, cutoff, Δt_rotation_shuttle, global_pause, log_data)
    @process naive_entangler(sim, net, n, F_link, link_success_prob, attempt_t, Δt_rotation_shuttle)
    return sim
end

## run the simulation

# setup parameters varied 
attempt_times = [0.1e-6, 0.5e-6, 1e-6, 1e-5] 
T_coherences = [0.01, 0.1, 1.0, 2.0, 10.0, 20.0] 
CNOTgate_times = [1e-6, 10e-6, 100e-6, 250e-6] 
CNOTgate_fidelities = [0.999, 0.9995, 0.9997, 0.9999, 0.99999] 
readout_times = [0.1e-3, 1e-3, 2e-3]
readout_fidelities = [0.999, 0.9999, 1.0] 
rotation_shuttle_times = [10e-6, 50e-6, 100e-6] 

# all combinations of parameters
parameter_combinations = [
    (attempt_time, T_coherence, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, Δt_rotation_shuttle)
    for attempt_time in attempt_times
        for T_coherence in T_coherences for gate_fidelity in CNOTgate_fidelities
            for Δt_CNOTgate in CNOTgate_times for Δt_readout in readout_times for readout_fidelity in readout_fidelities for Δt_rotation_shuttle in rotation_shuttle_times        
]
##

function estimate_runtime_for_samples(
    target_samples::Int;
    n::Int,
    attempt_time::Float64,
    link_success_prob::Float64,
    Δt_CNOTgate::Float64,
    Δt_readout::Float64,
    Δt_rotation_shuttle::Float64
)
    # Four Bell links are needed for one 4-client GHZ.
    t_link_per_sample = 4 * ( attempt_time / (n * link_success_prob) + Δt_rotation_shuttle )

    # A 4-client GHZ needs 3 fusions after the first piecemaker Bell pair.
    t_gate_per_sample = 3 * (Δt_CNOTgate + Δt_readout) + Δt_readout

    t_sample = t_link_per_sample + t_gate_per_sample

    return (target_samples * t_sample)
end

function get_cutoff(T_coherence, error_budget)
    return -T_coherence * log(1 - 4/3 * ( 1 - (1-error_budget)^0.25 ))
end

function run_sweep(F_link, link_success_prob)

    attempt_time =  parameter_combinations[global_idx][1] #10e-6 #
    T_coherence =  parameter_combinations[global_idx][2] # 1.0 #
    Δt_CNOTgate =  parameter_combinations[global_idx][3] #100e-6 #
    gate_fidelity =  parameter_combinations[global_idx][4] #0.9997 #
    Δt_readout =  parameter_combinations[global_idx][5] #1e-3 #
    readout_fidelity =  parameter_combinations[global_idx][6] #0.9999 #
    Δt_rotation_shuttle =  parameter_combinations[global_idx][7] #100e-6 #

    @debug "Running sweep with parameters: attempt_time=$(attempt_time), link_success_prob=$(link_success_prob), T_coherence=$(T_coherence), F_link=$(F_link), gate_fidelity=$(gate_fidelity), Δt_readout=$(Δt_readout), readout_fidelity=$(readout_fidelity)"

    log_data = LogRow[]
    dataframes = DataFrame[]

    runtime = estimate_runtime_for_samples(
        target_samples;
        n=codes[code][1],
        attempt_time=attempt_time,
        link_success_prob=link_success_prob,
        Δt_CNOTgate=Δt_CNOTgate,
        Δt_readout=Δt_readout,
        Δt_rotation_shuttle=Δt_rotation_shuttle
    )
    for cutoff in [get_cutoff(T_coherence, 0.05), Inf, get_cutoff(T_coherence, 0.1)] # 3
        Random.seed!(seed)

        empty!(log_data)

        sim = prepare_sim(
            codes[code][1],
            T_coherence,
            F_link,
            link_success_prob,
            attempt_time,
            Δt_CNOTgate,
            gate_fidelity,
            Δt_readout,
            readout_fidelity,
            error_model,
            cutoff,
            Δt_rotation_shuttle,
            log_data
        )
        @debug "Starting simulation with runtime $(runtime) seconds"

        t_wallclock = @elapsed run(sim, runtime)

        logs = DataFrame(
            log_data,
            [
                :timesteps,
                :generator_idx,
                :GHZfidel,
            ],
        )

        logs = combine(groupby(logs, :generator_idx), 
            :GHZfidel => mean => :mean_GHZfidel,
            :GHZfidel => std => :std_GHZfidel,
            :GHZfidel => (x -> std(x)/sqrt(length(x))) => :sem_GHZfidel,
            :timesteps => (x -> length(x)/maximum(x)) => :mean_rate,
            :timesteps => (x -> mean([first(x); diff(x)])) => :mean_generation_time,
            :timesteps => (x -> std([first(x); diff(x)])) => :std_generation_time,
            :timesteps => (x -> std([first(x); diff(x)])/sqrt(length(x))) => :sem_generation_time,
            nrow => :nlogs,
        )
        
        logs[!, :link_success_prob] .= link_success_prob
        logs[!, :attempt_time] .= attempt_time
        logs[!, :T_coherence] .= T_coherence
        logs[!, :F_link] .= F_link
        logs[!, :error_model] .= error_model
        logs[!, :cutoff] .= cutoff
        logs[!, :tRotationShuttle] .= Δt_rotation_shuttle
        logs[!, :tCNOT] .= Δt_CNOTgate
        logs[!, :gate_fidelity] .= gate_fidelity
        logs[!, :tReadout] .= Δt_readout
        logs[!, :readout_fidelity] .= readout_fidelity
        logs[!, :runtime] .= runtime
        logs[!, :wallclock_time] .= t_wallclock
        logs[!, :seed] .= seed

        push!(dataframes, logs)

        @info "cutoff: $cutoff, runtime: $runtime, wallclock_time: $t_wallclock"
    end
    df = vcat(dataframes...)
    return df
end

##

dfs = DataFrame[]
for link_success_prob in [10.0^(-x) for x in 1.0:5.0] 
    for F_link in [1.0 - 2.5^(-x) for x in 3.0:10.0]
        df = run_sweep(F_link, link_success_prob)
        push!(dfs, df)
        @info "Completed sweep for F_link=$(F_link), link_success_prob=$(link_success_prob)"
    end
end
df_out = vcat(dfs...)

@save "$(output_path)/summary_ghz_service_v1_$(code)_$(error_model)_$(global_idx).jld2" df_out #$(global_idx)