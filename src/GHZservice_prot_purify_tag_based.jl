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

const ghzs = [ghz(n) for n in 1:4] # make const in order to not build new every time

const steane_generators = [
    [1, 2, 3, 5],
    [1, 2, 4, 6],
    [2, 3, 4, 7],
]

const CLIENT_TO_GENERATORS = Dict(
    i => findall(g -> i in g, steane_generators)
    for i in 1:7
)

const CLIENT_TO_CLIENTINCOMMON = Dict(
    i => sort(setdiff(unique(vcat((collect(steane_generators[g]) for g in CLIENT_TO_GENERATORS[i])...)), [i]))
    for i in 1:7
)

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

@resumable function purify_fail_listener(sim, net)
    # while true
        # @yield onchange_tag(net[1])
        pm_msg_fail = query(net[1], :purifFail, ❓; filo = false)
        # isnothing(pm_msg_fail) && continue
        pm_msg = query(net[1][pm_msg_fail.slot.idx], :isPiecemaker, ❓; assigned = true, filo = false)
        @info "purify_fail_listener got message: $(pm_msg)"
        msgs = queryall(net[1], :PartOfGenSet, pm_msg_fail.tag[2], pm_msg.slot.idx, ❓; filo = false)
        # a piecemaker slot can only point to one to-be-built generator set. It cannot be a piecemaker for multiple generator sets.
        gen_set_ids = [msg.tag[2] for msg in msgs]
        @assert allequal(gen_set_ids) "Expected piecemaker slot $(pm_msg.slot.idx) to be associated with a single generator set, but found multiple: $(unique(gen_set_ids))."
        gen_set_id = gen_set_ids[1]
        tagids = [msg.tag[4] for msg in msgs]
        clientslot_idcs_msgs = [client_msg for switchslotmsg in msgs for client_msg in queryall(net[1][switchslotmsg.slot.idx], SwitchSlotInfo, ❓, ❓, ❓; filo=false) if client_msg.id in tagids]
        clientslot_idcs = [client_msg.tag[3] for client_msg in clientslot_idcs_msgs]

        @assert length(clientslot_idcs_msgs) == length(msgs)
        @assert length(clientslot_idcs) == length(msgs)
        clientslots_to_measure = [net[1+switchslot_idx][clientslot_idx] for (switchslot_idx, clientslot_idx) in zip([msg.slot.idx for msg in msgs], clientslot_idcs)]
        @info "Measuring client qubits at slots $(clientslot_idcs) to discard failed GHZ state"
        @yield reduce(&, lock.(clientslots_to_measure))
        @yield lock(pm_msg.slot)
        @info "Projecting out piecemaker qubit at slot $(pm_msg.slot.idx) to discard failed GHZ state for generator set $(gen_set_id)"
        res = project_traceout!(pm_msg.slot, σˣ)
        res == 2 && apply!(first(clientslots_to_measure), Z)
        @info "CLIENT SLOTS TO MEASURE: $(clientslots_to_measure)"
        obs_proj = SProjector(StabilizerState(ghzs[length(clientslots_to_measure)]))
        fidelity = real(observable(clientslots_to_measure, obs_proj))
        time_of_consumption = now(sim)
        for clientslot in clientslots_to_measure
            project_traceout!(clientslot, σˣ)
        end

        switch_slot_idcs = [msg.slot.idx for msg in msgs]
        for msg in msgs
            untag!(net[1], msg.id)
        end
        for msg in clientslot_idcs_msgs
            @info "Deleting tag $(msg.tag) with id $(msg.id) at slot $(msg.slot)"
            untag!(net[1], msg.id)
            unlock(net[1 + msg.slot.idx][msg.tag[3]])
        end
        untag!(net[1], pm_msg.id)
        untag!(net[1], pm_msg_fail.id)
        unlock(pm_msg.slot)

        n_purified = 0
        while true
            msg = querydelete!(net[1][pm_msg.slot.idx], :puriSucc; filo = false)
            isnothing(msg) && break
            n_purified += 1
        end

        # we can do  some logging here if we want to track consumption times etc.
        push!(log_data, (
            time_of_consumption,
            switch_slot_idcs,
            fidelity,
            true, # discarded
            n_purified,
        ))
    # end
end

@resumable function consume_listener(sim, net)
    # while true
        #@yield onchange_tag(net[1])
        pm_msg_complete = query(net[1], :isComplete, ❓; filo = false)
        #isnothing(pm_msg_complete) && continue

        gen_set_index = pm_msg_complete.tag[2]
        msgs = queryall(net[1], :PartOfGenSet, gen_set_index, pm_msg_complete.slot.idx, ❓; filo = false) # TODO: here we want only one unique generator set, however, if it happens that one slot functions as a piecemaker twice we could get two gen sets --> ACTUALLY if it is the piecemaker then the second bell pair will never arrive, so this must be unique
        @assert length(msgs) == 4 "Expected to find all slots of complete generator set for piecemaker slot $(pm_msg_complete.slot.idx) but received: $(msgs)"

        pm_msg = query(net[1][pm_msg_complete.slot.idx], :isPiecemaker, ❓; assigned = true, filo = false)
        @assert !isnothing(pm_msg) "Expected piecemaker slot $(pm_msg_complete.slot.idx) to be tagged with isPiecemaker but it is not!"
        
        # check that the four switch slots are exactly the correct generator
        switchslots_idcs = sort([msg.slot.idx for msg in msgs])
        @assert switchslots_idcs == steane_generators[gen_set_index] "The generator set is not correct."
        @assert allunique(switchslots_idcs) "All switch slots tagged as part of the generator set need to be unique!"
        @assert allequal([msg.tag[3] for msg in msgs]) "Not all slots tagged as part of the generator set have the same piecemaker slot idx!"

        tagids = [msg.tag[4] for msg in msgs]
        # choose arbitrary client slot idx among the 4 tagged as part of the generator set to apply correction if needed   
        clientslot_idcs_msgs = [client_msg for switchslotmsg in msgs for client_msg in queryall(net[1][switchslotmsg.slot.idx], SwitchSlotInfo, ❓, ❓, ❓; filo=false) if client_msg.id in tagids]
        clientslot_idcs = [client_msg.tag[3] for client_msg in clientslot_idcs_msgs]
        @info "SWICH SLOT INDICES: $([msg.slot.idx for msg in msgs]), CLIENT SLOT INDICES: $clientslot_idcs"
        

        # zip can truncate: If clientslot_idcs has length 3 or 5 because a SwitchSlotInfo tag is missing or duplicated, zip silently truncates. That could produce a smaller/wrong GHZ fidelity calculation without immediately failing.
        @assert length(clientslot_idcs_msgs) == length(msgs)
        @assert length(clientslot_idcs) == length(msgs)
        clientslots_to_measure = [net[1+switchslot_idx][clientslot_idx] for (switchslot_idx, clientslot_idx) in zip([msg.slot.idx for msg in msgs], clientslot_idcs)]
        @info "Measuring client qubits at slots $(clientslot_idcs) to consume GHZ state for generator set $(gen_set_index)"
        @info "WHICH SWITCH SLOTS ARE CURRENTLY LOCKED? $(islocked.(net[1][1:7]))"
        locked = [islocked(net[node][slot]) for node in 2:8, slot in 1:2]
        @info "WHICH CLIENT SLOTS ARE CURRENTLY LOCKED? $(locked)"
        @yield reduce(&, lock.(clientslots_to_measure))
        @yield lock(pm_msg.slot)
        @info "Projecting out piecemaker qubit at slot $(pm_msg.slot.idx) to consume GHZ state for generator set $(gen_set_index)"
        res = project_traceout!(pm_msg.slot, σˣ)

        res == 2 && apply!(first(clientslots_to_measure), Z)
        @info "CLIENT SLOTS TO MEASURE: $(clientslots_to_measure)"
        obs_proj = SProjector(StabilizerState(ghzs[length(clientslots_to_measure)]))
        fidelity = real(observable(clientslots_to_measure, obs_proj))
        time_of_consumption = now(sim)
        for clientslot in clientslots_to_measure
            project_traceout!(clientslot, σˣ)
        end

        switchslots_idcs = [msg.slot.idx for msg in msgs]
        for msg in msgs
            untag!(net[1], msg.id)
        end
        for msg in clientslot_idcs_msgs
            @info "Deleting tag $(msg.tag) with id $(msg.id) at slot $(msg.slot)"
            untag!(net[1], msg.id)
            unlock(net[1 + msg.slot.idx][msg.tag[3]])
        end
        untag!(net[1], pm_msg.id)
        untag!(net[1], pm_msg_complete.id)

        n_purified = 0
        while true
            msg = querydelete!(net[1][pm_msg.slot.idx], :puriSucc; filo = false)
            isnothing(msg) && break
            n_purified += 1
        end
        unlock(pm_msg.slot)

        # we can do  some logging here if we want to track consumption times etc.
        push!(log_data, (
            time_of_consumption,
            switchslots_idcs,
            fidelity,
            false, # discarded
            n_purified,
        ))
    # end
end

@resumable function switch_listener(sim, net, purify::Float64, attempt_counter::Ref{Int})
    # this process listens at the switch register for incoming bell pairs 
    # i.e., is triggered by a change of tag EntanglementCounterpart from the EntanglerProt
    # it tags a client's switch slot with tag SwitchSlotInfo containing the switchslot_idx, clientslot_idx and time_of_creation
    # finally, it runs the SwitchSlotProt
    while true
        @yield onchange_tag(net[1])
        while true
            msg = querydelete!(net[1], EntanglementCounterpart, ❓, ❓; assigned=true, filo = false)
            if !isnothing(msg)
                @info "switch_listener: $(msg)"
                # get corresponding slot idx at client
                clientslot_idx = msg.tag[3]
                # tag with switch slot info
                tagid = tag!(msg.slot, Tag(SwitchSlotInfo, msg.slot.idx, clientslot_idx, now(sim)))
                @yield @process SwitchSlotProt(sim, net, msg.slot.idx, clientslot_idx, Int(tagid), purify, attempt_counter)
            else
                break
            end
        end
        @debug "STATE INFO" queryall(net[1], :PartOfGenSet, ❓, ❓, ❓; filo = false)
    end
end

@resumable function SwitchSlotProt(sim, net, switchslot_idx::Int, clientslot_idx::Int, tagid::Int, purify::Float64, attempt_counter::Ref{Int})
    # this is a sequential protocol, meaning that it occupies the switch slots for its duration until it has decided for fusion or purfication
    # first it queries all other slots which have a generator set in common for the tag PartOfGenSet 
    # if the answer is 'nothing' it gets tagged with tag isPiecemaker
    # and it gets sorted into a (random if multiple are possible) generator set with the according tag PartOfGenSet
    # if the answer is (an)other slot(s) it tries to sort it into the generator set with the (oldest) piecemaker slot, where it gets fused
    # if the latter is not possible (i.e., this is the case when all potential piecemaker slots are already in an unsuitable generator set), 
    # it gets tagged with tag isPiecemaker and gets sorted into the a random suitable generator set with the according tag PartOfGenSet

    # if this protocol is launched on the second bell pair of the slot, it is already part of a generator set, so we need to make sure it is not fused with a state where it already is represented in.
    # we check if it is already part of a state in progress
    gen_set_idx = nothing
    # TODO: this should queryall ==> a slot can already be part of multiple, but wait no it can't because then we would not be here in this protocol as the second bell pair would never arrive
    part_of_gen_set_msg = query(net[1][switchslot_idx], :PartOfGenSet, ❓, ❓, ❓; assigned = true, filo = false)
    if !isnothing(part_of_gen_set_msg)
        gen_set_idx = part_of_gen_set_msg.tag[2]
    end

    !isnothing(gen_set_idx) && @assert isnothing(query(net[1][switchslot_idx], :isPiecemaker, ❓; assigned = true, filo = false)) "A slot already in a generator set must not be a piecemaker; otherwise the second Bell pair should not exist and this protocol should not run."
    if (!isnothing(gen_set_idx) && # this makes sure that purification is only attempted if the associated slot is already part of a fused state
        isnothing(query(net[1][part_of_gen_set_msg.tag[3]], :isComplete, ❓; filo = false)) && # and that the fused state is not complete yet (i.e., it is currently used by the consume_listener)
        isnothing(query(net[1][part_of_gen_set_msg.tag[3]], :purifFail, ❓; filo = false)) && # and that it has not failed yet (i.e., it is currently used by the purify_fail_listener)
        rand() < purify) # and that the purification is attempted with the given probability

        @info "This bell pair in slot $(switchslot_idx) in client slot $(clientslot_idx) is used for purification"
        # first we need to retrieve the piecemaker slot of the generatot set the client is already part of
        pm_slot_idx = part_of_gen_set_msg.tag[3]
        @yield lock(net[1][switchslot_idx]) & lock(net[1][pm_slot_idx]) & lock(net[1 + switchslot_idx][3-clientslot_idx]) & lock(net[1+switchslot_idx][clientslot_idx]) 
        res1 = Purify2to1Node(:Z)(net[1][pm_slot_idx], net[1][switchslot_idx])
        res2 = Purify2to1Node(:Z)(net[1 + switchslot_idx][3-clientslot_idx], net[1+switchslot_idx][clientslot_idx])
        unlock(net[1 + switchslot_idx][3-clientslot_idx])
        unlock(net[1 + switchslot_idx][clientslot_idx])
        unlock(net[1][switchslot_idx])
        unlock(net[1][pm_slot_idx])
        if res1 == 2 && res1 == res2 # we only accept states that correspond to the (+,+) eigenspaces of the ZZ stabilizers
            @info "Purification successful for slot $(clientslot_idx)"
            tag!(net[1][pm_slot_idx], Tag(:puriSucc))
            untag!(net[1][switchslot_idx], tagid)
            return
        else
            @info "Purification failed for slot $(clientslot_idx), tag slot as failed purification"
            tag!(net[1][pm_slot_idx], Tag(:purifFail, gen_set_idx))
            @yield @process purify_fail_listener(sim, net) # trigger the purify fail listener to discard the state and do the logging
            return
        end
    end

    # we query for all potential piecemaker slots among all potentially common clients that have a gen set in common, but are not part of the generator set that the current slot is already part of
    pmslot_msgs = []
    for otherswitchslot_idx in CLIENT_TO_CLIENTINCOMMON[switchslot_idx]
        pmslot_msg = query(net[1][otherswitchslot_idx], :isPiecemaker, ❓; assigned = true, filo = false)
        if isnothing(gen_set_idx) # if the slot is not part of a gen set yet we can proceed as normal
            push!(pmslot_msgs, pmslot_msg) 
        else # else we filter out the piecemaker slot that is part of the generator set the slot is already represented in
            isnothing(query(net[1][otherswitchslot_idx], :PartOfGenSet, gen_set_idx, ❓, ❓; assigned = true, filo = false)) && push!(pmslot_msgs, pmslot_msg)
        end
    end
    
    if all(isnothing.(pmslot_msgs))
        # in this case there is no piecemaker slot in any of the potentially common generator sets and this slot needs to become the piecemaker and start a GHZ generation
        @yield lock(net[1][switchslot_idx])
        attempt_id = next_attempt_id!(attempt_counter)
        tag!(net[1][switchslot_idx], Tag(:isPiecemaker, attempt_id))
        gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
        @info "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
        unlock(net[1][switchslot_idx])
    else
        # in this case there is at least one piecemaker slot in the potentially common generator sets, we want to assign to the oldest one and fuse with it
        min_counter = Inf
        pmslot_to_fuse_with = nothing
        gen_set_idx = nothing
        for pmslot_msg in pmslot_msgs[.!isnothing.(pmslot_msgs)]
            msg_switchinfo = query(pmslot_msg.slot, SwitchSlotInfo, ❓, ❓ ,❓; assigned = true, filo = false)
            msg_gensetinfo = query(pmslot_msg.slot, :PartOfGenSet, ❓, ❓, ❓; assigned = true, filo = false)
            msg_pm_info = query(pmslot_msg.slot, :isPiecemaker, ❓; assigned = true, filo = false)
            @assert !isnothing(msg_switchinfo) "Switch pm slot $(pmslot_msg.slot.idx) is assigned a tag isPiecemaker but has no SwitchSlotInfo tag! $(msg_switchinfo)"
            @assert !isnothing(msg_gensetinfo) "Switch pms lot $(pmslot_msg.slot.idx) is assigned a tag isPiecemaker but has no PartOfGenSet tag! $(msg_gensetinfo)"
            counter = msg_pm_info.tag[2]
            @info "PIECEMAKER INFO $(msg_pm_info)"
            if ((counter < min_counter) && 
                msg_gensetinfo.tag[2] ∈ CLIENT_TO_GENERATORS[switchslot_idx] && 
                isnothing(query(net[1][pmslot_msg.slot.idx], :purifFail, ❓; filo = false)) &&
                isnothing(query(net[1][pmslot_msg.slot.idx], :isComplete, ❓; filo = false)) ) # we check if the generator set is suitable and if the piecemaker slot is not tagged with failed purification from a previous attempt
                min_counter = counter
                pmslot_to_fuse_with = pmslot_msg.slot
                gen_set_idx = msg_gensetinfo.tag[2]
            end
        end

        if isnothing(pmslot_to_fuse_with) 
            # in this case all potential piecemaker slots in common are occupied within an unsuitable generator set, meaning that it needs to start a new generator set in which it is itself the piecemaker
            @info "in this case all potential piecemaker slots are in an unsuitable generator set, meaning that $(switchslot_idx) needs to start a new generator set in which it is itself the piecemaker"
            @yield lock(net[1][switchslot_idx])
            attempt_id = next_attempt_id!(attempt_counter)
            tag!(net[1][switchslot_idx], Tag(:isPiecemaker, attempt_id))
            gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
            @info "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
            unlock(net[1][switchslot_idx])
        else
            # in this case we found a piecemaker slot within a suitable generator set and fuse with it
            isnothing(gen_set_idx) && error("Cannot fuse as no generator set was associated!")
            @yield @process fusion(sim, net, pmslot_to_fuse_with, net[1][switchslot_idx], net[1+switchslot_idx][clientslot_idx], gen_set_idx, tagid)

            # check if after fusion the GHZ state is complete with 4 clients (i.e., if the current slot is the last one to complete the generator set)
            msgs = queryall(net[1], :PartOfGenSet, gen_set_idx, pmslot_to_fuse_with.idx, ❓; filo = false)
            if length(msgs) == 4 
                tag!(pmslot_to_fuse_with, Tag(:isComplete, gen_set_idx))
                @yield @process consume_listener(sim, net) # if the fused state is complete, trigger the consume listener to consume it;
            end
        end
    end
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, clientswitch_slot::RegRef, client_slot::RegRef, gen_set_idx::Int, tagid::Int)
    @yield lock(piecemaker_slot) & lock(clientswitch_slot) & lock(client_slot)
    apply!((piecemaker_slot, clientswitch_slot), CNOT)
    @yield timeout(sim, Δt_CNOT)
    res = project_traceout!(clientswitch_slot, Z)
    @yield timeout(sim, Δt_meas)
    res == 2 && apply!(client_slot, X) # TODO: correction gate is now faster than light, add timeout!
    tag!(clientswitch_slot, Tag(:PartOfGenSet, gen_set_idx, piecemaker_slot.idx, tagid))
    unlock(piecemaker_slot)
    unlock(clientswitch_slot)
    unlock(client_slot)
    @info "Fused client $(clientswitch_slot.idx) with first client $(piecemaker_slot.idx)"
end


function prepare_sim(n::Int, T_link::Float64, cutoff::Float64, F_link::Float64, link_success_prob::Float64, attempt_t, purify::Float64)
    states_representation = QuantumOpticsRepr()
    @debug "Preparing simulation with parameters: n=$(n), T_link=$(T_link), cutoff=$(cutoff), F_link=$(F_link), link_success_prob=$(link_success_prob), attempt_time=$(attempt_t), purify=$(purify)"
    noise_model = Depolarization(T_link)

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

    attempt_counter = Ref(0)

    @process switch_listener(sim, net, purify, attempt_counter)
    for i in 1:n
        # Entangler for generation of bell pairs
        entangler = EntanglerProt(
            sim=sim, net=net, nodeA=1, chooseA=i,
            nodeB=1+i, # choose the client slot arbitrarily
            pairstate=noisy_bell_state(F_link),
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = attempt_t,
            retry_lock_time = attempt_t/2, local_busy_time_post = 0.0
        )

        @process entangler()
    end

    return sim
end

##
n = 7

Δt_CNOT = 100e-6  # 100 µs
Δt_XZ = 10e-6     # 10 µs
Δt_meas = 100e-9  # 100 ns
Δt_cutoff_list = [Inf]

T_coherence = Inf #1.0
cutoff = Inf
F_link = 1.0
link_success_prob = 1.0
attempt_time = 1e-6
purify = 0.5
runtime = 0.1


dataframes = DataFrame[]

log_data = Tuple{Float64, Vector{Int}, Float64, Bool, Int}[]
sim = prepare_sim(n, T_coherence, cutoff, F_link, link_success_prob, attempt_time, purify)
t_wallclock = @elapsed run(sim, runtime)
logs = DataFrame(
    log_data,
    [
        :timesteps,
        :clients_serviced,
        :fidelity,
        :discarded,
        :num_purifications,
    ]
)
logs[!, :t_wallclock] .= t_wallclock
logs[!, :T_coherence] .= T_coherence
logs[!, :F_link] .= F_link
logs[!, :link_success_prob] .= link_success_prob
logs[!, :attempt_time] .= attempt_time
logs[!, :purify] .= purify
@info logs
