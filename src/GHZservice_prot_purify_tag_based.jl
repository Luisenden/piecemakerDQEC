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

# const steane_generators = [[1,2,3,5], [1,2,4,6], [2,3,4,7]]
# # think about having the clients paritioned into 4 tiles, such that each 
# const CLIENT_TO_GENERATORS = Dict(
#     1 => [1, 2],
#     2 => [1, 2, 3],
#     3 => [1, 3],
#     4 => [2, 3],
#     5 => [1],
#     6 => [2],
#     7 => [3]
# )

# const CLIENT_TO_CLIENTINCOMMON = Dict(
#     1 => [2, 3, 4, 5, 6],
#     2 => [1, 3, 4, 5, 6, 7],
#     3 => [1, 2, 4, 5, 7],
#     4 => [1, 2, 3, 6, 7],
#     5 => [1, 2, 3],
#     6 => [1, 2, 4],
#     7 => [2, 3, 4]
# )

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

@resumable function consume_listener(sim, net, gen_set_index::Int)
    while true
        @yield onchange_tag(net[1])
        pm_msgs = queryall(net[1], :isPiecemaker; filo = false)
        isempty(pm_msgs) && continue
        for pm_msg in pm_msgs
            msgs = queryall(net[1], :PartOfGenSet, gen_set_index, pm_msg.slot.idx, ❓; filo = false) # TODO: here we want only one unique generator set, however, if it happens that one slot functions as a piecemaker twice we could get two gen sets --> ACTUALLY if it is the piecemaker then the second bell pair will never arrive! sanity check this in the outcomes!
            isempty(msgs) && continue
            length(msgs) < 4 && continue
            if length(msgs) == 4 # if there are 4 slots tagged as part of the generator set, we can consume the state
                @info "DONE 4: consume_listener for generator set $(gen_set_index) got msgs: $(msgs)"
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
                @yield reduce(&, lock.(clientslots_to_measure))
                @yield lock(pm_msg.slot)
                @info "Projecting out piecemaker qubit at slot $(pm_msg.slot.idx) to consume GHZ state for generator set $(gen_set_index)"
                res = project_traceout!(pm_msg.slot, σˣ)

                res == 2 && apply!(first(clientslots_to_measure), Z)
                @info "CLIENT SLOTS TO MEASURE: $(clientslots_to_measure)"
                obs_proj = SProjector(StabilizerState(ghzs[length(clientslots_to_measure)]))
                fidelity = real(observable(clientslots_to_measure, obs_proj))
                for clientslot in clientslots_to_measure
                    project_traceout!(clientslot, σˣ)
                end

                # deleting tags TODO: can this be done nicer?
                for msg in msgs
                    untag!(net[1], msg.id)
                end
                for msg in clientslot_idcs_msgs
                    @info "Deleting tag $(msg.tag) with id $(msg.id) at slot $(msg.slot)"
                    untag!(net[1], msg.id)
                end
                untag!(net[1], pm_msg.id)

                unlock.(clientslots_to_measure)
                unlock(pm_msg.slot)
                @info "FIDELITY of consumed GHZ state for generator set $(gen_set_index): $(fidelity)"
                # we can do some logging here if we want to track consumption times etc.
            else
                error("The generator set is ill formed as there are more than 4 slots tagged as part of the generator set!")
            end
        end
    end
end

@resumable function switch_listener(sim, net)
    # this process listens at the switch register for incoming bell pairs 
    # i.e., is triggered by a change of tag EntanglementCounterpart from the EntanglerProt
    # it tags a client's switch slot with tag SwitchSlotInfo containing the switchslot_idx, clientslot_idx and time_of_creation
    # finally, it runs the SwitchSlotProt
    while true
        @yield onchange_tag(net[1])
        while true
            msg = querydelete!(net[1], EntanglementCounterpart, ❓, ❓; assigned=true, filo = false)
            if !isnothing(msg)
                switchslot, _, taglog = msg
                @info "switch_listener: $(msg)"
                
                # get corresponding slot idx at client
                clientslot_idx = taglog[3] 
                # tag with switch slot info
                tagid = tag!(switchslot, Tag(SwitchSlotInfo, switchslot.idx, clientslot_idx, now(sim)))
                @yield @process SwitchSlotProt(sim, net, switchslot.idx, clientslot_idx, Int(tagid))
            else
                break
            end
        end
        @debug "STATE INFO" queryall(net[1], :PartOfGenSet, ❓, ❓, ❓; filo = false)
    end
end

@resumable function SwitchSlotProt(sim, net, switchslot_idx::Int, clientslot_idx::Int, tagid::Int)
    # this is a sequential protocol, meaning that it occupies the switch slots for its duration until it has decided for fusion
    # first it queries all other slots which have a generator set in common for the tag PartOfGenSet 
    # if the answer is 'nothing' it gets tagged with tag isPiecemaker
    # and it gets sorted into a (random if multiple are possible) generator set with the according tag PartOfGenSet
    # if the answer is (an)other slot(s) it tries to sort it into the generator set with the (oldest) piecemaker slot, where it gets fused
    # if the latter is not possible (i.e., this is the case when all potential piecemaker slots are already in an unsuitable generator set), 
    # it gets tagged with tag isPiecemaker and gets sorted into the an arbitrary suitable generator set with the according tag PartOfGenSet

    # TODO: we need to make sure that when this protocol is launched on the second bell pair of the slot, that it does not fuse with a state where the slot is already represented in;

    # if this protocol is launched on the second bell pair of the slot, it is already part of a generator set, so we need to make sure it is not fused with a state where it already is represented in.
    # we check if it is already part of a state in progress
    gen_set_idx = nothing
    part_of_gen_set_msg = query(net[1][switchslot_idx], :PartOfGenSet, ❓, ❓, ❓; assigned = true, filo = false)
    if !isnothing(part_of_gen_set_msg)
        gen_set_idx = part_of_gen_set_msg.tag[2]
    end

    # we query for all potential piecemaker slots among all potentially common clients that have a gen set in common, but are not part of the generator set that the current slot is already part of
    pmslot_msgs = []
    for otherswitchslot_idx in CLIENT_TO_CLIENTINCOMMON[switchslot_idx]
        pmslot_msg = query(net[1][otherswitchslot_idx], :isPiecemaker; assigned = true, filo = false)
        if isnothing(gen_set_idx) # if the slot is not part of a gen set yet we can proceed as normal
            push!(pmslot_msgs, pmslot_msg) 
        else # else we filter out the piecemaker slot that is part of the generator set the slot is already represented in
            isnothing(query(net[1][otherswitchslot_idx], :PartOfGenSet, gen_set_idx, ❓, ❓; assigned = true, filo = false)) && push!(pmslot_msgs, pmslot_msg)
        end
    end
    
    if all(isnothing.(pmslot_msgs))
        # in this case there is no piecemaker slot in any of the potentially common generator sets and this slot needs to become the piecemaker and start a GHZ generation
        @yield lock(net[1][switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:isPiecemaker))
        gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
        @info "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
        unlock(net[1][switchslot_idx])
    else
        # in this case there is at least one piecemaker slot in the potentially common generator sets, we want to assign to the oldest one and fuse with it
        min_time_of_creation = Inf
        pmslot_to_fuse_with = nothing
        gen_set_idx = nothing
        for pmslot_msg in pmslot_msgs[.!isnothing.(pmslot_msgs)]
            msg_switchinfo = query(pmslot_msg.slot, SwitchSlotInfo, ❓, ❓ ,❓; assigned = true, filo = false)
            msg_gensetinfo = query(pmslot_msg.slot, :PartOfGenSet, ❓, ❓, ❓; assigned = true, filo = false)
            @assert !isnothing(msg_switchinfo) "Switch pm slot $(pmslot_msg.slot.idx) is assigned a tag isPiecemaker but has no SwitchSlotInfo tag! $(msg_switchinfo)"
            @assert !isnothing(msg_gensetinfo) "Switch pms lot $(pmslot_msg.slot.idx) is assigned a tag isPiecemaker but has no PartOfGenSet tag! $(msg_gensetinfo)"
            time_of_creation = msg_switchinfo.tag[4]
            if (time_of_creation < min_time_of_creation) && msg_gensetinfo.tag[2] ∈ CLIENT_TO_GENERATORS[switchslot_idx] 
                min_time_of_creation = time_of_creation
                pmslot_to_fuse_with = pmslot_msg.slot
                gen_set_idx = msg_gensetinfo.tag[2]
            end
        end

        if isnothing(pmslot_to_fuse_with) 
            # in this case all potential piecemaker slots in common are occupied within an unsuitable generator set, meaning that it needs to start a new generator set in which it is itself the piecemaker
            @info "in this case all potential piecemaker slots are in an unsuitable generator set, meaning that $(switchslot_idx) needs to start a new generator set in which it is itself the piecemaker"
            @yield lock(net[1][switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:isPiecemaker))
            gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
            @info "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
            unlock(net[1][switchslot_idx])
        else
            # in this case we found a piecemaker slot within a suitable generator set and fuse with it
            isnothing(gen_set_idx) && error("Cannot fuse as no generator set was associated!")
            @yield @process fusion(sim, net, pmslot_to_fuse_with, net[1][switchslot_idx], net[1+switchslot_idx][clientslot_idx], gen_set_idx, tagid)
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


function prepare_sim(n::Int, T_link::Float64, cutoff::Float64, F_link::Float64, link_success_prob::Float64, attempt_t, purify::Bool)
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

    @process consume_listener(sim, net, 1)
    @process consume_listener(sim, net, 2)
    @process consume_listener(sim, net, 3)
    @process switch_listener(sim, net)
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
purify = false #true
runtime = 0.01

dataframes = DataFrame[]
sim = prepare_sim(n, T_coherence, cutoff, F_link, link_success_prob, attempt_time, purify)
t_wallclock = @elapsed run(sim, runtime)