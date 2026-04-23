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

const steane_generators = [[1,2,3,5], [1,2,4,6], [2,3,4,7]]

const CLIENT_TO_GENERATORS = Dict(
    1 => [1, 2],
    2 => [1, 2, 3],
    3 => [1, 3],
    4 => [2, 3],
    5 => [1],
    6 => [2],
    7 => [3]
)

const CLIENT_TO_CLIENTINCOMMON = Dict(
    1 => [2, 3, 4, 5, 6],
    2 => [1, 3, 4, 5, 6, 7],
    3 => [1, 2, 4, 5, 7],
    4 => [1, 2, 3, 6, 7],
    5 => [1, 2, 3],
    6 => [1, 2, 4],
    7 => [2, 3, 4]
)

function noisy_bell_state(target_fidelity::Float64=0.97)
    λ = (4 * target_fidelity - 1) / 3
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return λ * perfect_pair_dm + (1 - λ) * mixed_dm
end

struct SwitchSlotInfo
    clientslot_idx::Int
    switchslot_idx::Int
    time_of_creation::Float64
end

@resumable function consume_listener(sim, net, gen_set_index::Int)
    while true
        @yield onchange_tag(net[1])
        pm_msgs = queryall(net[1], :isPiecemaker; filo = false)
        isempty(pm_msgs) && continue
        for pm_msg in pm_msgs
            @info "consume_listener for generator set $(gen_set_index) got pm_msg: $(pm_msg)"
            pm_slot, _, _ = pm_msg
            msgs = queryall(net[1], :PartOfGenSet, gen_set_index, pm_slot.idx; filo = false)
            length(msgs) < 4 && continue
            if length(msgs) == 4 # if there are 4 slots tagged as part of the generator set, we can consume the state
                allequal([msg[3][3] for msg in msgs]) || error("Not all slots tagged as part of the generator set have the same piecemaker slot idx!")
                @yield lock(pm_slot)
                @info "Projecting out piecemaker qubit at slot $(pm_slot.idx) to consume GHZ state for generator set $(gen_set_index)"
                res = project_traceout!(pm_slot, σˣ)
                unlock(pm_slot)

                #choose arbitrary client slot idx among the 4 tagged as part of the generator set to apply correction if needed
                clientslot_idcs = [query(switchslot, SwitchSlotInfo, ❓, ❓, ❓; filo = false)[3][3] for switchslot in [msg[1] for msg in msgs]] 
                # required TODO: we need something to differentiate between two different tags SwitchSlotInfo for the same slot, as otherwise we might access here the wrong client slot
                
                clientslots_to_measure = [net[1+switchslot_idx][clientslot_idx] for clientslot_idx in clientslot_idcs for switchslot_idx in [msg[1].idx for msg in msgs]]
                @info "Measuring client qubits at slots $(clientslot_idcs) to consume GHZ state for generator set $(gen_set_index)"
                @yield reduce(&, lock.(clientslots_to_measure))
                clientslot_for_correction_idx = clientslot_idcs[1]
                if res == 2
                    apply!(net[1+pm_slot.idx][clientslot_for_correction_idx], Z)
                end
                unlock.(clientslots_to_measure)

                obs_proj = SProjector(StabilizerState(ghzs[4]))
                fidelity = real(observable(clientslots_to_measure, obs_proj))
                @info "Fidelity of consumed GHZ state for generator set $(gen_set_index): $(fidelity)"
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
                tag!(switchslot, Tag(SwitchSlotInfo, switchslot.idx, clientslot_idx, now(sim)))
                @yield @process SwitchSlotProt(sim, net, switchslot.idx, clientslot_idx)
            else
                break
            end
        end
        @debug "STATE INFO" queryall(net[1], :PartOfGenSet, ❓, ❓; filo = false)
    end
end

@resumable function SwitchSlotProt(sim, net, switchslot_idx::Int, clientslot_idx::Int)
    # this is a sequential protocol, meaning that it occupies the switch slots for its duration until it has decided for fusion
    # first it queries all other slots which have a generator set in common for the tag PartOfGenSet 
    # if the answer is 'nothing' it gets tagged with tag isPiecemaker
    # and it gets sorted into a (random if multiple are possible) generator set with the according tag PartOfGenSet
    # if the answer is (an)other slot(s) it tries to sort it into the generator set with the (oldest) piecemaker slot, where it gets fused
    # if the latter is not possible (i.e., this is the case where all potential piecemaker slots are already in an unsuitable generator set), 
    # it gets tagged with tag isPiecemaker and gets sorted into the first suitable generator set with the according tag PartOfGenSet

    # TODO: we need to make sure that when this protocol is launched on the second bell pair of the slot, that is does not fuse with a state where the slot is already represented in;

    # if this protocol is launched on the second bell pair of the slot, it is already part of a generator set, so we need to make sure it is not fused with a state where it already is represented in.
    # we check if it is already part of a state in progress
    gen_set_idx = nothing
    part_of_gen_set_msg = query(net[1][switchslot_idx], :PartOfGenSet, ❓, ❓; assigned = true, filo = false)
    if !isnothing(part_of_gen_set_msg)
        gen_set_idx = part_of_gen_set_msg[3][2]
    end

    # we query for all potential piecemaker slots among all potentially common clients that have a gen set in common, but are not part of the generator set that the current slot is already part of
    pmslot_msgs = []
    for otherswitchslot_idx in CLIENT_TO_CLIENTINCOMMON[switchslot_idx]
        pmslot_msg = query(net[1][otherswitchslot_idx], :isPiecemaker; assigned = true, filo = false)
        if isnothing(gen_set_idx) # if the slot is not part of a gen set yet we can proceed as normal
            push!(pmslot_msgs, pmslot_msg) 
        else # else we filter out the piecemaker slot that is part of the generator set the slot is already represented in
            isnothing(query(net[1][otherswitchslot_idx], :PartOfGenSet, gen_set_idx, ❓; assigned = true, filo = false)) && push!(pmslot_msgs, pmslot_msg)
        end
    end
    
    if all(isnothing.(pmslot_msgs))
        # in this case there is no piecemaker slot in any of the potentially common generator sets and this slot needs to become the piecemaker and start a GHZ generation
        @yield lock(net[1][switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:isPiecemaker))
        gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx))
        @info "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
        unlock(net[1][switchslot_idx])
    else
        # in this case there is at least one piecemaker slot in the potentially common generator sets
        min_time_of_creation = Inf
        pmslot_to_fuse_with = nothing
        gen_set_idx = nothing
        for msg in pmslot_msgs[.!isnothing.(pmslot_msgs)]
            pmslot, _, taglog = msg
            msg_switchinfo = query(pmslot, SwitchSlotInfo, ❓, ❓ ,❓; assigned = true, filo = false)
            msg_gensetinfo = query(pmslot, :PartOfGenSet, ❓, ❓; assigned = true, filo = false)
            isnothing(msg_switchinfo) && error("Switch pmslot $(pmslot.idx) is assigned a tag isPiecemaker but does not have a tag SwitchSlotInfo!")
            isnothing(msg_gensetinfo) && error("Switch pmslot $(pmslot.idx) is assigned a tag isPiecemaker but does not have a tag PartOfGenSet")
            pmslot, _, taglog = msg_switchinfo
            time_of_creation = taglog[4]
            if (time_of_creation < min_time_of_creation) && msg_gensetinfo[3][2] ∈ CLIENT_TO_GENERATORS[switchslot_idx] 
                min_time_of_creation = time_of_creation
                pmslot_to_fuse_with = pmslot
                gen_set_idx = msg_gensetinfo[3][2]
            end
        end

        if isnothing(pmslot_to_fuse_with) 
            # in this case all potential piecemaker slots in common occupied within an unsuitable generator set, meaning that it needs to start a new generator set in which it is itself the piecemaker
            @info "in this case all potential piecemaker slots are in an unsuitable generator set, meaning that $(switchslot_idx) needs to start a new generator set in which it is itself the piecemaker"
            @yield lock(net[1][switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:isPiecemaker))
            gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx))
            @info "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
            unlock(net[1][switchslot_idx])
        else
            # in this case we found a piecemaker slot within a suitable generator set and fuse with it
            isnothing(gen_set_idx) && error("Cannot fuse as no generator set was associated!")
            @yield lock(net[1][switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, pmslot_to_fuse_with.idx))
            @info "Tagging switch slot idx $(switchslot_idx) as part of $(gen_set_idx)"
            unlock(net[1][switchslot_idx])
            @process fusion(sim, net, pmslot_to_fuse_with, net[1][switchslot_idx], net[1+switchslot_idx][clientslot_idx])
        end
    end
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, clientswitch_slot::RegRef, client_slot::RegRef)
    @yield lock(piecemaker_slot) & lock(clientswitch_slot) & lock(client_slot)
    apply!((piecemaker_slot, clientswitch_slot), CNOT)
    @yield timeout(sim, Δt_CNOT)
    res = project_traceout!(clientswitch_slot, Z)
    @yield timeout(sim, Δt_meas)
    tag!(client_slot, Tag(:updateX, res))
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

    # @process consume_listener(sim, net, 1)
    # @process consume_listener(sim, net, 2)
    # @process consume_listener(sim, net, 3)
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

T_coherence = 1.0
cutoff = Inf
F_link = 0.97
link_success_prob = 1.0
attempt_time = 1e-6
purify = true
runtime = 0.01

dataframes = DataFrame[]
sim = prepare_sim(n, T_coherence, cutoff, F_link, link_success_prob, attempt_time, purify)
t_wallclock = @elapsed run(sim, runtime)

