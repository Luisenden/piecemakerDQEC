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

@resumable function switch_listener(sim, net)
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
                @yield @process switchslot_listener(sim, net, switchslot.idx, clientslot_idx)
            else
                break
            end
        end
    end
end

@resumable function switchslot_listener(sim, net, switchslot_idx::Int, clientslot_idx::Int)
    # this process listens at each switch slot for incoming bell pairs 
    #  i.e., is triggered by a change of tag EntanglementCounterpart from the EntanglerProt
    # it tags a client's switch slot with tag SwitchSlotInfo containing the switchslot_idx, clientslot_idx and time_of_creation
    # it queries all other slots for the tag PartOfGenSet which have a generator set in common
    # if the answer is 'nothing', it becomes a tag isPiecemaker
    # and it gets sorted into a (random if multiple are possible) generator set with the according tag PartOfGenSet
    # if the answer is (an)other slot(s) it gets fused with the (oldest) piecemaker slot

    pmslot_msgs = []
    for otherswitchslot_idx in CLIENT_TO_CLIENTINCOMMON[switchslot_idx]
        push!(pmslot_msgs, query(net[1][otherswitchslot_idx], :isPiecemaker; assigned = true, filo = false))
    end
    
    if all(isnothing.(pmslot_msgs))
        @yield lock(net[1][switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:isPiecemaker))
        gen_set_idx = rand(CLIENT_TO_GENERATORS[switchslot_idx])
        tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx))
        @info "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
        unlock(net[1][switchslot_idx])
    else
        min_time_of_creation = Inf
        pmslot_to_fuse_with = nothing
        gen_set_idx = nothing
        for msg in pmslot_msgs[.!isnothing.(pmslot_msgs)]
            pmslot, _, taglog = msg
            msg_switchinfo = query(pmslot, SwitchSlotInfo, ❓, ❓ ,❓; assigned = true, filo = false)
            msg_gensetinfo = query(pmslot, :PartOfGenSet, ❓; assigned = true, filo = false)
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
            # in this case all potential piecemaker slots are in an unsuitable generator set, meaning that it needs to start a new generator set in which it is itself the piecemaker
            @info "in this case all potential piecemaker slots are in an unsuitable generator set, meaning that $(switchslot_idx) needs to start a new generator set in which it is itself the piecemaker"
            @yield lock(net[1][switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:isPiecemaker))
            tagged = false
            for gen_set_idx in CLIENT_TO_GENERATORS[switchslot_idx]
                if isempty(queryall(net[1], :PartOfGenSet, gen_set_idx))
                    tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx))
                    tagged = true
                    break
                end
            end
            !tagged && error("Could not find a suitable generator set for piecemaker slot $(switchslot_idx)")
            @info "Tagging switch slot idx $(switchslot_idx) as piecemaker and part of $(gen_set_idx)"
            unlock(net[1][switchslot_idx])
        else
            isnothing(gen_set_idx) && error("no generator set was associated!")
            @yield lock(net[1][switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx))
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
