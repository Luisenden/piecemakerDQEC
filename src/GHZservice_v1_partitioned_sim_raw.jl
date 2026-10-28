using QuantumSavory
using QuantumSavory: Register, X, Z, Y, CNOT
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using DataFrames
using Random

# This file contains only the simulation model and one-configuration runner.
@assert @isdefined(code) "Define `code` before including GHZservice_v1_partitioned_sim_raw.jl"
#@assert @isdefined(error_model) "Define `error_model` before including GHZservice_v1_partitioned_sim_raw.jl"
const error_model = ""

const codes = Dict(

    # Supports follow the row order of the corresponding parity-check tableau, so
    # generator_idx g indexes row g of H. Steane: QuantumClifford Steane7() labeling;
    # support g carries both X-check row g and Z-check row g+3 (CSS, shared supports).
    "Steane713" => (7, [
        [4, 5, 6, 7],
        [2, 3, 6, 7],
        [1, 3, 5, 7],
    ]),

    # [[12,2,3]] bivariate-bicycle code; order = rows of BB12_2_3 in utils_pseudothreshold.jl
    "BB12_2_3" => (12, [
        [1, 3, 8, 10], [1, 2, 9, 11], [2, 3, 7, 12], [4, 6, 7, 11], [4, 5, 8, 12],   # Z-type
        [3, 4, 7, 8],  [1, 5, 8, 9],  [2, 6, 7, 9],  [1, 6, 10, 11], [2, 4, 11, 12], # X-type
    ]),

    # [[26,2,5]] generalized bicycle code, l = 13, a = 1 + x^5, b = 1 + x
    # (Wang & Pryadko, arXiv:2203.17216; github.com/QEC-pages/GB-codes), up to qubit relabeling;
    # order = rows of GB26_2_5 in utils_pseudothreshold.jl
    "GB26_2_5" => (26, [
        [10, 12, 14, 24], [11, 13, 15, 25], [1, 12, 16, 26], [2, 13, 14, 17],
        [1, 3, 15, 18],   [2, 4, 16, 19],   [3, 5, 17, 20],  [4, 6, 18, 21],
        [5, 7, 19, 22],   [6, 8, 20, 23],   [7, 9, 21, 24],  [8, 10, 22, 25],   # Z-type
        [1, 4, 16, 18],   [2, 5, 17, 19],   [3, 6, 18, 20],  [4, 7, 19, 21],
        [5, 8, 20, 22],   [6, 9, 21, 23],   [7, 10, 22, 24], [8, 11, 23, 25],
        [9, 12, 24, 26],  [10, 13, 14, 25], [1, 11, 15, 26], [2, 12, 14, 16],   # X-type
    ]),
)

# Fixed data-qubit partitions for the codes that use multiple data qubits per node.
# Each inner vector contains the data qubits hosted by one remote node.
# Codes without an explicit entry fall back to one data qubit per node.
const NODE_PARTITIONS = Dict(
    # Steane keeps the original architecture: one data qubit / one communication node.
    # The three entries in GENERATORS are the three distinct support patterns;
    # X- and Z-type checks share these supports. For data-qubit waiting times, every
    # completed GHZ on a support is one stabilizer-measurement opportunity for each
    # data qubit in that support, independent of whether it is used for X or Z.
    "Steane713" => [[q] for q in 1:7],

    # 6 nodes, two data qubits per node. Stabilizer loads: [7, 6, 7, 7, 6, 7].
    "BB12_2_3_partitioned" => [
        [1, 12],
        [2, 10],
        [3, 11],
        [4, 9],
        [5, 7],
        [6, 8],
    ],
    # 12 nodes
    "BB12_2_3" => [[q] for q in 1:12],

    # 12 nodes. Every weight-4 generator spans four distinct nodes.
    # Stabilizer loads: [9, 10, 8, 8, 8, 8, 7, 8, 7, 8, 8, 7].
    "GB26_2_5" => [
        [3, 9, 11],
        [13, 23, 24],
        [1, 2],
        [4, 5],
        [6, 7],
        [8, 12],
        [10, 15],
        [14, 18],
        [16, 17],
        [19, 20],
        [21, 22],
        [25, 26],
    ],
)

# `code` may carry a "_partitioned" suffix: the suffix only selects the node layout in
# NODE_PARTITIONS, the stabilizer set comes from the base code.
const BASE_CODE = replace(code, r"_partitioned$" => "")
const N_DATA = codes[BASE_CODE][1]
const GENERATORS = codes[BASE_CODE][2]
const NODE_PARTITION = get(NODE_PARTITIONS, code, [[q] for q in 1:N_DATA])
const N_NODES = length(NODE_PARTITION)

# Validate that the partition is a proper partition of all data qubits.
const PARTITIONED_DATA_QUBITS = vcat(NODE_PARTITION...)
@assert length(PARTITIONED_DATA_QUBITS) == N_DATA "Node partition must contain exactly N_DATA qubits."
@assert allunique(PARTITIONED_DATA_QUBITS) "A data qubit occurs in more than one node."
@assert sort(PARTITIONED_DATA_QUBITS) == collect(1:N_DATA) "Node partition must contain every data qubit exactly once."

# Map the QEC-layer data-qubit supports to the network-layer node supports.
const DATA_TO_NODE = Dict(
    q => node_idx
    for (node_idx, qubits) in enumerate(NODE_PARTITION)
    for q in qubits
)

const GENERATOR_TO_NODES = [
    sort([DATA_TO_NODE[q] for q in generator])
    for generator in GENERATORS
]

# With one communication qubit per node, no stabilizer may require a node twice.
@assert all(
    length(node_support) == length(generator) && allunique(node_support)
    for (node_support, generator) in zip(GENERATOR_TO_NODES, GENERATORS)
) "Invalid node partition: at least one stabilizer contains multiple data qubits hosted by the same node."

const paulis = (nothing, X, Y, Z)
# @inline function sample_depol2q(gate_fidelity::Float64)
#     λ = 4/3 * (1-gate_fidelity)
#     rand() < (1-λ) && return (nothing, nothing)
#     return (rand(paulis), rand(paulis))
# end

const ghzs = [ghz(k) for k in 1:maximum(length.(GENERATORS))] # cache required GHZ sizes

# Network-level lookup tables. These are indexed by physical node / switch slot,
# not by data-qubit index.
const NODE_TO_GENERATORS = Dict(
    node_idx => findall(nodes -> node_idx in nodes, GENERATOR_TO_NODES)
    for node_idx in 1:N_NODES
)

const NODE_TO_NODEINCOMMON = Dict(
    node_idx => sort(setdiff(unique(vcat((GENERATOR_TO_NODES[g] for g in NODE_TO_GENERATORS[node_idx])...)), [node_idx]))
    for node_idx in 1:N_NODES
)

# Per-Bell-pair timing record, kept outside the QuantumSavory tag system.
#   birth: listener-observed Bell-pair birth time (b_j)
#   cnot:  time the fusion CNOT is applied (c_j); NaN for the piecemaker pair
#   meas:  time the incoming switch qubit is Z-measured, i.e. end of CNOT (g_j); NaN for the piecemaker pair
mutable struct PairTimes
    birth::Float64
    cnot::Float64
    meas::Float64
end

# One row per successfully consumed weight-4 GHZ state. Pairs are in FUSION order:
# pair 1 is the piecemaker, pairs 2..4 are ordered by their CNOT time.
#   timesteps: consumption time t_end (after the piecemaker readout wait)
#   t_pm_meas: time the piecemaker is X-measured
const RawGHZRow = NamedTuple{
    (:timesteps, :generator_idx, :t_pm_meas,
     :bellpair_1, :bellpair_2, :bellpair_3, :bellpair_4,
     :cnot_2, :cnot_3, :cnot_4,
     :meas_2, :meas_3, :meas_4),
    Tuple{Float64, Int, Float64,
          Float64, Float64, Float64, Float64,
          Float64, Float64, Float64,
          Float64, Float64, Float64},
}

# Completions per generator that make one QEC round: Steane checks share X/Z supports, so
# every support (one entry of GENERATORS) needs 2 completions.
const ROUND_REQUIRED_COUNT = BASE_CODE == "Steane713" ? 2 : 1

# Raw GHZ rows plus the round bookkeeping. A round of stabilizer measurements is complete once
# every generator has been completed ROUND_REQUIRED_COUNT times since the previous round
# (same rule as round_completion_times in utils_pseudothreshold.jl).
mutable struct GHZLog
    rows::Vector{RawGHZRow}
    counts::Vector{Int}     # completions per generator since the last completed round
    n_rounds::Int
    target_rounds::Int
end
GHZLog(target_rounds::Int) = GHZLog(RawGHZRow[], zeros(Int, length(GENERATORS)), 0, target_rounds)

function log_ghz!(log::GHZLog, row::RawGHZRow)
    push!(log.rows, row)
    log.counts[row.generator_idx] += 1
    if all(>=(ROUND_REQUIRED_COUNT), log.counts)
        log.n_rounds += 1
        fill!(log.counts, 0)
    end
    return log
end

@info "Using code $(code) with $(N_DATA) data qubits on $(N_NODES) physical nodes; node partition = $(NODE_PARTITION)"

# function noisy_bell_state(target_fidelity::Float64=0.97)
#     λ = (4 * target_fidelity - 1) / 3
#     perfect_pair::StabilizerState = StabilizerState("XX ZZ")
#     perfect_pair_dm = SProjector(perfect_pair)
#     mixed_dm = MixedState(SProjector(perfect_pair))
#     return λ * perfect_pair_dm + (1 - λ) * mixed_dm
# end

struct SwitchSlotInfo
    switchslot_idx::Int
    clientslot_idx::Int
    time_of_creation::Float64
end

function next_attempt_id!(counter)
    counter[] += 1
    return counter[]
end

@resumable function consumer(sim, net, pm_slot::RegRef, gen_set_index::Int, attempt_id::Int, iscutoff::Bool, log_data::GHZLog, pair_times::Dict{Int,PairTimes}, Δt_readout::Float64, readout_fidelity::Float64)
    @yield lock(pm_slot)
    # Re-check under the lock: the cutoff timer and the completing fusion can both schedule a
    # consumer for the same attempt; whichever gets the lock second must not touch the slot again
    # (it may already be consumed or even host a new attempt).
    pm_msg = query(pm_slot, :isPiecemaker, attempt_id; assigned = true, filo = false)
    if isnothing(pm_msg)
        @debug "Attempt $(attempt_id) at piecemaker slot $(pm_slot.idx) was already consumed, nothing to do."
        unlock(pm_slot)
        return
    end
    untag!(net[1], pm_msg.id) # untag such that is is not found anymore

    msgs = queryall(net[1], :PartOfGenSet, gen_set_index, pm_slot.idx, ❓; filo = false)

    expected_weight = length(GENERATOR_TO_NODES[gen_set_index])
    !iscutoff && @assert length(msgs) == expected_weight "Expected to find all $(expected_weight) node slots of complete generator set for piecemaker slot $(pm_slot.idx) but received: $(msgs)"

    # check that the participating switch slots are exactly the correct node support of the generator
    switchslots_idcs = sort([msg.slot.idx for msg in msgs])
    @debug "Generator set indices for generator set $(gen_set_index): $(switchslots_idcs)"

    
    !iscutoff && @assert switchslots_idcs == GENERATOR_TO_NODES[gen_set_index] "The generator set is not correct."
    @assert allunique(switchslots_idcs) "All switch slots tagged as part of the generator set need to be unique!"
    @assert allequal([msg.tag[3] for msg in msgs]) "Not all slots tagged as part of the generator set have the same piecemaker slot idx!"

    tagids = [msg.tag[4] for msg in msgs]
    # choose an arbitrary remote communication qubit among those tagged as part of the generator set to apply correction if needed   
    clientslot_idcs_msgs = [client_msg for switchslotmsg in msgs for client_msg in queryall(net[1][switchslotmsg.slot.idx], SwitchSlotInfo, ❓, ❓, ❓; filo=false) if client_msg.id in tagids]
    clientslot_idcs = [client_msg.tag[3] for client_msg in clientslot_idcs_msgs]
    @debug "SWICH SLOT INDICES: $([msg.slot.idx for msg in msgs]), CLIENT SLOT INDICES: $clientslot_idcs"
    
    # zip can truncate: If clientslot_idcs has length 3 or 5 because a SwitchSlotInfo tag is missing or duplicated, zip silently truncates. That could produce a smaller/wrong GHZ fidelity calculation without immediately failing.
    @assert length(clientslot_idcs_msgs) == length(msgs)
    @assert length(clientslot_idcs) == length(msgs)

    # The PartOfGenSet tag stores the SwitchSlotInfo tag id in field 4.
    # Birth times are logging metadata kept separately from QuantumSavory tags.
    @assert all(haskey(pair_times, Int(tagid)) for tagid in tagids) "Missing Bell-pair timing log entry."
    # Fusion order: piecemaker pair first, then the fused pairs by CNOT time.
    # (Hold references before the dictionary entries are deleted below.)
    pm_tagid = Int(only(msg.tag[4] for msg in msgs if msg.slot.idx == pm_slot.idx))
    fused_tagids = sort([Int(t) for t in tagids if Int(t) != pm_tagid]; by = t -> pair_times[t].cnot)
    ordered_times = PairTimes[pair_times[pm_tagid]; [pair_times[t] for t in fused_tagids]]

    clientslots_to_measure = [net[1+switchslot_idx][clientslot_idx] for (switchslot_idx, clientslot_idx) in zip([msg.slot.idx for msg in msgs], clientslot_idcs)]
    @debug "Measuring client qubits at slots $(clientslot_idcs) to consume GHZ state for generator set $(gen_set_index)"

    @yield reduce(&, lock.(clientslots_to_measure))
    @debug "Projecting out piecemaker qubit at slot $(pm_slot.idx) to consume GHZ state for generator set $(gen_set_index)"
    t_pm_meas = now(sim)
    res = project_traceout!(pm_slot, σˣ)
    # if rand() > readout_fidelity
    #     res = 3 - res
    # end
    # res == 2 && apply!(first(clientslots_to_measure), Z)
    !iscutoff && @yield timeout(sim, Δt_readout)
    #obs_proj = SProjector(StabilizerState(ghzs[length(clientslots_to_measure)]))
    #fidelity = real(observable(clientslots_to_measure, obs_proj))
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
        delete!(pair_times, Int(msg.id))
        unlock(net[1 + msg.slot.idx][msg.tag[3]])
    end

    unlock(pm_slot)

    # One raw row per SUCCESSFULLY completed GHZ state. Cutoff-discarded partial
    # constructions are deliberately not logged. Per pair we log the listener-observed
    # birth time b_j, the fusion CNOT time c_j and the Z-measurement time g_j, plus the
    # piecemaker X-measurement time and the consumption time: exactly the inputs of
    # ghz_fidelity_closedform (GHZfidelity_closedform.jl).
    if !iscutoff
        @assert length(ordered_times) == 4 "Raw output currently assumes weight-4 GHZ constructions."
        @assert isnan(ordered_times[1].cnot) "The piecemaker pair must not have a CNOT time."
        @assert all(!isnan(pt.cnot) && !isnan(pt.meas) for pt in ordered_times[2:4]) "Missing fusion times."
        pt = ordered_times
        log_ghz!(log_data, (
            timesteps = time_of_consumption,
            generator_idx = gen_set_index,
            t_pm_meas = t_pm_meas,
            bellpair_1 = pt[1].birth, bellpair_2 = pt[2].birth, bellpair_3 = pt[3].birth, bellpair_4 = pt[4].birth,
            cnot_2 = pt[2].cnot, cnot_3 = pt[3].cnot, cnot_4 = pt[4].cnot,
            meas_2 = pt[2].meas, meas_3 = pt[3].meas, meas_4 = pt[4].meas,
        ))
        # Ends the enclosing run() at exactly the GHZ state that completes the target round.
        log_data.n_rounds >= log_data.target_rounds && throw(StopSimulation())
    end
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
    pending_batches::Vector{Tuple{Vector{Any}, Float64}}, # (messages, listener-observed Bell-pair birth/arrival time)
    worker_running::Ref{Bool},
    attempt_counter::Ref{Int},
    Δt_CNOTgate::Float64,
    gate_fidelity::Float64,
    Δt_readout::Float64,
    readout_fidelity::Float64,
    cutoff::Float64,
    Δt_rotation_shuttle::Float64,
    global_pause::Ref{Float64},
    log_data::GHZLog,
    pair_times::Dict{Int,PairTimes},
    singlezone::Bool,
)
    while !isempty(pending_batches)
        msgs, batch_time = popfirst!(pending_batches)

        t_ready = batch_time + Δt_rotation_shuttle
        now(sim) < t_ready && @yield timeout(sim, t_ready - now(sim))

        for msg in msgs
            clientslot_idx = msg.tag[3]
            # Keep the original SwitchSlotInfo tag unchanged; cutoff logic uses its timestamp.
            tagid = Int(tag!(msg.slot, Tag(SwitchSlotInfo, msg.slot.idx, clientslot_idx, batch_time))) # tag! returns an Int128, which Tag does not accept
            # Store the listener-observed Bell-pair birth time outside the Tag system.
            pair_times[tagid] = PairTimes(batch_time, NaN, NaN)
            # get corresponding slot idx at client

            switchslot_idx = msg.slot.idx
            @assert isnothing(query(net[1][switchslot_idx], :PartOfGenSet, ❓, ❓, ❓; assigned = true, filo = false)) "This slot is already part of a GHZ attempt, it should not have received a new Bell pair."

            # we query for all potential piecemaker slots among clients that have a generator set in common
            pmslot_msgs = []
            for otherswitchslot_idx in NODE_TO_NODEINCOMMON[switchslot_idx]
                pmslot_msg = query(net[1][otherswitchslot_idx], :isPiecemaker, ❓; assigned = true, filo = false)
                if !isnothing(pmslot_msg)
                    @debug "Found potential piecemaker slot at switch slot idx $(otherswitchslot_idx) with info $(pmslot_msg)"
                    gensetinfo_msg = query(pmslot_msg.slot, :PartOfGenSet, ❓, pmslot_msg.slot.idx, ❓; assigned = true, filo = false)
                    @debug "checking if current switch slot idx $(switchslot_idx) and potential piecemaker slot idx $(otherswitchslot_idx) are part of generator node support $(GENERATOR_TO_NODES[gensetinfo_msg.tag[2]])"
                    (switchslot_idx ∈ GENERATOR_TO_NODES[gensetinfo_msg.tag[2]]) && push!(pmslot_msgs, pmslot_msg) # only consider it as potential piecemaker slot if the current switch slot is part of the same generator set as the piecemaker slot
                end
            end

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

            # if no piecemaker slot is found, we become the piecemaker and start a new GHZ attempt
            if isempty(pmslot_msgs)
                @yield lock(net[1][switchslot_idx])
                attempt_id = next_attempt_id!(attempt_counter)
                tag!(net[1][switchslot_idx], Tag(:isPiecemaker, attempt_id))
                gen_set_idx = rand(NODE_TO_GENERATORS[switchslot_idx])
                tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
                @debug "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
                isfinite(cutoff) && @process CutoffTimerProt(sim, net, msg.slot, cutoff, log_data, pair_times, batch_time, attempt_id) # this should not impart any simulation time?
                unlock(net[1][switchslot_idx])
            else
                @debug "Found potential piecemaker slots: $(pmslot_msgs)"
                fusion_proc = @process FusionConsumer(sim, net, pmslot_to_fuse_with, msg.slot.idx, clientslot_idx, gen_set_idx, Int(min_counter), Int(tagid), Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, log_data, pair_times)
                # Single fusion zone: the switch is busy until this fusion (and, if it completes the
                # GHZ state, the piecemaker readout) is done before it handles the next Bell pair.
                # Otherwise every fusion gets its own zone and only the slot locks serialise them.
                singlezone && @yield fusion_proc
            end
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
    log_data::GHZLog,
    pair_times::Dict{Int,PairTimes},
    singlezone::Bool,
)
    pending_batches = Tuple{Vector{Any}, Float64}[]
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

        push!(pending_batches, (msgs, batch_time)) # queue Bell pairs together with their listener-observed birth time
        # This is the physical switch pause for accepting/rotating/shuttling this batch.
        # The listener remains alive during this timeout and can discard newly arriving Bell pairs.
        global_pause[] = batch_time + Δt_rotation_shuttle

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
                pair_times,
                singlezone,
            )
        end
    end
end


@resumable function CutoffTimerProt(sim, net, pm_slot::RegRef, cutoff::Float64, log_data::GHZLog, pair_times::Dict{Int,PairTimes}, batch_time::Float64, attempt_id::Int)
    # this process is started by the switch_listener and consumes the GHZ state when the cutoff time is reached
    @yield timeout(sim, max(0.0, batch_time+ cutoff - now(sim)))
    msg = query(pm_slot, :isPiecemaker, attempt_id)
    if !isnothing(msg)
        gensetinfo_msg = query(pm_slot, :PartOfGenSet, ❓, ❓, ❓; filo = false)
        genset_idx = gensetinfo_msg.tag[2]
        @yield @process consumer(sim, net, pm_slot, genset_idx, attempt_id, true, log_data, pair_times, 0.0, 1.0) # if the cutoff time has been exceeded, we consume the GHZ state that was being built with the corresponding piecemaker slot to free up the switch slots
    end
end



@resumable function FusionConsumer(sim, net, pmslot_to_fuse_with::RegRef, switchslot_idx::Int, clientslot_idx::Int, gen_set_idx::Int, attempt_id::Int, tagid::Int, Δt_CNOTgate::Float64, gate_fidelity::Float64, Δt_readout::Float64, readout_fidelity::Float64, log_data::GHZLog, pair_times::Dict{Int,PairTimes})

    fused = @yield @process fusion(sim, net, pmslot_to_fuse_with, net[1][switchslot_idx], net[1+switchslot_idx][clientslot_idx], gen_set_idx, attempt_id, tagid, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, pair_times)
    fused || return # the attempt was discarded by the cutoff before this fusion could start

    # check if after fusion the GHZ state contains all nodes required by this generator
    msgs = queryall(net[1], :PartOfGenSet, gen_set_idx, pmslot_to_fuse_with.idx, ❓; filo = false)
    if length(msgs) == length(GENERATOR_TO_NODES[gen_set_idx])
        @yield @process consumer(sim, net, pmslot_to_fuse_with, gen_set_idx, attempt_id, false, log_data, pair_times, Δt_readout, readout_fidelity) # if the fused state is complete, run the consume listener to consume it;
    end

end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, clientswitch_slot::RegRef, client_slot::RegRef, gen_set_idx::Int, attempt_id::Int, tagid::Int, Δt_CNOTgate::Float64, gate_fidelity::Float64, Δt_readout::Float64, readout_fidelity::Float64, pair_times::Dict{Int,PairTimes})
    @yield lock(piecemaker_slot) & lock(clientswitch_slot) & lock(client_slot)
    # The piecemaker was chosen without holding its lock, so the attempt may have been consumed
    # by the cutoff in the meantime (the slot may then be empty or host a new attempt).
    # In that case the incoming Bell pair is discarded instead of fused.
    if isnothing(query(piecemaker_slot, :isPiecemaker, attempt_id; assigned = true, filo = false))
        @debug "Attempt $(attempt_id) at piecemaker slot $(piecemaker_slot.idx) is gone, discarding Bell pair at switch slot $(clientswitch_slot.idx)"
        project_traceout!(clientswitch_slot, σˣ)
        project_traceout!(client_slot, σˣ)
        delete!(pair_times, tagid)
        unlock(piecemaker_slot)
        unlock(clientswitch_slot)
        unlock(client_slot)
        untag!(net[1], Int128(tagid)) # SwitchSlotInfo tag (tag ids are Int128); also wakes the entangler waiting for a free slot
        return false
    end
    pair_times[tagid].cnot = now(sim)   # c_j
    apply!((piecemaker_slot, clientswitch_slot), CNOT)
    @yield timeout(sim, Δt_CNOTgate)
    pair_times[tagid].meas = now(sim)   # g_j: gate end = Z-measurement of the incoming switch qubit
    # noisygate = sample_depol2q(gate_fidelity)
    # !isnothing(noisygate[1]) && apply!(piecemaker_slot, noisygate[1])
    # !isnothing(noisygate[2]) && apply!(clientswitch_slot, noisygate[2])

    res = project_traceout!(clientswitch_slot, σᶻ)
    # if rand() > readout_fidelity
    #     res = 3 - res
    # end
    @yield timeout(sim, Δt_readout)
    # res == 2 && apply!(client_slot, X) # TODO: correction gate is now faster than light, add timeout!
    tag!(clientswitch_slot, Tag(:PartOfGenSet, gen_set_idx, piecemaker_slot.idx, tagid))
    unlock(piecemaker_slot)
    unlock(clientswitch_slot)
    unlock(client_slot)
    @debug "Fused client $(clientswitch_slot.idx) with first client $(piecemaker_slot.idx)"
    return true
end

@resumable function naive_entangler(
    sim,
    net,
    n_nodes,
    F_link,
    link_success_prob,
    attempt_t,
    Δt_rotation_shuttle,
)
    for i in 1:n_nodes
        entangler = EntanglerProt(
            sim = sim,
            net = net,
            nodeA = 1,
            nodeB = 1 + i,

            chooseslotA = i,
            chooseslotB = 1,

            #pairstate = noisy_bell_state(F_link),

            success_prob = link_success_prob,
            rounds = -1,
            attempts = -1,
            attempt_time = attempt_t,
            retry_lock_time = nothing,
        )

        @process entangler()
    end
end

function prepare_sim(n_nodes, T_link::Float64, F_link::Float64, link_success_prob::Float64, attempt_t::Float64, Δt_CNOTgate::Float64, gate_fidelity::Float64, Δt_readout::Float64, readout_fidelity::Float64,
    error_model::String, cutoff::Float64, Δt_rotation_shuttle::Float64, log_data::GHZLog, singlezone::Bool)

    states_representation = CliffordRepr()#QuantumOpticsRepr()
    @debug "Preparing simulation with parameters: n_nodes=$(n_nodes), T_link=$(T_link), cutoff=$(cutoff), F_link=$(F_link), link_success_prob=$(link_success_prob), attempt_time=$(attempt_t)"
    
    # noise_model = error_model == "dephasing" ? T2Dephasing(T_link) : Depolarization(T_link)

    # Network setup
    switch = Register(
        [Qubit() for _ in 1:n_nodes],
        [states_representation for _ in 1:n_nodes],
        # [noise_model for _ in 1:n_nodes]
    )
    clients = [
        Register(
            [Qubit()],
            [states_representation],
            # [noise_model]
        ) for _ in 1:n_nodes
    ]

    graph = star_graph(n_nodes + 1)
    net = RegisterNet(graph, [switch, clients...])

    sim = get_time_tracker(net)

    attempt_counter = Ref(0)
    global_pause = Ref(0.0)
    pair_times = Dict{Int,PairTimes}()

    @process switch_listener(sim, net, attempt_counter, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, cutoff, Δt_rotation_shuttle, global_pause, log_data, pair_times, singlezone)
    @process naive_entangler(sim, net, n_nodes, F_link, link_success_prob, attempt_t, Δt_rotation_shuttle)
    return sim
end


function get_cutoff(T_coherence, error_budget)
    return -T_coherence * log(1 - 4/3 * (1 - (1-error_budget)^0.25))
end

"""
    run_single_configuration(; ...)

Run exactly one physical/network parameter setting and one cutoff value.
No summary statistics are constructed. The returned `raw_events` DataFrame has
one row per successfully consumed GHZ state; see `RawGHZRow` for the columns.
Pairs are listed in fusion order (piecemaker first). Pass the rows to
`ghz_fidelities_from_log` (GHZfidelity_closedform.jl) to obtain fidelities.

The simulation stops at the GHZ state that completes round `target_rounds` of stabilizer
measurements (see `GHZLog`), or once `max_wallclock` seconds have passed, checked every
`add_runtime_batch` of simulated time; `converged` tells which.

`singlezone = true` gives the switch a single fusion zone, so fusions (and the readout of a
completed GHZ state) run one at a time in arrival order; `false` runs as many in parallel as
needed, limited only by the slot locks.
"""
function run_single_configuration(;
    F_link::Float64,
    link_success_prob::Float64,
    attempt_time::Float64,
    T_coherence::Float64,
    Δt_CNOTgate::Float64,
    gate_fidelity::Float64,
    Δt_readout::Float64,
    readout_fidelity::Float64,
    Δt_rotation_shuttle::Float64,
    cutoff::Float64,
    seed::Int = 1234,
    target_rounds::Int = 1000,
    max_wallclock::Float64 = 60.0,
    add_runtime_batch::Float64 = 0.1,
    singlezone::Bool = false,
)
    Random.seed!(seed)
    log_data = GHZLog(target_rounds)

    sim = prepare_sim(
        N_NODES,
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
        log_data,
        singlezone,
    )

    t_wallclock = 0.0

    # A batch returns early when the consumer throws StopSimulation at the target round.
    while log_data.n_rounds < target_rounds && t_wallclock <= max_wallclock
        t_wallclock += @elapsed run(sim, now(sim) + add_runtime_batch)
    end

    converged = log_data.n_rounds >= target_rounds
    if converged
        @info "Completed $(log_data.n_rounds) rounds after $(t_wallclock) seconds"
    else
        @warn "Reached maximum wallclock time $(max_wallclock) seconds after $(log_data.n_rounds) of $(target_rounds) rounds"
    end

    raw_events = DataFrame(log_data.rows)
    !isempty(raw_events) && sort!(raw_events, :timesteps)

    return (
        raw_events = raw_events,
        runtime = now(sim),
        wallclock_time = t_wallclock,
        n_rounds = log_data.n_rounds,
        converged = converged,
    )
end
