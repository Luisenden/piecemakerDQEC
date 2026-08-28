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
target_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
global_idx = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
output_path = length(ARGS) >= 5 ? ARGS[5] : "./"
max_wallclock = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 3600.0/2.0 # seconds (1/2 hour)

const codes = Dict(

    "Steane7" => (7, [
        [1, 2, 3, 5],
        [1, 2, 4, 6],
        [2, 3, 4, 7],
    ]),

    # [[12,2,3]] bivariate-bicycle code: 10 independent weight-4 generators
    "BB12_2_3" => (12, [
        [3, 4, 7, 8],
        [1, 5, 8, 9],
        [2, 6, 7, 9],
        [1, 6, 10, 11],
        [2, 4, 11, 12],

        [1, 3, 8, 10],
        [1, 2, 9, 11],
        [2, 3, 7, 12],
        [4, 6, 7, 11],
        [4, 5, 8, 12],
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

# Fixed data-qubit partitions for the codes that use multiple data qubits per node.
# Each inner vector contains the data qubits hosted by one remote node.
# Codes without an explicit entry fall back to one data qubit per node.
const NODE_PARTITIONS = Dict(
    # Steane keeps the original architecture: one data qubit / one communication node.
    # The three entries in GENERATORS are the three distinct support patterns;
    # X- and Z-type checks share these supports. For data-qubit waiting times, every
    # completed GHZ on a support is one stabilizer-measurement opportunity for each
    # data qubit in that support, independent of whether it is used for X or Z.
    "Steane7" => [[q] for q in 1:7],

    # 6 nodes, two data qubits per node. Stabilizer loads: [7, 6, 7, 7, 6, 7].
    "BB12_2_3" => [
        [1, 12],
        [2, 10],
        [3, 11],
        [4, 9],
        [5, 7],
        [6, 8],
    ],

    # 12 nodes. Every weight-4 generator spans four distinct nodes.
    # Stabilizer loads: [9, 9, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8].
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

const N_DATA = codes[code][1]
const GENERATORS = codes[code][2]
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
@inline function sample_depol2q(gate_fidelity::Float64)
    λ = 4/3 * (1-gate_fidelity)
    rand() < (1-λ) && return (nothing, nothing)
    return (rand(paulis), rand(paulis))
end

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

const LogRow = Tuple{Float64, Int, Float64}

# QEC-layer lookup: which stabilizer supports contain each data qubit.
# This remains defined in terms of the ORIGINAL data-qubit supports, even when
# several data qubits are hosted by the same physical node.
const DATA_QUBIT_TO_GENERATORS = Dict(
    q => findall(generator -> q in generator, GENERATORS)
    for q in 1:N_DATA
)

"""
    inter_event_times(times)

Return waiting times between consecutive events, including the initial waiting
interval from simulation time 0 to the first event. This matches the convention
used previously for generator-specific `mean_generation_time`.
"""
function inter_event_times(times::AbstractVector{<:Real})
    isempty(times) && return Float64[]
    ts = sort(Float64.(times))
    return [first(ts); diff(ts)]
end

safe_mean(x) = isempty(x) ? NaN : mean(x)
safe_std(x) = length(x) <= 1 ? NaN : std(x)
safe_sem(x) = length(x) <= 1 ? NaN : std(x) / sqrt(length(x))
safe_rate(times) = isempty(times) || maximum(times) <= 0 ? NaN : length(times) / maximum(times)

"""
    data_qubit_waiting_summary(completion_logs)

Construct the exact stabilizer-measurement event stream seen by every DATA QUBIT
from the global GHZ completion log. A GHZ completion for generator `g` counts as
an event for every q in `GENERATORS[g]`.

For a partitioned architecture this is intentionally NOT the same as the node
stream: a node may host several data qubits, but a valid partition guarantees
that a given stabilizer touches at most one of them.
"""
function data_qubit_waiting_summary(completion_logs::DataFrame)
    rows = NamedTuple[]

    for q in 1:N_DATA
        generator_indices = DATA_QUBIT_TO_GENERATORS[q]
        mask = in.(completion_logs.generator_idx, Ref(generator_indices))
        times = collect(completion_logs.timesteps[mask])
        fidelities = collect(completion_logs.GHZfidel[mask])
        waits = inter_event_times(times)

        push!(rows, (
            data_qubit_idx = q,
            node_idx = DATA_TO_NODE[q],
            n_generators = length(generator_indices),
            generator_indices = copy(generator_indices),
            mean_inter_measurement_time = safe_mean(waits),
            std_inter_measurement_time = safe_std(waits),
            sem_inter_measurement_time = safe_sem(waits),
            mean_measurement_rate = safe_rate(times),
            n_measurements = length(times),
            mean_GHZfidel = safe_mean(fidelities),
            std_GHZfidel = safe_std(fidelities),
            sem_GHZfidel = safe_sem(fidelities),
        ))
    end

    return DataFrame(rows)
end

"""
    node_waiting_summary(completion_logs)

Construct the GHZ inter-completion stream seen by every PHYSICAL NODE. A GHZ
completion for generator `g` counts as an event for every node in
`GENERATOR_TO_NODES[g]`.

With one communication qubit per node this quantity measures communication-qubit
service/load. If a node hosts multiple data qubits, its node stream is the union
of the measurement streams of those local data qubits.
"""
function node_waiting_summary(completion_logs::DataFrame)
    rows = NamedTuple[]

    for node_idx in 1:N_NODES
        generator_indices = NODE_TO_GENERATORS[node_idx]
        mask = in.(completion_logs.generator_idx, Ref(generator_indices))
        times = collect(completion_logs.timesteps[mask])
        fidelities = collect(completion_logs.GHZfidel[mask])
        waits = inter_event_times(times)

        push!(rows, (
            node_idx = node_idx,
            data_qubits = copy(NODE_PARTITION[node_idx]),
            n_data_qubits = length(NODE_PARTITION[node_idx]),
            n_generators = length(generator_indices),
            generator_indices = copy(generator_indices),
            mean_inter_completion_time = safe_mean(waits),
            std_inter_completion_time = safe_std(waits),
            sem_inter_completion_time = safe_sem(waits),
            mean_completion_rate = safe_rate(times),
            n_completions = length(times),
            mean_GHZfidel = safe_mean(fidelities),
            std_GHZfidel = safe_std(fidelities),
            sem_GHZfidel = safe_sem(fidelities),
        ))
    end

    return DataFrame(rows)
end

# Attach the hardware/run parameters to any per-run summary DataFrame.
function add_run_metadata!(
    df::DataFrame;
    link_success_prob,
    attempt_time,
    T_coherence,
    F_link,
    error_model,
    cutoff,
    Δt_rotation_shuttle,
    Δt_CNOTgate,
    gate_fidelity,
    Δt_readout,
    readout_fidelity,
    runtime,
    wallclock_time,
)
    df[!, :link_success_prob] .= link_success_prob
    df[!, :attempt_time] .= attempt_time
    df[!, :T_coherence] .= T_coherence
    df[!, :F_link] .= F_link
    df[!, :error_model] .= error_model
    df[!, :cutoff] .= cutoff
    df[!, :tRotationShuttle] .= Δt_rotation_shuttle
    df[!, :tCNOT] .= Δt_CNOTgate
    df[!, :gate_fidelity] .= gate_fidelity
    df[!, :tReadout] .= Δt_readout
    df[!, :readout_fidelity] .= readout_fidelity
    df[!, :runtime] .= runtime
    df[!, :wallclock_time] .= wallclock_time
    df[!, :seed] .= seed
    df[!, :global_idx] .= global_idx
    df[!, :n_data] .= N_DATA
    df[!, :n_nodes] .= N_NODES
    df[!, :node_partition] .= string(NODE_PARTITION)
    return df
end

@info "Using code $(code) with $(N_DATA) data qubits on $(N_NODES) physical nodes; node partition = $(NODE_PARTITION)"

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
    
    expected_weight = length(GENERATOR_TO_NODES[gen_set_index])
    !iscutoff && @assert length(msgs) == expected_weight "Expected to find all $(expected_weight) node slots of complete generator set for piecemaker slot $(pm_slot.idx) but received: $(msgs)"

    pm_msg = query(net[1][pm_slot.idx], :isPiecemaker, ❓; assigned = true, filo = false)
    @assert !isnothing(pm_msg) "Expected piecemaker slot $(pm_slot.idx) to be tagged with isPiecemaker but it is not!"
    
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
    # and receives tag isPiecemaker; in addition it receives a generator set tag PartOfGenSet uniformly at random
    # from the generators whose NODE SUPPORT contains the current switch slot / physical node.

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
    
    if isempty(pmslot_msgs)
        # there is no piecemaker slot in any of the potentially common generator sets, this slot needs therefore to become a piecemaker and start a new GHZ attempt
        @yield lock(net[1][switchslot_idx])
        attempt_id = next_attempt_id!(attempt_counter)
        tag!(net[1][switchslot_idx], Tag(:isPiecemaker, attempt_id))
        gen_set_idx = rand(NODE_TO_GENERATORS[switchslot_idx])
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
            gen_set_idx = rand(NODE_TO_GENERATORS[switchslot_idx])
            tag!(net[1][switchslot_idx], Tag(:PartOfGenSet, gen_set_idx, switchslot_idx, tagid))
            @debug "Tagging switch slot idx $(switchslot_idx) as PIECEMAKER and part of $(gen_set_idx)"
            unlock(net[1][switchslot_idx])
        else
            # in this case we found a piecemaker slot within a suitable generator set and fuse with it
            @yield @process fusion(sim, net, pmslot_to_fuse_with, net[1][switchslot_idx], net[1+switchslot_idx][clientslot_idx], gen_set_idx, tagid, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity)

            # check if after fusion the GHZ state contains all nodes required by this generator
            msgs = queryall(net[1], :PartOfGenSet, gen_set_idx, pmslot_to_fuse_with.idx, ❓; filo = false)
            if length(msgs) == length(GENERATOR_TO_NODES[gen_set_idx])
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

            pairstate = noisy_bell_state(F_link),

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
    error_model::String, cutoff::Float64, Δt_rotation_shuttle::Float64, log_data::Vector{LogRow})


    states_representation = QuantumOpticsRepr()
    @debug "Preparing simulation with parameters: n_nodes=$(n_nodes), T_link=$(T_link), cutoff=$(cutoff), F_link=$(F_link), link_success_prob=$(link_success_prob), attempt_time=$(attempt_t)"
    
    noise_model = error_model == "dephasing" ? T2Dephasing(T_link) : Depolarization(T_link)

    # Network setup
    switch = Register(
        [Qubit() for _ in 1:n_nodes],
        [states_representation for _ in 1:n_nodes],
        [noise_model for _ in 1:n_nodes]
    )
    clients = [
        Register(
            [Qubit()],
            [states_representation],
            [noise_model]
        ) for _ in 1:n_nodes
    ]

    graph = star_graph(n_nodes + 1)
    net = RegisterNet(graph, [switch, clients...])

    sim = get_time_tracker(net)

    attempt_counter = Ref(0)
    global_pause = Ref(0.0)

    @process switch_listener(sim, net, attempt_counter, Δt_CNOTgate, gate_fidelity, Δt_readout, readout_fidelity, cutoff, Δt_rotation_shuttle, global_pause, log_data)
    @process naive_entangler(sim, net, n_nodes, F_link, link_success_prob, attempt_t, Δt_rotation_shuttle)
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
function get_cutoff(T_coherence, error_budget)
    return -T_coherence * log(1 - 4/3 * ( 1 - (1-error_budget)^0.25 ))
end

function check_convergence(
    log_data,
    generators;
    min_samples = 1000,
)
    isempty(log_data) && return false

    times  = first.(log_data)
    genidx = getindex.(log_data, 2)

    n_data = maximum(vcat(generators...))

    # Every data qubit needs min_samples inter-measurement intervals
    for q in 1:n_data
        relevant_gens = findall(g -> q in g, generators)

        n_measurements = count(g -> g in relevant_gens, genidx)
        n_waits = max(n_measurements - 1, 0)

        n_waits < min_samples && return false
    end

    # Every stabilizer needs min_samples GHZ fidelity samples
    for g in eachindex(generators)
        count(==(g), genidx) < min_samples && return false
    end

    return true
end

function run_sweep(F_link, link_success_prob)

    # One PBS array index selects one of the 12,960 local-operation parameter combinations.
    attempt_time = parameter_combinations[global_idx][1]
    T_coherence = parameter_combinations[global_idx][2]
    Δt_CNOTgate = parameter_combinations[global_idx][3]
    gate_fidelity = parameter_combinations[global_idx][4]
    Δt_readout = parameter_combinations[global_idx][5]
    readout_fidelity = parameter_combinations[global_idx][6]
    Δt_rotation_shuttle = parameter_combinations[global_idx][7]

    @debug "Running sweep with parameters: attempt_time=$(attempt_time), link_success_prob=$(link_success_prob), T_coherence=$(T_coherence), F_link=$(F_link), gate_fidelity=$(gate_fidelity), Δt_readout=$(Δt_readout), readout_fidelity=$(readout_fidelity)"

    log_data = LogRow[]
    generator_dataframes = DataFrame[]
    data_qubit_dataframes = DataFrame[]
    node_dataframes = DataFrame[]

    add_runtime_batch = 0.1

    for cutoff in [get_cutoff(T_coherence, 0.1), get_cutoff(T_coherence, 0.25), Inf] # 3
        Random.seed!(seed)

        empty!(log_data)

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
            log_data
        )

        t_wallclock = 0.0
        runtime = 0.0
        while true
            t_wallclock += @elapsed run(sim, now(sim) + add_runtime_batch)
            runtime += add_runtime_batch

            if check_convergence(
                log_data,
                codes[code][2];
                min_samples = target_samples,
            )
                @info "Reached convergence"
                break
            end

            if t_wallclock > max_wallclock
                @warn "Reached maximum wallclock time $(max_wallclock) seconds without convergence"
                break
            end
        end

        # ------------------------------------------------------------------
        # RAW GHZ COMPLETION EVENTS
        # ------------------------------------------------------------------
        # One row = one successfully completed GHZ resource. `generator_idx`
        # still refers to the ORIGINAL DATA-QUBIT stabilizer support.
        completion_logs = DataFrame(
            log_data,
            [
                :timesteps,
                :generator_idx,
                :GHZfidel,
            ],
        )

        sort!(completion_logs, :timesteps)

        # ------------------------------------------------------------------
        # 1) ORIGINAL PER-GENERATOR STATISTICS
        # ------------------------------------------------------------------
        generator_logs = combine(groupby(completion_logs, :generator_idx),
            :GHZfidel => mean => :mean_GHZfidel,
            :GHZfidel => std => :std_GHZfidel,
            :GHZfidel => (x -> std(x)/sqrt(length(x))) => :sem_GHZfidel,
            :timesteps => (x -> length(x)/maximum(x)) => :mean_rate,
            :timesteps => (x -> mean([first(x); diff(x)])) => :mean_generation_time,
            :timesteps => (x -> std([first(x); diff(x)])) => :std_generation_time,
            :timesteps => (x -> std([first(x); diff(x)])/sqrt(length(x))) => :sem_generation_time,
            nrow => :nlogs,
        )

        # ------------------------------------------------------------------
        # 2) PER-DATA-QUBIT INTER-MEASUREMENT STATISTICS
        # ------------------------------------------------------------------
        # For q, merge completion events from ALL generators containing q.
        # This is the quantity to use for a data-qubit memory waiting time.
        data_qubit_logs = data_qubit_waiting_summary(completion_logs)

        # ------------------------------------------------------------------
        # 3) PER-NODE INTER-COMPLETION STATISTICS
        # ------------------------------------------------------------------
        # For node m, merge completion events from ALL generators whose node
        # support contains m. This characterizes communication-qubit workload.
        node_logs = node_waiting_summary(completion_logs)

        add_run_metadata!(generator_logs;
            link_success_prob=link_success_prob,
            attempt_time=attempt_time,
            T_coherence=T_coherence,
            F_link=F_link,
            error_model=error_model,
            cutoff=cutoff,
            Δt_rotation_shuttle=Δt_rotation_shuttle,
            Δt_CNOTgate=Δt_CNOTgate,
            gate_fidelity=gate_fidelity,
            Δt_readout=Δt_readout,
            readout_fidelity=readout_fidelity,
            runtime=runtime,
            wallclock_time=t_wallclock,
        )

        add_run_metadata!(data_qubit_logs;
            link_success_prob=link_success_prob,
            attempt_time=attempt_time,
            T_coherence=T_coherence,
            F_link=F_link,
            error_model=error_model,
            cutoff=cutoff,
            Δt_rotation_shuttle=Δt_rotation_shuttle,
            Δt_CNOTgate=Δt_CNOTgate,
            gate_fidelity=gate_fidelity,
            Δt_readout=Δt_readout,
            readout_fidelity=readout_fidelity,
            runtime=runtime,
            wallclock_time=t_wallclock,
        )

        add_run_metadata!(node_logs;
            link_success_prob=link_success_prob,
            attempt_time=attempt_time,
            T_coherence=T_coherence,
            F_link=F_link,
            error_model=error_model,
            cutoff=cutoff,
            Δt_rotation_shuttle=Δt_rotation_shuttle,
            Δt_CNOTgate=Δt_CNOTgate,
            gate_fidelity=gate_fidelity,
            Δt_readout=Δt_readout,
            readout_fidelity=readout_fidelity,
            runtime=runtime,
            wallclock_time=t_wallclock,
        )

        push!(generator_dataframes, generator_logs)
        push!(data_qubit_dataframes, data_qubit_logs)
        push!(node_dataframes, node_logs)

        @info "cutoff: $cutoff, runtime: $runtime, wallclock_time: $t_wallclock"
    end

    return (
        generator_summary = vcat(generator_dataframes...),
        data_qubit_summary = vcat(data_qubit_dataframes...),
        node_summary = vcat(node_dataframes...),
    )
end

##

# Keep the original `df_out` name for backwards compatibility. New outputs are
# saved alongside it for the per-data-qubit and per-node waiting-time analysis.
generator_dfs = DataFrame[]
data_qubit_dfs = DataFrame[]
node_dfs = DataFrame[]

for link_success_prob in [1e-4]#[10.0^(-x) for x in 1.0:5.0]
    for F_link in [0.99]#[1.0 - 2.5^(-x) for x in 3.0:10.0]
        result = run_sweep(F_link, link_success_prob)

        push!(generator_dfs, result.generator_summary)
        push!(data_qubit_dfs, result.data_qubit_summary)
        push!(node_dfs, result.node_summary)

        @info "Completed sweep for F_link=$(F_link), link_success_prob=$(link_success_prob)"
    end
end

df_out = vcat(generator_dfs...)
df_data_qubit_out = vcat(data_qubit_dfs...)
df_node_out = vcat(node_dfs...)

##
# `df_out`:                original per-generator GHZ statistics
# `df_data_qubit_out`:     exact per-data-qubit inter-measurement statistics
# `df_node_out`:           exact per-physical-node GHZ inter-completion statistics
# Raw GHZ completion streams are intentionally not persisted; they are only
# held temporarily within each run to construct the summary statistics above.
@save "$(output_path)/summary_ghz_service_v1_$(code)_$(error_model)_$(global_idx).jld2" df_out df_data_qubit_out df_node_out
