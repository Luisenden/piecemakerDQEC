using QuantumSavory
using QuantumSavory: Register, X, Z, Y, I, CNOT, Tag
using QuantumSavory.ProtocolZoo
using QuantumClifford
using QuantumClifford.ECC: Steane7, TableDecoder, decode
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames
using Statistics
using Plots

const ghzs = [ghz(n) for n in 1:10] # make const in order to not build new every time
const code = Steane7()
const decoder = TableDecoder(code)

ψ0 = StabilizerState("""
       ___XXXX
       _XX__XX
       X_X_X_X
       ___ZZZZ
       _ZZ__ZZ
       Z_Z_Z_Z
       ZZZZZZZ
       """)

# Pre-compute client to generator mapping for O(1) lookup
const CLIENT_TO_GENERATORS = Dict(
    1 => [3],
    2 => [2],
    3 => [2, 3],
    4 => [1],
    5 => [1, 3],
    6 => [1, 2],
    7 => [1, 2, 3]
)

# Encapsulate simulation state
mutable struct SimulationState
    progress::Vector{Vector{Vector{Int}}}
    to_be_measured::Vector{Vector{Int}}
    logs::Vector{Tuple{Float64, Vector{Int}, Float64, Bool}}
    syndrome_samples::Dict{PauliOperator, Vector{Int}}
end

function noisy_bell_state(target_fidelity::Float64=0.97)
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return target_fidelity*perfect_pair_dm + (1-target_fidelity)*mixed_dm
end

function majority(vec::Vector{Int})
    return sum(vec) > length(vec) ÷ 2 ? 1 : 0 # return 1 if majority is 1, else 0 (ties to false)
end

function get_pauli(key::Tuple{Vector{Int}, Bool})
    (genset, xtrue) = key
    xelem = [findall(xbit(stab)) for stab in stabs]
    zelem = [findall(zbit(stab)) for stab in stabs]

    if xtrue 
        index = findfirst(==(genset), xelem)
        return Stabilizer(code)[index]
    else
        index = findfirst(==(genset), zelem)
        return Stabilizer(code)[index]
    end
end

function correction_to_pauli(bits::AbstractVector{Bool})
    length(bits) % 2 != 0 && error("correction bits length must be even, got $(length(bits))")

    n = length(bits) ÷ 2
    x = bits[1:n]
    z = bits[n+1:end]
    sum(x) > 1 || sum(z) > 1 && error("More than 1 bit positive in correction!")

    for i in 1:n
        xi, zi = x[i], z[i]
        if xi || zi
            gate = xi && zi ? Y : (xi ? X : Z)
            return (gate, i)
        end
    end
end

@resumable function projectout(sim, net, slot_idx, gen_set_idx, n_clients_in_set, state::SimulationState)
    push!(state.to_be_measured, popfirst!(state.progress[gen_set_idx]))
    @yield lock(net[1][slot_idx])
    @debug "Projecting out piecemaker qubit at slot $(slot_idx), $(net[1][slot_idx])"
    res = Int(project_traceout!(net[1][slot_idx], σˣ))
    @debug "Tagging client $(slot_idx) with Z correction result $(res) for generator set $(gen_set_idx)"
    tag!(net[1+slot_idx][1], Tag(:updateZ, res, gen_set_idx, n_clients_in_set))
    unlock(net[1][slot_idx])
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, client_slot::RegRef)
    @yield lock(piecemaker_slot) & lock(client_slot)
    apply!((piecemaker_slot, client_slot), CNOT)
    res = Int(project_traceout!(client_slot, Z))
    tag!(net[1 + client_slot.idx][1], Tag(:updateX, res))
    unlock(piecemaker_slot)
    unlock(client_slot)
    @debug "Fused client $(client_slot.idx) with first client $(piecemaker_slot.idx)"
end

function get_oldest_generator_for_candidate(sim, net::RegisterNet, candidate::Int, state::SimulationState, steane_generators)
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
        @debug "Progress is empty, cannot find oldest generator for candidate $(candidate), return index $(potential_gens_indcs[1])"
        return now(sim), potential_gens_indcs[1]
    end

    # Optimized search: iterate through sorted clients, check if they're in any valid generator
    for oldest_idx in sorted_indices
        for gen_idx in potential_gens_indcs
            prog = state.progress[gen_idx]
            if !isempty(prog) && oldest_idx ∈ prog[1]
                @debug "Found generator index $(gen_idx) for candidate $(candidate) with oldest client index $(oldest_idx)"
                timestamp = accesstimes[oldest_idx]
                return timestamp, gen_idx
            end
        end
    end
    
    error("No generator found for candidate $(candidate), sorted_indices: $(sorted_indices)")
end

@resumable function GeneratorServiceProt(sim, net, candidate::Int, state::SimulationState, steane_generators, t_GHZ::Float64)
    notadded = true

    (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim, net, candidate, state, steane_generators)
    isempty(state.progress[idx_steane_take]) && push!(state.progress[idx_steane_take], Vector{Int}())

    while now(sim) - timestamp > t_GHZ
        @debug "Generator set $(idx_steane_take) with clients $(state.progress[idx_steane_take][1]) TOO OLD"
        @yield @process projectout(sim, net, state.progress[idx_steane_take][1][1], idx_steane_take, length(state.progress[idx_steane_take][1]), state)
        (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim, net, candidate, state, steane_generators)
    end

    for s in state.progress[idx_steane_take]
        if candidate ∉ s
            push!(s, candidate)
            if length(s) > 1
                @yield @process fusion(sim, net, net[1][s[1]], net[1][candidate])
                @debug "fusing candidate $(candidate) into generator set index $(idx_steane_take) with current starting index $(s[1])"
            end
            notadded = false
            if length(s) == length(steane_generators[idx_steane_take])
                @debug "Generator set $(idx_steane_take) completed with clients $(s), projecting out piecemaker qubit"
                @yield @process projectout(sim, net, s[1], idx_steane_take, length(s), state)
            end
            break
        end
    end

    if notadded
        # Use pre-computed mapping
        potential_gen_idcs = CLIENT_TO_GENERATORS[candidate]
        # Find generator with smallest progress
        min_length = typemax(Int)
        idx = potential_gen_idcs[1]
        for gen_idx in potential_gen_idcs
            len = length(state.progress[gen_idx])
            if len < min_length
                min_length = len
                idx = gen_idx
            end
        end
        push!(state.progress[idx], Vector{Int}([candidate]))
    end
end

@resumable function listen_fuse(sim, net, state::SimulationState, steane_generators, t_GHZ::Float64)
    while true
        @yield onchange_tag(net[1])
        
        while true
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if !isnothing(counterpart)
                slot, _, _ = counterpart
                @yield @process GeneratorServiceProt(sim, net, slot.idx, state, steane_generators, t_GHZ)
                @debug "Sorted client $(slot.idx) into generator sets, current progress: $(state.progress)"
            else
                break
            end
        end
    end
end

@resumable function listen_decode(sim, net, state::SimulationState, steane_generators)
    while true
        @yield onchange_tag(net[1])
        isdonemessage = querydelete!(net[1], :Zdone, ❓, ❓)

        if !isnothing(isdonemessage)
            genset = steane_generators[isdonemessage[3][2]]
            n_clients_in_set = isdonemessage[3][3]
            @debug "received Zdone tag: $(isdonemessage)"
            
            discarded = true
            fidelity = 0.0
            
            # Measure fidelity
            if length(genset) != n_clients_in_set
                @debug "MEASURE OUT BEFORE COMPLETION $(state.to_be_measured[1])"
                clients_to_measure = state.to_be_measured[1]
                @yield reduce(&, [lock(net[1+i][1]) for i in clients_to_measure])
                obs_proj = SProjector(StabilizerState(ghzs[length(clients_to_measure)]))
                fidelity = real(observable([net[1+i][1] for i in clients_to_measure], obs_proj))
                # Single cleanup - FIXED: removed double unlock
                for i in clients_to_measure
                    traceout!(net[1 + i][1])
                    unlock(net[1 + i][1])
                end
            else
                @yield reduce(&, [ [lock(net[1+i][1]) for i in genset]..., [lock(net[1+i][2]) for i in genset]...])
                obs_proj = SProjector(StabilizerState(ghzs[n_clients_in_set]))
                fidelity = real(observable([net[1+i][1] for i in genset], obs_proj))
                @info "clients serviced: $(genset) --> fidelity: $(fidelity)"
                # Single cleanup - FIXED: removed double unlock

                # measure syndrome
                rndn = rand(Bool)
                choice =  rndn ? CNOT : ZCZ
                @info "Applying choice gate $(choice) for syndrome measurement"
                for i in genset
                    apply!((net[1+i][1], net[1+i][2]), choice) # from ancilla to data
                end
                exp = 1 # qs convention for decoder -1 --> 1, 1 --> 0
                for i in genset
                    s_i_outcome = project_traceout!(net[1+i][1], X) - 1
                    exp += s_i_outcome
                    @info "Syndrome measurement for client $(i): $(s_i_outcome)"
                end
                syndrome = ((-1)^ exp + 1) ÷ 2
                pauli = get_pauli((genset, rndn))
                push!(state.syndrome_samples[pauli], syndrome)
                @info "Syndrome for generator set $(isdonemessage[3][2]) is $(syndrome), updated syndrome sample: $(state.syndrome_samples)"

                for i in genset
                    unlock(net[1 + i][1])
                    unlock(net[1 + i][2])
                end

                # Decode
                if !any(isempty(v) for v in values(state.syndrome_samples))
                    syndrome_sample = [majority(state.syndrome_samples[stab]) for stab in Stabilizer(code)]
                    @info "Decoding syndrome sample: $(syndrome_sample)"
                    correction = decode(decoder, syndrome_sample)
                    if !isnothing(correction) && sum(correction) > 0
                        ops = correction_to_pauli(correction)
                        @info "Decoded error correction: $(correction), applying: $(ops)"
                        @yield lock(net[1+ops[2]][2])
                        apply!(net[1+ops[2]][2], ops[1])
                        unlock(net[1+ops[2]][2])
                    else
                        @info "No correction from syndrome sample!"
                    end
                    for k in keys(state.syndrome_samples)
                        state.syndrome_samples[k] = []
                    end
                end
                discarded = false
            end
            
            timesteps = now(sim)
            push!(state.logs, (timesteps, popfirst!(state.to_be_measured), fidelity, discarded))
            @debug "Updated progress: $(state.progress)"
        end
    end
end

@resumable function correct_and_inform(sim, net::RegisterNet, client::Int)
    while true
        @yield onchange_tag(net[1+client][1])
        msg1 = querydelete!(net[1+client][1], :updateX, ❓)
        msg2 = querydelete!(net[1+client][1], :updateZ, ❓, ❓, ❓)
        
        if !isnothing(msg1) || !isnothing(msg2)
            if !isnothing(msg1)
                value = msg1[3][2]
                @debug "X received at client $(client), with value $(value)"
                @yield lock(net[1+client][1])
                if value == 2
                    apply!(net[1+client][1], X, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Xdone, client))
            end
            
            if !isnothing(msg2)
                @debug "Z received at client $(client)"
                value = msg2[3][2]
                gen_set_idx = msg2[3][3]
                n_clients_in_set = msg2[3][4]
                @debug "Z received at client $(client), with value $(value), gen_set_idx=$(gen_set_idx), n_clients_in_set=$(n_clients_in_set)"
                @yield lock(net[1+client][1])
                if value == 2
                    #noisyZ = NonInstantGate(Z, TXZ)
                    apply!(net[1+client][1], Z, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Zdone, gen_set_idx, n_clients_in_set))
            end
        end
    end
end

@resumable function init_data_qubits(sim, net)
    nmodules = (length(net.registers)-1)
    @yield reduce(&, [lock(net[1+i][2]) for i in 1:nmodules])
    initialize!([net[1+i] for i in 1:nmodules], [2 for _ in 1:nmodules], ψ0)
    for i in 1:nmodules
        unlock(net[1+i][2])
    end
end

function prepare_sim(n, T_link, t_GHZ::Float64, F_link::Float64, link_success_prob::Float64, steane_generators)
    states_representation = CliffordRepr()#QuantumOpticsRepr()
    noise_model = Depolarization(T_link)

    # Initialize simulation state
    state = SimulationState(
        [Vector{Vector{Int}}() for _ in 1:length(steane_generators)],
        Vector{Vector{Int}}(),
        Vector{Tuple{Float64, Vector{Int}, Float64, Bool}}(), 
        Dict([stab => Int64[] for stab in Stabilizer(code)]),
    )

    @info state.syndrome_samples

    # Network setup
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n])
    clients = [Register([Qubit(), Qubit()], [states_representation, states_representation], [noise_model, noise_model]) for _ in 1:n]

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    sim = get_time_tracker(net)

    @process init_data_qubits(sim, net)

    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1, 
            pairstate = noisy_bell_state(F_link),
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = 1.2e-6,
            retry_lock_time = 1e-7, local_busy_time_post = 0.0
        )

        @process entangler()
        @process correct_and_inform(sim, net, i)
    end

    @process listen_fuse(sim, net, state, steane_generators, t_GHZ)
    @process listen_decode(sim, net, state, steane_generators)

    return sim, state
end

# Main simulation, measures everything in seconds (s)
const steane_generators = [[4,5,6,7], [2,3,6,7], [1,3,5,7]]

n = 7

TCNOT = 0# 500e-6 # 500 microseconds
TXZ = 0#1e-6 # 1 microsecondx
cutoff_time= Inf # s

dataframes = DataFrame[]
link_success_prob = 2e-4
T_coherence = Inf # 0.01
F_link = 1.0# 0.941

runtime = 0.2 # 1e-4/link_success_prob
sim, state = prepare_sim(n, T_coherence, cutoff_time, F_link, link_success_prob, steane_generators)
t_wallclock = @elapsed run(sim, runtime)

logs = DataFrame(state.logs, [:timesteps, :clients_serviced, :GHZfidel, :discarded])
logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)
logs = transform(logs, :clients_serviced => (x -> [length(c) for c in x]) => :num_clients)

logs[!, "link_success_prob"] .= link_success_prob
logs[!, "runtime"] .= runtime
logs[!, "TCNOT"] .= TCNOT
logs[!, "T_coherence"] .= T_coherence
logs[!, "F_link"] .= F_link
logs[!, "cutoff"] .= cutoff_time
logs[!, "wallclock_time"] .= t_wallclock

push!(dataframes, logs)
@info "completed simulation for link_success_prob=$(link_success_prob), T_CNOT=$(TCNOT), T_coherence=$(T_coherence), F_link=$(F_link), cutoff=$(cutoff_time), wallclock=$(t_wallclock)s, collected $(nrow(logs)) logs"

alllogs = vcat(dataframes...)
@debug "Summary statistics:"
@debug describe(alllogs)