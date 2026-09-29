using QuantumSavory
using QuantumSavory: Register, X, Y, Z, CNOT
using QuantumClifford
using Random

function noisy_bell_state(target_fidelity::Float64 = 0.97)
    λ = (4 * target_fidelity - 1) / 3

    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(perfect_pair_dm)

    return λ * perfect_pair_dm + (1 - λ) * mixed_dm
end


const PAULIS = (nothing, X, Y, Z)

function sample_depol2q(
    rng::AbstractRNG,
    gate_fidelity::Float64,
)
    λ = 4 / 3 * (1 - gate_fidelity)

    rand(rng) < (1 - λ) && return (nothing, nothing)

    return (
        rand(rng, PAULIS),
        rand(rng, PAULIS),
    )
end


# ------------------------------------------------------------
# Reconstruct one weight-4 Piecemaker GHZ from four birth times
# ------------------------------------------------------------

function simulate_piecemaker_trace(
    birth_times::AbstractVector{<:Real};

    T_coherence::Float64 = 1.0,
    F_link::Float64 = 0.97,

    t_rotation_shuttle::Float64 = 0.0,

    t_CNOT::Float64 = 100e-6,
    F_CNOT::Float64 = 0.9997,

    t_readout::Float64 = 1e-3,
    F_readout::Float64 = 1.0,

    rng::AbstractRNG = Random.default_rng(),
)

    @assert length(birth_times) == 4

    # --------------------------------------------------------
    # 1. Sort pairs by arrival time
    # --------------------------------------------------------

    arrival_order = sortperm(birth_times)

    times = Float64.(birth_times[arrival_order])

    # Only time differences matter.
    times .-= times[1]

    # --------------------------------------------------------
    # 2. Make the two 4-slot registers
    #
    # switch[i]  = switch half of Bell pair i
    # remote[i]  = remote-node half
    #
    # Indices below refer to ARRIVAL ORDER.
    # --------------------------------------------------------

    repr = QuantumOpticsRepr()
    background = Depolarization(T_coherence)

    switch = Register(
        [Qubit() for _ in 1:4],
        [repr for _ in 1:4],
        [background for _ in 1:4],
    )

    remote = Register(
        [Qubit() for _ in 1:4],
        [repr for _ in 1:4],
        [background for _ in 1:4],
    )

    pair_state = noisy_bell_state(F_link)

    # --------------------------------------------------------
    # 3. First Bell pair becomes the Piecemaker
    # --------------------------------------------------------

    initialize!(
        (switch[1], remote[1]),
        pair_state;
        time = times[1],
    )

    # Your full protocol spends this time accepting /
    # rotating / shuttling the incoming pair before using it.
    current_time = times[1] + t_rotation_shuttle


    # --------------------------------------------------------
    # 4. Bell pairs 2, 3, 4 arrive and are fused sequentially
    # --------------------------------------------------------

    for k in 2:4

        t_birth = times[k]

        # Create Bell pair at its ACTUAL recorded birth time.
        initialize!(
            (switch[k], remote[k]),
            pair_state;
            time = t_birth,
        )

        # If it arrived while the switch was busy, it waits.
        t_start = max(current_time, t_birth)

        # Rotation / shuttle before fusion
        t_fusion = t_start + t_rotation_shuttle

        # ----------------------------------------------------
        # Piecemaker fusion:
        #
        # CNOT:
        #     switch[1] -> switch[k]
        # ----------------------------------------------------

        apply!(
            (switch[1], switch[k]),
            CNOT;
            time = t_fusion,
        )

        t_gate_end = t_fusion + t_CNOT

        # Explicitly evolve both CNOT qubits through gate time.
        uptotime!(
            (switch[1], switch[k]),
            t_gate_end,
        )

        # ----------------------------------------------------
        # Noisy CNOT, same stochastic model as current code
        # ----------------------------------------------------

        gate_error = sample_depol2q(rng, F_CNOT)

        if !isnothing(gate_error[1])
            apply!(
                switch[1],
                gate_error[1];
                time = t_gate_end,
            )
        end

        if !isnothing(gate_error[2])
            apply!(
                switch[k],
                gate_error[2];
                time = t_gate_end,
            )
        end

        # ----------------------------------------------------
        # Measure incoming switch qubit in Z
        #
        # Explicit basis ensures an outcome 1 or 2.
        # ----------------------------------------------------

        res = project_traceout!(
            switch[k],
            (Z1, Z2);
            time = t_gate_end,
        )

        # Readout error
        if rand(rng) > F_readout
            res = 3 - res
        end

        # Classical readout time
        t_correction = t_gate_end + t_readout

        # Feed-forward X on corresponding remote qubit
        if res == 2
            apply!(
                remote[k],
                X;
                time = t_correction,
            )
        end

        current_time = t_correction
    end


    # --------------------------------------------------------
    # 5. Measure Piecemaker qubit in X
    # --------------------------------------------------------

    res = project_traceout!(
        switch[1],
        (X1, X2);
        time = current_time,
    )

    if rand(rng) > F_readout
        res = 3 - res
    end

    # This placement matches your CURRENT consumer():
    # the Z correction happens before the final readout timeout.
    if res == 2
        apply!(
            remote[1],
            Z;
            time = current_time,
        )
    end

    final_time = current_time + t_readout


    # --------------------------------------------------------
    # 6. GHZ fidelity
    #
    # observable(...; time=final_time) automatically evolves
    # ALL four surviving remote qubits to final_time.
    # --------------------------------------------------------

    ghz4_projector =
        SProjector(StabilizerState(ghz(4)))

    remote_refs = (
        remote[1],
        remote[2],
        remote[3],
        remote[4],
    )

    fidelity = real(
        observable(
            remote_refs,
            ghz4_projector;
            time = final_time,
        )
    )

    # StateRef for inspection if desired
    # state = QuantumSavory.stateof(remote[1])

    return fidelity
    # return (
    #     fidelity = fidelity,
    #     state = state,

    #     remote = remote,

    #     birth_times = times,
    #     gaps = diff(times),

    #     arrival_order = arrival_order,

    #     final_time = final_time,
    # )
end