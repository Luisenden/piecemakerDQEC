# see tutorial at https://qc.quantumsavory.org/stable/ECC_evaluating/

# this script estimates the logical X- and Z-error probabilities of quantum error-correcting codes under a noisy memory and noisy Shor-style syndrome-extraction circuit.
# for each code and each memory-error probability, it performs Monte Carlo circuit (shor syndrome extraction) simulations

using Plots
using Colors
using QuantumClifford
using LaTeXStrings
using Measures

using QuantumClifford
using QuantumClifford.ECC

import QuantumClifford: applynoise!

const BB12_2_3 = S"""ZIZIIIIZIZII
ZZIIIIIIZIZI
IZZIIIZIIIIZ
IIIZIZZIIIZI
IIIZZIIZIIIZ
IIIIZZIIZZII
IIXXIIXXIIII
XIIIXIIXXIII
IXIIIXXIXIII
XIIIIXIIIXXI
IXIXIIIIIIXX
IIXIXIIIIXIX"""

function applynoise!(
    frame::QuantumClifford.PauliFrame,
    noise::QuantumClifford.DepolarizationNoise,
    indices::Base.AbstractVecOrTuple,
)
    qubits = Tuple(indices)
    n = length(qubits)

    n == 0 && return frame

    # Tableau storing the X and Z components of every trajectory.
    xzs = QuantumClifford.tab(frame.frame).xzs

    # Precompute the packed-bit location of every affected qubit.
    bit_locations = map(qubits) do q
        _, ibig, _, bitmask =
            QuantumClifford.get_bitmask_idxs(xzs, q)

        return ibig, bitmask
    end

    n_paulis = 4^n

    @inbounds for trajectory in eachindex(frame)

        # With probability 1-λ, leave this trajectory unchanged.
        rand() < noise.p || continue

        # Jointly sample one n-qubit Pauli.
        #
        # digit = 0: I
        # digit = 1: X
        # digit = 2: Z
        # digit = 3: Y
        pauli_index = rand(0:n_paulis-1)

        for (ibig, bitmask) in bit_locations
            pauli_index, digit = divrem(pauli_index, 4)

            if digit == 1
                # X
                xzs[ibig, trajectory] ⊻= bitmask

            elseif digit == 2
                # Z
                xzs[end ÷ 2 + ibig, trajectory] ⊻= bitmask

            elseif digit == 3
                # Y = XZ, up to an irrelevant phase
                xzs[ibig, trajectory] ⊻= bitmask
                xzs[end ÷ 2 + ibig, trajectory] ⊻= bitmask
            end
        end
    end

    return frame
end

##
# find the source code for the ShorSyndromeECCSetup struct in the QuantumClifford.ECC module https://raw.githubusercontent.com/QuantumSavory/QuantumClifford.jl/master/src/ecc/decoder_pipeline.jl
using QuantumClifford.ECC: AbstractECCSetup, AbstractSyndromeDecoder

function add_werner_GHZ_noise(H, F_GHZ)
    n_data = nqubits(H)
    noisy_GHZ_circ = QuantumClifford.AbstractOperation[]

    # shor_syndrome_circuit places the first ancilla at n_data + 1.
    next_ancilla = n_data + 1

    for check in H
        # One GHZ ancilla qubit is used for every non-identity
        # location in this stabilizer check.
        n_a = sum(
            check[q] != (false, false)
            for q in 1:n_data
        )

        n_a == 0 && continue

        d = 2.0^n_a

        # Dλ(|GHZ><GHZ|) has fidelity F_GHZ.
        λ = d * (1 - F_GHZ) / (d - 1)

        0 <= λ <= 1 ||
            throw(DomainError(
                λ,
                "F_GHZ must satisfy F_GHZ ≥ 1/2^n_a.",
            ))

        ancilla_indices =
            collect(next_ancilla : next_ancilla + n_a - 1)

        push!(
            noisy_GHZ_circ,
            NoiseOp(
                DepolarizationNoise(λ),
                ancilla_indices,
            ),
        )

        next_ancilla += n_a
    end

    return noisy_GHZ_circ
end

struct CShorSyndromeECCSetup <: AbstractECCSetup
    mem_noise::Float64
    two_qubit_gate_fidelity::Float64
    F_GHZ::Float64

    function CShorSyndromeECCSetup(
        mem_noise,
        two_qubit_gate_fidelity,
        F_GHZ,
    )
        0 <= mem_noise <= 1 ||
            throw(DomainError(mem_noise, "mem_noise must be between 0 and 1."))

        0 <= two_qubit_gate_fidelity <= 1 ||
            throw(DomainError(
                two_qubit_gate_fidelity,
                "two_qubit_gate_fidelity must be between 0 and 1.",
            ))

        0 <= F_GHZ <= 1 ||
            throw(DomainError(F_GHZ, "F_GHZ must be between 0 and 1."))

        new(mem_noise, two_qubit_gate_fidelity, F_GHZ)
    end
end

function add_two_qubit_gate_fidelity(g, gate_error)
    return ()
end

function add_two_qubit_gate_fidelity(g::AbstractTwoQubitOperator, F_gate)
    qubits = affectedqubits(g)

    λ_gate = 4 * (1 - F_gate) / 3

    return (
        NoiseOp(
            DepolarizationNoise(λ_gate),
            collect(qubits),
        ),
    )
end

function physical_ECC_circuit(
    H,
    setup::CShorSyndromeECCSetup,
)
    prep_anc, syndrome_circ, n_anc, syndrome_bits =
        shor_syndrome_circuit(H)

    noisy_syndrome_circ = QuantumClifford.AbstractOperation[]

    for op in syndrome_circ
        push!(noisy_syndrome_circ, op)

        for noise_op in add_two_qubit_gate_fidelity(
            op,
            setup.two_qubit_gate_fidelity,
        )
            push!(noisy_syndrome_circ, noise_op)
        end
    end

    mem_error_circ = [
        PauliError(i, setup.mem_noise)
        for i in 1:nqubits(H)
    ]

    werner_GHZ_circ =
        add_werner_GHZ_noise(H, setup.F_GHZ)

    circ = vcat(
        prep_anc,
        mem_error_circ,
        werner_GHZ_circ,
        noisy_syndrome_circ,
    )

    return circ, syndrome_bits, n_anc
end

function cevaluate_decoder(
    d::AbstractSyndromeDecoder,
    setup::AbstractECCSetup,
    nsamples::Int,
)
    H = parity_checks(d)

    n = code_n(H)
    k = code_k(H)

    # Matrix mapping physical correction guesses to logical faults
    O = faults_matrix(H)

    # Build the noisy ECC circuit for the chosen setup,
    # e.g. ShorSyndromeECCSetup or NaiveSyndromeECCSetup.
    physical_noisy_circ, syndrome_bits, n_anc = physical_ECC_circuit(H, setup)

    # Perfect encoding circuit
    encoding_circ = naive_encoding_circuit(H)

    # Used for testing logical Z failures by preparing/testing in X basis
    preX = sHadamard[sHadamard(i) for i in n-k+1:n]

    mdH = MixedDestabilizer(H)

    # Circuits that noiselessly measure logical X and logical Z observables
    logX_circ, _, logX_bits = naive_syndrome_circuit(
        logicalxview(mdH),
        n_anc + 1,
        last(syndrome_bits) + 1,
    )

    logZ_circ, _, logZ_bits = naive_syndrome_circuit(
        logicalzview(mdH),
        n_anc + 1,
        last(syndrome_bits) + 1,
    )

    # Logical X error:
    # run encoding + noisy ECC + logical Z measurement
    X_error = evaluate_decoder(
        d,
        nsamples,
        vcat(encoding_circ, physical_noisy_circ, logZ_circ),
        syndrome_bits,
        logZ_bits,
        O[end÷2+1:end, :],
    )

    # Logical Z error:
    # prepare in X basis, then run encoding + noisy ECC + logical X measurement
    Z_error = evaluate_decoder(
        d,
        nsamples,
        vcat(preX, encoding_circ, physical_noisy_circ, logX_circ),
        syndrome_bits,
        logX_bits,
        O[1:end÷2, :],
    )

    return X_error, Z_error
end

function cevaluate_decoder_pL(
    d::AbstractSyndromeDecoder,
    setup::AbstractECCSetup,
    nsamples::Int,
)
    H = parity_checks(d)
    n = code_n(H)
    O = faults_matrix(H)

     
    fmtab = QuantumClifford.Tableau( # this is from CommutationCheckECCSetup
        O[:, end÷2+1:end],
        O[:, 1:end÷2],
    )

    physical_noisy_circ, syndrome_bits, n_anc = # this is new
        physical_ECC_circuit(H, setup)


    n_total_qubits = n + n_anc
    n_total_bits = last(syndrome_bits)

    frames = PauliFrame(
        nsamples,
        n_total_qubits,
        n_total_bits,
    )

    fill!(QuantumClifford.tab(frames).xzs, 0) # physical error = I

    pftrajectories(
        frames,
        physical_noisy_circ,
    )

    syndromes = @view measurements(frames)[:, syndrome_bits]

    n_logical_faults = size(O, 1)

    measured_faults = zeros(UInt8, nsamples, n_logical_faults)
    frame_tableau = QuantumClifford.tab(frames) # frame to tableau representation

    for i in 1:nsamples

        err_i = frame_tableau[i][1:n]
 
        # analogous to CommutationCheckECCSetup
        QuantumClifford.comm!(
            @view(measured_faults[i, :]),
            fmtab,
            err_i,
        )
    end

    measured_faults .%= 2

    guesses =
        QuantumClifford.ECC.batchdecode(
            d,
            syndromes,
        )

    pL =
        QuantumClifford.ECC.evaluate_guesses(
            measured_faults,
            guesses,
            O,
        )

    return pL
end

p_mem(Δt_GHZ; T_coh=1.0) = (3/4) * (1 - exp(-Δt_GHZ / T_coh))

function extract_pL(gen_time, F_GHZ, code, T_coh; gate_fidelity=1.0, nsamples=100_000)
    p_mem_val = p_mem(gen_time; T_coh=T_coh)  # generation time per stabilizer generator takes twice as long

    setup = CShorSyndromeECCSetup(p_mem_val, gate_fidelity, F_GHZ)
    decoder = CSSTableDecoder(code)

    r = cevaluate_decoder_pL(decoder, setup, nsamples)

    return (
        pL = r,
        p_mem = p_mem_val,
    )
end