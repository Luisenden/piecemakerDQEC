# see tutorial at https://qc.quantumsavory.org/stable/ECC_evaluating/

# this script estimates the logical residual error probabilities of quantum error-correcting codes under a noisy memory and noisy Shor-style syndrome-extraction circuit.
# for each code and each memory-error probability, it performs Monte Carlo circuit (shor syndrome extraction) simulations

using Plots
using Colors
using QuantumClifford
using LaTeXStrings
using Measures

using QuantumClifford
using QuantumClifford.ECC

using QuantumClifford.ECC: AbstractECCSetup, AbstractSyndromeDecoder, batchdecode, evaluate_guesses, faults_matrix

import QuantumClifford: applynoise!

const GB26_2_5 = S"""IIIIIIIIIZIZIZIIIIIIIIIZII
IIIIIIIIIIZIZIZIIIIIIIIIZI
ZIIIIIIIIIIZIIIZIIIIIIIIIZ
IZIIIIIIIIIIZZIIZIIIIIIIII
ZIZIIIIIIIIIIIZIIZIIIIIIII
IZIZIIIIIIIIIIIZIIZIIIIIII
IIZIZIIIIIIIIIIIZIIZIIIIII
IIIZIZIIIIIIIIIIIZIIZIIIII
IIIIZIZIIIIIIIIIIIZIIZIIII
IIIIIZIZIIIIIIIIIIIZIIZIII
IIIIIIZIZIIIIIIIIIIIZIIZII
IIIIIIIZIZIIIIIIIIIIIZIIZI
XIIXIIIIIIIIIIIXIXIIIIIIII
IXIIXIIIIIIIIIIIXIXIIIIIII
IIXIIXIIIIIIIIIIIXIXIIIIII
IIIXIIXIIIIIIIIIIIXIXIIIII
IIIIXIIXIIIIIIIIIIIXIXIIII
IIIIIXIIXIIIIIIIIIIIXIXIII
IIIIIIXIIXIIIIIIIIIIIXIXII
IIIIIIIXIIXIIIIIIIIIIIXIXI
IIIIIIIIXIIXIIIIIIIIIIIXIX
IIIIIIIIIXIIXXIIIIIIIIIIXI
XIIIIIIIIIXIIIXIIIIIIIIIIX
IXIIIIIIIIIXIXIXIIIIIIIIII"""

const BB12_2_3 = S"""ZIZIIIIZIZII
ZZIIIIIIZIZI
IZZIIIZIIIIZ
IIIZIZZIIIZI
IIIZZIIZIIIZ
IIXXIIXXIIII
XIIIXIIXXIII
IXIIIXXIXIII
XIIIIXIIIXXI
IXIXIIIIIIXX"""


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

        # With probability 1-p, apply the depolarizing branch.
        # The sampled Pauli may still be identity.
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

    λ_gate = 16 * (1 - F_gate) / 15

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

# function cevaluate_decoder(
#     d::AbstractSyndromeDecoder,
#     setup::AbstractECCSetup,
#     nsamples::Int,
# )
#     H = parity_checks(d)

#     n = code_n(H)
#     k = code_k(H)

#     # Matrix mapping physical correction guesses to logical faults
#     O = faults_matrix(H)

#     # Build the noisy ECC circuit for the chosen setup,
#     # e.g. ShorSyndromeECCSetup or NaiveSyndromeECCSetup.
#     physical_noisy_circ, syndrome_bits, n_anc = physical_ECC_circuit(H, setup)

#     # Perfect encoding circuit
#     encoding_circ = naive_encoding_circuit(H)

#     # Used for testing logical Z failures by preparing/testing in X basis
#     preX = sHadamard[sHadamard(i) for i in n-k+1:n]

#     mdH = MixedDestabilizer(H)

#     # Circuits that noiselessly measure logical X and logical Z observables
#     logX_circ, _, logX_bits = naive_syndrome_circuit(
#         logicalxview(mdH),
#         n_anc + 1,
#         last(syndrome_bits) + 1,
#     )

#     logZ_circ, _, logZ_bits = naive_syndrome_circuit(
#         logicalzview(mdH),
#         n_anc + 1,
#         last(syndrome_bits) + 1,
#     )

#     # Logical X error:
#     # run encoding + noisy ECC + logical Z measurement
#     X_error = evaluate_decoder(
#         d,
#         nsamples,
#         vcat(encoding_circ, physical_noisy_circ, logZ_circ),
#         syndrome_bits,
#         logZ_bits,
#         O[end÷2+1:end, :],
#     )

#     # Logical Z error:
#     # prepare in X basis, then run encoding + noisy ECC + logical X measurement
#     Z_error = evaluate_decoder(
#         d,
#         nsamples,
#         vcat(preX, encoding_circ, physical_noisy_circ, logX_circ),
#         syndrome_bits,
#         logX_bits,
#         O[1:end÷2, :],
#     )

#     return X_error, Z_error
# end

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

    nlfails = round(Int, pL * nsamples)

    return (pL = pL, nlfails = nlfails)
end

p_mem(Δt_GHZ; T_coh=1.0) = (3/4) * (1.0 - exp(-Δt_GHZ / T_coh))

# binary search to find break-even infidelity for a given memory error probability in a code

function wilson_interval(k, N; z=1.645) # 90% confidence interval
    # defines uncertainty interval for a binomial proportion using the Wilson score interval
    p̂ = k / N

    D = 1 + z^2 / N

    center =
        (p̂ + z^2 / (2N)) / D

    halfwidth =
        z / D *
        sqrt(
            p̂ * (1 - p̂) / N +
            z^2 / (4N^2)
        )

    return (
        max(0.0, center - halfwidth),
        min(1.0, center + halfwidth),
    )
end


function classify_point(
    # classifies wether the logical error probability pL is above or below the physical memory error probability p_mem
    decoder,
    p_mem,
    gate_fidelity,
    F_ghz,
    nsamples_start = 1_000,
    nsamples_max = 1000_000,
)

    setup = CShorSyndromeECCSetup(p_mem, gate_fidelity, F_ghz)

    nsamples = nsamples_start

    while true

        res = cevaluate_decoder_pL(
            decoder,
            setup,
            nsamples,
        )

        lo, hi = wilson_interval(
            res.nlfails,
            nsamples,
        )

        if hi < p_mem
            return (
                status = :interval_below,
                result = res,
                ci = (lo, hi),
            )

        elseif lo > p_mem
            return (
                status = :interval_above,
                result = res,
                ci = (lo, hi),
            )

        elseif nsamples >= nsamples_max
            return (
                status = :interval_uncertain,
                result = res,
                ci = (lo, hi),
            )
        end

        nsamples = min(10 * nsamples, nsamples_max)
    end
end

function find_break_even(
    decoder,
    p_mem,
    gate_fidelity,
    logeps_min = -6.0,
    logeps_max = -1.0,
    logeps_tol = 0.05,
    nsamples_start = 1_000,
    nsamples_max = 1_000_000,
)

    lo = logeps_min
    hi = logeps_max

    while hi - lo > logeps_tol

        mid = (lo + hi) / 2

        eps_ghz = 10.0^mid
        F_ghz = 1 - eps_ghz

        classification = classify_point(
            decoder,
            p_mem,
            gate_fidelity,
            F_ghz;
            nsamples_start = nsamples_start,
            nsamples_max = nsamples_max,
        )

        if classification.status == :interval_below
            # pL < p_mem:
            # GHZ state can be made worse
            lo = mid

        elseif classification.status == :interval_above
            # pL > p_mem:
            # GHZ state must be better
            hi = mid

        else
            if hi - lo <= logeps_tol
                return (
                    status = :converged_statistically_unresolved,
                    logeps = mid,
                    eps_ghz = eps_ghz,
                    bracket = (lo, hi),
                    bracket_width = hi - lo,
                    details = classification,
                )
            else
                return (
                    status = :uncertain,
                    logeps = mid,
                    eps_ghz = eps_ghz,
                    bracket = (lo, hi),
                    bracket_width = hi - lo,
                    details = classification,
                )
            end
        end
    end

    logeps = (lo + hi) / 2
    eps_ghz = 10.0^logeps

    return (
        status = :converged,
        logeps = logeps,
        eps_ghz = eps_ghz,
    )
end

# calculate cycle time for a given memory error probability p_mem and coherence time T_coh
function t_cycle(p_mem; T_coh=1.0)
    0 <= p_mem < 0.75 ||
        throw(DomainError(p_mem, "p_mem must satisfy 0 ≤ p_mem < 0.75"))

    return -T_coh * log1p(-4p_mem / 3)
end