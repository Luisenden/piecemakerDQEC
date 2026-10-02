# see tutorial at https://qc.quantumsavory.org/stable/ECC_evaluating/

# this script estimates the logical residual error probabilities of quantum error-correcting codes under a noisy memory and noisy Shor-style syndrome-extraction circuit.
# for each code and each memory-error probability, it performs Monte Carlo circuit (shor syndrome extraction) simulations

using DataFrames
using JLD2
using Statistics
using StatsBase
using Printf

using QuantumClifford
using QuantumClifford.ECC
using QuantumClifford.ECC: AbstractECCSetup, AbstractSyndromeDecoder, batchdecode, evaluate_guesses, faults_matrix
import QuantumClifford: applynoise!

# Data-qubit coherence time T_coh,data in p_mem(τ) = 3/4 [1 - exp(-τ / T_coh,data)]
const T_COH_DATA = 1.0

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

p_mem(Δt_GHZ; T_coh=T_COH_DATA) = (3/4) * (1.0 - exp(-Δt_GHZ / T_coh))

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
    F_ghz;
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
function t_cycle(p_mem; T_coh=T_COH_DATA)
    0 <= p_mem < 0.75 ||
        throw(DomainError(p_mem, "p_mem must satisfy 0 ≤ p_mem < 0.75"))

    return -T_coh * log1p(-4p_mem / 3)
end

## utils for multi-objective hardware parameter optimization

"""
    round_completion_times(df_raw; required_count) -> Vector{Float64}

Times at which one complete round of stabilizer measurements has finished: every generator
has been completed at least `required_count` times since the previous round (Steane: the X and
Z checks share supports, so every support needs 2 completions).
"""
function round_completion_times(df_raw::DataFrame;
        required_count::Int = (code == "Steane713" ? 2 : 1))
    all_generators = collect(eachindex(GENERATORS))
    counts = Dict(g => 0 for g in all_generators)
    markers = Float64[]
    for row in sortperm(df_raw.timesteps)          # work in timestep order
        counts[df_raw.generator_idx[row]] += 1
        if all(counts[g] >= required_count for g in all_generators)
            push!(markers, df_raw.timesteps[row])
            for g in all_generators
                counts[g] = 0
            end
        end
    end
    return markers
end

"""
    estimate_cycle_time(df_raw; n_batches = 20) -> (τ, τ_se, n_rounds)

τ̂ = mean time between consecutive round completions. Consecutive round intervals are
serially correlated (partial GHZ states carry over between rounds), so the standard error is
a batch-means estimate rather than std/√n. τ = Inf when fewer than two rounds completed.
"""
function estimate_cycle_time(df_raw::DataFrame; n_batches::Int = 20)
    markers = isempty(df_raw) ? Float64[] : round_completion_times(df_raw)
    intervals = diff(markers)
    n = length(intervals)
    n == 0 && return (τ = Inf, τ_se = Inf, n_rounds = length(markers))

    τ = mean(intervals)
    nb = min(n_batches, n)
    τ_se = if nb < 2
        Inf
    else
        edges = round.(Int, range(0, n; length = nb + 1))
        batch_means = [mean(@view intervals[edges[b]+1:edges[b+1]]) for b in 1:nb]
        std(batch_means) / sqrt(nb)
    end
    return (τ = τ, τ_se = τ_se, n_rounds = length(markers))
end

# h_t = (t_att, t_CNOT, t_ro, t_buff, p_link), named as the keywords of run_single_configuration.
#   :asc  → a smaller value is more demanding (faster, i.e. more expensive hardware)
#   :desc → a larger value is more demanding (higher link success probability)
const TIMING_DIRECTIONS = (
    attempt_time        = :asc,   # t_att
    Δt_CNOTgate         = :asc,   # t_CNOT
    Δt_readout          = :asc,   # t_ro
    Δt_rotation_shuttle = :asc,   # t_buff
    link_success_prob   = :desc,  # p_link
)

# The timing simulation runs noise-free; the fidelity parameters h_f are applied afterwards
# to the logged trace (Step 4), so these values are dummies.
const NOISE_FREE_SIM = (F_link = 1.0, T_coherence = Inf, gate_fidelity = 1.0, readout_fidelity = 1.0)

"""
    order_timing_grid(grid) -> NamedTuple

Sort every axis from most to least demanding, so that grid index 1 is the fastest (most
expensive) value and a larger index means more relaxed hardware. A parameter is fixed by
giving a single value.
"""
function order_timing_grid(grid::NamedTuple)
    names = keys(TIMING_DIRECTIONS)
    Set(keys(grid)) == Set(names) ||
        throw(ArgumentError("timing grid must have exactly the fields $(names)"))
    sorted_axes = map(names) do name
        vals = unique(Float64.(collect(grid[name])))
        sort(vals; rev = TIMING_DIRECTIONS[name] == :desc)
    end
    return NamedTuple{names}(sorted_axes)
end

"""h_t at grid index tuple `idx` (one index per axis, in the order of the grid's fields)."""
timing_point(grid::NamedTuple, idx::Tuple) =
    NamedTuple{keys(grid)}(ntuple(d -> grid[d][idx[d]], length(grid)))

"""
    evaluate_timing!(cache, h_t, cutoff; sim_kwargs, T_coh_data) -> NamedTuple

Run the noise-free timing simulation T(h_t, c) once and store the trace with its cycle-time
estimate τ̂(h_t, c) and p_mem(h_t, c). Results are memoised in `cache`, so Step 3 reuses the
c = ∞ traces of Step 2. Use the same `seed` for every configuration (common random numbers):
differences between neighbouring configurations then reflect the parameters rather than
sampling noise, which matters because Steps 2-3 compare neighbours.
"""
function evaluate_timing!(cache::AbstractDict, h_t::NamedTuple, cutoff::Real;
        sim_kwargs::NamedTuple, T_coh_data::Real = T_COH_DATA)
    key = (h_t, Float64(cutoff), sim_kwargs)
    haskey(cache, key) && return cache[key]

    result = run_single_configuration(; NOISE_FREE_SIM..., h_t..., cutoff = Float64(cutoff), sim_kwargs...)
    est = estimate_cycle_time(result.raw_events)

    evaluation = (
        h_t        = h_t,
        cutoff     = Float64(cutoff),
        τ          = est.τ,
        τ_se       = est.τ_se,
        n_rounds   = est.n_rounds,
        p_mem      = p_mem(est.τ; T_coh = T_coh_data),
        converged  = result.converged,
        raw_events = result.raw_events,
    )
    cache[key] = evaluation
    return evaluation
end

"""
    cycle_time_feasible(evaluation, τ_max; z = 0) -> Bool

p_mem(τ) is strictly increasing in τ, so p_mem(h_t, c) ≤ p_mem^max ⇔ τ̂(h_t, c) ≤ τ_max with
τ_max = t_cycle(p_mem^max). For z > 0 the lower confidence bound τ̂ − z·SE is compared instead:
a configuration is then only discarded once it is significantly too slow (conservative pruning;
z = 1.645 ≈ one-sided 95 %). z = 0 is the plain point-estimate rule of the summary.
"""
function cycle_time_feasible(evaluation, τ_max::Real; z::Real = 0.0)
    isfinite(evaluation.τ) || return false
    τ_test = z == 0 ? evaluation.τ : evaluation.τ - z * evaluation.τ_se
    return τ_test <= τ_max
end

"""Table with one row per evaluation (without the traces)."""
function summary_table(evaluations)
    isempty(evaluations) && return DataFrame()
    return DataFrame([
        merge(e.h_t, (
            cutoff = e.cutoff, k = get(e, :k, 0), p_kept = get(e, :p_kept, 1.0),
            τ = e.τ, τ_se = e.τ_se, p_mem = e.p_mem, n_rounds = e.n_rounds, converged = e.converged,
        )) for e in evaluations
    ])
end

"""Parity checks of the QEC code, for the break-even model in utils_pseudothreshold.jl."""
function qec_parity_checks(name::AbstractString)
    name == "Steane713" && return Steane7()
    name == "BB12_2_3" && return BB12_2_3
    name == "GB26_2_5" && return GB26_2_5
    throw(ArgumentError("unknown code $(name)"))
end

"""
    find_pmem_max(code_checks, gate_fidelity; log_pmem_lo, log_pmem_hi, log_tol, ...) -> p_mem^max

p_mem^max = max{p_mem : ∃ε with p_L(p_mem, ε) ≤ p_mem}. Since p_L is non-decreasing in the GHZ
infidelity ε, such an ε exists iff break-even holds for a perfect GHZ state (ε = 0, F_GHZ = 1).
p_mem^max is therefore found by binary search over log10(p_mem) at F_GHZ = 1.

The lower bracket must be break-even. Note that with gate noise break-even also fails for very
small p_mem (the gate-noise floor of p_L exceeds p_mem), so the break-even set is an interval
[p_min, p_max]; choose `log_pmem_lo` inside it.

Statistically unresolved points (:interval_uncertain) are counted as feasible and the upper
end of the final bracket is returned. Both choices err towards a larger p_mem^max, so the
pruning discards no configuration that is feasible within Monte-Carlo resolution.
"""
function find_pmem_max(code_checks, gate_fidelity::Real;
        log_pmem_lo::Real = -2.0,
        log_pmem_hi::Real = log10(0.75), # the maximum p_mem for a depolarizing channel
        log_tol::Real = 0.05,
        nsamples_start::Int = 1_000,
        nsamples_max::Int = 100_000)

        decoder = TableDecoder(code_checks, error_weight=4)

    status(logp) = classify_point(decoder, 10.0^logp, gate_fidelity, 1.0;
        nsamples_start = nsamples_start, nsamples_max = nsamples_max).status

    status(log_pmem_lo) == :interval_above && throw(ArgumentError(
        "p_mem = 10^$(log_pmem_lo) is not break-even even with a perfect GHZ state; choose another lower bracket."))
    status(log_pmem_hi) == :interval_above ||
        @warn "p_mem = 10^$(log_pmem_hi) is not resolved as above break-even; p_mem^max may exceed the bracket."

    lo, hi = Float64(log_pmem_lo), Float64(log_pmem_hi)
    while hi - lo > log_tol
        mid = (lo + hi) / 2
        if status(mid) == :interval_above
            hi = mid
        else
            lo = mid
        end
    end
    return 10.0^hi
end

"""
Stricter alternative to p_mem^max: the memory error at which the break-even GHZ infidelity
ε*(p_mem) from Step 1 is largest.
"""
pmem_peak(pmem_values::AbstractVector, eps_star_values::AbstractVector) =
    pmem_values[argmax(eps_star_values)]


"""
    prune_timing_configurations!(cache, grid, τ_max; search_axis, z, sim_kwargs, T_coh_data)

Step 2: U_{t,∞} = {h_t : p_mem(h_t, ∞) ≤ p_mem^max} on the ordered grid (see `order_timing_grid`).

τ(h_t, ∞) is monotone in every timing parameter, so U_{t,∞} is a down-set ("staircase") in
grid-index space: if index vector i is feasible, every j ≤ i componentwise (at least as
demanding in every parameter) is feasible as well. Relaxing each parameter separately from
the fastest corner would only find where the staircase meets each axis. Here the whole
staircase is found, which Step 3 needs because it loops over every h_t ∈ U_{t,∞}:

  b(a) = largest feasible index along `search_axis` for each setting a of the other axes.

b is non-increasing in a. The settings a are visited in column-major order, so every a − e_j is
finished before a. Then b(a) ≤ min_j b(a − e_j), and b(a) is found by binary search inside
that bound. A setting whose bound is 0 needs no simulation. Taking the longest axis as
`search_axis` (the default) gives the fewest simulations.

Returns `members` (all of U_{t,∞}), `frontier` (its maximal, least demanding elements),
`boundary` (the array b), `search_axis` and `n_simulations` (new simulations run).
"""
function prune_timing_configurations!(cache::AbstractDict, grid::NamedTuple, τ_max::Real;
        search_axis::Int = argmax(collect(map(length, grid))),
        z::Real = 0.0,
        sim_kwargs::NamedTuple,
        T_coh_data::Real = T_COH_DATA)

    D = length(grid)
    lengths = collect(map(length, values(grid)))
    all(>(0), lengths) || throw(ArgumentError("every timing axis needs at least one value"))
    n_search = lengths[search_axis]
    other_axes = [d for d in 1:D if d != search_axis]
    prefix_dims = Tuple(lengths[other_axes])
    boundary = zeros(Int, prefix_dims)
    n_cached_before = length(cache)

    function full_index(a::CartesianIndex, i::Int)
        idx = Vector{Int}(undef, D)
        idx[other_axes] .= Tuple(a)
        idx[search_axis] = i
        return Tuple(idx)
    end

    feasible(a, i) = cycle_time_feasible(
        evaluate_timing!(cache, timing_point(grid, full_index(a, i)), Inf;
            sim_kwargs = sim_kwargs, T_coh_data = T_coh_data),
        τ_max; z = z)

    for a in CartesianIndices(boundary)
        # monotonicity: b(a) ≤ b(a − e_j) for every already-finished predecessor
        upper = n_search
        for j in eachindex(other_axes)
            a[j] > 1 || continue
            predecessor = CartesianIndex(Base.setindex(Tuple(a), a[j] - 1, j))
            upper = min(upper, boundary[predecessor])
        end
        if upper == 0
            continue            # boundary[a] stays 0, nothing to simulate
        end

        # invariant: lo is feasible (0 = sentinel), hi is infeasible (known, or beyond the grid)
        lo, hi = 0, upper + 1
        while hi - lo > 1
            mid = (lo + hi) ÷ 2
            if feasible(a, mid)
                lo = mid
            else
                hi = mid
            end
        end
        boundary[a] = lo
    end

    HT = typeof(timing_point(grid, ntuple(_ -> 1, D)))   # concrete element type for the outputs
    members = HT[]
    for a in CartesianIndices(boundary), i in 1:boundary[a]
        push!(members, timing_point(grid, full_index(a, i)))
    end

    return (
        members       = members,
        n_simulations = length(cache) - n_cached_before,
    )
end

"""
    ghz_generation_times(df_raw, Δt_readout) -> Vector{Float64}

T_GHZ^(j): time from the arrival of the first Bell pair (`bellpair_1`, the earliest birth time,
i.e. the piecemaker) to the piecemaker X measurement. In `consumer` the piecemaker is measured,
then Δt_readout elapses, then `timesteps` is logged, so t_pm_meas = timesteps − Δt_readout.
"""
function ghz_generation_times(df_raw::DataFrame, Δt_readout::Real)
    t_pm_meas = "t_pm_meas" in names(df_raw) ? df_raw.t_pm_meas : df_raw.timesteps .- Δt_readout
    return Float64.(t_pm_meas .- df_raw.bellpair_1)
end

"""
    quantile_cutoffs(T_ghz; k_max = 50) -> Vector{(k, p, cutoff)}

c_k = Q̂(p_k), p_k = 1/k, k = 1, 2, …, k_max, where Q̂ is the empirical quantile function in
the generalized-inverse sense, Q̂(p) = inf{x : F̂(x) ≥ p}. For sorted samples
x₍₁₎ ≤ … ≤ x₍ₙ₎ this is x₍⌈np⌉₎, and for p = 1/k the index is cld(n, k), computed exactly in
integer arithmetic. (`Statistics.quantile` defaults to Hyndman–Fan type 7, which interpolates
linearly and is not this inverse.) Repeated values (ties, or k > n) are dropped because they
give the same cutoff.
"""
function quantile_cutoffs(T_ghz::AbstractVector{<:Real}; k_max::Int = 50)
    out = NamedTuple{(:k, :p, :cutoff), Tuple{Int, Float64, Float64}}[]
    isempty(T_ghz) && return out
    x = sort(Float64.(T_ghz))
    n = length(x)
    for k in 1:k_max
        c = x[cld(n, k)]
        (isempty(out) || c < last(out).cutoff) && push!(out, (k = k, p = 1 / k, cutoff = c))
    end
    return out
end

"""
    cutoff_range!(cache, U_t, τ_max; k_max, patience, z, sim_kwargs, T_coh_data, verbose) -> U_{t,c}

Step 3: for every h_t ∈ U_{t,∞}, start at c = ∞ and lower the cutoff along c_k = Q̂_{T_GHZ}(1/k)
of the c = ∞ trace. The sweep for h_t stops after `patience` consecutive cutoffs with
p_mem(h_t, c_k) > p_mem^max. `patience = 1` is the rule of the summary (stop at the first
infeasible c_k, which is justified by τ(h_t, c) being non-increasing in c). A larger value
guards against a single noisy τ̂ ending the sweep too early.

`verbose` controls the progress output:
  0 silent except for warnings,
  1 one line per h_t (default): result of its sweep, elapsed time and ETA,
  2 additionally one line per simulated cutoff.

Returns U_{t,c} as a vector of evaluations (fields of `evaluate_timing!` plus `k` and
`p_kept`; k = 0 marks c = ∞). Each entry keeps its trace `raw_events` for Step 4.
"""
function cutoff_range!(cache::AbstractDict, U_t::AbstractVector, τ_max::Real;
        k_max::Int = 50,
        patience::Int = 1,
        z::Real = 0.0,
        sim_kwargs::NamedTuple,
        T_coh_data::Real = T_COH_DATA,
        verbose::Int = 1)

    fmt_ht(h) = join(("$(k)=$(round(v; sigdigits = 3))" for (k, v) in pairs(h)), ", ")
    function fmt_dur(seconds)
        isfinite(seconds) || return "?"
        s = round(Int, seconds)
        return @sprintf("%d:%02d:%02d", s ÷ 3600, (s % 3600) ÷ 60, s % 60)
    end
    fmt_ms(t) = isfinite(t) ? @sprintf("%.3g ms", 1e3 * t) : "Inf"

    # evaluate and report whether a new simulation was run, and how long it took
    function evaluate(h, c)
        n_before = length(cache)
        t0 = time()
        ev = evaluate_timing!(cache, h, c; sim_kwargs = sim_kwargs, T_coh_data = T_coh_data)
        return ev, length(cache) > n_before, time() - t0
    end

    function report_cutoff(label, ev, ok, is_new, dt)
        verbose >= 2 || return
        println("      ", rpad(label, 22), " τ = ", lpad(fmt_ms(ev.τ), 10),
                @sprintf("  p_mem = %.4f  ", ev.p_mem), ok ? "feasible  " : "INFEASIBLE",
                @sprintf("  (%s, %.1f s, %d GHZ, %d rounds)", is_new ? "new" : "cached", dt,
                         nrow(ev.raw_events), ev.n_rounds))
        flush(stdout)
    end

    N = length(U_t)
    U_tc = NamedTuple[]
    t_start = time()
    n_new_total = 0
    n_skipped = 0
    n_open_ended = 0

    verbose >= 1 && println("Step 3: cutoff sweep over $(N) timing configurations " *
                            "(k_max = $(k_max), patience = $(patience), τ_max = $(fmt_ms(τ_max)))")

    for (i, h_t) in enumerate(U_t)
        t_ht = time()
        verbose >= 2 && println(lpad(i, ndigits(N)), "/", N, "  ", fmt_ht(h_t))

        ev_inf, is_new, dt = evaluate(h_t, Inf)
        n_new = Int(is_new)
        ok_inf = cycle_time_feasible(ev_inf, τ_max; z = z)
        report_cutoff("c = ∞", ev_inf, ok_inf, is_new, dt)

        n_feasible = 0
        if !ok_inf
            # possible for members that Step 2 inferred by monotonicity without simulating them
            n_skipped += 1
            status = "SKIPPED: c = ∞ infeasible"
            verbose == 0 && @warn "h_t = $(h_t) was inferred to be in U_{t,∞}, but its own c = ∞ estimate is infeasible; skipped." ev_inf.τ τ_max
        else
            push!(U_tc, merge(ev_inf, (k = 0, p_kept = 1.0)))
            n_feasible = 1

            T_ghz = ghz_generation_times(ev_inf.raw_events, h_t.Δt_readout)
            n_infeasible_in_a_row = 0
            limit_found = false
            last_k = 0
            for q in quantile_cutoffs(T_ghz; k_max = k_max)
                ev, is_new, dt = evaluate(h_t, q.cutoff)
                n_new += is_new
                ok = cycle_time_feasible(ev, τ_max; z = z)
                report_cutoff(@sprintf("k = %2d, c = %s", q.k, fmt_ms(q.cutoff)), ev, ok, is_new, dt)
                last_k = q.k
                if ok
                    push!(U_tc, merge(ev, (k = q.k, p_kept = q.p)))
                    n_feasible += 1
                    n_infeasible_in_a_row = 0
                else
                    n_infeasible_in_a_row += 1
                    if n_infeasible_in_a_row >= patience
                        limit_found = true
                        break
                    end
                end
            end

            if limit_found
                status = "limit at k = $(last_k)"
            else
                # candidates exhausted: cutoffs tighter than c_{k_max} = Q̂(1/k_max) may still be feasible
                n_open_ended += 1
                status = "NO LIMIT up to k = $(last_k) (increase k_max)"
                verbose == 0 && @warn "Cutoff limit not reached for h_t = $(h_t) within k_max = $(k_max); " *
                                      "tighter cutoffs may still be feasible (increase k_max)."
            end
        end
        n_new_total += n_new

        if verbose >= 1
            elapsed = time() - t_start
            eta = elapsed / i * (N - i)     # rough: cost varies strongly between configurations
            println("[", lpad(i, ndigits(N)), "/", N, "] ",
                    verbose >= 2 ? "" : fmt_ht(h_t) * " | ",
                    "τ∞ = ", fmt_ms(ev_inf.τ), " | ", n_feasible, " feasible cutoffs, ", status,
                    @sprintf(" | %d new runs, %.1f s", n_new, time() - t_ht),
                    " | elapsed ", fmt_dur(elapsed), ", ETA ", fmt_dur(eta))
            flush(stdout)
        end
    end

    if verbose >= 1
        println("Step 3 done in $(fmt_dur(time() - t_start)): $(length(U_tc)) timing-cutoff pairs, " *
                "$(n_new_total) new simulations; skipped $(n_skipped), no limit within k_max for $(n_open_ended) of $(N).")
        flush(stdout)
    end
    return U_tc
end

# convenience method: a single timing configuration h_t
cutoff_range!(cache::AbstractDict, h_t::NamedTuple, τ_max::Real; kwargs...) =
    cutoff_range!(cache, [h_t], τ_max; kwargs...)