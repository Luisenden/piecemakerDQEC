# Exact GHZ fidelity of one Piecemaker construction from its timing trace.
#
# Replaces the density-matrix reconstruction in GHZfromtrace.jl. Exact (not sampled)
# for Pauli noise: Werner links, depolarizing or dephasing memory (QuantumSavory's
# `Depolarization(τ)` / `T2Dephasing(T2)` parameterizations), two-qubit depolarizing
# CNOT noise and classical readout flips.


# Pauli probabilities (pX, pY, pZ) of memory noise over a storage interval Δt.
@inline function _mem_probs(Δt::Real, T::Real, memory::Symbol)
    Δt <= 0 && return (0.0, 0.0, 0.0)
    λ = -expm1(-Float64(Δt) / Float64(T))                     # 1 - exp(-Δt/T); exactly 0 for T = Inf
    memory === :depolarizing && return (λ / 4, λ / 4, λ / 4)
    memory === :dephasing && return (0.0, 0.0, λ / 2)
    throw(ArgumentError("unknown memory model $(memory); use :depolarizing or :dephasing"))
end

@inline _anticommutes(sx::UInt64, sz::UInt64, ex::UInt64, ez::UInt64) =
    isodd(count_ones(sx & ez) + count_ones(sz & ex))

# Multiply each stabilizer's accumulator by E[χ_s] of one noise event.
# `outcomes` lists (probability, x-mask, z-mask); the identity carries the remaining mass.
function _apply_event!(acc::Vector{Float64}, stabs, outcomes)
    pI = 1.0 - sum(first, outcomes; init = 0.0)
    @inbounds for i in eachindex(stabs)
        sx, sz = stabs[i]
        e = pI
        for (p, ex, ez) in outcomes
            e += _anticommutes(sx, sz, ex, ez) ? -p : p
        end
        acc[i] *= e
    end
    return acc
end

"""
    ghz_fidelity_closedform(b, c, g, t_pm_meas, t_end; T_coherence, memory = :depolarizing,
                            F_link = 1.0, F_CNOT = 1.0, F_readout = 1.0)

Exact fidelity of the delivered n-GHZ state for one Piecemaker construction.

All vectors are in fusion order (entry 1 = piecemaker pair):
- `b[j]`: Bell-pair birth time
- `c[j]`: fusion-CNOT time (ignored for j = 1)
- `g[j]`: end of CNOT = Z-measurement time of switch qubit j (ignored for j = 1)
- `t_pm_meas`: X-measurement time of the piecemaker
- `t_end`: time the remote qubits are consumed (fidelity evaluation time)

Memory noise acts on every stored qubit (remotes until `t_end`, switch qubits until their
measurement) with coherence time `T_coherence` (τ for :depolarizing, T2 for :dephasing).
"""
function ghz_fidelity_closedform(b::AbstractVector{<:Real}, c::AbstractVector{<:Real}, g::AbstractVector{<:Real},
                                 t_pm_meas::Real, t_end::Real;
                                 T_coherence::Real, memory::Symbol = :depolarizing,
                                 F_link::Real = 1.0, F_CNOT::Real = 1.0, F_readout::Real = 1.0)
    n = length(b)
    @assert length(c) == n && length(g) == n "b, c, g must have equal length"
    @assert 2 <= n <= 62 "supports 2 ≤ n ≤ 62"
    T = Float64(T_coherence)
    tM, tE = Float64(t_pm_meas), Float64(t_end)
    # Timeline sanity: fusions are sequential and every qubit exists when it is used.
    prev = Float64(b[1])
    for j in 2:n
        @assert b[j] <= c[j] <= g[j] "pair $j: need b ≤ c ≤ g, got ($(b[j]), $(c[j]), $(g[j]))"
        @assert c[j] >= prev "pair $j: CNOT at $(c[j]) precedes previous fusion/piecemaker birth at $(prev)"
        prev = Float64(g[j])
    end
    @assert tM >= prev "piecemaker measured before the last fusion ended"
    @assert tE >= tM "consumption before piecemaker measurement"

    full = (UInt64(1) << n) - 1
    stabs = [(sx, sz) for sx in (UInt64(0), full) for sz in UInt64(0):full if iseven(count_ones(sz))]
    acc = ones(length(stabs))
    bit(j) = UInt64(1) << (j - 1)          # remote r_j
    fused(k) = (UInt64(1) << k) - 1        # remotes r_1..r_k
    r1 = bit(1)

    λL = 4 / 3 * (1 - F_link)              # Werner pair = one-sided depolarizing with λL
    λg = 4 / 3 * (1 - F_CNOT)
    ε = 1 - F_readout

    local_pauli(p, j) = ((p[1], bit(j), UInt64(0)), (p[2], bit(j), bit(j)), (p[3], UInt64(0), bit(j)))

    for j in 1:n
        # Werner link noise and remote memory (b_j → t_end)
        λL > 0 && _apply_event!(acc, stabs, local_pauli((λL / 4, λL / 4, λL / 4), j))
        _apply_event!(acc, stabs, local_pauli(_mem_probs(tE - b[j], T, memory), j))
        j == 1 && continue
        # switch qubit j before its CNOT: transposes onto r_j
        _apply_event!(acc, stabs, local_pauli(_mem_probs(c[j] - b[j], T, memory), j))
        # switch qubit j between CNOT and Z-measurement: X, Y flip the outcome → X on r_j
        pX, pY, _ = _mem_probs(g[j] - c[j], T, memory)
        _apply_event!(acc, stabs, ((pX + pY, bit(j), UInt64(0)),))
        # CNOT noise on (piecemaker, switch j) at g_j: with prob λg a uniform Pa ⊗ Pb
        if λg > 0
            A = fused(j)
            pm = ((UInt64(0), UInt64(0)), (A, UInt64(0)), (A, r1), (UInt64(0), r1))   # I, X, Y, Z
            sj = (UInt64(0), bit(j), bit(j), UInt64(0))                               # I, X, Y, Z
            outcomes = [(λg / 16, pm[a][1] ⊻ sj[bb], pm[a][2]) for a in 1:4 for bb in 1:4]
            _apply_event!(acc, stabs, outcomes)
        end
        # Z-readout flip on switch qubit j → wrong X correction on r_j
        ε > 0 && _apply_event!(acc, stabs, ((ε, bit(j), UInt64(0)),))
    end
    # Piecemaker memory, interval m has fused set {r_1..r_m}
    cuts = [Float64(b[1]); Float64.(c[2:n]); tM]
    for m in 1:n
        pX, pY, pZ = _mem_probs(cuts[m + 1] - cuts[m], T, memory)
        A = fused(m)
        _apply_event!(acc, stabs, ((pX, A, UInt64(0)), (pY, A, r1), (pZ, UInt64(0), r1)))
    end
    # Piecemaker X-readout flip → wrong Z correction on r_1
    ε > 0 && _apply_event!(acc, stabs, ((ε, UInt64(0), r1),))
    return sum(acc) / length(acc)
end

"""
    ghz_fidelities_from_log(df; kwargs...)

Fidelity for every row of the `raw_events` DataFrame produced by the patched simulation
(columns bellpair_1..4, cnot_2..4, meas_2..4, t_pm_meas, timesteps). Keyword arguments are
passed to `ghz_fidelity_closedform`.
"""
function ghz_fidelities_from_log(df; kwargs...)
    return [
        ghz_fidelity_closedform(
            [r.bellpair_1, r.bellpair_2, r.bellpair_3, r.bellpair_4],
            [NaN, r.cnot_2, r.cnot_3, r.cnot_4],
            [NaN, r.meas_2, r.meas_3, r.meas_4],
            r.t_pm_meas, r.timesteps; kwargs...)
        for r in eachrow(df)
    ]
end

"""
    trace_timeline(birth_times; t_rotation_shuttle, t_CNOT, t_readout)

Timeline (b, c, g, t_pm_meas, t_end) implied by `simulate_piecemaker_trace` for the same
inputs: the four pairs are processed alone on the switch, without other GHZ attempts.
Used only to cross-check this file against GHZfromtrace.jl.
"""
function trace_timeline(birth_times; t_rotation_shuttle, t_CNOT, t_readout)
    b = sort(Float64.(birth_times)); b .-= b[1]
    n = length(b)
    c = fill(NaN, n); g = fill(NaN, n)
    current = b[1] + t_rotation_shuttle
    for k in 2:n
        c[k] = max(current, b[k]) + t_rotation_shuttle
        g[k] = c[k] + t_CNOT
        current = g[k] + t_readout
    end
    return b, c, g, current, current + t_readout
end
