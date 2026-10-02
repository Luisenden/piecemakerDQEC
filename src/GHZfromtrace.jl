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
# Density-matrix reconstruction of one Piecemaker GHZ from a FULL timeline
# ------------------------------------------------------------
"""
    simulate_piecemaker_timeline(b, c, g, t_pm_meas, t_end; T_coherence, F_link, F_CNOT, F_readout, rng)

Density-matrix fidelity of one Piecemaker GHZ built on the given timeline. This is the
same model as `simulate_piecemaker_trace`, but all operation times are taken from the
input (e.g. the logged times of the patched event simulation), so queueing behind other
GHZ attempts is included.

All vectors are in fusion order (entry 1 = piecemaker pair):
- `b[j]`: Bell-pair birth time
- `c[j]`: fusion-CNOT time (ignored for j = 1)
- `g[j]`: CNOT end = Z-measurement time of switch qubit j (ignored for j = 1)
- `t_pm_meas`: X-measurement time of the piecemaker
- `t_end`: time at which the fidelity of the remote GHZ state is evaluated

Gate and readout errors are SAMPLED (one trajectory per call), as before.
Works for any GHZ size n = length(b); cost grows as 4^(2n), so keep n small.
"""
function simulate_piecemaker_timeline(
    b::AbstractVector{<:Real},
    c::AbstractVector{<:Real},
    g::AbstractVector{<:Real},
    t_pm_meas::Real,
    t_end::Real;

    T_coherence::Float64 = 1.0,
    F_link::Float64 = 0.97,
    F_CNOT::Float64 = 0.9997,
    F_readout::Float64 = 1.0,

    rng::AbstractRNG = Random.default_rng(),
)
    n = length(b)
    @assert length(c) == n && length(g) == n "b, c, g must have equal length"

    # Only time differences matter; shift so the piecemaker pair is born at t = 0.
    t0 = Float64(b[1])
    bt = Float64.(b) .- t0
    ct = Float64.(c) .- t0
    gt = Float64.(g) .- t0
    tM = Float64(t_pm_meas) - t0
    tE = Float64(t_end) - t0

    # Timeline sanity (same checks as the closed form)
    prev = bt[1]
    for j in 2:n
        @assert bt[j] <= ct[j] <= gt[j] "pair $j: need b ≤ c ≤ g"
        @assert ct[j] >= prev "pair $j: CNOT before the previous fusion ended"
        prev = gt[j]
    end
    @assert prev <= tM <= tE "need last fusion ≤ t_pm_meas ≤ t_end"

    repr = QuantumOpticsRepr()
    background = Depolarization(T_coherence)

    switch = Register([Qubit() for _ in 1:n], [repr for _ in 1:n], [background for _ in 1:n])
    remote = Register([Qubit() for _ in 1:n], [repr for _ in 1:n], [background for _ in 1:n])

    pair_state = noisy_bell_state(F_link)

    # Piecemaker pair
    initialize!((switch[1], remote[1]), pair_state; time = bt[1])

    for k in 2:n
        # Bell pair k is created at its logged birth time and decoheres in memory
        # until its logged CNOT time (this interval contains any queueing delay).
        initialize!((switch[k], remote[k]), pair_state; time = bt[k])

        apply!((switch[1], switch[k]), CNOT; time = ct[k])

        # Memory noise on both CNOT qubits during the gate
        uptotime!((switch[1], switch[k]), gt[k])

        gate_error = sample_depol2q(rng, F_CNOT)
        !isnothing(gate_error[1]) && apply!(switch[1], gate_error[1]; time = gt[k])
        !isnothing(gate_error[2]) && apply!(switch[k], gate_error[2]; time = gt[k])

        res = project_traceout!(switch[k], (Z1, Z2); time = gt[k])
        rand(rng) > F_readout && (res = 3 - res)

        # Feed-forward X on the remote qubit. Pauli corrections commute with the
        # Pauli memory noise, so applying it at gt[k] instead of after the readout
        # wait does not change the fidelity (and avoids needing the readout time).
        res == 2 && apply!(remote[k], X; time = gt[k])
    end

    # Piecemaker X-measurement at its logged time
    res = project_traceout!(switch[1], (X1, X2); time = tM)
    rand(rng) > F_readout && (res = 3 - res)
    res == 2 && apply!(remote[1], Z; time = tM)

    # Fidelity at the logged consumption time; observable() evolves all remotes to tE.
    ghz_projector = SProjector(StabilizerState(ghz(n)))
    fidelity = real(observable(Tuple(remote[j] for j in 1:n), ghz_projector; time = tE))
    return fidelity
end


"""
    isolated_timeline(birth_times; t_rotation_shuttle, t_CNOT, t_readout)

Timeline a single GHZ would have if its pairs were the only work on the switch
(no other GHZ attempts): the schedule the original `simulate_piecemaker_trace` assumed.
Returns (b, c, g, t_pm_meas, t_end) with the piecemaker pair born at t = 0.
"""
function isolated_timeline(birth_times; t_rotation_shuttle, t_CNOT, t_readout)
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


"""
    simulate_piecemaker_trace(birth_times; T_coherence, F_link, t_rotation_shuttle, t_CNOT, F_CNOT, t_readout, F_readout, rng)

Backward-compatible entry point: reconstructs the ISOLATED timeline from the birth
times alone (no queueing behind other GHZ attempts) and evaluates it with
`simulate_piecemaker_timeline`.
"""
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
    b, c, g, tM, tE = isolated_timeline(birth_times; t_rotation_shuttle, t_CNOT, t_readout)
    return simulate_piecemaker_timeline(b, c, g, tM, tE;
        T_coherence, F_link, F_CNOT, F_readout, rng)
end


"""
    simulate_piecemaker_from_log(df; kwargs...)

Density-matrix fidelity (one sampled trajectory per row) for every row of the patched
simulation's `raw_events`, using the logged timeline including queueing.
"""
function simulate_piecemaker_from_log(df; kwargs...)
    return [
        simulate_piecemaker_timeline(
            [r.bellpair_1, r.bellpair_2, r.bellpair_3, r.bellpair_4],
            [NaN, r.cnot_2, r.cnot_3, r.cnot_4],
            [NaN, r.meas_2, r.meas_3, r.meas_4],
            r.t_pm_meas, r.timesteps; kwargs...)
        for r in eachrow(df)
    ]
end
