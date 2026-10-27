# Side-by-side comparison of GHZfidelity_closedform.jl and GHZfromtrace.jl on a few
# representative Piecemaker timelines. Prints a table instead of pass/fail
# (see test_closedform_vs_trace.jl for the test suite).
#
# Run from src/:  julia --project compare_closedform_vs_trace.jl
# Optional environment variable:
#   N_TRAJ   trajectories per sampled (gate/readout) case (default 2000)
using Random, Statistics, Printf

include(joinpath(@__DIR__, "GHZfromtrace.jl"))
include(joinpath(@__DIR__, "GHZfidelity_closedform.jl"))

const rng = MersenneTwister(2026)
const N_TRAJ = parse(Int, get(ENV, "N_TRAJ", "2000"))

# Default hardware timings of the event simulation
const TK = (t_rotation_shuttle = 100e-6, t_CNOT = 100e-6, t_readout = 1e-3)

births4 = [0.0, 0.4e-3, 1.5e-3, 2.2e-3]
iso4 = trace_timeline(births4; TK...)

# Same pairs, but every fusion and the piecemaker readout wait behind other GHZ attempts
c_q = [NaN, 3.0e-3, 6.0e-3, 9.5e-3]
queued4 = (births4, c_q, c_q .+ TK.t_CNOT, 12.0e-3, 13.0e-3)

# Late pair: the switch idles until pair 4 arrives
late4 = trace_timeline([0.0, 0.2e-3, 0.3e-3, 20e-3]; TK...)

iso3 = trace_timeline([0.0, 0.8e-3, 1.1e-3]; TK...)
iso5 = trace_timeline([0.0, 0.4e-3, 1.5e-3, 2.2e-3, 2.9e-3]; TK...)

# (name, timeline, noise). Cases with F_CNOT = F_readout = 1 are deterministic in both methods.
cases = [
    ("ideal (no noise)",                 iso4,    (T_coherence = Inf,   F_link = 1.0)),
    ("link only, F_link=0.97",           iso4,    (T_coherence = Inf,   F_link = 0.97)),
    ("memory only, τ=0.1 s",             iso4,    (T_coherence = 0.1,   F_link = 1.0)),
    ("link + memory, τ=1 s",             iso4,    (T_coherence = 1.0,   F_link = 0.97)),
    ("link + memory, τ=10 ms (strong)",  iso4,    (T_coherence = 10e-3, F_link = 0.97)),
    ("queued, link + memory, τ=0.1 s",   queued4, (T_coherence = 0.1,   F_link = 0.97)),
    ("late 4th pair, τ=0.1 s",           late4,   (T_coherence = 0.1,   F_link = 0.97)),
    ("n=3, link + memory, τ=0.1 s",      iso3,    (T_coherence = 0.1,   F_link = 0.97)),
    ("n=5, link + memory, τ=0.1 s",      iso5,    (T_coherence = 0.1,   F_link = 0.97)),
    # Sampled in the trace: gate/readout errors exaggerated so the effect is resolvable
    ("gate only, F_CNOT=0.97",           iso4,    (T_coherence = Inf,   F_link = 1.0, F_CNOT = 0.97)),
    ("readout only, F_readout=0.98",     iso4,    (T_coherence = Inf,   F_link = 1.0, F_readout = 0.98)),
    ("all noise, queued, τ=0.1 s",       queued4, (T_coherence = 0.1,   F_link = 0.97, F_CNOT = 0.98, F_readout = 0.99)),
]

@printf("%-34s %2s  %12s  %12s  %10s  %10s  %6s\n",
        "case", "n", "closed form", "trace", "Δ", "SEM", "z")
println("-"^96)
for (name, tl, noise) in cases
    pars = merge((F_CNOT = 1.0, F_readout = 1.0), noise)
    sampled = pars.F_CNOT < 1 || pars.F_readout < 1
    F_cf = ghz_fidelity_closedform(tl...; pars...)
    trace_pars = merge(pars, (T_coherence = Float64(pars.T_coherence),))
    if sampled
        samples = [simulate_piecemaker_timeline(tl...; trace_pars..., rng) for _ in 1:N_TRAJ]
        F_tr, sem = mean(samples), std(samples) / sqrt(N_TRAJ)
        @printf("%-34s %2d  %12.8f  %12.8f  %10.2e  %10.2e  %6.2f\n",
                name, length(tl[1]), F_cf, F_tr, F_tr - F_cf, sem, (F_tr - F_cf) / sem)
    else
        F_tr = simulate_piecemaker_timeline(tl...; trace_pars..., rng)
        @printf("%-34s %2d  %12.8f  %12.8f  %10.2e  %10s  %6s\n",
                name, length(tl[1]), F_cf, F_tr, F_tr - F_cf, "exact", "")
    end
end
println("\nSampled rows: trace = mean of $(N_TRAJ) trajectories; z = Δ / SEM (|z| ≲ 3 expected).")
