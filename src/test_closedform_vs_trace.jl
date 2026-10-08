# Cross-check GHZfidelity_closedform.jl against the density-matrix model in GHZfromtrace.jl,
# on isolated timelines, on queued timelines, and on the logged output of the event simulation.
# Also reports the fidelity cost of queueing (logged timeline vs isolated timeline).
#
# Run from the project directory:  julia --project test_closedform_vs_trace.jl
# Optional environment variables:
#   N_STAT    trajectories for the statistical gate/readout test (default 400)
#   SIM_WALL  wallclock budget in s for the short event-simulation run (default 30; 0 skips it)
using Random, Statistics, Test, DataFrames

@isdefined(code) || @eval const code = "Steane713"
@isdefined(error_model) || @eval const error_model = "depolarizing"

include(joinpath(@__DIR__, "GHZservice_v1_partitioned_sim_raw.jl"))
include(joinpath(@__DIR__, "GHZfromtrace.jl"))
include(joinpath(@__DIR__, "GHZfidelity_closedform.jl"))

const rng = MersenneTwister(2026)
const τ = 1.0
const N_STAT = parse(Int, get(ENV, "N_STAT", "10000"))
const SIM_WALL = parse(Float64, get(ENV, "SIM_WALL", "30"))

# Event-simulation parameters. The isolated timelines are built from the same Δt_* values,
# so that the logged-vs-isolated comparison isolates queueing.
const SIMPARS = (
    F_link = 0.97, link_success_prob = 1e-3, attempt_time = 1e-6, T_coherence = Inf,
    Δt_CNOTgate = 100e-6, gate_fidelity = 0.9997, Δt_readout = 1e-3, readout_fidelity = 1.0,
    Δt_rotation_shuttle = 100e-6, cutoff = Inf,
)
const TK = (t_rotation_shuttle = SIMPARS.Δt_rotation_shuttle, t_CNOT = SIMPARS.Δt_CNOTgate,
            t_readout = SIMPARS.Δt_readout)

row_timeline(r) = ([r.bellpair_1, r.bellpair_2, r.bellpair_3, r.bellpair_4],
                   [NaN, r.cnot_2, r.cnot_3, r.cnot_4],
                   [NaN, r.meas_2, r.meas_3, r.meas_4],
                   r.t_pm_meas, r.timesteps)

# Random timeline with artificial queueing: exponential extra waits before each CNOT and
# before the piecemaker readout, on top of the isolated schedule.
function queued_timeline(rng; t_rotation_shuttle, t_CNOT, t_readout, mean_wait)
    b = sort(0.3 .* rand(rng, 4)); b .-= b[1]
    c = fill(NaN, 4); g = fill(NaN, 4)
    current = b[1] + t_rotation_shuttle
    for k in 2:4
        c[k] = max(current, b[k]) + t_rotation_shuttle + mean_wait * randexp(rng)
        g[k] = c[k] + t_CNOT
        current = g[k] + t_readout
    end
    tM = current + mean_wait * randexp(rng)
    return b, c, g, tM, tM + t_readout
end

@testset "closed form vs density-matrix trace" begin

    @testset "isolated_timeline (trace) == trace_timeline (closed-form file)" begin
        @test all(1:10) do _
            births = 0.3 .* rand(rng, 4)
            all(isequal.(isolated_timeline(births; TK...), trace_timeline(births; TK...)))
        end
    end

    # A) No gate/readout errors: both methods are deterministic -> agreement to round-off.
    @testset "exact: isolated timelines (old entry point)" begin
        for _ in 1:10
            births = 0.3 .* rand(rng, 4)
            tk = (t_rotation_shuttle = 0.01rand(rng), t_CNOT = 0.05rand(rng), t_readout = 0.05rand(rng))
            Fl = 0.9 + 0.1rand(rng)
            F_tr = simulate_piecemaker_trace(births; T_coherence = τ, F_link = Fl, tk...,
                                             F_CNOT = 1.0, F_readout = 1.0, rng)
            F_cf = ghz_fidelity_closedform(trace_timeline(births; tk...)...; T_coherence = τ, F_link = Fl)
            @test isapprox(F_tr, F_cf; atol = 1e-9)
        end
    end

    @testset "exact: queued timelines" begin
        for _ in 1:10
            tk = (t_rotation_shuttle = 0.01rand(rng), t_CNOT = 0.05rand(rng), t_readout = 0.05rand(rng))
            tl = queued_timeline(rng; tk..., mean_wait = 0.05)
            Fl = 0.9 + 0.1rand(rng)
            F_tr = simulate_piecemaker_timeline(tl...; T_coherence = τ, F_link = Fl,
                                                F_CNOT = 1.0, F_readout = 1.0, rng)
            F_cf = ghz_fidelity_closedform(tl...; T_coherence = τ, F_link = Fl)
            @test isapprox(F_tr, F_cf; atol = 1e-9)
        end
    end

    # B) Gate and readout errors are sampled by the trace: compare means within 4 SEM.
    @testset "statistical: gate + readout errors on a queued timeline" begin
        tl = queued_timeline(rng; t_rotation_shuttle = 0.005, t_CNOT = 0.02, t_readout = 0.03, mean_wait = 0.05)
        pars = (F_link = 0.97, F_CNOT = 0.97, F_readout = 0.98)
        samples = [simulate_piecemaker_timeline(tl...; T_coherence = τ, pars..., rng) for _ in 1:N_STAT]
        F_cf = ghz_fidelity_closedform(tl...; T_coherence = τ, pars...)
        sem = std(samples) / sqrt(N_STAT)
        @info "gate/readout check" trace_mean = mean(samples) sem closed_form = F_cf z = (mean(samples) - F_cf) / sem
        @test abs(mean(samples) - F_cf) < 4sem
    end

    # C) Real event-simulation output: logging invariants, method agreement, queueing cost.
    if SIM_WALL > 0
        @testset "event simulation ($(code))" begin
            result = run_single_configuration(; SIMPARS..., seed = 1234,
                                              target_samples = 100, max_wallclock = SIM_WALL)
            df = result.raw_events
            @info "simulated GHZ states" n = nrow(df) converged = result.converged
            @test nrow(df) > 0

            @testset "logging invariants" begin
                tls = [row_timeline(r) for r in eachrow(df)]
                isos = [isolated_timeline(tl[1]; TK...) for tl in tls]
                # fusion order equals FIFO arrival order
                @test all(issorted(tl[1]) for tl in tls)
                # each fusion lasts exactly Δt_CNOT; consumption = piecemaker readout + Δt_readout
                @test all(all(isapprox.(tl[3][2:4] .- tl[2][2:4], SIMPARS.Δt_CNOTgate; atol = 1e-12)) for tl in tls)
                @test all(isapprox(tl[5] - tl[4], SIMPARS.Δt_readout; atol = 1e-12) for tl in tls)
                # contention can only delay: never earlier than the isolated schedule
                @test all(all(tl[2][2:4] .- tl[1][1] .>= iso[2][2:4] .- 1e-12) for (tl, iso) in zip(tls, isos))
                @test all(tl[4] - tl[1][1] >= iso[4] - 1e-12 for (tl, iso) in zip(tls, isos))
            end

            noise = (T_coherence = 1.0, F_link = 0.97)   # memory + link only -> both methods exact
            sub = first(df, min(nrow(df), 25))

            @testset "closed form == trace on logged timelines" begin
                F_tr = simulate_piecemaker_from_log(sub; noise..., F_CNOT = 1.0, F_readout = 1.0, rng)
                F_cf = ghz_fidelities_from_log(sub; noise...)
                @test all(isapprox.(F_tr, F_cf; atol = 1e-9))
            end

            @testset "queueing cost: trace and closed form give the same ΔF" begin
                ΔF_tr = [simulate_piecemaker_trace(row_timeline(r)[1]; noise..., TK..., F_CNOT = 1.0, F_readout = 1.0, rng) -
                         simulate_piecemaker_timeline(row_timeline(r)...; noise..., F_CNOT = 1.0, F_readout = 1.0, rng)
                         for r in eachrow(sub)]
                ΔF_cf = [ghz_fidelity_closedform(isolated_timeline(row_timeline(r)[1]; TK...)...; noise...) -
                         ghz_fidelity_closedform(row_timeline(r)...; noise...) for r in eachrow(sub)]
                @test all(isapprox.(ΔF_tr, ΔF_cf; atol = 1e-9))
            end

            # Report (not a test): fidelity cost of queueing over the whole run, closed form.
            F_logged = ghz_fidelities_from_log(df; noise..., F_CNOT = SIMPARS.gate_fidelity)
            F_isol = [ghz_fidelity_closedform(isolated_timeline(row_timeline(r)[1]; TK...)...;
                                              noise..., F_CNOT = SIMPARS.gate_fidelity) for r in eachrow(df)]
            waits = reduce(vcat, [begin
                        b, c, _, tM, _ = row_timeline(r)
                        _, c_iso, _, tM_iso, _ = isolated_timeline(b; TK...)
                        [(c[2:4] .- b[1]) .- c_iso[2:4]; (tM - b[1]) - tM_iso]
                    end for r in eachrow(df)])
            ΔF = F_isol .- F_logged
            @info "queueing cost (isolated − logged), T_coherence = $(noise.T_coherence) s" mean_F_isolated = mean(F_isol) mean_F_logged = mean(F_logged) mean_ΔF = mean(ΔF) q90_ΔF = quantile(ΔF, 0.9) max_ΔF = maximum(ΔF) frac_ops_delayed = mean(waits .> 1e-12) mean_extra_wait_s = mean(waits) q90_extra_wait_s = quantile(waits, 0.9)
        end
    end
end
