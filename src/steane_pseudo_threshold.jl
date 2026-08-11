using QuantumClifford
using QuantumClifford.ECC: Steane7, TableDecoder
import CairoMakie

const QC = QuantumClifford
const ECC = QuantumClifford.ECC

# ---------------------------------------------------------------------
# [[12,2,3]] code from Andersen-Greplová 
#
# Row order here:
#   first 6 rows: Z stabilizers, HZ
#   last  6 rows: X stabilizers, HX
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

function make_decoder_figure(phys_errors, results; title="", labels=String[])
    fresults = copy(results)
    fresults[fresults .== 0] .= NaN

    nonzero_results = fresults[.!isnan.(fresults)]
    isempty(nonzero_results) && error(
        "All logical error estimates are zero. Increase nsamples or physical error range."
    )

    all_positive_vals = vcat(collect(phys_errors), collect(nonzero_results))

    minlim = 10^-3 #minimum(all_positive_vals)
    maxlim = min(1.0, maximum(all_positive_vals))

    f = CairoMakie.Figure()
    a = CairoMakie.Axis(
        f[1, 1],
        xscale = log10,
        yscale = log10,
        # limits = (minlim, maxlim, minlim, maxlim),
        aspect = CairoMakie.DataAspect(),
        xlabel = "physical error rate",
        ylabel = "logical error rate",
        title = title,
    )

    CairoMakie.lines!(
        a,
        [minlim, maxlim],
        [minlim, maxlim],
        color = :black,
        label = "pL = p",
    )

    #plot fault tolerant pL=A*p^2
    p = 10 .^ range(log10(minlim), log10(maxlim), length=100)
    CairoMakie.lines!(
        a,
        p,
        p.^2,
        color = :red,
        linestyle = :dash,
        label = "pL = p^2",
    )

    for (i, sresults) in enumerate(eachslice(fresults, dims = 1))
        label_base = isempty(labels) ? "curve $i" : labels[i]

        CairoMakie.scatter!(
            a,
            phys_errors,
            sresults[:, 1],
            marker = :circle,
            color = CairoMakie.Cycled(i),
            label = "$label_base: logical X",
        )

        CairoMakie.scatter!(
            a,
            phys_errors,
            sresults[:, 2],
            marker = :xcross,
            color = CairoMakie.Cycled(i),
            label = "$label_base: logical Z",
        )
    end

    CairoMakie.Legend(f[1, 2], a, "Legend")
    return f
end
##
# ---------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------

function add_two_qubit_noise_after_gates(circuit, two_qubit_gate_noise)
    noisy_circuit = QC.AbstractOperation[]

    for op in circuit
        push!(noisy_circuit, op)

        if two_qubit_gate_noise > 0 && op isa QC.AbstractTwoQubitOperator
            qs = QC.affectedqubits(op)
            push!(noisy_circuit, QC.PauliError(qs, two_qubit_gate_noise))
        end
    end

    return noisy_circuit
end

function data_memory_noise_circuit(H, mem_noise)
    n_data = QC.nqubits(H)

    return QC.AbstractOperation[
        QC.PauliError(q, mem_noise) for q in 1:n_data
    ]
end

function ghz_ancilla_noise_circuit(H, n_anc, ghz_noise)
    n_data = QC.nqubits(H)

    return QC.AbstractOperation[
        QC.PauliError(q, ghz_noise) for q in (n_data + 1):(n_data + n_anc)
    ]
end

# ---------------------------------------------------------------------
# Noisy GHZ / Shor-style physical ECC circuit
#
# This is the key model:
#
#   perfect GHZ preparation
#   + Pauli noise on every GHZ qubit
#   + memory noise on data
#   + syndrome-measurement circuit
#   + optional two-qubit gate noise on data-GHZ coupling gates
# ---------------------------------------------------------------------

function my_noisy_ghz_physical_circuit(
    H;
    mem_noise,
    two_qubit_gate_noise,
    ghz_noise,
)
    # Built-in Shor/GHZ-style syndrome circuit.
    #
    # prep_anc: perfect GHZ/cat preparation
    # syndrome_circ: data-GHZ interaction, GHZ measurement, XOR syndrome bits
    # n_anc: number of GHZ ancilla qubits
    # syndrome_bits: final syndrome-bit indices
    prep_anc, syndrome_circ, n_anc, syndrome_bits = ECC.shor_syndrome_circuit(H)

    # Make the GHZ imperfect:
    # ideal GHZ + independent Pauli error on each GHZ qubit.
    ghz_noise_circ = ghz_ancilla_noise_circuit(H, n_anc, ghz_noise)

    # Memory noise on data qubits.
    mem_error_circ = data_memory_noise_circuit(H, mem_noise)

    # Optional noisy data-GHZ coupling gates.
    noisy_syndrome_circ = add_two_qubit_noise_after_gates(
        syndrome_circ,
        two_qubit_gate_noise,
    )

    circ = vcat(
        prep_anc,
        ghz_noise_circ,
        mem_error_circ,
        noisy_syndrome_circ,
    )

    return circ, syndrome_bits, n_anc
end

# Same as above, but with perfect GHZ states for comparison.
function my_perfect_ghz_physical_circuit(
    H;
    mem_noise,
    two_qubit_gate_noise,
    ghz_noise = 0.0,
)
    return my_noisy_ghz_physical_circuit(
        H;
        mem_noise = mem_noise,
        two_qubit_gate_noise = two_qubit_gate_noise,
        ghz_noise = 0.0,
    )
end

# ---------------------------------------------------------------------
# Lower-level decoder evaluation
#
# This mirrors evaluate_decoder(decoder, setup, nsamples), but with our
# custom physical circuit instead of NaiveSyndromeECCSetup or
# ShorSyndromeECCSetup.
# ---------------------------------------------------------------------

function my_evaluate_decoder(
    decoder;
    physical_circuit_builder,
    mem_noise,
    two_qubit_gate_noise,
    ghz_noise,
    nsamples::Int,
)
    H = ECC.parity_checks(decoder)

    n = QC.nqubits(H)
    O = ECC.faults_matrix(H)

    physical_noisy_circ, syndrome_bits, n_anc =
        physical_circuit_builder(
            H;
            mem_noise = mem_noise,
            two_qubit_gate_noise = two_qubit_gate_noise,
            ghz_noise = ghz_noise,
        )

    # Perfect encoding, same as QuantumClifford's built-in evaluator.
    encoding_circ = ECC.naive_encoding_circuit(H)

    # Logical observables.
    mdH = QC.MixedDestabilizer(H)
    logX = QC.logicalxview(mdH)
    logZ = QC.logicalzview(mdH)

    k = size(logX, 1)

    # Final noiseless logical measurement circuits.
    logX_circ, _, logX_bits =
        ECC.naive_syndrome_circuit(logX, n_anc + 1, last(syndrome_bits) + 1)

    logZ_circ, _, logZ_bits =
        ECC.naive_syndrome_circuit(logZ, n_anc + 1, last(syndrome_bits) + 1)

    # Logical X failure:
    # measure logical Z and compare against decoded prediction.
    X_error = ECC.evaluate_decoder(
        decoder,
        nsamples,
        vcat(encoding_circ, physical_noisy_circ, logZ_circ),
        syndrome_bits,
        logZ_bits,
        O[end ÷ 2 + 1:end, :],
    )

    # Logical Z failure:
    # prepare logical |+⟩, then measure logical X.
    preX = QC.AbstractOperation[
        QC.sHadamard(i) for i in n-k+1:n
    ]

    Z_error = ECC.evaluate_decoder(
        decoder,
        nsamples,
        vcat(preX, encoding_circ, physical_noisy_circ, logX_circ),
        syndrome_bits,
        logX_bits,
        O[1:end ÷ 2, :],
    )

    return (X_error, Z_error)
end
##
# ---------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------

code = Steane7()
decoder = TableDecoder(code)

phys_errors = 10 .^ range(-3, 0, length = 20)
nsamples = 1_000_000

experiments = [
    ("perfect GHZ", my_perfect_ghz_physical_circuit),
    #("noisy GHZ", my_noisy_ghz_physical_circuit),
]

results = zeros(length(experiments), length(phys_errors), 2)

for (iexp, (name, builder)) in pairs(experiments)
    println("Running experiment: $name")

    for (i, p) in pairs(phys_errors)

        r = my_evaluate_decoder(
            decoder;
            physical_circuit_builder = builder,
            mem_noise = p,
            two_qubit_gate_noise = 0.0,
            ghz_noise = p,
            nsamples = nsamples,
        )

        results[iexp, i, :] .= r

        println(
            "  p = $p -> logical X = $(r[1]), logical Z = $(r[2])"
        )
    end
end
##
f = make_decoder_figure(
    phys_errors,
    results;
    title = "Steane7 with perfect vs imperfect GHZ syndrome extraction",
    labels = first.(experiments),
)

display(f)

##
# ---------------------------------------------------------------------
# 2D sweep: memory noise vs GHZ imperfection
# ---------------------------------------------------------------------

function logical_error_metric(r; mode = :max, k=1)
    if mode == :max
        return 1-(1-max(r[1], r[2]))^*(1/k)
    elseif mode == :mean
        return 0.5 * (r[1] + r[2])
    else
        error("Unknown mode: $mode")
    end
end

function run_mem_vs_ghz_sweep(
    decoder;
    mem_grid,
    ghz_grid,
    nsamples::Int = 10_000,
    two_qubit_gate_noise_function = p_mem -> 0.0,
    metric = :max,
    k=1,
)

    logical_errors = zeros(length(mem_grid), length(ghz_grid))
    threshold_ratio = zeros(length(mem_grid), length(ghz_grid))

    for (j, p_ghz) in pairs(ghz_grid)
        println("GHZ noise = $p_ghz")

        for (i, p_mem) in pairs(mem_grid)
            p_2q = two_qubit_gate_noise_function(p_mem)

            r = my_evaluate_decoder(
                decoder;
                physical_circuit_builder = my_noisy_ghz_physical_circuit,
                mem_noise = p_mem,
                two_qubit_gate_noise = p_2q,
                ghz_noise = p_ghz,
                nsamples = nsamples,
            )

            pL = logical_error_metric(r; mode = metric, k=k)

            logical_errors[i, j] = pL
            threshold_ratio[i, j] = pL / p_mem

            println(
                "  p_mem = $p_mem, p_ghz = $p_ghz -> ",
                "logical X = $(r[1]), logical Z = $(r[2]), pL/p_mem = $(pL / p_mem)"
            )
        end
    end

    return logical_errors, threshold_ratio
end

mem_grid = 10 .^ range(-3, -0.5, length = 50)
ghz_grid = 10 .^ range(-5, -1, length = 50)

nsamples = 100_000

code = Steane7()
decoder = TableDecoder(code)

logical_errors, threshold_ratio = run_mem_vs_ghz_sweep(
    k = 1,
    decoder;
    mem_grid = mem_grid,
    ghz_grid = ghz_grid,
    nsamples = nsamples,

    # This isolates memory noise + GHZ imperfection.
    two_qubit_gate_noise_function = p_mem -> 0.0,

    # Conservative choice: use worse of logical X and logical Z.
    metric = :max,
)
##
using DataFrames
using CSV
using JLD2
code_name = "Steane7"
df = DataFrame(
    mem_noise = repeat(mem_grid, outer = length(ghz_grid)),
    ghz_noise = repeat(ghz_grid, inner = length(mem_grid)),
    logical_error = vec(logical_errors),
    threshold_ratio = vec(threshold_ratio),
    metric = fill("max", length(logical_errors)),
    code = fill(code_name, length(logical_errors)),
    k = fill(1, length(logical_errors)),
)
@save "pseudo_threshold_heatmap_$(code_name)_data.jld2" mem_grid ghz_grid logical_errors threshold_ratio
@save "pseudo_threshold_heatmap_$(code_name)_data.csv" df

##
function local_depolarized_ghz_infidelity(p, n)
    F = 0.5 * ((1 - 2p/3)^n + (1 - 4p/3)^n) +
        2^(n - 1) * (p/3)^n
    return 1 - F
end

function make_threshold_heatmap(code_name, mem_grid, ghz_grid, threshold_ratio;
                                nsamples = nsamples,
                                n_anc::Int,
                                T_ms::Union{Vector, Nothing} = nothing)  # <-- new optional param

    infidelity_grid = local_depolarized_ghz_infidelity.(ghz_grid, 4)

    ratio_floor = (0.5 / nsamples) ./ mem_grid
    ratio_for_plot = copy(threshold_ratio)

    for i in eachindex(mem_grid), j in eachindex(ghz_grid)
        ratio_for_plot[i, j] = max(ratio_for_plot[i, j], ratio_floor[i])
    end

    heat_values = log10.(ratio_for_plot)

    # Convert x axis to time if T_ms is provided
    x_grids = []
    if !isnothing(T_ms)
        for T in T_ms
            push!(x_grids, 1 ./ (-T .* log.(1 .- 4 .* mem_grid ./ 3)) * 1e3)
        end
        x_label = "GHZ delivery rate (Hz)"
    else
        push!(x_grids, mem_grid)
        x_label = "memory noise p_mem"
    end

    f = CairoMakie.Figure(size = (700, length(x_grids) * 500))


    for (i, x_grid) in pairs(x_grids)
        
        a = CairoMakie.Axis(
            f[i, 1],
            xscale = log10,
            yscale = log10,
            xlabel = x_label,
            ylabel = "GHZ state infidelity 1 - F",
            title = isnothing(T_ms) ? "$(code_name): Logical error vs memory noise and noisy GHZ states" :
        "$(code_name): Logical error over GHZ delivery rate and GHZ infidelity(τ = $(T_ms[i]) ms)",
        )

        hm = CairoMakie.heatmap!(
            a,
            x_grid,
            infidelity_grid,
            heat_values,
            colormap = :RdYlGn_11,
            colorrange = (-maximum(abs.(heat_values)), maximum(abs.(heat_values))),
        )

        CairoMakie.Colorbar(
            f[i, 2],
            hm,
            label = "log10(logical error / memory noise)",
        )
    end

    return f
end

##
code_name = "steane7"
#@load "pseudo_threshold_heatmap_$(code_name)_data.jld2"

Tms_vec = [1000.0, 100.0]
f_heat = make_threshold_heatmap(
    code_name,
    mem_grid,
    ghz_grid,
    threshold_ratio,
    nsamples = nsamples,
    n_anc = 4,
    T_ms = nothing
)

save("pseudo_threshold_heatmap_$(code_name).pdf", f_heat)
