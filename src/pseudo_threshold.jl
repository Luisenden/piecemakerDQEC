# see tutorial at https://qc.quantumsavory.org/stable/ECC_evaluating/

using Plots
using Colors

error_rate_to_time(p, τ) = -τ * log1p(-4p/3)
time_to_error_rate(t, τ) = 3/4 * (1 - exp(-t/τ))
rate_to_error_rate(r, τ) = 3/4 * (1 - exp(-1/(r * τ)))

function make_decoder_figure(
    phys_errors,
    results;
    title = "",
    labels = String[],
    τ = nothing,
    xaxis = :error,   # :error, :time, or :rate
)
    fresults = copy(results)
    fresults[fresults .== 0] .= NaN

    p_axis = copy(phys_errors)

    if xaxis == :error
        x_axis = p_axis
        xlabel_str = "data-qubit storage Pauli error probability"

    elseif xaxis == :time
        isnothing(τ) && error("You need to provide τ when xaxis = :time.")

        valid = (p_axis .> 0) .& (p_axis .< 3/4)

        p_axis = p_axis[valid]
        fresults = fresults[:, valid, :]

        x_axis = error_rate_to_time.(p_axis, τ)
        xlabel_str = "storage time Δt [s]"

    elseif xaxis == :rate
        isnothing(τ) && error("You need to provide τ when xaxis = :rate.")

        valid = (p_axis .> 0) .& (p_axis .< 3/4)

        p_axis = p_axis[valid]
        fresults = fresults[:, valid, :]

        Δt_axis = error_rate_to_time.(p_axis, τ)
        x_axis = 1 ./ Δt_axis
        xlabel_str = "storage rate 1/Δt [s⁻¹]"

        # Sort so the reference line is drawn nicely from left to right.
        order = sortperm(x_axis)
        x_axis = x_axis[order]
        p_axis = p_axis[order]
        fresults = fresults[:, order, :]

    else
        error("xaxis must be :error, :time, or :rate.")
    end

    positive_results = fresults[.!isnan.(fresults)]

    if isempty(positive_results)
        error("All logical error estimates are zero. Increase nsamples or physical error range.")
    end

    xmin = minimum(x_axis)
    xmax = maximum(x_axis)

    ymin = max(1e-6, minimum(positive_results) / 2)
    ymax = min(1.0, max(maximum(positive_results) * 2, maximum(p_axis) * 2))

    plt = plot(
        xscale = :log10,
        yscale = :log10,
        xlims = (xmin, xmax),
        ylims = (ymin, ymax),
        xlabel = xlabel_str,
        ylabel = "logical error rate",
        title = title,
        legend = :outerright,
        size = (800, 500),
        grid = true,
    )

    # Pseudothreshold reference line.
    # This always means pL = p_mem.
    plot!(
        plt,
        x_axis,
        p_axis,
        color = :black,
        label = "pL = p_mem",
    )

    colors = [
        RGB(31/255, 119/255, 180/255),
        RGB(255/255, 127/255, 14/255),
        RGB(44/255, 160/255, 44/255),
        RGB(214/255, 39/255, 40/255),
        RGB(148/255, 103/255, 189/255),
        RGB(140/255, 86/255, 75/255),
        RGB(227/255, 119/255, 194/255),
        RGB(127/255, 127/255, 127/255),
        RGB(188/255, 189/255, 34/255),
        RGB(23/255, 190/255, 207/255),
    ]

    ncurves = size(fresults, 1)

    for i in 1:ncurves
        label_base = isempty(labels) ? "curve $i" : labels[i]
        color_i = colors[mod1(i, length(colors))]

        scatter!(
            plt,
            x_axis,
            fresults[i, :, 1],
            marker = :circle,
            label = "$label_base: logical X",
            color = color_i,
        )

        scatter!(
            plt,
            x_axis,
            fresults[i, :, 2],
            marker = :xcross,
            label = "$label_base: logical Z",
            color = color_i,
        )
    end

    return plt
end
##
# find the source code for the ShorSyndromeECCSetup struct in the QuantumClifford.ECC module https://raw.githubusercontent.com/QuantumSavory/QuantumClifford.jl/master/src/ecc/decoder_pipeline.jl
using QuantumClifford.ECC: AbstractECCSetup, AbstractSyndromeDecoder

struct CShorSyndromeECCSetup <: AbstractECCSetup
    mem_noise::Float64
    two_qubit_gate_noise::Float64
    ancilla_noise::Float64 # this is new

    function CShorSyndromeECCSetup(mem_noise, two_qubit_gate_noise, ancilla_noise)
        0 <= mem_noise <= 1 ||
            throw(DomainError(mem_noise,
                "The memory noise in `CShorSyndromeECCSetup` should be between 0 and 1."))

        0 <= two_qubit_gate_noise <= 1 ||
            throw(DomainError(two_qubit_gate_noise,
                "The two-qubit gate noise in `CShorSyndromeECCSetup` should be between 0 and 1."))

        0 <= ancilla_noise <= 1 ||
            throw(DomainError(ancilla_noise,
                "The ancilla noise in `CShorSyndromeECCSetup` should be between 0 and 1."))

        new(mem_noise, two_qubit_gate_noise, ancilla_noise)
    end
end

function add_two_qubit_gate_noise(g, gate_error)
    return ()
end

function add_two_qubit_gate_noise(g::AbstractTwoQubitOperator, gate_error)
    qubits = affectedqubits(g)
    return (PauliError(qubits, gate_error),)
end

function physical_ECC_circuit(H, setup::CShorSyndromeECCSetup)
    prep_anc, syndrome_circ, n_anc, syndrome_bits = shor_syndrome_circuit(H)

    noisy_syndrome_circ = []
    for op in syndrome_circ
        push!(noisy_syndrome_circ, op)
        for noise_op in add_two_qubit_gate_noise(op, setup.two_qubit_gate_noise)
            push!(noisy_syndrome_circ, noise_op)
        end
    end

    # this is new: add noise to the ancilla qubits after preparation
    noisy_ancilla_circ = []
    if setup.ancilla_noise > 0
        for i in nqubits(H)+1 : nqubits(H)+n_anc
            push!(noisy_ancilla_circ, PauliError(i, setup.ancilla_noise))
        end
    end

    mem_error_circ = [PauliError(i, setup.mem_noise) for i in 1:nqubits(H)]

    circ = vcat(prep_anc, mem_error_circ, noisy_ancilla_circ, noisy_syndrome_circ)

    circ, syndrome_bits, n_anc
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
##
##
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

mem_errors = 10 .^ range(-3, 0, length=20)
codes = [Shor9(), Steane7(), BB12_2_3]
results = zeros(length(codes), length(mem_errors), 2)

for (ic, c) in pairs(codes)
    for (i,m) in pairs(mem_errors)
        setup = CShorSyndromeECCSetup(m, 0, 0)
        decoder = TableDecoder(c)
        r = cevaluate_decoder(decoder, setup, 100_000)
        results[ic,i,:] .= r
    end
end

##

make_decoder_figure(mem_errors, 
results; 
title = "Pseudothreshold for Shor, Steane, and [[12,2,3]] code", 
labels = ["[[9,1,3]]", "[[7,1,3]]", "[[12,2,3]]"],
τ = 100e-3,
xaxis = :rate)  # 100 ms 
