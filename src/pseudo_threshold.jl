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
    @info "Adding two-qubit gate depolarization noise with λ = $λ_gate to gate $g on qubits $qubits"

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

## EVALUATION AND PLOTTING
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
        xlabel_str = L"Data-qubit Pauli error probability $p_{\mathrm{mem}}$"

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

    ymax = min(1.0, max(maximum(positive_results) * 2, maximum(p_axis) * 2))

    plt = plot(
        xscale = :log10,
        yscale = :log10,
        xlims = (1.05*xmin, 1.1*xmax),
        ylims = (1e-4, ymax),
        xlabel = L"GHZ Infidelity $1-F_{\mathrm{GHZ}}$", ##xlabel_str,
        ylabel = L"Logical error rate $p_L$",
        title = title,
        legend = :bottomright,
        size = (600, 500),
        grid = true,
        margin = 5mm,
        tickfontsize=12,
        labelfontsize=14,
        legendfontsize=12,
        minorgrid = true,
    )

    # Pseudothreshold reference line.
    # This always means pL = p_mem.
    # plot!(
    #     plt,
    #     x_axis,
    #     p_axis,
    #     color = :black,
    #     label = L"$p_L = p_{\mathrm{mem}}$",)

    hline!(
    plt,
    [0.1],
    linestyle = :dash,
    color = :black,
    label = L"$p_L = 10^{-1}$",)
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
            label = "$label_base:"*L"logical $X$",
            color = color_i,
        )

        scatter!(
            plt,
            x_axis,
            fresults[i, :, 2],
            marker = :xcross,
            label = "$label_base:"*L"logical $Z$",
            color = color_i,
        )
    end

    savefig(plt, "pseudothresholds_GHZ0_999.pdf")
    return plt
end

##
F_gate = 1.0
mem_errors = 10 .^ range(-4, -1, length=20)
codess = [Steane7(), BB12_2_3]
results = zeros(length(codess), length(mem_errors), 2)
fidelities = [1.0 - 2.5^(-x) for x in 4.0:12.0]

for (ic, c) in pairs(codess)
    for (i,m) in pairs(fidelities)
        setup = CShorSyndromeECCSetup(0.1, F_gate, m)
        decoder = TableDecoder(c)
        r = cevaluate_decoder(decoder, setup, 100_000)
        results[ic,i,:] .= r
    end
end

##

make_decoder_figure(1.0 .-fidelities, 
results; 
title = "", 
labels = ["[[7,1,3]]", "[[12,2,3]]"],
xaxis = :error)  # 100 ms 


##

# ---------------------------------------------------------------------
# 2D sweep: memory error × GHZ fidelity
# ---------------------------------------------------------------------

F_gate = 1.0
mem_errors = 10 .^ range(-4, -1, length=20)

fidelities = [
    1.0 - 2.5^(-x)
    for x in range(4.0, 12.0, length=20)
]

ghz_infidelities = 1.0 .- fidelities

codess = [Steane7()]#, BB12_2_3]

nsamples = 100_000

# dimensions:
# code × memory error × GHZ fidelity × logical error type
#
# last index:
#   1 = logical X error
#   2 = logical Z error
results_heatmap = zeros(
    length(codess),
    length(mem_errors),
    length(fidelities),
    2,
)

for (ic, c) in pairs(codess)

    decoder = TableDecoder(c)

    for (imem, p_mem) in pairs(mem_errors)
        for (ighz, F_GHZ) in pairs(fidelities)

            setup = CShorSyndromeECCSetup(
                p_mem,
                F_gate,       # two-qubit gate fidelity
                F_GHZ,
            )

            X_error, Z_error =
                cevaluate_decoder(
                    decoder,
                    setup,
                    nsamples,
                )

            results_heatmap[ic, imem, ighz, 1] = X_error
            results_heatmap[ic, imem, ighz, 2] = Z_error
        end
    end
end
##
function make_ratio_heatmap(
    mem_errors,
    fidelities,
    ratio;
    T_coh = nothing,
    title = "",
    nsamples = 100_000,
)

    ghz_infidelities = 1.0 .- fidelities

    # Choose y-axis representation
    if isnothing(T_coh)
        yvals = mem_errors
        ylabel_str = L"Memory error probability $p_{\mathrm{mem}}$"
    else
        yvals = -T_coh .* log.(1.0 .- (4.0 / 3.0) .* mem_errors)
        ylabel_str = L"GHZ generation time $\Delta t_{\mathrm{GHZ}}\;[\mathrm{s}]$"
    end

    # Sort axes
    xorder = sortperm(ghz_infidelities)
    yorder = sortperm(yvals)

    x = ghz_infidelities[xorder]
    y = yvals[yorder]

    z = ratio[yorder, xorder]

    # p_mem remains needed for the finite-sampling floor,
    # irrespective of which quantity is shown on the y-axis
    p_mem_sorted = mem_errors[yorder]

    # Smallest resolvable pL is approximately 1/nsamples
    min_ratio = (1 / nsamples) ./ reshape(p_mem_sorted, :, 1)

    z_plot = max.(z, min_ratio)

    logratio = log10.(z_plot)
    maxL = maximum(abs, logratio)

    plt = heatmap(
        x,
        y,
        logratio,

        xscale = :log10,
        yscale = :log10,

        xlabel = L"GHZ infidelity $1-F_{\mathrm{GHZ}}$",
        ylabel = ylabel_str,

        colorbar_title = "",

        clim = (-maxL, maxL),

        colormap = :RdYlGn_11,

        title = title * L"\qquad\log_{10}(p_L/p_{\mathrm{mem}})",

        size = (650, 500),
        margin = 5mm,

        tickfontsize = 11,
        labelfontsize = 13,
        titlefontsize = 13,
    )

    contour!(
        plt,
        x,
        y,
        logratio,
        levels = [0.0],
        linestyle = :dash,
        linewidth = 2,
        color = :grey,
        colorbar = true,
        label = "",
    )

    return plt
end

code_labels = [
    L"[[7,1,3]]",
    L"[[12,2,3]]",
]

plots = []
for ic in eachindex(codess)

    pL_X = results_heatmap[ic, :, :, 1]
    pL_Z = results_heatmap[ic, :, :, 2]

    # Conservative logical error probability
    pL = max.(pL_X, pL_Z)

    # Ratio pL / p_mem
    ratio = pL ./ reshape(mem_errors, :, 1)

    plt = make_ratio_heatmap(
        mem_errors,
        fidelities,
        ratio;
        T_coh = 1.0,
        title = code_labels[ic],
        nsamples = 100_000,
    )
    push!(plots, plt)
end

fig = plot(
    plots...,
    layout = (2, 1),
    size = (700, 1000),
    margin = 5mm,
)

display(fig)

savefig(fig, "pseudothresholds_heatmaps.pdf")