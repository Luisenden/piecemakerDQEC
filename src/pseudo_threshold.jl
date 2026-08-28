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
using QuantumClifford.ECC: batchdecode, evaluate_guesses, faults_matrix

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


function stabilizer_to_css(H::Stabilizer)
    n = nqubits(H)

    xrows = Vector{Vector{Bool}}()
    zrows = Vector{Vector{Bool}}()

    for g in H
        xs = Bool[g[q][1] for q in 1:n]
        zs = Bool[g[q][2] for q in 1:n]

        if any(xs) && !any(zs)
            # X-type stabilizer
            push!(xrows, xs)

        elseif any(zs) && !any(xs)
            # Z-type stabilizer
            push!(zrows, zs)

        else
            error("Generator is not purely X-type or Z-type: $g")
        end
    end

    Hx = reduce(vcat, permutedims.(xrows))
    Hz = reduce(vcat, permutedims.(zrows))

    return CSS(Hx, Hz)
end

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
    @debug "Adding two-qubit gate depolarization noise with λ = $λ_gate to gate $g on qubits $qubits"

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


function cevaluate_decoder_pLXZ(
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

    # xmin = minimum(x_axis)
    xmax = maximum(x_axis)

    # ymax = min(1.0, max(maximum(positive_results) * 2, maximum(p_axis) * 2))

    plt = plot(
        xscale = :log10,
        yscale = :log10,
        xlims = (0.004281332398719396, 1.0),
        ylims = (1.e-6, 1.05),
        xlabel = xlabel_str, #L"GHZ Infidelity $1-F_{\mathrm{GHZ}}$", #
        ylabel = L"Logical error rate $\widehat{p}_L$",
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
    plot!(
        plt,
        x_axis,
        p_axis,
        color = :black,
        label = L"$\widehat{p}_L = p_{\mathrm{mem}}$",)

    # hline!(
    # plt,
    # [0.01],
    # linestyle = :dash,
    # color = :black,
    # label = L"$\widehat{p}_L = p_{\mathrm{mem}}$",)

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

        plot!(
            plt,
            x_axis,
            fresults[i, :],
            linewidth = 2,
            marker = :circle,
            label = "$label_base",
            color = color_i,
        )
    end

    savefig(plt, "pseudothresholds_GHZ0_999.pdf")
    return plt
end
## 1D sweep: GHZ fidelity / memory error

codess = [
    Steane7(),
    stabilizer_to_css(BB12_2_3),
    stabilizer_to_css(GB26_2_5),
]

F_gate = 1.0
mem_errors = 10 .^ range(-3, 0, length=20) # :1

fidelities = [ # :2
    1.0 - 2.5^(-x)
    for x in range(4.0, 12.0, length=20)
]

ghz_infidelities = 1.0 .- fidelities

nsamples = 2000_000

# dimensions:
# code × GHZ fidelity / memory error
#
results = zeros(
    length(codess), 20)

for (ic, c) in pairs(codess)

    decoder = CSSTableDecoder(c; error_weight=3)

    for (ivar, var) in pairs(mem_errors) # :1 or :2

        setup = CShorSyndromeECCSetup(
            var,
            F_gate,       # two-qubit gate fidelity
            1.0,
        )

        pL =
            cevaluate_decoder_pL(
                decoder,
                setup,
                nsamples,
            )

        results[ic, ivar] = pL
    end
end
##
make_decoder_figure(mem_errors, 
results; 
title = "", 
labels = ["[[7,1,3]]", "[[12,2,3]]", "[[26,2,5]]"],
xaxis = :error)  # 100 ms 


##

# ---------------------------------------------------------------------
# 2D sweep: memory error × GHZ fidelity (this can take up to 1hour for 1M samples)
# ---------------------------------------------------------------------

F_gate = 1.0
mem_errors = 10 .^ range(log10(1e-4), log10(3/4), length=20)

fidelities = [
    1.0 - 2.5^(-x)
    for x in range(3.0, 10.0, length=20)
]

ghz_infidelities = 1.0 .- fidelities

codess = [
    Steane7(),
    stabilizer_to_css(BB12_2_3),
    stabilizer_to_css(GB26_2_5),
]

nsamples = 1000_000

# dimensions:
# code × memory error × GHZ fidelity
#
results_heatmap = zeros(
    length(codess),
    length(mem_errors),
    length(fidelities),
)

start = time()
for (ic, c) in pairs(codess)

    decoder = CSSTableDecoder(c; error_weight = 3)

    for (imem, p_mem) in pairs(mem_errors)
        for (ighz, F_GHZ) in pairs(fidelities)

            setup = CShorSyndromeECCSetup(
                p_mem,
                F_gate,       # two-qubit gate fidelity
                F_GHZ,
            )

            pL =
                cevaluate_decoder_pL(
                    decoder,
                    setup,
                    nsamples,
                )

            results_heatmap[ic, imem, ighz] = pL

        end
        @info "Code $ic, p_mem = $p_mem"
    end
end 
@info "Finished 2D sweep in $(time() - start) seconds."

##
@save "heatmap_data_Fgate1.0_allcodes_pmem75.jld2" results_heatmap mem_errors fidelities codess nsamples
##

function make_pL_ratio_heatmap(
    mem_errors,
    fidelities,
    pL;
    T_coh = nothing,
    title = "",
    nsamples = 1000_000,

)

    ghz_infidelities = 1.0 .- fidelities

    # -------------------------------------------------------------
    # y-axis
    # -------------------------------------------------------------

    if isnothing(T_coh)

        yvals = mem_errors
        ylabel_str = L"Memory error probability $p_{\mathrm{mem}}$"

    else

        # From
        # p_mem = 3/4 * (1 - exp(-Δt_GHZ / T_coh))
        #
        # => Δt_GHZ = -T_coh * log(1 - 4/3 p_mem)

        yvals =
            -T_coh .* log.(
                1.0 .- (4.0 / 3.0) .* mem_errors
            )

        ylabel_str =
            L"GHZ generation time $\Delta t_{\mathrm{GHZ}}\;[\mathrm{s}]$"
    end

    # -------------------------------------------------------------
    # Sort axes
    # -------------------------------------------------------------

    xorder = sortperm(ghz_infidelities)
    yorder = sortperm(yvals)

    x = ghz_infidelities[xorder]
    y = yvals[yorder]

    pL_sorted = pL[yorder, xorder]
    p_mem_sorted = mem_errors[yorder]

    # -------------------------------------------------------------
    # pL / p_mem
    # -------------------------------------------------------------

    ratio =
        pL_sorted ./
        reshape(p_mem_sorted, :, 1)

    # -------------------------------------------------------------
    # Finite-sampling floor
    #
    # If zero logical failures are observed, the Monte Carlo
    # estimate is pL = 0. For plotting on a logarithmic scale,
    # replace these values by approximately one failure in
    # nsamples trajectories.
    # -------------------------------------------------------------

    pL_floor = 1 / nsamples

    ratio_floor =
        pL_floor ./
        reshape(p_mem_sorted, :, 1)

    ratio_plot = max.(ratio, ratio_floor)

    logratio = log10.(ratio_plot)

    # Symmetric colour scale around log10(pL / p_mem) = 0
    maxL = maximum(abs, logratio)

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    plt = heatmap(
        x,
        y,
        logratio,

        xscale = :log10,
        yscale = :log10,

        xlabel = L"GHZ infidelity $1-F_{\mathrm{GHZ}}$",
        ylabel = L"Data error probability $p_{\mathrm{mem}}$",

        colorbar_title = "",
    

        clim = (-maxL, maxL),

        colormap = :RdYlGn_11,

        title =
            title *
            L"\qquad\log_{10}\!\left(\widehat{p}_{\mathrm{L}}/p_{\mathrm{mem}}\right)",

        size = (650, 500),
        margin = 5mm,

        tickfontsize = 11,
        labelfontsize = 13,
        titlefontsize = 13,
    )

    # -------------------------------------------------------------
    # Pseudothreshold / break-even contour:
    #
    # pL = p_mem
    # => log10(pL / p_mem) = 0
    # -------------------------------------------------------------

    contour!(
        plt,
        x,
        y,
        logratio,

        levels = [0.0],

        linestyle = :dash,
        linewidth = 2,
        color = :grey,

        label = "",
    )

    return plt
end

code_labels = [
    L"[[7,1,3]]",
    L"[[12,2,3]]",
    L"[[26,2,5]]",
]

plots = []

for ic in eachindex(codess)

    pL = results_heatmap[ic, :, :]

    plt = make_pL_ratio_heatmap(
        mem_errors,
        fidelities,
        pL;
        T_coh = nothing,
        title = code_labels[ic],
        nsamples = nsamples,
    )

    push!(plots, plt)
end

fig = plot(
    plots...,
    layout = (length(codess), 1),
    size = (700, 1500),
    margin = 5mm,
    leftmargin = 15mm,
)

display(fig)

savefig(
    fig,
    "pseudothresholds_pL_heatmaps.pdf",
)

