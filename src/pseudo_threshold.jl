# see tutorial at https://qc.quantumsavory.org/stable/ECC_evaluating/

# this script estimates the logical error probabilities of quantum error-correcting codes under a noisy memory and noisy Shor-style syndrome-extraction circuit.
# for each code and each memory-error probability, it performs Monte Carlo circuit (shor syndrome extraction) simulations

using Plots
using Colors
using LaTeXStrings
using Measures
using JLD2

include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))

## find break-even / pseudothreshold points for a given code and a range of memory error probabilities 

break_even_mem_errors = 10 .^ range(
    log10(1e-4),
    log10(0.1),
    length = 30,
)

break_even_gate_fidelity = 1.0
break_even_logeps_min = -6.0
break_even_logeps_max = -2.0
break_even_logeps_tol = 0.05
break_even_nsamples_start = 1_000
break_even_nsamples_max = 1000_000

codes= [
    Steane7(),
    BB12_2_3,
    GB26_2_5,
]

break_even_results = Matrix{Any}(
    undef,
    length(codes),
    length(break_even_mem_errors),
)

for (ic, code) in pairs(codes)
    decoder = TableDecoder(code; error_weight = 4)
    for (imem, p_mem) in pairs(break_even_mem_errors)
        result = find_break_even(
            decoder,
            p_mem,
            break_even_gate_fidelity,
            break_even_logeps_min,
            break_even_logeps_max,
            break_even_logeps_tol,
            break_even_nsamples_start,
            break_even_nsamples_max,
        )

        break_even_results[ic, imem] = result

        @info "Break-even search" code = ic p_mem = p_mem status = result.status eps_ghz = result.eps_ghz
    end
end
## Plot break-even GHZ infidelity versus memory error

break_even_code_labels = [
    "[[7,1,3]]",
    "[[12,2,3]]",
    "[[26,2,5]]",
]

break_even_code_colors = [
    RGB(31/255, 119/255, 180/255),
    RGB(255/255, 127/255, 14/255),
    RGB(44/255, 160/255, 44/255),
]

plt_break_even = plot(
    xscale = :log10,
    yscale = :log10,
    xlabel = L"Memory error probability $p_{\mathrm{mem}}$",
    ylabel = L"Break-even GHZ infidelity $\epsilon_{\mathrm{GHZ}}$",
    title = "Break-even GHZ infidelity",
    legend = :topleft,
    grid = true,
    minorgrid = true,
    size = (700, 500),
)


for ic in axes(break_even_results, 1)
    plot!(
        plt_break_even,
        break_even_mem_errors,
        [
            break_even_results[ic, imem].eps_ghz
            for imem in axes(break_even_results, 2)
        ],
        color = break_even_code_colors[ic],
        marker = :circle,
        linewidth = 2,
        label = break_even_code_labels[ic],
    )
end

savefig(plt_break_even, "break_even_infidelity_vs_pmem.pdf")
display(plt_break_even)

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
        legend = :topleft,
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

codes = [
    Steane7(),
    BB12_2_3,
    GB26_2_5,
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
    length(codes), 20)

for (ic, c) in pairs(codes)

    decoder = TableDecoder(c; error_weight=3)

    for (ivar, var) in pairs(mem_errors) # :1 or :2

        setup = CShorSyndromeECCSetup(
            var,
            F_gate,       # two-qubit gate fidelity
            1.0,
        )

        pL = cevaluate_decoder_pL(
            decoder,
            setup,
            nsamples,
        ).pL

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
mem_errors = 10 .^ range(log10(1e-4), log10(0.1), length=20)

fidelities = [
    1.0 - 2.5^(-x)
    for x in range(6.0, 12.0, length=20)
]

ghz_infidelities = 1.0 .- fidelities

codes = [
    Steane7(),
    BB12_2_3,
    GB26_2_5,
]

nsamples = 100_000

# dimensions:
# code × memory error × GHZ fidelity
#
results_heatmap = zeros(
    length(codes),
    length(mem_errors),
    length(fidelities),
)

start = time()
for (ic, c) in pairs(codes)

    decoder = TableDecoder(c; error_weight = 4)

    for (imem, p_mem) in pairs(mem_errors)
        for (ighz, F_GHZ) in pairs(fidelities)

            setup = CShorSyndromeECCSetup(
                p_mem,
                F_gate,       # two-qubit gate fidelity
                F_GHZ,
            )

            pL = cevaluate_decoder_pL(
                decoder,
                setup,
                nsamples,
            ).pL

            results_heatmap[ic, imem, ighz] = pL

        end
        @info "Code $ic, p_mem = $p_mem"
    end
end 
@info "Finished 2D sweep in $(time() - start) seconds."

##
@load "heatmap_data_Fgate1.0_allcodes5.jld2" results_heatmap mem_errors fidelities codess nsamples
##

function make_pL_ratio_heatmap(
    mem_errors,
    fidelities,
    pL;
    T_coh = nothing,
    title = "",
    nsamples = 100_000,

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

