using JLD2
using Statistics
using DataFrames
using CSV
using StatsPlots
include("utils_pseudothreshold.jl")

using Logging

##

folder = "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1"
## this can take up to a minute 
columns =  [:attempt_time, :T_coherence, :error_model, :tRotationShuttle, :tCNOT, :gate_fidelity, 
            :tReadout, :readout_fidelity, :runtime, :wallclock_time, :nlogs]

dfs = DataFrame[]
files = []
df_out = nothing
for folder in [folder]
    files = readdir(folder)
    for file in files
        path = joinpath(folder, file)
        try
            @load path df_out
        catch
        end
        !isnothing(df_out) && push!(dfs, df_out)
    end
end
@info "Loaded $(length(dfs)) dataframes from $(folder)."
df = vcat(dfs...)

##
files = readdir(folder)
indices = [
    parse(Int, match(r"_(\d+)\.jld2$", str).captures[1])
    for str in files
]
all_indices = 1:12960
missing_indices = setdiff(all_indices, indices)

##
df = df[df.generator_idx .== 1, :]
df[!, :mean_generation_time] = df.mean_generation_time .* 2
#df = df[df.cutoff .== Inf, :]


##
code = Steane7()
T_coh = 1.0
res = []
count = 0
##
df[!, :p_mem] = p_mem.(df.mean_generation_time; T_coh=T_coh)
df[!, :pL] = pL_fit.(df.p_mem, 1.0 .- df.mean_GHZfidel)

df_both_targets = df[(df.pL .<= df.p_mem) .&& (df.p_mem .< 0.1), :]
#@save "df_both_targets_pmem.jld2" df_both_targets

## pareto front analysis (this can take several minutes)

objectives = [
    (:readout_fidelity, :min),
    (:tReadout, :max),
    (:gate_fidelity, :min),
    (:tCNOT, :max),
    (:tRotationShuttle, :max),
    (:attempt_time, :max),
    (:link_success_prob, :min),
    (:T_coherence, :min),
    (:F_link, :min),
]

function better_or_equal(a, b, sense)
    if sense == :max
        return a >= b
    elseif sense == :min
        return a <= b
    else
        error("sense must be :min or :max")
    end
end

function strictly_better(a, b, sense)
    if sense == :max
        return a > b
    elseif sense == :min
        return a < b
    else
        error("sense must be :min or :max")
    end
end

function pareto_front(df, objectives)
    n = nrow(df)
    dominated = falses(n)

    for i in 1:n
        @info "done $(i)/$(n)"
        for j in 1:n
            i == j && continue

            j_better_or_equal_all = all(
                better_or_equal(df[j, col], df[i, col], sense)
                for (col, sense) in objectives
            )

            j_strictly_better_somewhere = any(
                strictly_better(df[j, col], df[i, col], sense)
                for (col, sense) in objectives
            )

            if j_better_or_equal_all && j_strictly_better_somewhere
                dominated[i] = true
                break
            end
        end
    end

    return df[.!dominated, :]
end

pareto_df = pareto_front(df_both_targets, objectives)
## add simulated logical error probability to pareto_df

code = Steane7()
T_coh = 1.0
transform!(
    pareto_df,
    [:mean_generation_time, :mean_GHZfidel] =>
        ByRow((gen_time, F_GHZ) ->
            extract_pL(gen_time, F_GHZ, code, T_coh; nsamples=1000_000)
        ) =>
        AsTable
)


##

pareto_df_hard = pareto_df[(pareto_df.pL .<= pareto_df.p_mem) .&& (pareto_df.p_mem .< 0.1), :]

##
objectives_ordered_easy_to_difficult = [
    :readout_fidelity,
    :gate_fidelity,
    :tRotationShuttle,
    :tCNOT,
    :attempt_time,
    :link_success_prob,
    :tReadout,
    :T_coherence,
    :F_link,
]

values_ordered_easy_to_difficult = [
    false,
    false,
    true,
    true,
    true,
    false,
    true,
    false,
    false,
]

# Reverse to difficult → easy
objectives_ordered_difficult_to_easy = reverse(objectives_ordered_easy_to_difficult)
values_ordered_difficult_to_easy = reverse(values_ordered_easy_to_difficult)

pareto_df_hard_infcut = pareto_df_hard[pareto_df_hard.cutoff .== Inf, :]

selected_pareto_df_hard_infcut = select(
    pareto_df_hard_infcut,
    Not([
        :generator_idx,
        :cutoff,
        :error_model,
        :nlogs,
        :runtime,
        :std_GHZfidel,
        :sem_GHZfidel,
        :std_generation_time,
        :sem_generation_time,
        :seed,
        :wallclock_time,
    ]),
)

sorted_df = sort(
    selected_pareto_df_hard_infcut,
    [
        order(col, rev=rev)
        for (col, rev) in zip(
            objectives_ordered_easy_to_difficult,
            .!(values_ordered_easy_to_difficult),
        )
    ],
)

df_save = DataFrame(sorted_df[
        :,
        [
            objectives_ordered_easy_to_difficult...,
            :mean_GHZfidel,
            :mean_generation_time,
            :pL,
            :p_mem,
        ],
    ])

CSV.write("pareto_data_sorted.csv", df_save)

##
#show(df_save[df_save.p_mem .== maximum(df_save.p_mem), :], allcols=true, allrows=true)

using PrettyTables
pretty_table(df_save, backend = :latex, formatters = [fmt__round(9)], show_row_number_column = true,)


##
using DataFrames, Plots, Printf, LaTeXStrings
using Plots.PlotMeasures
using StatsPlots

##
params = [obj[1] for obj in objectives]

function fmtval(x)
    if x isa AbstractFloat
        isinf(x) && return "Inf"
        isnan(x) && return "NaN"
        s = @sprintf("%.4g", x)
        return replace(s, "e-0" => "e-", "e+0" => "e+")
    else
        return string(x)
    end
end

function sorted_unique(vals)
    vals = collect(skipmissing(vals))

    if all(v -> v isa Number, vals)
        return sort(unique(vals))
    else
        return sort(unique(vals), by=string)
    end
end

function count_matrix_for_param(d, p, vals, models; groupcol=:error_model)
    counts = zeros(Int, length(vals), length(models))

    for (midx, model) in enumerate(models)
        subvals = collect(skipmissing(d[d[!, groupcol] .== model, p]))

        for (vidx, v) in enumerate(vals)
            counts[vidx, midx] = Base.count(
                x -> isequal(x, v),
                subvals
            )
        end
    end

    return counts
end

function plot_value_counts_many(
    dfs;
    labels = ["df$i" for i in eachindex(dfs)],
    groupcol = :generator_idx,
    params = [:attempt_time, :link_success_prob, :T_coherence, :F_link],
    map_params_to_latx_symbols = Dict(
        :attempt_time => L"\Delta t_\mathrm{attempt}",
        :link_success_prob => L"p_\mathrm{link}",
        :T_coherence => L"T_\mathrm{coherence}",
        :F_link => L"F_\mathrm{link}",
        :cutoff => L"\mathrm{cutoff}",
    ),
)

    # all error models across all dfs
    models = unique(vcat([collect(skipmissing(d[!, groupcol])) for d in dfs]...))
    model_labels = string.(models)

    # global unique x-values per parameter across all dfs
    global_vals = Dict(
        p => sorted_unique(vcat([collect(skipmissing(d[!, p])) for d in dfs]...))
        for p in params
    )

    subplots = Plots.Plot[]

    for (l, p) in enumerate(params)
        vals = global_vals[p]
        val_labels = fmtval.(vals)

        for (k, d) in enumerate(dfs)
            counts = count_matrix_for_param(d, p, vals, models; groupcol=groupcol)

        plt = groupedbar(
            1:length(vals),
            counts;
            label = reshape(model_labels, 1, :),
            bar_width = 0.75,
            title = l == 1 ? string(labels[k]) : "",
            ylabel = k == 1 ? string(map_params_to_latx_symbols[p]) : "",
            xticks = (1:length(vals), val_labels),
            xrotation = 35,
            legend = l == 1 && k == length(dfs) ? :topright : false,
            grid = true,
            bottom_margin = 8mm,
            left_margin = k == 1 ? 12mm : 4mm,
            right_margin = 4mm,
        )

            push!(subplots, plt)
        end
    end

    plot(
        subplots...,
        layout = (length(params), length(dfs)),
        size = (380 * length(dfs), 260 * length(params)),
        plot_title = "Counting feasible solutions per parameter",
        legend=false,
    )
end

plot_value_counts_many(
    [df_both_targets, pareto_df],;
    labels = [ 
        L"$p_L ≤ p_\mathrm{mem}$",
        L"\mathrm{Pareto\ front}",
    ],
)

savefig("minimum_requirements_analysis_pmem.pdf")


##

df_plot = pareto_df_hard
scatter(
    df_plot.p_mem,
    df_plot.pL,

    marker_z = (df_plot.tReadout 
        .+ df_plot.tCNOT 
        .+ df_plot.tRotationShuttle 
        .+ df_plot.attempt_time ./ df_plot.link_success_prob)
        .* df_plot.T_coherence,
    seriescolor = :viridis,
    colorbar = true,

    markersize = 3,
    markerstrokecolor = :black,
    markerstrokewidth = 0.7,

    xlabel = L"\mathrm{memory\ error\ probability\ } p_{\mathrm{mem}}",
    ylabel = L"\mathrm{logical\ error\ probability\ } p_L",

    colorbar_title =
        L"",

    grid = true,
    legend = false,
)



##

using Random, Measures, Plots, LaTeXStrings
gr()


baseline_values = (
    # readout_fidelity  = 0.9999,
    tReadout          = 0.001,
    gate_fidelity     = 0.9997,
    tCNOT             = 100e-6,
    tRotationShuttle  = 100e-6,
    attempt_time      = 10e-6,
    link_success_prob = 1e-4,
    T_coherence       = 1.0,
    F_link            = 0.97,
)


depolarizing_probability(F) = (4F - 1) / 3


function improvement_factors(dom_solution_values)
    probability_factor(value, baseline) =
        value == 1 ? 10. : log(baseline) / log(value)

    time_factor(value, baseline) = baseline / value

    Dict(
        # :readout_fidelity => probability_factor(
        #     dom_solution_values.readout_fidelity,
        #     baseline_values.readout_fidelity,
        # ),

        :tReadout => time_factor(
            dom_solution_values.tReadout,
            baseline_values.tReadout,
        ),

        :gate_fidelity => probability_factor(
            depolarizing_probability(dom_solution_values.gate_fidelity),
            depolarizing_probability(baseline_values.gate_fidelity),
        ),

        :tCNOT => time_factor(
            dom_solution_values.tCNOT,
            baseline_values.tCNOT,
        ),

        :tRotationShuttle => time_factor(
            dom_solution_values.tRotationShuttle,
            baseline_values.tRotationShuttle,
        ),

        :attempt_time => time_factor(
            dom_solution_values.attempt_time,
            baseline_values.attempt_time,
        ),

        :link_success_prob => probability_factor(
            dom_solution_values.link_success_prob,
            baseline_values.link_success_prob,
        ),

        :T_coherence =>
            dom_solution_values.T_coherence /
            baseline_values.T_coherence,

        :F_link => probability_factor(
            depolarizing_probability(dom_solution_values.F_link),
            depolarizing_probability(baseline_values.F_link),
        ),
    )
end
##
axis_order = [
    :gate_fidelity,
    :tReadout,
    # :readout_fidelity,
    :F_link,
    :T_coherence,
    :link_success_prob,
    :attempt_time,
    :tRotationShuttle,
    :tCNOT,
]


##

sweep_max_values = (
    # readout_fidelity   = 0.99999,   # 1.0 would give Inf
    tReadout           = 0.1e-3,
    gate_fidelity      = 0.99999,
    tCNOT              = 1e-6,
    tRotationShuttle   = 10e-6,
    attempt_time       = 1e-6,
    link_success_prob  = 0.5,
    T_coherence        = 20.0,
    F_link             = 1 - 2.5^(-10),
)



axis_order = [
    :gate_fidelity,
    :tReadout,
    # :readout_fidelity,
    :F_link,
    :T_coherence,
    :link_success_prob,
    :attempt_time,
    :tRotationShuttle,
    :tCNOT,
]


params = [
    L"F_{\mathrm{CNOT}}",
    L"t_{\mathrm{ro}}",
    # L"F_{\mathrm{ro}}",
    L"F_{\mathrm{link}}",
    L"T_{\mathrm{coh}}",
    L"p_{\mathrm{link}}",
    L"t_{\mathrm{att}}",
    L"t_{\mathrm{rs}}",
    L"t_{\mathrm{CNOT}}",
]


function spiderplot(
    solutions...;
    labels = nothing,
    reference_solution = nothing,
    reference_label = "sweep maximum",
    colors = [
        :blue,
        :red,
        :green,
        :orange,
        :purple,
        :brown,
        :magenta,
        :cyan,
    ],
)
    isempty(solutions) &&
        throw(ArgumentError("Provide at least one solution."))

    # Calculate improvement factors for every solution
    @info "Calculating improvement factors for $solutions..."
    factors = improvement_factors.(solutions)

    # Extract values in exactly the same order for every solution
    radii_original = [
        Float64[solution_factors[key] for key in axis_order]
        for solution_factors in factors
    ]

    # Use all plotted solutions to determine one common radial scale
    reference_radius = if reference_solution === nothing
        nothing
    else
        ref_factors = improvement_factors(reference_solution)
        Float64[ref_factors[key] for key in axis_order]
    end
    all_radii = reduce(vcat, radii_original)
    if reference_radius !== nothing
        all_radii = vcat(all_radii, reference_radius)
    end

    n = length(axis_order)
    θ = collect(range(0, 2π; length = n + 1))

    # Powers of ten that should appear on the radial axis
    min_exp = floor(Int, log10(minimum(all_radii)))
    max_exp = ceil(Int, log10(maximum(all_radii)))

    # Add an offset because polar radii cannot be negative
    offset = -min_exp + 0.5

    tick_exponents = collect(min_exp:max_exp)
    tick_positions = tick_exponents .+ offset

    # tick_labels = [
    #     L"0.01", L"0.1", L"1", L"10", L"100", L"1000"
    # ]

    tick_labels = [
    latexstring("10^{", e, "}")
    for e in tick_exponents
    ]
    
    # Preserve the original automatic label format unless labels are supplied
    if labels === nothing
        labels = [
            string(
                round(solution[:mean_GHZfidel]; digits = 4),
                ", ",
                round(1.0/solution[:mean_generation_time]; digits = 2), "Hz",
            )
            for solution in solutions
        ]
    elseif length(labels) != length(solutions)
        throw(
            ArgumentError(
                "The number of labels must equal the number of solutions.",
            ),
        )
    end

    # First solution creates the plot
    first_radius = log10.(radii_original[1]) .+ offset
    first_radius_closed = vcat(first_radius, first(first_radius))

    p = plot(
        θ,
        first_radius_closed;
        proj = :polar,
        ms = 3,
        marker = :circle,
        color = colors[1],
        fill = (0, 0.0),
        xaxis = false,
        yticks = (tick_positions, tick_labels),
        ylims = (0, maximum(tick_positions) + 0.4),
        ytickfont=12,
        rightmargin = 15mm,
        minorgrid = true,
        label = labels[1],
        xgridlinewidth = 0.0,
        ygridlinewidth = 1.0,
        gridcolor = "grey",
        fontsize=10,
        legendposition = (0.9,0.97),
        size = (600, 500),
        legendfontsize = 10,
        legendtitle = L"$F_\mathrm{GHZ},\ R_\mathrm{GHZ}$"
    )

    # Add every additional solution to the same spider plot
    for i in 2:length(solutions)
        radius = log10.(radii_original[i]) .+ offset
        radius_closed = vcat(radius, first(radius))

        plot!(
            p,
            θ,
            radius_closed;
            ms = 3,
            marker = :circle,
            color = colors[mod1(i, length(colors))],
            fill = (0, 0.0),
            label = labels[i],
            legend = :outertopright,
        )
    end
    if reference_radius !== nothing
    radius = log10.(reference_radius) .+ offset
    radius_closed = vcat(radius, first(radius))

    plot!(
            p,
            θ,
            radius_closed;
            linewidth = 1.5,
            linestyle = :dash,
            marker = :diamond,
            markersize = 0,
            color = :grey,
            fill = (0, 0.0),
            label = reference_label,
        )
    end

    # Parameter labels
    angles = 2π .* (0:n-1) ./ n
    z = 1.1 .* exp.(im .* angles)

    for (i, label) in enumerate(params)
        x = real(z[i])
        y = imag(z[i])

        # Make text extend away from the spider plot
        halign =
            x > 0.1  ? :left :
            x < -0.1 ? :right :
                    :center

        annotate!(
            p,
            x,
            y,
            text(label, 14, "Computer Modern", halign),
        )
    end

    display(p)

    return p
end

# sorted_pareto_df = sort(pareto_df, :p_mem)

selected_solutions =
    df_save[df_save.p_mem .== maximum(df_save.p_mem), :]

labels = [
    "$(round(row.mean_GHZfidel; digits=4)), " *
    "$(round(1.0 / row.mean_generation_time; digits=2))" *
    L"\,\mathrm{Hz}"
    for row in eachrow(selected_solutions)
]

p = spiderplot(
    eachrow(selected_solutions)...;
    labels = labels,
    reference_solution = nothing,
)

savefig(p, "IF_Steane_pmemT1s.pdf")




##
sorted_pareto_df = sort(pareto_df, :pL)[1:5, :]
perf_columns = [:generator_idx, :mean_GHZfidel, :mean_generation_time,]
df_out[!, :mean_generation_time] = df_out.mean_generation_time * 2
using PrettyTables
pretty_table(sorted_pareto_df[:, [perf_columns...; axis_order...;[:pL, :p_mem]]]; backend = :latex)

##

using Plots, LaTeXStrings

gr()

fs = 12

default(
    fontfamily = "Computer Modern",
    tickfontsize = fs,
    guidefontsize = fs,
    legendfontsize = fs,
    titlefontsize = fs,
)

function polar_improvement_histogram(
    df;
    nbins = 8,
    improvement_range = nothing,
    colormap = cgrad([:white, :lavender, :cornflowerblue, :mediumblue, :navy]),
    normalize = false ,
)

    # ---------------------------------------------------------
    # 1. Improvement factors for all configurations
    # ---------------------------------------------------------
    solutions = collect(eachrow(df))
    factors = improvement_factors.(solutions)

    # rows = configurations
    # columns = parameters
    values = [
        Float64(f[key])
        for f in factors, key in axis_order
    ]

    @assert all(values .> 0) "All improvement factors must be > 0."

    logvalues = log10.(values)

    nsolutions = size(values, 1)
    nparams = size(values, 2)

    println("Number of configurations: ", nsolutions)

    # ---------------------------------------------------------
    # 2. Define common logarithmic radial bins
    # ---------------------------------------------------------
    if improvement_range === nothing
        min_exp = floor(minimum(logvalues))
        max_exp = ceil(maximum(logvalues))
    else
        min_exp = log10(improvement_range[1])
        max_exp = log10(improvement_range[2])
    end

    log_edges = collect(
        range(min_exp, max_exp; length = nbins + 1)
    )

    # ---------------------------------------------------------
    # 3. Histogram for each parameter
    #
    # counts[b, j]:
    #   radial bin b
    #   parameter j
    # ---------------------------------------------------------
    counts = zeros(Int, nbins, nparams)

    for j in 1:nparams
        for x in logvalues[:, j]

            # Ignore points outside explicitly requested range
            if x < min_exp || x > max_exp
                continue
            end

            b = if x == max_exp
                nbins
            else
                searchsortedlast(log_edges, x)
            end

            b = clamp(b, 1, nbins)
            counts[b, j] += 1
        end
    end

    # Either show number of configurations or fraction
    zvalues = if normalize
        counts ./ nsolutions
    else
        Float64.(counts)
    end

    zmax = maximum(zvalues)

    # ---------------------------------------------------------
    # 4. Helper: annular sector as a Plots.Shape
    # ---------------------------------------------------------
    function annular_sector(r_inner, r_outer, θ1, θ2; npoints = 30)

        θ_outer = collect(range(θ1, θ2; length = npoints))
        θ_inner = reverse(θ_outer)

        x = vcat(
            r_outer .* cos.(θ_outer),
            r_inner .* cos.(θ_inner),
        )

        y = vcat(
            r_outer .* sin.(θ_outer),
            r_inner .* sin.(θ_inner),
        )

        Shape(x, y)
    end

    # ---------------------------------------------------------
    # 5. Coordinate transformation
    #
    # Radius is linear in log10(improvement factor).
    #
    # Leave a small hole in the center.
    # ---------------------------------------------------------
    radial_offset = 1.0

    log_to_radius(x) = radial_offset + x - min_exp

    radial_edges = log_to_radius.(log_edges)

    θ_edges = collect(
        range(0, 2π; length = nparams + 1)
    )

    # slight gap between angular sectors
    angular_gap = 0.015

    # ---------------------------------------------------------
    # 6. Base plot
    # ---------------------------------------------------------
    rmax = maximum(radial_edges)

    p = plot(
        aspect_ratio = :equal,
        axis = false,
        legend = false,
        size = (600, 400),
        margin = 0mm,
        rightmargin = 0mm,
    )

    gradient = cgrad(colormap)

    # ---------------------------------------------------------
    # 7. Draw colored histogram cells
    # ---------------------------------------------------------
    for j in 1:nparams

        θ1 = θ_edges[j] + angular_gap
        θ2 = θ_edges[j + 1] - angular_gap

        for b in 1:nbins

            r1 = radial_edges[b]
            r2 = radial_edges[b + 1]

            z = zvalues[b, j]

            # map count -> color
            color_fraction = zmax == 0 ? 0.0 : z / zmax
            cell_color = gradient[color_fraction]

            shape = annular_sector(r1, r2, θ1, θ2)

            plot!(
                p,
                shape;
                seriestype = :shape,
                fillcolor = cell_color,
                linecolor = :white,
                linewidth = 0.6,
                label = false,
            )
        end
    end

    # ---------------------------------------------------------
    # 8. Add radial grid lines at powers of 10
    # ---------------------------------------------------------
    integer_exponents =
        ceil(Int, min_exp):floor(Int, max_exp)

    θcircle = range(0, 2π; length = 300)

    for exponent in integer_exponents

        r = log_to_radius(exponent)

        plot!(
            p,
            r .* cos.(θcircle),
            r .* sin.(θcircle);
            linewidth = exponent == 0 ? 1.5 : 0.7,
            linestyle = exponent == 0 ? :solid : :dot,
            color = :black,
            alpha = 0.5,
            label = false,
        )

        # radial tick label
        θlabel = π / 2 + 0.03

        annotate!(
            p,
            r * cos(θlabel),
            r * sin(θlabel) - 0.2,
            text(
                latexstring("10^{", exponent, "}"),
                10,
                "Computer Modern",
            ),
        )
    end

    # ---------------------------------------------------------
    # 9. Angular separator lines
    # ---------------------------------------------------------
    for θ in θ_edges[1:end-1]

        plot!(
            p,
            [radial_offset * cos(θ), rmax * cos(θ)],
            [radial_offset * sin(θ), rmax * sin(θ)];
            color = :black,
            linewidth = 0.5,
            alpha = 0.5,
            label = false,
        )
    end

    # ---------------------------------------------------------
    # 10. Parameter labels
    # ---------------------------------------------------------
    θcenters = [
        (θ_edges[j] + θ_edges[j + 1]) / 2
        for j in 1:nparams
    ]

    label_radius = rmax + 0.45

    for (j, label) in enumerate(params)

        θ = θcenters[j]

        x = label_radius * cos(θ)
        y = label_radius * sin(θ)

        halign =
            cos(θ) > 0.2  ? :left :
            cos(θ) < -0.2 ? :right :
                            :center

        annotate!(
            p,
            x,
            y,
            text(
                label,
                13,
                "Computer Modern",
                halign,
            ),
        )
    end

    # ---------------------------------------------------------
    # 11. Dummy scatter series to generate the color bar
    # ---------------------------------------------------------
    scatter!(
        p,
        [NaN, NaN],
        [NaN, NaN];
        marker_z = [0.0, zmax],
        c = colormap,
        clims = (0, zmax),
        colorbar = true,
        colorbar_title =
            normalize ?
            "Fraction of configurations" :
            "Number of configurations",
        label = false,
        position = :right,
    )

    # give labels enough room
    xlims!(p, -label_radius - 0.1, label_radius + 2.0)
    ylims!(p, -label_radius - 0.5, label_radius + 0.5)

    return p, counts, log_edges
end

p, counts, log_edges = polar_improvement_histogram(
    df_save;
    nbins = 15,
    improvement_range = (1e-2, 1e3),
)

display(p)
savefig(
    p,
    "IF_distribution_Steane_pmemT1s.pdf",
)

##
using Plots
using LaTeXStrings

gr()

T_coh = 1.0  # s

# Single-qubit depolarizing memory model
p_mem(Δt) = 3 / 4 * (1 - exp(-Δt / T_coh))

Δt = range(0, 25, length=500)

# Example generation times from your table
Δt_examples = [3.942, 7.938, 22.507]
p_examples = p_mem.(Δt_examples)

plt = plot(
    Δt,
    p_mem.(Δt),
    linewidth = 2.5,
    xlabel = L"\Delta t_{\mathrm{GHZ}}\;(\mathrm{s})",
    ylabel = L"p_{\mathrm{mem}}",
    label = L"p_{\mathrm{mem}}=\frac{3}{4}\left(1-e^{-\Delta t_{\mathrm{GHZ}}/T_{\mathrm{coh}}}\right)",
    xlims = (0, 25),
    ylims = (0, 0.78),
    legend = :bottomright,
    grid = true,
    framestyle = :box,
)

# Saturation value
hline!(
    plt,
    [0.75],
    linestyle = :dash,
    linewidth = 1.5,
    label = L"p_{\mathrm{mem}}=3/4",
)

# Points from your table
scatter!(
    plt,
    Δt_examples,
    p_examples,
    markersize = 6,
    markerstrokewidth = 0.8,
    label = "Configurations from table",
)

display(plt)