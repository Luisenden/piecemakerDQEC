using JLD2
using Statistics
using DataFrames
using CSV
using StatsPlots
include("utils_pseudothreshold.jl")

using Logging
##
function extract_pL(gen_time, F_GHZ; nsamples=100_000)
    p_mem_val = p_mem(gen_time * 2)  # generation time per stabilizer generator takes twice as long

    setup = CShorSyndromeECCSetup(p_mem_val, 0.9995, F_GHZ)
    decoder = TableDecoder(code)

    r = cevaluate_decoder_pL(decoder, setup, nsamples)

    return (
        pL = r,
        p_mem = p_mem_val,
    )
end
##

folder1 = "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1"
folder2 = "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_T"
##
columns =  [:attempt_time, :T_coherence, :error_model, :tRotationShuttle, :tCNOT, :gate_fidelity, 
            :tReadout, :readout_fidelity, :runtime, :wallclock_time, :nlogs]

dfs = DataFrame[]
df_out = nothing
for folder in [folder1, folder2]
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
df = vcat(dfs...)

##
df = df[df.generator_idx .== 1, :]
#df = df[df.cutoff .== Inf, :]

##
T_coh = 10.0
p_mem(Δt_GHZ) = (3/4) * (1 - exp(-Δt_GHZ / T_coh))
#df = df[p_mem.(df.mean_generation_time .* 2) .<= 0.1, :]

##
code = Steane7()
res = []
count = 0

res = pL_fit.(p_mem.(df.mean_generation_time .* 2), 1.0 .- df.mean_GHZfidel) #extract_pL.(df.mean_generation_time .* 2, df.mean_GHZfidel; nsamples=100_000)
##
df_both_targets = df[res .<= p_mem.(df.mean_generation_time .* 2), :]
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
transform!(
    pareto_df,
    [:mean_generation_time, :mean_GHZfidel] =>
        ByRow((gen_time, F_GHZ) ->
            extract_pL(gen_time, F_GHZ; nsamples=1000_000)
        ) =>
        AsTable
)

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
            legend = l == 1 && k == length(dfs) ? :outerright : false,
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
    )
end

plot_value_counts_many(
    [df_both_targets, pareto_df],;
    labels = [ 
        L"\mathrm{both\ targets}",
        L"\mathrm{Pareto\ front}",
    ],
)

savefig("minimum_requirements_analysis_pmem.pdf")




##

using Random, Measures, Plots, LaTeXStrings
gr()


baseline_values = (
    readout_fidelity  = 0.9999,
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
        value == 1 ? Inf : log(baseline) / log(value)

    time_factor(value, baseline) = baseline / value

    Dict(
        :readout_fidelity => probability_factor(
            dom_solution_values.readout_fidelity,
            baseline_values.readout_fidelity,
        ),

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

sweep_max_values = (
    readout_fidelity   = 0.99999,   # 1.0 would give Inf
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
    :readout_fidelity,
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
    L"F_{\mathrm{ro}}",
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
        margin = 5mm,
        minorgrid = true,
        rightmargin = 5mm,
        label = labels[1],
        xgridlinewidth = 0.0,
        ygridlinewidth = 1.0,
        gridcolor = "black",
        fontsize=10,
        size = (700,720),
        legendfontsize = 10,
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
            rightmargin = 5mm,
            legend_title = L"$F_\mathrm{GHZ},\ R_\mathrm{GHZ}$"
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

# selected_solutions = sorted_pareto_df[end-5:end-3, :]
selected_solutions = sort(pareto_df, :pL)[1:5, :]
labels = [
    "$(round(row.mean_GHZfidel; digits=4)), $(round(1.0 / row.mean_generation_time; digits=2))"*L"\,\mathrm{Hz}"
    for row in eachrow(selected_solutions)
]

p = spiderplot(
    collect(eachrow(selected_solutions))...;
    labels = labels,
    reference_solution = nothing,#sweep_max_values,
    reference_label = "range maximum",
)

savefig("IF_Steane_pmemT1s.pdf")


##
sorted_pareto_df = sort(pareto_df, :pL)[1:5, :]
perf_columns = [:generator_idx, :mean_GHZfidel, :mean_generation_time,]
df_out[!, :mean_generation_time] = df_out.mean_generation_time * 2
using PrettyTables
pretty_table(sorted_pareto_df[:, [perf_columns...; axis_order...;[:pL, :p_mem]]]; backend = :latex)
##
function dominates(a, b, objectives)
    all(
        better_or_equal(a[col], b[col], sense)
        for (col, sense) in objectives
    ) &&
    any(
        strictly_better(a[col], b[col], sense)
        for (col, sense) in objectives
    )
end

for i in 1:nrow(pareto_df)
    for j in 1:nrow(pareto_df)
        i == j && continue

        if dominates(pareto_df[j, :], pareto_df[i, :], objectives)
            println("Row $i is dominated by row $j")
        end
    end
end
