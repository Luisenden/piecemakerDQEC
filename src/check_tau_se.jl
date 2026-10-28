# Check that the cycle-time estimate is well resolved: τ_se / τ ≲ rel_tol for every row of every
# step3_table (the U_t_member<index>_timing_cutoff_pairs_<code>.jld2 files of Step 3).
#
# Rows with a non-finite τ (fewer than two completed rounds) or τ_se (fewer than two batches) are
# counted separately, since no relative error can be formed for them.
#
# Usage:
#   julia --project=src src/check_tau_se.jl [rel_tol] [dir ...]

using JLD2
using DataFrames
using Printf

const backup = "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC"
rel_tol = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.01
dirs = length(ARGS) >= 2 ? ARGS[2:end] : [
    joinpath(backup, "output_v1multizone_2cutoffbfs"),
    joinpath(backup, "output_v1_2cutoffbfs"),
]

const ut_re = r"^U_t_member(\d+)_timing_cutoff_pairs_(.+)\.jld2$"

"""All U_t files below `dir`, as (path, code, member index)."""
function find_ut_files(dir)
    files = Tuple{String, String, Int}[]
    for (root, _, names) in walkdir(dir), name in names
        m = match(ut_re, name)
        m === nothing || push!(files, (joinpath(root, name), m.captures[2], parse(Int, m.captures[1])))
    end
    return sort(files; by = f -> (dirname(f[1]), f[2], f[3]))
end

for dir in dirs
    println("\n=== ", dir)
    isdir(dir) || (println("  directory not found, skipped"); continue)
    files = find_ut_files(dir)
    isempty(files) && (println("  no U_t files found"); continue)

    summary_rows = []
    offending = DataFrame[]
    for (path, code, index) in files
        step3_table = try
            jldopen(f -> f["step3_table"], path, "r")
        catch err
            @warn "could not read step3_table" path exception = err
            continue
        end
        n = nrow(step3_table)
        if !(hasproperty(step3_table, :τ) && hasproperty(step3_table, :τ_se))
            push!(summary_rows, (; code, member = index, n_rows = n, n_finite = 0, n_nonfinite = 0,
                                   n_above = 0, max_rel = NaN, note = "no τ/τ_se column"))
            continue
        end

        finite = isfinite.(step3_table.τ) .& isfinite.(step3_table.τ_se)
        rel = step3_table.τ_se[finite] ./ step3_table.τ[finite]
        above = rel .> rel_tol
        push!(summary_rows, (; code, member = index, n_rows = n,
                               n_finite = count(finite), n_nonfinite = count(!, finite),
                               n_above = count(above),
                               max_rel = isempty(rel) ? NaN : maximum(rel),
                               note = ""))
        if any(above)
            bad = step3_table[finite, :][above, :]
            insertcols!(bad, 1, :code => code, :member => index, :rel_se => rel[above])
            push!(offending, bad)
        end
    end

    isempty(summary_rows) && (println("  no readable U_t files"); continue)
    summary = DataFrame(map(identity, summary_rows))   # narrow Vector{Any} to a NamedTuple vector
    for code in unique(summary.code)
        s = summary[summary.code .== code, :]
        n_finite = sum(s.n_finite)
        n_above = sum(s.n_above)
        max_rel = maximum(filter(!isnan, s.max_rel); init = 0.0)
        @printf("  %-20s %3d files | %5d rows, %5d finite, %4d non-finite | %4d above %.3g%% (%.1f%%) | max τ_se/τ = %.3g%%\n",
                code, nrow(s), sum(s.n_rows), n_finite, sum(s.n_nonfinite),
                n_above, 100rel_tol, n_finite == 0 ? 0.0 : 100n_above / n_finite, 100max_rel)
    end

    missing_cols = summary[summary.note .!= "", :]
    nrow(missing_cols) > 0 && println("  files without τ/τ_se: ",
        join(string.(missing_cols.code, " member ", missing_cols.member), ", "))

    if isempty(offending)
        println("  all finite rows have τ_se/τ ≤ ", 100rel_tol, "%")
    else
        bad = sort!(vcat(offending...; cols = :union), :rel_se; rev = true)
        println("  rows above tolerance (worst first, at most 25):")
        for r in eachrow(first(bad, 25))
            @printf("    %-12s member %4d | c = %-10s | τ = %8.3g ms  τ_se = %8.3g ms  τ_se/τ = %6.2f%%  n_rounds = %s\n",
                    r.code, r.member,
                    isfinite(r.cutoff) ? @sprintf("%.3g ms", 1e3r.cutoff) : "Inf",
                    1e3r.τ, 1e3r.τ_se, 100r.rel_se,
                    hasproperty(r, :n_rounds) ? string(r.n_rounds) : "?")
        end
        nrow(bad) > 25 && println("    … and ", nrow(bad) - 25, " more")
    end
end
