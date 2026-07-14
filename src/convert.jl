using JLD2
using Statistics
using DataFrames
using CSV
using StatsPlots

folder = length(ARGS) >= 1 ? ARGS[1] : "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1"
outfolder = length(ARGS) >= 2 ? ARGS[2] : "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1/"
index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

for i in index:100:3888
    file = "ghz_service_v1_Steane7_depolarizing_$(i).jld2"
    start_time = time()

    endswith(file, ".jld2") || continue

    path = joinpath(folder, file)
    
    df_out = nothing
    try
        df_out = JLD2.load(path, "df_out")
        @info "Loaded file: $path."
    catch err
        @info "Failed to load file: $path"
        continue
    end

    isnothing(df_out) && continue

    df_out[!, :depolarizing] = df_out[!, :error_model] .== "depolarizing"
    df_out[!, :dephasing] = df_out[!, :error_model] .== "dephasing"

    select!(df_out, Not([:error_model, :time_diff, :n, :seed]))
    for i in 1:4
        df_out[!, Symbol("client_$i")] = in.(i, df_out[!, :clients_serviced])
    end

    for col in names(df_out, Int64)
        df_out[!, col] = Int32.(df_out[!, col])
    end

    for col in names(df_out, Float64)
        df_out[!, col] = Float32.(df_out[!, col])
    end

    select!(df_out, Not([:clients_serviced]))

    outpath = joinpath(outfolder, replace(file, ".jld2" => ".csv"))
    CSV.write(outpath, df_out)
    rm(path)

    end_time = time()
    @info "Conversion completed in $(end_time - start_time) seconds."
end
