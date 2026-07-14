using JLD2
using Statistics
using DataFrames
using CSV
using StatsPlots

folder = length(ARGS) >= 1 ? ARGS[1] : "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1"
outfolder = length(ARGS) >= 2 ? ARGS[2] : "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1/"
start_index = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

conversion_files = ["ghz_service_v1_Steane7_depolarizing_$(i).jld2" for i in 1:3888]  # 3888 files
end_index = min(start_index+100, length(conversion_files))


for file in readdir(folder)[start_index:end_index]
    start_time = time()

    endswith(file, ".jld2") || continue

    path = joinpath(folder, file)
    try 
        @load path df_out
    catch err
        @info "Failed to load file: $file. Error: $err"
        continue
    end

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
