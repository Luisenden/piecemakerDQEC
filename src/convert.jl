using JLD2
using Statistics
using DataFrames
using CSV
using StatsPlots

start_time = time()

folder = length(ARGS) >= 1 ? ARGS[1] : "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1"
outfolder = length(ARGS) >= 2 ? ARGS[2] : "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/4_backup_project_piecemakerDQEC/output_v1_cluster/output_v1/csv"

for file in readdir(folder)
    endswith(file, ".jld2") || continue

    path = joinpath(folder, file)
    @load path df_out

    df_out[!, :depolarizing] = df_out[!, :error_model] .== "depolarizing"
    df_out[!, :dephasing] = df_out[!, :error_model] .== "dephasing"
    select!(df_out, Not(:error_model))

    for i in 1:4
        df_out[!, Symbol("client_$i")] = in.(i, df_out[!, :clients_serviced])
    end

    for col in names(df_out, Int64)
        df_out[!, col] = Int32.(df_out[!, col])
    end

    for col in names(df_out, Float64)
        df_out[!, col] = Float32.(df_out[!, col])
    end

    select!(df_out, Not(:clients_serviced))

    outpath = joinpath(outfolder, replace(file, ".jld2" => ".csv"))
    CSV.write(outpath, df_out)
end

end_time = time()
@info "Conversion completed in $(end_time - start_time) seconds."