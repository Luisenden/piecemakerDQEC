using LinearAlgebra

function fit_2d_log_surface(
    mem_errors,
    ghz_infidelities,
    pL;
    pL_min = nothing,
)

    x = Float64[]
    y = Float64[]
    z = Float64[]

    for (imem, p_mem) in pairs(mem_errors)
        for (ighz, p_ghz) in pairs(ghz_infidelities)

            p = pL[imem, ighz]

            # Ignore unresolved Monte Carlo zeros
            if p > 0
                push!(x, log10(p_mem))
                push!(y, log10(p_ghz))
                push!(z, log10(p))
            elseif !isnothing(pL_min)
                push!(x, log10(p_mem))
                push!(y, log10(p_ghz))
                push!(z, log10(pL_min))
            end
        end
    end

    A = hcat(
        ones(length(x)),
        x,
        y,
        x.^2,
        x .* y,
        y.^2,
    )

    β = A \ z

    function predict(p_mem, p_ghz)
        x = log10(p_mem)
        y = log10(p_ghz)

        log_pL =
            β[1] +
            β[2] * x +
            β[3] * y +
            β[4] * x^2 +
            β[5] * x * y +
            β[6] * y^2

        return 10.0^log_pL
    end

    return β, predict
end

##

pL = results_heatmap[1, :, :, 1]

β, pL_fit = fit_2d_log_surface(
    mem_errors,
    ghz_infidelities,
    pL,

)