const code = length(ARGS) >= 1 ? ARGS[1] : ""
const gate_fidelity_qec = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
const seed = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 1234  # one fusion zone vs. as many as needed
code == "" && error("Missing code name (e.g. Steane713, BB12_2_3 or GB26_2_5) as first argument")

using Random 
Random.seed!(seed)

include(joinpath(@__DIR__, "utils_pseudothreshold.jl"))
## Step 1: find the break-even memory budget (p_mem^max, τ_max) for the QEC code and gate fidelity.
p_mem_max = find_pmem_max(qec_parity_checks(code), gate_fidelity_qec; seed=seed)
τ_max = t_cycle(p_mem_max; T_coh = T_COH_DATA)

@save joinpath(@__DIR__, "error_budget_$(code).jld2") p_mem_max τ_max