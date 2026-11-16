# The full hardware search space: single source of truth for the run scripts (Steps 2-4) and the
# plotting scripts, so every run and every figure uses the same ladders.
#
# TIMING_GRID / FIDELITY_GRID hold the raw candidate values (any order). The run scripts pass them
# through order_timing_grid / order_fidelity_grid (utils_pseudothreshold.jl), which sort every
# axis most → least demanding. A run on a reduced grid overrides single axes, e.g.
#     merge(TIMING_GRID, (Δt_rotation_shuttle = [10e-6, 50e-6, 100e-6],))
#
# SEARCH_GRID holds the same values ordered from least to most demanding: longer times, lower
# p_link, lower fidelities, shorter T_coh first. The value index k_i of a value on axis i is its
# position on that ladder, so k_i = 1 is the most relaxed value of the search space (the x axis of
# plot_minimal_codes.jl), and the index sum Σ_i k_i grows with demand. It does not depend on the
# (possibly shorter) grid a single code was run on.

const TIMING_GRID = (
    attempt_time        = [0.1e-6, 0.5e-6, 1e-6, 10e-6],          # t_att
    Δt_CNOTgate         = [1e-6, 10e-6, 100e-6, 250e-6],          # t_CNOT
    Δt_readout          = [0.1e-3, 1e-3, 2e-3],                   # t_ro
    Δt_rotation_shuttle = [10e-6, 50e-6, 100e-6, 500e-6],         # t_buff
    link_success_prob   = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],         # p_link
)

const FIDELITY_GRID = (
    F_link      = [1.0 - 2.5^(-x) for x in 3.0:10.0],           # F_link
    F_CNOT      = [0.999, 0.9995, 0.9997, 0.9999, 0.99999],     # F_CNOT
    F_readout   = [0.999, 0.9999, 1.0],                         # F_ro
    T_coherence = [0.5, 1.0, 2.0, 10.0, 20.0],                  # T_coh,comm
)

const SEARCH_GRID = (
    attempt_time        = sort(TIMING_GRID.attempt_time; rev = true),
    Δt_CNOTgate         = sort(TIMING_GRID.Δt_CNOTgate; rev = true),
    Δt_readout          = sort(TIMING_GRID.Δt_readout; rev = true),
    Δt_rotation_shuttle = sort(TIMING_GRID.Δt_rotation_shuttle; rev = true),
    link_success_prob   = sort(TIMING_GRID.link_success_prob),
    F_link              = sort(FIDELITY_GRID.F_link),
    F_CNOT              = sort(FIDELITY_GRID.F_CNOT),
    F_readout           = sort(FIDELITY_GRID.F_readout),
    T_coherence         = sort(FIDELITY_GRID.T_coherence),
)

"""Position (1 = least demanding) of value `v` on axis `name` of SEARCH_GRID."""
function search_slot(name::Symbol, v::Real)
    i = findfirst(a -> isapprox(a, v; rtol = 1e-9), SEARCH_GRID[name])
    i === nothing && error("$(name) = $(v) is not in SEARCH_GRID; update search_grid.jl to the grid the codes were run on")
    return i
end

"""Index sum Σ_i k_i over `names` of a hardware point, k_i = `search_slot`; `value(name)` gives its value on each axis."""
search_index_sum(value, names) = sum(search_slot(name, value(name)) for name in names)
