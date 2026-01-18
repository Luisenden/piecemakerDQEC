using QuantumSavory
using QuantumSavory: Register, X, Z, Y, CNOT, I, ZCZ
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames
using Statistics

function noisy_ghz(target_fidelity::Float64=1.0, n::Int=4)
    perfect_state::StabilizerState = StabilizerState(ghz(n))
    perfect_dm = SProjector(perfect_state)
    mixed_dm = MixedState(SProjector(perfect_state))
    return target_fidelity*perfect_dm + (1-target_fidelity)*mixed_dm
end

n = 4
const perfect_ghz = noisy_ghz(1.0, 4)
const error_patterns = Iterators.product([[I,Z,X,Y] for i in 1:n]...)

function get_superoperator(ghz_fidelity::Float64=1.0, n::Int=4)
    vec_s_plus = []
    s_p_flag = true
    vec_s_minus = []
    s_m_flag = true

    while s_p_flag || s_m_flag
        # 1. Prepare Choi state (this is what comes out of the system after applying the protocol)
        ancilla_register = Register(n, QuantumOpticsRepr())
        initialize!([ancilla_register[i] for i in 1:n], noisy_ghz(ghz_fidelity, 4)) # ancilla hold the GHZ (perfect in this dummy case)

        data_register = Register(n, QuantumOpticsRepr())
        ref_register = Register(n, QuantumOpticsRepr())
        for i in 1:n
            initialize!([data_register[i], ref_register[i]], noisy_ghz(ghz_fidelity, 2)) # the data + reference hold 4 Bell pairs
        end

        for i in 1:n
            apply!([ancilla_register[i], data_register[i]], ZCZ) # from ancilla to data
        end

        s_i_outcomes = [project_traceout!(ancilla_register[i], X) - 1 for i in 1:n] # Measure ancilla in X basis (outcomes 1 and 2 in Julia so -1 to get 0 and 1)

        s = (-1)^ sum(s_i_outcomes)
        @info s, s_p_flag, s_m_flag
        registers = RegRef[]
        for i in 1:n
            append!(registers, [ref_register[i], data_register[i]])
        end

        # 2. Prepare general theoretical state in different error patterns
        for prd in error_patterns

            Ψ = reduce(⊗, [StabilizerState(ghz(2)) for i in 1:n])

            P =  reduce(⊗, [(I ⊗ Z) for i in 1:n]) # Stabilizer (either XXXX or ZZZZ)
            I_full = reduce(⊗, [I for i in 1:2n])  # Identity on 2n qubits

            # Projector for data qubits onto eigenspaces
            P⁺ = (I_full + P) / sqrt(2)

            P_m = reduce(⊗, [(I ⊗ prd[i]) for i in 1:n]) # Error pattern on data qubits

            Ψ⁺ = P_m * P⁺ * Ψ 
            P_Ψ⁺ = projector(Ψ⁺)

            if s == 1 && s_p_flag
                p_plus = real(observable(registers, P_Ψ⁺))
                append!(vec_s_plus, p_plus)
            elseif s == -1 && s_m_flag
                p_plus = real(observable(registers, P_Ψ⁺))
                append!(vec_s_minus, p_plus)
            end
        end
        if s == 1
            s_p_flag = false
        elseif s == -1
            s_m_flag = false
        end
    end
    return vec_s_plus, vec_s_minus
end

trials = 50
all_s_plus, all_s_minus = [], []
s_avg_plus, s_avg_minus = get_superoperator(0.9, 4)
for _ in range(2,trials)
    s_plus, s_minus = get_superoperator(0.9, 4)
    s_avg_plus .+= s_plus
    s_avg_minus .+= s_minus
end

s_avg_plus ./= trials
s_avg_minus ./= trials