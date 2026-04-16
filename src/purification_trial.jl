using QuantumSavory
using QuantumSavory: Register, StabilizerState
using QuantumSavory.CircuitZoo: Purify2to1Node
using QuantumClifford: ghz

## GHZ purification using bell pairs

bell_pair = StabilizerState("XX ZZ")
perfect_pair_dm = SProjector(bell_pair)
mixed_bell_dm = MixedState(perfect_pair_dm)
noisy_pair_func(λ) = λ*perfect_pair_dm + (1-λ)*mixed_bell_dm

GHZ = StabilizerState(ghz(3))
perfect_GHZ_dm = SProjector(GHZ)
mixed_GHZ_dm = MixedState(perfect_GHZ_dm)
noisy_GHZ_func(λ) = λ*perfect_GHZ_dm + (1-λ)*mixed_GHZ_dm

F_ins = []
F_outs = []
F_GHZs = []
for F_GHZ in 0.99:0.002:1.0
    for F_in in 0.97:0.002:1.0
        while true
            #F_in = (1+3*λ_bell)/4
            λ_bell = (4*F_in - 1)/3
            # F_GHZ = (1+7*λ_GHZ)/8
            λ_GHZ = (8*F_GHZ - 1)/7
            reg1 = Register(3)
            reg2 = Register(3)
            reg3 = Register(3)
            initialize!([reg1[1], reg2[1], reg3[1]], noisy_GHZ_func(λ_GHZ))
            initialize!([reg1[2], reg2[2]], noisy_pair_func(λ_bell))
            initialize!([reg2[3], reg3[3]], noisy_pair_func(λ_bell))
            initialize!([reg1[3], reg3[2]], noisy_pair_func(λ_bell))

            

            resZ_1 = Purify2to1Node(:Z)(reg1[1], reg1[2]) == Purify2to1Node(:Z)(reg2[1], reg2[2])
            resZ_2 = Purify2to1Node(:Z)(reg2[1], reg2[3]) == Purify2to1Node(:Z)(reg3[1], reg3[3])
            #resZ_3 = Purify2to1Node(:Z)(reg1[1], reg1[3]) == Purify2to1Node(:Z)(reg3[1], reg3[2])

            if resZ_1 && resZ_2 #&& resZ_3
                fidel = real(observable([reg1[1], reg2[1], reg3[1]], perfect_GHZ_dm))
                push!(F_ins, F_in)
                push!(F_outs, fidel)
                push!(F_GHZs, F_GHZ)
                break
            end
        end
    end
end

using DataFrames
df = DataFrame(F_GHZ = F_GHZs, F_in = F_ins, F_out = F_outs, F_diff = F_outs .- F_ins)
df[!, :F_inf] = 1 .-round.(df[!, :F_out], digits=3)
df[!, :F_GHZ] = round.(df[!, :F_GHZ], digits=3)
##
using StatsPlots

#@df df plot(:F_in, :F_diff, group=:λ_GHZ, xlabel="Bell pair fidelity", ylabel="Fidelity Out - Fidelity In", title="GHZ Purification Trial", legendtitle ="λ_GHZ", markershape=:circle)
@df df plot(:F_in, :F_inf, group=:F_GHZ, xlabel="Bell pair fidelity", ylabel="GHZ final fidelity", title="3-GHZ Purification Trial", legendtitle ="F_GHZ", markershape=:circle)
#plot!(unique(F_ins), unique(F_GHZs), label="F_GHZ init", linestyle=:dash, color=:black)