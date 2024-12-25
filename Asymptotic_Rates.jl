"""
GENERAL INFORMATION

This code calculates asymptotic secret key rates under the cutoff assumption for the
QPSK protocol (prepare-and-measure) based on coarse-grained probabilities. The resulting 
states are also later processed for the finite key analysis in FKRates.jl.

The code is executed via the function Instance(.), which takes as input
    Nc  - Value of the cutoff, typically ≥10
    δ,Δ - Parameters of the modulation, usually 2 and ≥ 4 respectively
    f   - Error correction efficienty, take 0 for the Shannon limit or > 0 otherwise
    T   - DataType. Usually Float64 for distances below 100 km, and Double64 otherwise

As an output, the code returns the asymptotic secret key rate at Distances from 0 to 200 km
together with the optimal state (which is necessary for the finite analysis).
"""

using ConicQKD # ] add https://github.com/araujoms/ConicQKD.jl.git
using Ket


using  SpecialFunctions
using  LinearAlgebra
using  DoubleFloats
using  JuMP
using  Printf
import Hypatia
import Hypatia.Cones
import Integrals

include("Utils.jl")


"""
    hbe(::Type{T}, Nc::Integer, δ::T, Δ::T, f::T, D::Integer) where {T<:AbstractFloat}

Computes the conditional von Neumann entropy H(B|E) of Bob's raw key
register given Eve's quantum information. This function performs the
calculation of said entropy via conic optimization.
"""
function hbe(::Type{T}, Nc::Integer, δ::T, Δ::T, f::T, D::Integer) where {T<:AbstractFloat}

    α = T(optimal_amplitude(D,f))
    dim_τAB = 4*(Nc+1)

    model = GenericModel{T}()
    @variable(model, τAB[1:dim_τAB, 1:dim_τAB], Hermitian)

    p_τAB = constraint_probabilities(T,τAB,δ,Δ,Nc)
    p_sim = simulated_probabilities(T,δ,Δ,α,D)

    for x=1:4
        for z=1:5
            @constraint(model, p_τAB[x,z] == p_sim[x,z])
        end
      
        # z = 6
        # These constraints are redundant, like tr(τAB) == 1
        # @constraint(model, p_τAB[x,z] == p_sim[x,z])
    end

    τA = partial_trace(T(1)*τAB, 2, [4, Nc+1])
    @constraint(model, τA == alice_part(α)) #this already implies tr(τAB) == 1

    G = gkraus(T,Nc)
    Ghat = [I(dim_τAB)]
    Z = zkraus(Nc)
    Zhat = [Zi*G for Zi in Z]

    permutation = vec(reshape(1:16*(Nc+1),4,4*(Nc+1))')
    Zhatperm = [Zi[permutation,:] for Zi in Zhat]
    block_size = 4*(Nc+1)
    blocks = [(i-1)*block_size+1:i*block_size for i=1:4]
    
    vec_dim = Cones.svec_length(Complex,dim_τAB)
    τAB_vec = svec(τAB, Complex{T})

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; τAB_vec] in EpiQKDTriCone{T,Complex{T}}(Ghat,Zhatperm,1 + vec_dim;blocks))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)

    return value.(τAB), dual_objective_value(model)
end

"""
    CompleteCode(::Type{T},Nc::Integer,δ::T,Δ::T,f::T,D::Integer,NAMES::Vector{String}=[""]) where {T<:AbstractFloat}

Performs the complete computation of the asymptotic secret key rate at a given distance. Takes as inputs \n
    T   - DataType. Usually Float64
    Nc  - Value of the cutoff, typically ≥10
    δ,Δ - Parameters of the modulation, usually 2 and ≥ 4 respectively 
    f   - Error correction efficienty, take 0 for the Shannon limit or > 0 otherwise 
    D   - Distance in km 
    NAMES - a vector of strings of length 2. First string is the name of the rate file 
            (i.e. asymptotic rate, together with diverse parameters) and the second is 
            the name of the primal file (i.e. density matrix of the primal solution)
"""
function CompleteCode(::Type{T},Nc::Integer,δ::T,Δ::T,f::T,D::Integer,NAMES::Vector{String}=[""]) where {T<:AbstractFloat}

    # Compute the relative entropy H(B|E)
    τAB, dual_ObjF = hbe(T, Nc, δ, Δ, f, D)
    
    # Compute the EC cost H(B|A) + EC inefficiency
    α   = T(optimal_amplitude(D,f))
    hba = EC_cost(α,D,f)

    # Print data
    @printf("ObjVal:  %.6f \n",dual_ObjF)
    @printf("EC cost: %.6f \n",hba)

    # Derived quantities
    R_dl = dual_ObjF-hba

    # Store the data
    if NAMES !=[""]
        FILE_RATE = open(NAMES[1],"a")
        @printf(FILE_RATE,"%d, %.2f, %.8e, %.12f \n",D,α,R_dl,hba)
        close(FILE_RATE)

        FILE_PRIMAL = open(NAMES[2],"a")
        @printf(FILE_PRIMAL,"%d, %.2f",D,α)

        HτAB = Hvec(τAB)
        for pvar in HτAB
            @printf(FILE_PRIMAL,", %.16e",pvar)
        end
        @printf(FILE_PRIMAL,"\n")
        close(FILE_PRIMAL)
    end

end

"""
    Instance(Nc::Integer,δ::Real,Δ::Real,f::Real,T::DataType=Float64)

Computes the entire series of asymptotic secret key rates at different distances, given the parameters \n
    T   - DataType. Usually Float64
    Nc  - Value of the cutoff, typically ≥10
    δ,Δ - Parameters of the modulation, usually 2 and ≥ 4 respectively 
    f   - Error correction efficienty, take 0 for the Shannon limit or > 0 otherwise 
"""
function Instance(Nc::Integer,δ::Real,Δ::Real,f::Real,T::DataType=Float64)

    ### Friendly reminder of basic parameters
    # ξ     = 0.01 (excess noise in SNUs)
    # α_att = 0.2  (attenuation at the fiber in dB/km)

    # Change data types if necessary
    δ = T(δ)
    Δ = T(Δ)
    f = T(f)

    ############################################
    ### DATA OF OUTPUTS
    ############################################

    NAME_RATE   = "Rate_f"*string(Int(floor(f*100)))*"D"*string(Int(floor(Δ*10)))*"d"*string(Int(floor(δ*10)))*".csv"
    NAME_PRIMAL = "Primal_f"*string(Int(floor(f*100)))*"D"*string(Int(floor(Δ*10)))*"d"*string(Int(floor(δ*10)))*".csv"
    
    NAMES       = [NAME_RATE,NAME_PRIMAL]

    FILE_RATE = open(NAMES[1],"a")
    @printf(FILE_RATE,"xi, f, Nc, delta, Delta \n")
    @printf(FILE_RATE,"0.01, %.2f, %d, %.2f, %.2f  \n",f,Nc,δ,Δ)
    @printf(FILE_RATE,"D, amp, R_dl, deltaEC \n")
    close(FILE_RATE)

    FILE_PRIMAL = open(NAMES[2],"a")
    @printf(FILE_PRIMAL,"D, amp, tau \n")
    close(FILE_PRIMAL)

    ############################################
    ### MAIN LOOPS
    ############################################

    for D in 0:70
        @printf("Distance: %d ---------\n",D)
        CompleteCode(T,Nc,δ,Δ,f,D,NAMES)
    end

    for D in 75:5:200
        @printf("Distance: %d ---------\n",D)
        CompleteCode(T,Nc,δ,Δ,f,D,NAMES)
    end


end
