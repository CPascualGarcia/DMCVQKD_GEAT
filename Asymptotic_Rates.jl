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



function integrand(vars,pars)
    γ = vars[1]
    θ = vars[2]
    ξ = pars[1]
    η = pars[2]
    x = pars[3]
    α = pars[4]
    return γ*exp(-abs2(γ*exp(im*θ)-sqrt(η)*im^x*α)/(1+η*ξ/2))
end

function integrate(bounds, pars)
    T = eltype(pars)
    problem = Integrals.IntegralProblem(integrand, bounds, pars)
    tol = T == Float64 ? eps(T) : sqrt(eps(T))
    sol = Integrals.solve(problem, Integrals.HCubatureJL(); reltol = tol, abstol = tol)
    return sol.u
end

"""
    simulated_probabilities(::Type{T}, δ::T, Δ::T, α::T, D::Integer) where {T<:AbstractFloat}

Computes the probability distribution for the implementation of the protocol given the
modulation (δ, Δ) of the implementation, the amplitude of the coherent states α and 
the distance D between Alice and Bob.
"""
function simulated_probabilities(::Type{T}, δ::T, Δ::T, α::T, D::Integer) where {T<:AbstractFloat}
    α_att = T(2)/10
    α_eff = T(0)
    ξ = T(1)/100
    η = 10^(-(α_att*D+α_eff)/10)
    p_sim = zeros(T,4,6)
    for x=0:3
        pars = [ξ, η, x, α]
        for z = 0:3
            bounds = ([T(0), T(π)*(2*z-1)/4], [T(δ), T(π)*(2*z+1)/4])
            p_sim[x+1,z+1] = integrate(bounds,pars)
        end
        #z = 4
        bounds = ([T(δ), T(0)], [T(Δ), 2*T(π)])
        p_sim[x+1,5] = integrate(bounds,pars)
        #z = 5
        bounds = ([T(Δ), T(0)], [T(Inf), 2*T(π)])
        p_sim[x+1,6] = integrate(bounds,pars)
    end
    p_sim ./= 4*T(π)*(1+η*ξ/2)
    return p_sim
end

"""
    Hvec(H::AbstractMatrix)

Converts a Hermitian matrix H into a vector [diag(H);sqrt(2)*Re(Upp(H));sqrt(2)*Im(Upp(H))].
"""
function Hvec(H::AbstractMatrix)
    if  ~ishermitian(H)
        error("Wrong input! The matrix should be Hermitian!")
    end
    
    n   = size(H)[1]
    ind = BitMatrix(triu(ones(n,n),1))
    Hv  = real([diag(H);√2*[real(H[ind[:]]);imag(H[ind[:]])]])'

    return Hv
end

"""
    alice_part(α::Real)

Computes Alice's marginal state.
Tr{B} [τ{AB}] =  ∑_{j,k} |j><k| <α i^k|α i^j>/4
"""
function alice_part(α::Real)
    ρ = Hermitian(ones(Complex{typeof(α)},4,4))
    ρ.data[1,2] = exp(-(1+im)*α^2)
    ρ.data[1,3] = exp(-2*α^2)
    ρ.data[1,4] = exp(-(1-im)*α^2)
    ρ.data[2,3] = ρ.data[1,2]
    ρ.data[2,4] = ρ.data[1,3]
    ρ.data[3,4] = ρ.data[1,2]
    ρ *= 0.25
end

"""
    sinkpi4(::Type{T}, k::Integer) where {T<:AbstractFloat}

Computes sin(k*π/4) with high precision.
"""
function sinkpi4(::Type{T}, k::Integer) where {T<:AbstractFloat}
    if mod(k,4) == 0
        return T(0)
    else
        signal = T((-1)^div(k,4,RoundDown))
        if mod(k,2) == 0
            return signal
        else
            return signal/sqrt(T(2))
        end
    end
end

"""
    key_basis(::Type{T}, Nc::Integer) where {T<:AbstractFloat}

Computes the basis of operators for the key map.
"""
function key_basis(::Type{T}, Nc::Integer) where {T<:AbstractFloat}
    R = [Hermitian(zeros(Complex{T},Nc+1,Nc+1)) for z=0:3]
    for z = 0:3
        for n=0:Nc
            for m=n:Nc
                if n == m
                    R[z+1][n+1,m+1] = T(1)/4
                else
                    angular = 2*im^(mod(z*(n-m),4))*sinkpi4(T,n-m)/(n-m)
                    radial = gamma(1 + T(n+m)/2)/(2*T(π)*sqrt(gamma(T(1+n))*gamma(T(1+m))))                    
                    R[z+1].data[n+1,m+1] = angular*radial
                end
            end
        end
    end
    return R
end

"""
    test_basis(::Type{T}, δ::T, Δ::T, Nc::Integer) where {T<:AbstractFloat}

Computes the basis of operators for the parameter estimation map,
according to the chosen modulation.
"""
function test_basis(::Type{T}, δ::T, Δ::T, Nc::Integer) where {T<:AbstractFloat}
    R = [Hermitian(zeros(Complex{T},Nc+1,Nc+1)) for z=0:5]
    for z = 0:3
        for n=0:Nc
            for m=n:Nc
                if n == m
                    R[z+1][n+1,m+1] = (gamma(T(1+n)) - gamma(T(1+n),δ^2))/(4*gamma(T(1+n)))
                else
                    angular = 2*im^(mod(z*(n-m),4))*sinkpi4(T,n-m)/(n-m)
                    radial = (gamma(1 + T(n+m)/2) - gamma(1 + T(n+m)/2,δ^2))/(2*T(π)*sqrt(gamma(T(1+n))*gamma(T(1+m))))                    
                    R[z+1].data[n+1,m+1] = angular*radial
                end
            end
        end
    end
    for n=0:Nc
        R[5][n+1,n+1] = (gamma(T(1+n),δ^2) - gamma(T(1+n),Δ^2))/gamma(T(1+n))
        R[6][n+1,n+1] = gamma(T(1+n),Δ^2)/gamma(T(1+n))
    end
    return R
end

"""
    optimal_amplitude(D::Integer,f::Real)

Optimal amplitude for the coherent states according to the distance
between Alice and Bob, and the EC efficiency.
"""
function optimal_amplitude(D::Integer,f::Real)
    if f==0.05
        amplitudes = [1.05, 1.03, 1.01, 0.99, 0.97, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.85, 0.84, 0.83, 0.82, 0.81, 0.81, 0.80, 0.80, 0.79, 0.79, 0.79, 0.78, 0.78, 0.77, 0.76, 0.75, 0.75, 0.74, 0.74, 0.73, 0.73, 0.73, 0.72, 0.72, 0.71]
        return D <= 40 ? amplitudes[D+1] : 0.7
    elseif f==0.1
        amplitudes = [1.07, 1.02, 1.01, 1.00, 0.99, 0.98, 0.97, 0.96, 0.94, 0.92, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.79, 0.78, 0.77, 0.76, 0.76, 0.76, 0.75, 0.75, 0.74, 0.73, 0.73, 0.72, 0.72, 0.71]
        return D <= 35 ? amplitudes[D+1] : 0.71
    else # Default to f = 0
        amplitudes = [1.07, 1.05, 1.03, 1.01, 0.99, 0.96, 0.93, 0.90, 0.87, 0.85, 0.85, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.79, 0.79, 0.78, 0.78, 0.77, 0.77, 0.77, 0.77, 0.77, 0.76, 0.76, 0.75, 0.75, 0.75, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
        return D <= 70 ? amplitudes[D+1] : 0.71
    end
end    

"""
    constraint_probabilities(::Type{T}, ρ::AbstractMatrix, δ::T, Δ::T, Nc::Integer) where {T<:AbstractFloat}

Builds the trace of the state and each of the operators in the test basis
for Alice and Bob.
"""
function constraint_probabilities(::Type{T}, ρ::AbstractMatrix, δ::T, Δ::T, Nc::Integer) where {T<:AbstractFloat}
    R_B = test_basis(T,δ,Δ,Nc)
    bases_AB = [kron(proj(x+1,4),R_B[z+1]) for x=0:3, z=0:5]
    return real(dot.(Ref(ρ),bases_AB))
end

"""
    gmap(::Type{T}, ρ::AbstractMatrix, Nc::Integer) where {T<:AbstractFloat}

Quantum channel of the key map.
"""
function gmap(::Type{T}, ρ::AbstractMatrix, Nc::Integer) where {T<:AbstractFloat}
    V = gkraus(T,Nc)
    return Hermitian(V * ρ * V')
end

"""
    gkraus(::Type{T}, Nc::Integer) where {T<:AbstractFloat}

Isometry corresponding to the key map.
"""
function gkraus(::Type{T}, Nc::Integer) where {T<:AbstractFloat}
    sqrtbasis = sqrt.(key_basis(T,Nc))
    V = sum( kron(I(4),sqrtbasis[i],ket(i,4)) for i = 1:4)
    return V
end

"""
    zmap(ρ::AbstractMatrix, Nc::Integer)

Quantum channel of the pinching map.
"""
function zmap(ρ::AbstractMatrix, Nc::Integer)
    K = zkraus(Nc)
    return Hermitian(sum(K[i] * ρ * K[i] for i = 1:4))
end

"""
    zkraus(Nc::Integer)

Pinching map.
"""
function zkraus(Nc::Integer)
    K = [kron(I(4 * (Nc + 1)), proj(i, 4)) for i = 1:4]
    return K
end

"""
    EC_cost(α::T,D::Integer,f::T) where {T<:AbstractFloat}

Computes the cost of error correction, incorporating the efficiency
"""
function EC_cost(α::T,D::Integer,f::T) where {T<:AbstractFloat}
    α_att = T(2)/10
    α_eff = T(0)
    ξ = T(1)/100
    η = 10^(-(α_att*D+α_eff)/10)
    p_EC  = zeros(T,4,4)               # Conditional probability p(z|x)

    for x=0:3
        pars = [ξ, η, x, α]
        for z=0:3
            bounds        = ([T(0),T(π)*(2*z-1)/4],[T(Inf),T(π)*(2*z+1)/4])
            p_EC[x+1,z+1] = integrate(bounds,pars)
        end
    end
    p_EC ./= T(π)*(1+η*ξ/2)
    Hba   = -(1+T(f))*p_EC[:]'*log2.(p_EC[:])*T(0.25)
    return Hba
end


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
