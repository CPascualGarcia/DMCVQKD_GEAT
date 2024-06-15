push!(LOAD_PATH,"Ket/")
push!(LOAD_PATH,"ConicQKD/")
using ConicQKD
using Ket


using  SpecialFunctions
using  LinearAlgebra
#using  DoubleFloats
using  JuMP
using  Printf
import Hypatia
import Hypatia.Cones
import Convex
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
    sol = Integrals.solve(problem, Integrals.HCubatureJL(); reltol = Base.rtoldefault(T), abstol = Base.rtoldefault(T))
    return sol.u
end

function simulated_probabilities(::Type{T}, δ::Real, Δ::Real, α::Real, D::Integer) where {T}
    α_att = T(0.18)
    α_eff = T(0.0)
    ξ = T(0.01)
    η = 10^(-(α_att*D+α_eff)/T(10))
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
    p_sim /= 4*T(π)*(1+η*ξ/2)
    return p_sim
end

penalty_prob(p_sim) = sum(p_sim[:,6])

penalization_term(v) = sqrt(2*v-v^2)*2 + (1+sqrt(2*v-v^2))*binary_entropy(sqrt(2*v-v^2)/(1+sqrt(2*v-v^2)))

function Hvec(H::AbstractMatrix)
    if  ~ishermitian(H)
        error("Wrong input! The matrix should be Hermitian!")
    end
    
    n   = size(H)[1]
    ind = BitMatrix(triu(ones(n,n),1))
    Hv  = real([diag(H);√2*[real(H[ind[:]]);imag(H[ind[:]])]])'

    return Hv
end

function Hmat(v::AbstractArray)
    n   = Int(sqrt(length(v)))
    ind = BitMatrix(triu(ones(n,n),1))

    H  = zeros(ComplexF64,n,n)
    tn = Int((n^2-n)/2)
    vR = v[n+1:n+tn]
    vI = v[n+tn+1:end]

    H[ind] = (vR[:]+im*vI[:])/sqrt(2)
    H += diagm(v[1:n]) +H'
    
    return H

end


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

function sinkpi4(::Type{T}, k::Integer) where {T} #computes sin(k*π/4) with high precision
    if mod(k,4) == 0
        return 0
    else
        signal = (-1)^div(k,4,RoundDown)
        if mod(k,2) == 0
            return signal
        else
            return signal/sqrt(T(2))
        end
    end
end

function key_basis(::Type{T}, Nc::Integer) where {T}
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
#    R[4] = Hermitian(I(Nc+1)) - sum(R[z+1] for z=0:2)
    return R
end


function test_basis(::Type{T}, δ::Real, Δ::Real, Nc::Integer) where {T}
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

function optimal_amplitude(D::Integer)
    amplitudes = [1.07, 1.05, 1.03, 1.01, 0.99, 0.96, 0.93, 0.9, 0.87, 0.85, 0.85, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.83, 0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.79, 0.79, 0.78, 0.78, 0.77, 0.77, 0.77, 0.77, 0.77, 0.76, 0.76, 0.75, 0.75, 0.75, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
    return D <= 70 ? amplitudes[D+1] : 0.71
end    

function constraint_probabilities(::Type{T}, ρ::AbstractMatrix, δ::Real, Δ::Real, Nc::Integer) where {T}
    R_B = test_basis(T,δ,Δ,Nc)
    bases_AB = [kron(proj(x+1,4),R_B[z+1]) for x=0:3, z=0:5]
    return real(dot.(Ref(ρ),bases_AB))
end

function gmap(::Type{T}, ρ::AbstractMatrix, Nc::Integer) where {T}
    V = gkraus(T,Nc)
    return Hermitian(V * ρ * V')
end

function gkraus(::Type{T}, Nc::Integer) where {T}
    sqrtbasis = sqrt.(key_basis(T,Nc))
    V = sum( kron(I(4),sqrtbasis[i],ket(i,4)) for i = 1:4)
    return V
end

function zmap(ρ::AbstractMatrix, Nc::Integer, T::DataType = Float64)
    K = [kron(I(4*(Nc+1)), proj(i,4)) for i=1:4]
    return Hermitian(sum(K[i] * ρ * K[i] for i = 1:4))
end

function zkraus(Nc::Integer)
    K = [kron(I(4 * (Nc + 1)), proj(i, 4)) for i = 1:4]
    return K
end

function completecutoff(::Type{T}, Nc::Integer, δ::Real, Δ::Real, D::Integer) where {T}
    δ = T(δ)
    Δ = T(Δ)
    α  = T(optimal_amplitude(D))
    R4 = zeros(T,51)
    R5 = zeros(T,51)
    
    for n=0:50
        R4[n+1] = (gamma(T(1+n),δ^2) - gamma(T(1+n),Δ^2))/gamma(T(1+n))
        R5[n+1] = gamma(T(1+n),Δ^2)/gamma(T(1+n))
    end
    
    y5 = gamma(2+Nc)*(gamma(1,δ^2)-gamma(1,Δ^2))/(gamma(2+Nc,Δ^2)*gamma(1,δ^2) - gamma(2+Nc,δ^2)*gamma(1,Δ^2))
    y4 = -y5*gamma(1,Δ^2)/(gamma(1,δ^2)-gamma(1,Δ^2))
    
    p_sim = simulated_probabilities(T,δ,Δ,α,D)
    p4 = sum(p_sim[:,5])
    p5 = sum(p_sim[:,6])

    # Check the output
    constraint = (R4*y4+R5*y5) 

    # Note the eps(T) bc of slight unfeasibility of the solution
    @assert all(i-> i+eps(T) ≥ 0, constraint[1:Nc+1])     "Failure at cutoff choice"
    @assert all(i-> i+eps(T) ≥ 1, constraint[Nc+2:end])   "Failure at cutoff"
         
    v  = p4*y4 + p5*y5
    Ω  = penalization_term(v)
    @printf("Cutoff:  %.6e \n",Ω)
    # @printf("Penalization: %e \n",Ω)

    return Ω, v, y4, y5
end

function EC_cost(α::Real,D::Integer,f::Real,T::DataType=Float64)
    α_att = T(0.18) 
    α_eff = T(0)  #T(3.0)
    ξ     = T(0.01)
    η     = 10^(-(α_att*D+α_eff)/10)
    p_EC  = zeros(T,4,4)               # Conditional probability p(z|x)

    for x=0:3
        pars = [ξ, η, x, α]
        for z=0:3
            bounds        = ([T(0),T(π)*(2*z-1)/4],[T(Inf),T(π)*(2*z+1)/4])
            p_EC[x+1,z+1] = integrate(bounds,pars)
        end
    end
    p_EC /= T(π)*(1+η*ξ/2)
    Hba   = -(1+T(f))*p_EC[:]'*log2.(p_EC[:])*T(0.25)
    return Hba
end


function operator_norms(δ::T,Δ::T,Nc::Integer) where {T<:AbstractFloat}
    normR_B = zeros(T,6)
    R       = test_basis(T,δ,Δ,50)
    for z in 0:3
        normR_B[z+1] = opnorm(R[z+1],Inf)
    end
    normR_B[5] = opnorm(R[5][Nc+2:end,Nc+2:end],Inf)
    normR_B[6] = 1

    return normR_B
end


function MaxMinfunction(cons,v,y4,y5,a,b,T)
    constr = [JuMP.dual.(cons[1])'; JuMP.dual.(cons[2])';JuMP.dual.(cons[3])';JuMP.dual.(cons[4])']
    ad     = dual(a)
    bd     = dual.(b)[1]
    
    duals  = zeros(4,6)

    Derivative_v = -2*(1-v)/sqrt(2*v-v^2)
    # Dual variables related to Ω
    Derivative_Ω = (penalization_term(v + eps(T))-penalization_term(v-eps(T)))/(2eps(T))


    ν      = zeros(4,4)
    for x=1:4
        for z=1:4
            ν[x,z]     = constr[x,2*z-1]*Derivative_v
            duals[x,z] = constr[x,2*z-1] - constr[x,2*z]
        end
    end
    
    # This has to be added to the constraints of p(x,4), p(x,5)
    γ = zeros(4,2)
    for x=1:4
        γ[x,1] = (sum(ν[x,:])- ad - bd) *y4
        duals[x,5] = constr[x,9]  - constr[x,10] + γ[x,1]
        duals[x,5]-= Derivative_Ω*y4

        γ[x,2] = (sum(ν[x,:])-ad - bd)*y5
        if length(constr[x,:])>10
            duals[x,6] = constr[x,11] - constr[x,12] + γ[x,2]
        else
            duals[x,6] = γ[x,2]
        end
        duals[x,6]-= Derivative_Ω*y5
    end

    
    M = maximum(duals)
    m = minimum(duals)

    MaxMinf = M-m
    @printf("Maxminf: %f \n",MaxMinf)
    return MaxMinf, duals
end



function hae(::Type{T}, Nc::Integer, δ::Real, Δ::Real, D::Integer,minutes::Real) where {T}
    δ = T(δ)
    Δ = T(Δ)
    α = T(optimal_amplitude(D))
    dim_τAB = 4*(Nc+1)
    Ω, v, y4, y5 = completecutoff(T,Nc,δ,Δ,D)
    Ω  = T(Ω)
    v  = T(v)
    y4 = T(y4)
    y5 = T(y5)

    model = GenericModel{T}()
    @variable(model, τAB[1:dim_τAB, 1:dim_τAB], Hermitian)
    
    a = @constraint(model, tr(τAB) >= 1-v)
    @constraint(model,     tr(τAB) <= 1)

    set_time_limit_sec(model, 60*Float64(minutes))

    # normR_B    = opnorm.(test_basis(δ,Δ,Nc,T),Inf)
    normR_B = operator_norms(δ,Δ,Nc)
    p_τAB = constraint_probabilities(T,τAB,δ,Δ,Nc)
    p_sim = simulated_probabilities(T,δ,Δ,α,D)

    cons = []
    for x=1:4
        cx = []
        for z=1:4
            c = @constraint(model, p_τAB[x,z] ≥ p_sim[x,z] - 2*sqrt(2v-v^2)*normR_B[z])
            push!(cx,c)
            c = @constraint(model, p_τAB[x,z] <= p_sim[x,z])
            push!(cx,c)
        end

        z = 5
        c = @constraint(model, p_τAB[x,z] ≥  p_sim[x,z] - v*normR_B[z])
        push!(cx,c)
        c = @constraint(model, p_τAB[x,z] <= p_sim[x,z])
        push!(cx,c)

        
        z = 6
        # This constraint is redundant, but helps with MaxMinf
        c = @constraint(model, p_τAB[x,z] ≥ p_sim[x,z] - v*normR_B[z])
        push!(cx,c)
        c = @constraint(model, p_τAB[x,z] <= p_sim[x,z])
        push!(cx,c)

        push!(cons,cx)
    end

    τA = Hermitian(Convex.partialtrace(T(1)*τAB, 2, [4, Nc+1]))
    σA = alice_part(α)
    vec_dimA = 4^2
    τA_vec = Vector{GenericAffExpr{T, GenericVariableRef{T}}}(undef, vec_dimA)
    σA_vec = Vector{GenericAffExpr{T, GenericVariableRef{T}}}(undef, vec_dimA)
    Cones._smat_to_svec_complex!(τA_vec, τA, sqrt(T(2)))
    Cones._smat_to_svec_complex!(σA_vec, σA, sqrt(T(2)))
    b = @constraint(model, [2*sqrt(2v-v^2); σA_vec - τA_vec] in Hypatia.EpiNormSpectralTriCone{T,Complex{T}}(1 + vec_dimA, true))

    G = gkraus(T,Nc)
    Ghat = [I(dim_τAB)]
    Z = zkraus(Nc)
    Zhat = [Zi*G for Zi in Z]

    permutation = vec(reshape(1:16*(Nc+1),4,4*(Nc+1))')
    Zhatperm = [Zi[permutation,:] for Zi in Zhat]
    block_size = 4*(Nc+1)
    blocks = [(i-1)*block_size+1:i*block_size for i=1:4]

    
    vec_dim = Cones.svec_length(Complex,dim_τAB)
    τAB_vec = Vector{GenericAffExpr{T, GenericVariableRef{T}}}(undef, vec_dim)    
    Cones._smat_to_svec_complex!(τAB_vec, τAB, sqrt(T(2)))

    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; τAB_vec] in EpiQKDTriCone{T,Complex{T}}(Ghat,Zhatperm,1 + vec_dim;blocks))

    set_optimizer(model, Hypatia.Optimizer{T})
    set_attribute(model, "verbose", true)
    optimize!(model)
    @printf("%.5e \n",objective_value(model))

    # Get the dual variables of the optimization
    dual_vars = [dual.(cons[x]) for x=1:4]

    # Compute the spread and duals of the min-tradeoff function
    MaxMinf, dual_linear =  MaxMinfunction(cons,v,y4,y5,a,b,T)

    return dual_vars, dual_linear, value.(τAB), dual_objective_value(model), Ω, v  # objective_value(model) 
end

#hae(4,1,2,70) = 1.9595173004591733, 559.073 seconds
#hae(4,1,2,70) = 1.959578468709351, 113.778 seconds


function Instance(Nc::Integer,δ::Real,Δ::Real,minutes::Real,T::DataType=Float64)

    ### Friendly reminder of basic parameters
    # ξ     = 0.01
    # f     = 0.0
    # α_att = 0.18


    ############################################
    ### DATA OF OUTPUTS
    ############################################

    NAME_RATE   = "Rate_D"*string(Int(floor(Δ*10)))*"d"*string(Int(floor(δ*10)))*".csv"
    NAME_PRIMAL = "Primal_D"*string(Int(floor(Δ*10)))*"d"*string(Int(floor(δ*10)))*".csv"
    NAME_DUAL   = "Dual_D"*string(Int(floor(Δ*10)))*"d"*string(Int(floor(δ*10)))*".csv"
    NAMES       = [NAME_RATE,NAME_PRIMAL,NAME_DUAL]

    FILE_RATE = open(NAMES[1],"a")
    @printf(FILE_RATE,"xi, f, Nc, delta, Delta \n")
    @printf(FILE_RATE,"0.01, 0.0, %d, %.2f, %.2f \n",Nc,δ,Δ)
    @printf(FILE_RATE,"D, amp, R_dl, deltaEC, Omega, MaxMin \n")
    close(FILE_RATE)

    FILE_PRIMAL = open(NAMES[2],"a")
    @printf(FILE_PRIMAL,"D, amp, tau \n")
    close(FILE_PRIMAL)

    FILE_DUAL = open(NAMES[3],"a")
    @printf(FILE_DUAL,"D, amp, dvars \n")
    close(FILE_DUAL)


    for D in 0:70
        @printf("Distance: %d ---------\n",D)
        CompleteCode(Nc,δ,Δ,D,minutes,NAMES,T)
    end

    for D in 75:5:200
        @printf("Distance: %d ---------\n",D)
        CompleteCode(Nc,δ,Δ,D,minutes,NAMES,T)
    end


end


function CompleteCode(Nc::Integer,δ::Real,Δ::Real,D::Real,minutes::Real,NAMES::Vector{String}=[""],T::DataType=Float64)

    # Compute the relative entropy H(B|E)
    dual_vars, dual_linear, τAB, dual_ObjF, Ω, v = hae(T,Nc,δ,Δ,D,minutes)

    # Compute the EC cost H(B|A) at the Shannon limit
    α   = T(optimal_amplitude(D))
    hba = EC_cost(α,D,0.0,T)

    # Compute the spread of the min-tradeoff function
    M = maximum(dual_linear)
    m = minimum(dual_linear)

    MaxMinf = M-m
    @printf("Maxminf: %.6f \n",MaxMinf)

    # Print data
    @printf("ObjVal:  %.6f \n",dual_ObjF)
    @printf("EC cost: %.6f \n",hba)
    @printf("Cutoff:  %.6e \n",Ω)

    # Derived quantities
    R_dl = dual_ObjF-hba

    # Store the data
    if NAMES !=[""]
        FILE_RATE = open(NAMES[1],"a")
        @printf(FILE_RATE,"%d, %.2f, %.8e, %.12f, %.8e, %.6f \n",D,α,R_dl,hba,Ω,MaxMinf)
        close(FILE_RATE)

        FILE_PRIMAL = open(NAMES[2],"a")
        @printf(FILE_PRIMAL,"%d, %.2f",D,α)

        HτAB = Hvec(τAB)
        for pvar in HτAB
            @printf(FILE_PRIMAL,", %.12e",pvar)
        end
        @printf(FILE_PRIMAL,"\n")
        close(FILE_PRIMAL)

        FILE_DUAL = open(NAMES[3],"a")
        @printf(FILE_DUAL,"%d, %.2f",D,α)
        for dvar in dual_linear
            @printf(FILE_DUAL,", %.12f",dvar)
        end
        @printf(FILE_DUAL,"\n")
        close(FILE_DUAL)
    end

end


# Nc=3;δ=2.0;Δ=4.9;D=10;minutes=5;T=Float64;
