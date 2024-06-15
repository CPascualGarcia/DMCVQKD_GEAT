function integrand(vars,pars)
    γ = vars[1]
    θ = vars[2]
    ξ = pars[1]
    η = pars[2]
    x = pars[3]
    α = pars[4]
    return γ*exp(-abs2(γ*exp(im*θ)-sqrt(η)*im^x*α)/(1+η*ξ/2))
end

function integrate(bounds,pars)
    problem = Integrals.IntegralProblem(integrand,bounds,pars)
    sol = Integrals.solve(problem, Integrals.HCubatureJL(); reltol = Base.rtoldefault(T), abstol = Base.rtoldefault(T))
    return sol.u
end

function simulated_probabilities(::Type{T}, δ::Real, Δ::Real, α::Real, D::Integer) where {T}
    α_att = T(0.2)
    α_eff = T(0.0)
    ξ = T(0.01)
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
    p_sim /= 4*T(π)*(1+η*ξ/2)
    return p_sim
end

function Hvec(H::AbstractMatrix)
    if  ~ishermitian(H)
        error("Wrong input! The matrix should be Hermitian!")
    end
    
    n   = size(H)[1]
    ind = BitMatrix(triu(ones(n,n),1))
    Hv  = real([diag(H);√2*[real(H[ind[:]]);imag(H[ind[:]])]])'

    return Hv
end

function Hmat(v::AbstractVector)
    n   = Int(sqrt(length(v)))
    ind = BitMatrix(triu(ones(n,n),1))

    H  = zeros(ComplexF64,n,n) 
    tn = Int((n^2-n)/2)
    vR = v[n+1:n+tn]
    vI = v[n+tn+1:end]

    H[ind] = (vR[:]+im*vI[:])/sqrt(2)
    H .+= diagm(v[1:n]) +H'
    
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

function zmap(ρ::AbstractMatrix, Nc::Integer)
    K = [kron(I(4*(Nc+1)), proj(i,4)) for i=1:4]
    return Hermitian(sum(K[i] * ρ * K[i] for i = 1:4))
end

function zkraus(Nc::Integer)
    K = [kron(I(4 * (Nc + 1)), proj(i, 4)) for i = 1:4]
    return K
end

function EC_cost(α::Real,D::Integer,f::Real,T::DataType=Float64)
    α_att = T(0.2) 
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

function constraint_operators(::Type{T},δ::Real,Δ::Real,Nc::Integer) where {T}
    R_B = test_basis(T,δ,Δ,Nc)

    PE_AB = Vector{Matrix{Complex{T}}}(undef,0)
    for x=0:3
        PE_AB = append!(PE_AB,[kron(proj(x+1,4),R_B[z+1]) for z=0:5])
    end
    return PE_AB
end

function alice_tomography(::Type{T},α::Real,Nc::Int) where {T}
    Λ_A  = diagm(ones(T,16))
    Λ_AB = [kron(Hmat(Λ_A[5+x,:]),I(Nc+1)) for x=0:11]
    ρ_A  = alice_part(α)
    λ_AB = [real(ρ_A·Hmat(Λ_A[5+x,:])) for x=0:11]

    return Λ_AB, λ_AB
end