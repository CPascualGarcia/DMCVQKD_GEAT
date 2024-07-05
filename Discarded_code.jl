
#########################################################
#################### IN PROGRESS ########################
#########################################################

function Variance_f(Dvars::Array{T},pK::T) where {T<:AbstractFloat}
    # Note that the actual optimization of the variance includes
    # the pre-factor pK. Here we remove it for convenience, and 
    # add it in the main text

    Variance_f = GenericModel{T}()
    set_optimizer(Variance_f, Hypatia.Optimizer{T})

    @variable(Variance_f,prob[1:4,1:6])
    @constraint(Variance_f,prob.>=0)
    @constraint(Variance_f,sum(prob[1,:])==T(1/4))
    @constraint(Variance_f,sum(prob[2,:])==T(1/4))
    @constraint(Variance_f,sum(prob[3,:])==T(1/4))
    @constraint(Variance_f,sum(prob[4,:])==T(1/4))

    Max   = maximum(Dvars)
    "This below is wrong. One has to calculate g0, include it as a constant
    and then calculate the variance"
    coeff = [(Max - d) for d in Dvars]

    # I THINK THIS BELOW IS WRONG. DOUBLE CHECK
    Objf = (((coeff.^2)·prob'[:])/(1-pK) 
            - (coeff·prob'[:])^2)

    @objective(Variance_f,Max,Objf)
    
    optimize!(Variance_f)

    return value(Objf)*pK^2
end

