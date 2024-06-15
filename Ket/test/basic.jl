@testset "Basic" begin
    @testset "Kets" begin
        @test isa(ket(1, 3), Vector{ComplexF64})
        @test isa(proj(1, 3), Hermitian{ComplexF64})
        for R in [Int64, Float64, Double64, Float128, BigFloat]
            ψ = ket(R, 2, 3)
            P = proj(R, 2, 3)
            @test ψ == [0, 1, 0]
            @test isa(ψ, Vector{R})
            @test isa(P, Hermitian{R})
            @test P == ketbra(ψ)
            T = Complex{R}
            ψ = ket(T, 2, 3)
            P = proj(T, 2, 3)
            @test ψ == [0, 1, 0]
            @test isa(ψ, Vector{T})
            @test isa(P, Hermitian{T})
            @test P == ketbra(ψ)
        end
    end
    @testset "Shift and clock" begin
        @test isa(shift(4), Matrix{ComplexF64})
        @test clock(4) == Diagonal([1, im, -1, -im])
        @test isa(clock(4), Diagonal{ComplexF64})
        for R in [Float64, Double64, Float128, BigFloat]
            T = Complex{R}
            @test shift(T, 3) == [0 0 1; 1 0 0; 0 1 0]
            @test clock(T, 3) ≈ Diagonal([1, exp(2 * T(π) * im / 3), exp(-2 * T(π) * im / 3)])
            @test shift(T, 3, 2) == shift(T, 3)^2
            @test clock(T, 3, 2) ≈ clock(T, 3)^2
        end
    end
    @testset "Cleanup" begin
        for R in [Float64, Double64, Float128, BigFloat]
            a = zeros(R, 2, 2)
            a[1] = 0.5 * Ket._tol(R)
            a[4] = 1
            b = Hermitian(copy(a))
            c = UpperTriangular(copy(a))
            d = Diagonal(copy(a))
            cleanup!(a)
            cleanup!(b)
            cleanup!(c)
            cleanup!(d)
            @test a == [0 0; 0 1]
            @test b == [0 0; 0 1]
            @test c == [0 0; 0 1]
            @test d == [0 0; 0 1]
            T = Complex{R}
            a = zeros(T, 2, 2)
            a[1] = 0.5 * Ket._tol(T) + im
            a[3] = 1 + 0.5 * Ket._tol(T) * im
            a[4] = 1
            b = Hermitian(copy(a))
            c = UpperTriangular(copy(a))
            d = Diagonal(copy(a))
            cleanup!(a)
            cleanup!(b)
            cleanup!(c)
            cleanup!(d)
            @test a == [im 1; 0 1]
            @test b == [0 1; 1 1]
            @test c == [im 1; 0 1]
            @test d == [im 0; 0 1]
        end
    end
end
