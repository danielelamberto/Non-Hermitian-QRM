# Checks of the package against exactly known cases: the mock Liouvillian (closed-form spectrum, planted singularities)
# and the Lindblad boson dimer (modes from the characteristic polynomial disc). About a minute on one thread.
using Test
using NonHermitianQRM
using NonHermitianQRM: Circle
using QuantumToolbox, LinearAlgebra, SparseArrays, Accessors

mock = MockLiouvillian()
mock_at(δ) = setproperties(mock, (x=δ[1], y=δ[2]))
A_mock = AffineLiouvillian(δ -> mock_matrix(mock_at(δ)), 2)
pts, plines = mock_singularities(mock)

# greedy matching distance between two spectra
function spectrum_distance(a, b)
    b, e = copy(b), 0.0
    for v ∈ a
        k = argmin(abs.(b .- v))
        e = max(e, abs(b[k] - v))
        deleteat!(b, k)
    end
    return e
end

# the two eigenpairs of the mock's block-`block` dimer at δ (from a dense diagonalisation)
function dimer_pairs(δ, block)
    F = eigen(Matrix(A_mock(δ)))
    λ0 = -mock.blocks[block].γ0 - im*mock.blocks[block].ω0
    return [(F.values[k], F.vectors[:, k]) for k ∈ partialsortperm(abs.(F.values .- λ0), 1:2)]
end

# does tracking both eigenpairs once around the loop q swap them?
function swaps(A, q, pairs)
    ends = [track(A, q, λ0, r0)[1].u[end][end] for (λ0, r0) ∈ pairs]
    return argmin(abs.(first.(pairs) .- ends[1])) == 2
end

@testset "NonHermitianQRM" begin

    @testset "mock Liouvillian" begin
        for sector ∈ (:coherences, :within), δ ∈ ([0.2, 0.7], [0.55, 0.31])
            @test spectrum_distance(eigvals(Matrix(mock_matrix(mock_at(δ); sector))), mock_spectrum(mock_at(δ); sector)) < 1e-12
        end
        @test count(p -> p.kind == :EP2, pts) == 4 && count(p -> p.kind == :DP, pts) == 1
    end

    @testset "affine Liouvillian" begin
        δ = P2(0.37, 0.81)
        @test norm(A_mock(δ) - mock_matrix(mock_at(δ))) < 1e-12
        @test norm(derivative(A_mock, P2(1, 0)) - (mock_matrix(mock_at(P2(1, 0))) - mock_matrix(mock_at(P2(0, 0))))) < 1e-12
        @test_throws ArgumentError AffineLiouvillian(δ -> sparse(Diagonal([1.0, δ[1]^2])), 1; h=0.1)
    end

    @testset "tracking around loops (mock)" begin
        ep = only(filter(p -> p.block == 2 && p.y < 0.7, pts))          # one EP2 of the wide pair (block 2)
        dp = only(filter(p -> p.kind == :DP, pts))
        one_ep = Circle(P2(ep.x, ep.y), 0.05)
        @test swaps(A_mock, one_ep, dimer_pairs(position(one_ep, 0.), 2))
        both = Circle(P2(ep.x, 0.7031), 0.15)
        @test !swaps(A_mock, both, dimer_pairs(position(both, 0.), 2))
        around_dp = Circle(P2(dp.x, dp.y), 0.05)
        @test !swaps(A_mock, around_dp, dimer_pairs(position(around_dp, 0.), 3))
        # tracked eigenvalue = closed form along the loop; stale LU and fresh factorisations agree
        (λ0, r0), _ = dimer_pairs(position(one_ep, 0.), 2)
        sol, _ = track(A_mock, one_ep, λ0, r0)
        @test maximum(minimum(abs.(sol(u)[end] .- mock_spectrum(mock_at(position(one_ep, u))))) for u ∈ 0:0.01:1) < 1e-8
        sol_fresh, _ = track(A_mock, one_ep, λ0, r0; stale=false)
        @test abs(sol.u[end][end] - sol_fresh.u[end][end]) < 1e-10
    end

    @testset "tracking scan (mock)" begin
        F = eigen(Matrix(A_mock(P2(0, 0))))
        start = [(F.values[k], F.vectors[:, k]) for k ∈ findall(λ -> imag(λ) > 1e-8, F.values)]
        cells = tracked_scan(A_mock, range(0, 1, 13), range(0, 1, 13), start)
        leaves = refine_tracked(A_mock, cells, 9)
        inside(c, p) = c.rect[1] ≤ p.x ≤ c.rect[2] && c.rect[3] ≤ p.y ≤ c.rect[4]
        @test all(c -> c.lost == 0, cells)
        @test length(leaves) == length(pts)                              # one leaf per planted point, nothing else
        for p ∈ pts
            k = findfirst(c -> inside(c, p), leaves)
            @test k !== nothing
            k === nothing && continue
            @test p.kind == :EP2 ? !isempty(leaves[k].swaps) :
                                   (isempty(leaves[k].swaps) && abs(only(leaves[k].windings)[3]) == 2)
        end
    end

    @testset "bisection (mock)" begin
        ep = only(filter(p -> p.block == 2 && p.y < 0.7, pts))
        targets(δ) = first.(dimer_pairs(δ, 2))
        rect, _, _ = ep_bisect(A_mock, (ep.x - 0.05, ep.x + 0.04, ep.y - 0.06, ep.y + 0.05), targets; depth=10)
        @test rect[1] ≤ ep.x ≤ rect[2] && rect[3] ≤ ep.y ≤ rect[4]
        @test max(rect[2] - rect[1], rect[4] - rect[3]) < 0.01
    end

    @testset "boson dimer: Lindblad modes = roots of disc" begin
        n = 6                                     # the counter-rotating coupling at g = 0.2 needs n > 4
        â, b̂ = destroy(n) ⊗ eye(n), eye(n) ⊗ destroy(n)
        positive(ω) = filter(w -> real(w) > 0, ω)
        for approx ∈ ([RWA_env], [RWA_env, RWA_coupling]), g ∈ (0.0, 0.03, 0.2)
            mdl = BosonDimer(ωa=1.0, ωb=1.1, γa=0.1, γb=0.05, g=g, approx=approx)
            @test spectrum_distance(positive(modes(liouvillian(Lindbladian(â, b̂, mdl)...), [â, b̂])), positive(roots_sorted(mdl))) < 1e-8
        end
        g_EP, ω_EP = EP(BosonDimer(ωa=1.0, ωb=1.0, γa=0.3, γb=0.1, approx=[RWA_env]))
        ω = roots_sorted(BosonDimer(ωa=1.0, ωb=1.0, γa=0.3, γb=0.1, g=g_EP, approx=[RWA_env]))
        @test minimum(abs(ω[i] - ω[j]) for i ∈ eachindex(ω) for j ∈ i+1:length(ω)) < 1e-6
    end

    @testset "Jaynes–Cummings with RWA_env: closed form and the tower of EPs" begin
        Nc = 5
        â, σ̂ = qrm_operators(Nc)
        jc = QRM(ωa=1.0, ωb=0.97, g=0.02, γa=0.1, γb=0.0, approx=[RWA_env, RWA_coupling])
        L = liouvillian(Lindbladian(â, σ̂, jc)...).data
        @test spectrum_distance(eigvals(Matrix(L)), jc_spectrum(jc, Nc)) < 1e-12
        idx = excitation_sector(Nc, 1)
        @test norm(L[setdiff(1:size(L, 1), idx), idx]) == 0                # L conserves the excitation-number difference
        @test spectrum_distance(eigvals(Matrix(L[idx, idx])), jc_spectrum(jc, Nc; k=1)) < 1e-12
        # scan of the plane (g, ωb) in the sector k = 1: one leaf per EP_n (n = 1…4), with 3 or 4 swaps, nothing else
        A = AffineLiouvillian(δ -> liouvillian(Lindbladian(â, σ̂, setproperties(jc, (g=δ[1], ωb=δ[2])))...).data[idx, idx], 2)
        gs, ωbs = range(0.0101, 0.0303, 9), range(0.9713, 1.0291, 9)
        F = eigen(Matrix(A(P2(gs[1], ωbs[1]))))
        leaves = refine_tracked(A, tracked_scan(A, gs, ωbs, collect(zip(F.values, eachcol(F.vectors)))), 8)
        @test length(leaves) == Nc - 1
        for n ∈ 1:Nc - 1
            ωb, g, _ = jc_EP(jc, n)
            k = findfirst(c -> c.rect[1] ≤ g ≤ c.rect[2] && c.rect[3] ≤ ωb ≤ c.rect[4], leaves)
            @test k !== nothing && length(leaves[k].swaps) == (n ∈ (1, Nc - 1) ? 3 : 4) && isempty(leaves[k].windings)
        end
    end

end
