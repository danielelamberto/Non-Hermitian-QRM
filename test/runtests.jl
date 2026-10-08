# Checks of the package against exactly known cases: the mock Liouvillian (closed-form spectrum, planted singularities)
# and the Lindblad boson dimer (modes from the characteristic polynomial disc). About a minute on one thread.
using Test
using NonHermitianQRM
using NonHermitianQRM: Circle, SVector
using QuantumToolbox, LinearAlgebra, SparseArrays, Accessors

mock = MockLiouvillian()
mock_at(prm_P) = setproperties(mock, (x=prm_P[1], y=prm_P[2]))
A_mock = AffineLiouvillian(prm_P -> mock_matrix(mock_at(prm_P)), 2)
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

# the two eigenpairs of the mock's block-`block` dimer at prm_P (from a dense diagonalisation)
function dimer_pairs(prm_P, block)
    F = eigen(Matrix(A_mock(prm_P)))
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
        for sector ∈ (:coherences, :within), prm_P ∈ ([0.2, 0.7], [0.55, 0.31])
            @test spectrum_distance(eigvals(Matrix(mock_matrix(mock_at(prm_P); sector))),
                                    mock_spectrum(mock_at(prm_P); sector)) < 1e-12
        end
        @test count(p -> p.kind == :EP2, pts) == 4 && count(p -> p.kind == :DP, pts) == 1
    end

    @testset "affine Liouvillian" begin
        prm_P = P2(0.37, 0.81)
        @test norm(A_mock(prm_P) - mock_matrix(mock_at(prm_P))) < 1e-12
        @test norm(derivative(A_mock, P2(1, 0)) - (mock_matrix(mock_at(P2(1, 0))) - mock_matrix(mock_at(P2(0, 0))))) < 1e-12
        @test_throws ArgumentError AffineLiouvillian(prm_P -> sparse(Diagonal([1.0, prm_P[1]^2])), 1; h=0.1)
        # weak curvature: invisible at the scale h of the default check, caught by check_at in the domain
        weak(prm_P) = mock_matrix(mock_at(prm_P)) + 1e-8*prm_P[1]^2*mock_matrix(mock_at([1.0, 0.0]))
        @test AffineLiouvillian(weak, 2) isa AffineLiouvillian
        @test_throws ArgumentError AffineLiouvillian(weak, 2; check_at=[1.0, 0.0])
        @test AffineLiouvillian(prm_P -> sparse(2.0I, 3, 3), 2)(P2(0.3, 0.4)) == 2I(3)    # constant family: no 0/0
        # work matrices share the pattern of the family; a matrix on another pattern is refused
        W = work_matrix(A_mock)
        @test W.colptr === A_mock.L0.colptr && evaluate!(W, A_mock, prm_P) == A_mock(prm_P)
        @test derivative!(W, A_mock, prm_P, P2(1, 0)) == derivative(A_mock, P2(1, 0))
        @test_throws ArgumentError evaluate!(sparse(complex(1.0)*I, size(A_mock.L0)...), A_mock, prm_P)
    end

    @testset "paths (2D and 3D)" begin
        @test position(Circle(P2(0.3, 0.4), 0.1), 0.25) == P2(0.3, 0.5)      # 2D: counterclockwise from the right
        n, ρ = P3(0.3, -0.2, 1.0), 0.07
        q = Circle(P3(0.1, 0.2, 0.3), ρ, n)
        @test q.e1 × q.e2 ≈ normalize(n)
        @test all(u -> norm(position(q, u) - q.center) ≈ ρ && abs((position(q, u) - q.center) ⋅ n) < 1e-14, 0:0.1:1)
        @test position(q, 0.0) ≈ position(q, 1.0)
        h = 1e-6
        @test all(u -> norm((position(q, u + h) - position(q, u - h))/2h - velocity(q, u)) < 1e-7, 0.05:0.1:0.95)
        e1, e2 = plane_basis(P3(0, 0, 2))
        @test e1 × e2 ≈ P3(0, 0, 1)
        @test_throws ArgumentError Circle(P3(0, 0, 0), 1.0, P3(1, 0, 0), P3(1, 1, 0))
        vs = [P3(0, 0, 0), P3(1, 0, 0), P3(1, 1, 1)]
        tri = Polygon(vs)
        @test [position(tri, u) for u ∈ [0; breakpoints(tri)]] ≈ vs && position(tri, 1.0) ≈ vs[1]
        s = Segment(SVector(0, 0, 0), SVector(1, 2, 3))                              # integer vectors are converted
        @test s isa Segment{3} && position(s, 0.5) == P3(0.5, 1, 1.5) && velocity(s, 0.3) == P3(1, 2, 3)
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
        # tracked eigenvalue = closed form along the loop: exact at the steps (corrector), ~reltol in between (dense
        # output; tight with Vern7); stale LU and fresh factorisations agree
        (λ0, r0), _ = dimer_pairs(position(one_ep, 0.), 2)
        sol, _ = track(A_mock, one_ep, λ0, r0)
        exact_at(u) = mock_spectrum(mock_at(position(one_ep, u)))
        @test maximum(minimum(abs.(y[end] .- exact_at(u))) for (u, y) ∈ zip(sol.t, sol.u)) < 1e-10
        @test maximum(minimum(abs.(sol(u)[end] .- exact_at(u))) for u ∈ 0:0.01:1) < 1e-6
        sol_v, _ = track(A_mock, one_ep, λ0, r0; alg=Vern7(lazy=false), reltol=1e-10, abstol=1e-12)
        @test maximum(minimum(abs.(sol_v(u)[end] .- exact_at(u))) for u ∈ 0:0.01:1) < 1e-8
        @test abs(sol_v.u[end][end] - sol.u[end][end]) < 1e-10
        sol_r, _ = track(A_mock, one_ep, λ0, r0; gmres=false)
        @test abs(sol_r.u[end][end] - sol.u[end][end]) < 1e-10
        sol_fresh, _ = track(A_mock, one_ep, λ0, r0; stale=false)
        @test abs(sol.u[end][end] - sol_fresh.u[end][end]) < 1e-10
        # 3D: the mock with an inert third coordinate, so that the EP is a line along z. A tilted circle links it iff
        # its projection on (x, y), an ellipse, encloses the EP.
        A3 = AffineLiouvillian(prm_P -> mock_matrix(mock_at(prm_P)), 3)
        pairs_at(q) = dimer_pairs(position(q, 0.)[1:2], 2)
        tilted = Circle(P3(ep.x, ep.y, 0.4), 0.05, P3(0.3, 0.2, 1.0))
        @test swaps(A3, tilted, pairs_at(tilted))
        beside = Circle(P3(ep.x + 0.1, ep.y, 0.4), 0.05, P3(0.3, 0.2, 1.0))
        @test !swaps(A3, beside, pairs_at(beside))
    end

    @testset "reparametrised family (polar coordinates around a mock EP)" begin
        ep = only(filter(p -> p.block == 2 && p.y < 0.7, pts))
        c = P2(ep.x, ep.y)
        polar(ξ) = c + ξ[1]*SVector(cos(ξ[2]), sin(ξ[2]))                        # ξ = (r, θ); generic for ForwardDiff
        R = Reparametrised(A_mock, polar)
        @test norm(R(P2(0.05, 0.3)) - A_mock(polar(P2(0.05, 0.3)))) < 1e-12
        J_polar(ξ) = [cos(ξ[2]) -ξ[1]*sin(ξ[2]); sin(ξ[2]) ξ[1]*cos(ξ[2])]
        @test Reparametrised(A_mock, polar; Jφ=J_polar, check_at=P2(0.05, 0.7)) isa Reparametrised
        @test_throws ArgumentError Reparametrised(A_mock, polar; Jφ=ξ -> 2J_polar(ξ), check_at=P2(0.05, 0.7))
        # a segment in θ at fixed r is the circle around the EP: it swaps the pair, the tracked eigenvalue is the
        # closed form all along (this checks the chain rule), and the hand-written Jacobian gives the same result
        θ_loop = Segment(P2(0.05, 0.0), P2(0.05, 2π))
        pairs = dimer_pairs(polar(P2(0.05, 0.0)), 2)
        @test swaps(R, θ_loop, pairs)
        sol, _ = track(R, θ_loop, pairs[1]...)
        exact(u) = mock_spectrum(mock_at(polar(position(θ_loop, u))))
        @test maximum(minimum(abs.(y[end] .- exact(u))) for (u, y) ∈ zip(sol.t, sol.u)) < 1e-10
        @test maximum(minimum(abs.(sol(u)[end] .- exact(u))) for u ∈ 0:0.01:1) < 1e-6
        sol_J, _ = track(Reparametrised(A_mock, polar; Jφ=J_polar), θ_loop, pairs[1]...)
        @test abs(sol_J.u[end][end] - sol.u[end][end]) < 1e-10
        # a scan of an annulus around the EP flags nothing: no cell of (r, θ) contains it
        cells = tracked_scan(R, range(0.02, 0.08, 4), range(0, 2π, 9), pairs)
        @test !any(flagged, cells) && all(cell -> cell.lost == 0, cells)
    end

    @testset "bordered solves: stale LU with GMRES or refinement" begin
        # B at a point near an EP of the mock, factorised; then solved at a drifted point from the stale factorisation
        ep = only(filter(p -> p.block == 2 && p.y < 0.7, pts))
        (λ, r), _ = dimer_pairs(P2(ep.x + 0.003, ep.y), 2)
        c = NonHermitianQRM.normalisation(r)
        b = [r; 1.0] .+ 0.1                                    # a generic right-hand side
        L0, L1 = A_mock(P2(ep.x + 0.003, ep.y)), A_mock(P2(ep.x + 0.004, ep.y + 0.001))
        exact = NonHermitianQRM.bordered(L1, λ + 1e-3, r, c) \ b
        for (label, S) ∈ (("GMRES", StaleLU()), ("refinement", StaleLU(method=:richardson)),
                          ("GMRES, maxiter 1: refactorises", StaleLU(maxiter=1)),
                          ("GMRES, early 0: refactorises after", StaleLU(early=0)))
            NonHermitianQRM.solve_bordered!(S, L0, λ, r, c, b)  # the first solve factorises
            x = copy(NonHermitianQRM.solve_bordered!(S, L1, λ + 1e-3, r, c, b))
            @test norm(x - exact) ≤ 1e-10*norm(exact)
            label == "GMRES" && @test S.factorisations == 1
            startswith(label, "GMRES, ") && @test S.factorisations == 2
        end
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

    @testset "localisation of the EPs of a scan (mock, JC)" begin
        # coarse scan of the mock, then localise_eps: the four EP2 in boxes of 1e-9 (the close pair, in one coarse
        # cell with a winding ±2, after refinement), the DP classified as such
        F = eigen(Matrix(A_mock(P2(0, 0))))
        start = [(F.values[k], F.vectors[:, k]) for k ∈ findall(λ -> imag(λ) > 1e-8, F.values)]
        cells = tracked_scan(A_mock, range(0, 1, 13), range(0, 1, 13), start)
        @test all(c -> isempty(c.swaps) || all(w -> abs(w) == 1, c.swap_windings), cells)
        loc = localise_eps(A_mock, cells; tol=1e-9)
        inside(r, p) = r[1] ≤ p.x ≤ r[2] && r[3] ≤ p.y ≤ r[4]
        ep_pts = filter(p -> p.kind == :EP2, pts)
        @test length(loc.eps) == length(ep_pts) && isempty(loc.unresolved) && isempty(loc.lost)
        for p ∈ ep_pts
            k = findfirst(e -> inside(e.rect, p), loc.eps)
            @test k !== nothing && max(loc.eps[k].rect[2] - loc.eps[k].rect[1], loc.eps[k].rect[4] - loc.eps[k].rect[3]) ≤ 1e-9
            @test k !== nothing && minimum(abs.([loc.eps[k].λ, conj(loc.eps[k].λ)] .- p.λ)) < 1e-10
        end
        dp = only(filter(p -> p.kind == :DP, pts))
        @test length(loc.dps) == 1 && inside(only(loc.dps).rect, dp)
        # JC tower (sector k = 1): every EP_n bracketed once per swapped pair (3 or 4), at the closed form
        Nc = 5
        â, σ̂ = qrm_operators(Nc)
        jc = QRM(ωa=1.0, ωb=0.97, g=0.02, γa=0.1, γb=0.0, approx=[RWA_env, RWA_coupling])
        idx = excitation_sector(Nc, 1)
        A = AffineLiouvillian(P -> liouvillian(Lindbladian(â, σ̂, setproperties(jc, (g=P[1], ωb=P[2])))...).data[idx, idx], 2)
        gs, ωbs = range(0.0101, 0.0303, 9), range(0.9713, 1.0291, 9)
        F = eigen(Matrix(A(P2(gs[1], ωbs[1]))))
        loc = localise_eps(A, tracked_scan(A, gs, ωbs, collect(zip(F.values, eachcol(F.vectors)))); tol=1e-8)
        @test isempty(loc.dps) && isempty(loc.unresolved)
        for n ∈ 1:Nc - 1
            ωb, g, _ = jc_EP(jc, n)
            @test count(e -> e.rect[1] ≤ g ≤ e.rect[2] && e.rect[3] ≤ ωb ≤ e.rect[4], loc.eps) == (n ∈ (1, Nc - 1) ? 3 : 4)
        end
        @test length(loc.eps) == 3 + 4 + 4 + 3
    end

    @testset "bisection (mock)" begin
        ep = only(filter(p -> p.block == 2 && p.y < 0.7, pts))
        targets(prm_P) = first.(dimer_pairs(prm_P, 2))
        rect, _, _ = ep_bisect(A_mock, (ep.x - 0.05, ep.x + 0.04, ep.y - 0.06, ep.y + 0.05), targets; depth=10)
        @test rect[1] ≤ ep.x ≤ rect[2] && rect[3] ≤ ep.y ≤ rect[4]
        @test max(rect[2] - rect[1], rect[4] - rect[3]) < 0.01
        # greedy bracketing (regula falsi on D, 3 × 3 cuts): a certified box of 1e-9 in a few iterations, and the fit of
        # D on the sides of a small box around the EP points at it
        rect, hist, _ = ep_bracket(A_mock, (ep.x - 0.05, ep.x + 0.04, ep.y - 0.06, ep.y + 0.05), targets; tol=1e-9)
        @test rect[1] ≤ ep.x ≤ rect[2] && rect[3] ≤ ep.y ≤ rect[4]
        @test max(rect[2] - rect[1], rect[4] - rect[3]) ≤ 1e-9 && length(hist) ≤ 9
        small = (ep.x - 1e-3, ep.x + 1.2e-3, ep.y - 0.9e-3, ep.y + 1.1e-3)
        c = (P2(small[1], small[3]), P2(small[2], small[3]), P2(small[2], small[4]), P2(small[1], small[4]))
        lines = [track_line(A_mock, c[1], c[2], targets(c[1])), track_line(A_mock, c[2], c[3], targets(c[2])),
                 track_line(A_mock, c[4], c[3], targets(c[4])), track_line(A_mock, c[1], c[4], targets(c[1]))]
        zero, err = disc_fit(A_mock, lines, small)
        @test norm(zero - P2(ep.x, ep.y)) < max(3err, 1e-12) && err < 1e-4
        # the corrector of an EP-line tracker: bisection in a tilted plane of the 3D mock (inert z: the EP is a
        # vertical line), on the slice of the family, in the plane's own coordinates (s, t)
        A3 = AffineLiouvillian(prm_P -> mock_matrix(mock_at(prm_P)), 3)
        o = P3(ep.x + 0.01, ep.y - 0.005, 0.4)
        e1, e2 = plane_basis(P3(0.3, 0.2, 1.0))
        A_sl = slice(A3, o, e1, e2)
        @test norm(A_sl(P2(0.03, -0.02)) - A3(o + 0.03*e1 - 0.02*e2)) < 1e-12
        to_3d(st) = o + st[1]*e1 + st[2]*e2
        st_ep = [e1[1] e2[1]; e1[2] e2[2]] \ [ep.x - o[1], ep.y - o[2]]     # where the EP line crosses the plane
        targets_sl(st) = first.(dimer_pairs(to_3d(st)[1:2], 2))
        rect, _, _ = ep_bisect(A_sl, (-0.05, 0.04, -0.045, 0.05), targets_sl; depth=10)
        @test rect[1] ≤ st_ep[1] ≤ rect[2] && rect[3] ≤ st_ep[2] ≤ rect[4]
        @test max(rect[2] - rect[1], rect[4] - rect[3]) < 0.01
    end

    @testset "EP-line tracker (3D)" begin
        # synthetic family with a curved line of EP2s: a 2×2 block [λc + μ + w  1; c  λc + μ - w] with w = x + iy,
        # c = z0 - z - iη, μ = 0.1(x - iz), plus spectators far away. D = 4(w² + c): EPs on xy = η/2, z = z0 + x² - y²,
        # with λ = λc + μ there. Not a Liouvillian (no conjugate pairs): the tracker only needs the family interface.
        λc, z0, η = -0.2 - 0.3im, 0.5, 0.02
        spect = ComplexF64[-1.3-0.4im, -0.9+0.8im, -1.6, -1.1-1.4im, -2.0+0.3im, -0.7+1.5im, -1.8-0.9im, -2.4]
        n = 2 + length(spect)
        term(entries) = sparse(first.(first.(entries)), last.(first.(entries)), ComplexF64.(last.(entries)), n, n)
        A = AffineLiouvillian(term([(1, 1) => λc, (2, 2) => λc, (1, 2) => 1, (2, 1) => z0 - im*η,
                                    [(k + 2, k + 2) => s for (k, s) ∈ enumerate(spect)]...]),
                              [term([(1, 1) => 1.1, (2, 2) => -0.9]), term([(1, 1) => 1im, (2, 2) => -1im]),
                               term([(1, 1) => -0.1im, (2, 2) => -0.1im, (2, 1) => -1])])
        λ_EP(P) = λc + 0.1*P[1] - 0.1im*P[3]
        P_ex = P3(0.2, η/0.4, z0 + 0.2^2 - (η/0.4)^2)
        P0 = P_ex + P3(0.003, -0.002, 0.001)                               # an approximate seed
        bounds = (P3(-0.5, -0.5, 0), P3(0.5, 0.5, 1))
        for direction ∈ (1, -1)
            line = track_ep_line(A, P0, λ_EP(P0); h=0.02, direction, nsteps=40, bounds)
            @test line.status == :left_bounds && length(line.points) ≥ 5
            @test all(P -> abs(2P[1]*P[2] - η) < 1e-12 && abs(P[3] - z0 - P[1]^2 + P[2]^2) < 1e-12, line.points)
            @test maximum(abs.(line.λs .- λ_EP.(line.points))) < 1e-9
            # tangents: along the exact one, d(x, y, z)/dx = (1, -y/x, 2x + 2y²/x), oriented by direction
            exact_t(P) = normalize(P3(1, -P[2]/P[1], 2P[1] + 2P[2]^2/P[1]))
            @test all(((P, t),) -> abs(t ⋅ exact_t(P)) > 1 - 1e-8, zip(line.points, line.tangents))
            @test sign(line.tangents[1] ⋅ exact_t(line.points[1])) == direction
        end
        # the bisection corrector alone (no Newton-first) follows the same line
        line_nf = track_ep_line(A, P0, λ_EP(P0); h=0.02, nsteps=5, newton_first=false)
        line_n = track_ep_line(A, P0, λ_EP(P0); h=0.02, nsteps=5)
        @test maximum(norm.(line_nf.points .- line_n.points)) < 1e-12
        # several lines, half-lines as parallel tasks: the same as tracked one by one; joined through the seed
        seeds = [(P0, λ_EP(P0)), (P_ex + P3(-0.002, 0.001, 0.002), λ_EP(P_ex))]
        par = track_ep_lines(A, seeds; h=0.02, nsteps=5)
        for (k, (P_s, σ_s)) ∈ enumerate(seeds), (half, direction) ∈ ((:forward, 1), (:backward, -1))
            @test maximum(norm.(getfield(par[k], half).points .-
                                track_ep_line(A, P_s, σ_s; h=0.02, nsteps=5, direction).points)) < 1e-12
        end
        joined = join_halves(par[1]...)
        @test length(joined.points) == length(par[1].forward.points) + length(par[1].backward.points) - 1
        @test norm(joined.points[length(par[1].backward.points)] - par[1].forward.points[1]) < 1e-12   # the seed
        @test all(k -> joined.tangents[k] ⋅ (joined.points[k + 1] - joined.points[k]) > 0, 1:length(joined.points) - 1)
        # the pair of each check loop is carried from the previous one; without carrying, the same line
        line_c = track_ep_line(A, P0, λ_EP(P0); h=0.02, nsteps=6)
        line_s = track_ep_line(A, P0, λ_EP(P0); h=0.02, nsteps=6, continue_pair=false)
        @test line_c.counts.carried == line_c.counts.newton - 1 && line_s.counts.carried == 0
        @test maximum(norm.(line_c.points .- line_s.points)) < 1e-12
        # check_loop at the exact EP: with a carried pair, without one, and with a carry that cannot be continued
        # (the same eigenpair twice): it falls back to the pair's model; every loop swaps
        plane = NonHermitianQRM.NormalPlane(P_ex, P3(1, -P_ex[2]/P_ex[1], 2P_ex[1] + 2P_ex[2]^2/P_ex[1]))
        newton = NonHermitianQRM.newton_in_plane(slice(A, plane), λ_EP(P_ex); reach=0.01, fd=1e-6, xtol=1e-12)
        @test newton.converged && norm(newton.ξ) < 1e-10
        loop0, pairs0, carried0 = NonHermitianQRM.check_loop(A, plane, newton.ξ, 1e-3, nothing, newton.model)
        start0 = position(loop0, 0.0)
        shifted = NonHermitianQRM.NormalPlane(P_ex + 1e-3*plane.t̂, plane.t̂)       # a nearby plane, as at the next step
        loop1, pairs1, carried1 = NonHermitianQRM.check_loop(A, shifted, P2(0, 0), 1e-3, (x=start0, pairs=pairs0),
                                                             newton.model)
        loop2, pairs2, carried2 = NonHermitianQRM.check_loop(A, plane, newton.ξ, 1e-3,
                                                             (x=start0, pairs=[pairs0[1], pairs0[1]]), newton.model)
        @test !carried0 && carried1 && !carried2
        @test all(((q, ps),) -> pair_swaps(A, q, ps), ((loop0, pairs0), (loop1, pairs1), (loop2, pairs2)))
        # slice of a non-affine family (generic method): the same matrices as the affine slice
        e1, e2 = plane_basis(P3(0.3, 0.2, 1.0))
        R = Reparametrised(A, ξ -> ξ)
        @test norm(slice(R, P_ex, e1, e2)(P2(0.03, -0.02)) - slice(A, P_ex, e1, e2)(P2(0.03, -0.02))) < 1e-14
        # the 3D mock (inert z): a straight vertical EP line, eigenvalue -γ0 - iω0 (and its conjugate pair, far away)
        ep = only(filter(p -> p.block == 2 && p.y < 0.7, pts))
        A3 = AffineLiouvillian(prm_P -> mock_matrix(mock_at(prm_P)), 3)
        line = track_ep_line(A3, P3(ep.x + 0.002, ep.y - 0.001, 0.3), ep.λ; h=0.05, nsteps=6)
        @test line.status == :max_steps
        @test all(P -> norm(P[1:2] - SVector(ep.x, ep.y)) < 1e-9, line.points) && abs(line.points[end][3] - 0.3) > 0.2
        @test maximum(abs.(line.λs .- ep.λ)) < 1e-9
        # the corrector's bracketing path on a Liouvillian (no Newton-first): lands on the line
        c = ep_correct(A3, P3(ep.x + 0.002, ep.y - 0.001, 0.3), P3(0, 0, 1), ep.λ; w=0.01, newton_first=false)
        @test c.path == :bracket && norm(c.P[1:2] - SVector(ep.x, ep.y)) < 1e-9 && abs(c.λ - ep.λ) < 1e-9
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
        jc_at(prm_P) = setproperties(jc, (g=prm_P[1], ωb=prm_P[2]))
        A = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(â, σ̂, jc_at(prm_P))...).data[idx, idx], 2)
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
