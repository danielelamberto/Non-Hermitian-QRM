# Integration strategies for `track`: ODE solvers, tolerances, linear solves (stale LU refinement vs GMRES) and a
# hand-written predictor–corrector continuation, on representative workloads; scaling with the Liouvillian's size.
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/tracking.jl).
#
# Outcome (2026-10-08): the default of `track` is now Tsit5 at reltol 1e-6 with GMRES preconditioned by the stale LU,
# instead of Vern7 at 1e-10 with iterative refinement. The corrector puts every step back on an exact eigenpair, so the
# integrator only has to stay in Newton's basin: a tight tolerance was wasted. Near an EP the bordered matrix is
# ill-conditioned, the refinement from a stale LU diverges and refactorised every few solves; GMRES on the same
# preconditioner converges, with ~2 factorisations per tracking. Lines of bisection/bracketing (`track_line`) keep an
# accurate dense output (Vern7 at 1e-10, with GMRES), since they are read between steps.
#
# On the 3D tracker (ep_lines.jl, full QRM: 19 lines, 465 points, h = 0.002), with and without carrying the check
# loop's pair from one step to the next (`continue_pair`, now the default; 427 of 427 non-seed loops carried):
#   old default (Vern7 1e-10, refinement)   112.5 s / 117.4 s with carrying
#   new default (Tsit5 1e-6, GMRES)          24.9 s /  27.1 s
#   EulerNewton(α=0.1, quadratic), GMRES     19.5 s /  22.3 s
# No corrector failure in any of them, every point an EP by dense diagonalisation (pair gap ≤ 9e-8).

#%% Packages

using Revise
using NonHermitianQRM
using NonHermitianQRM: Circle
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, SparseArrays, Accessors, Printf
using OrdinaryDiffEqLowOrderRK: BS3                       # notebook environment only (not a package dependency)


#%% Strategies and workloads

# Strategies: keyword arguments of `track`. "g": GMRES (default now), "r": iterative refinement (`gmres=false`).
strategies = [
    "Vern7 1e-10 r (old default)" => (alg=Vern7(lazy=false), reltol=1e-10, abstol=1e-12, gmres=false),
    "Vern7 1e-6 r"                => (alg=Vern7(lazy=false), reltol=1e-6, abstol=1e-8, gmres=false),
    "Tsit5 1e-6 r"                => (alg=Tsit5(), reltol=1e-6, abstol=1e-8, gmres=false),
    "Tsit5 1e-4 r"                => (alg=Tsit5(), reltol=1e-4, abstol=1e-6, gmres=false),
    "BS3 1e-4 r"                  => (alg=BS3(), reltol=1e-4, abstol=1e-6, gmres=false),
    "EulerNewton2 r"              => (alg=EulerNewton(α=0.1, quadratic=true), gmres=false),
    "Vern7 1e-10 g"               => (alg=Vern7(lazy=false), reltol=1e-10, abstol=1e-12),
    "Tsit5 1e-6 g (new default)"  => (;),
    "EulerNewton2 g"              => (alg=EulerNewton(α=0.1, quadratic=true),),
]

# Families: the mock (18 states), the full QRM's odd sector (Nc = 10: 200 states), the Lindblad boson dimer (625)
mock = MockLiouvillian()
mock_at(P) = setproperties(mock, (x=P[1], y=P[2]))
A_mock = AffineLiouvillian(P -> mock_matrix(mock_at(P)), 2)
pts, _ = mock_singularities(mock)
function dimer_pairs(P, block)
    F = eigen(Matrix(A_mock(P)))
    λ0 = -mock.blocks[block].γ0 - im*mock.blocks[block].ω0
    return [(F.values[k], F.vectors[:, k]) for k ∈ partialsortperm(abs.(F.values .- λ0), 1:2)]
end
function rabi_family(Nc; ω0=1.0, γ0=0.05)
    â, σ̂ = qrm_operators(Nc)
    odd = sort(reduce(vcat, excitation_sector(Nc, k) for k ∈ -Nc:Nc if isodd(k)))
    at(P) = setproperties(QRM(approx=[RWA_env]), (ωa=ω0 + P[1], ωb=ω0 - P[1], γa=γ0 + P[2], γb=γ0 - P[2], g=P[3]))
    return AffineLiouvillian(P -> liouvillian(Lindbladian(â, σ̂, at(P))...).data[odd, odd], 3; h=0.01)
end
# an EP of the full QRM near `P0` (pair near λ0), in the plane normal to its line: (slice, EP in the slice, λ_EP)
function rabi_ep(A, P0, λ0)
    ev = eigvals(Matrix(A(P0)))
    t, σ = ep_tangent(A, P0, ev[argmin(abs.(ev .- λ0))]; fd=1e-6)
    A_sl = slice(A, P0, plane_basis(t)...)
    st, λm, _ = NonHermitianQRM.ep_newton(A_sl, P2(0, 0), σ; fd=1e-6, xtol=1e-14)
    return A_sl, st, λm
end
A_rabi = rabi_family(10)
rabi_slice, rabi_st, rabi_λ = rabi_ep(A_rabi, P3(9.53e-5, 0.0277, 0.008), -0.2443 - 0.9921im)        # an EP_3
nf_d = 5
âd, b̂d = destroy(nf_d) ⊗ eye(nf_d), eye(nf_d) ⊗ destroy(nf_d)
dimer_at(P) = BosonDimer(ωa=1 + P[1], ωb=1 - P[1], γa=0.1 + P[2], γb=0.1 - P[2], g=0.01, approx=[RWA_env])
A_dimer = AffineLiouvillian(P -> liouvillian(Lindbladian(âd, b̂d, dimer_at(P))...).data, 2)

# track every pair once along q: (solutions, work, wall time)
function run_pairs(A, q, pairs; kw...)
    t = @elapsed res = [track(A, q, λ0, r0; kw...) for (λ0, r0) ∈ pairs]
    sols, diags = first.(res), last.(res)
    work = (nf=sum(s.stats.nf for s ∈ sols), steps=sum(s.stats.naccept for s ∈ sols),
            lu=sum(d.factorisations for d ∈ diags), ok=all(tracking_succeeded, sols))
    return sols, work, t
end
swapped(sols, pairs) = argmin(abs.(first.(pairs) .- sols[1].u[end][end])) == 2
dense_error(sols, spec, q, us) = maximum(maximum(minimum(abs.(s(u)[end] .- spec(position(q, u)))) for u ∈ us) for s ∈ sols)
no_work = (nf=0, steps=0, lu=0, ok=true)

# Workloads: each returns (time, work, error of the dense output, correct)
workloads = [
    "mock loop" => kw -> begin                                 # around a mock EP2, r = 0.05
        ep = only(filter(p -> p.block == 2 && p.y < 0.7, pts))
        q = Circle(P2(ep.x, ep.y), 0.05)
        pairs = dimer_pairs(position(q, 0.0), 2)
        sols, w, t = run_pairs(A_mock, q, pairs; kw...)
        t, w, dense_error(sols, P -> mock_spectrum(mock_at(P)), q, range(0, 1, 201)), w.ok && swapped(sols, pairs)
    end,
    "mock weak EP" => kw -> begin                              # r = 2e-4 around one of two EPs 1e-3 apart
        ep = first(filter(p -> p.block == 1, pts))
        q = Circle(P2(ep.x, ep.y), 2e-4)
        pairs = dimer_pairs(position(q, 0.0), 1)
        sols, w, t = run_pairs(A_mock, q, pairs; kw...)
        t, w, dense_error(sols, P -> mock_spectrum(mock_at(P)), q, range(0, 1, 201)), w.ok && swapped(sols, pairs)
    end,
    "mock segment" => kw -> begin                              # all 18 eigenvalues across the domain (crosses a real EP line)
        q = Segment(P2(0.05, 0.1), P2(0.95, 0.9))
        F = eigen(Matrix(A_mock(position(q, 0.0))))
        sols, w, t = run_pairs(A_mock, q, [(F.values[k], F.vectors[:, k]) for k ∈ eachindex(F.values)]; kw...)
        t, w, dense_error(sols, P -> mock_spectrum(mock_at(P)), q, range(0, 1, 101)), w.ok
    end,
    "QRM check loop" => kw -> begin                            # the 3D tracker's check loop, r = w/16 = 6.25e-5
        q = Circle(rabi_st, 1e-3/16)
        pairs = pair_near(rabi_slice(position(q, 0.0)), rabi_λ)
        sols, w, t = run_pairs(rabi_slice, q, pairs; kw...)
        t, w, dense_error(sols, P -> eigvals(Matrix(rabi_slice(P))), q, range(0, 1, 41)), w.ok && swapped(sols, pairs)
    end,
    "QRM segment" => kw -> begin                               # a bracketing side passing 7e-4 from the EP
        q = Segment(rabi_st + P2(-1e-3, -7e-4), rabi_st + P2(1e-3, -7e-4))
        pairs = pair_near(rabi_slice(position(q, 0.0)), rabi_λ)
        sols, w, t = run_pairs(rabi_slice, q, pairs; kw...)
        t, w, dense_error(sols, P -> eigvals(Matrix(rabi_slice(P))), q, range(0, 1, 41)), w.ok
    end,
    "dimer loop" => kw -> begin                                # 625 states, around the upper EP
        q = Circle(P2(0, 0.005), 0.004)
        disc_λ(P) = -im .* filter(r -> real(r) > 0, roots_sorted(dimer_at(P)))
        pairs = eigenpairs_near(A_dimer(position(q, 0.0)), disc_λ(position(q, 0.0)))
        sols, w, t = run_pairs(A_dimer, q, pairs; kw...)
        t, w, dense_error(sols, disc_λ, q, range(0, 1, 201)), w.ok && swapped(sols, pairs)
    end,
    "mock scan" => kw -> begin                                 # 13 × 13 + 9 refinements: one leaf per planted point
        F = eigen(Matrix(A_mock(P2(0, 0))))
        start = [(F.values[k], F.vectors[:, k]) for k ∈ findall(λ -> imag(λ) > 1e-8, F.values)]
        t = @elapsed begin
            cells = tracked_scan(A_mock, range(0, 1, 13), range(0, 1, 13), start; kw...)
            leaves = refine_tracked(A_mock, cells, 9; kw...)
        end
        inside(c, p) = c.rect[1] ≤ p.x ≤ c.rect[2] && c.rect[3] ≤ p.y ≤ c.rect[4]
        correct = all(c -> c.lost == 0, cells) && length(leaves) == length(pts) && all(pts) do p
            k = findfirst(c -> inside(c, p), leaves)
            k !== nothing && (p.kind == :EP2 ? !isempty(leaves[k].swaps) :
                              (isempty(leaves[k].swaps) && abs(only(leaves[k].windings)[3]) == 2))
        end
        t, no_work, NaN, correct
    end,
    "JC scan" => kw -> begin                                   # 9 × 9 + 8 refinements: the tower EP_1 … EP_4
        Ncj = 5
        âj, σ̂j = qrm_operators(Ncj)
        jc = QRM(ωa=1.0, ωb=0.97, g=0.02, γa=0.1, γb=0.0, approx=[RWA_env, RWA_coupling])
        idx = excitation_sector(Ncj, 1)
        A = AffineLiouvillian(P -> liouvillian(Lindbladian(âj, σ̂j, setproperties(jc, (g=P[1], ωb=P[2])))...).data[idx, idx], 2)
        gs, ωbs = range(0.0101, 0.0303, 9), range(0.9713, 1.0291, 9)
        F = eigen(Matrix(A(P2(gs[1], ωbs[1]))))
        t = @elapsed leaves = refine_tracked(A, tracked_scan(A, gs, ωbs, collect(zip(F.values, eachcol(F.vectors))); kw...), 8; kw...)
        correct = length(leaves) == Ncj - 1 && all(1:Ncj - 1) do n
            ωb, g, _ = jc_EP(jc, n)
            k = findfirst(c -> c.rect[1] ≤ g ≤ c.rect[2] && c.rect[3] ≤ ωb ≤ c.rect[4], leaves)
            k !== nothing && length(leaves[k].swaps) == (n ∈ (1, Ncj - 1) ? 3 : 4) && isempty(leaves[k].windings)
        end
        t, no_work, NaN, correct
    end,
]


#%% Benchmark: every strategy on every workload (one warm-up pass, then the fastest of 3; ~5 min)

# The full comparison (23 variants, 2026-10-07) also had Vern6, Tsit5 at 1e-8/1e-5, BS3 at 1e-6/1e-3, refinement to 1e-9,
# and Euler–Newton with α = 0.05–0.2, linear predictor: same picture. Scans became wrong at reltol 1e-4 (Tsit5, BS3):
# the windings and labels need the steps to resolve the eigenvalues. 1e-6 keeps a decade of margin (1e-5 still passed).
results = Dict{Tuple{String,String},Any}()
for (sname, kw) ∈ strategies
    for (wname, wl) ∈ workloads
        r = try
            wl(kw)
            runs = [wl(kw) for _ ∈ 1:3]
            best = runs[argmin(first.(runs))]
            (best[1], best[2], best[3], all(x -> x[4], runs))
        catch err
            (NaN, (nf=0, steps=0, lu=0, ok=false), NaN, false)
        end
        results[(sname, wname)] = r
        @printf("%-28s %-15s %8.3f s  steps %5d  LU %5d  dense err %8.1e  %s\n", sname, wname, r[1], r[2].steps, r[2].lu,
                r[3], r[4] ? "ok" : "WRONG")
    end
end


#%% Figure: time relative to the old default, and factorisations per tracking

snames, wnames = first.(strategies), first.(workloads)
rel = [results[(s, w)][1]/results[(snames[1], w)][1] for s ∈ snames, w ∈ wnames]
wrong = [!results[(s, w)][4] for s ∈ snames, w ∈ wnames]
fig = Figure(size=(1500, 620))
ax = Axis(fig[1, 1]; title="time relative to Vern7 1e-10 with refinement (old default); ✗: wrong result",
          xticks=(1:length(wnames), wnames), yticks=(1:length(snames), snames), xticklabelrotation=π/6, yreversed=true)
hm = heatmap!(ax, 1:length(wnames), 1:length(snames), log10.(rel)'; colormap=clrs[:byz_div], colorrange=(-1.2, 1.2))
for i ∈ eachindex(snames), j ∈ eachindex(wnames)
    text!(ax, j, i; text=wrong[i, j] ? "✗" : @sprintf("%.2f", rel[i, j]), align=(:center, :center), fontsize=12,
          color=clrs[:text])
end
Colorbar(fig[1, 2], hm; label="log₁₀ (time / old default)")
loops = ["mock loop", "mock weak EP", "QRM check loop", "QRM segment", "dimer loop"]
ax_lu = Axis(fig[1, 3]; title="factorisations per tracked eigenpair", yscale=log10, xticks=(1:length(loops), loops),
             xticklabelrotation=π/6)
for (k, s) ∈ enumerate(snames)
    n_pairs(w) = w == "mock segment" ? 18 : 2
    scatterlines!(ax_lu, 1:length(loops), [max(results[(s, w)][2].lu/n_pairs(w), 0.5) for w ∈ loops];
                  color=Cycled(k), label=s, markersize=8, linestyle=occursin(r" g\b", s) ? :dash : :solid)   # GMRES dashed
end
Legend(fig[1, 4], ax_lu; labelsize=10, framevisible=false)
fig


#%% Scaling with the Liouvillian's size: full QRM odd sector, Nc = 6 … 28 (72 … 1568 states)

# One shift-invert (pair_near), and the check loop around EP_1 tracked with the old default, Tsit5 with refinement, and
# the new default. Expected: all like a sparse LU (≈ N^1.5 for this nearly 2D connectivity); the trackings' ratio to a
# shift-invert ≈ their number of factorisations (GMRES: ~2), plus Krylov iterations.
# Measured (2026-10-08): the new check loop costs ~11 shift-inverts at every size (old: 60 at N = 72, 130 at 1568), so
# the gain over the old default grows with N, 6× at N = 72 to 12× at 1568; the shift-invert approaches N^1.5 at large
# N (fixed overheads below N ~ 300).
scaling = []
for Nc ∈ (6, 8, 10, 14, 20, 28)
    A = rabi_family(Nc)
    A_sl, st, λm = rabi_ep(A, P3(3.3e-5, 0.016, 0.008), -0.083 - 0.992im)                       # EP_1
    q = Circle(st, 1e-3/16)
    x0 = position(q, 0.0)
    pair_near(A_sl(x0), λm)
    t_si = minimum(@elapsed(pair_near(A_sl(x0), λm)) for _ ∈ 1:3)
    times = map(["Vern7 1e-10 r (old default)", "Tsit5 1e-6 r", "Tsit5 1e-6 g (new default)"]) do s
        kw = Dict(strategies)[s]
        pairs = pair_near(A_sl(x0), λm)
        run_pairs(A_sl, q, pairs; kw...)
        minimum(run_pairs(A_sl, q, pairs; kw...)[3] for _ ∈ 1:2)/2                              # per eigenpair
    end
    push!(scaling, (N=size(A.L0, 1), si=t_si, loops=times))
    @printf("N = %5d: shift-invert %.4f s, check loop per eigenpair: old %.3f s, Tsit5 r %.3f s, new %.3f s\n",
            size(A.L0, 1), t_si, times...)
end
Ns = [s.N for s ∈ scaling]
fig_s = Figure(size=(800, 520))
ax_s = Axis(fig_s[1, 1]; xscale=log10, yscale=log10, xlabel="N (Liouvillian dimension)", ylabel="time (s)",
            title="full QRM odd sector: cost vs size")
for (k, (label, ys)) ∈ enumerate(["shift-invert (pair_near)" => [s.si for s ∈ scaling],
                                  "check loop, Vern7 1e-10 r (old)" => [s.loops[1] for s ∈ scaling],
                                  "check loop, Tsit5 1e-6 r" => [s.loops[2] for s ∈ scaling],
                                  "check loop, Tsit5 1e-6 g (new)" => [s.loops[3] for s ∈ scaling]])
    scatterlines!(ax_s, Ns, ys; color=Cycled(k), label)
end
lines!(ax_s, Ns, scaling[end].si .* (Ns ./ Ns[end]).^1.5; color=clrs[:overlay], linestyle=:dash, label="∝ N^1.5")
axislegend(ax_s; position=:lt, labelsize=10)
fig_s
