# EP lines in three-parameter spaces (δω, δγ, g): continuation of the complex EP of the Lindblad boson dimer, then of
# the tower of EPs of the quantum Rabi model, in its Jaynes–Cummings limit (RWA_coupling) and with the full coupling.
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/, here epline.jl).
# Start Julia with --threads=4: the half-lines of the EP lines are tracked in parallel (track_ep_lines).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, Accessors

# The EP lines through `seeds` ((prm_P0, σ0) pairs), their half-lines tracked in parallel (`track_ep_lines`: start
# Julia with --threads=4), each joined into one (points, λs, tangents); one printed line per half-line, with `labels`.
function track_lines(A, seeds, labels; kwargs...)
    t = @elapsed halves = track_ep_lines(A, seeds; kwargs...)
    for (label, hv) ∈ zip(labels, halves), (direction, l) ∈ ((1, hv.forward), (-1, hv.backward))
        println(label, ", direction $direction: $(length(l.points)) points, status $(l.status), g from ",
                round(l.points[1][3], sigdigits=3), " to ", round(l.points[end][3], sigdigits=3))
    end
    println(length(seeds), " lines (", 2length(seeds), " half-lines) in ", round(t, digits=1), " s, ",
            Threads.nthreads(), " threads")
    return [join_halves(hv...) for hv ∈ halves]
end


#%% Boson dimer in (δω, δγ, g): family, analytic EP lines

# Lindblad dimer (RWA_env, full coupling g/2(a + a†)(b + b†)): ωa,b = 1 ± δω, γa,b = γ0 ± δγ, coupling g, so
# P = (δω, δγ, g). Analytic EPs of the modes (`EP`): δω = 0 and g = 2|δγ|, i.e. two straight lines g = ±2δγ in the
# plane δω = 0, crossing at the origin (g = 0, δγ = 0: a DP, the uncoupled degenerate modes). The coalesced mode is
# ω_EP = √(1 - δγ²) - iγ0, the Liouvillian eigenvalue λ = -iω_EP (and its conjugate pair, far away).
# The line is straight but λ_EP is not constant along it. g → -g is the unitary b → -b: the lines are symmetric.
n_fock = 5
â, b̂ = destroy(n_fock) ⊗ eye(n_fock), eye(n_fock) ⊗ destroy(n_fock)
γ0 = 0.1
dimer_at(prm_P) = BosonDimer(ωa=1 + prm_P[1], ωb=1 - prm_P[1], γa=γ0 + prm_P[2], γb=γ0 - prm_P[2], g=prm_P[3],
                             approx=[RWA_env])
A3 = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(â, b̂, dimer_at(prm_P))...).data, 3; h=0.01)
modes_λ(prm_P) = -im .* filter(r -> real(r) > 0, roots_sorted(dimer_at(prm_P)))     # λ = -iω of the two modes
λ_exact(prm_P) = -γ0 - im*√(1 - prm_P[2]^2)
println("Liouvillian dimension ", size(A3.L0, 1))


#%% Seeds: the two EPs in the slice g = g0, by greedy bracketing

# The workflow of the tracker: find the EPs in a 2D slice (here by ep_bracket from a rectangle around each: regula
# falsi on D, bracketed by 3 × 3 cuts, down to a certified box of 1e-9; a tracked_scan of the slice would do the
# same), then continue each one in 3D. The slice of the family at g = g0 is a two-parameter family in (δω, δγ).
g0 = 0.04
A_g0 = slice(A3, P3(0, 0, g0), P3(1, 0, 0), P3(0, 1, 0))
targets_g0(st) = modes_λ(P3(st[1], st[2], g0))
seeds = map([1, -1]) do σ                                   # upper (δγ = g0/2) and lower (δγ = -g0/2) EP
    rect0 = (-0.011, 0.009, σ*g0/2 - 0.0081, σ*g0/2 + 0.0113)
    local t = @elapsed rect, hist, lines = ep_bracket(A_g0, rect0, targets_g0; tol=1e-9)
    seed = P3((rect[1] + rect[2])/2, (rect[3] + rect[4])/2, g0)
    println("EP δγ = $(σ*g0/2): box of size $(round(max(rect[2] - rect[1], rect[4] - rect[3]), sigdigits=2)) at ",
            round.(seed[1:2], sigdigits=8), " after $(length(hist) - 1) iterations, $(length(lines)) tracked lines, ",
            round(t, digits=1), " s")
    seed
end


#%% Continuation of both EP lines through 3D

# From each seed, track_ep_line in both directions (two half-lines, joined). Step h = 0.01, bounded to
# g ∈ [-0.06, 0.16] (γb = γ0 - δγ stays positive for |δγ| < 0.08), so the lines cross the DP at the origin and continue
# a little into g < 0.
bounds = (P3(-0.05, -0.09, -0.06), P3(0.05, 0.09, 0.16))
ep_lines = track_lines(A3, [(seed, λ_exact(seed)) for seed ∈ seeds],
                       ["seed δγ = $(round(seed[2], digits=4))" for seed ∈ seeds]; h=0.01, nsteps=60, bounds)
for (k, l) ∈ enumerate(ep_lines)
    σ = sign(l.points[end][2]*l.points[end][3])                            # g = 2σδγ on this line
    println("line $k (g = $(σ > 0 ? "" : "-")2δγ): max |δω| = ", round(maximum(abs.(getindex.(l.points, 1))), sigdigits=2),
            ",  max |g - 2σδγ| = ", round(maximum(abs(P[3] - 2σ*P[2]) for P ∈ l.points), sigdigits=2),
            ",  max |λ - λ_exact| = ", round(maximum(abs.(l.λs .- λ_exact.(l.points))), sigdigits=2))
end


#%% Plots: the lines in 3D, the coalesced mode along them, deviation from the analytic EPs

# The deviations grow with |g|: this is the Fock truncation (n_fock = 5), not the tracker. The counter-rotating
# coupling mixes in higher Fock states; at the analytic EP the truncated Liouvillian's pair midpoint is off by ~1e-13
# at g = 0.04 and ~1e-7 at g = 0.18 (n_fock = 6: ~2e-9). The tracker follows the EP of the truncated Liouvillian.
line_colours = (clrs[:byzantine], clrs[:selene])
fig = Figure(size=(1500, 620))
ax3 = Axis3(fig[1, 1]; protrusions=(60, 20, 30, 40), xlabel="δω", ylabel="δγ", zlabel="g", title="EP lines of the Lindblad dimer",
            limits=((-0.05, 0.05), (-0.09, 0.09), (-0.06, 0.16)), azimuth=0.35π, elevation=0.12π)
# the slice g = g0 where the seeds were found
mesh!(ax3, [Point3f(-0.05, -0.09, g0), Point3f(0.05, -0.09, g0), Point3f(0.05, 0.09, g0), Point3f(-0.05, 0.09, g0)],
      [1 2 3; 1 3 4]; color=(clrs[:overlay], 0.25), transparency=true, shading=NoShading)
for σ ∈ (1, -1)                                                            # analytic lines g = ±2δγ, δω = 0
    lines!(ax3, [Point3f(0, -0.03σ, -0.06), Point3f(0, 0.08σ, 0.16)]; color=clrs[:overlay], linestyle=:dash,
           label="analytic g = ±2δγ")
end
for (k, l) ∈ enumerate(ep_lines)
    scatterlines!(ax3, Point3f.(l.points); color=line_colours[k], markersize=7, label="tracked, line $k")
end
scatter!(ax3, Point3f.(seeds); marker=:star5, markersize=18, color=clrs[:helios], label="seeds (slice g = $g0)")
Legend(fig[2, 1], ax3; merge=true, unique=true, labelsize=10, nbanks=2, framevisible=false, tellheight=true)

# the coalesced mode ω_EP = iλ along the lines, against √(1 - δγ²) - iγ0 with δγ = g/2
ax_ω = Axis(fig[1, 2]; xlabel="g", ylabel="Re ω_EP", title="coalesced mode along the lines")
gs_ex = range(-0.06, 0.16, 200)
lines!(ax_ω, gs_ex, @. √(1 - (gs_ex/2)^2); color=clrs[:overlay], linestyle=:dash, label="√(1 - g²/4)")
for (k, l) ∈ enumerate(ep_lines)
    scatter!(ax_ω, getindex.(l.points, 3), real.(im .* l.λs); color=line_colours[k], markersize=8, label="line $k")
end
axislegend(ax_ω; position=:cb, labelsize=10)

# deviations from the analytic EP line and eigenvalue
ax_err = Axis(fig[1, 3]; xlabel="|g|", ylabel="deviation", yscale=log10, title="deviation from the analytic EPs")
for (k, l) ∈ enumerate(ep_lines)
    σ = sign(l.points[end][2]*l.points[end][3])
    gs = abs.(getindex.(l.points, 3))
    floor_ = 1e-17                                                         # log scale: exact zeros shown at the floor
    scatter!(ax_err, gs, max.(abs.(getindex.(l.points, 1)), floor_); color=line_colours[k], marker=:circle,
             markersize=7, label="|δω|, line $k")
    scatter!(ax_err, gs, max.([abs(P[3] - 2σ*P[2]) for P ∈ l.points], floor_); color=line_colours[k],
             marker=:utriangle, markersize=7, label="|g ∓ 2δγ|, line $k")
    scatter!(ax_err, gs, max.(abs.(l.λs .- λ_exact.(l.points)), floor_); color=line_colours[k], marker=:xcross,
             markersize=9, label="|λ - λ_exact|, line $k")
end
axislegend(ax_err; position=:lt, nbanks=2, labelsize=9)
fig


#%% Quantum Rabi model, Jaynes–Cummings limit: family in (δω, δγ, g), seeds by a scan of the slice g = g0

# QRM with all approximations (RWA_env, RWA_coupling, T = 0): ωa,b = ω0 ± δω (cavity, qubit), γa,b = γ0 ± δγ, coupling g.
# Closed form (qrm.jl): manifold n of H_eff has an EP2 where δω = 0 and g√n = |δγ|/2, so the tower is a family of
# straight lines g = |δγ|/(2√n) in the plane δω = 0, all through the origin. L conserves the excitation-number
# difference k: sector k = 1 (⟨a⟩, ⟨σ-⟩), where EP_n appears in the elements (n, n-1) and (n+1, n), as 3 or 4 swapped
# pairs at the same point. Seeds: a tracked scan of the slice g = g0, refined (as in qrm.jl), restricted to δγ > 0.
Nc_jc = 6                                                 # manifolds n ≤ 5 complete
â_q, σ̂_q = qrm_operators(Nc_jc)
idx_jc = excitation_sector(Nc_jc, 1)
ω0_q, γ0_q = 1.0, 0.05
qrm_at(mdl, prm_P) = setproperties(mdl, (ωa=ω0_q + prm_P[1], ωb=ω0_q - prm_P[1], γa=γ0_q + prm_P[2],
                                         γb=γ0_q - prm_P[2], g=prm_P[3]))
jc = QRM(approx=[RWA_env, RWA_coupling])
A_jc = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(â_q, σ̂_q, qrm_at(jc, prm_P))...).data[idx_jc, idx_jc], 3;
                         h=0.01)
println("JC, sector k = 1: dimension ", length(idx_jc))

g0_q = 0.008
slice_g0(A) = slice(A, P3(0, 0, g0_q), P3(1, 0, 0), P3(0, 1, 0))
xs_q, ys_q = range(-0.0031, 0.0029, 7), range(0.0043, 0.0457, 25)       # generic grid: no node on δω = 0
# seeds of the tower: the leaves of a refined tracked scan of the slice, from the eigenpairs `start` at its corner
function tower_seeds(A, start; depth=6)
    A_g0 = slice_g0(A)
    t = @elapsed cells = tracked_scan(A_g0, xs_q, ys_q, start)
    t_r = @elapsed leaves = refine_tracked(A_g0, cells, depth)
    println(length(start), " eigenvalues tracked; scan ", round(t, digits=1), " s, ", count(flagged, cells),
            " flagged cells; refinement ", round(t_r, digits=1), " s, ", length(leaves), " leaves")
    map(leaves) do c
        seed = P3((c.rect[1] + c.rect[2])/2, (c.rect[3] + c.rect[4])/2, g0_q)
        n = (seed[2]/(2g0_q))^2                                            # manifold, from the JC formula
        println("  leaf at (δω, δγ) = ", round.(seed[1:2], sigdigits=4), ", n ≈ ", round(n, digits=3), ", ",
                length(c.swaps), " swaps")
        (seed=seed, n=n, swaps=c.swaps)
    end
end
F_jc = eigen(Matrix(slice_g0(A_jc)(P2(xs_q[1], ys_q[1]))))
seeds_jc = tower_seeds(A_jc, collect(zip(F_jc.values, eachcol(F_jc.vectors))))


#%% Jaynes–Cummings limit: continuation of the tower

# From each leaf, one swapped pair (its mean is the shift σ) is followed in both directions: up to the positive-rate
# bound δγ < γ0 (here 0.048), down to g = 0.002, before the lines crowd at the origin. Steps fixed at h = 0.002 (hmax),
# for evenly resolved curves. Check against
# the closed form: δω = 0, g = δγ/(2√n), and the coalesced eigenvalue = midpoint of the two closed-form eigenvalues
# nearest it (comparing with one of them would show their √ splitting at the tracked point instead).
bounds_q = (P3(-0.01, -0.048, 0.002), P3(0.01, 0.048, 0.03))
# the tower lines through the seeds `ss` (one swapped pair each), with their manifold n
function tower_lines(A, ss, labels)
    ls = track_lines(A, [(s.seed, sum(s.swaps[1])/2) for s ∈ ss], labels; h=0.002, hmax=0.002, nsteps=80,
                     bounds=bounds_q)
    return [merge((n=round(Int, s.n),), l) for (s, l) ∈ zip(ss, ls)]
end
lines_jc = tower_lines(A_jc, seeds_jc, ["JC, n = $(round(Int, s.n))" for s ∈ seeds_jc])
function jc_midpoint(prm_P, λ)
    sp = jc_spectrum(qrm_at(jc, prm_P), Nc_jc; k=1)
    return sum(sp[partialsortperm(abs.(sp .- λ), 1:2)])/2
end
for l ∈ lines_jc
    println("JC n = $(l.n): ", length(l.points), " points, max |δω| = ", round(maximum(abs.(getindex.(l.points, 1))), sigdigits=2),
            ",  max |g - δγ/2√n| = ", round(maximum(abs(P[3] - P[2]/(2√l.n)) for P ∈ l.points), sigdigits=2),
            ",  max |λ - closed form| = ", round(maximum(abs.(l.λs .- jc_midpoint.(l.points, l.λs))), sigdigits=2))
end


#%% Quantum Rabi model, full coupling (RWA_env only): seeds and continuation of the tower

# Without RWA_coupling, H = ωa a†a + ωb σz/2 - ig(a - a†)σx conserves only the parity of N = a†a + σ+σ-: L is block
# diagonal in the parity of k, and the odd sector holds ⟨a⟩, ⟨σ-⟩. Only its eigenvalues near Im λ = -ω0 (the k = 1
# family: k = -1, ±3, … are at Im λ ≈ +ω0, ∓3ω0, far away) are tracked by the scan.
# - Truncation: the counter-rotating terms couple manifold n to n ± 2, so EP_n needs Fock states well above n. Checked at
#   the tracked points: Nc = 8 is converged for n = 1 but not n = 5 (a pair gap of ~3e-4 appears at Nc = 10), while
#   Nc = 10 and 12 agree to ~1e-10. Hence Nc = 10, and a window with n ≤ 5 only.
# - The 3–4 pairs that swap around EP_n in the JC limit no longer meet at one point: each has its own EP, ~2e-5 apart in
#   δγ at g = 0.008 (L is no longer block triangular). All the swapped pairs of every leaf are followed: one line each.
Nc_r = 10
â_r, σ̂_r = qrm_operators(Nc_r)
odd_r = sort(reduce(vcat, excitation_sector(Nc_r, k) for k ∈ -Nc_r:Nc_r if isodd(k)))
rabi = QRM(approx=[RWA_env])
A_rabi = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(â_r, σ̂_r, qrm_at(rabi, prm_P))...).data[odd_r, odd_r], 3;
                           h=0.01)
println("full QRM, odd sector: dimension ", length(odd_r))
ys_q = range(0.0043, 0.0373, 21)                          # n ≤ 5 (EP_6 at δγ = 0.0392 for g0 = 0.008)
F_r = eigen(Matrix(slice_g0(A_rabi)(P2(xs_q[1], ys_q[1]))))
start_r = [(F_r.values[k], F_r.vectors[:, k]) for k ∈ findall(λ -> -1.5 < imag(λ) < -0.5, F_r.values)]
seeds_rabi = tower_seeds(A_rabi, start_r)

# one seed per swapped pair, each pair once (a pair swaps in a single leaf)
pair_seeds = [(seed=s.seed, n=s.n, swaps=[sw]) for s ∈ seeds_rabi for sw ∈ s.swaps]
lines_rabi = tower_lines(A_rabi, pair_seeds,
                         ["QRM, n = $(round(Int, s.n)), λ_EP ≈ $(round(sum(s.swaps[1])/2, digits=4))" for s ∈ pair_seeds])
# independent check: dense eigenvalues of the sector at the tracked points, the pair nearest λ: its gap (~√|D| at the
# point: ~1e-8 for D at round-off) and the distance of its midpoint to λ
function dense_check(A, prm_P, λ)
    ev = eigvals(Matrix(A(prm_P)))
    k = partialsortperm(abs.(ev .- λ), 1:2)
    return abs(ev[k[1]] - ev[k[2]]), abs(sum(ev[k])/2 - λ)
end
for l ∈ lines_rabi
    chk = [dense_check(A_rabi, P, λ) for (P, λ) ∈ zip(l.points[1:3:end], l.λs[1:3:end])]
    println("QRM n = $(l.n), λ_EP ≈ $(round(l.λs[1], digits=4)): ", length(l.points), " points, δω from ",
            round(minimum(getindex.(l.points, 1)), sigdigits=2), " to ", round(maximum(getindex.(l.points, 1)), sigdigits=2),
            ";  dense check: max gap ", round(maximum(first.(chk)), sigdigits=2), ", max |midpoint - λ| ",
            round(maximum(last.(chk)), sigdigits=2))
end


#%% Plots: the tower in the JC limit and with the full coupling

# Left: both towers in 3D, with δω stretched (the full-coupling lines leave the plane δω = 0 by a few 1e-4).
# Middle: that shift δω along the lines (zero in the JC limit): δω ≈ n g²/2 for every line of manifold n (δω/g² → 0.50,
# 1.00, 1.50, 2.00, 2.50 within ~1% at the large-g ends; the pairs of one manifold scatter around it at small g), a
# Bloch–Siegert-like shift of the resonance from the counter-rotating terms. Right: the deviation from the JC line
# g = δγ/(2√n), which separates the lines of the different pairs of one manifold. Colour: manifold n.
n_colours = [clrs[k] for k ∈ (:byzantine, :selene, :minthe, :hesperides, :hyacinth)]
fig_q = Figure(size=(1600, 620))
ax3_q = Axis3(fig_q[1, 1]; protrusions=(60, 20, 30, 40), xlabel="δω", ylabel="δγ", zlabel="g",
              title="EP tower: JC limit (dashed) and full QRM", azimuth=0.3π, elevation=0.1π)
for l ∈ lines_jc
    lines!(ax3_q, Point3f.(l.points); color=n_colours[l.n], linestyle=:dash, linewidth=2, label="JC, n = $(l.n)")
end
for l ∈ lines_rabi
    scatterlines!(ax3_q, Point3f.(l.points); color=n_colours[l.n], markersize=4, label="full QRM, n = $(l.n)")
end
scatter!(ax3_q, [Point3f(s.seed) for s ∈ seeds_rabi]; marker=:star5, markersize=12, color=clrs[:helios],
         label="seeds (slice g = $g0_q)")
Legend(fig_q[2, 1], ax3_q; merge=true, unique=true, nbanks=4, labelsize=10, framevisible=false, tellheight=true)

ax_δω = Axis(fig_q[1, 2]; xlabel="g", ylabel="δω", title="shift of the EP lines off δω = 0")
for l ∈ lines_jc
    lines!(ax_δω, getindex.(l.points, 3), getindex.(l.points, 1); color=n_colours[l.n], linestyle=:dash)
end
for l ∈ lines_rabi
    scatterlines!(ax_δω, getindex.(l.points, 3), getindex.(l.points, 1); color=n_colours[l.n], markersize=5,
                  label="n = $(l.n)")
end
gs_guide = range(0, 0.026, 100)
for n ∈ 1:5                                                                 # observed: δω ≈ n g²/2
    lines!(ax_δω, gs_guide, n .* gs_guide.^2 ./ 2; color=n_colours[n], linestyle=:dot, linewidth=1, label="n g²/2")
end
ylims!(ax_δω, -2e-5, 3.9e-4)
axislegend(ax_δω; position=:lt, merge=true, unique=true, labelsize=10)

ax_dg = Axis(fig_q[1, 3]; xlabel="g", ylabel="g - δγ/(2√n)", title="deviation from the JC line (one curve per pair)")
for l ∈ lines_rabi
    scatterlines!(ax_dg, getindex.(l.points, 3), [P[3] - P[2]/(2√l.n) for P ∈ l.points]; color=n_colours[l.n],
                  markersize=5, label="n = $(l.n)")
end
hlines!(ax_dg, [0]; color=clrs[:overlay], linestyle=:dash)
axislegend(ax_dg; position=:lb, merge=true, unique=true, labelsize=10)
fig_q
