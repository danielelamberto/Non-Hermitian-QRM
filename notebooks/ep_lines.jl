# EP lines in three-parameter spaces (δω, δγ, g): continuation of the complex EP of the Lindblad boson dimer, then of
# the tower of EPs of the quantum Rabi model, in its Jaynes–Cummings limit (RWA_coupling) and with the full coupling;
# last, the JC tower in (δω, γa, γb) at fixed g, and in (g, δω, γa) at fixed γb with its defective crossings, which the
# full coupling splits into pairs of EP2s.
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/, here epline.jl).
# Start Julia with --threads=4: the half-lines of the EP lines are tracked in parallel (track_ep_lines).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, Accessors

# The EP lines through `seeds` ((prm_P0, σ0) pairs), their half-lines tracked in parallel (`track_ep_lines`: start
# Julia with --threads=4), each joined into one (points, λs, tangents); one printed line per half-line, with `labels`
# and the range of the third coordinate (named `zname`).
function track_lines(A, seeds, labels; zname="g", kwargs...)
    t = @elapsed halves = track_ep_lines(A, seeds; kwargs...)
    for (label, hv) ∈ zip(labels, halves), (direction, l) ∈ ((1, hv.forward), (-1, hv.backward))
        println(label, ", direction $direction: $(length(l.points)) points, status $(l.status), $zname from ",
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
    local t = @elapsed rect, hist, lines = ep_bracket(A_g0, rect0, targets_g0; box_tol=1e-9)
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
# pairs at the same point. Seeds: a tracked scan of the slice g = g0, its EPs localised (`localise_eps`), δγ > 0.
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
# seeds of the tower: a tracked scan of the slice from the eigenpairs `start` at its corner, then `localise_eps`
# (each swapping cell bracketed on its pair, down to 1e-5: the tracker's first correction finds the EP to round-off):
# one seed per swapped pair, with its coalesced eigenvalue
function tower_seeds(A, start)
    A_g0 = slice_g0(A)
    t = @elapsed cells = tracked_scan(A_g0, xs_q, ys_q, start)
    t_l = @elapsed loc = localise_eps(A_g0, cells; box_tol=1e-5)
    println(length(start), " eigenvalues tracked; scan ", round(t, digits=1), " s, ", count(flagged, cells),
            " flagged cells; localisation ", round(t_l, digits=1), " s: ", length(loc.eps), " EPs, ", length(loc.dps),
            " DPs, ", length(loc.unresolved), " unresolved")
    seeds = map(loc.eps) do e
        seed = P3((e.rect[1] + e.rect[2])/2, (e.rect[3] + e.rect[4])/2, g0_q)
        n = (seed[2]/(2g0_q))^2                                            # manifold, from the JC formula
        println("  EP at (δω, δγ) = ", round.(seed[1:2], sigdigits=6), ", n ≈ ", round(n, digits=3), ", λ = ",
                round(e.λ, digits=5))
        (seed=seed, n=n, λ=e.λ)
    end
    return seeds, cells, loc                                  # the cells and the localisation, for plots
end
F_jc = eigen(Matrix(slice_g0(A_jc)(P2(xs_q[1], ys_q[1]))))
seeds_jc, cells_jc, loc_jc = tower_seeds(A_jc, collect(zip(F_jc.values, eachcol(F_jc.vectors))))


#%% Jaynes–Cummings limit: continuation of the tower

# From each leaf, one swapped pair (its mean is the shift σ) is followed in both directions: up to the positive-rate
# bound δγ < γ0 (here 0.048), down to g = 0.002, before the lines crowd at the origin. Steps fixed at h = 0.002 (hmax),
# for evenly resolved curves. Check against
# the closed form: δω = 0, g = δγ/(2√n), and the coalesced eigenvalue = midpoint of the two closed-form eigenvalues
# nearest it (comparing with one of them would show their √ splitting at the tracked point instead).
bounds_q = (P3(-0.01, -0.048, 0.002), P3(0.01, 0.048, 0.03))
# the tower lines through the seeds `ss` (one swapped pair each), with their manifold n
function tower_lines(A, ss, labels)
    ls = track_lines(A, [(s.seed, s.λ) for s ∈ ss], labels; h=0.002, hmax=0.002, nsteps=80,
                     bounds=bounds_q)
    return [merge((n=round(Int, s.n),), l) for (s, l) ∈ zip(ss, ls)]
end
seeds_jc1 = unique(s -> round(Int, s.n), seeds_jc)        # JC: the 3–4 pairs of EP_n share its line, one each
lines_jc = tower_lines(A_jc, seeds_jc1, ["JC, n = $(round(Int, s.n))" for s ∈ seeds_jc1])
# the midpoint of the two closed-form eigenvalues (sector k = 1, Nc Fock states) of the JC model `mdl` nearest λ
function jc_midpoint(mdl, λ; Nc=Nc_jc)
    sp = jc_spectrum(mdl, Nc; k=1)
    return sum(sp[partialsortperm(abs.(sp .- λ), 1:2)])/2
end
for l ∈ lines_jc
    println("JC n = $(l.n): ", length(l.points), " points, max |δω| = ", round(maximum(abs.(getindex.(l.points, 1))), sigdigits=2),
            ",  max |g - δγ/2√n| = ", round(maximum(abs(P[3] - P[2]/(2√l.n)) for P ∈ l.points), sigdigits=2),
            ",  max |λ - closed form| = ",
            round(maximum(abs(λ - jc_midpoint(qrm_at(jc, P), λ)) for (P, λ) ∈ zip(l.points, l.λs)), sigdigits=2))
end


#%% Quantum Rabi model, full coupling (RWA_env only): seeds of the tower

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
seeds_rabi, cells_rabi, loc_rabi = tower_seeds(A_rabi, start_r)


#%% Full coupling: how the seeds were localised in the slice g = g0

# Needs the previous cell. Left: the tracked scan of the slice (the grid xs_q × ys_q), its flagged cells (shaded: each
# a clean swap of one or more pairs, so no refinement was needed) and the localised EPs (one per swapped pair, colour:
# manifold n). Middle: the flagged cell of EP_n_zoom, with the lines tracked by the bracketing of each of its pairs
# (colour: pair) and the rectangle kept at each iteration: the first cuts at x* ± δ, y* ± δ around the zero of the fit
# of D, which already keep a small central box (the four pairs' first boxes nearly coincide), then the final ones.
# Right: zoom on the final boxes only (filled, size ≤ tol = 1e-5), each with its pair's EP at its centre: the 3–4 pairs
# of EP_n_zoom, which coincide in the JC limit, sit apart here.
n_zoom = 3
rect_pts(r) = Point2f[(r[1], r[3]), (r[2], r[3]), (r[2], r[4]), (r[1], r[4]), (r[1], r[3])]
centre(r) = Point2f((r[1] + r[2])/2, (r[3] + r[4])/2)
manifold(e) = round(Int, (centre(e.rect)[2]/(2g0_q))^2)
seed_colours = [clrs[k] for k ∈ (:byzantine, :selene, :minthe, :hesperides, :hyacinth)]
pair_colours = [clrs[k] for k ∈ (:byzantine, :selene, :minthe, :helios)]
fig_seeds = Figure(size=(1600, 600))
ax_s = Axis(fig_seeds[1, 1]; xlabel="δω", ylabel="δγ", title="scan of the slice g = $g0_q and the localised EPs")
for c ∈ cells_rabi
    if flagged(c)
        poly!(ax_s, rect_pts(c.rect); color=(clrs[:overlay], 0.4), strokecolor=clrs[:subtext], strokewidth=0.8)
    else
        lines!(ax_s, rect_pts(c.rect); color=clrs[:surface], linewidth=0.6)
    end
end
for n ∈ 1:5
    es = filter(e -> manifold(e) == n, loc_rabi.eps)
    scatter!(ax_s, [centre(e.rect) for e ∈ es]; color=seed_colours[n], marker=:star5, markersize=11, label="EP_$n")
end
axislegend(ax_s; position=:rb, labelsize=10)
es_zoom = filter(e -> manifold(e) == n_zoom, loc_rabi.eps)
cell_zoom = es_zoom[1].cell
ax_c = Axis(fig_seeds[1, 2]; xlabel="δω", ylabel="δγ",
            title="EP_$n_zoom's cell: cut lines and kept boxes of its $(length(es_zoom)) pairs")
ax_f = Axis(fig_seeds[1, 3]; xlabel="δω", ylabel="δγ", title="zoom: final boxes, one per pair, and its EP (★)")
for (k, e) ∈ enumerate(es_zoom)
    colour = pair_colours[mod1(k, length(pair_colours))]
    for (a, b) ∈ e.segments
        lines!(ax_c, [Point2f(a), Point2f(b)]; color=(colour, 0.5), linewidth=0.7)
    end
    for r ∈ e.history[2:end]                                   # history[1] is the cell itself
        lines!(ax_c, rect_pts(r); color=colour, linewidth=1.5, label="λ ≈ $(round(e.λ, digits=4))")
    end
    poly!(ax_f, rect_pts(e.rect); color=(colour, 0.25), strokecolor=colour, strokewidth=1.5)
    scatter!(ax_f, [centre(e.rect)]; color=colour, marker=:star5, markersize=16)
end
limits!(ax_c, cell_zoom...)
finals = [e.rect for e ∈ es_zoom]
lo_x, hi_x = minimum(r -> r[1], finals), maximum(r -> r[2], finals)
lo_y, hi_y = minimum(r -> r[3], finals), maximum(r -> r[4], finals)
pad = 0.2*max(hi_x - lo_x, hi_y - lo_y)
limits!(ax_f, lo_x - pad, hi_x + pad, lo_y - pad, hi_y + pad)
Legend(fig_seeds[2, 2:3], ax_c; orientation=:horizontal, framevisible=false, labelsize=11, merge=true)
fig_seeds


#%% Full coupling: continuation of the tower

# Needs the seeds cell. One seed per swapped pair (localise_eps brackets each pair of a swapping cell separately).
lines_rabi = tower_lines(A_rabi, seeds_rabi,
                         ["QRM, n = $(round(Int, s.n)), λ_EP ≈ $(round(s.λ, digits=4))" for s ∈ seeds_rabi])
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


#%% Jaynes–Cummings limit in (δω, γa, γb) at fixed g: family, seeds by a scan of the slice γb = γb0

# Needs the JC cell (operators, sector, `jc`) and the JC continuation cell (`jc_midpoint`). Same model and sector k = 1,
# now with the coupling fixed at g = g_r and the two rates free: P = (δω, γa, γb), ωa,b = ω0 ± δω. With
# d₁ - d₂ = 2δω - i(γa - γb)/2, the discriminant of manifold n is ∝ (δω - i(γa - γb)/4)² + g²n: Im D = 0 needs δω = 0,
# then Re D = 0 gives γa - γb = ±4√n g. So the tower is two families of straight lines along (0, 1, 1) in the plane
# δω = 0, one per sign. (At fixed resonance δω = 0, D is real everywhere: in (γa, γb, g) the EPs are surfaces, which
# the line tracker cannot follow; hence δω as the third parameter.)
# Seeds: the slice γb = γb0 = 0.2, scanned in (δω, γa), crosses all of them for n ≤ 5, at γa = γb0 ± 4√n g_r. The
# scan also finds "DPs" at small γa (winding ±2, no swap): defective crossings, see the JC cells in (g, δω, γa) below.
g_r, γb0 = 0.02, 0.2
jc_rates_at(prm_P) = setproperties(jc, (ωa=ω0_q + prm_P[1], ωb=ω0_q - prm_P[1], γa=prm_P[2], γb=prm_P[3], g=g_r))
A_jcr = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(â_q, σ̂_q, jc_rates_at(prm_P))...).data[idx_jc, idx_jc], 3;
                          h=0.01)
A_γb0 = slice(A_jcr, P3(0, 0, γb0), P3(1, 0, 0), P3(0, 1, 0))
xs_r, ys_r = range(-0.0031, 0.0029, 7), range(0.0153, 0.3947, 39)     # generic grid: no node on an EP
F_jcr = eigen(Matrix(A_γb0(P2(xs_r[1], ys_r[1]))))
t = @elapsed cells_jcr = tracked_scan(A_γb0, xs_r, ys_r, collect(zip(F_jcr.values, eachcol(F_jcr.vectors))))
t_l = @elapsed loc_jcr = localise_eps(A_γb0, cells_jcr; box_tol=1e-5)
println(length(F_jcr.values), " eigenvalues tracked; scan ", round(t, digits=1), " s, ", count(flagged, cells_jcr),
        " flagged cells; localisation ", round(t_l, digits=1), " s: ", length(loc_jcr.eps), " EPs, ", length(loc_jcr.dps),
        " DPs, ", length(loc_jcr.unresolved), " unresolved")
# one seed per swapped pair, with its manifold n and the sign σ of γa - γb (from the closed form)
seeds_jcr = map(loc_jcr.eps) do e
    seed = P3((e.rect[1] + e.rect[2])/2, (e.rect[3] + e.rect[4])/2, γb0)
    Δγ = seed[2] - seed[3]
    (seed=seed, n=round(Int, (Δγ/(4g_r))^2), σ=Int(sign(Δγ)), λ=e.λ)
end
for s ∈ unique(s -> (s.n, s.σ), seeds_jcr)
    println("  EP at γa = ", round(s.seed[2], sigdigits=6), " (closed form ", round(γb0 + 4s.σ*√s.n*g_r, sigdigits=6),
            "), n = $(s.n), ", count(t -> (t.n, t.σ) == (s.n, s.σ), seeds_jcr), " swapped pairs")
end


#%% JC in (δω, γa, γb): continuation of the tower

# One swapped pair per line (the 3–4 pairs of EP_n share it), followed both ways along (0, 1, 1) to the bounds: a rate
# reaching 0.01 below, 0.5 above (the last point may overshoot the bound by a fraction of a step). Checked against the
# closed form: δω = 0, γa - γb = 4σ√n g_r, and the coalesced eigenvalue = midpoint of the closed-form pair.
bounds_r = (P3(-0.01, 0.01, 0.01), P3(0.01, 0.5, 0.5))
seeds_jcr1 = unique(s -> (s.n, s.σ), seeds_jcr)
lines_jcr = [merge((n=s.n, σ=s.σ), l) for (s, l) ∈ zip(seeds_jcr1,
             track_lines(A_jcr, [(s.seed, s.λ) for s ∈ seeds_jcr1],
                         ["JC, n = $(s.n), γa $(s.σ > 0 ? '>' : '<') γb" for s ∈ seeds_jcr1];
                         zname="γb", h=0.01, hmax=0.01, nsteps=80, bounds=bounds_r))]
for l ∈ lines_jcr
    println("n = $(l.n), σ = $(l.σ): ", length(l.points), " points, max |δω| = ",
            round(maximum(abs(P[1]) for P ∈ l.points), sigdigits=2), ",  max |γa - γb - 4σ√n g| = ",
            round(maximum(abs(P[2] - P[3] - 4l.σ*√l.n*g_r) for P ∈ l.points), sigdigits=2), ",  max |λ - closed form| = ",
            round(maximum(abs(λ - jc_midpoint(jc_rates_at(P), λ)) for (P, λ) ∈ zip(l.points, l.λs)), sigdigits=2))
end


#%% Plots: the tower in (δω, γa, γb)

# Needs the previous cells, and rect_pts, centre (full-coupling plot of the seeds).
# Left: the scan of the slice γb = γb0 (flagged cells shaded), its EPs (colour: manifold n) and DPs. Middle: the lines
# in 3D (circles: γa > γb, triangles: γa < γb) over the closed-form lines (dashed), and the seed slice. Right: the
# deviations from the closed form along the lines, against γb.
rate_colours = [clrs[k] for k ∈ (:byzantine, :selene, :minthe, :hesperides, :hyacinth)]
rate_marker(σ) = σ > 0 ? :circle : :utriangle
fig_r = Figure(size=(1600, 620))
ax_sr = Axis(fig_r[1, 1]; xlabel="δω", ylabel="γa", title="scan of the slice γb = $γb0 (g = $g_r)")
for c ∈ cells_jcr
    if flagged(c)
        poly!(ax_sr, rect_pts(c.rect); color=(clrs[:overlay], 0.4), strokecolor=clrs[:subtext], strokewidth=0.8)
    else
        lines!(ax_sr, rect_pts(c.rect); color=clrs[:surface], linewidth=0.6)
    end
end
for s ∈ sort(seeds_jcr1, by=s -> s.n)
    scatter!(ax_sr, [Point2f(s.seed[1], s.seed[2])]; color=rate_colours[s.n], marker=:star5, markersize=12,
             label="EP_$(s.n)")
end
scatter!(ax_sr, [centre(d.rect) for d ∈ loc_jcr.dps]; color=clrs[:subtext], marker=:diamond, markersize=7, label="DP")
Legend(fig_r[2, 1], ax_sr; merge=true, unique=true, orientation=:horizontal, labelsize=10, framevisible=false)

ax3_r = Axis3(fig_r[1, 2]; protrusions=(60, 20, 30, 40), xlabel="δω", ylabel="γa", zlabel="γb",
              title="EP tower of JC in (δω, γa, γb)", limits=((-0.01, 0.01), (0, 0.5), (0, 0.5)),
              azimuth=0.3π, elevation=0.1π)
mesh!(ax3_r, [Point3f(-0.01, 0, γb0), Point3f(0.01, 0, γb0), Point3f(0.01, 0.5, γb0), Point3f(-0.01, 0.5, γb0)],
      [1 2 3; 1 3 4]; color=(clrs[:overlay], 0.25), transparency=true, shading=NoShading)
for n ∈ 1:5, σ ∈ (1, -1)                                                # closed form: δω = 0, γa = γb + 4σ√n g
    Δ = 4σ*√n*g_r
    γbs = (max(0, -Δ), min(0.5, 0.5 - Δ))
    lines!(ax3_r, [Point3f(0, γb + Δ, γb) for γb ∈ γbs]; color=clrs[:overlay], linestyle=:dash)
end
for l ∈ lines_jcr
    scatterlines!(ax3_r, Point3f.(l.points); color=rate_colours[l.n], marker=rate_marker(l.σ), markersize=5)
end
scatter!(ax3_r, [Point3f(s.seed) for s ∈ seeds_jcr1]; marker=:star5, markersize=12, color=clrs[:helios])

ax_er = Axis(fig_r[1, 3]; xlabel="γb", ylabel="deviation", yscale=log10, title="deviation from the closed form")
for l ∈ lines_jcr
    γbs = getindex.(l.points, 3)
    floor_ = 1e-17                                                      # log scale: exact zeros shown at the floor
    scatter!(ax_er, γbs, max.([abs(P[2] - P[3] - 4l.σ*√l.n*g_r) for P ∈ l.points], floor_); color=rate_colours[l.n],
             marker=rate_marker(l.σ), markersize=6, label="EP_$(l.n)")
    scatter!(ax_er, γbs, max.([abs(λ - jc_midpoint(jc_rates_at(P), λ)) for (P, λ) ∈ zip(l.points, l.λs)], floor_);
             color=(rate_colours[l.n], 0.5), marker=:xcross, markersize=7)
end
Legend(fig_r[2, 2:3], [MarkerElement(marker=m, color=clrs[:subtext]) for m ∈ (:circle, :utriangle, :xcross)],
       ["|γa - γb - 4σ√n g|, γa > γb", "same, γa < γb", "|λ - closed form|"]; orientation=:horizontal,
       labelsize=10, framevisible=false)
fig_r


#%% JC in (g, δω, γa) at fixed γb: family, seeds by a scan of the slice g = g0

# Needs the JC continuation cell (`jc_midpoint`). JC at T = 0 again, sector k = 1, now with ωa,b = 1 ± δω, γb = 0.05
# fixed, and P = (g, δω, γa) over g ∈ [0.001, 0.01], γa ∈ [0.02, 0.08]: the region where extra EPs were reported. The
# tower: δω = 0, γa = γb ± 4√n g. In the slice g = g0 = 0.003, n = 1, …, 6 fall in the window, as close as 0.0026 in
# γa: Nc = 8 (manifolds n ≤ 7 complete), a grid step of 0.001 in γa. Lines with n ≥ 7 enter the window only below g0
# (n ≈ 56 at g = 0.001): they are the tower too, but this slice does not seed them.
# The scan finds the 12 EPs of the tower (46 swapped pairs) and two "DPs" (winding ±2, no swap): see the next cells.
Nc_g = 8
â_g, σ̂_g = qrm_operators(Nc_g)
idx_g = excitation_sector(Nc_g, 1)
γb_g, g0_g = 0.05, 0.003
jc_g_at(prm_P) = setproperties(jc, (ωa=1 + prm_P[2], ωb=1 - prm_P[2], γa=prm_P[3], γb=γb_g, g=prm_P[1]))
L_g(prm_P) = liouvillian(Lindbladian(â_g, σ̂_g, jc_g_at(prm_P))...).data[idx_g, idx_g]
A_jcg = AffineLiouvillian(L_g, 3; h=0.01)
A_g0g = slice(A_jcg, P3(g0_g, 0, 0), P3(0, 1, 0), P3(0, 0, 1))
xs_g, ys_g = range(-0.00613, 0.00587, 13), range(0.02013, 0.07993, 61)    # generic grid: no node on an EP
F_g = eigen(Matrix(A_g0g(P2(xs_g[1], ys_g[1]))))
t = @elapsed cells_g = tracked_scan(A_g0g, xs_g, ys_g, collect(zip(F_g.values, eachcol(F_g.vectors))))
t_l = @elapsed loc_g = localise_eps(A_g0g, cells_g; box_tol=1e-6)
println(length(F_g.values), " eigenvalues tracked; scan ", round(t, digits=1), " s, ", count(flagged, cells_g),
        " flagged cells; localisation ", round(t_l, digits=1), " s: ", length(loc_g.eps), " EPs, ", length(loc_g.dps),
        " DPs, ", length(loc_g.unresolved), " unresolved, ", length(loc_g.lost), " lost")
seeds_g = map(loc_g.eps) do e
    seed = P3(g0_g, (e.rect[1] + e.rect[2])/2, (e.rect[3] + e.rect[4])/2)
    Δγ = seed[3] - γb_g
    (seed=seed, n=round(Int, (Δγ/(4g0_g))^2), σ=Int(sign(Δγ)), λ=e.λ)
end
for s ∈ sort(unique(s -> (s.n, s.σ), seeds_g), by=s -> s.seed[3])
    println("  EP at γa = ", round(s.seed[3], sigdigits=6), " (closed form ", round(γb_g + 4s.σ*√s.n*g0_g, sigdigits=6),
            "), n = $(s.n), ", count(t -> (t.n, t.σ) == (s.n, s.σ), seeds_g), " swapped pairs")
end
for d ∈ loc_g.dps
    println("  DP at (δω, γa) = ", round.(((d.rect[1] + d.rect[2])/2, (d.rect[3] + d.rect[4])/2), sigdigits=6), ", λ ≈ ", round(d.windings[1][1], digits=5))
end


#%% JC in (g, δω, γa): continuation of the tower

# One swapped pair per line, both ways to the box g ∈ [0.001, 0.01], γa ∈ [0.02, 0.08] (steps h = 5e-4): the lines
# leave it through g = 0.001 below, γa = 0.02 or 0.08 above. Checked against the closed form.
bounds_g = (P3(0.001, -0.01, 0.02), P3(0.01, 0.01, 0.08))
seeds_g1 = unique(s -> (s.n, s.σ), seeds_g)
lines_g = [merge((n=s.n, σ=s.σ), l) for (s, l) ∈ zip(seeds_g1,
           track_lines(A_jcg, [(s.seed, s.λ) for s ∈ seeds_g1],
                       ["JC, n = $(s.n), γa $(s.σ > 0 ? '>' : '<') γb" for s ∈ seeds_g1];
                       zname="γa", h=0.0005, hmax=0.0005, nsteps=120, bounds=bounds_g))]
for l ∈ lines_g
    println("n = $(l.n), σ = $(l.σ): ", length(l.points), " points, max |δω| = ",
            round(maximum(abs(P[2]) for P ∈ l.points), sigdigits=2), ",  max |γa - γb - 4σ√n g| = ",
            round(maximum(abs(P[3] - γb_g - 4l.σ*√l.n*P[1]) for P ∈ l.points), sigdigits=2), ",  max |λ - closed form| = ",
            round(maximum(abs(λ - jc_midpoint(jc_g_at(P), λ; Nc=Nc_g)) for (P, λ) ∈ zip(l.points, l.λs)), sigdigits=2))
end


#%% JC in (g, δω, γa): the extra "EPs" are defective crossings

# The two DPs of the slice are not noise. In closed form the eigenvalues of L are λ = -i(εᵢ - conj εⱼ) over the levels
# of H_eff; at δω = 0 the levels of a PT-broken manifold share their real part, so two eigenvalues of different pairs
# can cross on a curve of the plane δω = 0 (one real condition). Searching all pairs of closed-form eigenvalues on a
# grid of that plane (`labelled`: the eigenvalues with their levels; the coalescences of the tower are left out),
# the box holds such curves where λ(n-, n-1 -) = λ(n+1 +, n +), n = 2, …, Nc - 2 (their number grows with Nc, each is
# converged once its manifolds are complete): small γa, g ≲ 0.004; the slice g = g0 crosses n = 2 and 3, its two DPs.
# In closed form: with manifolds n - 1, n, n + 1 PT-broken, Im ε = -(γa(2n - 1) + γb)/4 ± ρₙ, ρₙ = √((γa - γb)²/16 - g²n),
# and the crossing is at γa = ρₙ₋₁ + 2ρₙ + ρₙ₊₁ ≤ |γa - γb|: it needs γb ≥ 2γa. Hence none in the tower scans above
# (γa,b = γ0 ± δγ with δγ > 0, so γa > γb), and many in the rates scan at γb = 0.2 (its DPs at small γa).
# At every crossing L is defective: L - λ has a one-dimensional kernel (one zero singular value), the two eigenvectors
# coalesce. The jumps map |n+1⟩⟨n| to |n⟩⟨n-1|, one way only (L is block triangular), and that coupling makes the
# crossing a Jordan block. But the eigenvalues cross linearly, without a √ branch: no swap around them, winding ±2 of
# D, so the scan calls them DPs. A criterion based on eigenvectors (coalescence, condition number, Petermann factor)
# sees them as EPs, and a dense eigensolver splits them by ~√eps ≈ 1e-8 there, as at a true EP2.
function labelled(prm_P)
    lv = jc_effective_energies(jc_g_at(prm_P), Nc_g)
    return [(λ=-im*(lv[i][2] - conj(lv[j][2])), i=i, j=j, Ni=lv[i][1], Nj=lv[j][1])
            for j ∈ eachindex(lv) for i ∈ eachindex(lv) if lv[i][1] - lv[j][1] == 1]
end
level_name(N, i) = N == 0 || N == Nc_g ? "$N" : "$N$(iseven(i) ? '+' : '-')"   # level 2N: +√, 2N+1: -√
pair_name(e) = level_name(e.Ni, e.i) * "←" * level_name(e.Nj, e.j)
gs_c, γs_c = range(0.001, 0.01, 181), range(0.02, 0.08, 241)
S_c = [labelled(P3(g, 0, γ)) for g ∈ gs_c, γ ∈ γs_c]
n_lab = length(S_c[1])
# crossing curves (a, b) => points (g, γa): Re(λa - λb) changes sign along γa where Im(λa - λb) = 0
crossings = Dict{Tuple{Int, Int}, Vector{Point2f}}()
for a ∈ 1:n_lab, b ∈ a + 1:n_lab
    ea, eb = S_c[1][a], S_c[1][b]
    ea.Ni == eb.Ni && (ea.i == eb.i || ea.j == eb.j) && continue          # the two branches of one tower EP
    for ig ∈ eachindex(gs_c), iγ ∈ 1:length(γs_c) - 1
        d1, d2 = S_c[ig, iγ][a].λ - S_c[ig, iγ][b].λ, S_c[ig, iγ + 1][a].λ - S_c[ig, iγ + 1][b].λ
        if sign(real(d1)) != sign(real(d2)) && max(abs(imag(d1)), abs(imag(d2))) < 1e-12
            γc = γs_c[iγ] - real(d1)*(γs_c[iγ + 1] - γs_c[iγ])/(real(d2) - real(d1))
            push!(get!(crossings, (a, b), Point2f[]), Point2f(gs_c[ig], γc))
        end
    end
end
# the exact crossing at g (bisection in γa from the grid point γ0), and the two smallest singular values of L - λ there
function crossing_at(a, b, g, γ0; w=3e-4)
    f(γ) = (s = labelled(P3(g, 0, γ)); real(s[a].λ - s[b].λ))
    lo, hi = γ0 - w, γ0 + w
    for _ ∈ 1:60
        m = (lo + hi)/2
        sign(f(m)) == sign(f(lo)) ? (lo = m) : (hi = m)
    end
    prm_P = P3(g, 0, (lo + hi)/2)
    λ = labelled(prm_P)[a].λ
    return prm_P, λ, svdvals(Matrix(L_g(prm_P)) - λ*I)[end-1:end]
end
println(length(crossings), " crossing curves in the plane δω = 0:")
for ((a, b), pts) ∈ sort(collect(crossings), by=c -> -length(c[2]))
    prm_P, λ, sv = crossing_at(a, b, pts[end ÷ 2 + 1]...)
    println("  ", pair_name(S_c[1][a]), " × ", pair_name(S_c[1][b]), ": g ", round(pts[1][1], digits=4), "–",
            round(pts[end][1], digits=4), ", γa ", round(minimum(last.(pts)), digits=4), "–",
            round(maximum(last.(pts)), digits=4), ";  at g = ", round(prm_P[1], digits=4), ": σ_min of L - λ = ",
            round.(sv, sigdigits=2), sv[1] > 1e-8 ? " (one zero: defective)" : " (two zeros: diabolic)")
end


#%% Plots: the tower and the defective crossings in the plane δω = 0

# Needs the previous cells, and rect_pts, centre (full-coupling plot of the seeds).
# Left: the plane δω = 0 (every line lies in it): the tracked tower (markers, colour: manifold n) over the closed form
# (dashed), the defective crossing curves (solid), the seed slice g = g0 with its EPs and DPs. Middle: zoom on the
# crossings. Right: why the crossings are not EP2s: the splitting |λa - λb| of the two eigenvalues against the distance
# δω off the plane, at a tower EP (∝ √δω) and at a defective crossing (∝ δω), from the closed form; dense eigenvalues
# of L (circles) agree.
manifold_colours = [clrs[k] for k ∈ (:byzantine, :selene, :minthe, :hesperides, :hyacinth, :helios)]
fig_g = Figure(size=(1700, 640))
function plot_plane!(ax)
    gs_l = range(0.001, 0.01, 400)
    for n ∈ 1:Nc_g - 1, σ ∈ (1, -1)
        lines!(ax, gs_l, γb_g .+ 4σ*√n .* gs_l; color=clrs[:overlay], linestyle=:dash, label="tower, closed form")
    end
    for l ∈ lines_g
        scatter!(ax, getindex.(l.points, 1), getindex.(l.points, 3); color=manifold_colours[l.n], markersize=5,
                 label="tracked EP_$(l.n)")
    end
    for pts ∈ values(crossings)
        lines!(ax, sort(pts, by=first); color=clrs[:text], linewidth=2, label="defective crossing")
    end
    vlines!(ax, [g0_g]; color=clrs[:subtext], linewidth=0.8)
    scatter!(ax, [Point2f(g0_g, s.seed[3]) for s ∈ seeds_g1]; marker=:star5, markersize=12, color=clrs[:subtext],
             label="seeds (slice g = $g0_g)")
    scatter!(ax, [Point2f(g0_g, centre(d.rect)[2]) for d ∈ loc_g.dps]; marker=:diamond, markersize=11,
             color=clrs[:text], strokecolor=clrs[:subtext], strokewidth=1, label="DPs of the slice")
end
ax_p = Axis(fig_g[1, 1]; xlabel="g", ylabel="γa", title="JC, γb = $γb_g, δω = 0: EP tower and defective crossings",
            limits=((0.001, 0.01), (0.02, 0.08)))
plot_plane!(ax_p)
ax_z = Axis(fig_g[1, 2]; xlabel="g", ylabel="γa", title="zoom on the crossings", limits=((0.001, 0.004), (0.02, 0.026)))
plot_plane!(ax_z)
Legend(fig_g[2, 1:2], ax_p; merge=true, unique=true, orientation=:horizontal, nbanks=2, labelsize=10,
       framevisible=false)

# splitting off the plane: a tower EP (seed of EP_2, γa < γb) and the crossing n = 2 of the slice
ax_s = Axis(fig_g[1, 3]; xlabel="δω", ylabel="|λa - λb|", xscale=log10, yscale=log10,
            title="splitting off the plane δω = 0")
δωs = 10 .^ range(-7, -3, 30)
s_ep = only(filter(s -> (s.n, s.σ) == (2, -1), seeds_g1))
ab_ep = partialsortperm([abs(e.λ - s_ep.λ) for e ∈ labelled(s_ep.seed)], 1:2)
(a_x, b_x), pts_x = only(filter(c -> pair_name(S_c[1][c[1][1]]) == "2-←1-", collect(crossings)))
P_x, _, _ = crossing_at(a_x, b_x, g0_g, pts_x[argmin(abs.(first.(pts_x) .- g0_g))][2])
for (label, P0, (a, b), colour) ∈ (("tower EP_2", s_ep.seed, ab_ep, manifold_colours[2]),
                                   ("defective crossing", P_x, (a_x, b_x), clrs[:text]))
    split(δω) = (s = labelled(P0 + P3(0, δω, 0)); abs(s[a].λ - s[b].λ))
    lines!(ax_s, δωs, split.(δωs); color=colour, linewidth=2, label=label)
    λ0 = labelled(P0)[a].λ                                               # dense check at a few points
    dense = map(δωs[1:6:end]) do δω
        ev = eigvals(Matrix(L_g(P0 + P3(0, δω, 0))))
        k = partialsortperm(abs.(ev .- λ0), 1:2)
        abs(ev[k[1]] - ev[k[2]])
    end
    scatter!(ax_s, δωs[1:6:end], dense; color=colour, markersize=9)
end
lines!(ax_s, δωs, 0.5 .* δωs; color=clrs[:overlay], linestyle=:dot, label="∝ δω")
lines!(ax_s, δωs, 0.05 .* sqrt.(δωs); color=clrs[:overlay], linestyle=:dash, label="∝ √δω")
axislegend(ax_s; position=:rb, labelsize=10)
fig_g


#%% Full QRM in (g, δω, γa) at fixed γb: seeds by a scan of the slice g = g0

# Needs the JC cells in (g, δω, γa) (`labelled`, the crossings). The same box with the full coupling (RWA_env only),
# odd parity sector, window -1.5 < Im λ < -0.5 as for the full tower above. Nc = 10: the EPs of the slice agree with
# Nc = 12 to ~1e-10. The slice holds 50 EPs and no DP (JC: 46 and 2 DPs): the tower, its pairs now at distinct points
# shifted off δω = 0, and 4 more, two at each defective crossing of JC. Each crossing has split into two EP2s of the
# same pair, ~1e-4 apart in δω: the 2 × 2 picture of a crossing [λ₁(s), c; ε, λ₂(s)] with linear λ₁ - λ₂ ∝ s, where
# the counter-rotating terms give the missing back-coupling ε, and D = (λ₁ - λ₂)² + 4cε has two simple zeros.
# Classified by the closed form of JC at the same point: an EP is "extra" if its two nearest JC eigenvalues are a
# crossing pair rather than the two branches of one level.
Nc_rg = 10
â_rg, σ̂_rg = qrm_operators(Nc_rg)
odd_rg = sort(reduce(vcat, excitation_sector(Nc_rg, k) for k ∈ -Nc_rg:Nc_rg if isodd(k)))
rabi_g_at(prm_P) = setproperties(rabi, (ωa=1 + prm_P[2], ωb=1 - prm_P[2], γa=prm_P[3], γb=γb_g, g=prm_P[1]))
L_rg(prm_P) = liouvillian(Lindbladian(â_rg, σ̂_rg, rabi_g_at(prm_P))...).data[odd_rg, odd_rg]
A_rg = AffineLiouvillian(L_rg, 3; h=0.01)
A_g0rg = slice(A_rg, P3(g0_g, 0, 0), P3(0, 1, 0), P3(0, 0, 1))
F_rg = eigen(Matrix(A_g0rg(P2(xs_g[1], ys_g[1]))))
start_rg = [(F_rg.values[k], F_rg.vectors[:, k]) for k ∈ findall(λ -> -1.5 < imag(λ) < -0.5, F_rg.values)]
t = @elapsed cells_rg = tracked_scan(A_g0rg, xs_g, ys_g, start_rg)
t_l = @elapsed loc_rg = localise_eps(A_g0rg, cells_rg; box_tol=1e-6)
println("full QRM, odd sector: dimension ", length(odd_rg), ", ", length(start_rg), " eigenvalues tracked; scan ",
        round(t, digits=1), " s, ", count(flagged, cells_rg), " flagged cells; localisation ", round(t_l, digits=1),
        " s: ", length(loc_rg.eps), " EPs, ", length(loc_rg.dps), " DPs, ", length(loc_rg.unresolved), " unresolved, ",
        length(loc_rg.lost), " lost")
function is_extra(prm_P, λ)
    sp = labelled(prm_P)
    ea, eb = sp[partialsortperm([abs(e.λ - λ) for e ∈ sp], 1:2)]
    return !(ea.Ni == eb.Ni && (ea.i == eb.i || ea.j == eb.j))
end
seeds_rg = map(loc_rg.eps) do e
    seed = P3(g0_g, (e.rect[1] + e.rect[2])/2, (e.rect[3] + e.rect[4])/2)
    (seed=seed, λ=e.λ, extra=is_extra(seed, e.λ))
end
for s ∈ filter(s -> s.extra, seeds_rg)
    println("  extra EP at (δω, γa) = ", round.(s.seed[2:3], sigdigits=6), ", λ = ", round(s.λ, digits=6))
end


#%% Full QRM in (g, δω, γa): continuation of the tower and of the split crossings

# Every swapped pair of the slice is followed (one line each), to the box. The extra lines run ~1e-4 from their partner:
# steps of 1e-4 (2e-4 at most), against 5e-4 for the tower. Independent check at every fourth point: dense eigenvalues,
# the gap of the pair nearest λ (~√eps at an EP2) and the distance of its midpoint to λ. Then the splitting of each
# extra pair against g: Δδω ∝ g² (Δδω/g² ≈ 13 and 23–32 for g from 0.001 to 0.003), the counter-rotating terms acting
# at second order; both lines follow the crossing curve of JC in (g, γa).
seeds_rgx, seeds_rgt = filter(s -> s.extra, seeds_rg), filter(s -> !s.extra, seeds_rg)
lines_rgx = track_lines(A_rg, [(s.seed, s.λ) for s ∈ seeds_rgx],
                        ["split crossing, λ ≈ $(round(s.λ, digits=5))" for s ∈ seeds_rgx];
                        zname="γa", h=1e-4, hmax=2e-4, nsteps=200, bounds=bounds_g)
lines_rgt = track_lines(A_rg, [(s.seed, s.λ) for s ∈ seeds_rgt],
                        ["tower, λ ≈ $(round(s.λ, digits=5))" for s ∈ seeds_rgt];
                        zname="γa", h=0.0005, hmax=0.0005, nsteps=120, bounds=bounds_g)
dense_rg = [[dense_check(A_rg, P, λ) for (P, λ) ∈ zip(l.points[1:4:end], l.λs[1:4:end])] for l ∈ [lines_rgx; lines_rgt]]
println("dense check of the ", length(dense_rg), " lines: max gap ", round(maximum(maximum(first.(c)) for c ∈ dense_rg),
        sigdigits=2), ", max |midpoint - λ| ", round(maximum(maximum(last.(c)) for c ∈ dense_rg), sigdigits=2))
# the extra lines in pairs: the two lines from one crossing (nearest seeds)
pairs_rgx = [(i, argmin(j -> j == i ? Inf : abs(seeds_rgx[j].λ - seeds_rgx[i].λ), eachindex(seeds_rgx)))
             for i ∈ eachindex(seeds_rgx)]
pairs_rgx = unique(p -> minmax(p...), pairs_rgx)
point_at(l, g) = l.points[argmin(abs.(getindex.(l.points, 1) .- g))]   # the tracked point nearest g
for (i, j) ∈ pairs_rgx, g ∈ (0.001, 0.0015, 0.002, 0.003)
    Pi, Pj = point_at(lines_rgx[i], g), point_at(lines_rgx[j], g)
    println("  crossing λ ≈ ", round(seeds_rgx[i].λ, digits=4), ", g ≈ $g: Δδω = ", round(abs(Pi[2] - Pj[2]), sigdigits=3),
            " (Δδω/g² = ", round(abs(Pi[2] - Pj[2])/g^2, sigdigits=3), "), centre δω = ",
            round((Pi[2] + Pj[2])/2, sigdigits=3), ", Δγa = ", round(abs(Pi[3] - Pj[3]), sigdigits=2))
end


#%% Plots: the split crossings of the full QRM

# Needs the previous cells. Left: projection on (g, γa), zoomed on the crossings: the defective crossing curves of JC
# (solid) and the tower of JC (dashed), the full-QRM lines (tower: thin, colour by pair; extra: markers). Right: their
# distance δω off the plane of JC, against g: each crossing gives two lines straddling a common centre, ~g² apart; the
# tower lines (thin) shift as n g²/2.
fig_rg = Figure(size=(1400, 620))
ax_pr = Axis(fig_rg[1, 1]; xlabel="g", ylabel="γa", title="full QRM, γb = $γb_g: split crossings, projection on (g, γa)",
             limits=((0.001, 0.004), (0.0195, 0.026)))
gs_l = range(0.001, 0.004, 200)
for n ∈ 1:Nc_g - 1, σ ∈ (1, -1)
    lines!(ax_pr, gs_l, γb_g .+ 4σ*√n .* gs_l; color=clrs[:overlay], linestyle=:dash, label="JC tower")
end
for pts ∈ values(crossings)
    lines!(ax_pr, sort(pts, by=first); color=clrs[:text], linewidth=2, label="JC defective crossing")
end
for l ∈ lines_rgt
    lines!(ax_pr, getindex.(l.points, 1), getindex.(l.points, 3); color=(clrs[:subtext], 0.6), linewidth=1,
           label="full QRM tower")
end
extra_colours = (clrs[:helios], clrs[:selene])
for (k, (i, j)) ∈ enumerate(pairs_rgx), (m, l) ∈ ((:circle, lines_rgx[i]), (:utriangle, lines_rgx[j]))
    scatter!(ax_pr, getindex.(l.points, 1), getindex.(l.points, 3); color=extra_colours[k], marker=m, markersize=6,
             label="full QRM, split crossing $k")
end
Legend(fig_rg[2, 1], ax_pr; merge=true, unique=true, orientation=:horizontal, nbanks=2, labelsize=10,
       framevisible=false)

ax_dr = Axis(fig_rg[1, 2]; xlabel="g", ylabel="δω", title="distance off the plane δω = 0",
             limits=((0.001, 0.004), nothing))
for l ∈ lines_rgt
    lines!(ax_dr, getindex.(l.points, 1), getindex.(l.points, 2); color=(clrs[:subtext], 0.6), linewidth=1)
end
for (k, (i, j)) ∈ enumerate(pairs_rgx), (m, l) ∈ ((:circle, lines_rgx[i]), (:utriangle, lines_rgx[j]))
    scatterlines!(ax_dr, getindex.(l.points, 1), getindex.(l.points, 2); color=extra_colours[k], marker=m,
                  markersize=6)
end
hlines!(ax_dr, [0]; color=clrs[:overlay], linestyle=:dash)
fig_rg
