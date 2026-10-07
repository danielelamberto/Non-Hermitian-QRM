# Mock Liouvillian with planted singularities: test bed for the monodromy scans.
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using NonHermitianQRM: Circle          # also exported by Makie: ours is the loop in parameter space
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, SparseArrays, Polynomials, Accessors, Random
figdir = joinpath(pkgdir(NonHermitianQRM), "figures")
mkpath(figdir)


#%% Helpers: windows, comparison with the planted singularities, plots of a scan

# The eigenpairs with Im λ > 0 of the dense L at δ: the window followed by the scans (one diagonalisation).
function upper_window(A, δ)
    F = eigen(Matrix(A(δ)))
    return [(F.values[k], F.vectors[:, k]) for k ∈ findall(λ -> imag(λ) > 1e-8, F.values)]
end

# largest distance between two spectra, matched greedily
function spectrum_distance(a, b)
    b, e = copy(b), 0.0
    for v ∈ a
        k = argmin(abs.(b .- v))
        e = max(e, abs(b[k] - v))
        deleteat!(b, k)
    end
    return e
end

# a planted point inside a cell; a planted line a x + b y + c = 0 crossing a cell
inside(c, p) = c.rect[1] ≤ p.x ≤ c.rect[2] && c.rect[3] ≤ p.y ≤ c.rect[4]
function crossed(c, l)
    a, b, cc = l.coeffs
    f = [a*x + b*y + cc for x ∈ c.rect[1:2] for y ∈ c.rect[3:4]]
    return minimum(f) ≤ 1e-12 && maximum(f) ≥ -1e-12
end

# what a leaf of the refinement says about the point it contains
verdict(c) = !isempty(c.swaps) ? :EP2 : !isempty(c.windings) ? :DP : :unclear

# a cell drawn by its result: swap (red), nonzero winding (orange), longer cycle (magenta), lost labels (gray)
cell_colour(c) = !isempty(c.swaps) ? clrs[:byzantine] : !isempty(c.windings) ? clrs[:hesperides] : c.cycles > 0 ? clrs[:hyacinth] :
                 c.lost > 0 ? (clrs[:overlay], 0.6) : nothing
rect_poly(r) = Point2f[(r[1], r[3]), (r[2], r[3]), (r[2], r[4]), (r[1], r[4])]
function draw_cells!(ax, cells)
    for c ∈ cells
        col = cell_colour(c)
        col === nothing || poly!(ax, rect_poly(c.rect); color=col)
    end
end

# planted lines (solid: real-axis EP2, dashed: DP) across the unit square, and planted points (stars: EP2, circles: DP)
function draw_planted!(ax, pts, plines; markersize=12)
    for l ∈ plines
        a, b, c = l.coeffs
        xx, yy = abs(b) > abs(a) ? ([0.0, 1.0], [-c/b, -(a + c)/b]) : ([-c/a, -(b + c)/a], [0.0, 1.0])
        lines!(ax, xx, yy; color=clrs[:text], linestyle=(l.kind == :real_EP2 ? :solid : :dash))
    end
    scatter!(ax, [p.x for p ∈ pts], [p.y for p ∈ pts]; marker=[p.kind == :EP2 ? :star5 : :circle for p ∈ pts],
        markersize, color=:transparent, strokecolor=clrs[:text], strokewidth=1.2)
end
scan_legend(pos) = Legend(pos,
    [PolyElement(color=clrs[:byzantine]), PolyElement(color=clrs[:hesperides]), PolyElement(color=clrs[:hyacinth]), PolyElement(color=(clrs[:overlay], 0.6)),
     LineElement(color=clrs[:text]), LineElement(color=clrs[:text], linestyle=:dash),
     MarkerElement(marker=:star5, color=:transparent, strokecolor=clrs[:text], strokewidth=1.2),
     MarkerElement(marker=:circle, color=:transparent, strokecolor=clrs[:text], strokewidth=1.2)],
    ["swap", "identity, nonzero winding", "cycle of ≥ 3", "lost labels", "real-axis EP2 line", "DP line", "planted EP2",
     "planted DP"]; orientation=:horizontal, nbanks=2, tellheight=true)


#%% Mock Liouvillian: spectrum and planted singularities

# Test bed for the monodromy tools (see MockLiouvillian): a Lindblad sector affine in (x, y) ∈ [0, 1]², with an exactly
# known spectrum. This cell checks it: closed-form vs numerical eigenvalues (both sectors), affinity, and the scaling of
# the gap at every planted singularity (√ at an EP2, linear at a DP), then maps the smallest gap over the square in the
# default coherence sector, where the planted singularities are the only ones.
mock = MockLiouvillian()
mock_at(δ) = setproperties(mock, (x=δ[1], y=δ[2]))
pts, plines = mock_singularities(mock)              # planted points and lines
for p ∈ pts
    println("planted ", rpad(p.kind, 4), " block $(p.block) at (x, y) = (", p.x, ", ", round(p.y, digits=6), "),  λ = ", p.λ)
end
for l ∈ plines
    println("planted ", rpad(l.kind, 9), " block $(l.block): ", l.coeffs[1], " x + ", l.coeffs[2], " y + ", l.coeffs[3], " = 0")
end

# closed form vs numerical spectrum at random points, and affinity in (x, y)
for sector ∈ (:coherences, :within)
    err = maximum(spectrum_distance(eigvals(Matrix(mock_matrix(mock_at(δ); sector))), mock_spectrum(mock_at(δ); sector))
                  for δ ∈ (rand(2) for _ ∈ 1:20))
    println("sector $sector ($(length(mock_sector(mock; sector))) states): max |λ_numerical - λ_closed form| over 20 random points = ", err)
end
A_mock = AffineLiouvillian(δ -> mock_matrix(mock_at(δ)), 2)
println("affine in (x, y): accepted by AffineLiouvillian")

# gap scaling at the planted singularities, from the numerical eigenvalues: exponent of gap ∝ distance^p
pair_gap(δ, λ0) = (v = eigvals(Matrix(mock_matrix(mock_at(δ)))); i = sortperm(abs.(v .- λ0)); abs(v[i[1]] - v[i[2]]))
gap_exponent(gap) = log10(gap(1e-4)/gap(1e-6))/2
for p ∈ pts
    println(rpad("$(p.kind) at ($(p.x), $(round(p.y, digits=4)))", 30), " exponent ",
        round(gap_exponent(d -> pair_gap(P2(p.x + d, p.y), p.λ)), digits=3), "  (expected ", p.kind == :EP2 ? 0.5 : 1.0, ")")
end
for l ∈ plines
    a, b, c = l.coeffs
    n = P2(a, b)/hypot(a, b)
    δ0 = P2(-(0.6b + c)/a, 0.6)                         # the point of the line at y = 0.6, approached along its normal
    # the coalescing pair: -3γ/4 ± ... for the driven qubit (3rd entry), -γ/2 ± iΔ for the detuned one (1st entry)
    λ0 = block_spectrum(mock.blocks[l.block], δ0...)[l.kind == :real_EP2 ? 3 : 1]
    println(rpad("$(l.kind) at ($(δ0[1]), $(round(δ0[2], digits=4)))", 30), " exponent ",
        round(gap_exponent(d -> pair_gap(δ0 + d*n, λ0)), digits=3), "  (expected ", l.kind == :real_EP2 ? 0.5 : 1.0, ")")
end

# map of the smallest gap between eigenvalues (steady states, one per block, excluded) with the planted structures
xs = ys = range(0, 1, 201)
gapmap = [begin
              v = filter(λ -> abs(λ) > 1e-9, mock_spectrum(mock_at((x, y))))
              minimum(abs(v[i] - v[j]) for i ∈ eachindex(v) for j ∈ i+1:length(v))
          end for x ∈ xs, y ∈ ys]
fig = Figure(size=(760, 700))
ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="mock Liouvillian (coherence sector): log₁₀ smallest gap",
    aspect=DataAspect(), limits=((0, 1), (0, 1)))
hm = heatmap!(ax, xs, ys, log10.(gapmap); colormap=clrs[:byz])
Colorbar(fig[1, 2], hm)
draw_planted!(ax, pts, plines; markersize=14)
fig


#%% Mock Liouvillian: monodromy scan by eigenpair tracking (one diagonalisation)

# Needs the mock check cell (mock, pts, plines, A_mock). The spectrum is computed once, at (0, 0); its window (the
# eigenvalues with Im λ > 0) is followed along every edge of a 39 × 39 grid with the predictor–corrector tracker
# (tracked_scan), and the flagged cells are refined 7 times from the eigenpairs carried to their corners
# (refine_tracked), down to cells of 2e-4, enough to separate the weak EP2 pair 1e-3 apart. Expected: one swapping leaf
# at each planted EP2, an identity with winding ±2 at the planted DP, nothing else.
start = upper_window(A_mock, P2(0, 0))
println("window at (0, 0): ", length(start), " eigenvalues")

xs = ys = range(0, 1, 39)
t_scan = @elapsed cells = tracked_scan(A_mock, xs, ys, start)
t_refine = @elapsed leaves = refine_tracked(A_mock, cells, 7)
println("tracked scan: ", length(cells), " cells in ", round(t_scan, digits=1), " s; ", count(flagged, cells), " flagged, ",
    count(c -> c.lost > 0, cells), " with lost labels.  Refinement (7 levels): ", length(leaves), " leaves in ",
    round(t_refine, digits=1), " s")

# leaves against the planted points
center(c) = ((c.rect[1] + c.rect[2])/2, (c.rect[3] + c.rect[4])/2)
for p ∈ pts
    k = findfirst(c -> inside(c, p), leaves)
    found = k === nothing ? "NOT FOUND" : begin
        c = leaves[k]
        "$(verdict(c)) in a leaf of size $(round(c.rect[2] - c.rect[1], sigdigits=2)), centre off by " *
        "$(round(hypot((center(c) .- (p.x, p.y))...), sigdigits=2)), λ ≈ " *
        "$(round(isempty(c.swaps) ? c.windings[1][1] : c.swaps[1][1], digits=3))"
    end
    println("planted ", rpad(p.kind, 4), " at (", p.x, ", ", round(p.y, digits=4), "): ", found)
end
println("leaves not at a planted point: ", count(c -> !any(p -> inside(c, p), pts), leaves))

fig = Figure(size=(620, 680))
ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="39 × 39 scan by tracking (one diagonalisation)", aspect=DataAspect(),
    limits=((0, 1), (0, 1)))
draw_cells!(ax, cells)
draw_planted!(ax, pts, plines)
scan_legend(fig[2, 1])
fig


#%% Mock Liouvillian: eigenvalue trajectories around loops enclosing one or several singularities

# Needs the mock check cell (pts, A_mock). Each loop starts at its rightmost point (u = 0) and runs counterclockwise.
# For every dimer block with a planted point inside the loop, its pair of coherence eigenvalues (near λ = -γ0 - iω0) is
# tracked once around the loop, starting from one diagonalisation at u = 0. The panels show ω - ω̄ = i(λ - λ̄), with λ̄
# the pair's mean at u = 0 (ω̄ in the title): filled circle at u = 0, arrowheads at u = 1/4, 1/2, 3/4, open circle at
# u = 1. A swap shows as each branch ending on the other's start; a DP or a pair of EP2 as both branches returning,
# after winding around each other (then both run along the same curve, branch 2 dashed).
ep_w = filter(p -> p.block == 2, pts)                 # the EP2 pair 0.2 apart
ep_n = filter(p -> p.block == 1, pts)                 # the EP2 pair 1e-3 apart
dp = only(filter(p -> p.kind == :DP, pts))
mid(ps) = P2(sum(p.x for p ∈ ps)/length(ps), sum(p.y for p ∈ ps)/length(ps))
mock_loops = [
    ("one EP2",                         Circle(P2(ep_w[1].x, ep_w[1].y), 0.05)),
    ("two EP2, 0.2 apart",              Circle(mid(ep_w), 0.15)),
    ("one EP2 of the close pair",       Circle(P2(ep_n[1].x, ep_n[1].y), 3e-4)),
    ("two EP2, 1e-3 apart",             Circle(mid(ep_n), 0.02)),
    ("DP",                              Circle(P2(dp.x, dp.y), 0.05)),
    ("close pair + one EP2 + DP",       Polygon([P2(0.85, 0.2), P2(0.85, 0.68), P2(0.22, 0.68), P2(0.22, 0.2)])),
]
encloses(q::Circle, p) = hypot(p.x - q.center[1], p.y - q.center[2]) < q.ρ
encloses(q::Polygon, p) = (v = q.vertices; minimum(first, v) < p.x < maximum(first, v) && minimum(last, v) < p.y < maximum(last, v))
us = range(0, 1, 801)
n_col = 1 + maximum(length(unique(p.block for p ∈ pts if encloses(q, p))) for (_, q) ∈ mock_loops)

# one branch: its curve, start and end markers, and arrowheads showing the direction
function draw_branch!(ax, ω, j, label)
    col = (clrs[:byzantine], clrs[:selene])[j]
    lines!(ax, real(ω), imag(ω); color=col, linestyle=(j == 1 ? :solid : :dash), label)
    scatter!(ax, [real(ω[1])], [imag(ω[1])]; color=col, markersize=10)
    scatter!(ax, [real(ω[end])], [imag(ω[end])]; color=:transparent, strokecolor=col, strokewidth=1.5, markersize=16)
    for u ∈ (0.25, 0.5, 0.75)
        m = searchsortedfirst(us, u)
        scatter!(ax, [real(ω[m])], [imag(ω[m])]; marker=:rtriangle, rotation=angle(ω[m + 1] - ω[m - 1]), color=col,
            markersize=11)
    end
end

fig = Figure(size=(330*n_col, 300*length(mock_loops)))
for (row, (name, q)) ∈ enumerate(mock_loops)
    inner = filter(p -> encloses(q, p), pts)
    ax_p = Axis(fig[row, 1]; xlabel="x", ylabel="y", title=name, aspect=DataAspect(), xticks=WilkinsonTicks(3))
    δs = position.(Ref(q), us)
    lines!(ax_p, first.(δs), last.(δs); color=clrs[:overlay])
    scatter!(ax_p, [position(q, 0.)[1]], [position(q, 0.)[2]]; color=clrs[:overlay], markersize=8)
    near = q isa Circle ? filter(p -> hypot(p.x - q.center[1], p.y - q.center[2]) < 3q.ρ, pts) : pts
    scatter!(ax_p, [p.x for p ∈ near], [p.y for p ∈ near]; marker=[p.kind == :EP2 ? :star5 : :circle for p ∈ near],
        markersize=11, color=[p ∈ inner ? RGBAf(clrs[:byzantine]) : RGBAf(0, 0, 0, 0) for p ∈ near], strokecolor=clrs[:text], strokewidth=1)
    q isa Polygon && limits!(ax_p, 0, 1, 0, 1)

    F = eigen(Matrix(A_mock(position(q, 0.))))         # one diagonalisation per loop, at u = 0
    println(name)
    for (col, k) ∈ enumerate(unique(p.block for p ∈ inner))
        λp = first(filter(p -> p.block == k, inner)).λ
        idx = partialsortperm(abs.(F.values .- λp), 1:2)
        ω̄ = im*sum(F.values[idx])/2
        local ax = Axis(fig[row, col + 1]; xlabel="Re (ω - ω̄)", ylabel="Im (ω - ω̄)", xticks=WilkinsonTicks(4),
            title="block $k (" * join(string.(unique(p.kind for p ∈ inner if p.block == k)), ", ") *
                  "), ω̄ = $(round(ω̄, digits=3))")
        for (j, i) ∈ enumerate(idx)
            t = @elapsed sol, _ = track(A_mock, q, F.values[i], F.vectors[:, i])
            lands = argmin(abs.(F.values[idx] .- sol.u[end][end]))
            println("  block $k, branch $j → branch $lands  (|λ(1) - λ_start| = ",
                round(abs(sol.u[end][end] - F.values[idx[lands]]), sigdigits=2), ", ", sol.stats.naccept, " steps, ",
                round(t, digits=2), " s)")
            draw_branch!(ax, [im*sol(u)[end] - ω̄ for u ∈ us], j, "branch $j" * (lands == j ? " (returns)" : " → $lands"))
        end
        axislegend(ax; position=:rt, labelsize=9, framevisible=false)
    end
end
save(joinpath(figdir, "mock_loop_trajectories.png"), fig)
fig


#%% Mock Liouvillian: robustness of the tracking scan on randomly placed features

# Needs the packages and helpers cells only. Each instance moves every feature of the mock: a weak EP2 pair (5e-4 to
# 2e-3 apart), a second EP2 pair (0.1 to 0.4 apart, one point may fall outside the square), an off-axis DP, a real-axis
# EP line and a DP line, at random positions and slopes. Constraints: rates positive on [0, 1]², the dimers' frequency
# bands separated (no unplanted crossing between blocks), the driven qubit's Ω > 0 (a single EP line), distinct
# constant real parts. Each instance: one diagonalisation at (0, 0), tracking scan on 39 × 39, 7 refinement levels,
# compared with the planted points inside the square. Figures in figures/ (figdir).
function random_mock_blocks(rng)
    u(a, b) = a + (b - a)*rand(rng)
    ωs = shuffle(rng, [0.8, 1.25, 1.7])                  # bands ω0 ± 0.16 never overlap
    function dimer(ω0, sep)                             # EP2 pair at (x0, y0 ± sep/2); sep = 0: DP at (x0, y0)
        x0, y0, γ0 = u(0.1, 0.9), u(0.1, 0.9), u(0.08, 0.2)
        s = min(0.12, 0.9γ0/max(y0, 1 - y0))            # γa, γb = γ0 ± s(y - y0) > 0 on the square
        return DimerBlock((x0, y0), s, ω0, γ0, sep*s)
    end
    # a random line a x + b y + c = 0 through a point of [0.2, 0.8]²
    function line()
        θ, px, py = u(0, π), u(0.2, 0.8), u(0.2, 0.8)
        return cos(θ), sin(θ), -(cos(θ)*px + sin(θ)*py)
    end
    a, b, c = line()
    γq = (u(0.1, 0.15), u(0, 0.1), u(0, 0.1))           # driven qubit decay, γ ≥ 0.1
    κ = u(0.01, 0.016)                                   # Ω = γ/4 + κ (a x + b y + c) > 0: EP line where Ω = γ/4
    Ω = (γq[1]/4 + κ*c, γq[2]/4 + κ*a, γq[3]/4 + κ*b)
    ad, bd, cd = line()
    κd = u(0.2, 0.28)                                    # detuning |Δ| ≤ 0.4, below the dimers' bands
    return MockBlock[dimer(ωs[1], exp(u(log(5e-4), log(2e-3)))), dimer(ωs[2], u(0.1, 0.4)), dimer(ωs[3], 0.0),
                     DrivenQubit(Ω, γq), DetunedQubit((κd*cd, κd*ad, κd*bd), 0.8)]
end

seeds = 1:8
xs = ys = range(0, 1, 39)
summary = NamedTuple[]
overview = Figure(size=(1600, 860))
for (n, seed) ∈ enumerate(seeds)
    m = MockLiouvillian(blocks=random_mock_blocks(MersenneTwister(seed)))
    A = AffineLiouvillian(δ -> mock_matrix(setproperties(m, (x=δ[1], y=δ[2]))), 2)
    local start = upper_window(A, P2(0, 0))              # the only diagonalisation
    t = @elapsed begin
        cells_m = tracked_scan(A, xs, ys, start)
        leaves_m = refine_tracked(A, cells_m, 7)
    end
    pts_m, lines_m = mock_singularities(m)
    pts_m = filter(p -> 0 < p.x < 1 && 0 < p.y < 1, pts_m)
    found = [(k = findfirst(c -> inside(c, p), leaves_m); k === nothing ? :missed : verdict(leaves_m[k])) for p ∈ pts_m]
    correct = count(i -> found[i] == pts_m[i].kind, eachindex(pts_m))
    spurious = count(c -> !any(p -> inside(c, p), pts_m), leaves_m)
    with_lost = filter(c -> c.lost > 0, cells_m)
    off_line = count(c -> !any(l -> l.kind == :real_EP2 && crossed(c, l), lines_m), with_lost)
    push!(summary, (seed=seed, planted=length(pts_m), correct=correct, spurious=spurious, with_lost=length(with_lost),
                    lost_off_EP_line=off_line, window=length(start), time=t))
    println("seed $seed: window $(length(start)), planted inside $(length(pts_m)), correct $correct, spurious leaves $spurious, ",
        "cells with lost labels $(length(with_lost)) ($(off_line) not on the EP line), $(round(t, digits=1)) s",
        correct < length(pts_m) ? "   ← " * join(["$(p.kind) at ($(round(p.x, digits=4)), $(round(p.y, digits=4))) found as $(f)"
                                                for (p, f) ∈ zip(pts_m, found) if f != p.kind], "; ") : "")

    local fig = Figure(size=(620, 680))
    for (f, pos) ∈ ((fig, fig[1, 1]), (overview, overview[fldmod1(n, 4)...]))
        local ax = Axis(pos; title="seed $seed", aspect=DataAspect(), limits=((0, 1), (0, 1)),
            xlabel=(f === fig ? "x" : ""), ylabel=(f === fig ? "y" : ""))
        draw_cells!(ax, cells_m)
        draw_planted!(ax, pts_m, lines_m; markersize=(f === fig ? 12 : 9))
    end
    scan_legend(fig[2, 1])
    save(joinpath(figdir, "mock_tracked_scan_seed$(seed).png"), fig)
end
scan_legend(overview[3, 1:4])
save(joinpath(figdir, "mock_tracked_scan_robustness.png"), overview)
println("total: ", sum(r.correct for r ∈ summary), " / ", sum(r.planted for r ∈ summary), " planted points correctly found, ",
    sum(r.spurious for r ∈ summary), " spurious leaves, ", sum(r.lost_off_EP_line for r ∈ summary), " cells with lost labels off the EP line")
overview
