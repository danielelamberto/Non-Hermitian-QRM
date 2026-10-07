# Non-linear models: Duffing oscillator (Lindblad and Bloch–Redfield), driven optomechanics (linearised EP, monodromy, bisection).
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using NonHermitianQRM: Circle          # also exported by Makie: ours is the loop in parameter space
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, SparseArrays, Polynomials, Accessors


#%% Duffing oscillator: minimal Liouvillian gap in (g, γ), RWA_env, ω0 = 1, nB = 0

# Odd parity sector (where ⟨a⟩ lives), K = 6 slowest eigenvalues. n_fock = 15 is converged to 5 digits for g ≤ 1.
# With RWA_env the linear oscillator has no EP (modes ±ω0 - iγ for all γ): any EP here comes from the non-linearity.
n_fock = 15
â = destroy(n_fock)
idx_odd = parity_sector(n_fock, -1)

gs = range(0, 1, 61)
γs = range(0.02, 2, 61)
gap_map = zeros(length(gs), length(γs))
t = @elapsed Threads.@threads for j ∈ eachindex(γs)
    for i ∈ eachindex(gs)
        L = liouvillian(Lindbladian(â, Duffing(ω0=1., γ=γs[j], g=gs[i], nB=0., approx=[RWA_env]))...)
        gap_map[i, j] = min_gap(L, idx_odd)[1]
    end
end
println("scan: ", length(gs)*length(γs), " points in ", round(t, digits=1), " s")

fig = Figure(size=(700, 550))
ax = Axis(fig[1, 1]; xlabel="g", ylabel="γ", title="log₁₀ minimal gap, odd sector, 6 slowest eigenvalues")
hm = heatmap!(ax, gs, γs, log10.(gap_map); colormap=clrs[:byz])
Colorbar(fig[1, 2], hm)
fig


#%% Duffing oscillator: Bloch–Redfield (no RWA_env), EP line in (g, γ), ω0 = 1, nB = 0

# Indicator: |Re ω| of the dominant ⟨a⟩ mode, half the gap to its mirror partner -Re ω + i Im ω; it vanishes on the
# overdamped side, so the EP line is where it reaches 0 (γ = ω0 at g = 0, critical damping).
# Smoke tests: for g ≲ 0.5 the dominant mode is converged at n_fock = 15–20; at g ~ 1 the truncated Redfield tensor has
# spurious weakly damped, even unstable (Re λ > 0) high-frequency eigenvalues. Each point is computed at both cutoffs
# and flagged when they disagree or when any eigenvalue of the sector is unstable.
cutoffs = (15, 20)
ops = [(N, destroy(N), parity_sector(N, -1)) for N ∈ cutoffs]

gs = range(0, 0.5, 51)
γs = range(0.6, 1.4, 41)
ω_dom = zeros(ComplexF64, length(gs), length(γs), length(cutoffs))
max_reλ = fill(-Inf, length(gs), length(γs))
t = @elapsed Threads.@threads for j ∈ eachindex(γs)
    for i ∈ eachindex(gs), (k, (N, a, odd)) ∈ enumerate(ops)
        ω, reλ = dominant_mode(Redfield(a, Duffing(ω0=1., γ=γs[j], g=gs[i], nB=0., approx=[])), a, odd)
        ω_dom[i, j, k] = ω
        max_reλ[i, j] = max(max_reλ[i, j], reλ)
    end
end
half_gap = abs.(real.(ω_dom[:, :, end]))
unconverged = abs.(ω_dom[:, :, 1] .- ω_dom[:, :, 2]) .> 1e-6
unstable = max_reλ .> 0
println("scan: ", length(gs)*length(γs), " points × ", length(cutoffs), " cutoffs in ", round(t, digits=1), " s;  ",
    count(unconverged), " points not converged in n_fock, ", count(unstable), " unstable")

fig = Figure(size=(750, 550))
ax = Axis(fig[1, 1]; xlabel="g", ylabel="γ", title="|Re ω| of the dominant ⟨a⟩ mode (Redfield, n_fock = $(cutoffs[end]))")
hm = heatmap!(ax, gs, γs, half_gap; colormap=clrs[:byz])
Colorbar(fig[1, 2], hm)
# points where the two cutoffs disagree (crosses) or the sector is unstable (red)
for (mask, marker, color) ∈ ((unconverged, :xcross, clrs[:text]), (unstable, :circle, clrs[:byzantine]))
    bad = findall(mask)
    isempty(bad) || scatter!(ax, [gs[I[1]] for I ∈ bad], [γs[I[2]] for I ∈ bad]; marker, color, markersize=6)
end
scatter!(ax, [0.], [1.]; marker=:star5, color=clrs[:text], markersize=14)     # g = 0: critical damping γ = ω0
fig


#%% Driven optomechanics: setup (linearised EP, displaced-frame Liouvillian)

# Linearised fluctuations ≡ BosonDimer with ωa = -(Δ + g x̄) and coupling 2g|α| (linearised_dimer): EP at resonance and
# 2g|α| = |γa - γb| (linearised_EP). The full model keeps the non-linear term -g d†d (e + e†). It is written in the
# displaced frame a = α + d, b = β + e (reference at the linearised EP), where n_fock = 6 per mode is converged to ~1e-7.
# The next two cells work in the plane P = (Δ, F/F_EP), where L is affine (H is linear in Δ and F).
mdl = Optomech(ωb=1., γa=0.2, γb=0.05, g=0.1, nBa=0., nBb=0., approx=[RWA_env])
Δ_EP, F_EP, n_branches = linearised_EP(mdl)
point(prm_P) = setproperties(mdl, (Δ=prm_P[1], F=prm_P[2]*F_EP))
println("linearised EP: Δ = ", round(Δ_EP, digits=5), ", F = ", round(F_EP, digits=5), ", ", n_branches, " classical branch(es)")

Na, Nb = 6, 6
â = destroy(Na) ⊗ eye(Nb)
b̂ = eye(Na) ⊗ destroy(Nb)
α, β = classical_displacement(point(P2(Δ_EP, 1.0)))
âd, b̂d = â + α*one(â), b̂ + β*one(b̂)        # displaced frame: the model functions see a = α + d, b = β + e
A = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(âd, b̂d, point(prm_P))...).data, 2)

# starting targets at P = (Δ, F/F_EP): the linearised modes, λ = -iω (≈ 0.01 from the full ones at g = 0.1)
linearised_targets(prm_P) = -im .* filter(r -> real(r) > 0, roots_sorted(linearised_dimer(point(prm_P))))


#%% Driven optomechanics: monodromy around the linearised EP in the (Δ, F) plane

# Needs the setup cell. Both fluctuation modes are tracked around a large and a small circle centred on the linearised EP,
# and around a control circle away from it.
om_loops = [
    ("ρ = 0.05 around the linearised EP", Circle(P2(Δ_EP, 1.0), 0.05)),
    ("ρ = 0.01 around the linearised EP", Circle(P2(Δ_EP, 1.0), 0.01)),
    ("control, centre Δ_EP + 0.3",        Circle(P2(Δ_EP + 0.3, 1.0), 0.05)),
]
us = range(0, 1, 401)

fig = Figure(size=(1100, 900))
ax_p = Axis(fig[1, 1]; xlabel="Δ", ylabel="F / F_EP", title="loops in parameter space")
scatter!(ax_p, [Δ_EP], [1.0]; marker=:star5, markersize=16, color=clrs[:text], label="linearised EP")
for (j, (name, q)) ∈ enumerate(om_loops)
    prm_Ps = position.(Ref(q), us)
    lines!(ax_p, first.(prm_Ps), last.(prm_Ps); color=Cycled(j + 2), label=name)
    local ax = Axis(fig[fldmod1(j + 1, 2)...]; xlabel="Re ω", ylabel="Im ω", title=name)
    pairs0 = eigenpairs_near(A(position(q, 0.)), linearised_targets(position(q, 0.)))
    println(name)
    for (k, (λ0, r0)) ∈ enumerate(pairs0)
        local t = @elapsed sol, info = track(A, q, λ0, r0)
        lands = argmin(abs.(first.(pairs0) .- sol.u[end][end]))
        println("  mode $k → mode $lands,  |λ(1) - λ_start| = ", round(abs(sol.u[end][end] - first(pairs0[lands])), sigdigits=2),
            ",  ", sol.stats.naccept, " steps, ", info.factorisations, " LU, ", round(t, digits=1), " s")
        ω = [im*sol(u)[end] for u ∈ us]
        lines!(ax, real(ω), imag(ω); color=Cycled(k), label="mode $k" * (lands == k ? " (returns)" : " → mode $lands"))
        scatter!(ax, [real(im*λ0)], [imag(im*λ0)]; color=Cycled(k), marker=:star5, markersize=14)
    end
    axislegend(ax; position=:rb, labelsize=10)
end
ylims!(ax_p, 0.93, 1.2)                     # room for the legend above the loops
axislegend(ax_p; position=:ct, labelsize=10)
fig


#%% Driven optomechanics: locating the EP by bisection of rectangles with edge reuse

# Needs the setup cell. The starting square contains the ρ = 0.05 circle, which swaps the pair (previous cell).
rect0 = (Δ_EP - 0.05, Δ_EP + 0.05, 0.95, 1.05)
t = @elapsed rect, hist, tracked_lines = ep_bisect(A, rect0, linearised_targets; depth=16)
x0, x1, y0, y1 = rect
Δ_c, f_c = (x0 + x1)/2, (y0 + y1)/2
println("bisection: ", length(hist) - 1, " levels, ", length(tracked_lines), " tracked lines, ", round(t, digits=1), " s")
println("EP of the full model: Δ = ", round(Δ_c, digits=5), " ± ", round((x1 - x0)/2, sigdigits=2),
    ",  F = ", round(f_c*F_EP, digits=5), " ± ", round((y1 - y0)/2*F_EP, sigdigits=2))
println("shift from the linearised EP: δΔ = ", round(Δ_c - Δ_EP, sigdigits=3), ",  δF = ", round((f_c - 1)*F_EP, sigdigits=3),
    " (", round(100(f_c - 1), sigdigits=3), " %)")

# midpoint of the pair at the corner (x0, y0) of a box: shift for pair_near (the pair is close near the EP)
box_midpoint(r) = pair_midpoint(tracked_lines, P2(r[1], r[3]), P2(r[2], r[3]))

# Independent confirmation by direct loops of radius 2 × the box diagonal: around the box (swap) and next to it (none)
ρ = 2hypot(x1 - x0, y1 - y0)
for (name, q) ∈ (("around the box", Circle(P2(Δ_c, f_c), ρ)), ("control, shifted by 3ρ", Circle(P2(Δ_c + 3ρ, f_c), ρ)))
    pairs0 = pair_near(A(position(q, 0.)), box_midpoint(rect))
    sol, _ = track(A, q, pairs0[1]...)
    println("loop ", name, " (ρ = ", round(ρ, sigdigits=2), "): mode 1 → mode ", argmin(abs.(first.(pairs0) .- sol.u[end][end])))
end

# Square-root signature: gap of the pair at the box centres against the box size
sizes = [hypot(r[2] - r[1], r[4] - r[3]) for r ∈ hist]
gaps = map(hist) do r
    (λ1, _), (λ2, _) = pair_near(A(P2((r[1] + r[2])/2, (r[3] + r[4])/2)), box_midpoint(r))
    abs(λ1 - λ2)
end

fig = Figure(size=(1200, 450))
ax1 = Axis(fig[1, 1]; xlabel="Δ", ylabel="F / F_EP", title="bisection boxes")
ax2 = Axis(fig[1, 2]; xlabel="Δ", ylabel="F / F_EP", title="zoom on the last levels")
for (k, r) ∈ enumerate(hist), ax ∈ (ax1, ax2)
    lines!(ax, [r[1], r[2], r[2], r[1], r[1]], [r[3], r[3], r[4], r[4], r[3]]; color=k, colorrange=(1, length(hist)), colormap=clrs[:byz])
end
for ax ∈ (ax1, ax2)
    scatter!(ax, [Δ_EP], [1.0]; marker=:star5, markersize=14, color=clrs[:text], label="linearised EP")
    scatter!(ax, [Δ_c], [f_c]; marker=:xcross, markersize=12, color=clrs[:byzantine], label="full EP (box centre)")
end
zr = hist[min(7, end)]
limits!(ax2, zr[1], zr[2], zr[3], zr[4])
axislegend(ax1; position=:rt, labelsize=10)
ax3 = Axis(fig[1, 3]; xscale=log10, yscale=log10, xlabel="box diagonal", ylabel="|λ₁ - λ₂| at the box centre", title="gap ∝ √size")
scatter!(ax3, sizes, gaps)
lines!(ax3, sizes, gaps[end]*sqrt.(sizes ./ sizes[end]); linestyle=:dash, color=clrs[:overlay], label="∝ √size")
axislegend(ax3; position=:lt)
fig
