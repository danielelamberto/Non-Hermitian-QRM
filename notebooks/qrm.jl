# Quantum Rabi model: Jaynes–Cummings limit with local Lindblad dissipators (tower of EPs towards g → 0), consistency
# with the coworker's functions_QRM.jl.
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using NonHermitianQRM: Circle          # also exported by Makie: ours is the loop in parameter space
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, SparseArrays, Accessors
figdir = joinpath(pkgdir(NonHermitianQRM), "figures")
mkpath(figdir)

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


#%% QRM: consistency with the coworker's functions_QRM.jl

# Our QRM type follows the coworker's conventions (dipole gauge, photon ⊗ qubit, energy-decay rates, temperatures):
# its Hamiltonian must equal H_QRM_sb and its DressedLiouvillian gen_liouvillian_QRM.
include(joinpath(pkgdir(NonHermitianQRM), "functions_QRM.jl"))
using .MyFunctions: H_QRM_sb, gen_liouvillian_QRM
Nc = 8
â, σ̂ = qrm_operators(Nc)
qrm = QRM(ωa=1.0, ωb=1.1, g=0.3, γa=0.1, γb=0.02, Ta=0.1, Tb=0.2, ε=0.05)
println("‖H - H_QRM_sb‖ = ", norm(Hamiltonian(â, σ̂, qrm).data - H_QRM_sb(0.3, 0.05, (ωa=1.0, ωb=1.1, Nc=Nc)).data))
_, _, L_ours = DressedLiouvillian(â, σ̂, qrm; N_trunc=10, σ_filter=10.0)
_, _, L_his = gen_liouvillian_QRM(0.3, (ωa=1.0, ωb=1.1, Nc=Nc), [0.1, 0.02]; T=[0.1, 0.2], ε=0.05, N_trunc=10, σ_filter=10.0)
println("dressed Liouvillian vs gen_liouvillian_QRM: spectra ", spectrum_distance(eigvals(Matrix(L_ours.data)), eigvals(Matrix(L_his.data))))


#%% Jaynes–Cummings with RWA_env: closed form and the tower of EPs of H_eff

# At T = 0 the jumps only lower N = a†a + σ+σ-, so L is block-triangular: its eigenvalues are -i(εᵢ - conj(εⱼ)) over
# the eigenvalues ε of H_eff in each N-manifold (jc_effective_energies). Manifold n has an EP2 at ωb = ωa,
# g_n = |γa - γb|/(4√n): a tower accumulating at g → 0. Check of the closed form (full L and excitation sectors), then
# the splitting ε₊ - ε₋ of the first manifolds at resonance against g: purely imaginary below g_n (overdamped),
# real above.
Nc = 12
â, σ̂ = qrm_operators(Nc)
jc = QRM(ωa=1.0, ωb=0.97, g=0.02, γa=0.1, γb=0.0, approx=[RWA_env, RWA_coupling])
L = liouvillian(Lindbladian(â, σ̂, jc)...).data
println("full L ($(size(L, 1)) states) vs closed form: ", spectrum_distance(eigvals(Matrix(L)), jc_spectrum(jc, Nc)))
for k ∈ 0:2
    idx = excitation_sector(Nc, k)
    println("sector k = $k ($(length(idx)) states) vs closed form: ",
        spectrum_distance(eigvals(Matrix(L[idx, idx])), jc_spectrum(jc, Nc; k)))
end

gs = range(1e-4, 0.035, 2000)
fig = Figure(size=(1000, 420))
ax_re = Axis(fig[1, 1]; xlabel="g", ylabel="Re (ε₊ - ε₋)", title="JC manifolds at ωb = ωa: splitting of H_eff")
ax_im = Axis(fig[1, 2]; xlabel="g", ylabel="|Im (ε₊ - ε₋)|")
for n ∈ 1:6
    split = map(gs) do g
        e = [ε for (N, ε) ∈ jc_effective_energies(setproperties(jc, (ωb=1.0, g=g)), Nc) if N == n]
        e[1] - e[2]
    end
    lines!(ax_re, gs, abs.(real.(split)); color=Cycled(n), label="n = $n")
    lines!(ax_im, gs, abs.(imag.(split)); color=Cycled(n))
    vlines!(ax_re, [jc_EP(jc, n)[2]]; color=Cycled(n), linestyle=:dash)
end
axislegend(ax_re; position=:lt)
save(joinpath(figdir, "jc_tower_splitting.png"), fig)
fig


#%% Jaynes–Cummings with RWA_env: tracking scan of the tower in the plane (g, ωb)

# Needs the previous cell (Nc, â, σ̂, jc). All 44 eigenvalues of the sector k = 1 (where ⟨a⟩ and ⟨σ-⟩ live) are tracked
# from one diagonalisation at the lower-left corner, along a 39 × 39 grid, and the flagged cells refined 8 times.
# Expected: one leaf at each EP_n inside the window (n = 1 … Nc - 1), with 4 swaps: the EP of manifold n appears in the
# elements (n, n-1) (two choices in manifold n - 1) and (n+1, n) (two choices in manifold n + 1); 3 swaps for n = 1
# (manifold 0 is one state) and n = Nc - 1 (the truncated manifold Nc is one state). The grid is generic: no node row
# lies on the resonance ωb = ωa.
idx = excitation_sector(Nc, 1)
A_jc = AffineLiouvillian(δ -> liouvillian(Lindbladian(â, σ̂, setproperties(jc, (g=δ[1], ωb=δ[2])))...).data[idx, idx], 2)
gs, ωbs = range(0.0061, 0.0303, 39), range(0.9713, 1.0291, 39)
F = eigen(Matrix(A_jc(P2(gs[1], ωbs[1]))))                # the only diagonalisation
start = collect(zip(F.values, eachcol(F.vectors)))
t_scan = @elapsed cells = tracked_scan(A_jc, gs, ωbs, start)
t_refine = @elapsed leaves = refine_tracked(A_jc, cells, 8)
println(length(start), " eigenvalues tracked; scan ", round(t_scan, digits=1), " s, ", count(flagged, cells), " flagged cells, ",
    count(c -> c.lost > 0, cells), " with lost labels; refinement ", round(t_refine, digits=1), " s, ", length(leaves), " leaves")

inside(c, g, ωb) = c.rect[1] ≤ g ≤ c.rect[2] && c.rect[3] ≤ ωb ≤ c.rect[4]
tower = [(n, jc_EP(jc, n)...) for n ∈ 1:Nc - 1 if gs[1] < jc_EP(jc, n)[2] < gs[end]]
for (n, ωb, g, ε) ∈ tower
    k = findfirst(c -> inside(c, g, ωb), leaves)
    println("EP n = $n at g = ", round(g, digits=5), ": ", k === nothing ? "MISSED" :
        "$(length(leaves[k].swaps)) swaps, $(length(leaves[k].windings)) windings, leaf of size $(round(leaves[k].rect[2] - leaves[k].rect[1], sigdigits=2))")
end
println("leaves away from the tower: ", count(c -> !any(e -> inside(c, e[3], e[2]), tower), leaves))

rect_poly(r) = Point2f[(r[1], r[3]), (r[2], r[3]), (r[2], r[4]), (r[1], r[4])]
fig = Figure(size=(1100, 520))
ax = Axis(fig[1, 1]; xlabel="g", ylabel="ωb", title="JC, sector k = 1: tracking scan (39 × 39)")
ax_z = Axis(fig[1, 2]; xlabel="g", ylabel="ωb - 1", title="refined leaves (centres; leaves of 2.5e-6)")
for c ∈ cells
    flagged(c) && poly!(ax, rect_poly(c.rect); color=clrs[:byzantine])
end
scatter!(ax, [e[3] for e ∈ tower], [e[2] for e ∈ tower]; marker=:star5, markersize=12, color=:transparent,
    strokecolor=clrs[:text], strokewidth=1.2)
centres = [((c.rect[1] + c.rect[2])/2, (c.rect[3] + c.rect[4])/2) for c ∈ leaves]
scatter!(ax_z, first.(centres), last.(centres) .- 1; marker=:xcross, markersize=12, color=clrs[:byzantine], label="leaf centre")
scatter!(ax_z, [e[3] for e ∈ tower], [e[2] - 1 for e ∈ tower]; marker=:star5, markersize=14, color=:transparent,
    strokecolor=clrs[:text], strokewidth=1.2, label="EP_n (closed form)")
ylims!(ax_z, -1e-5, 1e-5)
axislegend(ax_z; position=:rt)
save(joinpath(figdir, "jc_tower_scan.png"), fig)
fig


#%% Jaynes–Cummings with RWA_env: refinement of the scan around one EP of the tower

# Needs the previous cell (A_jc, cells, jc, inside). The refinement of the flagged coarse cell containing EP_n is replayed
# level by level (refine_tracked keeps only the leaves): at each level the cell is split into 2 × 2 sub-cells, scanned
# from the eigenpairs at its corner, and the flagged sub-cell is kept.
# Top: the sub-cells of every level on the same axes (colour = level, flagged sub-cells filled), and a zoom on the last
# levels. Bottom: the pairs swapped around the flagged cell of level `loop_level`, each eigenvalue followed once around
# that rectangle (counterclockwise from its lower-left corner), shown as ω - ω̄ = i(λ - λ̄) with λ̄ the pair's mean at
# the start: filled circle at the start, arrowheads at 1/4, 1/2, 3/4 of the loop, open circle at the end, on the
# other branch's start.
n_ep = 2
ωb_ep, g_ep, _ = jc_EP(jc, n_ep)
levels = [only(filter(c -> inside(c, g_ep, ωb_ep), cells))]       # levels[k + 1]: the flagged cell of level k
subs = Vector{Vector{ScanCell}}()
for level ∈ 1:8
    x0, x1, y0, y1 = levels[end].rect
    sub = tracked_scan(A_jc, range(x0, x1, 3), range(y0, y1, 3), levels[end].corner)
    push!(subs, sub)
    push!(levels, only(filter(flagged, sub)))
end

# a rectangle in coordinates relative to the EP
shifted(r) = Point2f[(r[1] - g_ep, r[3] - ωb_ep), (r[2] - g_ep, r[3] - ωb_ep), (r[2] - g_ep, r[4] - ωb_ep), (r[1] - g_ep, r[4] - ωb_ep)]
level_colours = Makie.resample_cmap(clrs[:byz], 12)[5:12]
loop_level = 2
loop_cell = levels[loop_level + 1]

fig = Figure(size=(1400, 1050))
top = fig[1, 1] = GridLayout()
ax_all = Axis(top[1, 1]; title="sub-cells of levels 1–8 (colour: level)", xlabel="g - g_$n_ep", ylabel="ωb - ωa")
ax_zoom = Axis(top[1, 2]; title="zoom on levels 5–8", xlabel="g - g_$n_ep", ylabel="ωb - ωa")
for (ax, ks) ∈ ((ax_all, 1:8), (ax_zoom, 5:8)), k ∈ ks, c ∈ subs[k]
    poly!(ax, shifted(c.rect); color=(flagged(c) ? (level_colours[k], 0.35) : :transparent), strokecolor=level_colours[k],
        strokewidth=1.5)
end
lines!(ax_all, [shifted(loop_cell.rect); shifted(loop_cell.rect)[1:1]]; color=clrs[:helios], linewidth=2.5,
    label="loop of the bottom panels (level $loop_level)")
r5 = levels[5].rect
limits!(ax_zoom, r5[1] - g_ep, r5[2] - g_ep, r5[3] - ωb_ep, r5[4] - ωb_ep)
for ax ∈ (ax_all, ax_zoom)
    scatter!(ax, [0.0], [0.0]; marker=:star5, markersize=16, color=clrs[:helios], strokecolor=clrs[:base], strokewidth=1,
        label="EP (closed form)")
end
axislegend(ax_all; position=:rb, merge=true)
Colorbar(top[1, 3]; colormap=cgrad(level_colours; categorical=true), limits=(0.5, 8.5), ticks=1:8, label="level")

# the swapped pairs around the loop cell: labels at its lower-left corner, tracked once around it
loop = Polygon(P2.(Tuple.(shifted(loop_cell.rect))) .+ Ref(P2(g_ep, ωb_ep)))
corner = loop_cell.corner
us = range(0, 1, 801)
bottom = fig[2, 1] = GridLayout()
for (col, (λa, λb)) ∈ enumerate(loop_cell.swaps)
    labels = [argmin(abs.([p[1] for p ∈ corner] .- λ)) for λ ∈ (λa, λb)]
    ω̄ = im*(λa + λb)/2
    local ax = Axis(bottom[1, col]; title="ω̄ = $(round(ω̄, digits=3))", xlabel="Re (ω - ω̄)", ylabel=(col == 1 ? "Im (ω - ω̄)" : ""),
        xticks=WilkinsonTicks(2), yticks=WilkinsonTicks(3), aspect=DataAspect())
    for (j, l) ∈ enumerate(labels)
        sol, _ = track(A_jc, loop, corner[l]...)
        ω = [im*sol(u)[end] - ω̄ for u ∈ us]
        colour = (clrs[:byzantine], clrs[:selene])[j]
        lines!(ax, real(ω), imag(ω); color=colour, linestyle=(j == 1 ? :solid : :dash))
        scatter!(ax, [real(ω[1])], [imag(ω[1])]; color=colour, markersize=10)
        scatter!(ax, [real(ω[end])], [imag(ω[end])]; color=:transparent, strokecolor=colour, strokewidth=1.5, markersize=16)
        for u ∈ (0.25, 0.5, 0.75)
            m = searchsortedfirst(us, u)
            scatter!(ax, [real(ω[m])], [imag(ω[m])]; marker=:rtriangle, rotation=angle(ω[m + 1] - ω[m - 1]), color=colour,
                markersize=11)
        end
    end
end
Label(bottom[0, 1:length(loop_cell.swaps)], "the $(length(loop_cell.swaps)) pairs swapped around the level-$loop_level cell " *
    "(highlighted above), followed once around it"; fontsize=14, tellwidth=false)
rowsize!(fig.layout, 2, Relative(0.4))
Label(fig[0, 1], "JC, EP n = $n_ep at (g, ωb) = ($(round(g_ep, digits=6)), $ωb_ep): refinement of the coarse cell"; fontsize=16, tellwidth=false)
save(joinpath(figdir, "jc_refinement_EP$(n_ep).png"), fig)
fig
