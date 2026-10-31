# Bloch–Redfield master equation of the quantum Rabi model: our assembly (redfield_liouvillian) against QuantumToolbox's,
# its fixed sparsity pattern and parity sectors, its spectrum against the Lindblad and dressed master equations, its
# smoothness in the parameters at T = 0 and small T > 0, and its tracking as a finite-difference family
# (FDLiouvillian): the check of the step h, the error of the derivative against h, a swap around an EP, and the
# first EP lines of the full QRM tower with Redfield baths, against the Lindblad ones (as in ep_lines.jl).
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using NonHermitianQRM: ohmic
using NonHermitianQRM: Circle          # also exported by Makie: ours is the loop in parameter space
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, SparseArrays, Accessors, JLD2
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

# odd parity sector of the QRM (N = a†a + σ+σ-, |i⟩⟨j| with Nᵢ - Nⱼ odd): ⟨a⟩ and ⟨σ-⟩ live there
odd_sector(Nc) = sort(reduce(vcat, excitation_sector(Nc, k) for k ∈ -Nc:Nc if isodd(k)))


#%% Our assembly against QuantumToolbox's bloch_redfield_tensor

# redfield_liouvillian builds Λ = Σᵢⱼ S(Eᵢ - Eⱼ)|i⟩⟨i|X|j⟩⟨j| for each bath and assembles L in the Fock basis. Same
# matrix as bloch_redfield_tensor(...; sec_cutoff=-1, fock_basis=Val(true)) up to round-off, but stored on a pattern
# fixed by the coupling operators (redfield_pattern). QuantumToolbox drops the entries below its tidyup tolerance, so
# its pattern changes with the parameters: sparser where a symmetry or a zero rate empties blocks, as full as ours
# where nothing does (ε ≠ 0). Ours keeps the blocks that parity empties at ε = 0 (holding round-off, see the next
# cell), and is a few times slower to build for now (assembly by sparse sums, then copy onto the pattern).
Nc = 10
â, σ̂ = qrm_operators(Nc)
qrm = QRM(ωa=1.0, ωb=0.9, g=0.3, γa=0.05, γb=0.02, Ta=0.02, Tb=0.02)
bths(q) = [(â + â', ohmic(q.γa/q.ωa, q.Ta)), (σ̂ + σ̂', ohmic(q.γb/q.ωb, q.Tb))]
qt(q) = bloch_redfield_tensor(Hamiltonian(â, σ̂, q), bths(q); sec_cutoff=-1, fock_basis=Val(true))
L, R = Redfield(â, σ̂, qrm), qt(qrm)
println("‖L - L_QT‖/‖L_QT‖ = ", norm(L.data - R.data)/norm(R.data))
println("stored entries: ours ", nnz(L.data), ", QuantumToolbox ", nnz(R.data), " (of ", length(L.data), ")")
println("time: ours ", @elapsed(Redfield(â, σ̂, qrm)), " s, QuantumToolbox ", @elapsed(qt(qrm)), " s")
for q ∈ (@set(qrm.g = 0.0), @set(qrm.γa = 0.0), @set(qrm.Ta = 0.0), @set(qrm.ε = 0.1))
    Lq = Redfield(â, σ̂, q).data
    same = Lq.colptr == L.data.colptr && rowvals(Lq) == rowvals(L.data)
    println("g = $(q.g), γa = $(q.γa), Ta = $(q.Ta), ε = $(q.ε): same pattern ", same, ", QuantumToolbox's ", nnz(qt(q).data))
end


#%% Parity sectors

# The baths couple through a + a† and σx, both odd under the parity exp(iπN): at ε = 0, L is block diagonal in the
# parity of Nᵢ - Nⱼ, as the Lindbladian. Λ is built from eigenvectors that need not respect parity (exactly degenerate
# levels of opposite parity can mix), but Λ itself does not depend on them: the leak between sectors is round-off.
odd = odd_sector(Nc)
even = setdiff(1:size(L, 1), odd)
for q ∈ (qrm, @set(qrm.ε = 0.1))
    Lq = Redfield(â, σ̂, q).data
    println("ε = $(q.ε): ‖L[even, odd]‖/‖L‖ = ", norm(Lq[even, odd])/norm(Lq), ", ‖L[odd, even]‖/‖L‖ = ", norm(Lq[odd, even])/norm(Lq))
end


#%% Spectra: Lindblad, Redfield and dressed master equations

# Odd-sector eigenvalues near Im λ = -ω (the ⟨a⟩, ⟨σ-⟩ family), for the three master equations of the QRM. Weak
# coupling: the local Lindblad dissipators are close to Redfield (the bare levels are the eigenstates). Strong
# coupling: they are not, while the dressed master equation (RWA in the bath, in the eigenbasis of H) stays close to
# Redfield. The dressed L is written in the eigenbasis of H: rotated back to the Fock basis (ρ = UρU†, i.e.
# vec ρ ↦ (conj(U) ⊗ U) vec ρ, no truncation), it is block diagonal in parity like the others.
dressed_fock(q) = ((E, U, L) = DressedLiouvillian(â, σ̂, q); S = kron(conj(U.data), U.data); S*Matrix(L.data)*S')
fig = Figure(size=(1000, 450))
for (col, g) ∈ enumerate((0.05, 0.4))
    q = @set qrm.g = g
    λs = [
        ("Lindblad (local)", :circle, eigvals(Matrix(liouvillian(Lindbladian(â, σ̂, @set(q.approx = [RWA_env]))...).data[odd, odd]))),
        ("Redfield", :xcross, eigvals(Matrix(Redfield(â, σ̂, q).data[odd, odd]))),
        ("dressed", :cross, eigvals(dressed_fock(q)[odd, odd])),
    ]
    window(λ) = filter(l -> real(l) > -0.15 && -1.6 < imag(l) < -0.4, λ)
    ax = Axis(fig[1, col]; xlabel="Re λ", ylabel="Im λ", title="g = $g, odd sector near Im λ = -ω")
    for (k, (name, mk, λ)) ∈ enumerate(λs)
        scatter!(ax, real.(window(λ)), imag.(window(λ)); marker=mk, markersize=(11, 16, 9)[k], label=name,
                 color=clrs[:dark][(:byzantine, :helios, :minthe)[k]])
    end
    col == 1 && axislegend(ax; position=:lb)
    λR = partialsort(window(λs[2][3]), 1:4; by=real, rev=true)          # the four slowest Redfield eigenvalues
    println("g = $g: four slowest, distance to Redfield: Lindblad ", spectrum_distance(λR, λs[1][3]),
            ", dressed ", spectrum_distance(λR, λs[3][3]))
end
save(joinpath(figdir, "redfield_spectra.png"), fig)
fig


#%% Smoothness in g: level crossings at T = 0 and T > 0

# Finite-difference tracking needs L smooth in the parameters. Λ depends on H through S(Eᵢ - Eⱼ): at T = 0, S(ω) = κω
# for ω > 0 and 0 below, a kink at ω = 0, so L has a kink in g wherever two levels coupled by the baths cross: in the
# full QRM, levels of opposite parity (Juddian points). For T > 0, S is analytic, and the kink is rounded over a width
# ~ T in Eᵢ - Eⱼ. Top: the lowest levels of H against g, by parity. Bottom: the curvature of L, by the second
# difference ‖L(g + h) - 2L(g) + L(g - h)‖/h². At T = 0 it spikes, with a height ~ 1/h, wherever a crossing falls
# within [g - h, g + h]: also crossings of the higher levels (not drawn, all 2Nc levels enter Λ), and with h smaller
# than the grid step fewer crossings are hit, but higher. At T > 0 the spikes become bumps of height ~ κ/T and width
# ~ T, independent of h once h ≪ T: L is smooth, with a curvature that grows as T decreases.
Nc = 16
â, σ̂ = qrm_operators(Nc)
base = QRM(ωa=1.0, ωb=0.6, γa=0.05, γb=0.02)
gs = range(0.05, 0.8, 301)
# the four lowest levels of each parity block of H (N even, N odd): H is block diagonal at ε = 0
blocks = [filter(i -> NonHermitianQRM.excitation_number(i) % 2 == p, 1:2Nc) for p ∈ (0, 1)]
levels = [reduce(hcat, eigvals(Hermitian(Matrix(Hamiltonian(â, σ̂, @set(base.g = g)).data[b, b])))[1:4] for g ∈ gs)
          for b ∈ blocks]
curvature(T, h) = map(gs) do g
    L_at(x) = Redfield(â, σ̂, setproperties(base, (g=x, Ta=T, Tb=T))).data
    norm(L_at(g + h) - 2L_at(g) + L_at(g - h))/h^2
end
curves = [(T, h, curvature(T, h)) for (T, h) ∈ ((0.0, 1e-3), (0.0, 1e-4), (0.01, 1e-4), (0.05, 1e-4))]

fig = Figure(size=(900, 700))
ax_E = Axis(fig[1, 1]; ylabel="Eᵢ", title="QRM levels (ωb = $(base.ωb)): even, odd parity")
for (p, lv) ∈ enumerate(levels), i ∈ 1:4
    lines!(ax_E, gs, lv[i, :]; color=clrs[:dark][(:byzantine, :helios)[p]])
end
ax_c = Axis(fig[2, 1]; xlabel="g", ylabel="‖Δ²L‖/h²", yscale=log10, title="Curvature of the Redfield L along g")
for (k, (T, h, c)) ∈ enumerate(curves)
    lines!(ax_c, gs, c; label="T = $T, h = $h", color=clrs[:dark][(:subtext, :hyacinth, :minthe, :selene)[k]],
           linestyle=k == 1 ? :dash : :solid)
end
axislegend(ax_c; position=:lt)
linkxaxes!(ax_E, ax_c)
save(joinpath(figdir, "redfield_smoothness.png"), fig)
fig


#%% Finite-difference family: the step h, and tracking around an EP

# Redfield is not affine in (g, ωb): FDLiouvillian rebuilds L at every evaluation and differentiates it by central
# differences along the path's velocity. Its constructor checks h at the reference point and at check_at (D(h) and
# D(h/2) must agree to rtol = 1e-6 relative to the derivative), here near the JC EP₁ (g = γa/4, ωb = ωa), where weak
# coupling Redfield should have an EP close to Lindblad's.
# Left: ‖D(h) - D(h/2)‖/‖D‖ along g at (0.3, 0.9) against h, the error estimate of the check: truncation ∝ h² above
# the optimum h ≈ 1e-5, round-off ∝ eps‖L‖/h below. Right: the pair nearest λ = -γa/4 - iω tracked around a circle
# of radius 0.01 in (g, ωb), around the EP (it swaps: each eigenvalue ends where the other started) and beside it (each
# comes back), with dense eigenvalues at a few points of the loops.
Nc = 5
â, σ̂ = qrm_operators(Nc)
odd = odd_sector(Nc)
q = QRM(ωa=1.0, ωb=1.0, γa=0.1, γb=0.0, Ta=0.02, Tb=0.02)
L_gω(prm_P) = Redfield(â, σ̂, setproperties(q, (g=prm_P[1], ωb=prm_P[2]))).data[odd, odd]
A = FDLiouvillian(L_gω, P2(0.3, 0.9); check_at=P2(0.025, 1.0))
println("FDLiouvillian: h = $(A.h) validated, $(size(A.L0, 1)) states, $(nnz(A.L0)) stored entries")

prm_P, e_g = P2(0.3, 0.9), P2(1.0, 0.0)
D(h) = derivative(FDLiouvillian(L_gω, prm_P; h, check=false), prm_P, e_g)
hs = 10.0 .^ range(-11, -1.5, 20)
δs = [norm(D(h) - D(h/2))/norm(D(h)) for h ∈ hs]

fig = Figure(size=(1000, 430))
ax_h = Axis(fig[1, 1]; xlabel="h", ylabel="‖D(h) - D(h/2)‖/‖D‖", xscale=log10, yscale=log10,
            title="Central difference along g: error estimate")
scatterlines!(ax_h, hs, δs; color=clrs[:dark][:byzantine], label="Redfield QRM")
above, below = filter(≥(1e-6), hs), filter(≤(1e-4), hs)                  # the guides, each in its regime
lines!(ax_h, above, δs[end]*(above/hs[end]).^2; color=clrs[:dark][:subtext], linestyle=:dash, label="∝ h²")
lines!(ax_h, below, δs[1]*hs[1]./below; color=clrs[:dark][:subtext], linestyle=:dot, label="∝ 1/h")
hlines!(ax_h, [1e-6]; color=clrs[:dark][:helios], label="rtol of the check")
vlines!(ax_h, [A.h]; color=clrs[:dark][:minthe], label="default h")
axislegend(ax_h; position=:lb)

λ_EP = -q.γa/4 - 1.0im
ax_λ = Axis(fig[1, 2]; xlabel="Re λ", ylabel="Im λ", title="Pair near λ = -γa/4 - iω around a loop")
for (k, (centre, name)) ∈ enumerate(((P2(0.025, 1.0), "around EP₁"), (P2(0.06, 1.0), "beside")))
    loop = Circle(centre, 0.01)
    pair = pair_near(A(position(loop, 0.0)), λ_EP)
    ends = map(pair) do (λ0, r0)
        sol, diag = track(A, loop, λ0, r0)
        us = range(0, 1, 400)
        lines!(ax_λ, real.(sol.(us; idxs=length(r0) + 1)), imag.(sol.(us; idxs=length(r0) + 1));
               color=clrs[:dark][(:byzantine, :hyacinth)[k]], label=name)
        println(name, ": ", diag.corrections, " steps, max residual before correction ", diag.max_residual)
        sol.u[end][end]
    end
    swapped = argmin(abs.(first.(pair) .- ends[1])) == 2
    println(name, ": swap ", swapped, ", start ", first.(pair), ", end ", ends)
    for u ∈ range(0, 1, 9)[1:end-1]
        λ = eigvals(Matrix(A(position(loop, u))))
        λ = partialsort(λ, 1:2; by=l -> abs(l - λ_EP))
        scatter!(ax_λ, real.(λ), imag.(λ); color=clrs[:dark][:text], markersize=6)
    end
end
axislegend(ax_λ; position=:lt, unique=true)
save(joinpath(figdir, "redfield_fd_tracking.png"), fig)
fig


#%% EP lines of the full QRM with Redfield baths: seeds

# The setup of ep_lines.jl (full coupling): odd sector, Nc = 10, ωa,b = ω0 ± δω, γa,b = γ0 ± δγ, coupling g, tower of
# EP_n near δω = 0, g = δγ/(2√n), seeds in the slice g = g0. A Redfield scan of that slice costs a few times the
# Lindblad one (18 s) even with RedfieldMap: instead, the Lindblad seeds (scan and localise_eps on the affine family), then the Redfield EP
# of the same pair from each, by the corrector of the line tracker in the slice (ep_correct with normal t = ĝ: Newton
# on D from the Lindblad seed, certified by a check loop that swaps the pair, bracketing otherwise). Two seeds: one
# pair of EP_1 and one of EP_2. Redfield at T = 0.02: smooth (h checked inside the tower region), with thermal
# occupations e^(-50) at ω ≈ 1, so the rates are those of T = 0.
Nc = 10
â, σ̂ = qrm_operators(Nc)
odd = odd_sector(Nc)
ω0, γ0, g0, T_R = 1.0, 0.05, 0.008, 0.02
qrm_at(mdl, prm_P) = setproperties(mdl, (ωa=ω0 + prm_P[1], ωb=ω0 - prm_P[1], γa=γ0 + prm_P[2], γb=γ0 - prm_P[2],
                                         g=prm_P[3]))
slice_g0(A) = slice(A, P3(0, 0, g0), P3(1, 0, 0), P3(0, 1, 0))
A_lind = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(â, σ̂, qrm_at(QRM(approx=[RWA_env]), prm_P))...).data[odd, odd],
                           3; h=0.01)
# Redfield by its precomputed linear map (RedfieldMap): the X's and the sector are fixed along the family, so L is
# M x with x = (H, Λ's), ~20× faster than redfield_liouvillian's sparse assembly, to round-off the same matrix
R_map = RedfieldMap(first.(redfield_baths(â, σ̂, QRM())); idx=odd)
red_at(prm_P) = qrm_at(QRM(Ta=T_R, Tb=T_R), prm_P)
A_red = FDLiouvillian(prm_P -> R_map(Hamiltonian(â, σ̂, red_at(prm_P)), redfield_baths(â, σ̂, red_at(prm_P))),
                      P3(0, 0.016, g0); check_at=P3(0.002, 0.03, 0.02))

xs_s, ys_s = range(-0.0031, 0.0029, 7), range(0.0043, 0.0373, 21)       # as ep_lines.jl: n ≤ 5, no node on δω = 0
F = eigen(Matrix(slice_g0(A_lind)(P2(xs_s[1], ys_s[1]))))
start = [(F.values[k], F.vectors[:, k]) for k ∈ findall(λ -> -1.5 < imag(λ) < -0.5, F.values)]
t = @elapsed loc_lind = localise_eps(slice_g0(A_lind), tracked_scan(slice_g0(A_lind), xs_s, ys_s, start); box_tol=1e-5)
println("Lindblad: $(length(loc_lind.eps)) EPs in the slice g = $g0, ", round(t, digits=1), " s")
box_centre(e) = P3((e.rect[1] + e.rect[2])/2, (e.rect[3] + e.rect[4])/2, g0)
manifold_of(P) = round(Int, (P[2]/(2g0))^2)
picked = [argmin(e -> abs(e.λ - λ), loc_lind.eps) for λ ∈ (-0.025 - 1.0im, -0.0783 - 1.0001im)]   # EP_1, EP_2
seeds_lind = [(box_centre(e), e.λ) for e ∈ picked]
seeds_red = map(seeds_lind) do (prm_P, λ)
    t = @elapsed c = NonHermitianQRM.ep_correct(A_red, prm_P, P3(0, 0, 1), λ; w=0.005)
    println("EP_$(manifold_of(prm_P)): Lindblad (δω, δγ) = ", round.(prm_P[1:2], sigdigits=5), ", λ = ", round(λ, digits=5),
            ";  Redfield ", round.(c.P[1:2], sigdigits=5), ", λ = ", round(c.λ, digits=5), " (", c.path, ", ",
            round(t, digits=1), " s)")
    (c.P, c.λ)
end


#%% EP lines of the full QRM with Redfield baths: continuation

# Needs the previous cell. Both lines tracked in g with the settings of ep_lines.jl, for Redfield (four half-lines in
# parallel: start Julia with --threads=4) and for Lindblad from the same pairs, then a dense check of the Redfield
# points (gap of the pair). Left: the shift δω of the lines off δω = 0, against g. Lindblad: δω = +n g²/2 (the
# Bloch–Siegert-like shift of ep_lines.jl); Redfield: δω = -n g²/2, the opposite shift (to be understood: the
# non-secular terms of the x-coupled baths, against the p-like coupling -ig(a - a†)σx of the dipole gauge?). Redfield
# with RedfieldMap: ~3× the Lindblad time (12× faster than with redfield_liouvillian at every build). Right: the
# deviation from g = δγ/(2√n), a few 1e-6 for both.
bounds_s = (P3(-0.01, -0.048, 0.002), P3(0.01, 0.048, 0.03))
track_lines(A, seeds) = [join_halves(hv...) for hv ∈ track_ep_lines(A, seeds; h=0.002, hmax=0.002, nsteps=80, bounds=bounds_s)]
t_lind = @elapsed lines_lind = track_lines(A_lind, seeds_lind)
t_red = @elapsed lines_red = track_lines(A_red, seeds_red)
println("two lines: Lindblad ", round(t_lind, digits=1), " s, Redfield ", round(t_red, digits=1), " s (",
        Threads.nthreads(), " threads)")
for l ∈ lines_red
    gaps = map(zip(l.points[1:4:end], l.λs[1:4:end])) do (prm_P, λ)
        ev = eigvals(Matrix(A_red(prm_P)))
        k = partialsortperm(abs.(ev .- λ), 1:2)
        abs(ev[k[1]] - ev[k[2]])
    end
    n = manifold_of(l.points[findmin(p -> abs(p[3] - g0), l.points)[2]])
    println("Redfield EP_$n: ", length(l.points), " points, g from ", round(minimum(p -> p[3], l.points), sigdigits=3),
            " to ", round(maximum(p -> p[3], l.points), sigdigits=3), ", δω/g² from ",
            round.(extrema(p -> p[1]/p[3]^2, l.points), digits=3), ", dense max gap ", round(maximum(gaps), sigdigits=2))
end

fig = Figure(size=(1000, 430))
ax_δω = Axis(fig[1, 1]; xlabel="g", ylabel="δω", title="Shift of the EP lines off δω = 0")
ax_dev = Axis(fig[1, 2]; xlabel="g", ylabel="g - δγ/(2√n)", title="Deviation from the JC line")
for (name, ls, style) ∈ (("Lindblad", lines_lind, :dash), ("Redfield", lines_red, :solid)), l ∈ ls
    n = manifold_of(l.points[findmin(p -> abs(p[3] - g0), l.points)[2]])
    colour = clrs[:dark][(:byzantine, :minthe)[n]]
    gs_l = getindex.(l.points, 3)
    lines!(ax_δω, gs_l, getindex.(l.points, 1); color=colour, linestyle=style, label="$name, n = $n")
    lines!(ax_dev, gs_l, [p[3] - p[2]/(2√n) for p ∈ l.points]; color=colour, linestyle=style)
end
gs_guide = range(0, 0.025, 100)
for n ∈ 1:2, sgn ∈ (1, -1)
    lines!(ax_δω, gs_guide, sgn*n .* gs_guide.^2 ./ 2; color=clrs[:dark][:subtext], linestyle=:dot, linewidth=1)
end
axislegend(ax_δω; position=:lb)
save(joinpath(figdir, "redfield_ep_lines.png"), fig)
fig


#%% Scan of the plane (γa, g) at ωa = ωb = 1, γb = 0.05, against the coworker's EPs

# A tracked scan of the Redfield QRM (odd sector, T = 0.02) over γa ∈ [0.005, 0.04], g ∈ [0.1, 0.5], from the eigenvalues
# near Im λ = -ω at the corner (0.005, 0.1), then localise_eps: no line tracking. Compared with the coworker's EP
# table (data/, Redfield at T = 0, Ĥ diagonalised at Nc = 60 and L built on its N_trunc = 30 lowest dressed levels;
# their λ is the conjugate of ours; their "reliable" cut is exactly maxlevel ≤ 15, maxlevel being the highest dressed
# level in the EP's eigenvector).
# - Truncation: Nc = 12 (first run, 2026-10-30) is not enough for g ≲ 0.37. Re-localised at Nc = 16 and 20, its EPs
#   there moved by up to ~5e-3, some were missed, and two pairs were artefacts (g ≈ 0.113; γa ≈ 0.026, λ ≈ -0.31 - 1.2i,
#   30% of its weight on the top Fock levels). For g ≥ 0.37, Nc = 16 and 20 agree to ~1e-8, and the γa ≈ 0.0198
#   pairs match the coworker's to 1e-7–2e-6 (T = 0.02 vs 0 makes no visible difference). Hence Nc = 16 here: ~3× the
#   cost of Nc = 12 (whose scan took ~35 min with 12 threads, plus ~15 min of localisation). Pairs involving dressed
#   levels ≳ 20 (the coworker's maxlevel) still move at Nc = 16 → 20 by ~1e-5.
# - The EPs come in split pairs (two EP2s of one pair, 1e-6–1e-4 apart): a coarse cell sees a winding +2, and
#   localise_eps refines it until each EP2 has its own swapping sub-cell.
# - Lost labels (outlined cells, g ≳ 0.22): the window, not the truncation. A few labels at its edges sweep far in
#   Im λ with g, pass EPs with eigenvalues outside the tracked set, and come back on none of the labels. EPs among the
#   other labels are fully covered. Also left out by the window: the towers |Im λ| ≈ 3…13 and the |Im λ| ≈ 1 branch
#   with Re λ < -0.15, both in the coworker's table.
Nc = 16
â, σ̂ = qrm_operators(Nc)
odd = odd_sector(Nc)
base_sc = QRM(ωa=1.0, ωb=1.0, γb=0.05, Ta=0.02, Tb=0.02)
qrm_sc(prm_P) = setproperties(base_sc, (γa=prm_P[1], g=prm_P[2]))
R_sc = RedfieldMap(first.(redfield_baths(â, σ̂, base_sc)); idx=odd)
A_sc = FDLiouvillian(prm_P -> R_sc(Hamiltonian(â, σ̂, qrm_sc(prm_P)), redfield_baths(â, σ̂, qrm_sc(prm_P))),
                     P2(0.012, 0.3); check_at=P2(0.006, 0.45))
xs_sc, ys_sc = range(0.005, 0.04, 36), range(0.1, 0.5, 41)
F = eigen(Matrix(A_sc(P2(xs_sc[1], ys_sc[1]))))
start_sc = [(F.values[k], F.vectors[:, k]) for k ∈ findall(λ -> -1.5 < imag(λ) < -0.5 && real(λ) > -0.15, F.values)]
t = @elapsed cells_sc = tracked_scan(A_sc, xs_sc, ys_sc, start_sc)
t_l = @elapsed loc_sc = localise_eps(A_sc, cells_sc; box_tol=1e-5)
println(length(start_sc), " eigenvalues tracked; scan ", round(t, digits=1), " s, ", count(flagged, cells_sc),
        " flagged cells, ", count(c -> c.lost > 0, cells_sc), " with lost labels; localisation ", round(t_l, digits=1),
        " s: ", length(loc_sc.eps), " EPs, ", length(loc_sc.dps), " DPs, ", length(loc_sc.unresolved), " unresolved")
centre(r) = Point2f((r[1] + r[2])/2, (r[3] + r[4])/2)

# our EPs as a table in the style of the coworker's (same names where the meaning is the same; λ in our sign
# convention, the conjugate of theirs), with a light summary of the cells to redraw the scan without rerunning it
point_of(prm_P) = merge((g=prm_P[2], γa=prm_P[1], Nc=Nc), (; (k => getfield(base_sc, k) for k ∈ (:ωa, :ωb, :γb, :ε, :Ta, :Tb))...),
                        (liouvillian=:redfield, rwa=false))
R_ours = [(point=point_of(((e.rect[1] + e.rect[2])/2, (e.rect[3] + e.rect[4])/2)), verdict="EP2", λ=e.λ, rect=e.rect, pair=e.pair, cell=e.cell) for e ∈ loc_sc.eps]
jldsave(joinpath(pkgdir(NonHermitianQRM), "data", "redfield_scan_gammaa_g_Nc$Nc.jld2");
        R=R_ours, cells=[(rect=c.rect, flagged=flagged(c), lost=c.lost) for c ∈ cells_sc],
        dps=[c.rect for c ∈ loc_sc.dps], unresolved=[c.rect for c ∈ loc_sc.unresolved], xs=collect(xs_sc), ys=collect(ys_sc))

theirs = filter(r -> xs_sc[1] ≤ r.point.γa ≤ xs_sc[end] && ys_sc[1] ≤ r.point.g ≤ ys_sc[end],
                load(joinpath(pkgdir(NonHermitianQRM), "data", "ep_prod_complex_eps.jld2"), "R"))
maxlevel_cut = 25

C = clrs[:dark]
rect_pts(r) = Point2f[(r[1], r[3]), (r[2], r[3]), (r[2], r[4]), (r[1], r[4]), (r[1], r[3])]
λs = [e.λ for e ∈ loc_sc.eps]
fig = Figure(size=(1500, 700))
Label(fig[0, 1:4], "Redfield QRM, ωa = ωb = 1, γb = 0.05, T = 0.02, Nc = $Nc: tracked scan and localised EPs; " *
                   "circles: coworker's EPs (T = 0, Nc = 60, N_trunc = 30; filled: maxlevel ≤ $maxlevel_cut)")
ax = Axis(fig[1, 1]; xlabel="γa", ylabel="g", title="scan (shaded: flagged, outlined: lost labels)")
for c ∈ cells_sc
    if flagged(c)
        poly!(ax, rect_pts(c.rect); color=(C[:overlay], 0.4), strokecolor=C[:subtext], strokewidth=0.8)
    else
        lines!(ax, rect_pts(c.rect); color=C[:surface], linewidth=0.5)
    end
    c.lost > 0 && lines!(ax, rect_pts(c.rect); color=(C[:helios], 0.6), linewidth=1, label="lost labels")
end
for (cs, mk, col, name) ∈ ((loc_sc.dps, :diamond, C[:minthe], "DP (winding ±2)"),
                           (loc_sc.unresolved, :xcross, C[:selene], "unresolved"))
    isempty(cs) || scatter!(ax, [centre(c.rect) for c ∈ cs]; marker=mk, color=col, markersize=12, label=name)
end
sc = scatter!(ax, [centre(e.rect) for e ∈ loc_sc.eps]; color=imag.(λs), colormap=C[:byz], marker=:star5,
              markersize=13, label="EP2 (ours)")
Colorbar(fig[1, 2], sc; label="Im λ_EP (ours)")
# coworker's EPs: colour = branch (|Im λ| ≈ 1 as ours, or the towers outside our window), filled = maxlevel ≤ cut
for kept ∈ (true, false), near ∈ (true, false)
    rs = filter(r -> (r.maxlevel ≤ maxlevel_cut) == kept && (abs(imag(r.λ)) < 1.5) == near, theirs)
    isempty(rs) && continue
    col = near ? C[:hesperides] : C[:selene]
    scatter!(ax, [Point2f(r.point.γa, r.point.g) for r ∈ rs]; color=kept ? (col, 0.5) : :transparent, strokecolor=col,
             strokewidth=1.2, markersize=11,
             label="coworker, |Im λ| ≈ $(near ? "1" : "3…13"), maxlevel $(kept ? "≤" : ">") $maxlevel_cut")
end
axislegend(ax; position=:rb, unique=true, merge=true)
limits!(ax, xs_sc[1], xs_sc[end], ys_sc[1], ys_sc[end])
ax_λ = Axis(fig[1, 3]; xlabel="Re λ", ylabel="Im λ", title="coalesced eigenvalues (coworker's: conj λ, |Im λ| < 1.5)")
scatter!(ax_λ, real.(λs), imag.(λs); color=[centre(e.rect)[2] for e ∈ loc_sc.eps], colormap=C[:byz], marker=:star5,
         markersize=13, colorrange=extrema(ys_sc))
near1 = filter(r -> abs(imag(r.λ)) < 1.5, theirs)
scatter!(ax_λ, real.(conj.([r.λ for r ∈ near1])), imag.(conj.([r.λ for r ∈ near1]));
         color=[r.maxlevel ≤ maxlevel_cut ? (C[:hesperides], 0.5) : (C[:hesperides], 0.0) for r ∈ near1],
         strokecolor=C[:hesperides], strokewidth=1.2, markersize=11)
Colorbar(fig[1, 4], limits=extrema(ys_sc), colormap=C[:byz], label="g")
save(joinpath(figdir, "redfield_scan_gammaa_g.png"), fig)
fig
