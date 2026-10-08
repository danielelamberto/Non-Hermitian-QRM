# Bloch–Redfield master equation of the quantum Rabi model: our assembly (redfield_liouvillian) against QuantumToolbox's,
# its fixed sparsity pattern and parity sectors, its spectrum against the Lindblad and dressed master equations, its
# smoothness in the parameters at T = 0 and small T > 0, and its tracking as a finite-difference family
# (FDLiouvillian): the check of the step h, the error of the derivative against h, and a swap around an EP.
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using NonHermitianQRM: ohmic
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
