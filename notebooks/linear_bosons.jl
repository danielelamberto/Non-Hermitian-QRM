# Linear bosons: single boson and coupled bosons (analytical response vs master equations), monodromy around the dimer EP.
# Cell script (#%% cells); the generic functions are in the package NonHermitianQRM (src/).

#%% Packages

# Revise (global environment) picks up edits to src/ without restarting Julia: load it before the package.
using Revise
using NonHermitianQRM
using NonHermitianQRM: Circle          # also exported by Makie: ours is the loop in parameter space
using QuantumToolbox, CairoMakie, MakieStyles, LinearAlgebra, SparseArrays, Polynomials, Accessors


#%% Single boson: EP at critical damping, Q = ω0/2γ = 1/2

n_fock = 10
â = destroy(n_fock)

mdl = Boson(ω0=1.0, approx=[])
γ_EP, ω_EP = EP(mdl)
γs = sort([range(0.02, 2.0, 100); γ_EP])

fig = Figure(size=(900, 400))
ax_re = Axis(fig[1, 1]; xlabel="γ/ω0", ylabel="Re ω", title="single boson, EP at γ = ω0 (Q = 1/2)")
ax_im = Axis(fig[1, 2]; xlabel="γ/ω0", ylabel="Im ω")
# (label, colour, marker, mode frequencies as a function of the model)
series = [
    ("disc, []",            2, :circle,  m -> roots_sorted(@set m.approx = [])),
    ("Bloch–Redfield",      2, :xcross,  m -> modes(Redfield(â, m), [â])),
    ("disc, [RWA_env]",     1, :circle,  m -> roots_sorted(@set m.approx = [RWA_env])),
    ("Lindblad, [RWA_env]", 1, :xcross,  m -> modes(liouvillian(Lindbladian(â, @set m.approx = [RWA_env])...), [â])),
    ("dressed non-secular", 3, :cross,   m -> dressed_modes(DressedLiouvillian(â, m), [â])),
]
for (label, col, marker, f) ∈ series
    ω = reduce(hcat, f(@set mdl.γ = γ) for γ ∈ γs)
    for ax ∈ (ax_re, ax_im), row ∈ eachrow(ω)
        scatter!(ax, γs, (ax === ax_re ? real : imag).(row); color=Cycled(col), marker, markersize=(marker == :circle ? 5 : 9), label)
    end
end
for ax ∈ (ax_re, ax_im)
    vlines!(ax, [γ_EP]; color=:gray, linestyle=:dash)
end
axislegend(ax_im; position=:lb, merge=true, unique=true)

ωR = modes(Redfield(â, @set mdl.γ = γ_EP), [â])
println("Redfield at γ_EP: ", ωR, ",  gap ", abs(ωR[1] - ωR[2]), "  vs ω_EP = ", ω_EP)
println("max |Redfield - disc([])| over the sweep: ",
    maximum(maximum(minimum(abs.(w .- roots_sorted(m))) for w ∈ modes(Redfield(â, m), [â])) for m ∈ (@set(mdl.γ = γ) for γ ∈ γs)))
fig


#%% Coupled bosons: EP at finite frequency

n_fock = 5
â = destroy(n_fock) ⊗ eye(n_fock)
b̂ = eye(n_fock) ⊗ destroy(n_fock)

Ω, γa, γb = 1.0, 0.3, 0.1
panels = [
    ("equal bare ω (EP with RWA_env)",                   BosonDimer(ωa=Ω, ωb=Ω, γa=γa, γb=γb, approx=[])),
    ("equal damped Ω = √(ω² - γ²) (EP without RWA_env)", BosonDimer(ωa=√(Ω^2 + γa^2), ωb=√(Ω^2 + γb^2), γa=γa, γb=γb, approx=[])),
]
gs = range(0, 0.4, 31)
pos(ω) = filter(w -> real(w) > 0, ω)
gap(ω) = abs(ω[1] - ω[2])
# (label, colour, marker, positive-frequency modes as a function of the model, approximations of the matching analytics)
series = [
    ("disc, []",                        2, :circle, m -> pos(roots_sorted(@set m.approx = [])),                                        []),
    ("Bloch–Redfield",                  2, :xcross, m -> pos(modes(Redfield(â, b̂, m), [â, b̂])),                                         []),
    ("disc, [RWA_env]",                 1, :circle, m -> pos(roots_sorted(@set m.approx = [RWA_env])),                                 [RWA_env]),
    ("Lindblad, [RWA_env]",             1, :xcross, m -> pos(modes(liouvillian(Lindbladian(â, b̂, @set m.approx = [RWA_env])...), [â, b̂])), [RWA_env]),
    ("disc, [RWA_env, RWA_coupling]",   4, :circle, m -> pos(roots_sorted(@set m.approx = [RWA_env, RWA_coupling])),                   [RWA_env, RWA_coupling]),
    ("Lindblad, [RWA_env, RWA_coupling]", 4, :xcross, m -> pos(modes(liouvillian(Lindbladian(â, b̂, @set m.approx = [RWA_env, RWA_coupling])...), [â, b̂])), [RWA_env, RWA_coupling]),
    ("dressed non-secular",             3, :cross,  m -> pos(dressed_modes(DressedLiouvillian(â, b̂, m), [â, b̂])),                        nothing),
]

fig = Figure(size=(1100, 500))
for (j, (title, mdl)) ∈ enumerate(panels)
    ax = Axis(fig[1, j]; xlabel="Re ω", ylabel="Im ω", title)
    println(title)
    for (label, col, marker, f, approx) ∈ series
        for g ∈ gs
            ω = f(@set mdl.g = g)
            scatter!(ax, real(ω), imag(ω); color=Cycled(col), marker, markersize=(marker == :circle ? 5 : 9), label)
        end
        # analytic EP of this approximation (stars), and the gap of the corresponding modes there
        isnothing(approx) && continue
        ep = EP(@set mdl.approx = approx)
        if isnothing(ep)
            println("  ", rpad(label, 36), "no EP (detuned)")
        else
            g_ep, ω_ep = ep
            scatter!(ax, [real(ω_ep)], [imag(ω_ep)]; color=Cycled(col), marker=:star5, markersize=18)
            println("  ", rpad(label, 36), "g_EP = ", round(g_ep, digits=5), ",  gap there = ", round(gap(f(@set mdl.g = g_ep)), sigdigits=2))
        end
    end
    j == 1 && axislegend(ax; position=:lb, merge=true, unique=true, labelsize=10)
end
fig


#%% Monodromy, coupled bosons: tracking the Liouvillian modes around the EP (Lindblad, RWA_env)

# Lindblad dimer in the plane δ = (δω, δγ): detuning ±δω and damping asymmetry ±δγ around ω0 = 1, γ0 = 0.1, g = 0.01.
# EPs at δω = 0, δγ = ±g/2. Both modes are tracked around five loops and compared with -iω from the roots of disc
# (same parametrisation). Expected: a swap for a loop around one EP, none around zero or two EPs.
n_fock = 5
â = destroy(n_fock) ⊗ eye(n_fock)
b̂ = eye(n_fock) ⊗ destroy(n_fock)
g = 0.01
mdl = BosonDimer(g=g, approx=[RWA_env])
dimer(δ) = setproperties(mdl, (ωa=1 + δ[1], ωb=1 - δ[1], γa=0.1 + δ[2], γb=0.1 - δ[2]))
A = AffineLiouvillian(δ -> liouvillian(Lindbladian(â, b̂, dimer(δ))...).data, 2)

# EP(⋅) returns g_EP ∝ |γa - γb| at δω = 0: rescale to find the δγ where g_EP = g
δγ_EP = g*0.01/EP(dimer(P2(0, 0.01)))[1]
disc_λ(δ) = -im .* filter(r -> real(r) > 0, roots_sorted(dimer(δ)))

dimer_loops = [
    ("circle, 1 EP",                Circle(P2(0, 0.01), 0.01)),
    ("rectangle, 1 EP",             Polygon([P2(-0.01, 0), P2(0.01, 0), P2(0.01, 0.02), P2(-0.01, 0.02)])),
    ("circle, no EP",               Circle(P2(0, 0.03), 0.01)),
    ("circle, both EPs",            Circle(P2(0, 0), 0.01)),
    ("small circle, 1e-4 from EP",  Circle(P2(0, δγ_EP + 1e-4), 2e-4)),
]
us = range(0, 1, 401)

fig = Figure(size=(1000, 1300))
ax_p = Axis(fig[1, 1]; xlabel="δω", ylabel="δγ", title="loops in parameter space")
scatter!(ax_p, [0, 0], [δγ_EP, -δγ_EP]; marker=:star5, markersize=14, color=:white, label="EPs")
for (j, (name, q)) ∈ enumerate(dimer_loops)
    δs = position.(Ref(q), us)
    lines!(ax_p, first.(δs), last.(δs); color=Cycled(j + 2), label=name)
    ax = Axis(fig[fldmod1(j + 1, 2)...]; xlabel="Re ω", ylabel="Im ω", title=name)
    scatter!(ax, vcat((real.(im .* disc_λ(δ)) for δ ∈ δs[1:4:end])...), vcat((imag.(im .* disc_λ(δ)) for δ ∈ δs[1:4:end])...);
        color=:gray, markersize=4)
    pairs0 = eigenpairs_near(A(position(q, 0.)), disc_λ(position(q, 0.)))
    println(name)
    for (k, (λ0, r0)) ∈ enumerate(pairs0)
        t = @elapsed sol, info = track(A, q, λ0, r0)
        lands = argmin(abs.(first.(pairs0) .- sol.u[end][end]))
        dev = maximum(minimum(abs.(sol(u)[end] .- disc_λ(position(q, u)))) for u ∈ us)
        println("  mode $k → mode $lands,  max |λ - λ_disc| = ", round(dev, sigdigits=2), ",  ", sol.stats.naccept, " steps, ",
            info.factorisations, " LU, ", round(t, digits=2), " s")
        ω = [im*sol(u)[end] for u ∈ us]
        lines!(ax, real(ω), imag(ω); color=Cycled(k), label="mode $k" * (lands == k ? " (returns)" : " → mode $lands"))
        scatter!(ax, [real(im*λ0)], [imag(im*λ0)]; color=Cycled(k), marker=:star5, markersize=14)
    end
    axislegend(ax; position=:rb, labelsize=10)
end
ylims!(ax_p, -0.012, 0.075)                 # room for the legend above the loops
axislegend(ax_p; position=:ct, labelsize=9)
fig
