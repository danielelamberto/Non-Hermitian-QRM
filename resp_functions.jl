#%% Generic definitions: models, analytical response, master equations

using QuantumToolbox
using CairoMakie
using MakieStyles

using LinearAlgebra
using Polynomials
using Accessors

abstract type Model end

@enum Approximation RWA_env RWA_coupling

# accepts a single Approximation, a tuple/vector of them, or an empty collection
to_approx(a::Approximation) = [a]
to_approx(a) = collect(Approximation, a)

struct boson{T<:AbstractFloat} <: Model
    ω0::T
    γ::T
    nB::T
    approx::Vector{Approximation}
end
boson(ω0::Real, γ::Real, nB::Real, approx=[RWA_env]) = boson(
    promote(float(ω0), float(γ), float(nB))..., to_approx(approx)
)
boson(;ω0=1., γ=1., nB=0., approx=[RWA_env]) = boson(ω0, γ, nB, approx)

struct bosonDimer{T<:AbstractFloat} <: Model
    ωa::T
    ωb::T
    γa::T
    γb::T
    g::T
    nB::T
    approx::Vector{Approximation}
end
bosonDimer(ωa::Real, ωb::Real, γa::Real, γb::Real, g::Real, nB::Real, approx=[RWA_env, RWA_coupling]) = bosonDimer(
    promote(float(ωa), float(ωb), float(γa), float(γb), float(g), float(nB))..., to_approx(approx)
)
bosonDimer(;ωa=1., ωb=1., γa=1., γb=1., g=0., nB=0., approx=[RWA_env, RWA_coupling]) = bosonDimer(ωa, ωb, γa, γb, g, nB, approx)

## Analytical response

D_R(ω, mdl::boson) = RWA_env ∈ mdl.approx ? ((ω + im*mdl.γ)^2 - mdl.ω0^2)/mdl.ω0 : (ω^2 - mdl.ω0^2 + 2im*ω*mdl.γ)/mdl.ω0

# RWA_coupling, g/2(a†b + b†a): rotating-frame 2×2 in the (a, b) basis.
# Otherwise, g/2(a+a†)(b+b†) = g x_a x_b: 2×2 in the (x_a, x_b) basis, diagonal from the single boson (respects RWA_env).
function D_R(ω, mdl::bosonDimer)
    (; ωa, ωb, γa, γb, g, nB, approx) = mdl
    if RWA_coupling ∈ approx
        return [
            ω-ωa+1im*γa g/2
            g/2 ω-ωb+1im*γb
        ]
    end
    Da = D_R(ω, boson(ωa, γa, nB, approx))
    Db = D_R(ω, boson(ωb, γb, nB, approx))
    return [
        Da g
        g Db
    ]
end

# Characteristic polynomial in ω, disc(mdl)(ω) == det(D_R(ω, mdl)): roots are the complex mode frequencies.
# Evaluating D_R on the polynomial variable inherits its dispatch on the model and the approximations.
disc(mdl::boson) = D_R(Polynomial([0, 1], :ω), mdl)
function disc(mdl::bosonDimer)
    D = D_R(Polynomial([0, 1], :ω), mdl)
    return D[1, 1]*D[2, 2] - D[1, 2]*D[2, 1]
end

# EP of the single boson, (γ_EP, ω_EP): critical damping γ = ω0 (Q = ω0/2γ = 1/2), where -iγ ± √(ω0² - γ²) merge
# on the imaginary axis. With RWA_env the modes are ±ω0 - iγ for any γ: no EP.
EP(mdl::boson) = RWA_env ∈ mdl.approx ? nothing : (mdl.ω0, -im*mdl.ω0)

# EP of the positive-frequency modes, (g_EP, ω_EP), or nothing if the modes are detuned (no EP by tuning g alone).
# Without RWA_coupling the frequencies that must match are Ω_i = √(ω_i² - γ_i²), or the bare ω_i with RWA_env.
function EP(mdl::bosonDimer)
    (;ωa, ωb, γa, γb, approx) = mdl
    δ, γ̄ = (γa - γb)/2, (γa + γb)/2
    if RWA_coupling ∈ approx
        return ωa ≈ ωb ? (2abs(δ), ωa - im*γ̄) : nothing
    end
    Ωa, Ωb = RWA_env ∈ approx ? (ωa, ωb) : (√(ωa^2 - γa^2), √(ωb^2 - γb^2))
    (Ωa ≈ Ωb && Ωa > abs(δ)) || return nothing
    return Ωa*2abs(δ)/√(ωa*ωb), √(Ωa^2 - δ^2) - im*γ̄
end

## Master equations
# â and b̂ must already act on the joint Hilbert space, e.g. â = destroy(N) ⊗ eye(N).

Hamiltonian(â::QuantumObject, mdl::boson) = mdl.ω0*â'*â

# Coupling as in D_R: g/2(a†b + b†a) with RWA_coupling, g/2(a+a†)(b+b†) otherwise.
function Hamiltonian(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer)
    (;ωa, ωb, g, approx) = mdl
    Ĥ_int = RWA_coupling ∈ approx ? g/2*(â'*b̂ + b̂'*â) : g/2*(â + â')*(b̂ + b̂')
    return ωa*â'*â + ωb*b̂'*b̂ + Ĥ_int
end

# Each mode's annihilation operator with its Ohmic bath (γ, ω0), coupled through x = â + â'.
baths(â::QuantumObject, mdl::boson) = [(â, mdl.γ, mdl.ω0)]
baths(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer) = [(â, mdl.γa, mdl.ωa), (b̂, mdl.γb, mdl.ωb)]

# Bath temperature from the thermal occupation nB at the reference frequency ω.
temperature(nB, ω) = nB > 0 ? ω/log(1 + 1/nB) : 0.0
temperature(mdl::boson) = temperature(mdl.nB, mdl.ω0)
temperature(mdl::bosonDimer) = temperature(mdl.nB, mdl.ωa)

# Ohmic spectrum S(ω) = κω(1 + n(ω)) (ω > 0 emission, ω < 0 absorption).
ohmic(κ, T) = ω -> T == 0 ? (ω > 0 ? κ*ω : zero(ω)) : (ω == 0 ? κ*T : κ*ω/(1 - exp(-ω/T)))

# Local Lindblad form, secular in the bath: jump operators √(2γ(1+nB)) â and √(2γnB) â'. Returns (Ĥ, L̂s).
function _lindbladian(Ĥ, bths, nB, approx)
    RWA_env ∈ approx || @warn "Lindbladian called without RWA_env: the Lindblad form is secular in the bath, so its spectrum will not match D_R/disc for this model." maxlog=1
    return Ĥ, [L̂ for (â, γ, _) ∈ bths for L̂ ∈ (√(2γ*nB)*â', √(2γ*(1+nB))*â)]
end

# Dressed non-secular master equation (Settineri 2018). Field c(â+â') has emission rate ω0c², so c = √(2γ/ω0) matches √(2γ)â.
# Non-secular only among transitions of the same sign: the bath is still treated in the RWA, in the eigenbasis of H.
# σ_filter → 0 is the dressed secular limit, Inf keeps all cross terms. No Lamb shift. Returns (E, U, L) in the eigenbasis.
function _dressed(Ĥ, bths, T; N_trunc=nothing, σ_filter=Inf)
    fields = [√(2γ/ω)*(â + â') for (â, γ, ω) ∈ bths]
    return liouvillian_dressed_nonsecular(Ĥ, fields, fill(T, length(bths)); N_trunc, σ_filter)
end

# Non-secular Bloch–Redfield (sec_cutoff = -1) with Ohmic spectra, κ = 2γ/ω0, in the Fock basis. Keeps the terms pairing
# ω and -ω transitions: velocity damping, as in D_R without RWA_env. No Lamb shift (none is needed for a strictly Ohmic bath).
_redfield(Ĥ, bths, T) = bloch_redfield_tensor(Ĥ, [(â + â', ohmic(2γ/ω, T)) for (â, γ, ω) ∈ bths]; sec_cutoff=-1, fock_basis=Val(true))

Lindbladian(â::QuantumObject, mdl::boson) = _lindbladian(Hamiltonian(â, mdl), baths(â, mdl), mdl.nB, mdl.approx)
Lindbladian(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer) = _lindbladian(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), mdl.nB, mdl.approx)

DressedLiouvillian(â::QuantumObject, mdl::boson; kwargs...) = _dressed(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl); kwargs...)
DressedLiouvillian(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer; kwargs...) = _dressed(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), temperature(mdl); kwargs...)

Redfield(â::QuantumObject, mdl::boson) = _redfield(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl))
Redfield(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer) = _redfield(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), temperature(mdl))

# Complex mode frequencies ω = iλ of a Liouvillian L, given the annihilation operators of the modes in L's basis.
# For these quadratic models (â|, (â'| span an L†-invariant subspace, so only 2 eigenvectors per mode carry ⟨â⟩, ⟨â'⟩:
# select them by that overlap rather than by position, which fails once the modes reach the imaginary axis.
# Dense: L is n²×n² for an n-dimensional Hilbert space.
function modes(L::QuantumObject, ops)
    λ, V = eigen(Matrix(L.data))
    n = size(first(ops), 1)
    w = [sum(abs(tr(ô.data * reshape(V[:, i], n, n))) for â ∈ ops for ô ∈ (â, â')) for i ∈ eachindex(λ)]
    idx = partialsortperm(w, 1:2length(ops), rev=true)
    return sort(im .* λ[idx], by=ω -> (real(ω), imag(ω)))
end

# Modes of the dressed Liouvillian: its operators must be taken to the eigenbasis first.
dressed_modes((E, U, L), ops) = modes(L, [U'*â*U for â ∈ ops])

roots_sorted(mdl) = sort(roots(disc(mdl)), by=ω -> (real(ω), imag(ω)))


#%% Single boson: EP at critical damping, Q = ω0/2γ = 1/2

n_fock = 10
â = destroy(n_fock)

mdl = boson(ω0=1.0, approx=[])
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
    ("equal bare ω (EP with RWA_env)",                   bosonDimer(ωa=Ω, ωb=Ω, γa=γa, γb=γb, approx=[])),
    ("equal damped Ω = √(ω² - γ²) (EP without RWA_env)", bosonDimer(ωa=√(Ω^2 + γa^2), ωb=√(Ω^2 + γb^2), γa=γa, γb=γb, approx=[])),
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
