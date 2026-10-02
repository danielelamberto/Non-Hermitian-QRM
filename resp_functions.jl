#%% Generic definitions: models, analytical response, master equations

using QuantumToolbox
using CairoMakie
using MakieStyles

using LinearAlgebra
using Polynomials
using Accessors

abstract type Model end

const 𝑖 = 1im
@enum Approximation RWA_env RWA_coupling N_conserving linearised

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

# Damped anharmonic (Duffing) oscillator, H = ω0 a†a + g/4! x̂⁴ with x̂ = (a + a†)/√2, the quadrature described by D_R.
# (In unit-mass units, x̂ = (a + a†)/√(2ω0), the same term reads g/(4! ω0²) x̂⁴.)
# N_conserving keeps only the excitation-conserving (Kerr) part of x̂⁴; RWA_env selects the Lindblad bath as for the bosons.
struct duffing{T<:AbstractFloat} <: Model
    ω0::T
    γ::T
    g::T
    nB::T
    approx::Vector{Approximation}
end
duffing(ω0::Real, γ::Real, g::Real, nB::Real, approx=[RWA_env]) = duffing(
    promote(float(ω0), float(γ), float(g), float(nB))..., to_approx(approx)
)
duffing(;ω0=1., γ=1., g=0., nB=0., approx=[RWA_env]) = duffing(ω0, γ, g, nB, approx)

# Driven optomechanical system, in the frame rotating at the drive frequency (RWA on the drive):
# H = -Δ a†a + ωb b†b - g a†a (b + b†) + F (a + a†). Cavity a: damping γa, thermal occupation nBa; mechanics b: γb, nBb.
# The frame change makes a's bath time dependent unless it is treated in the RWA, which is excellent for ω_drive ≫ γa, ωb
# (the shifted bath spectrum is flat over the rotating-frame band): a always gets Lindblad dissipators.
# b's coupling x_b commutes with the frame change: RWA_env selects Lindblad for b, otherwise Ohmic Redfield.
# linearised: fluctuations around the classical steady state α (see classical_steady_state), coupling -g(α* a + α a†)(b + b†).
# Fock cutoff for a: the full model holds the coherent amplitude, so N_a must exceed |α|² by several √|α|².
struct optomech{T<:AbstractFloat} <: Model
    ωb::T
    γa::T
    γb::T
    g::T
    Δ::T
    F::T
    nBa::T
    nBb::T
    approx::Vector{Approximation}
end
optomech(ωb::Real, γa::Real, γb::Real, g::Real, Δ::Real, F::Real, nBa::Real, nBb::Real, approx=[RWA_env]) = optomech(
    promote(float(ωb), float(γa), float(γb), float(g), float(Δ), float(F), float(nBa), float(nBb))..., to_approx(approx)
)
optomech(;ωb=1., γa=0.1, γb=0.1, g=0., Δ=-1., F=0., nBa=0., nBb=0., approx=[RWA_env]) = optomech(ωb, γa, γb, g, Δ, F, nBa, nBb, approx)

# Classical steady state, a → α, b → β: the static displacement x̄ = β + β* = 2g|α|² c with c = ωb/(ωb² + γb²) (Lindblad
# damping of b, RWA_env) or 1/ωb (velocity damping), and α = F/(Δ + g x̄ + iγa). Then n = |α|² solves the cubic
# n((Δ + 2g²c n)² + γa²) = F², with up to three positive roots (bistability). Returns all (α, x̄), by increasing |α|².
function classical_steady_state(mdl::optomech)
    (;ωb, γa, γb, g, Δ, F, approx) = mdl
    c = RWA_env ∈ approx ? ωb/(ωb^2 + γb^2) : 1/ωb
    k = 2g^2*c
    cubic = Polynomial([-F^2, Δ^2 + γa^2, 2Δ*k, k^2])
    ns = sort([real(n) for n ∈ roots(cubic) if abs(imag(n)) < 1e-10 && real(n) ≥ 0])
    iszero(k) && (ns = [F^2/(Δ^2 + γa^2)])
    return [(F/(Δ + k*n + im*γa), 2g*c*n) for n ∈ ns]
end

# Reference displacement (α, β = x̄/2) of the classical steady state, for a displaced frame a = α + d, b = β + e.
function classical_displacement(mdl::optomech; branch=1)
    α, x̄ = classical_steady_state(mdl)[branch]
    return α, x̄/2
end

# Linearised fluctuations around a classical steady state, as the equivalent bosonDimer: a at -(Δ + g x̄), coupling
# -g(α* a + α a†)(b + b†) = (2g|α|/2)(a' + a'†)(b + b†) after absorbing the phase of α. Same bath treatment (approx).
function linearised_dimer(mdl::optomech; branch=1)
    (;ωb, γa, γb, g, Δ, nBb, approx) = mdl
    α, x̄ = classical_steady_state(mdl)[branch]
    return bosonDimer(ωa=-(Δ + g*x̄), ωb=ωb, γa=γa, γb=γb, g=2g*abs(α), nB=nBb, approx=filter(!=(linearised), approx))
end

# (Δ, F) of the linearised EP: resonance Δ + g x̄ = -ωb and coupling 2g|α| = g_EP of the dimer (EP of bosonDimer).
# Only for RWA_env, where the dimer EP needs equal bare frequencies. Returns (Δ, F, number of classical branches there).
function linearised_EP(mdl::optomech)
    (;ωb, γa, γb, g) = mdl
    RWA_env ∈ mdl.approx || throw(ArgumentError("linearised_EP is implemented for RWA_env only"))
    g_EP, = EP(bosonDimer(ωa=ωb, ωb=ωb, γa=γa, γb=γb, g=1., approx=[RWA_env]))
    n = (g_EP/2g)^2                                  # |α|² at the EP
    x̄ = 2g*n*ωb/(ωb^2 + γb^2)
    Δ = -ωb - g*x̄
    F = √(n*(ωb^2 + γa^2))
    return Δ, F, length(classical_steady_state(setproperties(mdl, (Δ=Δ, F=F))))
end

## Analytical response

D_R(ω, mdl::boson) = RWA_env ∈ mdl.approx ? ((ω + im*mdl.γ)^2 - mdl.ω0^2)/mdl.ω0 : (ω^2 - mdl.ω0^2 + 2im*ω*mdl.γ)/mdl.ω0

# RWA_coupling, g/2(a†b + b†a): rotating-frame 2×2 in the (a, b) basis.
# Otherwise, g/2(a+a†)(b+b†) = g x_a x_b: 2×2 in the (x_a, x_b) basis, diagonal from the single boson (respects RWA_env).
function D_R(ω, mdl::bosonDimer)
    (; ωa, ωb, γa, γb, g, nB, approx) = mdl
    if RWA_coupling ∈ approx
        return [
            ω-ωa+𝑖*γa g/2
            g/2 ω-ωb+𝑖*γb
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

Hamiltonian(â::QuantumObject, mdl::boson) = mdl.ω0*â'*â

# x̂⁴ connects n to n ± 2, n ± 4: with a Fock cutoff N the top levels are distorted, keep N well above the populated ones.
# With N_conserving, x̂⁴ → its part with two â and two â', (6â'²â² + 12â'â + 3)/4 = (3/2)n̂² + (3/2)n̂ + 3/4 (Kerr):
# the diagonal of x̂⁴ in the Fock basis, exact up to the cutoff.
function Hamiltonian(â::QuantumObject, mdl::duffing)
    n̂ = â'*â
    if N_conserving ∈ mdl.approx
        return mdl.ω0*n̂ + mdl.g/24*(3/2*n̂^2 + 3/2*n̂ + 3/4*one(n̂))
    end
    x̂ = (â + â')/√2
    return mdl.ω0*n̂ + mdl.g/24*x̂^4
end

# Full model, or (linearised) the fluctuation Hamiltonian around the lowest-|α| classical steady state, with Δ → Δ + g x̄.
function Hamiltonian(â::QuantumObject, b̂::QuantumObject, mdl::optomech; branch=1)
    (;ωb, g, Δ, F, approx) = mdl
    if linearised ∈ approx
        α, x̄ = classical_steady_state(mdl)[branch]
        return -(Δ + g*x̄)*â'*â + ωb*b̂'*b̂ - g*(conj(α)*â + α*â')*(b̂ + b̂')
    end
    return -Δ*â'*â + ωb*b̂'*b̂ - g*â'*â*(b̂ + b̂') + F*(â + â')
end

# Coupling as in D_R: g/2(a†b + b†a) with RWA_coupling, g/2(a+a†)(b+b†) otherwise.
function Hamiltonian(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer)
    (;ωa, ωb, g, approx) = mdl
    Ĥ_int = RWA_coupling ∈ approx ? g/2*(â'*b̂ + b̂'*â) : g/2*(â + â')*(b̂ + b̂')
    return ωa*â'*â + ωb*b̂'*b̂ + Ĥ_int
end

# Each mode's annihilation operator with its Ohmic bath (γ, ω0), coupled through x = â + â'.
baths(â::QuantumObject, mdl::boson) = [(â, mdl.γ, mdl.ω0)]
baths(â::QuantumObject, mdl::duffing) = [(â, mdl.γ, mdl.ω0)]
baths(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer) = [(â, mdl.γa, mdl.ωa), (b̂, mdl.γb, mdl.ωb)]

# Bath temperature from the thermal occupation nB at the reference frequency ω.
temperature(nB, ω) = nB > 0 ? ω/log(1 + 1/nB) : 0.0
temperature(mdl::boson) = temperature(mdl.nB, mdl.ω0)
temperature(mdl::duffing) = temperature(mdl.nB, mdl.ω0)
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

Lindbladian(â::QuantumObject, mdl::boson) = _lindbladian(Hamiltonian(â, mdl), baths(â, mdl), mdl.nB, mdl.approx)
Lindbladian(â::QuantumObject, mdl::duffing) = _lindbladian(Hamiltonian(â, mdl), baths(â, mdl), mdl.nB, mdl.approx)
Lindbladian(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer) = _lindbladian(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), mdl.nB, mdl.approx)

DressedLiouvillian(â::QuantumObject, mdl::boson; kwargs...) = _dressed(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl); kwargs...)
DressedLiouvillian(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer; kwargs...) = _dressed(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), temperature(mdl); kwargs...)

Redfield(â::QuantumObject, mdl::boson) = _redfield(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl))
Redfield(â::QuantumObject, mdl::duffing) = _redfield(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl))
Redfield(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer) = _redfield(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), temperature(mdl))

# Optomechanics: a always Lindblad (see optomech); b Lindblad (RWA_env) or Ohmic Bloch–Redfield in the eigenbasis of the
# full rotating-frame H, with a's Lindblad terms passed as collapse operators. Returns (Ĥ, L̂s), resp. the superoperator.
function _cavity_jumps(â, mdl::optomech)
    (;γa, nBa) = mdl
    return [√(2γa*(1 + nBa))*â, √(2γa*nBa)*â']
end
function Lindbladian(â::QuantumObject, b̂::QuantumObject, mdl::optomech)
    (;γb, nBb, approx) = mdl
    RWA_env ∈ approx || @warn "Lindbladian called without RWA_env: the mechanical bath of this model is then Redfield." maxlog=1
    return Hamiltonian(â, b̂, mdl), [_cavity_jumps(â, mdl); √(2γb*(1 + nBb))*b̂; √(2γb*nBb)*b̂']
end
Redfield(â::QuantumObject, b̂::QuantumObject, mdl::optomech) = bloch_redfield_tensor(
    Hamiltonian(â, b̂, mdl), [(b̂ + b̂', ohmic(2mdl.γb/mdl.ωb, temperature(mdl.nBb, mdl.ωb)))], _cavity_jumps(â, mdl);
    sec_cutoff=-1, fock_basis=Val(true))

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

# Indices of the vectorised ρ of one mode with N Fock states (column-major, ρ[m, n] ↦ m + N n + 1 for 0-based m, n)
# in the parity sector (-1)^(m + n) = parity. A parity-conserving Liouvillian (x̂-coupled bath, even H) is block diagonal
# in these sectors; ⟨a⟩ and ⟨a'⟩ live in the odd one (parity = -1).
parity_sector(N, parity) = [m + N*n + 1 for n ∈ 0:N-1 for m ∈ 0:N-1 if iseven(m + n) == (parity == 1)]

# Smallest distance between two of the K slowest eigenvalues (largest real part) of L restricted to the indices idx,
# as (gap, λ1, λ2). A vanishing gap is a candidate EP; restricting to one symmetry sector avoids level crossings
# between sectors, which also close the gap but are not EPs. Dense.
function min_gap(L::QuantumObject, idx; K=6)
    λ = eigvals(Matrix(L.data[idx, idx]))
    λ = partialsort(λ, 1:K, by=real, rev=true)
    gaps = [(abs(λ[i] - λ[j]), λ[i], λ[j]) for i ∈ 1:K for j ∈ i+1:K]
    return gaps[argmin(first.(gaps))]
end

# Dominant ⟨a⟩ mode of L restricted to the sector idx of one mode: the eigenvalue (as ω = iλ) whose right eigenvector has
# the largest overlap |tr(a ρ)| + |tr(a'ρ)|, among those with |Re ω| < ωmax (excludes high-frequency cutoff artefacts).
# L is real, so ω and its mirror -conj(ω) have the same overlap: the one with Re ω ≥ 0 is returned.
# Returns (ω, largest Re λ in the sector), the latter as a stability check. Unambiguous on the underdamped side;
# past an EP other overdamped eigenvalues compete for the overlap. Dense.
function dominant_mode(L::QuantumObject, a::QuantumObject, idx; ωmax=3.)
    λ, V = eigen(Matrix(L.data[idx, idx]))
    ta, tad = vec(transpose(a.data))[idx], vec(transpose(a.data'))[idx]     # tr(a ρ) = vec(aᵀ) ⋅ vec(ρ)
    ok = findall(l -> abs(imag(l)) < ωmax, λ)
    w = [abs(dot(conj(ta), V[:, i])) + abs(dot(conj(tad), V[:, i])) for i ∈ ok]
    ω = im*λ[ok[argmax(w)]]
    return complex(abs(real(ω)), imag(ω)), maximum(real, λ)
end


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
        L = liouvillian(Lindbladian(â, duffing(ω0=1., γ=γs[j], g=gs[i], nB=0., approx=[RWA_env]))...)
        gap_map[i, j] = min_gap(L, idx_odd)[1]
    end
end
println("scan: ", length(gs)*length(γs), " points in ", round(t, digits=1), " s")

fig = Figure(size=(700, 550))
ax = Axis(fig[1, 1]; xlabel="g", ylabel="γ", title="log₁₀ minimal gap, odd sector, 6 slowest eigenvalues")
hm = heatmap!(ax, gs, γs, log10.(gap_map); colormap=:viridis)
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
        ω, reλ = dominant_mode(Redfield(a, duffing(ω0=1., γ=γs[j], g=gs[i], nB=0., approx=[])), a, odd)
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
hm = heatmap!(ax, gs, γs, half_gap; colormap=:viridis)
Colorbar(fig[1, 2], hm)
# points where the two cutoffs disagree (crosses) or the sector is unstable (red)
for (mask, marker, color) ∈ ((unconverged, :xcross, :white), (unstable, :circle, :red))
    bad = findall(mask)
    isempty(bad) || scatter!(ax, [gs[I[1]] for I ∈ bad], [γs[I[2]] for I ∈ bad]; marker, color, markersize=6)
end
scatter!(ax, [0.], [1.]; marker=:star5, color=:white, markersize=14)     # g = 0: critical damping γ = ω0
fig


#%% Generic definitions: monodromy tracking of Liouvillian eigenvalues

# Following a pair of Liouvillian eigenvalues continuously around a closed loop in parameter space: if they swap, the
# loop encloses an (odd number of) EP(s) of the pair. Method and validation: notes/monodromy_tracking.typ.
#   1. paths in parameter space                       ParamPath, Circle, Polygon, Segment
#   2. Liouvillians affine in the path parameters     AffineLiouvillian, derivative
#   3. starting eigenpairs (shift-invert)             eigenpairs_near, pair_near
#   4. bordered system and its linear solves          normalisation, bordered, bordered_mul, StaleLU, solve_bordered!
#   5. predictor–corrector tracking along a path      tangent, correct, track
#   6. locating an EP by bisection with edge reuse    TrackedLine, track_line, transport, rect_swaps, ep_bisect
# The parametrisation of a path (which model fields its coordinates δ move) and the choice of operators (e.g. a displaced
# frame) are written where they are used, in the function δ ↦ L(δ) given to AffineLiouvillian.

using SparseArrays
using OrdinaryDiffEqVerner
import QuantumToolbox: SVector
const P2 = SVector{2,Float64}

## 1. Paths in parameter space
# A path p is parametrised by u ∈ [0, 1] and provides position(p, u), velocity(p, u) = d position/du, and breakpoints(p),
# the u where the velocity jumps (passed as tstops to the ODE solver). Loops have position(p, 0) == position(p, 1).

abstract type ParamPath end
breakpoints(::ParamPath) = Float64[]

struct Circle <: ParamPath
    center::P2
    ρ::Float64
end
position(c::Circle, u) = c.center + c.ρ*P2(cospi(2u), sinpi(2u))
velocity(c::Circle, u) = 2π*c.ρ*P2(-sinpi(2u), cospi(2u))

# Closed polygon through the vertices in order (a rectangle is 4 vertices), uniform in u along each edge.
struct Polygon <: ParamPath
    vertices::Vector{P2}
end
# number of edges, index of the current edge (0-based) and position t ∈ [0, 1] along it
function _edge(p::Polygon, u)
    n = length(p.vertices)
    k = min(floor(Int, n*u), n - 1)
    return n, k, n*u - k
end
function position(p::Polygon, u)
    n, k, t = _edge(p, u)
    return (1 - t)*p.vertices[k + 1] + t*p.vertices[mod1(k + 2, n)]
end
function velocity(p::Polygon, u)
    n, k, _ = _edge(p, u)
    return n*(p.vertices[mod1(k + 2, n)] - p.vertices[k + 1])
end
breakpoints(p::Polygon) = collect(1:length(p.vertices) - 1) ./ length(p.vertices)

# Open straight path from a to b (not a loop): tracking along single edges, see section 6.
struct Segment <: ParamPath
    a::P2
    b::P2
end
position(p::Segment, u) = p.a + u*(p.b - p.a)
velocity(p::Segment, u) = p.b - p.a

## 2. Liouvillians affine in the path parameters
# L(δ) = L0 + Σₖ δₖ Lₖ (sparse matrices). Along a path p: L(u) = A(position(p, u)), dL/du = derivative(A, velocity(p, u)),
# so nothing is rebuilt along the path. The tracker (sections 5 and 6) only uses these two operations: any other type
# providing them (e.g. a non-affine family with finite-difference derivatives) can replace AffineLiouvillian.

struct AffineLiouvillian{M<:AbstractMatrix}
    L0::M
    Ls::Vector{M}
end
(A::AffineLiouvillian)(δ) = A.L0 + sum(δ[k]*A.Ls[k] for k ∈ eachindex(A.Ls))
derivative(A::AffineLiouvillian, v) = sum(v[k]*A.Ls[k] for k ∈ eachindex(A.Ls))

# From any function δ ↦ L(δ) of n coordinates that is affine: L0 = L(0), Lₖ = (L(h eₖ) - L0)/h, exact for an affine
# family up to round-off. The small step h keeps the model physical (e.g. positive rates) at the evaluation points.
# check = true evaluates L at δ = h(1, …, 1) and compares the deviation from the decomposition with the change of L over
# that step: the ratio is round-off for an affine family (~1e-14 here), and grows with h and the curvature otherwise
# (Redfield dimer: 7e-8); above rtol an error is raised.
function AffineLiouvillian(L_at::Function, n::Integer; h=1e-3, check=true, rtol=1e-10)
    L0 = L_at(zeros(n))
    e(k) = [j == k ? h : 0.0 for j ∈ 1:n]
    A = AffineLiouvillian(L0, typeof(L0)[(L_at(e(k)) - L0)/h for k ∈ 1:n])
    if check
        δ = fill(h, n)
        L1 = L_at(δ)
        ratio = norm(L1 - A(δ))/norm(L1 - L0)
        ratio ≤ rtol || throw(ArgumentError("L_at is not affine in its $n coordinates (relative deviation $ratio)"))
    end
    return A
end

## 3. Starting eigenpairs
# Eigenpairs (λ, r) of the sparse L nearest to each target λ, one shift-invert each. The shift is offset by δσ so that
# L - σ is not singular when a target is an exact eigenvalue.
function eigenpairs_near(L::AbstractMatrix, targets; δσ=1e-3)
    pairs = map(targets) do λt
        res = eigsolve(L; sigma=λt + δσ, eigvals=1)
        (res.values[1], res.vectors[:, 1])
    end
    allunique(round.(first.(pairs), digits=10)) || @warn "eigenpairs_near: two targets converged to the same eigenvalue"
    return pairs
end

# The two eigenpairs of L nearest σ ≈ the midpoint of a close pair, from a single shift-invert. Near an EP the pair is
# close, and separate searches from approximate targets can converge to the same eigenvalue.
function pair_near(L::AbstractMatrix, σ; δσ=1e-6)
    res = eigsolve(L; sigma=σ + δσ, eigvals=2)
    return [(res.values[i], res.vectors[:, i]) for i ∈ 1:2]
end

## 4. Bordered system and its linear solves
# An eigenpair is fixed by (L - λ) r = 0 and the normalisation c'r = 1, with c fixed. Its Jacobian in (r, λ) is
#     B = [L - λI  -r; c'  0],
# non-singular while λ is simple and c'r ≠ 0, singular at an EP (cond(B) ∝ distance^(-1/2) near an EP2).

# normalisation vector for r: c'r = 1
normalisation(r) = r/dot(r, r)

function bordered(L::SparseMatrixCSC, λ, r, c)
    T = promote_type(eltype(L), typeof(λ), eltype(r), eltype(c))
    return [L - λ*I         sparse(reshape(-r, :, 1))
            sparse(reshape(conj(c), 1, :))   spzeros(T, 1, 1)]
end

# B x, without assembling B
function bordered_mul(L::SparseMatrixCSC, λ, r, c, x)
    xr, xλ = @view(x[1:end-1]), x[end]
    return [L*xr - λ*xr - xλ*r; dot(c, xr)]
end

# B depends on (u, λ, r) but changes little along a path: solve B x = b with the LU of an earlier B ("stale"), refined by
# x ← x + F \ (b - B x) until ‖b - B x‖ ≤ rtol ‖b‖. Refactorise with the current B when refinement does not converge
# within maxiter iterations or stops decreasing, and after any solve that needed more than `early` iterations (B has
# drifted: keep the next solves cheap). Counts factorisations and refinement iterations.
mutable struct StaleLU
    F::Any                # factorisation of the reference B, nothing before the first solve
    rtol::Float64
    maxiter::Int
    early::Int
    factorisations::Int
    refinements::Int
end
StaleLU(; rtol=1e-12, maxiter=20, early=8) = StaleLU(nothing, rtol, maxiter, early, 0, 0)

function refactorise!(S::StaleLU, L, λ, r, c)
    S.F = lu(bordered(L, λ, r, c))
    S.factorisations += 1
    return S
end

function solve_bordered!(S::StaleLU, L, λ, r, c, b)
    if S.F !== nothing
        x = S.F \ b
        res_prev = Inf
        for it ∈ 0:S.maxiter
            res = b - bordered_mul(L, λ, r, c, x)
            nres = norm(res)
            if nres ≤ S.rtol*norm(b)
                it > S.early && refactorise!(S, L, λ, r, c)
                return x
            end
            nres < res_prev || break
            res_prev = nres
            x += S.F \ res
            S.refinements += 1
        end
    end
    refactorise!(S, L, λ, r, c)
    return S.F \ b
end
# without a StaleLU: fresh factorisation
solve_bordered!(::Nothing, L, λ, r, c, b) = bordered(L, λ, r, c) \ b

## 5. Predictor–corrector tracking along a path

# Predictor: (dr/du, dλ/du) at the eigenpair (λ, r) of L, from B [dr; dλ] = [-(dL/du) r; 0].
function tangent(L::SparseMatrixCSC, dL::SparseMatrixCSC, λ, r, c; solver=nothing)
    x = solve_bordered!(solver, L, λ, r, c, [-(dL*r); 0])
    return x[1:end-1], x[end]
end

# Corrector: Newton iterations at fixed u back onto an eigenpair of L, B [δr; δλ] = [-(L - λ) r; 1 - c'r].
# Returns (λ, r, residual ‖(L - λ)r‖/‖r‖ before correction, number of iterations).
function correct(L::SparseMatrixCSC, λ, r, c; tol=1e-12, maxiter=5, solver=nothing)
    res0 = norm(L*r - λ*r)/norm(r)
    res, it = res0, 0
    while (res > tol || abs(c'*r - 1) > tol) && it < maxiter
        x = solve_bordered!(solver, L, λ, r, c, [λ*r - L*r; 1 - c'*r])
        r = r + x[1:end-1]
        λ += x[end]
        res, it = norm(L*r - λ*r)/norm(r), it + 1
    end
    return λ, r, res0, it
end

# Follows the eigenpair (λ0, r0) of A (see section 2) along the path p by integrating d[r; λ]/du = tangent(...) over tspan (default the
# whole path), with the path's breakpoints as tstops. Returns (sol, diagnostics): λ(u) = sol(u)[end], r(u) = sol(u)[1:end-1].
#   corrector = true   Newton correction (correct) after every accepted step, and reset of c to normalisation(r) when
#                      cos∠(c, r) = 1/(‖c‖‖r‖) < reset_cos (c'r = 1 is kept, r unchanged; ‖r‖ would grow otherwise)
#   stale = true       bordered systems solved with StaleLU (false: fresh factorisation at every solve)
# c is the ODE parameter, reset in place by the callback: the solver must not interpolate lazily (Vern7(lazy=false)),
# lazy stages would be computed when sol(u) is called, with the latest c.
# diagnostics: corrections, Newton iterations, largest residual before a correction, c resets, factorisations, refinements.
function track(A, p::ParamPath, λ0, r0; alg=Vern7(lazy=false), reltol=1e-10, abstol=1e-12,
               corrector=true, tol=1e-12, reset_cos=0.1, stale=true, tspan=(0.0, 1.0), kwargs...)
    c = normalisation(r0)
    solver = stale ? StaleLU() : nothing
    function rhs!(dy, y, c, u)
        dr, dλ = tangent(A(position(p, u)), derivative(A, velocity(p, u)), y[end], @view(y[1:end-1]), c; solver)
        dy[1:end-1] .= dr
        dy[end] = dλ
        return nothing
    end

    diagnostics = (corrections=Ref(0), iterations=Ref(0), max_residual=Ref(0.), resets=Ref(0))
    function correct!(integ)
        c = integ.p
        λ, r, res0, it = correct(A(position(p, integ.t)), integ.u[end], integ.u[1:end-1], c; tol, solver)
        if 1/(norm(c)*norm(r)) < reset_cos
            c .= normalisation(r)
            diagnostics.resets[] += 1
        end
        integ.u[1:end-1] .= r
        integ.u[end] = λ
        diagnostics.corrections[] += 1
        diagnostics.iterations[] += it
        diagnostics.max_residual[] = max(diagnostics.max_residual[], res0)
        u_modified!(integ, true)
    end
    callback = corrector ? DiscreteCallback((y, u, integ) -> true, correct!; save_positions=(false, false)) : nothing

    prob = ODEProblem(rhs!, [r0; λ0], tspan, c)
    sol = solve(prob, alg; reltol, abstol, tstops=filter(u -> tspan[1] < u < tspan[2], breakpoints(p)), callback, kwargs...)
    lu_counts = stale ? (factorisations=solver.factorisations, refinements=solver.refinements) : (factorisations=-1, refinements=0)
    return sol, merge(map(getindex, diagnostics), lu_counts)
end

## 6. Locating an EP by bisection of rectangles, with edge reuse
# Both eigenvalues of the pair are tracked along straight lines (TrackedLine: one ODE solution per branch). Every side of
# every rectangle lies on a tracked line; following an eigenvalue along a side means finding the branch that matches it
# at the start and reading that branch at the end. A rectangle swaps the pair iff it encloses an odd number of their EPs.
# Splitting a rectangle only requires tracking the new dividing line: the other sides are parts of earlier lines.

struct TrackedLine
    a::P2
    b::P2
    sols::Vector{Any}          # one ODE solution per branch, u ∈ [0, 1] from a to b
end
branch(l::TrackedLine, j, s) = l.sols[j](s)[end]

# parameter s of the point x along l, or nothing if x is not on l
function line_param(l::TrackedLine, x; tol=1e-9)
    d = l.b - l.a
    s = dot(x - l.a, d)/dot(d, d)
    on = abs(d[1]*(x - l.a)[2] - d[2]*(x - l.a)[1]) ≤ tol*dot(d, d) && -tol ≤ s ≤ 1 + tol
    return on ? clamp(s, 0., 1.) : nothing
end

# the tracked line containing both x and y, with their parameters along it
function find_line(lines, x, y)
    for l ∈ lines
        sx, sy = line_param(l, x), line_param(l, y)
        sx !== nothing && sy !== nothing && return l, sx, sy
    end
    error("no tracked line contains both $x and $y")
end

# Tracks both eigenvalues of the pair from a to b, starting from shift-invert near the targets at a.
function track_line(A, a::P2, b::P2, targets; kwargs...)
    pairs = eigenpairs_near(A(a), targets)
    sols = Any[]
    for (λ0, r0) ∈ pairs
        sol, _ = track(A, Segment(a, b), λ0, r0; kwargs...)
        string(sol.retcode) == "Success" || error("tracking along $a → $b failed ($(sol.retcode))")
        push!(sols, sol)
    end
    return TrackedLine(a, b, sols)
end

# midpoint of the pair at x (on a tracked line, towards y): the shift for pair_near close to the EP
pair_midpoint(lines, x, y) = (l = find_line(lines, x, y); (branch(l[1], 1, l[2]) + branch(l[1], 2, l[2]))/2)

# Follows the eigenvalue λ from x to y along the tracked line containing both. The branch must be unambiguous: its
# distance to λ at x below `ratio` times the gap between the two branches there.
function transport(lines, λ, x, y; ratio=0.25)
    l, sx, sy = find_line(lines, x, y)
    v = [branch(l, j, sx) for j ∈ eachindex(l.sols)]
    d = abs.(v .- λ)
    j = argmin(d)
    d[j] ≤ ratio*abs(v[1] - v[2]) || error("ambiguous branch at $x (the EP may lie on this line)")
    return branch(l, j, sy)
end

# Does the rectangle (x0, x1, y0, y1) swap the pair? Follows one eigenvalue counterclockwise from (x0, y0).
function rect_swaps(lines, (x0, x1, y0, y1))
    c = (P2(x0, y0), P2(x1, y0), P2(x1, y1), P2(x0, y1))
    l, s, _ = find_line(lines, c[1], c[2])
    λa, λb = branch(l, 1, s), branch(l, 2, s)
    λ = λa
    for k ∈ 1:4
        λ = transport(lines, λ, c[k], c[mod1(k + 1, 4)])
    end
    return abs(λ - λb) < abs(λ - λa)
end

# Bisection from a rectangle (x0, x1, y0, y1) that swaps the pair: split across the longer side, track the dividing line,
# keep the half that swaps (exactly one must). A dividing line that fails (EP too close to it) is moved by `shift` of the
# side, up to three times. corner_targets(x) gives the starting targets at the outer corners.
# Returns (final rectangle, rectangles at every level, all tracked lines).
function ep_bisect(A, rect, corner_targets; depth=12, shift=0.1, kwargs...)
    x0, x1, y0, y1 = rect
    c = (P2(x0, y0), P2(x1, y0), P2(x1, y1), P2(x0, y1))
    lines = TrackedLine[track_line(A, c[1], c[2], corner_targets(c[1]); kwargs...),
                        track_line(A, c[2], c[3], corner_targets(c[2]); kwargs...),
                        track_line(A, c[4], c[3], corner_targets(c[4]); kwargs...),
                        track_line(A, c[1], c[4], corner_targets(c[1]); kwargs...)]
    rect_swaps(lines, rect) || error("the initial rectangle does not swap the pair: no (or an even number of) EP inside")
    history = [rect]
    for level ∈ 1:depth
        x0, x1, y0, y1 = rect
        vertical = (x1 - x0) ≥ (y1 - y0)                 # split across the longer side
        for attempt ∈ 0:3
            f = 0.5 + shift*attempt*(isodd(attempt) ? 1 : -1)/2
            m = vertical ? x0 + f*(x1 - x0) : y0 + f*(y1 - y0)
            a, b = vertical ? (P2(m, y0), P2(m, y1)) : (P2(x0, m), P2(x1, m))
            halves = vertical ? ((x0, m, y0, y1), (m, x1, y0, y1)) : ((x0, x1, y0, m), (x0, x1, m, y1))
            try
                # starting targets at a: the two branches of the side it lies on
                l, s, _ = find_line(lines, a, vertical ? P2(x1, y0) : P2(x0, y1))
                new = track_line(A, a, b, [branch(l, j, s) for j ∈ 1:2]; kwargs...)
                sw = [rect_swaps([lines; new], h) for h ∈ halves]
                count(sw) == 1 || error("$(count(sw)) halves swap")
                push!(lines, new)
                rect = halves[findfirst(sw)]
                break
            catch err
                attempt == 3 && rethrow()
                @warn "level $level: dividing line at f = $f failed ($(sprint(showerror, err))), shifting it"
            end
        end
        push!(history, rect)
    end
    return rect, history, lines
end


#%% Monodromy, coupled bosons: tracking the Liouvillian modes around the EP (Lindblad, RWA_env)

# Lindblad dimer in the plane δ = (δω, δγ): detuning ±δω and damping asymmetry ±δγ around ω0 = 1, γ0 = 0.1, g = 0.01.
# EPs at δω = 0, δγ = ±g/2. Both modes are tracked around five loops and compared with -iω from the roots of disc
# (same parametrisation). Expected: a swap for a loop around one EP, none around zero or two EPs.
n_fock = 5
â = destroy(n_fock) ⊗ eye(n_fock)
b̂ = eye(n_fock) ⊗ destroy(n_fock)
g = 0.01
mdl = bosonDimer(g=g, approx=[RWA_env])
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
        t = @elapsed sol, diag = track(A, q, λ0, r0)
        lands = argmin(abs.(first.(pairs0) .- sol.u[end][end]))
        dev = maximum(minimum(abs.(sol(u)[end] .- disc_λ(position(q, u)))) for u ∈ us)
        println("  mode $k → mode $lands,  max |λ - λ_disc| = ", round(dev, sigdigits=2), ",  ", sol.stats.naccept, " steps, ",
            diag.factorisations, " LU, ", round(t, digits=2), " s")
        ω = [im*sol(u)[end] for u ∈ us]
        lines!(ax, real(ω), imag(ω); color=Cycled(k), label="mode $k" * (lands == k ? " (returns)" : " → mode $lands"))
        scatter!(ax, [real(im*λ0)], [imag(im*λ0)]; color=Cycled(k), marker=:star5, markersize=14)
    end
    axislegend(ax; position=:rb, labelsize=10)
end
ylims!(ax_p, -0.012, 0.075)                 # room for the legend above the loops
axislegend(ax_p; position=:ct, labelsize=9)
fig


#%% Driven optomechanics: setup (linearised EP, displaced-frame Liouvillian)

# Linearised fluctuations ≡ bosonDimer with ωa = -(Δ + g x̄) and coupling 2g|α| (linearised_dimer): EP at resonance and
# 2g|α| = |γa - γb| (linearised_EP). The full model keeps the non-linear term -g d†d (e + e†). It is written in the
# displaced frame a = α + d, b = β + e (reference at the linearised EP), where n_fock = 6 per mode is converged to ~1e-7.
# The next two cells work in the plane δ = (Δ, F/F_EP), where L is affine (H is linear in Δ and F).
mdl = optomech(ωb=1., γa=0.2, γb=0.05, g=0.1, nBa=0., nBb=0., approx=[RWA_env])
Δ_EP, F_EP, n_branches = linearised_EP(mdl)
point(δ) = setproperties(mdl, (Δ=δ[1], F=δ[2]*F_EP))
println("linearised EP: Δ = ", round(Δ_EP, digits=5), ", F = ", round(F_EP, digits=5), ", ", n_branches, " classical branch(es)")

Na, Nb = 6, 6
â = destroy(Na) ⊗ eye(Nb)
b̂ = eye(Na) ⊗ destroy(Nb)
α, β = classical_displacement(point(P2(Δ_EP, 1.0)))
âd, b̂d = â + α*one(â), b̂ + β*one(b̂)        # displaced frame: the model functions see a = α + d, b = β + e
A = AffineLiouvillian(δ -> liouvillian(Lindbladian(âd, b̂d, point(δ))...).data, 2)

# starting targets at δ = (Δ, F/F_EP): the linearised modes, λ = -iω (≈ 0.01 from the full ones at g = 0.1)
linearised_targets(δ) = -im .* filter(r -> real(r) > 0, roots_sorted(linearised_dimer(point(δ))))


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
scatter!(ax_p, [Δ_EP], [1.0]; marker=:star5, markersize=16, color=:white, label="linearised EP")
for (j, (name, q)) ∈ enumerate(om_loops)
    δs = position.(Ref(q), us)
    lines!(ax_p, first.(δs), last.(δs); color=Cycled(j + 2), label=name)
    ax = Axis(fig[fldmod1(j + 1, 2)...]; xlabel="Re ω", ylabel="Im ω", title=name)
    pairs0 = eigenpairs_near(A(position(q, 0.)), linearised_targets(position(q, 0.)))
    println(name)
    for (k, (λ0, r0)) ∈ enumerate(pairs0)
        t = @elapsed sol, diag = track(A, q, λ0, r0)
        lands = argmin(abs.(first.(pairs0) .- sol.u[end][end]))
        println("  mode $k → mode $lands,  |λ(1) - λ_start| = ", round(abs(sol.u[end][end] - first(pairs0[lands])), sigdigits=2),
            ",  ", sol.stats.naccept, " steps, ", diag.factorisations, " LU, ", round(t, digits=1), " s")
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
t = @elapsed rect, hist, lines = ep_bisect(A, rect0, linearised_targets; depth=16)
x0, x1, y0, y1 = rect
Δ_c, f_c = (x0 + x1)/2, (y0 + y1)/2
println("bisection: ", length(hist) - 1, " levels, ", length(lines), " tracked lines, ", round(t, digits=1), " s")
println("EP of the full model: Δ = ", round(Δ_c, digits=5), " ± ", round((x1 - x0)/2, sigdigits=2),
    ",  F = ", round(f_c*F_EP, digits=5), " ± ", round((y1 - y0)/2*F_EP, sigdigits=2))
println("shift from the linearised EP: δΔ = ", round(Δ_c - Δ_EP, sigdigits=3), ",  δF = ", round((f_c - 1)*F_EP, sigdigits=3),
    " (", round(100(f_c - 1), sigdigits=3), " %)")

# midpoint of the pair at the corner (x0, y0) of a box: shift for pair_near (the pair is close near the EP)
box_midpoint(r) = pair_midpoint(lines, P2(r[1], r[3]), P2(r[2], r[3]))

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
    lines!(ax, [r[1], r[2], r[2], r[1], r[1]], [r[3], r[3], r[4], r[4], r[3]]; color=k, colorrange=(1, length(hist)), colormap=:viridis)
end
for ax ∈ (ax1, ax2)
    scatter!(ax, [Δ_EP], [1.0]; marker=:star5, markersize=14, color=:white, label="linearised EP")
    scatter!(ax, [Δ_c], [f_c]; marker=:xcross, markersize=12, color=:red, label="full EP (box centre)")
end
zr = hist[min(7, end)]
limits!(ax2, zr[1], zr[2], zr[3], zr[4])
axislegend(ax1; position=:rt, labelsize=10)
ax3 = Axis(fig[1, 3]; xscale=log10, yscale=log10, xlabel="box diagonal", ylabel="|λ₁ - λ₂| at the box centre", title="gap ∝ √size")
scatter!(ax3, sizes, gaps)
lines!(ax3, sizes, gaps[end]*sqrt.(sizes ./ sizes[end]); linestyle=:dash, color=:gray, label="∝ √size")
axislegend(ax3; position=:lt)
fig
