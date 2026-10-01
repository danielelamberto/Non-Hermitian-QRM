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


#%% Generic definitions: monodromy (closed paths in parameter space)

# A path p is defined for u ∈ [0, 1], with position(p, 0) == position(p, 1), and provides
# position(p, u), velocity(p, u) = d position/du, and breakpoints(p): the u where velocity jumps (tstops for ODE solvers).

using SparseArrays
using OrdinaryDiffEqVerner
import QuantumToolbox: SVector
const P2 = SVector{2,Float64}

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

# Dimer detuned by ±δω and with damping asymmetry ±δγ around (ω0, γ0); g, nB and approx are taken from mdl.
dimer_at(mdl::bosonDimer, (δω, δγ); ω0=1., γ0=0.1) =
    setproperties(mdl, (ωa=ω0 + δω, ωb=ω0 - δω, γa=γ0 + δγ, γb=γ0 - δγ))

# Liouvillian (as a sparse matrix) affine in the path parameters, L(δ) = L0 + Σₖ δₖ Lₖ.
# Along a path p: L(u) = A(position(p, u)) and dL/du = derivative(A, velocity(p, u)).
struct AffineLiouvillian{M<:AbstractMatrix}
    L0::M
    Ls::Vector{M}
end
(A::AffineLiouvillian)(δ) = A.L0 + sum(δ[k]*A.Ls[k] for k ∈ eachindex(A.Ls))
derivative(A::AffineLiouvillian, v) = sum(v[k]*A.Ls[k] for k ∈ eachindex(A.Ls))

# Lindblad dimer as a function of (δω, δγ), see dimer_at: the Hamiltonian is linear in ωa, ωb and the dissipators in γa, γb.
# L_ω = -i[a'a - b'b, ⋅],  L_γ = 2(1 + nB)(D[a] - D[b]) + 2nB(D[a'] - D[b']).
function lindblad_affine(â::QuantumObject, b̂::QuantumObject, mdl::bosonDimer; ω0=1., γ0=0.1)
    nB = mdl.nB
    L0 = liouvillian(Lindbladian(â, b̂, dimer_at(mdl, P2(0, 0); ω0, γ0))...).data
    Lω = liouvillian(â'*â - b̂'*b̂).data
    Lγ = (2(1 + nB)*(lindblad_dissipator(â) - lindblad_dissipator(b̂)) + 2nB*(lindblad_dissipator(â') - lindblad_dissipator(b̂'))).data
    return AffineLiouvillian(L0, typeof(L0)[Lω, Lγ])
end

# Eigenpairs (λ, r) of the sparse matrix L nearest to each target λ (shift-invert): the starting points for tracking.
# The shift is offset by δσ so that L - σ is not singular when a target is an exact eigenvalue.
function eigenpairs_near(L::AbstractMatrix, targets; δσ=1e-3)
    pairs = map(targets) do λt
        res = eigsolve(L; sigma=λt + δσ, eigvals=1)
        (res.values[1], res.vectors[:, 1])
    end
    allunique(round.(first.(pairs), digits=10)) || @warn "eigenpairs_near: two targets converged to the same eigenvalue"
    return pairs
end

## Eigenpair tracking: (L(u) - λ) r = 0 with the normalisation c'r = 1, c fixed.
# Differentiating in u gives the bordered system  B [dr; dλ] = [-(dL/du) r; 0],  B = [L - λI  -r; c'  0].
# B is non-singular while λ is simple and c'r ≠ 0, and becomes singular at an EP. It depends on (u, λ, r), but changes
# little along the path: systems are solved with an LU of an earlier B plus iterative refinement (StaleLU).

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

# Solves B x = b with the LU of an earlier B (stale), refined by x ← x + F \ (b - B x) until ‖b - B x‖ ≤ rtol ‖b‖.
# Refactorises with the current B when refinement does not converge within maxiter iterations or stops decreasing,
# and also after a converged solve that needed more than `early` iterations (B has drifted: keep later solves cheap).
# One refinement costs ~1/50 of a factorisation here (n_fock = 5), and converges by a factor ~0.07 per typical step.
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

# (dr/du, dλ/du) at the eigenpair (λ, r) of L, given dL = dL/du
function tangent(L::SparseMatrixCSC, dL::SparseMatrixCSC, λ, r, c; solver=nothing)
    x = solve_bordered!(solver, L, λ, r, c, [-(dL*r); 0])
    return x[1:end-1], x[end]
end

# Newton correction of (λ, r) back onto an eigenpair of L at fixed c: solves (L - λ)r = 0, c'r = 1.
# The Jacobian of these equations is the bordered matrix B.
# Returns (λ, r, residual before correction, number of iterations), residual = ‖(L - λ)r‖/‖r‖.
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

# Follow the eigenpair (λ0, r0) of A along the closed path p, u ∈ [0, 1], by integrating d[r; λ]/du = tangent(...).
# The state is y = [r; λ]; λ(u) = sol(u)[end], r(u) = sol(u)[1:end-1]. The path's breakpoints are passed as tstops.
# With corrector = true, every accepted step is followed by a Newton correction (see correct), and the normalisation
# vector c is reset to normalisation(r) when cos∠(c, r) = 1/(‖c‖‖r‖) drops below reset_cos (c'r = 1 is kept, r is unchanged).
# With stale = true, the bordered systems reuse an earlier LU with iterative refinement (StaleLU); false refactorises every time.
# Returns (sol, diagnostics): number of corrections, Newton iterations, largest residual before a correction, c resets,
# LU factorisations and refinement iterations.
# The solver must not use lazy interpolation (Vern7(lazy=false)): lazy stages are computed when sol(u) is called,
# with the current c, which is wrong for steps taken before a reset.
function track(A::AffineLiouvillian, p::ParamPath, λ0, r0; alg=Vern7(lazy=false), reltol=1e-10, abstol=1e-12,
               corrector=true, tol=1e-12, reset_cos=0.1, stale=true, kwargs...)
    c = normalisation(r0)        # passed as the ODE parameter, so that the callback can reset it in place
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

    prob = ODEProblem(rhs!, [r0; λ0], (0.0, 1.0), c)
    sol = solve(prob, alg; reltol, abstol, tstops=breakpoints(p), callback, kwargs...)
    lu_counts = stale ? (factorisations=solver.factorisations, refinements=solver.refinements) : (factorisations=-1, refinements=0)
    return sol, merge(map(getindex, diagnostics), lu_counts)
end


#%% Monodromy test 

g = 0.01
mdl = bosonDimer(g=g, approx=[RWA_env])

# EPs at δω = 0. EP(⋅) returns g_EP ∝ |γa - γb| there, so rescale to find the δγ where g_EP = g.
# (Valid with RWA_env; without it the EPs move to δω = (γ0/ω0) δγ.)
δγ_EP = g*0.01/EP(dimer_at(mdl, P2(0, 0.01)))[1]

paths = [
    ("circle",    Circle(P2(0, 0.01), 0.01)),
    ("rectangle", Polygon([P2(-0.01, 0), P2(0.01, 0), P2(0.01, 0.02), P2(-0.01, 0.02)])),
]

# velocity must be the derivative of position (centred finite difference, away from the polygon corners)
for (name, p) ∈ paths, u ∈ (0.1, 0.4, 0.6, 0.9)
    h = 1e-6
    err = maximum(abs.((position(p, u + h) - position(p, u - h))/2h - velocity(p, u)))
    err < 1e-6 || @warn "velocity of the $name path is not d position/du at u = $u" err
end

n_samples = 60
us = range(0, 1, n_samples + 1)[1:end-1]
colors = resample_cmap(:twilight, n_samples)

fig = Figure(size=(1000, 900))
for (j, (name, p)) ∈ enumerate(paths)
    ax_param = Axis(fig[1, j]; xlabel="δω", ylabel="δγ", title=name)
    ax_cmplx = Axis(fig[2, j]; xlabel="Re ω", ylabel="Im ω")
    scatter!(ax_param, [0, 0], [δγ_EP, -δγ_EP]; marker=:star4, markersize=20, color=:gray)
    for (u, color) ∈ zip(us, colors)
        δ = position(p, u)
        scatter!(ax_param, δ[1], δ[2]; color)
        ω = filter(r -> real(r) > 0, roots_sorted(dimer_at(mdl, δ)))
        scatter!(ax_cmplx, real(ω), imag(ω); color)
    end
end
fig

#%% Lindbladian eigenvalue tracking. 

# Step 1: Liouvillian along the loop, its u-derivative, and the starting eigenpairs.
# Same model and circle as the monodromy test; n_fock = 5 per mode, so L is 625×625 (sparse).
n_fock = 5
â = destroy(n_fock) ⊗ eye(n_fock)
b̂ = eye(n_fock) ⊗ destroy(n_fock)

g = 0.01
mdl = bosonDimer(g=g, approx=[RWA_env])
p = Circle(P2(0, 0.01), 0.01)
A = lindblad_affine(â, b̂, mdl)

# reference: Liouvillian rebuilt from scratch at a point of the path
L_direct(u) = liouvillian(Lindbladian(â, b̂, dimer_at(mdl, position(p, u)))...).data

# 1. affine decomposition: L0 + δω Lω + δγ Lγ equals the direct construction
err_affine = maximum(norm(A(position(p, u)) - L_direct(u)) for u ∈ (0., 0.3, 0.7))
println("‖A(δ) - L_direct‖ = ", err_affine)

# 2. dL/du = derivative(A, velocity) against a centred finite difference of the direct construction (relative error)
h = 1e-6
err_dL = maximum(u -> norm((L_direct(u + h) - L_direct(u - h))/2h - derivative(A, velocity(p, u)))/norm(derivative(A, velocity(p, u))), (0.1, 0.45, 0.8))
println("dL/du vs finite difference (relative) = ", err_dL)

# 3. starting pair at u = 0: Liouvillian eigenvalues λ = -iω nearest to the positive-frequency roots of disc
L_start = A(position(p, 0.))
targets = -im .* filter(r -> real(r) > 0, roots_sorted(dimer_at(mdl, position(p, 0.))))
start_pairs = eigenpairs_near(L_start, targets)
for ((λ, r), λt) ∈ zip(start_pairs, targets)
    println("target ", λt, "  found ", λ, "  |Δλ| = ", abs(λ - λt), "  ‖(L - λ)r‖/‖r‖ = ", norm(L_start*r - λ*r)/norm(r))
end


# Step 3: one tangent step of the bordered system at fixed u, for both modes.
pos_roots_at(u) = filter(r -> real(r) > 0, roots_sorted(dimer_at(mdl, position(p, u))))
h = 1e-5
for u ∈ (0.1, 0.45, 0.8)
    L, dL = A(position(p, u)), derivative(A, velocity(p, u))
    println("u = $u")
    for (λ, r) ∈ eigenpairs_near(L, -im .* pos_roots_at(u))
        c = normalisation(r)
        dr, dλ = tangent(L, dL, λ, r, c)

        # 1. first-order perturbation theory, with the left eigenvector l (L'l = conj(λ) l)
        (_, l), = eigenpairs_near(sparse(L'), [conj(λ)])
        dλ_pt = (l'*dL*r)/(l'*r)

        # 2. analytic slope from the roots of disc, λ = -iω
        nearest(rs, x) = rs[argmin(abs.(rs .- x))]
        dλ_disc = -im*(nearest(pos_roots_at(u + h), im*λ) - nearest(pos_roots_at(u - h), im*λ))/2h

        # 3. dr against eigenvectors at u ± h, scaled to the same normalisation c'r = 1
        function r_at(v)
            (_, rv), = eigenpairs_near(A(position(p, v)), [λ + (v - u)*dλ])
            return rv/(c'*rv)
        end
        dr_fd = (r_at(u + h) - r_at(u - h))/2h

        # 4. condition number of B
        κ = cond(Matrix(bordered(L, λ, r, c)))

        println("  λ = ", round(λ, digits=6), "  dλ = ", round(dλ, digits=6),
            "  |dλ - dλ_pt| = ", abs(dλ - dλ_pt), "  |dλ - dλ_disc| = ", abs(dλ - dλ_disc),
            "  ‖dr - dr_fd‖/‖dr‖ = ", norm(dr - dr_fd)/norm(dr), "  cond(B) = ", round(κ, sigdigits=3))
    end
end

# cond(B) grows as the point approaches the EP at (0, δγ_EP) = (0, g/2)
for d ∈ (1e-2, 1e-3, 1e-4, 1e-5)
    L = A(P2(0, g/2 + d))
    λ, r = eigenpairs_near(L, [-im*filter(r -> real(r) > 0, roots_sorted(dimer_at(mdl, P2(0, g/2 + d))))[1]])[1]
    println("distance $d to the EP:  cond(B) = ", round(cond(Matrix(bordered(L, λ, r, normalisation(r)))), sigdigits=3))
end


# Step 4: track both modes around closed loops, with the Newton corrector, refactorising B every time or reusing a stale LU.
# Expected: swap for a loop around one EP, none around zero or two EPs. Along the way λ(u) must stay on -iω from disc.
loops = [
    ("circle, 1 EP",          Circle(P2(0, 0.01), 0.01)),
    ("rectangle, 1 EP",       Polygon([P2(-0.01, 0), P2(0.01, 0), P2(0.01, 0.02), P2(-0.01, 0.02)])),
    ("circle, no EP",         Circle(P2(0, 0.03), 0.01)),
    ("circle, both EPs",      Circle(P2(0, 0), 0.01)),
    ("small circle, 1 EP at 1e-4", Circle(P2(0, g/2 + 1e-4), 2e-4)),
]
us_check = range(0, 1, 401)

fig = Figure(size=(1000, 1300))
for (j, (name, q)) ∈ enumerate(loops)
    disc_λ(u) = -im .* filter(r -> real(r) > 0, roots_sorted(dimer_at(mdl, position(q, u))))
    pairs0 = eigenpairs_near(A(position(q, 0.)), disc_λ(0.))
    ax = Axis(fig[fldmod1(j, 2)...]; xlabel="Re ω", ylabel="Im ω", title=name)
    println(name)
    for (k, (λ0, r0)) ∈ enumerate(pairs0), stale ∈ (false, true)
        t = @elapsed sol, diag = track(A, q, λ0, r0; stale)
        λ1, r1 = sol.u[end][end], sol.u[end][1:end-1]
        lands_on = argmin(abs.(first.(pairs0) .- λ1))
        dev = maximum(minimum(abs.(sol(u)[end] .- disc_λ(u))) for u ∈ us_check)
        res = norm(A(position(q, 1.))*r1 - λ1*r1)/norm(r1)
        println("  mode $k → $lands_on", stale ? "  stale LU   " : "  fresh LU   ",
            "max |λ - λ_disc| = ", round(dev, sigdigits=2), ",  final residual ", round(res, sigdigits=2),
            ",  ", sol.stats.naccept, " steps, ", sol.stats.nf, " rhs, ", round(t, digits=2), " s",
            ",  $(diag.iterations) Newton its, $(diag.resets) c resets",
            stale ? ",  $(diag.factorisations) LU, $(diag.refinements) refinements" : "")
        stale || continue
        ω = [im*sol(u)[end] for u ∈ us_check]
        lines!(ax, real(ω), imag(ω); color=Cycled(k), label="mode $k")
        scatter!(ax, [real(im*λ0)], [imag(im*λ0)]; color=Cycled(k), marker=:star5, markersize=14)
    end
    j == 1 && axislegend(ax; position=:lt)
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
