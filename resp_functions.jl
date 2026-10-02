#%% Generic definitions: models, analytical response, master equations

using QuantumToolbox
using CairoMakie
using MakieStyles

using LinearAlgebra
using SparseArrays
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

## Mock Liouvillian with planted, analytically known singularities (test bed for the monodromy tools)
# Direct sum of small independent Lindblad blocks; each block's frequencies and rates are affine in the two parameters
# (x, y) ∈ [0, 1]², so the Liouvillian is affine in (x, y), and its spectrum is known in closed form:
#   DimerBlock      levels |0⟩, |a⟩, |b⟩: single-excitation sector of two coupled damped modes, ωa,b = ω0 ± s(x - x0),
#                   γa,b = γ0 ± s(y - y0), coupling g/2. Pair of EP2 of the coherences |a⟩⟨0|, |b⟩⟨0| at
#                   (x0, y0 ± g/2s), λ = -γ0 - iω0 (and conjugate); g = 0: DP at (x0, y0).
#   DrivenQubit     resonant drive Ω/2 σx, decay γ (Ω, γ affine): line of real-axis EP2 where Ω = γ/4, λ = -3γ/4.
#   DetunedQubit    H = Δ|e⟩⟨e|, decay γ (Δ affine): the pair -γ/2 ± iΔ touches the real axis on the line Δ = 0
#                   without coalescing (DP line, no EP).
# Coherences between different blocks have eigenvalues repeated once per other block (degenerate everywhere), so
# mock_matrix restricts L to a sector (the block label is conserved):
#   sector = :coherences (default)  dimers: |x⟩⟨0|, |0⟩⟨x| (closed: the excitation difference is conserved); detuned
#                                   qubit: |e⟩⟨g|, |g⟩⟨e|; driven qubit: whole block. 20 states, and the only singularities
#                                   are the planted ones (the blocks' eigenvalues never meet on [0, 1]²).
#   sector = :within                all |i_k⟩⟨j_k| (35 states). The dimers' populations add, on the real axis: a DP line
#                                   x = x0 (outside the EP segment), a crossing of two real eigenvalues on y = y0, a 4-fold
#                                   coalescence at -2γ0 at the EP points, and real crossings between blocks.

abstract type MockBlock end
struct DimerBlock <: MockBlock
    center::NTuple{2,Float64}
    s::Float64
    ω0::Float64
    γ0::Float64
    g::Float64
end
struct DrivenQubit <: MockBlock
    Ω::NTuple{3,Float64}       # Ω = Ω[1] + Ω[2] x + Ω[3] y
    γ::NTuple{3,Float64}
end
struct DetunedQubit <: MockBlock
    Δ::NTuple{3,Float64}       # Δ = Δ[1] + Δ[2] x + Δ[3] y
    γ::Float64
end
affine3(c, x, y) = c[1] + c[2]*x + c[3]*y

# local Hamiltonian and jump operators of a block at (x, y) (plain matrices; |0⟩ or |g⟩ first)
function block_terms(b::DimerBlock, x, y)
    δω, δγ = b.s*(x - b.center[1]), b.s*(y - b.center[2])
    γa, γb = b.γ0 + δγ, b.γ0 - δγ
    H = ComplexF64[0 0 0; 0 b.ω0+δω b.g/2; 0 b.g/2 b.ω0-δω]
    return H, [ComplexF64[0 √(2γa) 0; 0 0 0; 0 0 0], ComplexF64[0 0 √(2γb); 0 0 0; 0 0 0]]
end
function block_terms(b::DrivenQubit, x, y)
    Ω, γ = affine3(b.Ω, x, y), affine3(b.γ, x, y)
    return ComplexF64[0 Ω/2; Ω/2 0], [ComplexF64[0 √γ; 0 0]]
end
block_terms(b::DetunedQubit, x, y) = ComplexF64[0 0; 0 affine3(b.Δ, x, y)], [ComplexF64[0 √b.γ; 0 0]]
block_dim(b::MockBlock) = size(block_terms(b, 0.5, 0.5)[1], 1)

# Closed-form eigenvalues of each block's own Liouvillian. Dimer and detuned qubit: the only jumps go to the ground level,
# so L is triangular and its eigenvalues are -i(εᵢ - conj(εⱼ)) over the eigenvalues ε of H_eff = H - (i/2)Σ J†J
# (ground level ε = 0). Driven qubit: Bloch equations, 0, -γ/2, -3γ/4 ± √(γ²/16 - Ω²).
function effective_energies(b::DimerBlock, x, y)
    δω, δγ = b.s*(x - b.center[1]), b.s*(y - b.center[2])
    r = sqrt(complex((δω - im*δγ)^2 + b.g^2/4))
    return [0.0im, b.ω0 - im*b.γ0 + r, b.ω0 - im*b.γ0 - r]
end
effective_energies(b::DetunedQubit, x, y) = [0.0im, affine3(b.Δ, x, y) - im*b.γ/2]
# elements |i⟩⟨j| (local levels, ground first) of a block kept in a sector
sector_elements(b::MockBlock, sector) = (d = block_dim(b); [(i, j) for j ∈ 1:d for i ∈ 1:d])
sector_elements(b::Union{DimerBlock,DetunedQubit}, sector) =
    sector == :coherences ? [(i, j) for j ∈ 1:block_dim(b) for i ∈ 1:block_dim(b) if (i == 1) ⊻ (j == 1)] :
                            [(i, j) for j ∈ 1:block_dim(b) for i ∈ 1:block_dim(b)]

# eigenvalue of the element |i⟩⟨j| (exact for the triangular blocks)
function block_spectrum(b::MockBlock, x, y; sector=:coherences)
    ε = effective_energies(b, x, y)
    return [-im*(ε[i] - conj(ε[j])) for (i, j) ∈ sector_elements(b, sector)]
end
function block_spectrum(b::DrivenQubit, x, y; sector=:coherences)
    Ω, γ = affine3(b.Ω, x, y), affine3(b.γ, x, y)
    r = sqrt(complex(γ^2/16 - Ω^2))
    return [0.0im, -γ/2 + 0im, -3γ/4 + r, -3γ/4 - r]
end

struct mockLiouvillian <: Model
    x::Float64
    y::Float64
    blocks::Vector{MockBlock}
end
# Default test bed: a pair of weak EP2 1e-3 apart, a pair of EP2 0.2 apart, an off-axis DP, a real-axis EP line and a
# DP line. Generic (non-round) values, so that no point or line falls on the nodes of regular or refined grids; all
# rates stay positive on [0, 1]².
default_mock_blocks() = MockBlock[
    DimerBlock((0.3071, 0.2943), 0.1,  1.0, 0.1,  1e-4),    # EP2 pair at (0.3071, 0.2943 ± 5e-4)
    DimerBlock((0.6113, 0.7031), 0.15, 1.3, 0.15, 0.03),    # EP2 pair at (0.6113, 0.6031) and (0.6113, 0.8031)
    DimerBlock((0.7489, 0.2617), 0.2,  0.7, 0.2,  0.0),     # DP at (0.7489, 0.2617)
    DrivenQubit((0.0213, 0.0587, 0.0), (0.1, 0.0, 0.2)),   # real-axis EP line Ω = γ/4: y = 1.174x - 0.074
    DetunedQubit((-0.3071, 0.5, 0.2133), 0.4),             # DP line 0.5x + 0.2133y = 0.3071 (meets the EP line at λ ≠ -3γ/4)
]
mockLiouvillian(; x=0.5, y=0.5, blocks=default_mock_blocks()) = mockLiouvillian(x, y, blocks)

# Block-diagonal H and jump operators of the whole system, with the level ranges of the blocks.
function block_ranges(mdl::mockLiouvillian)
    d = block_dim.(mdl.blocks)
    return [sum(d[1:k-1]) + 1:sum(d[1:k]) for k ∈ eachindex(d)]
end
function Lindbladian(mdl::mockLiouvillian)
    rs = block_ranges(mdl)
    D = last(last(rs))
    H = zeros(ComplexF64, D, D)
    jumps = QuantumObject[]
    for (b, r) ∈ zip(mdl.blocks, rs)
        Hb, Jb = block_terms(b, mdl.x, mdl.y)
        H[r, r] .= Hb
        for J ∈ Jb
            Jfull = zeros(ComplexF64, D, D)
            Jfull[r, r] .= J
            push!(jumps, Qobj(sparse(Jfull)))
        end
    end
    return Qobj(sparse(H)), jumps
end

# indices of the sector's elements in the vectorised ρ (column-major), and L restricted to them (sparse)
function mock_sector(mdl::mockLiouvillian; sector=:coherences)
    D = last(last(block_ranges(mdl)))
    return sort([r[i] + D*(r[j] - 1) for (b, r) ∈ zip(mdl.blocks, block_ranges(mdl)) for (i, j) ∈ sector_elements(b, sector)])
end
mock_matrix(mdl::mockLiouvillian; sector=:coherences) =
    (idx = mock_sector(mdl; sector); liouvillian(Lindbladian(mdl)...).data[idx, idx])

# exact spectrum of mock_matrix
mock_spectrum(mdl::mockLiouvillian; sector=:coherences) =
    reduce(vcat, block_spectrum(b, mdl.x, mdl.y; sector) for b ∈ mdl.blocks)

# Planted singularities: points (x, y) with their λ (upper or lower half-plane: both conjugates are present), and lines
# a x + b y + c = 0 (coefficients (a, b, c)) with their kind.
function mock_singularities(mdl::mockLiouvillian)
    pts, lines = NamedTuple[], NamedTuple[]
    for (k, b) ∈ enumerate(mdl.blocks)
        if b isa DimerBlock
            λ = -b.γ0 - im*b.ω0
            if b.g > 0
                for σ ∈ (-1, 1)
                    push!(pts, (block=k, kind=:EP2, x=b.center[1], y=b.center[2] + σ*b.g/(2b.s), λ=λ))
                end
            else
                push!(pts, (block=k, kind=:DP, x=b.center[1], y=b.center[2], λ=λ))
            end
        elseif b isa DrivenQubit
            push!(lines, (block=k, kind=:real_EP2, coeffs=(b.Ω[2] - b.γ[2]/4, b.Ω[3] - b.γ[3]/4, b.Ω[1] - b.γ[1]/4)))
        elseif b isa DetunedQubit
            push!(lines, (block=k, kind=:DP_line, coeffs=(b.Δ[2], b.Δ[3], b.Δ[1])))
        end
    end
    return pts, lines
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
#   7. systematic scan on a grid (dense spectra)      EdgeMatch, match_spectra, edge_match, cell_monodromy, monodromy_scan,
#                                                     refine_cells
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

## 7. Systematic scan on a grid (dense spectra, all eigenvalues at once)
# For matrices small enough to diagonalise densely, every eigenvalue is followed along every grid edge by matching the
# spectra at adaptively spaced points: a step is accepted when each eigenvalue moves by less than ρ times its distance
# to the others. The four edges of a cell compose into a permutation of the eigenvalues at its lower-left corner:
# a transposition (a b) ⇔ an odd number of EP2 of that pair inside. Along the same loop, the winding of D = (λa - λb)²
# of a pair that returns to itself counts its EPs with orientation (±1 each) and DPs (±2): an identity with a nonzero
# winding is an even number of EPs or a DP, told apart by refinement (EPs separate into swapping sub-cells).
# An exact coalescence on an edge (a real-axis EP or DP line crossing it) stalls the stepping: below hmin the step is
# forced and the labels involved are flagged ambiguous on that edge (their cells say nothing about that pair).

struct EdgeMatch
    vals::Matrix{ComplexF64}     # eigenvalues along the edge: row = label at the start node, column = sample
    perm::Vector{Int}            # label at the start node ↦ label (index) at the end node
    ambiguous::BitVector         # labels whose matching was forced somewhere on the edge
end

# Greedy nearest matching of the labelled eigenvalues `cur` to `new` (most certain first): p[i] = index in new of label i.
# bad[i] when label i moved by more than ρ times its distance to the nearest other eigenvalue.
function match_spectra(cur, new; ρ=0.2)
    n = length(cur)
    p, taken = zeros(Int, n), falses(n)
    for i ∈ sortperm([minimum(abs.(new .- c)) for c ∈ cur])
        p[i] = argmin(j -> taken[j] ? Inf : abs(new[j] - cur[i]), 1:n)
        taken[p[i]] = true
    end
    sep = [minimum(abs(cur[i] - cur[k]) for k ∈ 1:n if k != i) for i ∈ 1:n]
    bad = BitVector([abs(new[p[i]] - cur[i]) > ρ*sep[i] for i ∈ 1:n])
    return p, bad
end

# Follows all eigenvalues from a to b (spectra va, vb at the nodes, as returned by eigs_at).
function edge_match(eigs_at, a::P2, b::P2, va, vb; ρ=0.2, h0=0.05, hmin=1e-7)
    n = length(va)
    cur, cols, amb = copy(va), [copy(va)], falses(n)
    s, h, perm = 0.0, h0, collect(1:n)
    while s < 1
        last = h ≥ 1 - s
        snew = last ? 1.0 : s + h
        new = last ? vb : eigs_at(a + snew*(b - a))
        p, bad = match_spectra(cur, new; ρ)
        if any(bad) && h > hmin
            h /= 2                                       # refine the step
            continue
        end
        amb .|= bad                                      # forced step (coalescence on the edge)
        cur = new[p]
        push!(cols, cur)
        last && (perm = p)
        s = snew
        any(bad) || (h *= 1.5)
    end
    return EdgeMatch(reduce(hcat, cols), perm, amb)
end

# Loop around a cell, counterclockwise from its lower-left corner. Edges as stored: bottom and top from left to right,
# left and right from bottom to top. Returns the permutation σ of the corner labels, the windings w[a, k] of
# (λa - λk)² for the pairs that return to themselves (0 otherwise), and the labels ambiguous on some edge of the loop.
function cell_monodromy(bottom::EdgeMatch, right::EdgeMatch, top::EdgeMatch, left::EdgeMatch)
    n = length(bottom.perm)
    σ, amb = zeros(Int, n), falses(n)
    traj = [ComplexF64[] for _ ∈ 1:n]
    itop, ileft = invperm(top.perm), invperm(left.perm)
    for i ∈ 1:n
        l = i
        amb[i] |= bottom.ambiguous[l]; append!(traj[i], bottom.vals[l, :]); l = bottom.perm[l]
        amb[i] |= right.ambiguous[l];  append!(traj[i], right.vals[l, 2:end]); l = right.perm[l]
        l = itop[l];  amb[i] |= top.ambiguous[l];  append!(traj[i], reverse(top.vals[l, :])[2:end])
        l = ileft[l]; amb[i] |= left.ambiguous[l]; append!(traj[i], reverse(left.vals[l, :])[2:end])
        σ[i] = l
    end
    w = zeros(Int, n, n)
    for i ∈ 1:n, k ∈ i+1:n
        Set((σ[i], σ[k])) == Set((i, k)) || continue     # D closes only for pairs returning to themselves
        D = (traj[i] .- traj[k]).^2
        w[i, k] = w[k, i] = round(Int, sum(angle(D[t + 1]/D[t]) for t ∈ 1:length(D) - 1)/2π)
    end
    return σ, w, amb
end

# Scan of the grid xs × ys (eigs_at(δ): all eigenvalues at δ). For every cell: its rectangle, the swapped pairs and the
# identity pairs with nonzero winding (eigenvalues at the lower-left corner, upper half-plane representative of each
# conjugate pair, ambiguous labels excluded), longer cycles, and whether some label was ambiguous. Edges in parallel.
function monodromy_scan(eigs_at, xs, ys; kwargs...)
    nodes = [eigs_at(P2(x, y)) for x ∈ xs, y ∈ ys]
    H = Matrix{EdgeMatch}(undef, length(xs) - 1, length(ys))
    V = Matrix{EdgeMatch}(undef, length(xs), length(ys) - 1)
    Threads.@threads for k ∈ 1:length(H) + length(V)
        if k ≤ length(H)
            i, j = Tuple(CartesianIndices(H)[k])
            H[i, j] = edge_match(eigs_at, P2(xs[i], ys[j]), P2(xs[i + 1], ys[j]), nodes[i, j], nodes[i + 1, j]; kwargs...)
        else
            i, j = Tuple(CartesianIndices(V)[k - length(H)])
            V[i, j] = edge_match(eigs_at, P2(xs[i], ys[j]), P2(xs[i], ys[j + 1]), nodes[i, j], nodes[i, j + 1]; kwargs...)
        end
    end
    cells = NamedTuple[]
    for j ∈ 1:length(ys) - 1, i ∈ 1:length(xs) - 1
        σ, w, amb = cell_monodromy(H[i, j], V[i + 1, j], H[i, j + 1], V[i, j])
        λ = nodes[i, j]
        upper(a, b) = imag(λ[a]) + imag(λ[b]) > 0
        swaps = [(λ[a], λ[σ[a]]) for a ∈ eachindex(σ) if σ[a] > a && σ[σ[a]] == a && !amb[a] && !amb[σ[a]] && upper(a, σ[a])]
        even = [(λ[a], λ[k], w[a, k]) for a ∈ eachindex(σ) for k ∈ a+1:length(σ)
                if σ[a] == a && σ[k] == k && w[a, k] != 0 && !amb[a] && !amb[k] && upper(a, k)]
        cycles = count(a -> σ[a] != a && σ[σ[a]] != a && !amb[a], eachindex(σ))
        push!(cells, (rect=(xs[i], xs[i + 1], ys[j], ys[j + 1]), swaps=swaps, even=even, cycles=cycles, ambiguous=any(amb)))
    end
    return cells
end

flagged(c) = !isempty(c.swaps) || !isempty(c.even) || c.cycles > 0

# Recursive refinement of the flagged cells (2 × 2 sub-cells per level, `depth` levels): returns the flagged leaves.
function refine_cells(eigs_at, cells, depth; kwargs...)
    depth == 0 && return filter(flagged, cells)
    leaves = NamedTuple[]
    for c ∈ filter(flagged, cells)
        x0, x1, y0, y1 = c.rect
        sub = monodromy_scan(eigs_at, range(x0, x1, 3), range(y0, y1, 3); kwargs...)
        append!(leaves, refine_cells(eigs_at, sub, depth - 1; kwargs...))
    end
    return leaves
end


#%% Mock Liouvillian: spectrum and planted singularities

# Test bed for the monodromy tools (see mockLiouvillian): a Lindblad sector affine in (x, y) ∈ [0, 1]², with an exactly
# known spectrum. This cell checks it: closed-form vs numerical eigenvalues (both sectors), affinity, and the scaling of
# the gap at every planted singularity (√ at an EP2, linear at a DP), then maps the smallest gap over the square in the
# default coherence sector, where the planted singularities are the only ones.
mock = mockLiouvillian()
mock_at(δ) = setproperties(mock, (x=δ[1], y=δ[2]))
pts, lines = mock_singularities(mock)
for p ∈ pts
    println("planted ", rpad(p.kind, 4), " block $(p.block) at (x, y) = (", p.x, ", ", round(p.y, digits=6), "),  λ = ", p.λ)
end
for l ∈ lines
    println("planted ", rpad(l.kind, 9), " block $(l.block): ", l.coeffs[1], " x + ", l.coeffs[2], " y + ", l.coeffs[3], " = 0")
end

# closed form vs numerical spectrum (greedy matching) at random points, and affinity in (x, y)
function match_error(a, b)
    b, e = copy(b), 0.0
    for v ∈ a
        k = argmin(abs.(b .- v))
        e = max(e, abs(b[k] - v))
        deleteat!(b, k)
    end
    return e
end
for sector ∈ (:coherences, :within)
    err = maximum(begin
                      δ = rand(2)
                      match_error(eigvals(Matrix(mock_matrix(mock_at(δ); sector))), mock_spectrum(mock_at(δ); sector))
                  end for _ ∈ 1:20)
    println("sector $sector ($(length(mock_sector(mock; sector))) states): max |λ_numerical - λ_closed form| over 20 random points = ", err)
end
A_mock = AffineLiouvillian(δ -> mock_matrix(mock_at(δ)), 2)
println("affine in (x, y): accepted by AffineLiouvillian")

# gap scaling at the planted singularities, from the numerical eigenvalues: exponent of gap ∝ distance^p
pair_gap(δ, λ0) = (v = eigvals(Matrix(mock_matrix(mock_at(δ)))); i = sortperm(abs.(v .- λ0)); abs(v[i[1]] - v[i[2]]))
exponent(gap) = log10(gap(1e-4)/gap(1e-6))/2
for p ∈ pts
    println(rpad("$(p.kind) at ($(p.x), $(round(p.y, digits=4)))", 30), " exponent ",
        round(exponent(d -> pair_gap(P2(p.x + d, p.y), p.λ)), digits=3), "  (expected ", p.kind == :EP2 ? 0.5 : 1.0, ")")
end
for l ∈ lines
    a, b, c = l.coeffs
    n = P2(a, b)/hypot(a, b)
    δ0 = P2(-(0.6b + c)/a, 0.6)                         # the point of the line at y = 0.6, approached along its normal
    # the coalescing pair: -3γ/4 ± ... for the driven qubit (3rd entry), -γ/2 ± iΔ for the detuned one (1st entry)
    λ0 = block_spectrum(mock.blocks[l.block], δ0...)[l.kind == :real_EP2 ? 3 : 1]
    println(rpad("$(l.kind) at ($(δ0[1]), $(round(δ0[2], digits=4)))", 30), " exponent ",
        round(exponent(d -> pair_gap(δ0 + d*n, λ0)), digits=3), "  (expected ", l.kind == :real_EP2 ? 0.5 : 1.0, ")")
end

# map of the smallest gap between eigenvalues (steady states, one per block, excluded) with the planted structures
xs = ys = range(0, 1, 201)
gapmap = [begin
              v = filter(λ -> abs(λ) > 1e-9, mock_spectrum(mock_at((x, y))))
              minimum(abs(v[i] - v[j]) for i ∈ eachindex(v) for j ∈ i+1:length(v))
          end for x ∈ xs, y ∈ ys]
fig = Figure(size=(760, 650))
ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="mock Liouvillian (coherence sector): log₁₀ smallest gap", aspect=DataAspect(), limits=((0, 1), (0, 1)))
hm = heatmap!(ax, xs, ys, log10.(gapmap); colormap=:viridis)
Colorbar(fig[1, 2], hm)
for l ∈ lines
    a, b, c = l.coeffs
    lines!(ax, [0, 1], [-c/b, -(a + c)/b]; color=:white, linestyle=(l.kind == :real_EP2 ? :solid : :dash),
        label=(l.kind == :real_EP2 ? "real-axis EP2 line" : "DP line"))
end
for (kind, marker) ∈ ((:EP2, :star5), (:DP, :circle))
    sel = filter(p -> p.kind == kind, pts)
    scatter!(ax, [p.x for p ∈ sel], [p.y for p ∈ sel]; marker, markersize=14, color=:red, strokecolor=:white, strokewidth=1,
        label="planted $kind")
end
axislegend(ax; position=:lt, labelsize=11)
fig


#%% Mock Liouvillian: systematic monodromy scan

# Needs the mock check cell (mock, mock_at, pts, lines, A_mock). All 18 eigenvalues of the coherence sector are followed
# along the edges of a 39 × 39 grid, each cell gets a
# permutation, and the flagged cells (a swap, or an identity with nonzero winding) are refined 7 times (cells of 2e-4,
# enough to separate the weak EP2 pair 1e-3 apart). Expected: one swapping leaf at each planted EP2, a DP (winding ±2,
# no swap at every level) at the planted DP, ambiguous cells only along the two real-axis lines, nothing else.
# scan_sector = :within adds the dimers' populations and their real-axis structures (see mockLiouvillian).
scan_sector = :coherences
A_scan = AffineLiouvillian(δ -> mock_matrix(mock_at(δ); sector=scan_sector), 2)
eigs_at(δ) = eigvals(Matrix(A_scan(δ)))
xs = ys = range(0, 1, 39)
t_scan = @elapsed cells = monodromy_scan(eigs_at, xs, ys)
t_refine = @elapsed leaves = refine_cells(eigs_at, cells, 7)
println("sector $scan_sector, scan: ", length(cells), " cells in ", round(t_scan, digits=2), " s; ", count(flagged, cells), " flagged, ",
    count(c -> c.ambiguous, cells), " with ambiguous labels.  Refinement (7 levels): ", length(leaves), " leaves in ",
    round(t_refine, digits=2), " s")

# leaves against the planted points
inside(c, p) = c.rect[1] ≤ p.x ≤ c.rect[2] && c.rect[3] ≤ p.y ≤ c.rect[4]
center(c) = ((c.rect[1] + c.rect[2])/2, (c.rect[3] + c.rect[4])/2)
used = falses(length(leaves))
for p ∈ pts
    k = findall(c -> inside(c, p), leaves)
    used[k] .= true
    found = isempty(k) ? "NOT FOUND" : begin
        c = leaves[k[1]]
        kind = !isempty(c.swaps) ? "swap (EP2)" : !isempty(c.even) ? "identity, winding $(c.even[1][3]) (DP)" : "?"
        "$kind in a leaf of size $(round(c.rect[2] - c.rect[1], sigdigits=2)), centre off by " *
        "$(round(hypot((center(c) .- (p.x, p.y))...), sigdigits=2)), λ ≈ $(round(conj(isempty(c.swaps) ? c.even[1][1] : c.swaps[1][1]), digits=3))"
    end
    println("planted ", rpad(p.kind, 4), " at (", p.x, ", ", round(p.y, digits=4), "): ", found)
end
println("leaves not at a planted point: ", count(!, used))

# ambiguous coarse cells: are they all crossed by one of the two lines?
crossed(c, l) = (f = [l.coeffs[1]*x + l.coeffs[2]*y + l.coeffs[3] for x ∈ c.rect[1:2] for y ∈ c.rect[3:4]]; minimum(f) ≤ 1e-12 && maximum(f) ≥ -1e-12)
amb = filter(c -> c.ambiguous, cells)
println("ambiguous cells crossed by a real-axis line: ", count(c -> any(l -> crossed(c, l), lines), amb), " / ", length(amb),
    ";  cells crossed by a line: ", count(c -> any(l -> crossed(c, l), lines), cells))

fig = Figure(size=(1150, 560))
ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="39 × 39 scan, sector $scan_sector", aspect=DataAspect(), limits=((0, 1), (0, 1)))
rect_poly(r) = Point2f[(r[1], r[3]), (r[2], r[3]), (r[2], r[4]), (r[1], r[4])]
for c ∈ cells
    col = !isempty(c.swaps) ? :red : !isempty(c.even) ? :orange : c.ambiguous ? (:gray, 0.6) : nothing
    col === nothing || poly!(ax, rect_poly(c.rect); color=col)
end
for l ∈ lines
    a, b, cc = l.coeffs
    lines!(ax, [0, 1], [-cc/b, -(a + cc)/b]; color=:white, linestyle=(l.kind == :real_EP2 ? :solid : :dash))
end
scatter!(ax, [p.x for p ∈ pts], [p.y for p ∈ pts]; marker=[p.kind == :EP2 ? :star5 : :circle for p ∈ pts],
    markersize=12, color=:transparent, strokecolor=:white, strokewidth=1.5)
# zoom on the weak EP2 pair: refined leaves
pw = filter(p -> p.block == 1, pts)
xw, yw = pw[1].x, (pw[1].y + pw[2].y)/2
ax2 = Axis(fig[1, 2]; xlabel="x", ylabel="y", title="weak EP2 pair: refined leaves", aspect=DataAspect(),
    limits=((xw - 1e-3, xw + 1e-3), (yw - 1e-3, yw + 1e-3)))
for c ∈ leaves
    poly!(ax2, rect_poly(c.rect); color=(!isempty(c.swaps) ? :red : :orange), strokecolor=:white, strokewidth=0.5)
end
scatter!(ax2, [p.x for p ∈ pw], [p.y for p ∈ pw]; marker=:star5, markersize=16, color=:transparent, strokecolor=:white, strokewidth=1.5)
Legend(fig[2, 1:2], [PolyElement(color=:red), PolyElement(color=:orange), PolyElement(color=(:gray, 0.6)),
        LineElement(color=:white), LineElement(color=:white, linestyle=:dash), MarkerElement(marker=:star5, color=:transparent, strokecolor=:white, strokewidth=1.5),
        MarkerElement(marker=:circle, color=:transparent, strokecolor=:white, strokewidth=1.5)],
    ["swap (odd number of EP2)", "identity, nonzero winding", "ambiguous labels", "real-axis EP2 line", "DP line", "planted EP2", "planted DP"];
    orientation=:horizontal, nbanks=2, tellheight=true)
fig


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
