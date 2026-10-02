# Models (parameters and approximations) and their analytical response: response matrix D_R, characteristic
# polynomial disc, analytic EPs.

"""
    Model

Supertype of the physical models: a model holds parameters and the approximations to apply, and the functions of
`master_equations.jl` and of this file dispatch on it.
"""
abstract type Model end

"""
    Approximation

Approximations a model can be built with (field `approx`, a vector of them):
- `RWA_env`: rotating-wave approximation in the system–bath coupling (local Lindblad dissipators); without it, the bath
  is Ohmic and treated by Bloch–Redfield (velocity damping).
- `RWA_coupling`: rotating-wave approximation in the coupling between modes, g/2(a†b + b†a).
- `N_conserving`: keep only the excitation-conserving part of a non-linearity (Kerr part of x̂⁴).
- `linearised`: fluctuations around the classical steady state (optomechanics).
"""
@enum Approximation RWA_env RWA_coupling N_conserving linearised

# a single Approximation, a tuple/vector of them, or an empty collection, as a Vector{Approximation}
to_approx(a::Approximation) = [a]
to_approx(a) = collect(Approximation, a)

"""
    Boson(; ω0=1, γ=1, nB=0, approx=[RWA_env])

Damped harmonic oscillator: frequency `ω0`, damping rate `γ` (amplitude decay), thermal occupation `nB` of its bath.
"""
struct Boson{T<:AbstractFloat} <: Model
    ω0::T
    γ::T
    nB::T
    approx::Vector{Approximation}
end
Boson(ω0::Real, γ::Real, nB::Real, approx=[RWA_env]) = Boson(
    promote(float(ω0), float(γ), float(nB))..., to_approx(approx)
)
Boson(;ω0=1., γ=1., nB=0., approx=[RWA_env]) = Boson(ω0, γ, nB, approx)

"""
    BosonDimer(; ωa=1, ωb=1, γa=1, γb=1, g=0, nB=0, approx=[RWA_env, RWA_coupling])

Two damped oscillators a, b (frequencies `ωa`, `ωb`, damping rates `γa`, `γb`, same bath occupation `nB`) with
coupling g/2(a + a†)(b + b†), or g/2(a†b + b†a) with `RWA_coupling`.
"""
struct BosonDimer{T<:AbstractFloat} <: Model
    ωa::T
    ωb::T
    γa::T
    γb::T
    g::T
    nB::T
    approx::Vector{Approximation}
end
BosonDimer(ωa::Real, ωb::Real, γa::Real, γb::Real, g::Real, nB::Real, approx=[RWA_env, RWA_coupling]) = BosonDimer(
    promote(float(ωa), float(ωb), float(γa), float(γb), float(g), float(nB))..., to_approx(approx)
)
BosonDimer(;ωa=1., ωb=1., γa=1., γb=1., g=0., nB=0., approx=[RWA_env, RWA_coupling]) = BosonDimer(ωa, ωb, γa, γb, g, nB, approx)

"""
    Duffing(; ω0=1, γ=1, g=0, nB=0, approx=[RWA_env])

Damped anharmonic (Duffing) oscillator, H = ω0 a†a + g/4! x̂⁴ with x̂ = (a + a†)/√2, the quadrature described by D_R.
(In unit-mass units, x̂ = (a + a†)/√(2ω0), the same term reads g/(4! ω0²) x̂⁴.) `N_conserving` keeps only the
excitation-conserving (Kerr) part of x̂⁴; `RWA_env` selects the Lindblad bath as for the bosons.
"""
struct Duffing{T<:AbstractFloat} <: Model
    ω0::T
    γ::T
    g::T
    nB::T
    approx::Vector{Approximation}
end
Duffing(ω0::Real, γ::Real, g::Real, nB::Real, approx=[RWA_env]) = Duffing(
    promote(float(ω0), float(γ), float(g), float(nB))..., to_approx(approx)
)
Duffing(;ω0=1., γ=1., g=0., nB=0., approx=[RWA_env]) = Duffing(ω0, γ, g, nB, approx)

"""
    Optomech(; ωb=1, γa=0.1, γb=0.1, g=0, Δ=-1, F=0, nBa=0, nBb=0, approx=[RWA_env])

Driven optomechanical system, in the frame rotating at the drive frequency (RWA on the drive):
H = -Δ a†a + ωb b†b - g a†a (b + b†) + F (a + a†). Cavity a: damping `γa`, thermal occupation `nBa`; mechanics b:
`γb`, `nBb`.
- The frame change makes a's bath time dependent unless it is treated in the RWA, which is excellent for
  ω_drive ≫ γa, ωb (the shifted bath spectrum is flat over the rotating-frame band): a always gets Lindblad dissipators.
- b's coupling x_b commutes with the frame change: `RWA_env` selects Lindblad for b, otherwise Ohmic Redfield.
- `linearised`: fluctuations around the classical steady state α (see `classical_steady_state`), coupling
  -g(α* a + α a†)(b + b†).
Fock cutoff for a: the full model holds the coherent amplitude, so N_a must exceed |α|² by several √|α|².
"""
struct Optomech{T<:AbstractFloat} <: Model
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
Optomech(ωb::Real, γa::Real, γb::Real, g::Real, Δ::Real, F::Real, nBa::Real, nBb::Real, approx=[RWA_env]) = Optomech(
    promote(float(ωb), float(γa), float(γb), float(g), float(Δ), float(F), float(nBa), float(nBb))..., to_approx(approx)
)
Optomech(;ωb=1., γa=0.1, γb=0.1, g=0., Δ=-1., F=0., nBa=0., nBb=0., approx=[RWA_env]) = Optomech(ωb, γa, γb, g, Δ, F, nBa, nBb, approx)

## Classical steady state and linearisation of the optomechanical system

"""
    classical_steady_state(mdl::Optomech)

Classical steady states, a → α, b → β: the static displacement x̄ = β + β* = 2g|α|² c with c = ωb/(ωb² + γb²)
(Lindblad damping of b, `RWA_env`) or 1/ωb (velocity damping), and α = F/(Δ + g x̄ + iγa). Then n = |α|² solves the
cubic n((Δ + 2g²c n)² + γa²) = F², with up to three positive roots (bistability). Returns all (α, x̄), by increasing
|α|².
"""
function classical_steady_state(mdl::Optomech)
    (;ωb, γa, γb, g, Δ, F, approx) = mdl
    c = RWA_env ∈ approx ? ωb/(ωb^2 + γb^2) : 1/ωb
    k = 2g^2*c
    cubic = Polynomial([-F^2, Δ^2 + γa^2, 2Δ*k, k^2])
    ns = sort([real(n) for n ∈ roots(cubic) if abs(imag(n)) < 1e-10 && real(n) ≥ 0])
    iszero(k) && (ns = [F^2/(Δ^2 + γa^2)])
    return [(F/(Δ + k*n + im*γa), 2g*c*n) for n ∈ ns]
end

"""
    classical_displacement(mdl::Optomech; branch=1)

Reference displacement (α, β = x̄/2) of a classical steady state, for a displaced frame a = α + d, b = β + e.
"""
function classical_displacement(mdl::Optomech; branch=1)
    α, x̄ = classical_steady_state(mdl)[branch]
    return α, x̄/2
end

"""
    linearised_dimer(mdl::Optomech; branch=1)

Linearised fluctuations around a classical steady state, as the equivalent `BosonDimer`: a at -(Δ + g x̄), coupling
-g(α* a + α a†)(b + b†) = (2g|α|/2)(a' + a'†)(b + b†) after absorbing the phase of α. Same bath treatment (approx).
"""
function linearised_dimer(mdl::Optomech; branch=1)
    (;ωb, γa, γb, g, Δ, nBb, approx) = mdl
    α, x̄ = classical_steady_state(mdl)[branch]
    return BosonDimer(ωa=-(Δ + g*x̄), ωb=ωb, γa=γa, γb=γb, g=2g*abs(α), nB=nBb, approx=filter(!=(linearised), approx))
end

"""
    linearised_EP(mdl::Optomech)

(Δ, F) of the linearised EP: resonance Δ + g x̄ = -ωb and coupling 2g|α| = g_EP of the dimer (`EP` of `BosonDimer`).
Only for `RWA_env`, where the dimer EP needs equal bare frequencies. Returns (Δ, F, number of classical branches
there).
"""
function linearised_EP(mdl::Optomech)
    (;ωb, γa, γb, g) = mdl
    RWA_env ∈ mdl.approx || throw(ArgumentError("linearised_EP is implemented for RWA_env only"))
    g_EP, = EP(BosonDimer(ωa=ωb, ωb=ωb, γa=γa, γb=γb, g=1., approx=[RWA_env]))
    n = (g_EP/2g)^2                                  # |α|² at the EP
    x̄ = 2g*n*ωb/(ωb^2 + γb^2)
    Δ = -ωb - g*x̄
    F = √(n*(ωb^2 + γa^2))
    return Δ, F, length(classical_steady_state(setproperties(mdl, (Δ=Δ, F=F))))
end

## Analytical response

"""
    D_R(ω, mdl::Boson)
    D_R(ω, mdl::BosonDimer)

Inverse response function of the position quadrature(s), whose zeros in ω are the complex mode frequencies.
- `Boson`: ((ω + iγ)² - ω0²)/ω0 with `RWA_env` (Lindblad), (ω² - ω0² + 2iωγ)/ω0 otherwise (velocity damping).
- `BosonDimer`: with `RWA_coupling`, the rotating-frame 2×2 matrix in the (a, b) basis; otherwise, since
  g/2(a + a†)(b + b†) = g x_a x_b, the 2×2 matrix in the (x_a, x_b) basis, with the single-boson D_R on the diagonal
  (which respects `RWA_env`).
"""
D_R(ω, mdl::Boson) = RWA_env ∈ mdl.approx ? ((ω + im*mdl.γ)^2 - mdl.ω0^2)/mdl.ω0 : (ω^2 - mdl.ω0^2 + 2im*ω*mdl.γ)/mdl.ω0
function D_R(ω, mdl::BosonDimer)
    (; ωa, ωb, γa, γb, g, nB, approx) = mdl
    if RWA_coupling ∈ approx
        return [
            ω-ωa+im*γa g/2
            g/2 ω-ωb+im*γb
        ]
    end
    Da = D_R(ω, Boson(ωa, γa, nB, approx))
    Db = D_R(ω, Boson(ωb, γb, nB, approx))
    return [
        Da g
        g Db
    ]
end

"""
    disc(mdl)

Characteristic polynomial in ω, disc(mdl)(ω) == det(D_R(ω, mdl)): its roots are the complex mode frequencies.
Evaluating D_R on the polynomial variable inherits its dispatch on the model and the approximations.
"""
disc(mdl::Boson) = D_R(Polynomial([0, 1], :ω), mdl)
function disc(mdl::BosonDimer)
    D = D_R(Polynomial([0, 1], :ω), mdl)
    return D[1, 1]*D[2, 2] - D[1, 2]*D[2, 1]
end

"""
    roots_sorted(mdl)

The complex mode frequencies (roots of `disc`), sorted by real then imaginary part.
"""
roots_sorted(mdl) = sort(roots(disc(mdl)), by=ω -> (real(ω), imag(ω)))

"""
    EP(mdl::Boson)
    EP(mdl::BosonDimer)

Analytic exceptional point, or `nothing` if there is none.
- `Boson`: (γ_EP, ω_EP), critical damping γ = ω0 (Q = ω0/2γ = 1/2), where -iγ ± √(ω0² - γ²) merge on the imaginary
  axis. With `RWA_env` the modes are ±ω0 - iγ for any γ: no EP.
- `BosonDimer`: (g_EP, ω_EP) of the positive-frequency modes, or nothing if the modes are detuned (no EP by tuning g
  alone). Without `RWA_coupling` the frequencies that must match are Ω_i = √(ω_i² - γ_i²), or the bare ω_i with
  `RWA_env`.
"""
EP(mdl::Boson) = RWA_env ∈ mdl.approx ? nothing : (mdl.ω0, -im*mdl.ω0)
function EP(mdl::BosonDimer)
    (;ωa, ωb, γa, γb, approx) = mdl
    δ, γ̄ = (γa - γb)/2, (γa + γb)/2
    if RWA_coupling ∈ approx
        return ωa ≈ ωb ? (2abs(δ), ωa - im*γ̄) : nothing
    end
    Ωa, Ωb = RWA_env ∈ approx ? (ωa, ωb) : (√(ωa^2 - γa^2), √(ωb^2 - γb^2))
    (Ωa ≈ Ωb && Ωa > abs(δ)) || return nothing
    return Ωa*2abs(δ)/√(ωa*ωb), √(Ωa^2 - δ^2) - im*γ̄
end
