# Quantum Rabi model (QRM) and its Jaynes–Cummings (JC) approximation, in the conventions of the coworker's
# functions_QRM.jl: dipole gauge, photon ⊗ qubit ordering, rates γa, γb are energy-decay rates (jump √γa â), baths at
# temperatures Ta, Tb.
# With RWA_coupling and RWA_env (local Lindblad dissipators) at T = 0 the spectrum of L is known in closed form: the
# jumps only lower the excitation number N = a†a + σ+σ-, which H conserves, so L is block-triangular and its eigenvalues
# are -i(εᵢ - conj(εⱼ)) over the eigenvalues ε of H_eff = H - (i/2)Σ J†J in each N-manifold. Manifold n ≥ 1 is the 2×2
# problem {|n, g⟩, |n-1, e⟩} with coupling g√n: an EP2 of H_eff at ωb = ωa, g = |γa - γb|/(4√n), a tower of EPs
# accumulating at g → 0.

"""
    QRM(; ωa=1, ωb=1, g=0, γa=0, γb=0, Ta=0, Tb=0, ε=0, approx=Approximation[])

Quantum Rabi model in the dipole gauge, as in the coworker's `H_QRM_sb`:
H = ωa a†a + ωb σz/2 - i g (a - a†) σx + ε σx.
- `RWA_coupling`: Jaynes–Cummings coupling -i g (a σ+ - a† σ-), which conserves N = a†a + σ+σ- (unitarily equivalent to
  g(a σ+ + a† σ-) by a → i a). The bias ε σx breaks that conservation: it must be 0.
- `RWA_env`: local Lindblad dissipators (`Lindbladian`); otherwise the dressed non-secular master equation
  (`DressedLiouvillian`), the coworker's `gen_liouvillian_QRM`.
Rates are energy-decay rates (the community convention): the photon decays with jump operator √γa a, the qubit with
√γb σ-, at the bath temperatures Ta (photon) and Tb (qubit). The operators are built by `qrm_operators`.
"""
struct QRM{T<:AbstractFloat} <: Model
    ωa::T
    ωb::T
    g::T
    γa::T
    γb::T
    Ta::T
    Tb::T
    ε::T
    approx::Vector{Approximation}
end
QRM(ωa::Real, ωb::Real, g::Real, γa::Real, γb::Real, Ta::Real, Tb::Real, ε::Real, approx=Approximation[]) = QRM(
    promote(float(ωa), float(ωb), float(g), float(γa), float(γb), float(Ta), float(Tb), float(ε))..., to_approx(approx)
)
QRM(; ωa=1., ωb=1., g=0., γa=0., γb=0., Ta=0., Tb=0., ε=0., approx=Approximation[]) = QRM(ωa, ωb, g, γa, γb, Ta, Tb, ε, approx)

"""
    qrm_operators(Nc)

The photon annihilation operator â and the qubit lowering operator σ̂ = σ- on the joint space of `Nc` Fock states ⊗
qubit (photon first, as `generate_a` in functions_QRM.jl). The qubit basis is QuantumToolbox's: |e⟩ first (σz = +1).
"""
qrm_operators(Nc) = (destroy(Nc) ⊗ eye(2), eye(Nc) ⊗ sigmam())

# Bose occupation at frequency ω and temperature T
occupation(ω, T) = T > 0 ? 1/expm1(ω/T) : 0.0

"""
    Hamiltonian(â, σ̂, mdl::QRM)

H = ωa a†a + ωb σz/2 - i g (a - a†) σx + ε σx, or with `RWA_coupling` the JC coupling -i g (a σ+ - a† σ-), with
σz = σ+σ- - σ-σ+ and σx = σ- + σ+.
"""
function Hamiltonian(â::QuantumObject, σ̂::QuantumObject, mdl::QRM)
    (;ωa, ωb, g, ε, approx) = mdl
    σz, σx = σ̂'*σ̂ - σ̂*σ̂', σ̂ + σ̂'
    if RWA_coupling ∈ approx
        iszero(ε) || throw(ArgumentError("the bias ε σx breaks the excitation number conserved by the JC coupling: set ε = 0"))
        return ωa*â'*â + ωb/2*σz - im*g*(â*σ̂' - â'*σ̂)
    end
    return ωa*â'*â + ωb/2*σz - im*g*(â - â')*σx + ε*σx
end

"""
    Lindbladian(â, σ̂, mdl::QRM)

Local Lindblad form: jump operators √(γa(1 + na)) a, √(γa na) a†, √(γb(1 + nb)) σ-, √(γb nb) σ+, with the Bose
occupations na, nb at ωa, ωb and Ta, Tb (zero-rate jumps are dropped). Returns (Ĥ, jump operators). Physical only
with `RWA_env` (a warning is given otherwise): local dissipators ignore the dressing of the states by the coupling.
"""
function Lindbladian(â::QuantumObject, σ̂::QuantumObject, mdl::QRM)
    (;ωa, ωb, γa, γb, Ta, Tb, approx) = mdl
    RWA_env ∈ approx || @warn "Lindbladian called without RWA_env: the QRM's own bath treatment is DressedLiouvillian." maxlog=1
    na, nb = occupation(ωa, Ta), occupation(ωb, Tb)
    rates = (γa*(1 + na), γa*na, γb*(1 + nb), γb*nb)
    ops = (â, â', σ̂, σ̂')
    return Hamiltonian(â, σ̂, mdl), [√r*op for (r, op) ∈ zip(rates, ops) if r > 0]
end

"""
    DressedLiouvillian(â, σ̂, mdl::QRM; N_trunc=nothing, σ_filter=Inf, kwargs...)

Dressed non-secular master equation, as the coworker's `gen_liouvillian_QRM`: the cavity couples to its bath through
A ∝ a + a† (field √(γa/ωa)(a + a†)), the qubit through σx (field √(γb/ωb) σx), at temperatures Ta, Tb. Returns
(E, U, L) in the eigenbasis of H. Not affine in the parameters (the eigenbasis moves with them).
"""
function DressedLiouvillian(â::QuantumObject, σ̂::QuantumObject, mdl::QRM; kwargs...)
    (;ωa, ωb, γa, γb, Ta, Tb) = mdl
    fields, Ts = QuantumObject[], Float64[]
    γa > 0 && (push!(fields, √(γa/ωa)*(â + â')); push!(Ts, Ta))
    γb > 0 && (push!(fields, √(γb/ωb)*(σ̂ + σ̂')); push!(Ts, Tb))
    return liouvillian_dressed_nonsecular(Hamiltonian(â, σ̂, mdl), fields, Ts; kwargs...)
end

## Excitation-number sectors

# excitation number N = n + 1 (excited qubit) or n (ground) of the basis state with index idx of qrm_operators(Nc)
excitation_number(idx) = (idx - 1) ÷ 2 + ((idx - 1) % 2 == 0)

"""
    excitation_sector(Nc, k)

Indices in the vectorised ρ (column-major) of the elements |i⟩⟨j| with N_i - N_j = k, N = a†a + σ+σ-. With
`RWA_coupling` and `RWA_env`, L conserves this difference (a U(1) weak symmetry): it is block diagonal in these
sectors. ⟨a⟩ and ⟨σ-⟩ live in k = 1; without `RWA_coupling` only the parity of k is conserved.
"""
function excitation_sector(Nc, k)
    D = 2Nc
    return [i + D*(j - 1) for j ∈ 1:D for i ∈ 1:D if excitation_number(i) - excitation_number(j) == k]
end

## Closed forms for JC with RWA_env at T = 0

# requirements of the closed forms
function check_jc_closed_form(mdl::QRM)
    (RWA_coupling ∈ mdl.approx && RWA_env ∈ mdl.approx) || throw(ArgumentError("closed forms need RWA_coupling and RWA_env"))
    (iszero(mdl.Ta) && iszero(mdl.Tb)) || throw(ArgumentError("closed forms need Ta = Tb = 0"))
end

"""
    jc_effective_energies(mdl::QRM, Nc)

Eigenvalues ε of H_eff = H - (i/2)(γa a†a + γb σ+σ-) in the truncated space (`Nc` Fock states), with their manifold
N, as a vector of (N, ε). Manifold 0 is |0, g⟩; manifold 1 ≤ n ≤ Nc - 1 is {|n, g⟩, |n-1, e⟩}, with diagonal
d₁ = nωa - ωb/2 - iγa n/2, d₂ = (n-1)ωa + ωb/2 - i(γa(n-1) + γb)/2 and coupling g√n; the truncation leaves manifold
Nc with |Nc-1, e⟩ only.
"""
function jc_effective_energies(mdl::QRM, Nc)
    check_jc_closed_form(mdl)
    (;ωa, ωb, g, γa, γb) = mdl
    levels = [(0, -ωb/2 + 0im)]
    for n ∈ 1:Nc - 1
        d1 = n*ωa - ωb/2 - im*γa*n/2
        d2 = (n - 1)*ωa + ωb/2 - im*(γa*(n - 1) + γb)/2
        r = sqrt(complex(((d1 - d2)/2)^2 + g^2*n))
        push!(levels, (n, (d1 + d2)/2 + r), (n, (d1 + d2)/2 - r))
    end
    push!(levels, (Nc, (Nc - 1)*ωa + ωb/2 - im*(γa*(Nc - 1) + γb)/2))
    return levels
end

"""
    jc_spectrum(mdl::QRM, Nc; k=nothing)

Exact spectrum of the Lindblad Liouvillian of JC at T = 0 with `Nc` Fock states: -i(εᵢ - conj(εⱼ)) over all pairs of
`jc_effective_energies`, or only those with Nᵢ - Nⱼ = k (the sector `excitation_sector(Nc, k)`).
"""
function jc_spectrum(mdl::QRM, Nc; k=nothing)
    levels = jc_effective_energies(mdl, Nc)
    return [-im*(εi - conj(εj)) for (Nj, εj) ∈ levels for (Ni, εi) ∈ levels if k === nothing || Ni - Nj == k]
end

"""
    jc_EP(mdl::QRM, n)

EP2 of H_eff in the manifold n ≥ 1: the point (ωb, g) = (ωa, |γa - γb|/(4√n)), with the coalesced eigenvalue
ε = (n - 1/2)ωa - i(γa(2n - 1) + γb)/4. Returns (ωb, g, ε). In the Liouvillian, every eigenvalue -i(ε - conj(εⱼ)) and
-i(εⱼ - conj(ε)) with εⱼ in another manifold is an EP2 there (with both indices in manifold n, four eigenvalues meet).
"""
function jc_EP(mdl::QRM, n)
    (;ωa, γa, γb) = mdl
    return ωa, abs(γa - γb)/(4√n), (n - 1/2)*ωa - im*(γa*(2n - 1) + γb)/4
end
