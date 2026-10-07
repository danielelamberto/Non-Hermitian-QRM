# Master equations of the models, built with QuantumToolbox, and spectral analysis of the resulting Liouvillians.
# The mode operators â, b̂ must already act on the joint Hilbert space, e.g. â = destroy(N) ⊗ eye(N).

## Hamiltonians and baths

"""
    Hamiltonian(â, mdl::Boson)
    Hamiltonian(â, mdl::Duffing)
    Hamiltonian(â, b̂, mdl::BosonDimer)
    Hamiltonian(â, b̂, mdl::Optomech; branch=1)

The Hamiltonian of the model, as a QuantumObject.
- `Duffing`: x̂⁴ connects n to n ± 2, n ± 4: with a Fock cutoff N the top levels are distorted, keep N well above the
  populated ones. With `N_conserving`, x̂⁴ → its part with two â and two â', (6â'²â² + 12â'â + 3)/4
  = (3/2)n̂² + (3/2)n̂ + 3/4 (Kerr): the diagonal of x̂⁴ in the Fock basis, exact up to the cutoff.
- `BosonDimer`: coupling as in `D_R`, g/2(a†b + b†a) with `RWA_coupling`, g/2(a + a†)(b + b†) otherwise.
- `Optomech`: the full model, or (`linearised`) the fluctuation Hamiltonian around the classical steady state number
  `branch`, with Δ → Δ + g x̄.
"""
Hamiltonian(â::QuantumObject, mdl::Boson) = mdl.ω0*â'*â
function Hamiltonian(â::QuantumObject, mdl::Duffing)
    n̂ = â'*â
    if N_conserving ∈ mdl.approx
        return mdl.ω0*n̂ + mdl.g/24*(3/2*n̂^2 + 3/2*n̂ + 3/4*one(n̂))
    end
    x̂ = (â + â')/√2
    return mdl.ω0*n̂ + mdl.g/24*x̂^4
end
function Hamiltonian(â::QuantumObject, b̂::QuantumObject, mdl::BosonDimer)
    (;ωa, ωb, g, approx) = mdl
    Ĥ_int = RWA_coupling ∈ approx ? g/2*(â'*b̂ + b̂'*â) : g/2*(â + â')*(b̂ + b̂')
    return ωa*â'*â + ωb*b̂'*b̂ + Ĥ_int
end
function Hamiltonian(â::QuantumObject, b̂::QuantumObject, mdl::Optomech; branch=1)
    (;ωb, g, Δ, F, approx) = mdl
    if linearised ∈ approx
        α, x̄ = classical_steady_state(mdl)[branch]
        return -(Δ + g*x̄)*â'*â + ωb*b̂'*b̂ - g*(conj(α)*â + α*â')*(b̂ + b̂')
    end
    return -Δ*â'*â + ωb*b̂'*b̂ - g*â'*â*(b̂ + b̂') + F*(â + â')
end

# Each mode's annihilation operator with its Ohmic bath (γ, ω0), coupled through x = â + â'.
baths(â::QuantumObject, mdl::Boson) = [(â, mdl.γ, mdl.ω0)]
baths(â::QuantumObject, mdl::Duffing) = [(â, mdl.γ, mdl.ω0)]
baths(â::QuantumObject, b̂::QuantumObject, mdl::BosonDimer) = [(â, mdl.γa, mdl.ωa), (b̂, mdl.γb, mdl.ωb)]

# Bath temperature from the thermal occupation nB at the reference frequency ω.
temperature(nB, ω) = nB > 0 ? ω/log(1 + 1/nB) : 0.0
temperature(mdl::Boson) = temperature(mdl.nB, mdl.ω0)
temperature(mdl::Duffing) = temperature(mdl.nB, mdl.ω0)
temperature(mdl::BosonDimer) = temperature(mdl.nB, mdl.ωa)

# Ohmic spectrum S(ω) = κω(1 + n(ω)) at temperature T (ω > 0 emission, ω < 0 absorption).
ohmic(κ, T) = ω -> T == 0 ? (ω > 0 ? κ*ω : zero(ω)) : (ω == 0 ? κ*T : κ*ω/(1 - exp(-ω/T)))

# Optomechanics: the cavity a always gets Lindblad jump operators (see Optomech).
function _cavity_jumps(â, mdl::Optomech)
    (;γa, nBa) = mdl
    return [√(2γa*(1 + nBa))*â, √(2γa*nBa)*â']
end

## Master equations

"""
    Lindbladian(â, mdl::Boson)
    Lindbladian(â, mdl::Duffing)
    Lindbladian(â, b̂, mdl::BosonDimer)
    Lindbladian(â, b̂, mdl::Optomech)

Local Lindblad form, secular in the bath: jump operators √(2γ(1 + nB)) â and √(2γ nB) â' for every mode. Returns
(Ĥ, jump operators), to pass to `liouvillian`. Its spectrum matches `D_R`/`disc` only with `RWA_env` (a warning is
given otherwise). For `Optomech`, the mechanics b gets Lindblad terms, and the cavity a always does.
"""
Lindbladian(â::QuantumObject, mdl::Boson) = _lindbladian(Hamiltonian(â, mdl), baths(â, mdl), mdl.nB, mdl.approx)
Lindbladian(â::QuantumObject, mdl::Duffing) = _lindbladian(Hamiltonian(â, mdl), baths(â, mdl), mdl.nB, mdl.approx)
Lindbladian(â::QuantumObject, b̂::QuantumObject, mdl::BosonDimer) = _lindbladian(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), mdl.nB, mdl.approx)
function Lindbladian(â::QuantumObject, b̂::QuantumObject, mdl::Optomech)
    (;γb, nBb, approx) = mdl
    RWA_env ∈ approx || @warn "Lindbladian called without RWA_env: the mechanical bath of this model is then Redfield." maxlog=1
    return Hamiltonian(â, b̂, mdl), [_cavity_jumps(â, mdl); √(2γb*(1 + nBb))*b̂; √(2γb*nBb)*b̂']
end
function _lindbladian(Ĥ, bths, nB, approx)
    RWA_env ∈ approx || @warn "Lindbladian called without RWA_env: the Lindblad form is secular in the bath, so its spectrum will not match D_R/disc for this model." maxlog=1
    return Ĥ, [L̂ for (â, γ, _) ∈ bths for L̂ ∈ (√(2γ*nB)*â', √(2γ*(1+nB))*â)]
end

"""
    Redfield(â, mdl::Boson)
    Redfield(â, mdl::Duffing)
    Redfield(â, b̂, mdl::BosonDimer)
    Redfield(â, b̂, mdl::Optomech)

Non-secular Bloch–Redfield tensor (`sec_cutoff = -1`) with Ohmic spectra, κ = 2γ/ω0, in the Fock basis. It keeps the
terms pairing ω and -ω transitions: velocity damping, as in `D_R` without `RWA_env`. No Lamb shift (none is needed for
a strictly Ohmic bath). For `Optomech`, only the mechanics b is treated this way, in the eigenbasis of the full
rotating-frame H, with the cavity's Lindblad terms passed as collapse operators.
"""
Redfield(â::QuantumObject, mdl::Boson) = _redfield(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl))
Redfield(â::QuantumObject, mdl::Duffing) = _redfield(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl))
Redfield(â::QuantumObject, b̂::QuantumObject, mdl::BosonDimer) = _redfield(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), temperature(mdl))
Redfield(â::QuantumObject, b̂::QuantumObject, mdl::Optomech) = bloch_redfield_tensor(
    Hamiltonian(â, b̂, mdl), [(b̂ + b̂', ohmic(2mdl.γb/mdl.ωb, temperature(mdl.nBb, mdl.ωb)))], _cavity_jumps(â, mdl);
    sec_cutoff=-1, fock_basis=Val(true))
_redfield(Ĥ, bths, T) = bloch_redfield_tensor(Ĥ, [(â + â', ohmic(2γ/ω, T)) for (â, γ, ω) ∈ bths]; sec_cutoff=-1, fock_basis=Val(true))

"""
    DressedLiouvillian(â, mdl::Boson; N_trunc=nothing, σ_filter=Inf)
    DressedLiouvillian(â, b̂, mdl::BosonDimer; N_trunc=nothing, σ_filter=Inf)

Dressed non-secular master equation (Settineri 2018). The field c(â + â') has emission rate ω0c², so c = √(2γ/ω0)
matches √(2γ)â. Non-secular only among transitions of the same sign: the bath is still treated in the RWA, in the
eigenbasis of H. `σ_filter` → 0 is the dressed secular limit, Inf keeps all cross terms. No Lamb shift.
Returns (E, U, L) in the eigenbasis of H (see `dressed_modes`).
"""
DressedLiouvillian(â::QuantumObject, mdl::Boson; kwargs...) = _dressed(Hamiltonian(â, mdl), baths(â, mdl), temperature(mdl); kwargs...)
DressedLiouvillian(â::QuantumObject, b̂::QuantumObject, mdl::BosonDimer; kwargs...) = _dressed(Hamiltonian(â, b̂, mdl), baths(â, b̂, mdl), temperature(mdl); kwargs...)
function _dressed(Ĥ, bths, T; N_trunc=nothing, σ_filter=Inf)
    fields = [√(2γ/ω)*(â + â') for (â, γ, ω) ∈ bths]
    return liouvillian_dressed_nonsecular(Ĥ, fields, fill(T, length(bths)); N_trunc, σ_filter)
end

## Spectral analysis

"""
    modes(L::QuantumObject, ops)

Complex mode frequencies ω = iλ of a Liouvillian `L`, given the annihilation operators `ops` of the modes in L's basis.
For these quadratic models (â|, (â'| span an L†-invariant subspace, so only 2 eigenvectors per mode carry ⟨â⟩, ⟨â'⟩:
they are selected by that overlap rather than by position, which fails once the modes reach the imaginary axis.
Dense: L is n²×n² for an n-dimensional Hilbert space.
"""
function modes(L::QuantumObject, ops)
    λ, V = eigen(Matrix(L.data))
    n = size(first(ops), 1)
    w = [sum(abs(tr(ô.data * reshape(V[:, i], n, n))) for â ∈ ops for ô ∈ (â, â')) for i ∈ eachindex(λ)]
    idx = partialsortperm(w, 1:2length(ops), rev=true)
    return sort(im .* λ[idx], by=ω -> (real(ω), imag(ω)))
end

"""
    dressed_modes((E, U, L), ops)

`modes` of a `DressedLiouvillian`: the mode operators are first taken to the eigenbasis of H.
"""
dressed_modes((E, U, L), ops) = modes(L, [U'*â*U for â ∈ ops])

"""
    parity_sector(N, parity)

Indices of the vectorised ρ of one mode with N Fock states (column-major, ρ[m, n] ↦ m + N n + 1 for 0-based m, n)
in the parity sector (-1)^(m + n) = `parity`. A parity-conserving Liouvillian (x̂-coupled bath, even H) is block
diagonal in these sectors; ⟨a⟩ and ⟨a'⟩ live in the odd one (parity = -1).
"""
parity_sector(N, parity) = [m + N*n + 1 for n ∈ 0:N-1 for m ∈ 0:N-1 if iseven(m + n) == (parity == 1)]

"""
    min_gap(L::QuantumObject, idx; K=6)

Smallest distance between two of the `K` slowest eigenvalues (largest real part) of `L` restricted to the indices
`idx`, as (gap, λ1, λ2). A vanishing gap is a candidate EP; restricting to one symmetry sector avoids level crossings
between sectors, which also close the gap but are not EPs. Dense.
"""
function min_gap(L::QuantumObject, idx; K=6)
    λ = eigvals(Matrix(L.data[idx, idx]))
    λ = partialsort(λ, 1:K, by=real, rev=true)
    gaps = [(abs(λ[i] - λ[j]), λ[i], λ[j]) for i ∈ 1:K for j ∈ i+1:K]
    return gaps[argmin(first.(gaps))]
end

"""
    dominant_mode(L::QuantumObject, a::QuantumObject, idx; ωmax=3.)

Dominant ⟨a⟩ mode of `L` restricted to the sector `idx` of one mode: the eigenvalue (as ω = iλ) whose right
eigenvector has the largest overlap |tr(a ρ)| + |tr(a'ρ)|, among those with |Re ω| < `ωmax` (excludes high-frequency
cutoff artefacts). L is real, so ω and its mirror -conj(ω) have the same overlap: the one with Re ω ≥ 0 is returned.
Returns (ω, largest Re λ in the sector), the latter as a stability check. Unambiguous on the underdamped side; past
an EP other overdamped eigenvalues compete for the overlap. Dense.
"""
function dominant_mode(L::QuantumObject, a::QuantumObject, idx; ωmax=3.)
    λ, V = eigen(Matrix(L.data[idx, idx]))
    ta, tad = vec(transpose(a.data))[idx], vec(transpose(a.data'))[idx]     # tr(a ρ) = vec(aᵀ) ⋅ vec(ρ)
    ok = findall(l -> abs(imag(l)) < ωmax, λ)
    w = [abs(conj(ta) ⋅ V[:, i]) + abs(conj(tad) ⋅ V[:, i]) for i ∈ ok]
    ω = im*λ[ok[argmax(w)]]
    return complex(abs(real(ω)), imag(ω)), maximum(real, λ)
end
