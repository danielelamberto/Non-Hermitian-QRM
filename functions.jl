module MyFunctions

using LinearAlgebra
using QuantumToolbox

export get_shifted_eigvals, get_shifted_eigvals_dense, H_Dicke, H_Dicke_sb, H_comp_separated, H_Dicke_separated, H_Dicke_polaron, J_separated
export get_shifted_eigvals_NH, get_shifted_eigvals_dense_NH, H_Dicke_NH, H_comp_NH_separated, H_Dicke_NH_separated
export parity_blocks_NH, parity_blocks_NH_coll
export gen_liouvillian_qrm, qrm_emission_field, lep_indicators, lep_min_gap, track_pair, encircle_swap, ep_feshbach

# Wrap in a QuantumObject only if needed, so the eigenvalue helpers accept both raw matrices and QuantumObject blocks.
_as_qobj(H::QuantumObject) = H
_as_qobj(H) = Qobj(H)

# Hermitian eigenvalue helpers: return the real part of the eigenvalues, shifted so that the ground state is at zero.

function get_shifted_eigvals(H, nev, sigma, tol)
    vals, _ = eigsolve(H, eigvals=nev, sigma=sigma, tol=tol)
    return real.(vals .- vals[1])
end

function get_shifted_eigvals_dense(H, nev)
    res_vals = eigenstates(Qobj(H))
    vals = res_vals.values
    return real.(vals[1:nev] .- vals[1])
end

# Non-Hermitian eigenvalue helpers: keep complex eigenvalues. The spectrum is ordered by real part; only the real part of the ground state is subtracted, so the imaginary parts stay absolute.

function get_shifted_eigvals_NH(H, nev, sigma, tol)
    vals, _ = eigsolve(_as_qobj(H); eigvals=nev, sigma=sigma, tol=tol)
    e = sort(vals; by=real)
    return e .- real(e[1])
end

function get_shifted_eigvals_dense_NH(H, nev)
    vals = eigvals(to_dense(_as_qobj(H)).data)
    e = sort(vals; by=real)
    return e[1:nev] .- real(e[1])
end

generate_op(O, idx::Int, N::Int) = mapreduce(i -> i == idx ? O : eye(2), kron, 1:N)
generate_a(N::Int, Nc::Int) = kron(destroy(Nc), [eye(2) for i in 1:N]...)
generate_collective_op(O, N::Int, Nc::Int) = mapreduce(i -> kron(eye(Nc), generate_op(O, i, N)), +, 1:N)

# Hermitian Hamiltonians for the Dicke model and its variants.

"""
    H_Dicke(g, vars)

Build the Dicke Hamiltonian for N two-level systems.
"""
function H_Dicke(g::Float64, vars::NamedTuple)
    Nc, N, ωa, ωb = vars.Nc, vars.N, vars.ωa, vars.ωb

    a = generate_a(N, Nc)
    Jx = generate_collective_op(sigmax()/2, N, Nc)
    Jz = generate_collective_op(sigmaz()/2, N, Nc)
    return ωb * Jz + ωa * a' * a + g * 2 / sqrt(N) * (a + a') * Jx
end

"""
    H_Dicke_sb(g, ε, vars)

Build the Dicke Hamiltonian for N two-level systems with the addition of a symmetry-breaking term in the Hamiltonian. The symmetry-breaking term is proportional to the collective spin operator Jx, and its strength is given by the parameter ε. 
"""
function H_Dicke_sb(g::Float64, ε::Float64, vars::NamedTuple)
    Nc, N, ωa, ωb = vars.Nc, vars.N, vars.ωa, vars.ωb

    a = generate_a(N, Nc)
    Jx = generate_collective_op(sigmax()/2, N, Nc)
    Jz = generate_collective_op(sigmaz()/2, N, Nc)
    return ωb * Jz + ωa * a' * a + g * 2 / sqrt(N) * (a + a') * Jx + ε * Jx
end

"""
    J_separated(N)

Build the total angular momentum operators Jx, Jy, Jz for N two-level systems, separated into blocks of conserved angular momentum. Returns a tuple of three arrays, each containing the corresponding operator for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function J_separated(N::Int)
    js = range(N/2, step=-1, length=Int(floor(N/2 + 1)))
    return ([spin_Jx(j) for j in js], [spin_Jy(j) for j in js], [spin_Jz(j) for j in js])
end

"""
    H_comp_separated(vars)

Build the bare Hamiltonian H_0 and interaction Hamiltonian H_int for N two-level systems, separated into blocks of conserved angular momentum. Returns a tuple of two arrays, H_0 and H_int, each containing the corresponding operator for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function H_comp_separated(vars::NamedTuple)
    Nc, N, ωa, ωb = vars.Nc, vars.N, vars.ωa, vars.ωb
    J_sep = J_separated(N)
    H_0_sep = QuantumObject[]
    H_int_sep = QuantumObject[]

    for i in eachindex(J_sep[1])
        a = kron(destroy(Nc), eye(size(J_sep[1][i], 1)))
        push!(H_0_sep, ωb * kron(eye(Nc), J_sep[3][i]) + ωa * a' * a)
        push!(H_int_sep, 2 / sqrt(N) * (a + a') * kron(eye(Nc), J_sep[1][i]))
    end

    return H_0_sep, H_int_sep
end
    
"""
    H_Dicke_separated(H_0, H_int, g)

Build the Dicke Hamiltonian for N two-level systems, separated into blocks of conserved angular momentum. Returns a tuple containing the corresponding Hamiltonian for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function H_Dicke_separated(H_0, H_int, g::Float64)
    result = QuantumObject[]

    for i in range(1, length(H_0))
        push!(result, H_0[i] + g * H_int[i])
    end

    return result
end

"""
    H_Dicke_polaron(H_0, H_int, g)

Build the Dicke Hamiltonian for N two-level systems in the polaron frame, separated into blocks of conserved angular momentum. Returns a tuple containing the corresponding Hamiltonian for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function H_Dicke_polaron(g, vars)
    Nc, N, ωa, ωb = vars.Nc, vars.N, vars.ωa, vars.ωb
    J_sep = J_separated(N)
    a_ph = destroy(Nc)
    n = a_ph' * a_ph

    λ = 2 * g / (ωa * sqrt(N))
    A = to_dense(a_ph' - a_ph).data
    cosh_op = Qobj(cosh(λ * A))
    sinh_op = Qobj(sinh(λ * A))

    result = QuantumObject[]
    for i in eachindex(J_sep[1])
        Is = eye(size(J_sep[1][i], 1))
        Jx = kron(eye(Nc), J_sep[1][i])
        Jy = kron(eye(Nc), J_sep[2][i])
        Jz = kron(eye(Nc), J_sep[3][i])

        push!(result, ωa * kron(n, Is) + ωb * (Jz * kron(cosh_op, Is) - 1im * Jy * kron(sinh_op, Is)) - 4 * g^2 / (ωa * N) * Jx^2)
    end

    return result
end

# Non-Hermitian Hamiltonians for the Dicke model and its variants.

"""
    H_Dicke_NH(g, vars)

Build the effective non-Hermitian Dicke Hamiltonian for N two-level systems, with photon and qubit damping. The dissipation enters through the effective frequencies ωa → ωa - i·γa/2 (photon) and ωb → ωb - i·γb/2 (qubit).
"""
function H_Dicke_NH(g::Real, vars::NamedTuple)
    Nc, N, ωa, ωb, γa, γb = vars.Nc, vars.N, vars.ωa, vars.ωb, vars.γa, vars.γb
    ωa_eff = ωa - 1im * γa / 2
    ωb_eff = ωb - 1im * γb / 2

    a = generate_a(N, Nc)
    Jx = generate_collective_op(sigmax()/2, N, Nc)
    Jz = generate_collective_op(sigmaz()/2, N, Nc)
    return ωb_eff * Jz + ωa_eff * a' * a + g * 2 / sqrt(N) * (a + a') * Jx
end

"""
    H_comp_NH_separated(vars)

Non-Hermitian counterpart of `H_comp_separated`: build the effective bare Hamiltonian H_0 (now complex, carrying the -i·γa/2 and -i·γb/2 damping terms) and the interaction Hamiltonian H_int (unchanged, real) for each block of conserved angular momentum.
Returns (H_0, H_int), each an array of blocks ordered by decreasing angular momentum.
"""
function H_comp_NH_separated(vars::NamedTuple)
    Nc, N, ωa, ωb, γa, γb = vars.Nc, vars.N, vars.ωa, vars.ωb, vars.γa, vars.γb
    ωa_eff = ωa - 1im * γa / 2
    ωb_eff = ωb - 1im * γb / 2
    J_sep = J_separated(N)
    H_0_sep = QuantumObject[]
    H_int_sep = QuantumObject[]

    for i in eachindex(J_sep[1])
        a = kron(destroy(Nc), eye(size(J_sep[1][i], 1)))
        push!(H_0_sep, ωb_eff * kron(eye(Nc), J_sep[3][i]) + ωa_eff * a' * a)
        push!(H_int_sep, 2 / sqrt(N) * (a + a') * kron(eye(Nc), J_sep[1][i]))
    end

    return H_0_sep, H_int_sep
end

"""
    H_Dicke_NH_separated(H_0, H_int, g)

Assemble the effective non-Hermitian Dicke Hamiltonian per block, given the components from `H_comp_NH_separated`. The combination H_0 + g·H_int is identical in form to the Hermitian case: all the non-Hermiticity is already baked into H_0.
"""
function H_Dicke_NH_separated(H_0, H_int, g::Real)
    result = QuantumObject[]

    for i in eachindex(H_0)
        push!(result, H_0[i] + g * H_int[i])
    end

    return result
end

"""
    parity_blocks_NH(g, vars)

Split the effective non-Hermitian Dicke Hamiltonian into its two Z₂-parity sectors.
Parity Π = (-1)^(a†a + Jz + N/2) commutes with H, so H is block-diagonal in the computational basis (where Π is diagonal). Returns (H_even, H_odd) as dense matrices (the even/odd sectors), which can then be diagonalized independently, useful to tell apart exceptional points (same-parity level coalescence, defective) from diabolic points (different-parity crossings, non-defective).
"""
function parity_blocks_NH(g, vars::NamedTuple)
    Nc, N = vars.Nc, vars.N
    H = to_dense(H_Dicke_NH(g, vars)).data

    a   = generate_a(N, Nc)
    nph = real.(diag(to_dense(a' * a).data))
    exc = real.(diag(to_dense(generate_collective_op(sigmaz()/2, N, Nc)).data)) .+ N/2
    signs = round.(Int, nph .+ exc)

    even = findall(iseven, signs)
    odd  = findall(isodd,  signs)
    return H[even, even], H[odd, odd]
end

"""
    parity_blocks_NH_coll(g, vars)

Parity (Z₂) blocks of the effective NH Dicke Hamiltonian in the collective (maximal-spin j = N/2) sector, of dimension Nc·(N+1). 
Returns (H_even, H_odd) as dense matrices. For N = 1 it coincides with `parity_blocks_NH`.
"""
function parity_blocks_NH_coll(g, vars::NamedTuple)
    Nc, N, ωa, ωb, γa, γb = vars.Nc, vars.N, vars.ωa, vars.ωb, vars.γa, vars.γb
    ωa_eff = ωa - 1im * γa / 2
    ωb_eff = ωb - 1im * γb / 2

    j  = N / 2
    Jz = spin_Jz(j); Jx = spin_Jx(j)
    d  = size(Jz, 1)                       # = N + 1
    a   = kron(destroy(Nc), eye(d))
    Jzf = kron(eye(Nc), Jz)
    Jxf = kron(eye(Nc), Jx)
    H   = ωb_eff * Jzf + ωa_eff * a' * a + g * 2 / sqrt(N) * (a + a') * Jxf
    Hd  = to_dense(H).data

    nph = real.(diag(to_dense(a' * a).data))
    exc = real.(diag(to_dense(Jzf).data)) .+ N/2
    signs = round.(Int, nph .+ exc)
    even = findall(iseven, signs)
    odd  = findall(isodd,  signs)
    return Hd[even, even], Hd[odd, odd]
end


# Genralized Liouvillian of the QRM with photon and qubit baths, and LEP indicators.

"""
    gen_liouvillian_qrm(g, vars, γ; T = 0.0, kwargs...)

Wrapper over `liouvillian_dressed_nonsecular`: generalized (dressed, non-secular) Liouvillian of the QRM, with **gauge-consistent** system–bath coupling fields.

`H_Dicke` is the dipole-gauge QRM with coupling `g(a+a†)σx`, i.e. the paper's dipole
Hamiltonian in the photon frame rotated by `a → -ia`. In that frame the **physical
vector potential** that couples the cavity to its bath is `A ∝ i(a-a†)` (NOT `a+a†`),
while the qubit couples via `σx` (gauge-invariant). Using the bare `a+a†` overestimates
the cavity rates by orders of magnitude in USC/DSC — it misses the light–matter
decoupling. The matching photodetection (electric-field) operator is `qrm_emission_field`.

`γ` and `T` are each a scalar (shared photon/qubit value) or a 2-element `[γa, γb]` /
`[Ta, Tb]` (independent photon/qubit channels); a channel is added only if its rate is
`> 0`. Returns `(E, U, L)`: dressed energies (truncated if `N_trunc` is given), the
eigenvector map, and the generalized Liouvillian.

`ε` adds a **parity-breaking** bias `ε·σx` to `H` (a flux offset in a flux-qubit
realization). `σx` is parity-odd (`Π σx Π† = −σx`, `Π = (−1)^{a†a}σz`), so `ε ≠ 0` breaks
the Z₂ parity, merges the two weak-symmetry blocks of the Liouvillian, and turns on the
interblock coupling `∝ ε` that converts the parity-protected diabolic points into genuine
`EP2`s. At `ε = 0` (default) parity is preserved and light–matter degeneracies are diabolic.
"""
function gen_liouvillian_qrm(g::Real, vars::NamedTuple, γ::Union{Real, AbstractVector{<:Real}}; T::Union{Real, AbstractVector{<:Real}} = 0.0, ε::Real = 0.0, kwargs...)
    Nc, N = vars.Nc, vars.N
    a   = generate_a(N, Nc)
    A   = 1im * (a - a')
    sq  = generate_collective_op(sigmax(), N, Nc)
    H   = H_Dicke_sb(g, ε, vars)

    γa, γb = γ isa Real ? (γ, γ) : (γ[1], γ[2])
    Ta, Tb = T isa Real ? (T, T) : (T[1], T[2])

    fields = QuantumObject[]
    T_list = Float64[]
    γa > 0 && (push!(fields, sqrt(γa) * A);  push!(T_list, Ta))
    γb > 0 && (push!(fields, sqrt(γb) * sq); push!(T_list, Tb))

    return liouvillian_dressed_nonsecular(H, fields, T_list; kwargs...)
end

"""
    qrm_emission_field(g, vars)

Gauge-consistent photodetection operator (physical electric field) for the QRM in the frame of `H_Dicke`, in the bare (photon ⊗ qubit) basis:

    Ê ∝ (a + a†) + 2η·σx ,     η = g / ωc      (collective σx for N > 1).

This is the dipole-gauge electric field, transformed through the `a → -ia` rotation.
"""
function qrm_emission_field(g::Real, vars::NamedTuple)
    Nc, N = vars.Nc, vars.N
    a  = generate_a(N, Nc)
    η  = g / vars.ωa
    sq = generate_collective_op(sigmax(), N, Nc)
    return (a + a') + 2η * sq
end

"""
    lep_indicators(L; rewin = (-0.1, -0.003), imcut = 0.15)

Locate the coalescing eigenpair of a Liouvillian within a physical eigenvalue window
and quantify its exceptional-point character. Among the right eigenvectors whose
eigenvalues fall in the window `real ∈ rewin`, `|imag| < imcut` (chosen to isolate the
slow, single-excitation "vacuum-Rabi" modes near `Re ≈ -γa/2`), it selects the pair with
**maximal eigenmatrix overlap** — the EP signature — rather than merely the smallest gap.

`L` may be a `SuperOperator` `QuantumObject` or a plain matrix. Returns a NamedTuple:
- `gap`     = |λ₁ − λ₂|                         → 0 at an EP;
- `overlap` = |⟨ρ₁|ρ₂⟩|/(‖ρ₁‖‖ρ₂‖) (HS)        → 1 at an EP;
- `λ1, λ2`  the coalescing Liouvillian eigenvalues;
- `ρ1, ρ2`  the two eigenmatrices (reshaped), so their coalescence can be inspected directly.
"""
function lep_indicators(L; rewin = (-0.1, -0.003), imcut = 0.15)
    M = L isa AbstractMatrix ? Matrix(L) : Matrix(L.data)
    F = eigen(M)
    vals = F.values
    V = copy(F.vectors)
    for j in axes(V, 2)
        V[:, j] ./= norm(V[:, j])
    end
    idx = filter(i -> rewin[1] < real(vals[i]) < rewin[2] && abs(imag(vals[i])) < imcut, eachindex(vals))
    length(idx) < 2 && return (gap = NaN, overlap = NaN, λ1 = NaN + 0im, λ2 = NaN + 0im, ρ1 = nothing, ρ2 = nothing)
    best = -1.0; pa = 0; pb = 0
    for ii in 1:length(idx), jj in ii+1:length(idx)
        a, b = idx[ii], idx[jj]
        ov = abs(dot(V[:, a], V[:, b]))
        ov > best && (best = ov; pa = a; pb = b)
    end
    d = isqrt(size(M, 1))
    return (gap = abs(vals[pa] - vals[pb]), overlap = best,
            λ1 = vals[pa], λ2 = vals[pb],
            ρ1 = reshape(V[:, pa], d, d), ρ2 = reshape(V[:, pb], d, d))
end

"""
    lep_min_gap(L; exclude_zero=true, tol_zero=1e-9, window=nothing, k=1, overlap=false)

Locate near-degeneracies of a Liouvillian by the **minimum pairwise eigenvalue distance**
— an eigenvalue-only estimator, reliable in USC where the eigenvectors of the non-normal
generator are ill-conditioned (unlike the overlap in `lep_indicators`). Meant to be mapped
over a parameter grid to find the dips, then to classify each dip separately.

`L` may be a `SuperOperator` `QuantumObject` or a plain matrix. Options:
- `exclude_zero`  drop the trivial steady-state mode (|λ| < `tol_zero`) so it does not
                  masquerade as a degeneracy paired with a slow decaying mode;
- `window = (remin, remax, immin, immax)`  keep only eigenvalues inside this rectangle of
                  the complex plane (e.g. the slow modes near `Re ≈ -κ/2`); `nothing` = all.
                  Note: a conjugate pair `λ, λ*` merging on the real axis is a *genuine* LEP
                  (critical-damping type); include `Im = 0` in the window to see it;
- `k`             number of smallest-gap pairs returned (to catch a second/near EP);
- `overlap`       also compute the eigenmatrix overlap |⟨ρᵢ|ρⱼ⟩| (EP → 1) of each returned
                  pair. Off by default: for a *scan* use the gap only (values only, no
                  eigenvectors); switch it on just at the candidates.

Returns a NamedTuple for the closest pair — `gap = |λᵢ − λⱼ|`, indices `i, j`, `λi, λj`,
`loc = (λi+λj)/2` (position in the complex plane) — plus `pairs`, the list of the `k`
closest pairs (each with the same fields, and `overlap, ρi, ρj` when `overlap=true`).

CAUTION: a small `gap` at a *single* point is not an EP — both diabolic points and EPs give
gap → 0. Map `gap(params)`, find localized dips, then classify each by the **scaling** of
`gap` along a cut through it (∝ √ ⇒ EP, ∝ linear ⇒ DP); use the overlap only as a cross-check.
Grid minima are generically *between* nodes, so refine (nested grid / local optimizer)
before quoting an EP location.
"""
function lep_min_gap(L; exclude_zero::Bool = true, tol_zero::Real = 1e-9,
                     window = nothing, k::Int = 1, overlap::Bool = false)
    M = L isa AbstractMatrix ? Matrix(L) : Matrix(L.data)
    if overlap                                   # need eigenvectors only for the overlap
        F = eigen(M); vals = F.values; V = copy(F.vectors)
        for j in axes(V, 2)
            V[:, j] ./= norm(V[:, j])
        end
    else                                         # scan path: eigenvalues only (cheaper)
        vals = eigvals(M); V = nothing
    end
    idx = collect(eachindex(vals))
    exclude_zero && (idx = filter(i -> abs(vals[i]) > tol_zero, idx))
    if window !== nothing
        rmn, rmx, imn, imx = window
        idx = filter(i -> rmn ≤ real(vals[i]) ≤ rmx && imn ≤ imag(vals[i]) ≤ imx, idx)
    end
    n = length(idx)
    n < 2 && return (gap = NaN, i = 0, j = 0, λi = NaN + 0im, λj = NaN + 0im,
                     loc = NaN + 0im, pairs = NamedTuple[], overlap = NaN)
    gaps = Tuple{Float64,Int,Int}[]              # every pairwise gap in the filtered set
    for ii in 1:n-1, jj in ii+1:n
        a, b = idx[ii], idx[jj]
        push!(gaps, (abs(vals[a] - vals[b]), a, b))
    end
    sort!(gaps; by = first)
    kk = min(k, length(gaps))
    d  = isqrt(size(M, 1))
    pairs = map(1:kk) do t
        g, a, b = gaps[t]
        p = (gap = g, i = a, j = b, λi = vals[a], λj = vals[b], loc = (vals[a] + vals[b]) / 2)
        overlap ? merge(p, (overlap = abs(dot(V[:, a], V[:, b])),
                            ρi = reshape(V[:, a], d, d), ρj = reshape(V[:, b], d, d))) : p
    end
    top = pairs[1]
    return (gap = top.gap, i = top.i, j = top.j, λi = top.λi, λj = top.λj,
            loc = top.loc, pairs = pairs, overlap = overlap ? top.overlap : NaN)
end

"""
    track_pair(Llist, λ1, λ2)

Follow a single eigenpair across a sequence of (super)operators `Llist` **by continuity**,
starting from the reference eigenvalues `(λ1, λ2)`. Unlike `lep_min_gap` — which re-picks the
globally closest pair at each point and can jump between different pairs — this locks onto the
same two modes step by step, so an EP scan yields clean, continuous trajectories.

At each step it diagonalizes, selects the eigenvalues nearest to the previous references
(falling back to the 2nd-nearest if the two would collide onto one index), and records the
Hilbert–Schmidt overlap of the corresponding right eigenmatrices. Returns a NamedTuple of
vectors: `λ1s, λ2s` (the two trajectories in ℂ), `mid = (λ1+λ2)/2` (coalescence location),
`gaps = |λ1−λ2|` (→ 0 at the EP), `overlaps = |⟨ρ1|ρ2⟩|` (→ 1 at the EP).

Diagnostics: near a genuine EP the gap shows a **√-cusp** in the scan parameter and the
overlap peaks toward 1; a diabolic point gives a **linear V** gap and no eigenvector alignment.
Overlap alone is unreliable in USC (ill-conditioned eigenvectors) — read it together with the
gap shape and the ℂ-plane trajectories.
"""
function track_pair(Llist, λ1::Number, λ2::Number)
    λ1s = ComplexF64[]; λ2s = ComplexF64[]; gaps = Float64[]; ovs = Float64[]
    r1, r2 = ComplexF64(λ1), ComplexF64(λ2)
    for L in Llist
        M = L isa AbstractMatrix ? Matrix(L) : Matrix(L.data)
        F = eigen(M); vals = F.values; V = F.vectors
        for k in axes(V, 2)
            V[:, k] ./= norm(V[:, k])
        end
        i1 = argmin(abs.(vals .- r1))
        p2 = sortperm(abs.(vals .- r2))
        i2 = p2[1] == i1 ? p2[2] : p2[1]              # avoid both tracks landing on one mode
        push!(λ1s, vals[i1]); push!(λ2s, vals[i2])
        push!(gaps, abs(vals[i1] - vals[i2]))
        push!(ovs, abs(dot(V[:, i1], V[:, i2])))
        r1, r2 = vals[i1], vals[i2]                   # advance references (continuity)
    end
    return (λ1s = λ1s, λ2s = λ2s, mid = (λ1s .+ λ2s) ./ 2, gaps = gaps, overlaps = ovs)
end

"""
    encircle_swap(Ls, λ1, λ2)

Monodromy (encirclement) test for an exceptional point. `Ls` is a sequence of operators
sampled around a **closed loop** in the 2-real-parameter plane (first and last point equal,
the loop enclosing the candidate). The pair `(λ1, λ2)` is followed by continuity around the
loop (`track_pair`); after one full turn the two eigenvalues **swap** at a genuine EP (the
√-branch point needs two loops to return) but **return to themselves** at a diabolic point
or when no EP is enclosed.

Returns `(tr, swapped, d_swap, d_same)`: the full `track_pair` trajectory, the boolean verdict,
and the two endpoint distances it is based on (`d_swap < d_same` ⇒ swapped). Plot `tr.λ1s`,
`tr.λ2s` in ℂ to see the interleaving arcs of a true EP.
"""
function encircle_swap(Ls, λ1::Number, λ2::Number)
    tr = track_pair(Ls, λ1, λ2)
    d_swap = abs(tr.λ1s[end] - λ2) + abs(tr.λ2s[end] - λ1)
    d_same = abs(tr.λ1s[end] - λ1) + abs(tr.λ2s[end] - λ2)
    return (tr = tr, swapped = d_swap < d_same, d_swap = d_swap, d_same = d_same)
end

"""
    ep_feshbach(L; window = nothing, tol = 1e-3)

Reliable EP2-vs-DP classifier for a Liouvillian, via the **Feshbach 2×2 reduction** —
the method that survives the dense, non-normal spectrum where eigenvector overlap and
closest-pair monodromy mislead. It compresses `L` onto the 2-D invariant subspace of its
closest eigenpair (in the optional complex-plane `window`) using an **ordered complex
Schur** decomposition, `M2 = Z₂† L Z₂`, and cross-checks with the kernel dimension.

`L` may be a `SuperOperator` `QuantumObject` or a plain matrix. Returns a NamedTuple:
- `gap`     = |λᵢ − λⱼ| of the closest pair;
- `t`       = |M2[1,2]|, the Schur off-diagonal. **The decisive quantity:** as the pair is
              driven to coalescence (`gap → 0`), `t` stays **finite** at an EP2 (M2 → a
              Jordan block) but **→ 0** at a DP (M2 → a scalar). Do this "t-pinning" test
              across a parameter sweep for a gold-standard verdict;
- `tratio`  = t / gap;
- `overlap` = eigenvector overlap of the (well-conditioned) 2×2 → 1 at an EP;
- `λ`       = coalescence location (λᵢ+λⱼ)/2;
- `svmin, sv2, svratio` = the two smallest singular values of `L − λ·I` and their ratio.
              `dim ker(L − λ·I) = 1` (svratio ≫ 1) ⇒ **EP2** (defective); `= 2` (svratio ~ 1)
              ⇒ **DP** (semisimple);
- `verdict` — from **dim ker** (`svratio = sv2/svmin`), the robust discriminator: `svratio ≫
              10³` ⇒ `EP2` (one SV → 0, defective); coalesced `gap` with `svratio ~ O(1)` ⇒
              `DP` (two SVs → 0 together, semisimple); otherwise refine. **`t/gap` is NOT
              used for the verdict** — it diverges spuriously at a floored gap, or when the
              only route to `gap → 0` is a rate `κ → 0` (then `t ∝ κ → 0`, M2 → scalar = DP);
- `M2`      — the 2×2 block, for inspection.

Definitive test: drive `gap → 0` at **fixed physical parameters** (2-knob refinement, not by
lowering a rate); a true EP2 keeps `t` pinned (finite) with `svratio` blowing up, whereas a DP
has `t → 0` and `svratio ~ O(1)`. At a single, poorly-localized point the verdict is `ambiguo`.
"""
function ep_feshbach(L; window = nothing, tol = 1e-4)
    M = L isa AbstractMatrix ? Matrix(L) : Matrix(L.data)
    F = schur(M); vals = copy(F.values); n = length(vals)
    inw(λ) = window === nothing ? true :
        (window[1] ≤ real(λ) ≤ window[2] && window[3] ≤ imag(λ) ≤ window[4])
    cand = findall(inw, vals)
    length(cand) < 2 && return (gap = NaN, t = NaN, tratio = NaN, overlap = NaN, λ = NaN + 0im,
                                svmin = NaN, sv2 = NaN, svratio = NaN, verdict = "no pair",
                                M2 = zeros(ComplexF64, 2, 2))
    bi = bj = 0; best = Inf
    for ii in 1:length(cand)-1, jj in ii+1:length(cand)
        i, j = cand[ii], cand[jj]; d = abs(vals[i] - vals[j]); d < best && (best = d; bi = i; bj = j)
    end
    sel = falses(n); sel[bi] = true; sel[bj] = true; ordschur!(F, sel)
    Z2 = F.Z[:, 1:2]; M2 = Z2' * M * Z2
    e = eigen(M2)
    v1 = e.vectors[:, 1] / norm(e.vectors[:, 1]); v2 = e.vectors[:, 2] / norm(e.vectors[:, 2])
    λc = (vals[bi] + vals[bj]) / 2
    sv = svdvals(M - λc * I); svmin, sv2 = sv[end], sv[end-1]
    t = abs(M2[1, 2]); tratio = t / max(best, 1e-300); svratio = sv2 / max(svmin, 1e-300)
    # Verdict from dim ker (svratio), the ROBUST discriminator: at a genuine EP2 exactly ONE
    # singular value of (L−λ·I) → 0 (svratio ≫ 1); at a DP two vanish together (svratio ~ 1).
    # NB: `tratio = t/gap` alone is NOT reliable — it diverges spuriously at a FLOORED gap, or
    # when the only route to gap→0 is a rate κ→0 with t ∝ κ → 0 (M2 → scalar = DP, not Jordan).
    # A true EP2 keeps t PINNED (finite) as gap→0 at FIXED physical parameters. So drive the gap
    # to ~0 at fixed rates first, then trust svratio: ≫10³ ⇒ EP2, O(1) ⇒ DP.
    verdict = svratio > 1e3 ? "EP2 (dim ker = 1: only 1 SV → 0)" : "ambigous"
    return (gap = best, t = t, tratio = tratio, overlap = abs(dot(v1, v2)), λ = λc, svmin = svmin, sv2 = sv2, svratio = svratio, verdict = verdict, M2 = M2)
end


end # module