module MyFunctions

using LinearAlgebra
using QuantumToolbox
using ProgressMeter

export get_shifted_eigvals, get_shifted_eigvals_dense, H_QRM, H_QRM_sb
export gen_liouvillian_QRM, QRM_emission_field
export parity_charge, liouvillian_matrix, parity_leakage
export scanned_keys, liouvillian_at, ep_grid, find_real_eps, find_complex_eps, check_ep, check_eps, ep_search
# previous interface, still used by the ε ≠ 0 cells of the notebook (to be migrated)
export ep_feshbach, locate_ep, confirm_ep, gap_scaling

# Wrap in a QuantumObject
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

generate_a(Nc::Int) = kron(destroy(Nc), eye(2))
generate_mat_op(O, Nc::Int) = kron(eye(Nc), O)

# Hermitian Hamiltonians for the QRM model and its variants.

"""
    H_QRM(g, vars)

Build the QRM Hamiltonian for a two-level system coupled to a single bosonic mode, with coupling strength `g`, in the **dipole gauge**:

    H = ωa a†a + ωb σz/2 - i g (a - a†) σx .

It follows from the Coulomb-gauge QRM (vector potential A ∝ a + a†, electric field E ∝ i(a - a†)) via `U = exp(-iη(a + a†)σx)`, η = g/ωa, up to the constant ωa η². Gauge-consistent operators in this frame: E ∝ `QRM_emission_field`, A ∝ a + a† (cavity bath), σx (qubit bath).
"""
function H_QRM(g::Real, vars::NamedTuple)
    Nc, ωa, ωb = vars.Nc, vars.ωa, vars.ωb

    a  = generate_a(Nc)
    σx = generate_mat_op(sigmax(), Nc)          # FULL σx: coupling/bias couple to σx, not σx/2
    Jz = generate_mat_op(sigmaz()/2, Nc)        # free-qubit splitting ωb·σz/2
    return ωb * Jz + ωa * a' * a - 1im * g * (a - a') * σx
end

"""
    H_QRM_sb(g, ε, vars)

Build the QRM Hamiltonian for a two-level system coupled to a single bosonic mode, with the term of the form `-ig(a - a†)σx`, with the addition of a symmetry-breaking term in the Hamiltonian `ε·σx`.
"""
function H_QRM_sb(g::Real, ε::Real, vars::NamedTuple)
    Nc, ωa, ωb = vars.Nc, vars.ωa, vars.ωb

    a  = generate_a(Nc)
    σx = generate_mat_op(sigmax(), Nc)          # FULL σx (coupling AND parity-breaking bias)
    Jz = generate_mat_op(sigmaz()/2, Nc)
    return ωb * Jz + ωa * a' * a - 1im * g * (a - a') * σx + ε * σx
end

# Generalized Liouvillian of the QRM with photon and qubit baths.

"""
    gen_liouvillian_QRM(g, vars, γ; T = 0.0, kwargs...)

Wrapper over `liouvillian_dressed_nonsecular`: generalized (dressed, non-secular) Liouvillian of the QRM, with **gauge-consistent** system–bath coupling fields.

`H_QRM_sb` is the dipole-gauge QRM with coupling `-ig(a - a†)σx`, plus a parity-breaking bias `ε·σx` (default `ε = 0`, i.e. the standard cavity-QED QRM). The vector potential that couples the cavity to its bath is `A ∝ (a+a†)` (unchanged by the gauge transformation), while the qubit couples via `σx` (gauge-invariant). The matching photodetection (electric-field) operator is `QRM_emission_field`.

`γ` and `T` are each a scalar (shared photon/qubit value) or a 2-element `[γa, γb]` / `[Ta, Tb]` (independent photon/qubit channels). By default `T = 0.0` (zero-temperature baths). 
Returns `(E, U, L)`: dressed energies (truncated if `N_trunc` is given), the eigenvector map, and the generalized Liouvillian.
"""
function gen_liouvillian_QRM(g::Real, vars::NamedTuple, γ::Union{Real, AbstractVector{<:Real}}; T::Union{Real, AbstractVector{<:Real}} = 0.0, ε::Real = 0.0, kwargs...)
    Nc, ωa, ωb = vars.Nc, vars.ωa, vars.ωb
    a   = generate_a(Nc)
    A   = a + a'                                 # cavity bath field (this gauge: A ∝ a+a†)
    sq  = generate_mat_op(sigmax(), Nc)          # qubit bath field: FULL σx (gauge-invariant)
    H   = H_QRM_sb(g, ε, vars)

    γa, γb = γ isa Real ? (γ, γ) : (γ[1], γ[2])
    Ta, Tb = T isa Real ? (T, T) : (T[1], T[2])

    fields = QuantumObject[]
    T_list = Float64[]
    γa > 0 && (push!(fields, sqrt(γa/ωa) * A);  push!(T_list, Ta))
    γb > 0 && (push!(fields, sqrt(γb/ωb) * sq); push!(T_list, Tb))

    return liouvillian_dressed_nonsecular(H, fields, T_list; kwargs...)
end

"""
    QRM_emission_field(g, vars)

Gauge-consistent photodetection operator (physical electric field) for the QRM in the frame of `H_QRM`, in the bare (photon ⊗ qubit) basis:

    Ê ∝ i (a - a†) - 2η·σx ,     η = g / ωa .

This is the Coulomb-gauge field `E_C ∝ i(a - a†)` transformed to the dipole gauge by `U = exp(-iη(a + a†)σx)` (`U a U† = a + iησx`). The relative sign is fixed by the coupling `-ig(a - a†)σx`: in the displaced ground state ⟨i(a - a†)⟩ = 2η⟨σx⟩, so Ê decouples in the deep-strong-coupling limit.
"""
function QRM_emission_field(g::Real, vars::NamedTuple)
    Nc = vars.Nc
    a  = generate_a(Nc)
    η  = g / vars.ωa
    sq = generate_mat_op(sigmax(), Nc)          # FULL σx
    return 1im*(a - a') - 2η * sq
end

# LEP indicators: eigenvalue-only and eigenvector-based diagnostics of near-degeneracies.

"""
    ep_feshbach(L; window = nothing, ep_svratio = 1e4)

Reliable EP2-vs-DP classifier for a Liouvillian, via the Feshbach 2×2 reduction. It compresses `L` onto the 2-D invariant subspace of its closest eigenpair (in the optional complex-plane `window`) using an ordered complex Schur decomposition, `M2 = Z₂† L Z₂`, and cross-checks with the kernel dimension.

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

Definitive test: drive `gap → 0` at fixed physical parameters (2-knob refinement, not by lowering a rate); a true EP2 keeps `t` pinned (finite) with `svratio` blowing up, whereas a DP has `t → 0` and `svratio ~ O(1)`. At a single, poorly-localized point the verdict is `ambiguous` — use `confirm_ep` for the full 2-step protocol.
"""
function ep_feshbach(L; window = nothing, ep_svratio = 1e4)
    M = L isa AbstractMatrix ? Matrix(L) : Matrix(L.data)
    F = schur(M)
    cand = _in_window(F.values, window)
    length(cand) < 2 && return (gap = NaN, t = NaN, tratio = NaN, overlap = NaN, λ = NaN + 0im, svmin = NaN, sv2 = NaN, svratio = NaN, verdict = "no pair", M2 = zeros(ComplexF64, 2, 2))

    _, bi, bj = _sorted_pairs(F.values, cand)[1]
    # NB: ep_feshbach is a pure per-pair classifier — it always computes t/svratio, even at a
    # large (floored) gap (needed e.g. to read t across a κ-sweep at a diabolic point). Gate on a
    # gap tolerance in the caller (`confirm_ep`), not here.
    r = _feshbach_pair(M, F, bi, bj)
    verdict = r.svratio > ep_svratio ? "EP2 (dim ker = 1: only 1 SV → 0)" : "ambiguous"
    return merge(r, (verdict = verdict,))
end

# ── shared internals: window selection, pair sorting, Feshbach core ──────────────────────────

_dense(L) = L isa AbstractMatrix ? Matrix(L) : Matrix(L.data)
_vars(vars::NamedTuple, ωb::Real) = (ωa = vars.ωa, ωb = ωb, Nc = vars.Nc)

# indices of the eigenvalues inside `window = (remin, remax, immin, immax)` (all if `nothing`),
# optionally dropping the steady state (|λ| < tol_zero)
function _in_window(vals, window; exclude_zero::Bool = false, tol_zero::Real = 1e-9)
    idx = collect(eachindex(vals))
    exclude_zero && (idx = filter(i -> abs(vals[i]) > tol_zero, idx))
    window === nothing && return idx
    return filter(i -> window[1] ≤ real(vals[i]) ≤ window[2] && window[3] ≤ imag(vals[i]) ≤ window[4], idx)
end

# all pairs among `idx`, sorted by gap: Vector of (gap, i, j)
function _sorted_pairs(vals, idx)
    prs = Tuple{Float64,Int,Int}[]
    for a in 1:length(idx)-1, b in a+1:length(idx)
        push!(prs, (abs(vals[idx[a]] - vals[idx[b]]), idx[a], idx[b]))
    end
    return sort!(prs; by = first)
end

# run f() with single-threaded BLAS: the scans are multithreaded over grid points, and nested BLAS
# threads on small (≲200×200) matrices only cause contention
function _blas1(f)
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(nb)
    end
end

# the two eigenvalues nearest to λref (continuity tracking of a pair)
function _pair_near(vals, λref)
    i = sortperm(abs.(vals .- λref))
    return vals[i[1]], vals[i[2]]
end

# indices of the eigenpair nearest to the given values (λi, λj), never the same index twice
function _nearest_pair(vals, λi, λj)
    bi = argmin(abs.(vals .- λi))
    p  = sortperm(abs.(vals .- λj))
    return bi, (p[1] == bi ? p[2] : p[1])
end

# Feshbach 2×2 reduction of M onto the invariant subspace of eigenvalues (bi, bj) of its Schur
# factorization F (ordered Schur, non-mutating), plus the kernel-dimension indicator.
function _feshbach_pair(M::AbstractMatrix, F::Schur, bi::Int, bj::Int)
    vals = F.values
    sel = falses(length(vals)); sel[bi] = true; sel[bj] = true
    Z2 = ordschur(F, sel).Z[:, 1:2]
    M2 = Z2' * M * Z2
    e  = eigen(M2)
    v1 = normalize(e.vectors[:, 1]); v2 = normalize(e.vectors[:, 2])
    gap = abs(vals[bi] - vals[bj])
    λc  = (vals[bi] + vals[bj]) / 2
    sv  = svdvals(M - λc * I)
    svmin, sv2 = sv[end], sv[end-1]
    t = abs(M2[1, 2])
    return (gap = gap, t = t, tratio = t / max(gap, 1e-300), overlap = abs(dot(v1, v2)), λ = λc,
            svmin = svmin, sv2 = sv2, svratio = sv2 / max(svmin, 1e-300), M2 = M2)
end

# ── Weak parity symmetry: charge of the Liouvillian basis and block extraction ──────────────

"""
    parity_charge(U, vars)

Weak-symmetry charge `q = πᵢ πⱼ = ±1` of every Liouvillian basis element `|i⟩⟨j|` in the dressed
basis returned by `gen_liouvillian_QRM` (`U` = eigenvector map). `πᵢ = ⟨i|Π|i⟩` with the QRM
parity `Π = (-1)^{a†a} σz`. Ordered as `vec(ρ)` (column stacking, `i` fastest), i.e. as the rows of
the Liouvillian matrix. Meaningful only at `ε = 0` (warns if a dressed state is not a parity
eigenstate, e.g. for `ε ≠ 0` or at an exact dressed degeneracy).
"""
function parity_charge(U, vars::NamedTuple)
    Nc = vars.Nc
    Π  = Diagonal(kron((-1.0) .^ (0:Nc-1), [1.0, -1.0]))       # (-1)^{a†a} ⊗ σz (photon ⊗ qubit)
    Ud = Matrix(U isa AbstractMatrix ? U : U.data)
    πd = real.(diag(Ud' * Π * Ud))
    any(x -> abs(abs(x) - 1) > 1e-6, πd) &&
        @warn "parity_charge: some dressed states are not parity eigenstates (ε ≠ 0 or exact degeneracy)"
    πs = sign.(πd)
    NT = length(πs)
    return [πs[i] * πs[j] for j in 1:NT for i in 1:NT]
end

# Unitary T from the |i⟩⟨j| basis (column-stacked vec) to an orthonormal basis of HERMITIAN
# matrices: |i⟩⟨i|, (|i⟩⟨j| + |j⟩⟨i|)/√2, i(|i⟩⟨j| - |j⟩⟨i|)/√2 (i < j). `lab[k] = (i, j)`.
const _HERM_CACHE = Dict{Int,Tuple{Matrix{ComplexF64},Vector{Tuple{Int,Int}}}}()
const _HERM_LOCK  = ReentrantLock()
function _herm_basis(N::Int)
    lock(_HERM_LOCK) do
        get!(_HERM_CACHE, N) do
            T = zeros(ComplexF64, N^2, N^2); lab = Tuple{Int,Int}[]; c = 0
            idx(i, j) = i + (j - 1) * N
            for j in 1:N, i in 1:j
                if i == j
                    c += 1; T[idx(i, i), c] = 1; push!(lab, (i, i))
                else
                    c += 1; T[idx(i, j), c] = 1 / sqrt(2);  T[idx(j, i), c] = 1 / sqrt(2);  push!(lab, (i, j))
                    c += 1; T[idx(i, j), c] = 1im / sqrt(2); T[idx(j, i), c] = -1im / sqrt(2); push!(lab, (i, j))
                end
            end
            (T, lab)
        end
    end
end

"""
    liouvillian_matrix(g, vars, γ; ε = 0.0, q = nothing, real_basis = false, kwargs...)

Dense generalized Liouvillian `gen_liouvillian_QRM(g, vars, γ; ε, kwargs...)` as a plain matrix.
With `q = +1` or `q = -1` it returns only the **parity block** of that weak-symmetry charge
(`q = +1`: populations, same-parity coherences and the steady state; `q = -1`: opposite-parity
coherences). Blocks exist only at `ε = 0`. All other `kwargs` (`N_trunc`, `σ_filter`, `T`, ...)
are forwarded.

`real_basis = true` returns the same operator (unitarily equivalent: same spectrum) in an
orthonormal basis of Hermitian matrices, where a Hermiticity-preserving L is a **real** matrix
(checked: the imaginary part is dropped only if ≲ 1e-10‖L‖). The real eigensolver then returns
real eigenvalues with `Im = 0` *exactly* and complex ones in exact conjugate pairs, so "is this
eigenvalue real?" becomes an exact test — the basis of `find_real_eps` and `find_complex_eps`.
"""
function liouvillian_matrix(g::Real, vars::NamedTuple, γ; ε::Real = 0.0, q = nothing,
                            real_basis::Bool = false, kwargs...)
    _, U, L = gen_liouvillian_QRM(g, vars, γ; ε = ε, kwargs...)
    M = Matrix(L.data)
    (q === nothing || ε == 0) || throw(ArgumentError("parity blocks exist only for ε = 0"))
    if !real_basis
        q === nothing && return M
        idx = findall(==(q), parity_charge(U, vars))
        return M[idx, idx]
    end
    N = isqrt(size(M, 1))
    T, lab = _herm_basis(N)
    MH = T' * M * T
    norm(imag(MH)) ≤ 1e-10 * norm(M) || @warn "liouvillian_matrix: L is not Hermiticity-preserving (‖Im‖/‖L‖ = $(norm(imag(MH)) / norm(M)))"
    MR = real(MH)
    q === nothing && return MR
    Π  = Diagonal(kron((-1.0) .^ (0:vars.Nc-1), [1.0, -1.0]))
    Ud = Matrix(U.data)
    πs = sign.(real.(diag(Ud' * Π * Ud)))
    idx = findall(k -> πs[lab[k][1]] * πs[lab[k][2]] == q, eachindex(lab))
    return MR[idx, idx]
end

"""
    parity_leakage(g, vars, γ; kwargs...)

Relative norm of the cross-block part of the Liouvillian, `(‖L₊₋‖ + ‖L₋₊‖)/‖L‖`. At `ε = 0` it
is ~1e-16: exact weak Z₂ symmetry ⇒ block-diagonal L (check before using `q` blocks).
"""
function parity_leakage(g::Real, vars::NamedTuple, γ; kwargs...)
    _, U, L = gen_liouvillian_QRM(g, vars, γ; kwargs...)
    M  = Matrix(L.data)
    c  = parity_charge(U, vars)
    ip = findall(==(1), c); iq = findall(==(-1), c)
    return (norm(M[ip, iq]) + norm(M[iq, ip])) / norm(M)
end

# ════════════════════════════════════════════════════════════════════════════════════════════
# Unified LEP search: SCAN (locate) → LOCATE (2-knob Newton) → CONFIRM (Feshbach + dim ker).
# The same pipeline serves the full Liouvillian (q = nothing) at any bias ε, and a single parity
# block (q = ±1, ε = 0 only).
# ════════════════════════════════════════════════════════════════════════════════════════════

"""
    locate_ep(g0, ωb0, vars, γ; λ0, ε=0.0, q=nothing, iters=40, h=1e-6, maxstep=0.03, kwargs...)

**Step 2 of the EP search.** Drive the eigenpair nearest `λ0` to coalescence by a damped 2-knob
Newton in `(g, ωb)` on `δ² = ((λ₁ - λ₂)/2)² = 0` — analytic at an EP2, whereas the gap itself has
a √-cusp. The pair is re-selected at every step as the two eigenvalues nearest the current midpoint
(continuity: `λ0` is essential, otherwise the Newton drifts onto another near-degeneracy). A
backtracking line search on `|δ²|` makes the iteration converge to the local **minimum** of the
gap when no coalescence exists at these rates (avoided crossing: the gap *floors*) instead of
wandering. All other parameters (`γ`, `ε`, `N_trunc`, `σ_filter`, ...) stay fixed. With
`box = ((gmin, gmax), (ωbmin, ωbmax))` the iteration is confined to that rectangle.

Returns `(g, ωb, λ, λ1, λ2, gap)`.
"""
function locate_ep(g0::Real, ωb0::Real, vars::NamedTuple, γ; λ0::Number, ε::Real = 0.0, q = nothing,
                   iters::Int = 40, h::Real = 1e-6, maxstep::Real = 0.03, box = nothing, kwargs...)
    pair(g, w, λr) = _pair_near(eigvals(liouvillian_matrix(g, _vars(vars, w), γ; ε = ε, q = q, kwargs...)), λr)
    δ2(p) = ((p[1] - p[2]) / 2)^2
    # optional search box ((gmin, gmax), (ωbmin, ωbmax)): trial points are clamped inside it
    inbox(g, w) = box === nothing ? (g, w) : (clamp(g, box[1]...), clamp(w, box[2]...))
    g, w = inbox(float(g0), float(ωb0)); λr = ComplexF64(λ0)
    p = pair(g, w, λr); f = δ2(p); λr = (p[1] + p[2]) / 2
    for _ in 1:iters
        fg = δ2(pair(g + h, w, λr)); fw = δ2(pair(g, w + h, λr))
        J  = [real(fg - f) real(fw - f); imag(fg - f) imag(fw - f)] ./ h
        r  = [real(f); imag(f)]
        μ  = 1e-12 * (sum(abs2, J) + eps())                  # tiny LM shift: finite step if J is singular
        s  = -((J' * J + μ * I) \ (J' * r))
        n  = norm(s); n > maxstep && (s .*= maxstep / n)
        accepted = false
        for _ in 1:30                                         # backtracking on |δ²|
            gt, wt = inbox(g + s[1], w + s[2])
            (gt, wt) == (g, w) && break                       # blocked by the box
            pn = pair(gt, wt, λr); fn = δ2(pn)
            if abs(fn) < abs(f)
                s = [gt - g, wt - w]; g, w = gt, wt; p, f = pn, fn; λr = (p[1] + p[2]) / 2
                accepted = true; break
            end
            s ./= 2
        end
        (!accepted || norm(s) < 1e-14) && break
    end
    return (g = g, ωb = w, λ = λr, λ1 = p[1], λ2 = p[2], gap = abs(p[1] - p[2]))
end

"""
    confirm_ep(g0, ωb0, vars, γ; λ0=nothing, ε=0.0, q=nothing, window=nothing, iters=40,
               ep_gap=5e-7, ep_svratio=1e4, kwargs...)

**Step 3 of the EP search: the 2-step protocol.** `locate_ep` drives the pair's gap → 0 at **fixed
physical parameters**, then the Feshbach reduction (`ep_feshbach` core) of *that* pair classifies:
  (1) does the gap reach ~0 (`< ep_gap`)? If not, it **floors** ⇒ avoided crossing, no EP at these
      rates (it may coalesce only as κ → 0, as a DP);
  (2) only if it does, the kernel dimension decides: `svratio > ep_svratio` ⇒ **EP2** (defective,
      `t` stays pinned); `svratio < dp_svratio` ⇒ **DP** (semisimple, `t → 0`; e.g. a crossing
      between the two parity blocks at `ε = 0`); in between ⇒ **ambiguous** (typically a weak EP,
      small `t`, not localized deeply enough: rerun with more `iters`).
This removes the false positives of a single grid point, where `svratio` is large also at a floored
gap. `λ0` seeds the pair; if `nothing`, the closest pair in `window` at
`(g0, ωb0)` is used. Returns `(g, ωb, gap, t, svratio, λ, verdict)`.

NB: L preserves Hermiticity, so its spectrum is symmetric under conjugation. An EP where a real pair
turns into a complex-conjugate pair is then **codimension 1**: it forms a curve in `(g, ωb)`, and
the Newton lands on one point of that curve (which point depends on the seed).
"""
function confirm_ep(g0::Real, ωb0::Real, vars::NamedTuple, γ::Union{Real,AbstractVector{<:Real}};
                    λ0 = nothing, ε::Real = 0.0, q = nothing, window = nothing, iters::Int = 40,
                    h::Real = 1e-6, box = nothing, ep_gap::Real = 5e-7, ep_svratio::Real = 1e4,
                    dp_svratio::Real = 1e2, kwargs...)
    Lm(g, w) = liouvillian_matrix(g, _vars(vars, w), γ; ε = ε, q = q, kwargs...)
    λ0 === nothing && (λ0 = ep_feshbach(Lm(g0, ωb0); window = window).λ)
    loc = locate_ep(g0, ωb0, vars, γ; λ0 = λ0, ε = ε, q = q, iters = iters, h = h, box = box, kwargs...)
    M = Lm(loc.g, loc.ωb); F = schur(M)
    r = _feshbach_pair(M, F, _nearest_pair(F.values, loc.λ1, loc.λ2)...)
    verdict = r.gap ≥ ep_gap         ? "no EP (gap floors at $(round(r.gap, sigdigits = 2)))" :
              r.svratio > ep_svratio ? "EP2" :
              r.svratio < dp_svratio ? "DP (coalesces, dim ker = 2)" :
                                       "ambiguous (coalesces, svratio $(round(r.svratio, sigdigits = 2)): refine)"
    return (g = loc.g, ωb = loc.ωb, gap = r.gap, t = r.t, svratio = r.svratio, λ = r.λ, verdict = verdict)
end

"""
    gap_scaling(g, ωb, vars, γ; λ, s, dir=(1,1)./√2, ε=0.0, q=nothing, kwargs...)

Gap of the pair nearest `λ` along the straight cut `(g, ωb) + s·dir` away from a located
coalescence, for every distance in `s`. Returns `(s, gap, p)` with `p` the log–log slope:
`p → 0.5` at an EP2 (√ branch point), `p → 1` at a DP (linear crossing). Reliable only with a deep
localization (gap ≲ 1e-8, `locate_ep`) and `s` inside the √-region, whose width shrinks with the
Schur off-diagonal `t` (s ≲ 1e-5 for t ~ 1e-3; s ≲ 1e-8 for a weak EP with t ~ 1e-6). Farther
away the gap turns linear and the fitted slope drifts to 1.
"""
function gap_scaling(g::Real, ωb::Real, vars::NamedTuple, γ; λ::Number, s,
                     dir = (1, 1) ./ sqrt(2), ε::Real = 0.0, q = nothing, kwargs...)
    λr  = ComplexF64(λ)
    gap = map(s) do x
        p = _pair_near(eigvals(liouvillian_matrix(g + x * dir[1], _vars(vars, ωb + x * dir[2]), γ;
                                                  ε = ε, q = q, kwargs...)), λr)
        abs(p[1] - p[2])
    end
    return (s = collect(s), gap = gap, p = _loglog_slope(s, gap))
end

_loglog_slope(x, y) = (xs = log.(x); ys = log.(y); xm = sum(xs) / length(xs); ym = sum(ys) / length(ys);
                       sum((xs .- xm) .* (ys .- ym)) / sum(abs2, xs .- xm))

# ════════════════════════════════════════════════════════════════════════════════════════════
# Grid-independent search of REAL-AXIS EPs (a real pair ↔ a complex-conjugate pair).
# L is Hermiticity-preserving ⇒ real in a Hermitian basis ⇒ "λ is real" is an exact test, and the
# number of real eigenvalues in a Re-window changes by 2 exactly at a real-axis EP.
# ════════════════════════════════════════════════════════════════════════════════════════════

# δ² = ((λ₁-λ₂)/2)² of the pair closest to the real point x: -Im² for a conjugate pair, +Δ²/4 for
# a real pair. Analytic across the EP (where it changes sign), unlike the gap.
# The pair is looked for within |Re λ - x| < r: if a conjugate pair with Im < im_cap is there, the
# tracked mode is still complex; otherwise it is the CLOSEST real pair (other, unrelated real
# eigenvalues — e.g. population modes at -nκ — are much farther apart than the pair near an EP).
function _delta2_near(vals, x::Real; r::Real = 5e-3, im_cap::Real = 0.05)
    cz = [λ for λ in vals if 0 < imag(λ) < im_cap && abs(real(λ) - x) < r]
    if !isempty(cz)
        z = cz[argmin(imag.(cz))]
        return -imag(z)^2, real(z)
    end
    rz = sort([real(λ) for λ in vals if imag(λ) == 0 && abs(real(λ) - x) < r])
    length(rz) < 2 && return -Inf, float(x)
    d = diff(rz); k = argmin(d)
    return (d[k] / 2)^2, (rz[k] + rz[k+1]) / 2
end

# At a converged count bracket (spectra Sa, Sb): is the change a coalescence on the real axis
# (→ λ*), or a real eigenvalue crossing the edge of the Re-window (→ nothing)?
function _edge_event(Sa, Sb, inw, match_tol)
    na = count(λ -> imag(λ) == 0 && inw(λ), Sa); nb = count(λ -> imag(λ) == 0 && inw(λ), Sb)
    R, C = na > nb ? (Sa, Sb) : (Sb, Sa)                      # R = side with the extra real pair
    cpx = filter(λ -> imag(λ) > 0 && inw(λ), C)
    isempty(cpx) && return nothing
    z  = cpx[argmin(imag.(cpx))]
    rs = sort(filter(λ -> imag(λ) == 0, R); by = λ -> abs(λ - real(z)))
    length(rs) < 2 && return nothing
    ok = imag(z) < match_tol && abs(rs[1] - rs[2]) < 2match_tol && abs(real(z) - real(rs[1] + rs[2]) / 2) < match_tol
    return ok ? real(z) : nothing
end

# ════════════════════════════════════════════════════════════════════════════════════════════
# PARAMETER-SPACE EP SEARCH (N scanned parameters).
#  · parameters: one NamedTuple — a Number = fixed, a range/vector = scanned;
#  · real-axis EPs (codim 1): exact count of real eigenvalues on ALL grid edges + V-minima of Im λ
#    on all grid lines (bisection / golden section);
#  · off-axis EPs (codim 2): winding number of the window discriminant D_W on every 2-D face cell,
#    quadtree refinement, 2-knob Newton;
#  · checks: approach along the normal to the EP curve (real axis) or along both face axes
#    (off axis), always on the whole matrix.
# The search only needs a map `point ↦ real matrix`, so the same code runs on the QRM Liouvillian
# (`ep_grid(p; q)`) and on any test model (`ep_grid(matfun, p)`).
# ════════════════════════════════════════════════════════════════════════════════════════════

const _QRM_DEFAULTS = (ωa = 1.0, ωb = 1.0, γa = 0.0, γb = 0.0, ε = 0.0, Ta = 0.0, Tb = 0.0,
                       N_trunc = nothing, σ_filter = nothing)

"""
    scanned_keys(p)

Names of the scanned parameters of `p` (the entries given as a range or vector), in order.
"""
scanned_keys(p::NamedTuple) = Tuple(k for k in keys(p) if p[k] isa AbstractVector)

# the point of `p` with the scanned keys `ks` set to the values `v`
_at(p::NamedTuple, ks::Tuple, v) = merge(p, NamedTuple{ks}(Tuple(v)))

"""
    liouvillian_at(pt; q = nothing, real_basis = false)

Dense generalized Liouvillian of the QRM at ONE parameter point `pt` (all entries numbers).
Recognized entries (defaults in brackets): `g`, `Nc` (required), `ωa` [1], `ωb` [1], `γa` [0],
`γb` [0], `ε` [0], `Ta` [0], `Tb` [0], `N_trunc` [none], `σ_filter` [QuantumToolbox default].
`q = ±1` returns a parity block (only at `ε = 0`); `real_basis = true` the real representation
in the Hermitian-matrix basis (see `liouvillian_matrix`).
"""
function liouvillian_at(pt::NamedTuple; q = nothing, real_basis::Bool = false)
    pt = merge(_QRM_DEFAULTS, pt)
    (haskey(pt, :g) && haskey(pt, :Nc)) || throw(ArgumentError("the parameters must contain at least g and Nc"))
    any(v -> v isa AbstractVector, values(pt)) &&
        throw(ArgumentError("liouvillian_at needs a single point: all parameters must be numbers"))
    return liouvillian_matrix(pt.g, (ωa = pt.ωa, ωb = pt.ωb, Nc = pt.Nc), [pt.γa, pt.γb];
                              ε = pt.ε, T = [pt.Ta, pt.Tb], q = q, real_basis = real_basis,
                              N_trunc = pt.N_trunc, σ_filter = pt.σ_filter)
end

"""
    ep_grid(p; q = nothing)
    ep_grid(matfun, p)

Spectra on the grid of the scanned parameters of `p` (multithreaded). The first form uses the QRM
Liouvillian in its real (Hermitian-basis) representation, full or parity block `q`; the second any
map `matfun(point::NamedTuple) → real matrix` (e.g. a test model). Returns a NamedTuple `G` with
`p`, `keys` (scanned names), `axes` (their values), `spec` (Array of spectra, one per node) and
`mat` (the matrix map), used by `find_real_eps`, `find_complex_eps`, `check_eps`.
"""
function ep_grid(mat::Function, p::NamedTuple; progress::Bool = true)
    ks = scanned_keys(p)
    isempty(ks) && throw(ArgumentError("no scanned parameter: pass at least one range"))
    ax   = [collect(Float64, p[k]) for k in ks]
    dims = Tuple(length.(ax))
    G0   = (p = p, keys = ks, axes = ax, mat = mat)
    spec = Array{Vector{ComplexF64}}(undef, dims)
    nodes = vec(collect(CartesianIndices(dims)))
    prog  = _progress(length(nodes), "[1/5] spectra (nodes)", progress)
    _blas1() do
        Threads.@threads for I in nodes
            spec[I] = _spec_at(G0, _node(G0, I))
            next!(prog)
        end
    end
    return merge(G0, (spec = spec,))
end
ep_grid(p::NamedTuple; q = nothing, progress::Bool = true) =
    ep_grid(pt -> liouvillian_at(pt; q = q, real_basis = true), merge(_QRM_DEFAULTS, p); progress = progress)

# progress bar of one phase (ProgressMeter's next! is thread-safe); disabled if progress = false or n = 0
_progress(n, desc, on) = Progress(max(n, 1); desc = rpad(desc, 34), enabled = on && n > 0, showspeed = true)

# spectrum at the scanned-parameter vector v (complex vector even if all eigenvalues are real)
_spec_at(G, v) = ComplexF64.(eigvals(G.mat(_at(G.p, G.keys, v))))
_node(G, I) = [G.axes[d][I[d]] for d in eachindex(G.keys)]
_unit(D, d) = CartesianIndex(ntuple(j -> j == d ? 1 : 0, D))

# ── real-axis EPs on ONE line (1-D core) ─────────────────────────────────────────────────────

# Follow the eigenvalue z0 (Im ≥ 0) from x0 to x by continuity: steps are halved while the match
# is ambiguous (the 2nd closest eigenvalue less than 3× farther than the closest), except when the
# two closest are both real (that is the coalescing pair itself, inside a real interval).
# Returns δ² of its pair at x (-Im² if complex; +(Δ/2)² with the nearest real partner if real),
# the followed eigenvalue at x and the spectrum at x.
function _track_delta2(spec, x0, z0, x, hmin)
    xc, zc = x0, ComplexF64(z0)
    Sx = nothing
    h  = x - x0
    while true
        xn = abs(x - xc) ≤ abs(h) ? x : xc + h
        T  = spec(xn)
        up = filter(λ -> imag(λ) ≥ 0, T)
        d  = abs.(up .- zc); i = sortperm(d)
        amb = length(i) ≥ 2 && d[i[2]] < 3d[i[1]] && !(imag(up[i[1]]) == 0 && imag(up[i[2]]) == 0)
        if amb && abs(xn - xc) > hmin
            h /= 2
            continue
        end
        xc, zc, Sx = xn, up[i[1]], T
        xc == x && break
        h *= 2
    end
    imag(zc) > 0 && return -imag(zc)^2, zc, Sx
    re = [λ for λ in Sx if imag(λ) == 0 && λ != zc]
    isempty(re) && return -Inf, zc, Sx
    r2 = re[argmin(abs.(re .- zc))]
    return real((zc - r2) / 2)^2, zc, Sx
end

# xs: node coordinates on the line, S: spectra at the nodes, spec(x): spectrum anywhere on it.
# Returns EPs (x, λ) and near misses (x, λ, immin); `touch_tol` rejects "slivers" whose half-width
# √δ²_max is at round-off level (a conjugate pair TOUCHING the axis = DP, split only by noise).
function _line_real_eps(xs::Vector{Float64}, S::Vector{Vector{ComplexF64}}, spec::Function, inw::Function;
                        im_cap::Real = 0.05, gtol::Real = 1e-13, match_tol::Real = 1e-4, touch_tol::Real = 1e-10)
    n = length(xs)
    nreal(T) = count(λ -> imag(λ) == 0 && inw(λ), T)
    imin(T)  = (x = [imag(λ) for λ in T if inw(λ) && 0 < imag(λ) < im_cap]; isempty(x) ? Inf : minimum(x))
    NR = nreal.(S); IM = imin.(S)
    eps = NamedTuple[]; nm = NamedTuple[]
    function bisect!(a, b, Sa, Sb)                          # (a) exact count change ⇒ bisection
        if b - a < gtol * max(1.0, abs(a))
            λs = _edge_event(Sa, Sb, inw, match_tol)
            λs === nothing || push!(eps, (x = (a + b) / 2, λ = λs))
            return
        end
        m = (a + b) / 2; Sm = spec(m)
        nreal(Sm) != nreal(Sa) && bisect!(a, m, Sa, Sm)
        nreal(Sm) != nreal(Sb) && bisect!(m, b, Sm, Sb)
    end
    for k in 1:n-1
        NR[k] != NR[k+1] && bisect!(xs[k], xs[k+1], S[k], S[k+1])
    end
    # (b) V-minima of Im λ, eigenvalue by eigenvalue. Continuation to the neighbouring nodes among
    # COMPLEX eigenvalues; if the match is ambiguous (another eigenvalue almost as close) the
    # candidate is KEPT (examined below rather than risk discarding it).
    function nb_im(T, z)
        u = filter(λ -> imag(λ) > 0, T); isempty(u) && return 0.0
        d = abs.(u .- z); i = sortperm(d)
        (length(i) ≥ 2 && d[i[2]] < 3d[i[1]]) && return Inf
        return imag(u[i[1]])
    end
    cands = Tuple{Int,ComplexF64}[]
    for k in 1:n, z in S[k]
        (inw(z) && 0 < imag(z) < im_cap) || continue
        lo = k > 1 ? nb_im(S[k-1], z) : Inf
        hi = k < n ? nb_im(S[k+1], z) : Inf
        (imag(z) ≤ lo && imag(z) ≤ hi) || continue
        any(c -> c[1] == k && abs(c[2] - z) < 1e-9, cands) || push!(cands, (k, z))
    end
    hmin = 1e-13 * max(1.0, abs(xs[end] - xs[1]))
    for (k, z0) in cands
        kl, kr = max(k - 1, 1), min(k + 1, n)
        NR[kl] == NR[kr] || continue
        # δ² of the candidate's OWN pair, followed by continuity from the closest point already
        # visited (golden-section points are reached by adaptive steps, never by a jump)
        memo = [(xs[k], z0)]
        function f(x)
            j = argmin([abs(m[1] - x) for m in memo])
            d2, zx, _ = _track_delta2(spec, memo[j][1], memo[j][2], x, hmin)
            push!(memo, (x, zx))
            return d2
        end
        a, b = xs[kl], xs[kr]; φ = (sqrt(5) - 1) / 2
        c, d = b - φ * (b - a), a + φ * (b - a); fc, fd = f(c), f(d)
        while b - a > 1e-14 * max(1.0, abs(a))               # golden section: maximize δ²
            if fc > fd
                b, d, fd = d, c, fc; c = b - φ * (b - a); fc = f(c)
            else
                a, c, fc = c, d, fd; d = a + φ * (b - a); fd = f(d)
            end
        end
        xm = (a + b) / 2
        j  = argmin([abs(m[1] - xm) for m in memo])
        d2, zm, Sm = _track_delta2(spec, memo[j][1], memo[j][2], xm, hmin)
        interior = xs[kl] + 1e-9 < xm < xs[kr] - 1e-9
        if d2 > touch_tol^2                                  # a real sliver ⇒ two EPs
            bisect!(xs[kl], xm, S[kl], Sm); bisect!(xm, xs[kr], Sm, S[kr])
        elseif interior                                      # near miss, or touching within round-off
            push!(nm, (x = xm, λ = real(zm), immin = sqrt(max(-d2, 0.0)), touching = d2 > -touch_tol^2))
        end
    end
    unique!(e -> (round(e.x, digits = 11), round(e.λ, digits = 6)), eps)
    unique!(e -> (round(e.x, digits = 7), round(e.λ, digits = 4)), nm)
    return (eps = eps, nearmiss = nm, nreal = NR, imin = IM)
end

"""
    find_real_eps(G; re_window = (-0.45, -1e-3), im_cap = 0.05, gtol = 1e-13, match_tol = 1e-4,
                  touch_tol = 1e-10)

Real-axis EPs (a real pair ↔ a complex-conjugate pair; codimension 1) on the grid `G` of
`ep_grid`. The 1-D search (exact count change + bisection; V-minima of Im λ + golden section on
δ²) runs on EVERY grid line in EVERY scanned direction, so an EP curve (surface) is intercepted
whatever its orientation. Returns `(eps, nearmiss)`; each EP is `(point, λ, along, kind = :real)`,
`point` = NamedTuple of the scanned parameters, `along` = the direction of the line on which it
was found. A near miss has `immin` (closest approach to the axis) and `touching` (reached the
axis within round-off: a DP, not an EP).
"""
function find_real_eps(G; re_window = (-0.45, -1e-3), im_cap::Real = 0.05, gtol::Real = 1e-13,
                       match_tol::Real = 1e-4, touch_tol::Real = 1e-10, progress::Bool = true)
    inw(λ) = re_window[1] ≤ real(λ) ≤ re_window[2]
    D = length(G.keys); dims = size(G.spec)
    lines = [(d, J) for d in 1:D for J in CartesianIndices(dims) if J[d] == 1 && dims[d] > 1]
    res = Vector{Any}(undef, length(lines))
    prog = _progress(length(lines), "[2/5] real axis (grid lines)", progress)
    _blas1() do
        Threads.@threads for li in eachindex(lines)
            d, J = lines[li]
            base  = _node(G, J)
            nodes = [J + (i - 1) * _unit(D, d) for i in 1:dims[d]]
            spec  = x -> (v = copy(base); v[d] = x; _spec_at(G, v))
            r = _line_real_eps(G.axes[d], [G.spec[I] for I in nodes], spec, inw;
                               im_cap = im_cap, gtol = gtol, match_tol = match_tol, touch_tol = touch_tol)
            res[li] = (r = r, d = d, base = base)
            next!(prog)
        end
    end
    eps = NamedTuple[]; nm = NamedTuple[]
    pt = v -> NamedTuple{G.keys}(Tuple(v))
    for x in res
        for e in x.r.eps
            v = copy(x.base); v[x.d] = e.x
            push!(eps, (point = pt(v), λ = e.λ, along = G.keys[x.d], kind = :real))
        end
        for e in x.r.nearmiss
            v = copy(x.base); v[x.d] = e.x
            push!(nm, (point = pt(v), λ = e.λ, immin = e.immin, touching = e.touching, along = G.keys[x.d]))
        end
    end
    return (eps = eps, nearmiss = nm)
end

# ── off-axis EPs: winding number of the window discriminant ─────────────────────────────────

_wrap(x) = mod(x + π, 2π) - π

# W = the open upper half-plane Im λ > 0. The matrix is real, so an eigenvalue is either exactly
# real or one of an exact conjugate pair: membership in W is exact, and the set in W can change only
# through a coalescence ON the real axis (the codim-1 objects of `find_real_eps`).

# phase of D_W = ∏_{i<j ∈ W} (λᵢ-λⱼ)² (label-free: symmetric in the eigenvalues), eigenvalues in W
function _discr_info(vals; touch::Real = 1e-9)
    w = [λ for λ in vals if imag(λ) > 0]
    # a conjugate pair TOUCHING the real axis (a DP) may come out of the eigensolver as two real
    # eigenvalues split by round-off: keep it as one member of W (at the touching point), so that
    # W changes only through genuine real intervals (split ≫ round-off)
    re = sort([real(λ) for λ in vals if imag(λ) == 0])
    k = 1
    while k < length(re)
        if re[k+1] - re[k] < touch * max(1.0, abs(re[k]))
            push!(w, complex((re[k] + re[k+1]) / 2, 0.0)); k += 2
        else
            k += 1
        end
    end
    θ = 0.0
    for a in 1:length(w)-1, b in a+1:length(w)
        θ += 2angle(w[a] - w[b])
    end
    return (θ = mod2pi(θ), n = length(w), w = w)
end

# Hausdorff distance between two finite sets of complex numbers (label-free displacement)
function _hausdorff(A, B)
    (isempty(A) || isempty(B)) && return isempty(A) && isempty(B) ? 0.0 : Inf
    h1 = maximum(a -> minimum(b -> abs(a - b), B), A)
    h2 = maximum(b -> minimum(a -> abs(a - b), A), B)
    return max(h1, h2)
end

# Rigorous bound on the phase change of D_W between two samples A → B: every eigenvalue moves by
# ≲ r = Hausdorff(A, B), so every difference δᵢⱼ stays in a disc of radius 2r around its value at A;
# if the disc excludes 0, arg δᵢⱼ turns by < arcsin(2r/|δᵢⱼ|). Summing 2·arcsin over the pairs bounds
# |Δ arg D_W|; if the bound is < bmax (< π), wrap() of the sampled difference is exact.
function _phase_bound_ok(dA, dB, bmax)
    r = _hausdorff(dA.w, dB.w); B = 0.0
    for a in 1:length(dA.w)-1, b in a+1:length(dA.w)
        B += 2asin(min(1.0, 2r / abs(dA.w[a] - dA.w[b])))
        B ≥ bmax && return false
    end
    return true
end

# phase increment of D_W on [ta, tb] of an edge (f(t) = spectrum). The midpoint is always sampled;
# each half is accepted only if the rigorous bound holds, otherwise it is bisected (up to maxdepth).
# Invalid (reported, never used) if the number of eigenvalues in W changes (a real-axis event) or if
# the phase cannot be resolved (a zero lying on the edge).
function _seg_phase(f, ta, tb, da, db, depth, o)
    tm = (ta + tb) / 2; dm = _discr_info(f(tm))
    (da.n == dm.n == db.n) || return (Δ = NaN, ok = false, why = :realaxis)
    ok1 = _phase_bound_ok(da, dm, o.bmax); ok2 = _phase_bound_ok(dm, db, o.bmax)
    (ok1 && ok2) && return (Δ = _wrap(dm.θ - da.θ) + _wrap(db.θ - dm.θ), ok = true, why = :ok)
    depth ≥ o.maxdepth && return (Δ = NaN, ok = false, why = :unresolved)
    r1 = ok1 ? (Δ = _wrap(dm.θ - da.θ), ok = true, why = :ok) : _seg_phase(f, ta, tm, da, dm, depth + 1, o)
    r1.ok || return r1
    r2 = ok2 ? (Δ = _wrap(db.θ - dm.θ), ok = true, why = :ok) : _seg_phase(f, tm, tb, dm, db, depth + 1, o)
    r2.ok || return r2
    return (Δ = r1.Δ + r2.Δ, ok = true, why = :ok)
end

# total phase increment along a whole edge, starting from o.n0 equal sub-segments (anti-aliasing)
function _edge_phase(f, d0, d1, o)
    ts = range(0.0, 1.0, o.n0 + 1)
    ds = vcat([d0], [_discr_info(f(t)) for t in ts[2:end-1]], [d1])
    Δ = 0.0
    for k in 1:o.n0
        r = _seg_phase(f, ts[k], ts[k+1], ds[k], ds[k+1], 0, o)
        r.ok || return r
        Δ += r.Δ
    end
    return (Δ = Δ, ok = true, why = :ok)
end

# winding number of D_W around the rectangle [ua,ub]×[va,vb] of the (u, v) plane (counter-clockwise)
function _rect_winding(spec2, ua, ub, va, vb, o)
    cs  = ((ua, va), (ub, va), (ub, vb), (ua, vb))
    inf = [_discr_info(spec2(c...)) for c in cs]
    tot = 0.0
    for k in 1:4
        pa, pb = cs[k], cs[mod1(k + 1, 4)]
        f = t -> spec2(pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2]))
        r = _edge_phase(f, inf[k], inf[mod1(k + 1, 4)], o)
        r.ok || return (w = NaN, ok = false, why = r.why)
        tot += r.Δ
    end
    w = tot / 2π
    ok = abs(w - round(w)) < 0.05
    return (w = w, ok = ok, why = ok ? :ok : :unresolved)
end

# winding numbers of the 4 sub-rectangles of [ua,ub]×[va,vb]. Split at the midpoint; if a sub-rectangle
# is unresolved (typically a zero lying on a split line), retry with the split point shifted (golden
# ratios), so that a zero is never left on an internal boundary by accident.
function _split4(spec2, ua, ub, va, vb, o)
    subs = nothing
    for θ in (0.5, 0.4381966011250105, 0.5618033988749895)
        um = ua + θ * (ub - ua); vm = va + θ * (vb - va)
        rects = ((ua, um, va, vm), (um, ub, va, vm), (ua, um, vm, vb), (um, ub, vm, vb))
        subs  = [(r, _rect_winding(spec2, r..., o)) for r in rects]
        any(s -> !s[2].ok && s[2].why == :unresolved, subs) || return subs
    end
    return subs
end

# damped 2-knob Newton on δ² = ((λ₁-λ₂)/2)² of the pair nearest λ0 (complex δ² ⇒ 2 real equations)
function _newton2(spec2, u0, v0, λ0, box; iters = 40, h = 1e-8, maxstep = Inf)
    pair(u, v, λr) = _pair_near(spec2(u, v), λr)
    δ2(p) = ((p[1] - p[2]) / 2)^2
    inbox(u, v) = (clamp(u, box[1]...), clamp(v, box[2]...))
    u, v = inbox(u0, v0); λr = ComplexF64(λ0)
    p = pair(u, v, λr); f = δ2(p); λr = (p[1] + p[2]) / 2
    for _ in 1:iters
        fu = δ2(pair(u + h, v, λr)); fv = δ2(pair(u, v + h, λr))
        J  = [real(fu - f) real(fv - f); imag(fu - f) imag(fv - f)] ./ h
        r  = [real(f); imag(f)]
        s  = -((J' * J + 1e-12 * (sum(abs2, J) + eps()) * I) \ (J' * r))
        nr = norm(s); nr > maxstep && (s .*= maxstep / nr)
        accepted = false
        for _ in 1:30
            ut, vt = inbox(u + s[1], v + s[2])
            (ut, vt) == (u, v) && break
            pn = pair(ut, vt, λr); fn = δ2(pn)
            if abs(fn) < abs(f)
                s = [ut - u, vt - v]; u, v = ut, vt; p, f = pn, fn; λr = (p[1] + p[2]) / 2
                accepted = true; break
            end
            s ./= 2
        end
        (!accepted || norm(s) < 1e-15) && break
    end
    return (u = u, v = v, λ = λr, gap = abs(p[1] - p[2]))
end

# Quadtree refinement of one face cell of the grid:
#  · a cell (or sub-cell) with nonzero winding is split until each zero is isolated, then located by
#    Newton: |index| = 1 stops at level `refine`; |index| ≥ 2 keeps splitting up to `refine_max`
#    (two close weak EPs separate; a true DP keeps index 2 down to the last level);
#  · an UNRELIABLE cell (crossed by a real-axis event) is split too, up to `unreliable_depth` levels:
#    the sub-cells not touched by the real-axis structure become valid and are searched like any
#    other, so the blind region shrinks from one grid cell to cell/2^unreliable_depth.
# Returns zeros (kind :complex), the remaining small unreliable rectangles (:unreliable) and the
# sub-cells whose phase could not be resolved (:unresolved).
function _refine_cell(G, c, o; refine = 4, refine_max = 12, iters = 40, ep_gap = 5e-7, nseed = 5,
                      unreliable_depth = 4)
    base = _node(G, c.I)
    u0, u1 = base[c.d1], G.axes[c.d1][c.I[c.d1] + 1]
    v0, v1 = base[c.d2], G.axes[c.d2][c.I[c.d2] + 1]
    pt(u, v)    = (x = copy(base); x[c.d1] = u; x[c.d2] = v; x)
    spec2(u, v) = _spec_at(G, pt(u, v))
    face = (G.keys[c.d1], G.keys[c.d2])
    out  = NamedTuple[]
    blank(kind, r, level, why) = (point = NamedTuple{G.keys}(Tuple(pt((r[1] + r[2]) / 2, (r[3] + r[4]) / 2))), λ = NaN + 0im,
                                  index = 0, gap = NaN, located = false, kind = kind, in_window = false,
                                  face = face, cell = r, level = level, why = why)
    # one level of splitting, dispatching every sub-rectangle
    function descend!(ua, ub, va, vb, level)
        for (r, s) in _split4(spec2, ua, ub, va, vb, o)
            if s.ok
                round(Int, s.w) != 0 && rect!(r..., level + 1, round(Int, s.w))
            elseif s.why == :realaxis
                unrel!(r..., level + 1)
            else
                push!(out, blank(:unresolved, r, level + 1, s.why))
            end
        end
    end
    function unrel!(ua, ub, va, vb, level)
        level ≥ unreliable_depth && return push!(out, blank(:unreliable, (ua, ub, va, vb), level, :realaxis))
        descend!(ua, ub, va, vb, level)
    end
    function rect!(ua, ub, va, vb, level, n)
        if level ≥ refine_max || (abs(n) == 1 && level ≥ refine)
            uc, vc = (ua + ub) / 2, (va + vb) / 2
            w  = [λ for λ in spec2(uc, vc) if imag(λ) > 0]
            length(w) < 2 && return push!(out, blank(:unresolved, (ua, ub, va, vb), level, :few_eigenvalues))
            # Newton from the `nseed` closest pairs at the centre; accept the first that converges
            # INSIDE this sub-cell (the zero counted by the winding). Otherwise keep refining: the
            # sub-cell may still hold several zeros whose indices sum to ±1.
            du, dv = ub - ua, vb - va
            box  = ((ua - du, ub + du), (va - dv, vb + dv))
            best = nothing
            for (_, i, j) in _sorted_pairs(w, collect(eachindex(w)))[1:min(nseed, end)]
                loc = _newton2(spec2, uc, vc, (w[i] + w[j]) / 2, box; iters = iters, h = 1e-6 * min(du, dv), maxstep = min(du, dv))
                inside = ua - 1e-9du ≤ loc.u ≤ ub + 1e-9du && va - 1e-9dv ≤ loc.v ≤ vb + 1e-9dv
                (best === nothing || (inside && loc.gap < best[1].gap)) && (best = (loc, inside))
                inside && loc.gap < ep_gap && break
            end
            loc, inside = best
            (!(inside && loc.gap < ep_gap) && level < refine_max) && return descend!(ua, ub, va, vb, level)
            push!(out, (point = NamedTuple{G.keys}(Tuple(pt(loc.u, loc.v))), λ = loc.λ, index = n,
                        gap = loc.gap, located = inside && loc.gap < ep_gap, kind = :complex,
                        in_window = o.re_window[1] ≤ real(loc.λ) ≤ o.re_window[2],
                        face = face, cell = (ua, ub, va, vb), level = level, why = :ok))
            return
        end
        descend!(ua, ub, va, vb, level)
    end
    if c.ok
        rect!(u0, u1, v0, v1, 0, round(Int, c.w))
    elseif c.why == :realaxis
        unrel!(u0, u1, v0, v1, 0)
    else                                                              # a grid edge could not be resolved:
        descend!(u0, u1, v0, v1, 0)                                   # the split lines avoid the bad spot
    end
    return out
end

"""
    find_complex_eps(G; re_window = (-0.45, -1e-3), n0 = 4, bmax = π/2, maxdepth = 14,
                     refine = 4, refine_max = 12, iters = 40, ep_gap = 5e-7)

Off-axis coalescences (Im λ* > 0; codimension 2) on the grid `G` of `ep_grid` (needs ≥ 2 scanned
parameters). On every 2-D face cell of the grid it computes the winding number of the discriminant
D_W = ∏_{i<j}(λᵢ-λⱼ)² of ALL eigenvalues in the upper half-plane W around the cell boundary
(Brouwer degree: sum of the indices of the zeros inside; EP2 → ±1, conical DP → ±2). D_W is
symmetric in the eigenvalues, so no eigenvalue is tracked. Membership in W is exact (real matrix),
so W can only change through a real-axis coalescence.

The boundary is sampled adaptively with a RIGOROUS anti-aliasing criterion: a sub-segment is
accepted only if the eigenvalues move (Hausdorff distance between the sets) so little compared with
their mutual distances that |Δ arg D_W| < `bmax` < π (see `_phase_bound_ok`); otherwise it is
bisected, up to `maxdepth`. A conjugate pair touching the real axis (split into two reals only by
round-off) counts as one member of W. A cell is **unreliable** if the number of eigenvalues in W
changes along its boundary (a real interval, i.e. a real-axis EP curve, crosses the cell): it is
split `unreliable_depth` times, the sub-cells away from the real-axis structure are searched
normally, and only the small sub-cells still crossed by it are reported (never silently used).
Cells with nonzero winding are refined (quadtree, see `_refine_cell`; a split producing an
unresolved sub-cell is retried with shifted split lines) and each zero is located by a 2-knob
Newton in the face parameters. `re_window` is only a post-filter: each zero carries `in_window`.

Limitation (degree theory): zeros of opposite index in the same cell cancel (winding 0) ⇒ refine
the grid to separate them. Only the upper half-plane is searched (conjugate twins are implied).

Returns `(zeros, unreliable, unresolved, cells, n_unreliable_grid)`: each zero is `(point, λ, index,
gap, located, kind, in_window, face, cell, level)`; `unreliable` / `unresolved` are the remaining
sub-rectangles (`cell = (u_min, u_max, v_min, v_max)` in the face parameters).
"""
function find_complex_eps(G; re_window = (-0.45, -1e-3), n0::Int = 4, bmax::Real = π / 2,
                          maxdepth::Int = 14, refine::Int = 4, refine_max::Int = 12,
                          iters::Int = 40, ep_gap::Real = 5e-7, unreliable_depth::Int = 4,
                          progress::Bool = true)
    D = length(G.keys); dims = size(G.spec)
    D ≥ 2 || throw(ArgumentError("the winding number needs at least 2 scanned parameters"))
    o = (re_window = re_window, n0 = n0, bmax = bmax, maxdepth = maxdepth)
    info  = map(_discr_info, G.spec)
    edges = [(d, I) for d in 1:D for I in CartesianIndices(dims) if I[d] < dims[d]]
    evals = Vector{Any}(undef, length(edges))
    prog  = _progress(length(edges), "[3/5] off axis (edge phases)", progress)
    _blas1() do
        Threads.@threads for k in eachindex(edges)
            d, I = edges[k]; J = I + _unit(D, d)
            a, b = _node(G, I), _node(G, J)
            evals[k] = _edge_phase(t -> _spec_at(G, a .+ t .* (b .- a)), info[I], info[J], o)
            next!(prog)
        end
    end
    E = Dict(edges[k] => evals[k] for k in eachindex(edges))
    cells = NamedTuple[]
    for d1 in 1:D-1, d2 in d1+1:D, I in CartesianIndices(dims)
        (I[d1] < dims[d1] && I[d2] < dims[d2]) || continue
        es = (E[(d1, I)], E[(d2, I + _unit(D, d1))], E[(d1, I + _unit(D, d2))], E[(d2, I)])
        eok = all(e -> e.ok, es)
        w   = eok ? (es[1].Δ + es[2].Δ - es[3].Δ - es[4].Δ) / 2π : NaN
        ok  = eok && abs(w - round(w)) < 0.05
        why = ok ? :ok : eok ? :unresolved : first(e.why for e in es if !e.ok)
        push!(cells, (d1 = d1, d2 = d2, I = I, w = w, ok = ok, why = why))
    end
    # cells to refine: nonzero winding (zeros inside) and unreliable ones (split to shrink the blind area)
    todo  = filter(c -> !c.ok || round(Int, c.w) != 0, cells)
    found = Vector{Any}(undef, length(todo))
    prog  = _progress(length(todo), "[4/5] off axis (cell refinement)", progress)
    _blas1() do
        Threads.@threads for k in eachindex(todo)
            found[k] = _refine_cell(G, todo[k], o; refine = refine, refine_max = refine_max, iters = iters,
                                    ep_gap = ep_gap, unreliable_depth = unreliable_depth)
            next!(prog)
        end
    end
    res = reduce(vcat, found; init = NamedTuple[])
    return (zeros      = filter(z -> z.kind == :complex, res),
            unreliable = filter(z -> z.kind == :unreliable, res),
            unresolved = filter(z -> z.kind == :unresolved, res),
            cells = cells, n_unreliable_grid = count(c -> !c.ok, cells))
end

# ── dynamic checks on the whole matrix ─────────────────────────────────────────────────────

# gap, Schur off-diagonal t, svratio, |Im λ| of the pair nearest λ at x0 + σ·dir, for σ ∈ s
function _approach_dir(G, x0, dir, λ, s, s_fit)
    rs = map(s) do σ
        M = ComplexF64.(G.mat(_at(G.p, G.keys, x0 .+ σ .* dir)))
        F = schur(M); i = sortperm(abs.(F.values .- λ))
        r = _feshbach_pair(M, F, i[1], i[2])
        (gap = r.gap, t = r.t, svratio = r.svratio, im = abs(imag(F.values[i[1]])))
    end
    gap = [r.gap for r in rs]; t = [r.t for r in rs]; im = [r.im for r in rs]
    # stop the fit well before the pair changes type (real ↔ complex): a second EP is there
    isreal_(k) = im[k] < 1e-3 * gap[k]
    kc = findfirst(k -> isreal_(k) != isreal_(1), eachindex(s))
    smax = kc === nothing ? s_fit : min(s_fit, s[kc] / 10)
    fit = s .≤ smax; count(fit) < 4 && (fit = (1:length(s)) .≤ 4)
    return (dir = dir, s = s, gap = gap, t = t, svratio = [r.svratio for r in rs], im = im,
            p_gap = _loglog_slope(s[fit], gap[fit]), p_t = _loglog_slope(s[fit], t[fit]))
end

"""
    check_ep(G, ep; s = 10.0 .^ range(-10, -5, 21), s_fit = 1e-7, hgrad = 1e-7)

Dynamic test of one EP (from `find_real_eps` or `find_complex_eps`) on the whole matrix of `G`:
move away from it and follow the pair nearest λ* (gap, Schur off-diagonal `t` from the ordered Schur
form of the full matrix, svratio).
  · real-axis EP: along ± the NORMAL to the EP curve/surface, i.e. the gradient of δ² w.r.t. the
    scanned parameters (finite differences, step `hgrad`·axis span);
  · off-axis EP: along ± both axes of the face where it was found.
EP2 ⇔ on every side gap ∝ s^½ and `t` pinned (slope ≈ 0); DP ⇔ gap ∝ s and t ∝ s. For an off-axis
zero the winding `index` must also be ±1 (EP2) or ±2 (DP). Returns `(sides, verdict)`.
"""
function check_ep(G, ep; s = 10.0 .^ range(-10, -5, 21), s_fit::Real = 1e-7, hgrad::Real = 1e-7)
    D  = length(G.keys)
    x0 = [ep.point[k] for k in G.keys]
    sv = sort(collect(Float64, s))
    if ep.kind == :real
        span = [max(ax[end] - ax[1], eps()) for ax in G.axes]
        # δ² of the pair nearest λ* (at the EP it is the coalescing pair; δ² real: -Im² or +(Δ/2)²)
        δ2(x) = (p = _pair_near(_spec_at(G, x), ep.λ); real(((p[1] - p[2]) / 2)^2))
        grad = map(1:D) do k
            h = hgrad * span[k]; e = zeros(D); e[k] = h
            (δ2(x0 .+ e) - δ2(x0 .- e)) / 2h
        end
        n = norm(grad) > 0 ? grad ./ norm(grad) : Float64.(collect(G.keys) .== ep.along)
        dirs = [n, -n]
    else
        dirs = Vector{Float64}[]
        for k in ep.face
            e = Float64.(collect(G.keys) .== k); push!(dirs, e, -e)
        end
    end
    sides = [_approach_dir(G, x0, d, ep.λ, sv, s_fit) for d in dirs]
    isep(r) = abs(r.p_gap - 0.5) < 0.1 && abs(r.p_t) < 0.1
    isdp(r) = abs(r.p_gap - 1.0) < 0.15 && abs(r.p_t - 1.0) < 0.2
    beh = all(isep, sides) ? "EP2" : all(isdp, sides) ? "DP" : "inconclusive"
    verdict = ep.kind == :real ? beh :
              (beh == "EP2" && abs(ep.index) == 1) ? "EP2" :
              (beh == "DP"  && abs(ep.index) == 2) ? "DP"  : "inconclusive ($(beh), index $(ep.index))"
    return (sides = sides, verdict = verdict)
end

"""
    check_eps(G, eps; kwargs...)

`check_ep` on every entry of `eps` (multithreaded; entries with `kind = :unresolved` are skipped).
Returns each EP merged with `verdict`, `t` (at the smallest distance, first side), `p_gap`, `p_t`
(tuples over the sides) and the full `check` result.
"""
function check_eps(G, eps; progress::Bool = true, kwargs...)
    todo = filter(e -> e.kind != :unresolved, eps)
    out  = Vector{NamedTuple}(undef, length(todo))
    prog = _progress(length(todo), "[5/5] checks (approach test)", progress)
    _blas1() do
        Threads.@threads for i in eachindex(todo)
            c = check_ep(G, todo[i]; kwargs...)
            out[i] = merge(todo[i], (verdict = c.verdict, t = c.sides[1].t[1],
                                     p_gap = Tuple(r.p_gap for r in c.sides), p_t = Tuple(r.p_t for r in c.sides), check = c))
            next!(prog)
        end
    end
    return out
end

# ── driver: one call, compact results (no spectra kept) ─────────────────────────────────────

# per-node scalars for the maps: # real eigenvalues in the Re-window, smallest Im > 0 below im_cap,
# smallest distance between eigenvalues (Im ≥ 0) in the Re-window
function _node_maps(G, re_window, im_cap)
    inw(λ) = re_window[1] ≤ real(λ) ≤ re_window[2]
    nreal  = map(S -> count(λ -> imag(λ) == 0 && inw(λ), S), G.spec)
    imin   = map(S -> (x = [imag(λ) for λ in S if inw(λ) && 0 < imag(λ) < im_cap]; isempty(x) ? Inf : minimum(x)), G.spec)
    gapmin = map(G.spec) do S
        w = [λ for λ in S if inw(λ) && imag(λ) ≥ 0]
        length(w) < 2 ? Inf : minimum(abs(w[a] - w[b]) for a in 1:length(w)-1 for b in a+1:length(w))
    end
    return (nreal = nreal, imin = imin, gapmin = gapmin)
end

# dressed content (QRM only) of the Liouvillian eigenmode nearest λ at the point pt: main |i⟩⟨j|
# elements (i = 1 ground state) with weights, and E_i - E_j of the leading one
function _composition_at(pt::NamedTuple, λ; top::Int = 4)
    pt = merge(_QRM_DEFAULTS, pt)
    E, _, L = gen_liouvillian_QRM(pt.g, (ωa = pt.ωa, ωb = pt.ωb, Nc = pt.Nc), [pt.γa, pt.γb];
                                  ε = pt.ε, T = [pt.Ta, pt.Tb], N_trunc = pt.N_trunc, σ_filter = pt.σ_filter)
    F = eigen(Matrix(L.data)); k = argmin(abs.(F.values .- λ))
    w = abs2.(normalize(F.vectors[:, k])); N = isqrt(length(w))
    o = sortperm(w; rev = true)[1:top]
    content = [((mod1(i, N), (i - 1) ÷ N + 1), w[i]) for i in o]
    (i, j) = content[1][1]; Ev = real.(E)
    return (content = content, ΔE = Ev[i] - Ev[j])
end

# compact record of one (checked) coalescence
function _ep_record(G, P, e; spectra::Bool, composition::Bool)
    x  = [e.point[k] for k in G.keys]
    pt = _at(P, G.keys, x)                                   # ALL parameters, as numbers
    S  = _spec_at(G, x)
    λ1, λ2 = _pair_near(S, e.λ)
    rec = (point = pt, kind = e.kind, λ = e.λ, λ1 = λ1, λ2 = λ2, gap = abs(λ1 - λ2),
           index = get(e, :index, missing), along = get(e, :along, missing), face = get(e, :face, missing),
           verdict = get(e, :verdict, "unchecked"), t = get(e, :t, NaN),
           p_gap = get(e, :p_gap, ()), p_t = get(e, :p_t, ()))
    if haskey(e, :check)
        rec = merge(rec, (approach = [(dir = r.dir, s = r.s, gap = r.gap, t = r.t, svratio = r.svratio, im = r.im)
                                      for r in e.check.sides],))
    end
    composition && (rec = merge(rec, _composition_at(pt, e.λ)))
    spectra     && (rec = merge(rec, (spectrum = S,)))
    return rec
end

function _ep_search(G, P, q, t0; re_window = (-0.45, -1e-3), im_cap::Real = 0.05, complex_search::Bool = true,
                    check::Bool = true, s = 10.0 .^ range(-10, -5, 21), spectra::Bool = false,
                    composition::Bool = true, real_kw = (;), complex_kw = (;), progress::Bool = true)
    t1 = time()
    RE = find_real_eps(G; re_window = re_window, im_cap = im_cap, progress = progress, real_kw...)
    t2 = time()
    CE = (complex_search && length(G.keys) ≥ 2) ?
         find_complex_eps(G; re_window = re_window, progress = progress, complex_kw...) :
         (zeros = NamedTuple[], unreliable = NamedTuple[], unresolved = NamedTuple[], cells = NamedTuple[],
          n_unreliable_grid = 0)
    t3 = time()
    zs    = filter(z -> z.kind == :complex && z.in_window, CE.zeros)
    found = vcat(RE.eps, zs)
    chk   = check ? check_eps(G, found; s = s, progress = progress) : found
    t4 = time()
    recs = Vector{NamedTuple}(undef, length(chk))
    _blas1() do
        Threads.@threads for i in eachindex(chk)
            recs[i] = _ep_record(G, P, chk[i]; spectra = spectra, composition = composition)
        end
    end
    t5 = time()
    return (eps          = filter(r -> r.verdict == "EP2", recs),
            dps          = filter(r -> r.verdict == "DP", recs),
            inconclusive = filter(r -> r.verdict ∉ ("EP2", "DP", "unchecked"), recs),
            unchecked    = filter(r -> r.verdict == "unchecked", recs),
            nearmiss     = RE.nearmiss,
            unreliable   = [(point = z.point, face = z.face, cell = z.cell, level = z.level) for z in CE.unreliable],
            unresolved   = [(point = z.point, face = z.face, cell = z.cell, level = z.level, why = z.why) for z in CE.unresolved],
            n_outside    = count(z -> z.kind == :complex && !z.in_window, CE.zeros),
            maps         = _node_maps(G, re_window, im_cap),
            meta         = (p = P, keys = G.keys, axes = G.axes, q = q, re_window = re_window, im_cap = im_cap,
                            n_cells = length(CE.cells), n_unreliable_grid = CE.n_unreliable_grid,
                            nthreads = Threads.nthreads(),
                            timing = (spectra = t1 - t0, real = t2 - t1, complex = t3 - t2, checks = t4 - t3, records = t5 - t4)))
end

"""
    ep_search(p; q = nothing, re_window = (-0.45, -1e-3), im_cap = 0.05, complex_search = true,
              check = true, s = 10.0 .^ range(-10, -5, 21), spectra = false, composition = true,
              real_kw = (;), complex_kw = (;), progress = true)
    ep_search(matfun, p; ...)

`progress = true` (default) shows one progress bar per phase: [1/5] spectra, [2/5] real axis,
[3/5] off-axis edge phases, [4/5] off-axis cell refinement, [5/5] checks.

Complete EP search in one call. `p` = NamedTuple of parameters: a Number is FIXED, a range/vector is
SCANNED. The first form uses the QRM Liouvillian (full, or parity block `q = ±1` at ε = 0); the
second any map `matfun(point) → real matrix` (test models; no dressed composition).

Pipeline: `ep_grid` (spectra at the nodes, kept only during the search) → `find_real_eps` (real-axis
EPs, all grid lines) → `find_complex_eps` (off-axis EPs by winding number, ≥ 2 scanned parameters)
→ `check_eps` (approach test on the whole matrix) → compact records.

Returns a NamedTuple with NO spectra (unless `spectra = true`, then only at the found points):
- `eps`, `dps`, `inconclusive` (and `unchecked` if `check = false`): one record per coalescence:
  `point` (ALL parameters), `kind` (:real / :complex), `λ`, `λ1`, `λ2`, `gap`, `index` (winding,
  off-axis only), `along` / `face`, `verdict`, `t`, `p_gap`, `p_t` (per side), `approach` (gap and t
  vs distance, per side), and for the QRM `content` (main |i⟩⟨j| of the mode) and `ΔE` (E_i - E_j);
- `nearmiss` (real axis, with `touching` = DP within round-off), `unreliable` rectangles (the small
  sub-cells still crossed by a real-axis EP curve after the splitting: centre, face, `cell` bounds,
  level), `unresolved` sub-cells, `n_outside` (off-axis zeros outside `re_window`);
- `maps` (per-node scalars for plots: `nreal`, `imin`, `gapmin`), `meta` (parameters, axes, settings,
  timings). Everything is plain data: save it with JLD2 (`jldsave(file; R)`).
"""
function ep_search(p::NamedTuple; q = nothing, progress::Bool = true, kwargs...)
    t0 = time(); P = merge(_QRM_DEFAULTS, p)
    G  = ep_grid(P; q = q, progress = progress)
    return _ep_search(G, P, q, t0; progress = progress, kwargs...)
end
function ep_search(mat::Function, p::NamedTuple; progress::Bool = true, kwargs...)
    t0 = time()
    G  = ep_grid(mat, p; progress = progress)
    return _ep_search(G, p, nothing, t0; composition = false, progress = progress, kwargs...)
end

end # module