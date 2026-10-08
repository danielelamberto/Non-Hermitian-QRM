# Following one eigenpair (λ, r) of L continuously along a path in parameter space, by predictor–corrector continuation.
# Method and validation: notes/monodromy_tracking.typ.
#   starting eigenpairs     eigenpairs_near, pair_near (shift-invert)
#   bordered system         the Jacobian B of the eigenpair equations, and its linear solves (StaleLU)
#   tracking                tangent (predictor), correct! (corrector), track (adaptive ODE integration along the path)

"""
    Eigenpair

An eigenvalue with its right eigenvector, `(λ, r)`.
"""
const Eigenpair = Tuple{ComplexF64,Vector{ComplexF64}}

## Starting eigenpairs

"""
    eigenpairs_near(L, targets; δσ=1e-3)

The eigenpairs of the sparse `L` nearest to each target eigenvalue, one shift-invert each. The shift is offset by `δσ`
so that L - σ is not singular when a target is an exact eigenvalue. Warns if two targets converge to the same
eigenvalue (near an EP, use `pair_near`).
"""
function eigenpairs_near(L::AbstractMatrix, targets; δσ=1e-3)
    pairs = map(targets) do λt
        res = eigsolve(L; sigma=λt + δσ, eigvals=1)
        (res.values[1], res.vectors[:, 1])
    end
    allunique(round.(first.(pairs), digits=10)) || @warn "eigenpairs_near: two targets converged to the same eigenvalue"
    return pairs
end

"""
    pair_near(L, σ; δσ=1e-6)

The two eigenpairs of `L` nearest `σ` (≈ the midpoint of a close pair), from a single shift-invert. Near an EP the pair
is close, and separate searches from approximate targets can converge to the same eigenvalue.
"""
function pair_near(L::AbstractMatrix, σ; δσ=1e-6)
    res = eigsolve(L; sigma=σ + δσ, eigvals=2)
    return [(res.values[i], res.vectors[:, i]) for i ∈ 1:2]
end

## Bordered system
# An eigenpair is fixed by (L - λ) r = 0 and the normalisation c'r = 1, with c fixed. Its Jacobian in (r, λ) is
#     B = [L - λI  -r; c'  0],
# non-singular while λ is simple and c'r ≠ 0, singular at an EP (cond(B) ∝ distance^(-1/2) near an EP2).

"""
    normalisation(r)

The normalisation vector c = r/(r'r), for which c'r = 1.
"""
normalisation(r) = r/(r ⋅ r)

"""
    bordered(L, λ, r, c)

The bordered matrix B = [L - λI  -r; c'  0], assembled (allocating; the tracker fills a `BorderedPattern` in place).
"""
function bordered(L::SparseMatrixCSC, λ, r, c)
    T = promote_type(eltype(L), typeof(λ), eltype(r), eltype(c))
    return [L - λ*I         sparse(reshape(-r, :, 1))
            sparse(reshape(conj(c), 1, :))   spzeros(T, 1, 1)]
end

"""
    bordered_mul!(y, L, λ, r, c, x)

y = B x, without assembling B.
"""
function bordered_mul!(y, L::SparseMatrixCSC, λ, r, c, x)
    n = size(L, 1)
    xr, yr = view(x, 1:n), view(y, 1:n)
    mul!(yr, L, xr)
    yr .-= λ .* xr .+ x[n + 1] .* r
    y[n + 1] = c ⋅ xr
    return y
end

"""
    BorderedPattern(L, r, c)

Preallocated bordered matrix B, filled in place by `fill_bordered!`, with the positions in `nonzeros(B)` of L's entries,
of its diagonal, of -r (last column) and of c' (last row). Its pattern is L's (whose diagonal must be stored, as in
`AffineLiouvillian`), bordered on the support of (r, c) only, the indices where r or c is nonzero, with no corner
entry. The eigenvectors of a Liouvillian with a symmetry vanish exactly outside their sector, and stay so along a
path: storing those zeros (or the corner) adds LU fill and pivoting work (factorisations 1.4 times slower for the
625-state dimer). `fits(P, r, c)` tells whether r and c are still zero outside the support.
"""
struct BorderedPattern{T}
    B::SparseMatrixCSC{T,Int}
    Lpos::Vector{Int}           # positions of L's entries in nonzeros(B)
    diag::Vector{Int}           # positions of L's diagonal
    support::Vector{Int}        # indices where r or c is nonzero
    outside::Vector{Int}        # the other indices
    rpos::Vector{Int}           # positions of -r[support] (last column)
    cpos::Vector{Int}           # positions of conj(c[support]) (last row)
end
function BorderedPattern(L::SparseMatrixCSC, r, c)
    T = promote_type(eltype(L), ComplexF64)
    n = size(L, 1)
    inside = [!iszero(r[i]) || !iszero(c[i]) for i ∈ 1:n]
    colptr, rowval, Lpos, cpos = [1], Int[], Int[], Int[]
    for j ∈ 1:n
        for k ∈ nzrange(L, j)
            push!(rowval, rowvals(L)[k])
            push!(Lpos, length(rowval))
        end
        if inside[j]
            push!(rowval, n + 1)
            push!(cpos, length(rowval))
        end
        push!(colptr, length(rowval) + 1)
    end
    support = findall(inside)
    rpos = collect(length(rowval) .+ eachindex(support))
    append!(rowval, support)
    push!(colptr, length(rowval) + 1)
    B = SparseMatrixCSC(n + 1, n + 1, colptr, rowval, zeros(T, length(rowval)))
    diag = [Lpos[entry_index(L, j, j)] for j ∈ 1:n]
    return BorderedPattern(B, Lpos, diag, support, findall(!, inside), rpos, cpos)
end
fits(P::BorderedPattern, r, c) = all(i -> iszero(r[i]) && iszero(c[i]), P.outside)

"""
    fill_bordered!(P::BorderedPattern, L, λ, r, c)

Writes B = [L - λI  -r; c'  0] into `P.B` and returns it.
"""
function fill_bordered!(P::BorderedPattern, L, λ, r, c)
    v = nonzeros(P.B)
    @views v[P.Lpos] .= nonzeros(L)
    @views v[P.diag] .-= λ
    @views v[P.rpos] .= .-r[P.support]
    @views v[P.cpos] .= conj.(c[P.support])
    return P.B
end

"""
    StaleLU(; rtol=1e-12, maxiter=20, early=8, method=:gmres)

Solver for the bordered systems of one tracked eigenpair (not to be shared between threads). B depends on (u, λ, r) but
changes little along a path: B x = b is solved with the LU factorisation F of an earlier B ("stale"), refined by
x ← x + F \\ (b - B x) until ‖b - B x‖ ≤ rtol ‖b‖. It refactorises with the current B when the refinement does not
converge within `maxiter` iterations or stops decreasing, and after any solve that needed more than `early` iterations
(B has drifted: keep the next solves cheap). Counts `factorisations` and `refinements` (iterations).
`method=:gmres` (default): GMRES preconditioned by F instead of the refinement (a Richardson iteration on the same
splitting, `method=:richardson`), which still converges when ‖I - B F⁻¹‖ ≳ 1, as near an EP where B is ill-conditioned.

The matrix B (a `BorderedPattern`, rebuilt only if r or c leaves its support) and the work vectors are allocated once
and reused; refactorisations on the same pattern reuse the symbolic analysis of the LU (`lu!`).
"""
mutable struct StaleLU
    F::Any                    # factorisation of the reference B, nothing before the first solve
    P::Any                    # BorderedPattern, nothing before the first factorisation
    b::Vector{ComplexF64}     # work vectors of length n + 1: right-hand side (filled by the callers),
    x::Vector{ComplexF64}     # solution,
    res::Vector{ComplexF64}   # residual,
    dx::Vector{ComplexF64}    # correction
    rtol::Float64
    maxiter::Int
    early::Int
    method::Symbol            # :gmres (GMRES preconditioned by F) or :richardson (iterative refinement)
    V::Matrix{ComplexF64}     # GMRES workspace: Krylov basis (n + 1) × (maxiter + 1),
    H::Matrix{ComplexF64}     # Hessenberg matrix, rotated to triangular,
    g::Vector{ComplexF64}     # rotated right-hand side,
    cs::Vector{Float64}       # Givens rotations (cosines,
    sn::Vector{ComplexF64}    # sines)
    factorisations::Int
    refinements::Int
end
StaleLU(; rtol=1e-12, maxiter=20, early=8, method=:gmres) =
    StaleLU(nothing, nothing, ComplexF64[], ComplexF64[], ComplexF64[], ComplexF64[], rtol, maxiter, early, method,
            Matrix{ComplexF64}(undef, 0, 0), zeros(ComplexF64, maxiter + 1, maxiter), zeros(ComplexF64, maxiter + 1),
            zeros(maxiter), zeros(ComplexF64, maxiter), 0, 0)

"""
    rhs_buffer(S::StaleLU, n)

The right-hand-side work vector of `S`, for an eigenvector of length `n` (sizes the work vectors at the first call).
"""
function rhs_buffer(S::StaleLU, n)
    if length(S.b) != n + 1
        foreach(v -> resize!(v, n + 1), (S.b, S.x, S.res, S.dx))
        S.method == :gmres && (S.V = Matrix{ComplexF64}(undef, n + 1, S.maxiter + 1))
    end
    return S.b
end

"""
    refactorise!(S::StaleLU, L, λ, r, c)

Factorises the current B: in place on the same pattern, from a new `BorderedPattern` if r or c left it.
"""
function refactorise!(S::StaleLU, L, λ, r, c)
    if S.P === nothing || !fits(S.P, r, c)
        S.P = BorderedPattern(L, r, c)
        S.F = lu(fill_bordered!(S.P, L, λ, r, c))
    else
        S.F = lu!(S.F, fill_bordered!(S.P, L, λ, r, c))     # same pattern: reuses the symbolic analysis
    end
    S.factorisations += 1
    return S
end

"""
    solve_bordered!(S, L, λ, r, c, b)

Solves B x = b. With a `StaleLU`, by refinement from a stale factorisation; x is returned in the work vector `S.x`,
overwritten by the next solve. With `nothing`, by a fresh factorisation (allocating).
"""
function solve_bordered!(S::StaleLU, L, λ, r, c, b)
    rhs_buffer(S, length(r))
    fresh = S.F === nothing
    fresh && refactorise!(S, L, λ, r, c)
    ldiv!(S.x, S.F, b)
    fresh && return S.x
    S.method == :gmres && return gmres_bordered!(S, L, λ, r, c, b)
    nb, res_prev = norm(b), Inf
    for it ∈ 0:S.maxiter
        S.res .= b .- bordered_mul!(S.res, L, λ, r, c, S.x)
        nres = norm(S.res)
        if nres ≤ S.rtol*nb
            it > S.early && refactorise!(S, L, λ, r, c)
            return S.x
        end
        nres < res_prev || break
        res_prev = nres
        S.x .+= ldiv!(S.dx, S.F, S.res)
        S.refinements += 1
    end
    refactorise!(S, L, λ, r, c)
    return ldiv!(S.x, S.F, b)
end
solve_bordered!(::Nothing, L, λ, r, c, b) = bordered(L, λ, r, c) \ b

# GMRES on B x = b, right-preconditioned by the stale factorisation F (x = F⁻¹z), from x0 = F⁻¹b (in S.x): converges
# in a few iterations where iterative refinement (a Richardson iteration on the same splitting) diverges, i.e. when
# ‖I - B F⁻¹‖ ≳ 1 near an EP. Least squares by Givens rotations, updated at each iteration; workspace in S (no
# allocation). Refactorises if not converged within maxiter, or after more than `early` iterations.
function gmres_bordered!(S::StaleLU, L, λ, r, c, b)
    nb = norm(b)
    bordered_mul!(S.res, L, λ, r, c, S.x)
    S.res .= b .- S.res
    β = norm(S.res)
    β ≤ S.rtol*nb && return S.x
    V, H, g, cs, sn = S.V, S.H, S.g, S.cs, S.sn
    z, w = S.res, S.dx                                        # work vectors: F⁻¹ vⱼ and B F⁻¹ vⱼ
    @views V[:, 1] .= S.res ./ β
    fill!(g, 0)
    g[1] = β
    for j ∈ 1:S.maxiter
        @views ldiv!(z, S.F, V[:, j])
        bordered_mul!(w, L, λ, r, c, z)
        for i ∈ 1:j                                           # modified Gram–Schmidt
            @views H[i, j] = V[:, i] ⋅ w
            @views w .-= H[i, j] .* V[:, i]
        end
        hn = norm(w)
        @views V[:, j + 1] .= w ./ hn
        for i ∈ 1:j - 1                                       # previous rotations on the new column
            hi = H[i, j]
            H[i, j] = cs[i]*hi + sn[i]*H[i + 1, j]
            H[i + 1, j] = -conj(sn[i])*hi + cs[i]*H[i + 1, j]
        end
        ρ = hypot(abs(H[j, j]), hn)                           # new rotation, zeroing hn below H[j, j]
        cs[j] = abs(H[j, j])/ρ
        sn[j] = (iszero(H[j, j]) ? one(ComplexF64) : H[j, j]/abs(H[j, j]))*hn/ρ     # [c s; -s̄ c][H[j, j]; hn] = [·; 0]
        H[j, j] = cs[j]*H[j, j] + sn[j]*hn
        g[j + 1] = -conj(sn[j])*g[j]
        g[j] = cs[j]*g[j]
        S.refinements += 1
        if abs(g[j + 1]) ≤ S.rtol*nb
            y = UpperTriangular(view(H, 1:j, 1:j)) \ view(g, 1:j)
            @views mul!(z, V[:, 1:j], y)                      # x = x0 + F⁻¹ V y
            ldiv!(w, S.F, z)
            S.x .+= w
            j > S.early && refactorise!(S, L, λ, r, c)
            return S.x
        end
    end
    refactorise!(S, L, λ, r, c)
    return ldiv!(S.x, S.F, b)
end

## Predictor–corrector tracking along a path

"""
    tangent(L, dL, λ, r, c; solver=nothing)

Predictor: the derivatives (dr/du, dλ/du) of the eigenpair (λ, r) of L along the path, from
B [dr; dλ] = [-(dL/du) r; 0]. With a `StaleLU`, dr is a view of its work vector (valid until the next solve).
"""
function tangent(L::SparseMatrixCSC, dL::SparseMatrixCSC, λ, r, c; solver=nothing)
    n = length(r)
    b = solver === nothing ? similar(r, promote_type(eltype(r), ComplexF64), n + 1) : rhs_buffer(solver, n)
    mul!(view(b, 1:n), dL, r, -1, 0)
    b[n + 1] = 0
    x = solve_bordered!(solver, L, λ, r, c, b)
    return view(x, 1:n), x[n + 1]
end

"""
    correct!(r, L, λ, c; tol=1e-12, maxiter=5, solver=nothing)

Corrector: Newton iterations at fixed u back onto an eigenpair of L, B [δr; δλ] = [-(L - λ) r; 1 - c'r], updating `r`
in place. Returns (λ, residual ‖(L - λ)r‖/‖r‖ before the correction, number of iterations).
"""
function correct!(r, L::SparseMatrixCSC, λ, c; tol=1e-12, maxiter=5, solver=nothing)
    n = length(r)
    b = solver === nothing ? similar(r, promote_type(eltype(r), ComplexF64), n + 1) : rhs_buffer(solver, n)
    Lr = view(b, 1:n)
    mul!(Lr, L, r)
    Lr .-= λ .* r
    res0 = norm(Lr)/norm(r)
    res, it = res0, 0
    while (res > tol || abs(c ⋅ r - 1) > tol) && it < maxiter
        Lr .*= -1                                       # b = [λr - Lr; 1 - c'r]
        b[n + 1] = 1 - c ⋅ r
        x = solve_bordered!(solver, L, λ, r, c, b)
        r .+= view(x, 1:n)
        λ += x[n + 1]
        mul!(Lr, L, r)
        Lr .-= λ .* r
        res, it = norm(Lr)/norm(r), it + 1
    end
    return λ, res0, it
end

"""
    track(A, p::ParamPath, λ0, r0; alg=Tsit5(), reltol=1e-6, abstol=1e-8, corrector=true, tol=1e-12, reset_cos=0.1,
          stale=true, gmres=true, tspan=(0.0, 1.0), kwargs...)

Follows the eigenpair (λ0, r0) of `A` (an `AffineLiouvillian`, or any family with its interface: see affine.jl)
along the path `p`, by integrating d[r; λ]/du = `tangent` over `tspan` (default: the whole path) with the path's
breakpoints as tstops.
Returns `(sol, diagnostics)`: λ(u) = sol(u)[end], r(u) = sol(u)[1:end-1].
- `alg`, `reltol`, `abstol`: the ODE solver. The corrector puts every step back on an exact eigenpair, so the
  integrator only has to stay in Newton's basin and on the right branch: Tsit5 at 1e-6 (2.5–4× faster than Vern7 at
  1e-10, the former default, on loops, lines and scans; scans became wrong at 1e-4). The values at the steps
  (sol.t, sol.u) are exact eigenpairs (to `tol`: the corrector overwrites the saved step); the dense output sol(u)
  between them is accurate to ~reltol, and less near an EP. For an accurate dense output,
  `alg=Vern7(lazy=false), reltol=1e-10, abstol=1e-12`.
- `corrector`: Newton correction (`correct!`) after every accepted step, and reset of c to `normalisation(r)` when
  cos∠(c, r) = 1/(‖c‖‖r‖) < `reset_cos` (c'r = 1 is kept, r unchanged; ‖r‖ would grow otherwise).
- `stale`: bordered systems solved with a `StaleLU` (false: a fresh factorisation at every solve); `gmres`: by GMRES
  preconditioned by the stale factorisation (false: iterative refinement, which diverges near an EP and refactorises
  much more often: 50–250 factorisations per tracking instead of ~2; GMRES is 2–4× faster on 200–1600 states).
- `diagnostics`: corrections, Newton iterations, largest residual before a correction, c resets, factorisations,
  refinements (GMRES or refinement iterations).

L(u) and dL/du are written in place into two `work_matrix(A)` owned by this call (they share only A's read-only index
arrays), so concurrent calls on the same `A` are independent. c is the ODE parameter, reset in place by the callback:
the solver must not interpolate lazily (Vern7 needs `lazy=false`: its lazy stages would be computed when sol(u) is
called, with the latest c; Tsit5's interpolant uses only the stages of the step).
"""
function track(A, p::ParamPath, λ0, r0; alg=Tsit5(), reltol=1e-6, abstol=1e-8, corrector=true, tol=1e-12,
               reset_cos=0.1, stale=true, gmres=true, tspan=(0.0, 1.0), kwargs...)
    n = length(r0)
    c = normalisation(r0)
    solver = stale ? StaleLU(method=gmres ? :gmres : :richardson) : nothing
    L, dL = evaluate!(work_matrix(A), A, position(p, tspan[1])), work_matrix(A)     # dL: filled by rhs!
    function rhs!(dy, y, c, u)
        evaluate!(L, A, position(p, u))
        derivative!(dL, A, position(p, u), velocity(p, u))
        dr, dλ = tangent(L, dL, y[n + 1], view(y, 1:n), c; solver)
        view(dy, 1:n) .= dr
        dy[n + 1] = dλ
        return nothing
    end

    diagnostics = (corrections=Ref(0), iterations=Ref(0), max_residual=Ref(0.), resets=Ref(0))
    function correct_step!(integ)
        c = integ.p
        r = view(integ.u, 1:n)
        λ, res0, it = correct!(r, evaluate!(L, A, position(p, integ.t)), integ.u[n + 1], c; tol, solver)
        integ.u[n + 1] = λ
        # the step was saved before this callback: store the corrected eigenpair there instead (sol.u exact)
        if !isempty(integ.sol.t) && integ.sol.t[end] == integ.t
            integ.sol.u[end] .= integ.u
        end
        if 1/(norm(c)*norm(r)) < reset_cos
            c .= normalisation(r)
            diagnostics.resets[] += 1
        end
        diagnostics.corrections[] += 1
        diagnostics.iterations[] += it
        diagnostics.max_residual[] = max(diagnostics.max_residual[], res0)
        derivative_discontinuity!(integ, true)               # the state was changed: restart the step from it
    end
    callback = corrector ? DiscreteCallback((y, u, integ) -> true, correct_step!; save_positions=(false, false)) : nothing

    prob = ODEProblem(rhs!, [r0; λ0], tspan, c)
    sol = solve(prob, alg; reltol, abstol, tstops=filter(u -> tspan[1] < u < tspan[2], breakpoints(p)), callback, kwargs...)
    lu_counts = stale ? (factorisations=solver.factorisations, refinements=solver.refinements) : (factorisations=-1, refinements=0)
    return sol, merge(map(getindex, diagnostics), lu_counts)
end

"""
    tracking_succeeded(sol)

Whether the integration of `track` reached the end of its path.
"""
tracking_succeeded(sol) = string(sol.retcode) == "Success"
