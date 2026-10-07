# Liouvillians affine in the parameters, L(P) = L0 + Σₖ Pₖ Lₖ (sparse matrices), their restriction to planes (slice)
# and their reparametrisations (Reparametrised). The tracker uses a family A through four functions only:
#   A(P)                        L at the point P, as a new matrix
#   work_matrix(A)              a matrix to be filled in place, on the fixed sparsity pattern of the family
#   evaluate!(L, A, P)          L(P), in place
#   derivative!(dL, A, P, v)    the derivative of L at P along the velocity v, in place
# so along a path p, L(u) = L(position(p, u)) and dL/du = derivative at position(p, u) along velocity(p, u), with
# nothing rebuilt. Any other type providing them can replace AffineLiouvillian, e.g. `Reparametrised`, an affine family
# in new coordinates ξ, P = φ(ξ): not affine in ξ, its derivative at ξ along w is the affine one along Jφ(ξ) w.

"""
    AffineLiouvillian(L0, Ls)
    AffineLiouvillian(L_at, n; h=1e-3, check=true, check_at=nothing, rtol=1e-10)

The family L(P) = L0 + Σₖ Pₖ Ls[k] of sparse matrices. All terms are stored on one sparsity pattern, the union of
theirs and the diagonal (sharing its index arrays), so that `evaluate!` and `derivative!` only combine the stored
values.

The second form builds the decomposition from any function `prm_P ↦ L(prm_P)` of `n` coordinates that is affine:
L0 = L(0), Lₖ = (L(h eₖ) - L0)/h. For an affine L this is exact for any h, up to round-off: Lₖ carries an error
~ eps‖L0‖/h, and L(P) one ~ eps‖L0‖|P|/h. So take `h` as large as the model allows (it must stay physical at 0, h eₖ
and the check point, e.g. positive rates), and coordinates centred on the region of interest.

With `check`, L is also evaluated at a point not used to build the decomposition, h(1 + 1/n, 1 + 2/n, …, 2): all
coordinates nonzero (cross terms PᵢPⱼ show), distinct (no symmetry hides a term), and not h(1, …, 1), where a pure Pₖ²
term would be reproduced exactly. The deviation from the decomposition, relative to the change of L, is round-off
for an affine family and above `rtol` an `ArgumentError` is raised. This tests the curvature at the scale h only: a
term κ P² gives a deviation ~ κh there but an error ~ κ|P|² over a domain of size |P|. `check_at`, a point in the
domain of interest, adds a check at that scale.

`A(prm_P)` returns L(prm_P) and `derivative(A, v)` returns Σₖ vₖ Lₖ, as new, independent matrices.
"""
struct AffineLiouvillian{T}
    L0::SparseMatrixCSC{T,Int}
    Ls::Vector{SparseMatrixCSC{T,Int}}
    function AffineLiouvillian(L0::AbstractMatrix, Ls::AbstractVector)
        T = promote_type(eltype(L0), eltype.(Ls)...)
        terms = [SparseMatrixCSC{T,Int}(M) for M ∈ (L0, Ls...)]
        ones_on(M) = SparseMatrixCSC(size(M)..., M.colptr, M.rowval, ones(nnz(M)))
        pattern = reduce(+, ones_on.(terms); init=sparse(1.0I, size(L0)...))      # entries ≥ 1: no cancellation
        L0p, Lsp... = [on_pattern(M, pattern) for M ∈ terms]
        return new{T}(L0p, Lsp)
    end
end

function AffineLiouvillian(L_at::Function, n::Integer; h=1e-3, check=true, check_at=nothing, rtol=1e-10)
    L0 = L_at(zeros(n))
    e(k) = [j == k ? h : 0.0 for j ∈ 1:n]
    A = AffineLiouvillian(L0, typeof(L0)[(L_at(e(k)) - L0)/h for k ∈ 1:n])
    check || return A
    for prm_P ∈ (h .* (1 .+ (1:n) ./ n), check_at)
        prm_P === nothing && continue
        L1 = L_at(collect(Float64, prm_P))
        deviation, change = norm(L1 - A(prm_P)), norm(L1 - L0)
        deviation ≤ rtol*change ||                                                 # a constant L passes (0 ≤ 0)
            throw(ArgumentError("L_at is not affine in its $n coordinates " *
                                "(relative deviation $(deviation/change) at $prm_P)"))
    end
    return A
end

"""
    entry_index(S::SparseMatrixCSC, i, j)

Position of the entry (i, j) in `nonzeros(S)`; it must be stored.
"""
function entry_index(S::SparseMatrixCSC, i, j)
    rg = nzrange(S, j)
    k = searchsortedfirst(view(rowvals(S), rg), i)
    (k ≤ length(rg) && rowvals(S)[rg[k]] == i) || throw(ArgumentError("entry ($i, $j) is not stored"))
    return rg[k]
end

"""
    on_pattern(M, P)

`M` stored on the sparsity pattern of `P`, which must contain `M`'s. The result shares the index arrays of `P`.
"""
function on_pattern(M::SparseMatrixCSC{T}, P::SparseMatrixCSC) where T
    S = SparseMatrixCSC(size(P)..., P.colptr, P.rowval, zeros(T, nnz(P)))
    for j ∈ 1:size(M, 2), k ∈ nzrange(M, j)
        nonzeros(S)[entry_index(S, rowvals(M)[k], j)] = nonzeros(M)[k]
    end
    return S
end

"""
    work_matrix(A::AffineLiouvillian)

A matrix on the pattern of `A`, to be filled by `evaluate!` or `derivative!` (values uninitialised). It shares A's index
arrays, so that these recognise its pattern in O(1): its structure must not be changed (no entries added or removed),
which would corrupt A. Use `A(prm_P)` for an independent matrix.
"""
work_matrix(A::AffineLiouvillian) = SparseMatrixCSC(size(A.L0)..., A.L0.colptr, A.L0.rowval, similar(nonzeros(A.L0)))

# throws unless L is stored on the pattern of A (identical index arrays: O(1); otherwise compared)
function check_pattern(L::SparseMatrixCSC, A::AffineLiouvillian)
    S = A.L0
    (L.colptr === S.colptr && L.rowval === S.rowval) ||
        (size(L) == size(S) && L.colptr == S.colptr && rowvals(L) == rowvals(S)) ||
        throw(ArgumentError("the matrix is not stored on the sparsity pattern of the AffineLiouvillian"))
    return nothing
end

"""
    evaluate!(L, A::AffineLiouvillian, prm_P)

Writes L(prm_P) into `L`, a matrix on the pattern of `A` (`work_matrix(A)`, or a copy of `A.L0`), and returns it.
"""
function evaluate!(L::SparseMatrixCSC, A::AffineLiouvillian, prm_P)
    check_pattern(L, A)
    v = nonzeros(L)
    v .= nonzeros(A.L0)
    for k ∈ eachindex(A.Ls)
        v .+= prm_P[k] .* nonzeros(A.Ls[k])
    end
    return L
end

"""
    derivative!(dL, A::AffineLiouvillian, prm_P, v)
    derivative!(dL, A::AffineLiouvillian, v)

Writes the derivative of L along the velocity `v`, Σₖ vₖ Lₖ, into `dL` (pattern of `A`), and returns it. The point
`prm_P` is that of the family interface (see the head of affine.jl); an affine family's derivative does not depend
on it.
"""
function derivative!(dL::SparseMatrixCSC, A::AffineLiouvillian, v)
    check_pattern(dL, A)
    w = nonzeros(dL)
    fill!(w, 0)
    for k ∈ eachindex(A.Ls)
        w .+= v[k] .* nonzeros(A.Ls[k])
    end
    return dL
end
derivative!(dL::SparseMatrixCSC, A::AffineLiouvillian, prm_P, v) = derivative!(dL, A, v)

(A::AffineLiouvillian)(prm_P) = evaluate!(copy(A.L0), A, prm_P)
derivative(A::AffineLiouvillian, v) = derivative!(copy(A.L0), A, v)

"""
    slice(A::AffineLiouvillian, prm_P, e1, e2)

Restriction of `A` to the plane through `prm_P` spanned by `e1` and `e2`: the two-coordinate family
(s, t) ↦ L(prm_P + s e1 + t e2), itself affine and on the pattern of `A`. The 2D tools (`tracked_scan`, `ep_bisect`, …)
run on it unchanged, and a point (s, t) of the slice is prm_P + s e1 + t e2 in the full space. With
`(e1, e2) = plane_basis(n)`: the plane through prm_P normal to n, in orthonormal coordinates.
"""
slice(A::AffineLiouvillian, prm_P, e1, e2) = AffineLiouvillian(A(prm_P), [derivative(A, e1), derivative(A, e2)])

"""
    Reparametrised(A, φ; Jφ=nothing, check_at=nothing)

The family `A` (an `AffineLiouvillian`, or any family with its interface) in new coordinates ξ, with P = φ(ξ): the
family ξ ↦ L(φ(ξ)), e.g. logarithmic coordinates for rates over several decades, or polar coordinates around a point.
Not affine in ξ, but exact and as cheap: L at ξ is A at φ(ξ), and its derivative at ξ along w is A's along Jφ(ξ) w
(chain rule), on A's sparsity pattern. The tracking tools take it in place of `A`, working in ξ.
- Jφ, the Jacobian of φ, is computed by ForwardDiff, which runs φ on dual numbers: φ must not fix the element type of
  its result (`SVector(…)` or `[…]`, not `P2(…)`). Or give `Jφ(ξ)` yourself; `check_at`, a point ξ, then compares it
  with ForwardDiff there (`ArgumentError` if they differ).
- φ must be a diffeomorphism on the region searched (smooth, invertible), so that loops in ξ map to equivalent loops
  in P: around a singular point of φ (r = 0 in polar coordinates), a loop in ξ is not a loop in P.
- φ and Jφ must be thread-safe (pure functions): the scans track edges in parallel.
"""
struct Reparametrised{Fam,F,J}
    A::Fam
    φ::F
    Jφ::J
end
function Reparametrised(A, φ; Jφ=nothing, check_at=nothing)
    J_ad = ξ -> ForwardDiff.jacobian(φ, ξ)
    Jφ === nothing && return Reparametrised(A, φ, J_ad)
    if check_at !== nothing
        J, J_ref = Jφ(check_at), J_ad(check_at)
        norm(J - J_ref) ≤ 1e-8*(1 + norm(J_ref)) ||
            throw(ArgumentError("Jφ differs from the Jacobian of φ at $check_at (ForwardDiff: $J_ref, given: $J)"))
    end
    return Reparametrised(A, φ, Jφ)
end

(R::Reparametrised)(ξ) = R.A(R.φ(ξ))
work_matrix(R::Reparametrised) = work_matrix(R.A)
evaluate!(L::SparseMatrixCSC, R::Reparametrised, ξ) = evaluate!(L, R.A, R.φ(ξ))
derivative!(dL::SparseMatrixCSC, R::Reparametrised, ξ, w) = derivative!(dL, R.A, R.φ(ξ), R.Jφ(ξ)*w)
