# Liouvillians affine in the parameters, L(δ) = L0 + Σₖ δₖ Lₖ (sparse matrices). Along a path p, L(u) = A(position(p, u))
# and dL/du = derivative(A, velocity(p, u)), so nothing is rebuilt along the path. The tracker only uses evaluate! and
# derivative! (in place, on a fixed sparsity pattern): any other type providing them (e.g. a non-affine family with
# finite-difference derivatives) can replace AffineLiouvillian.

"""
    AffineLiouvillian(L0, Ls)
    AffineLiouvillian(L_at, n; h=1e-3, check=true, rtol=1e-10)

The family L(δ) = L0 + Σₖ δₖ Ls[k] of sparse matrices. All terms are stored on one sparsity pattern, the union of
theirs and the diagonal, so that `evaluate!` and `derivative!` only combine the stored values.

The second form builds the decomposition from any function `δ ↦ L(δ)` of `n` coordinates that is affine:
L0 = L(0), Lₖ = (L(h eₖ) - L0)/h, exact up to round-off. The small step `h` keeps the model physical (e.g. positive
rates) at the evaluation points. With `check`, L is also evaluated at δ = 2h(1, …, 1), a point not used to build the
decomposition (at h(1, …, 1) a pure δₖ² term would be reproduced exactly): the deviation from the decomposition,
relative to the change of L, is round-off for an affine family (~1e-14) and grows with h and the curvature otherwise;
above `rtol` an `ArgumentError` is raised.

`A(δ)` returns L(δ) and `derivative(A, v)` returns Σₖ vₖ Lₖ, as new matrices.
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
        return new{T}(L0p, collect(Lsp))
    end
end

function AffineLiouvillian(L_at::Function, n::Integer; h=1e-3, check=true, rtol=1e-10)
    L0 = L_at(zeros(n))
    e(k) = [j == k ? h : 0.0 for j ∈ 1:n]
    A = AffineLiouvillian(L0, typeof(L0)[(L_at(e(k)) - L0)/h for k ∈ 1:n])
    if check
        δ = fill(2h, n)
        L1 = L_at(δ)
        ratio = norm(L1 - A(δ))/norm(L1 - L0)
        ratio ≤ rtol || throw(ArgumentError("L_at is not affine in its $n coordinates (relative deviation $ratio)"))
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

`M` stored on the sparsity pattern of `P`, which must contain `M`'s.
"""
function on_pattern(M::SparseMatrixCSC{T}, P::SparseMatrixCSC) where T
    S = SparseMatrixCSC(size(P)..., copy(P.colptr), copy(P.rowval), zeros(T, nnz(P)))
    for j ∈ 1:size(M, 2), k ∈ nzrange(M, j)
        nonzeros(S)[entry_index(S, rowvals(M)[k], j)] = nonzeros(M)[k]
    end
    return S
end

"""
    evaluate!(L, A::AffineLiouvillian, δ)

Writes L(δ) into `L`, a matrix with the pattern of `A` (e.g. a copy of `A.L0`), and returns it.
"""
function evaluate!(L::SparseMatrixCSC, A::AffineLiouvillian, δ)
    v = nonzeros(L)
    v .= nonzeros(A.L0)
    for k ∈ eachindex(A.Ls)
        v .+= δ[k] .* nonzeros(A.Ls[k])
    end
    return L
end

"""
    derivative!(dL, A::AffineLiouvillian, v)

Writes the derivative of L along the velocity `v`, Σₖ vₖ Lₖ, into `dL` (pattern of `A`), and returns it.
"""
function derivative!(dL::SparseMatrixCSC, A::AffineLiouvillian, v)
    w = nonzeros(dL)
    fill!(w, 0)
    for k ∈ eachindex(A.Ls)
        w .+= v[k] .* nonzeros(A.Ls[k])
    end
    return dL
end

(A::AffineLiouvillian)(δ) = evaluate!(copy(A.L0), A, δ)
derivative(A::AffineLiouvillian, v) = derivative!(copy(A.L0), A, v)
