# Liouvillians smooth but not affine in the parameters, e.g. Redfield, built in the eigenbasis of H(P): the family is
# given by its builder P ↦ L(P) only, and provides the interface of affine.jl (A(P), work_matrix, evaluate!,
# derivative!) by rebuilding L at every evaluation and differentiating it by central differences. The tracking tools,
# `slice` and `Reparametrised` take it in place of an AffineLiouvillian, at the cost of one build per evaluation and
# two per derivative.

"""
    FDLiouvillian(L_at, prm_P0; h=1e-5, check=true, check_at=nothing, rtol=1e-6)

The family P ↦ L(P) = `L_at(P)`, a function returning a sparse matrix, smooth in P but not necessarily affine. L is
evaluated by calling `L_at`, its derivative at P along the velocity v by a central difference along v̂ = v/|v|:

    dL ≈ |v| (L(P + h v̂) - L(P - h v̂))/2h,

so that the step `h` is a distance in parameter space whatever the speed |v| of the path. Its error is the truncation
h²‖∂³L‖/6 plus the round-off ~ eps‖L‖/h: h ~ (eps‖L‖/‖∂³L‖)^(1/3), about 1e-5 when L varies on a scale 1 of the
coordinates. Use coordinates of comparable scales (`Reparametrised` otherwise), and h well below the scale on which L
varies fastest: for Redfield at temperature T, near a crossing of two levels coupled by the bath, L varies on a scale
~ T in the level spacing, so h ≪ T/|d(Eᵢ - Eⱼ)/dP|.

With `check`, h is validated at `prm_P0`, and at `check_at` (a point in the domain of interest) if given: along each
coordinate eₖ, the derivatives Dₖ with steps h and h/2 must agree, ‖Dₖ(h) - Dₖ(h/2)‖ ≤ `rtol` maxⱼ‖Dⱼ(h)‖ (relative
to the largest derivative there, so that a coordinate L barely depends on is not held to its own small scale). This
difference is 3/4 of the truncation error of Dₖ(h) when it dominates, and grows as 1/h when round-off does, so it
bounds the error on both sides; above `rtol`, an `ArgumentError` gives it with the ratio
‖Dₖ(2h) - Dₖ(h)‖/‖Dₖ(h) - Dₖ(h/2)‖: ≈ 4, truncation (h too large, or L not smooth on the scale h); ≲ 2, round-off
(h too small). The ratio itself is no criterion: near the optimal h both errors are comparable, and it is ~2. The
default `rtol` is well below the tracker's default reltol. 4n builds per point (6n on failure); like any check at
points, it does not see a sharper feature elsewhere, e.g. a crossing at small T.

`L_at` must return its matrices on one sparsity pattern, that of `L_at(prm_P0)`, with possibly fewer stored entries
(placed on it, an `ArgumentError` if an entry falls outside): `redfield_liouvillian` does, and so does its restriction
to a sector, `L[idx, idx]`. `L_at` must be thread-safe (the scans track edges in parallel).
"""
struct FDLiouvillian{F,T}
    L_at::F
    L0::SparseMatrixCSC{T,Int}        # L at the reference point: the pattern of the family
    h::Float64
end
function FDLiouvillian(L_at, prm_P0; h=1e-5, check=true, check_at=nothing, rtol=1e-6)
    A = FDLiouvillian(L_at, SparseMatrixCSC(L_at(prm_P0)), float(h))
    check || return A
    n = length(prm_P0)
    e(k) = [float(j == k) for j ∈ 1:n]
    with_step(s) = FDLiouvillian(L_at, A.L0, s)
    for prm_P ∈ (prm_P0, check_at)
        prm_P === nothing && continue
        D = [derivative(A, prm_P, e(k)) for k ∈ 1:n]
        scale = maximum(norm, D)
        for k ∈ 1:n
            δ = norm(D[k] - derivative(with_step(h/2), prm_P, e(k)))
            δ ≤ rtol*scale && continue                                    # a constant L passes (0 ≤ 0)
            ratio = norm(derivative(with_step(2h), prm_P, e(k)) - D[k])/δ
            throw(ArgumentError("step h = $h not validated at $prm_P along coordinate $k: " *
                                "‖D(h) - D(h/2)‖ = $(δ/scale) relative to the derivative (rtol = $rtol), " *
                                "ratio ‖D(2h) - D(h)‖/‖D(h) - D(h/2)‖ = $ratio " *
                                (ratio > 3 ? "(truncation: decrease h, or L is not smooth on the scale h)" :
                                             "(round-off: increase h)")))
        end
    end
    return A
end

# the values of the matrix M on the pattern of A, without a copy if M is stored on it
function values_on(A::FDLiouvillian, M::SparseMatrixCSC)
    S = A.L0
    (M.colptr == S.colptr && rowvals(M) == rowvals(S)) && return nonzeros(M)
    return nonzeros(on_pattern(M, S))
end

work_matrix(A::FDLiouvillian) = SparseMatrixCSC(size(A.L0)..., A.L0.colptr, A.L0.rowval, similar(nonzeros(A.L0)))

"""
    evaluate!(L, A::FDLiouvillian, prm_P)

Writes `A.L_at(prm_P)` into `L`, a matrix on the pattern of `A` (`work_matrix(A)`), and returns it.
"""
function evaluate!(L::SparseMatrixCSC, A::FDLiouvillian, prm_P)
    check_pattern(L, A)
    nonzeros(L) .= values_on(A, A.L_at(prm_P))
    return L
end

"""
    derivative!(dL, A::FDLiouvillian, prm_P, v)

Writes the derivative of L at `prm_P` along the velocity `v`, by a central difference of step `A.h` along v/|v|,
into `dL` (pattern of `A`), and returns it. Two builds of L.
"""
function derivative!(dL::SparseMatrixCSC, A::FDLiouvillian, prm_P, v)
    check_pattern(dL, A)
    s = norm(v)
    iszero(s) && (fill!(nonzeros(dL), 0); return dL)
    δ = (A.h/s)*v
    nonzeros(dL) .= (values_on(A, A.L_at(prm_P + δ)) .- values_on(A, A.L_at(prm_P - δ))) .* (s/(2A.h))
    return dL
end

(A::FDLiouvillian)(prm_P) = evaluate!(copy(A.L0), A, prm_P)
derivative(A::FDLiouvillian, prm_P, v) = derivative!(copy(A.L0), A, prm_P, v)
