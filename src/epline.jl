# Following a line of EP2s through a three-parameter space. Complex EP2s have codimension 2: in 3D they form curves,
# the intersection of the surfaces Re D = 0 and Im D = 0, with D = (λa - λb)² the discriminant of the pair. D is
# smooth in the parameters (the eigenvalues have a √ branch point at the EP, D does not), vanishes linearly at an EP2
# and is symmetric in a, b: no branch labelling. Predictor–corrector continuation along the curve:
#   predictor   a step h along the tangent t = ∇Re D × ∇Im D (finite differences of D)
#   corrector   in the plane through the predicted point normal to t (`slice`): `ep_bisect` from a small square,
#               which brackets the EP (the square swaps the pair), then Newton on D from the final rectangle
# Real-axis EPs of a real Liouvillian have codimension 1 (surfaces in 3D): this tracker is for complex EP2s only.
# The tangent and the normal planes use the Euclidean metric of the coordinates: rescale them (`Reparametrised`) if
# the parameters have very different scales.

"""
    slice(A, prm_P, e1, e2)

Restriction of any family `A` (with the interface of affine.jl) to the plane through `prm_P` spanned by `e1`, `e2`:
the `Reparametrised` family (s, t) ↦ L(prm_P + s e1 + t e2). For an `AffineLiouvillian`, the method of affine.jl
gives an affine family instead.
"""
slice(A, prm_P, e1, e2) = Reparametrised(A, st -> prm_P + st[1]*e1 + st[2]*e2; Jφ=_ -> hcat(e1, e2))

"""
    pair_disc(A, prm_P, σ)

The discriminant D = (λa - λb)² and the midpoint (λa + λb)/2 of the two eigenvalues of A(prm_P) nearest `σ`
(`pair_near`). Both are symmetric in the pair.
"""
function pair_disc(A, prm_P, σ)
    (λa, _), (λb, _) = pair_near(A(prm_P), σ)
    return (λa - λb)^2, (λa + λb)/2
end

# (Re D, Im D) as a real vector
reim_vec(D) = SVector(real(D), imag(D))

"""
    disc_gradient(A, prm_P, σ; fd)

The gradients of Re D and Im D at `prm_P` (central differences of step `fd` along each coordinate), as the columns of
an N×2 matrix, and the midpoint of the pair at prm_P.
"""
function disc_gradient(A, prm_P::SVector{N}, σ; fd) where N
    _, λm = pair_disc(A, prm_P, σ)
    G = zeros(N, 2)
    for k ∈ 1:N
        e = SVector(ntuple(i -> i == k ? fd : 0.0, N))
        G[k, :] = (reim_vec(pair_disc(A, prm_P + e, λm)[1]) - reim_vec(pair_disc(A, prm_P - e, λm)[1]))/2fd
    end
    return G, λm
end

"""
    ep_tangent(A, prm_P, σ; fd)

Unit tangent of the EP line through `prm_P` (3D), ∇Re D × ∇Im D, and the midpoint of the pair there. Its sign is
arbitrary: the tracker orients it along the previous one.
"""
function ep_tangent(A, prm_P::P3, σ; fd)
    G, λm = disc_gradient(A, prm_P, σ; fd)
    t = P3(G[:, 1]) × P3(G[:, 2])
    norm(t) > 0 || error("∇Re D ∥ ∇Im D at $prm_P: not a regular point of an EP line")
    return normalize(t), λm
end

"""
    ep_newton(A, st0, σ; fd, xtol, maxiter=10)

Newton iterations on D = 0 for a two-parameter family `A` (e.g. a `slice`), from the point `st0`: Re D = Im D = 0 is
a 2×2 real system, its Jacobian from central differences of step `fd`. The shift of `pair_near` follows the midpoint
of the pair. Returns (point, midpoint of the pair there, converged): converged when a step is below `xtol`.
"""
function ep_newton(A, st0::P2, σ; fd, xtol, maxiter=10)
    st, λm = st0, σ
    for _ ∈ 1:maxiter
        G, λm = disc_gradient(A, st, λm; fd)
        D, λm = pair_disc(A, st, λm)
        δ = -(transpose(G) \ reim_vec(D))                   # the Jacobian's rows are ∇Re D, ∇Im D
        st += δ
        norm(δ) < xtol && return st, pair_disc(A, st, λm)[2], true
    end
    return st, λm, false
end

"""
    ep_correct(A, prm_P, t, σ; w, depth=4, fd=1e-3w, xtol=1e-8w, kwargs...)

Corrector of the EP-line tracker: the EP in the plane through `prm_P` normal to `t` (3D), near the eigenvalue `σ`.
`ep_bisect` on the `slice` of `A` from a square of half-width `w` around prm_P, slightly off-centre (it must swap the
pair: one EP inside), `depth` levels, then `ep_newton` from the centre of the final rectangle, which must converge within that
rectangle (up to half its size). `kwargs` are passed to `track`. Throws if any of this fails. Returns (EP, midpoint
of the pair there).
"""
function ep_correct(A, prm_P::P3, t::P3, σ; w, depth=4, fd=1e-3w, xtol=1e-8w, kwargs...)
    e1, e2 = plane_basis(t)
    A_sl = slice(A, prm_P, e1, e2)
    targets(st) = first.(pair_near(A_sl(st), σ))
    # off-centre by a generic fraction of w: with a good predictor the EP is near prm_P, which must not lie on the
    # first dividing lines of the bisection (the midlines of the square)
    a, b = 0.1373w, 0.0719w
    (x0, x1, y0, y1), _, _ = ep_bisect(A_sl, (a - w, a + w, b - w, b + w), targets; depth, kwargs...)
    st, λm, converged = ep_newton(A_sl, P2((x0 + x1)/2, (y0 + y1)/2), σ; fd, xtol)
    converged || error("Newton on D did not converge from the bisection's rectangle")
    hx, hy = (x1 - x0)/2, (y1 - y0)/2
    (x0 - hx ≤ st[1] ≤ x1 + hx && y0 - hy ≤ st[2] ≤ y1 + hy) || error("Newton on D left the bisection's rectangle")
    return prm_P + st[1]*e1 + st[2]*e2, λm
end

"""
    EPLine

A tracked line of EP2s: the `points` (3D), the coalesced eigenvalue `λs` there (midpoint of the pair), the unit
`tangents` (oriented along the tracking), and the `status` that ended the tracking: `:max_steps`, `:left_bounds` (the
last point is the first one outside), `:closed` (back near the first point, same direction), `:step_too_small` (the
corrector failed down to `hmin`: the line may end at a higher-order point, or leave the region where the pair is
found).
"""
struct EPLine
    points::Vector{P3}
    λs::Vector{ComplexF64}
    tangents::Vector{P3}
    status::Symbol
end

"""
    track_ep_line(A, prm_P0, σ0; h, direction=1, nsteps=200, hmin=h/64, hmax=4h, w_ratio=0.5, θmax=0.15,
                  bounds=nothing, depth=4, fd_ratio=1e-3, xtol_ratio=1e-8, kwargs...)

Follows the line of EP2s of the three-parameter family `A` through `prm_P0`, an approximate EP (e.g. from a scan or
`ep_bisect` in a 2D slice, mapped to 3D) whose coalesced eigenvalue is near `σ0`. Returns an `EPLine`.
- Seed: the tangent at prm_P0, oriented by `direction` (±1: call twice for both halves of the line), then the
  corrector in the plane through prm_P0 normal to it.
- Step: predictor P + h t, corrector `ep_correct` from a square of half-width `w_ratio*h` (the prediction error,
  ~ θ h/2 for a turn θ per step, must stay well inside it), new tangent ∇Re D × ∇Im D there.
- Step control: a step whose corrector fails, or whose tangent turns by more than `θmax` (radians), is rejected and h
  halved; after a step turning by less than θmax/3, h grows by 1.5, up to `hmax`.
- Stops after `nsteps` points, when a point leaves `bounds = (lo, hi)` (corners of a box, `P3`), when the line closes,
  or when h < `hmin`.
- Finite-difference steps `fd_ratio` times the box half-width (Newton) or h (tangent); Newton converged when a step is
  below `xtol_ratio` times the box half-width. `kwargs` are passed to `track` (bisection lines).
"""
function track_ep_line(A, prm_P0::P3, σ0; h, direction=1, nsteps=200, hmin=h/64, hmax=4h, w_ratio=0.5, θmax=0.15,
                       bounds=nothing, depth=4, fd_ratio=1e-3, xtol_ratio=1e-8, kwargs...)
    correct(prm_P, t, σ, h) = ep_correct(A, prm_P, t, σ; w=w_ratio*h, depth, fd=fd_ratio*w_ratio*h,
                                         xtol=xtol_ratio*w_ratio*h, kwargs...)
    t, σ = ep_tangent(A, prm_P0, σ0; fd=fd_ratio*h)
    P, λ = correct(prm_P0, direction*t, σ, h)
    t_P, λ = ep_tangent(A, P, λ; fd=fd_ratio*h)
    points, λs, tangents = [P], [λ], [sign(t_P ⋅ (direction*t))*t_P]
    status = :max_steps
    while length(points) < nsteps
        if h < hmin
            status = :step_too_small
            break
        end
        P, λ, t = points[end], λs[end], tangents[end]
        # shift: the coalesced eigenvalue extrapolated linearly from the last two points
        σ = length(points) ≥ 2 ? λ + (λ - λs[end - 1])*h/norm(P - points[end - 1]) : λ
        Q, λQ, t_Q = try
            Q, λQ = correct(P + h*t, t, σ, h)
            t_Q, λQ = ep_tangent(A, Q, λQ; fd=fd_ratio*h)
            Q, λQ, sign(t_Q ⋅ t)*t_Q
        catch err
            err isa InterruptException && rethrow()
            @debug "step h = $h from $P failed: $(sprint(showerror, err))"
            h /= 2
            continue
        end
        θ = acos(clamp(t_Q ⋅ t, -1, 1))
        if θ > θmax
            h /= 2
            continue
        end
        push!(points, Q); push!(λs, λQ); push!(tangents, t_Q)
        if bounds !== nothing && !all(bounds[1] .≤ Q .≤ bounds[2])
            status = :left_bounds
            break
        end
        if length(points) ≥ 4 && norm(Q - points[1]) < h && t_Q ⋅ tangents[1] > 0
            status = :closed
            break
        end
        θ < θmax/3 && (h = min(1.5h, hmax))
    end
    return EPLine(points, λs, tangents, status)
end
