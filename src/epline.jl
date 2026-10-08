# Following a line of EP2s through a three-parameter space. Complex EP2s have codimension 2: in 3D they form curves,
# the intersection of the surfaces Re D = 0 and Im D = 0, with D = (λa - λb)² the discriminant of the pair. D is
# smooth in the parameters (the eigenvalues have a √ branch point at the EP, D does not), vanishes linearly at an EP2
# and is symmetric in a, b: no branch labelling. Predictor–corrector continuation along the curve (`track_ep_line`):
#   predictor   a step h along the tangent ∇Re D × ∇Im D (finite differences of D)
#   corrector   in the plane through the predicted point normal to the tangent (`ep_correct`): Newton on D, accepted
#               if a small loop around the result swaps the pair (`check_loop`, with the pair carried from the
#               previous step's loop); otherwise a bracketing of the EP (`ep_bracket`), then Newton
# Real-axis EPs of a real Liouvillian have codimension 1 (surfaces in 3D): this tracker is for complex EP2s only.
# The tangent and the normal planes use the Euclidean metric of the coordinates: rescale them (`Reparametrised`) if
# the parameters have very different scales. Points of a normal plane are written ξ = (ξ₁, ξ₂) in its own basis.

"""
    slice(A, prm_P, e1, e2)

Restriction of any family `A` (with the interface of affine.jl) to the plane through `prm_P` spanned by `e1`, `e2`:
the `Reparametrised` family (s, t) ↦ L(prm_P + s e1 + t e2). For an `AffineLiouvillian`, the method of affine.jl
gives an affine family instead.
"""
slice(A, prm_P, e1, e2) = Reparametrised(A, st -> prm_P + st[1]*e1 + st[2]*e2; Jφ=_ -> hcat(e1, e2))

"""
    NormalPlane(origin, t)

The plane through `origin` (3D) normal to `t`, with the orthonormal basis (e1, e2) of `plane_basis`: `to_3d(plane, ξ)`
and `to_plane(plane, x)` convert between its coordinates ξ and 3D points, `slice(A, plane)` restricts a family to it.
"""
struct NormalPlane
    origin::P3
    t̂::P3
    e1::P3
    e2::P3
end
NormalPlane(origin, t) = NormalPlane(origin, normalize(t), plane_basis(t)...)
to_3d(plane::NormalPlane, ξ) = plane.origin + ξ[1]*plane.e1 + ξ[2]*plane.e2
to_plane(plane::NormalPlane, x) = P2((x - plane.origin) ⋅ plane.e1, (x - plane.origin) ⋅ plane.e2)
slice(A, plane::NormalPlane) = slice(A, plane.origin, plane.e1, plane.e2)

## The discriminant and its derivatives

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
    ep_newton(A, ξ0, σ; fd, xtol, maxiter=10)

Newton iterations on D = 0 for a two-parameter family `A` (e.g. a `slice`), from the point `ξ0`: Re D = Im D = 0 is a
2×2 real system, its Jacobian from central differences of step `fd`. The shift of `pair_near` follows the midpoint of
the pair. Returns (point, midpoint of the pair there, converged, G): converged when a step is below `xtol`; G, the last
Jacobian (columns ∇Re D, ∇Im D), gives the local model of the pair (`pair_model`).
"""
function ep_newton(A, ξ0::P2, σ; fd, xtol, maxiter=10)
    ξ, λm, G = ξ0, σ, zeros(2, 2)
    for _ ∈ 1:maxiter
        G, λm = disc_gradient(A, ξ, λm; fd)
        D, λm = pair_disc(A, ξ, λm)
        δξ = -(transpose(G) \ reim_vec(D))                   # the Jacobian's rows are ∇Re D, ∇Im D
        ξ += δξ
        norm(δξ) < xtol && return ξ, pair_disc(A, ξ, λm)[2], true, G
    end
    return ξ, λm, false, G
end

"""
    pair_model(ξ_EP, λm, G)

The pair predicted near its EP `ξ_EP` (coalesced eigenvalue `λm`, Jacobian `G` of (Re D, Im D) from `ep_newton`): the
function ξ ↦ [λm + √D(ξ)/2, λm - √D(ξ)/2] with D(ξ) ≈ ∇Re D⋅(ξ - ξ_EP) + i ∇Im D⋅(ξ - ξ_EP). Targets for the pair's
eigenvalues at ξ: unlike the two eigenvalues nearest λm, they stay on the right pair when another eigenvalue lies
within the pair's splitting √|D(ξ)| (crowded spectra).
"""
function pair_model(ξ_EP, λm, G)
    return ξ -> (D = (ξ - ξ_EP) ⋅ G[:, 1] + im*((ξ - ξ_EP) ⋅ G[:, 2]); [λm + sqrt(D)/2, λm - sqrt(D)/2])
end

## Corrector

"""
    pair_swaps(A, q, pairs; kwargs...)

Whether following the two eigenpairs `pairs` (at the start of the loop `q`) once around q swaps them: an odd number
of their EPs inside. `kwargs` are passed to `track`; a tracking that fails counts as no swap.
"""
function pair_swaps(A, q::ParamPath, pairs; kwargs...)
    sols = [first(track(A, q, λ0, r0; kwargs...)) for (λ0, r0) ∈ pairs]
    all(tracking_succeeded, sols) || return false
    λ_end = sols[1].u[end][end]
    return abs(λ_end - pairs[2][1]) < abs(λ_end - pairs[1][1])
end

# the eigenpairs of L nearest the two `targets` (shift offset: 1% of their separation)
pairs_near(L, targets) = eigenpairs_near(L, targets; δσ=0.01*abs(targets[1] - targets[2]))

"""
    check_loop(A, plane, ξ_EP, ρ, carry, model; kwargs...)

The loop that certifies an EP found by Newton at `ξ_EP` in the normal `plane`: a circle of radius `ρ` around it in
that plane (3D), and the pair at its start, as `(loop, pairs, carried)`.
- With `carry = (x, pairs)`, the pair at the previous loop's start x: the loop starts at its point nearest x, and the
  pair is tracked from x to there (a short segment beside the EP line), so that it is the same pair by continuation
  (`carried`).
- Without it, or if that tracking fails, the loop starts along e1 and the pair is picked near the targets of `model`
  (the local model of the pair, `pair_model`, a function of ξ).
The radius only has to exceed Newton's error, and must stay below the distance to any other EP involving one of the
pair's eigenvalues (e.g. the neighbouring lines of the QRM tower, which all meet at g = 0): a loop enclosing one cycles
four eigenvalues instead of swapping two. The cost of a loop near an EP does not depend on its radius (the √ structure
is scale invariant), so it is kept small.
"""
function check_loop(A, plane::NormalPlane, ξ_EP, ρ, carry, model; kwargs...)
    center = to_3d(plane, ξ_EP)
    if carry !== nothing
        d = carry.x - center
        d -= (d ⋅ plane.t̂)*plane.t̂                            # its component in the plane
        if norm(d) > 0
            u1 = normalize(d)
            loop = Circle(center, ρ, u1, plane.t̂ × u1)
            start = position(loop, 0.0)
            sols = [first(track(A, Segment(carry.x, start), λ0, r0; kwargs...)) for (λ0, r0) ∈ carry.pairs]
            if all(tracking_succeeded, sols)
                pairs = [(s.u[end][end], s.u[end][1:end - 1]) for s ∈ sols]
                pairs[1][1] != pairs[2][1] && return loop, pairs, true
            end
            @debug "carrying the pair to the new check loop failed"
        end
    end
    loop = Circle(center, ρ, plane.e1, plane.e2)
    start = position(loop, 0.0)
    return loop, pairs_near(A(start), model(to_plane(plane, start))), false
end

# Newton on D from the plane's origin, on the family `A_pl` restricted to the plane: (EP, midpoint, converged, model of
# the pair), or nothing if it fails or lands farther than `reach`
function newton_in_plane(A_pl, σ; reach, fd, xtol)
    ξ, λm, converged, G = try
        ep_newton(A_pl, P2(0, 0), σ; fd, xtol)
    catch err
        err isa InterruptException && rethrow()
        @debug "Newton from the predicted point failed: $(sprint(showerror, err))"
        return nothing
    end
    (norm(ξ) ≤ reach && all(isfinite, G)) || return nothing
    return (ξ=ξ, λ=λm, converged=converged, model=pair_model(ξ, λm, G))
end

"""
    ep_correct(A, prm_P, t, σ; w, newton_first=true, loop_ratio=1/16, carry=nothing, greedy=true, depth=4, fd=1e-3w,
               xtol=1e-8w, kwargs...)

Corrector of the EP-line tracker: the EP of the pair near the eigenvalue `σ` in the plane through `prm_P` normal to
`t` (3D), within about `w` of prm_P.
- `newton_first`: `ep_newton` from prm_P, accepted if it converges within w and a loop of radius `loop_ratio*w` around
  the result swaps the pair (`check_loop`, which carries the pair from the previous step's loop if `carry` is given).
  Cheap: two tracked loops instead of a bracketing's lines.
- Otherwise: a bracketing from a square of half-width w around prm_P, slightly off-centre (it must swap the pair: one
  EP inside), `ep_bracket` down to a rectangle of size 1e-2 w if `greedy`, else `ep_bisect` with `depth` levels; then
  `ep_newton` from the centre of the final rectangle, which must converge within it (up to half its size). The pair at
  the corners is picked from Newton's local model of the pair when available, not as the two eigenvalues nearest σ,
  which fails in crowded spectra.
`kwargs` are passed to `track`. Throws if the bracketing fails. Returns `(P, λ, path, carry, carried)`: the EP (3D),
the coalesced eigenvalue, `:newton` or `:bracket`, the check loop's start and pair `(x, pairs)` to carry to the next
step (or `nothing`), and whether the pair was carried.
"""
function ep_correct(A, prm_P::P3, t::P3, σ; w, newton_first=true, loop_ratio=1/16, carry=nothing, greedy=true,
                    depth=4, fd=1e-3w, xtol=1e-8w, kwargs...)
    plane = NormalPlane(prm_P, t)
    A_pl = slice(A, plane)
    newton = newton_in_plane(A_pl, σ; reach=2w, fd, xtol)
    if newton_first && newton !== nothing && newton.converged && norm(newton.ξ) ≤ w
        loop, pairs, carried = check_loop(A, plane, newton.ξ, loop_ratio*w, carry, newton.model; kwargs...)
        pair_swaps(A, loop, pairs; kwargs...) &&
            return (P=to_3d(plane, newton.ξ), λ=newton.λ, path=:newton, carry=(x=position(loop, 0.0), pairs=pairs),
                    carried=carried)
        @debug "Newton-first rejected: the check loop does not swap the pair (carried: $carried)"
    end
    ξ, λ = bracket_in_plane(A_pl, σ, newton; w, greedy, depth, fd, xtol, kwargs...)
    return (P=to_3d(plane, ξ), λ=λ, path=:bracket, carry=nothing, carried=false)
end

# the bracketing fallback of `ep_correct`, in the plane's coordinates: (EP, coalesced eigenvalue)
function bracket_in_plane(A_pl, σ, newton; w, greedy, depth, fd, xtol, kwargs...)
    targets = newton === nothing ? (ξ -> first.(pair_near(A_pl(ξ), σ))) : newton.model
    σ_EP = newton === nothing ? σ : newton.λ
    # off-centre by a generic fraction of w: with a good predictor the EP is near the origin, which must not lie on the
    # first cuts (the midlines of the square for the bisection)
    a, b = 0.1373w, 0.0719w
    square = (a - w, a + w, b - w, b + w)
    (x0, x1, y0, y1), _, _ = greedy ? ep_bracket(A_pl, square, targets; tol=1e-2w, kwargs...) :
                                      ep_bisect(A_pl, square, targets; depth, kwargs...)
    ξ, λ, converged = ep_newton(A_pl, P2((x0 + x1)/2, (y0 + y1)/2), σ_EP; fd=min(fd, (x1 - x0)/10), xtol)
    converged || error("Newton on D did not converge from the bracketing rectangle")
    hx, hy = (x1 - x0)/2, (y1 - y0)/2
    (x0 - hx ≤ ξ[1] ≤ x1 + hx && y0 - hy ≤ ξ[2] ≤ y1 + hy) || error("Newton on D left the bracketing rectangle")
    return ξ, λ
end

## Tracker

"""
    EPLine

A tracked line of EP2s: the `points` (3D), the coalesced eigenvalue `λs` there (midpoint of the pair), the unit
`tangents` (oriented along the tracking), the `status` that ended the tracking: `:max_steps`, `:left_bounds` (the
last point is the first one outside), `:closed` (back near the first point, same direction), `:step_too_small` (the
corrector failed down to `hmin`: the line may end at a higher-order point, or leave the region where the pair is
found), and `counts` of the corrections: `newton` and `bracket` (successful corrections, by the path of `ep_correct`,
including the seed and steps then rejected for turning), `failed` (the corrector threw), `turned` (steps rejected:
the tangent turned by more than θmax), and `carried` (check loops whose pair was carried from the previous one).
"""
struct EPLine
    points::Vector{P3}
    λs::Vector{ComplexF64}
    tangents::Vector{P3}
    status::Symbol
    counts::NamedTuple{(:newton, :bracket, :failed, :turned, :carried),NTuple{5,Int}}
end

"""
    track_ep_line(A, prm_P0, σ0; h, direction=1, nsteps=200, hmin=h/64, hmax=4h, w_ratio=0.5, θmax=0.15,
                  bounds=nothing, newton_first=true, continue_pair=true, greedy=true, depth=4, fd_ratio=1e-3,
                  xtol_ratio=1e-8, kwargs...)

Follows the line of EP2s of the three-parameter family `A` through `prm_P0`, an approximate EP (e.g. from a scan or
`ep_bracket` in a 2D slice, mapped to 3D) whose coalesced eigenvalue is near `σ0`. Returns an `EPLine`.
- Seed: the tangent at prm_P0, oriented by `direction` (±1: call twice for both halves of the line), then the
  corrector in the plane through prm_P0 normal to it.
- Step: predictor P + h t, corrector `ep_correct` within the distance `w_ratio*h` (the prediction error, ~ θ h/2 for
  a turn θ per step, must stay well inside it; `newton_first`, `greedy` and `depth` are passed to it), new tangent
  ∇Re D × ∇Im D there. With `continue_pair`, the pair of each check loop is carried from the previous accepted one
  (see `check_loop`), so that the line follows the same pair by continuation.
- Step control: a step whose corrector fails, or whose tangent turns by more than `θmax` (radians), is rejected and h
  halved; after a step turning by less than θmax/3, h grows by 1.5, up to `hmax`.
- Stops after `nsteps` points, when a point leaves `bounds = (lo, hi)` (corners of a box, `P3`), when the line closes,
  or when h < `hmin`.
- Finite-difference steps `fd_ratio` times the box half-width (Newton) or h (tangent); Newton converged when a step is
  below `xtol_ratio` times the box half-width. `kwargs` are passed to `track`.
"""
function track_ep_line(A, prm_P0::P3, σ0; h, direction=1, nsteps=200, hmin=h/64, hmax=4h, w_ratio=0.5, θmax=0.15,
                       bounds=nothing, newton_first=true, continue_pair=true, greedy=true, depth=4, fd_ratio=1e-3,
                       xtol_ratio=1e-8, kwargs...)
    counts = Dict(:newton => 0, :bracket => 0, :failed => 0, :turned => 0, :carried => 0)
    carry = nothing                                   # the pair at the last accepted check loop's start
    function correct(prm_P, t, σ, h)
        c = ep_correct(A, prm_P, t, σ; w=w_ratio*h, newton_first, greedy, depth, carry=continue_pair ? carry : nothing,
                       fd=fd_ratio*w_ratio*h, xtol=xtol_ratio*w_ratio*h, kwargs...)
        counts[c.path] += 1
        counts[:carried] += c.carried
        return c
    end
    function failed!(err, what)                       # a failed correction: halve h
        err isa InterruptException && rethrow()
        @debug "$what with h = $h failed: $(sprint(showerror, err))"
        counts[:failed] += 1
        h /= 2
    end

    # seed: corrected in the plane normal to the tangent at prm_P0, h halved on failure as for the steps
    t0, σ = ep_tangent(A, prm_P0, σ0; fd=fd_ratio*h)
    seed = nothing
    while seed === nothing
        try
            seed = correct(prm_P0, direction*t0, σ, h)
        catch err
            h/2 < hmin && rethrow()
            failed!(err, "seed correction")
        end
    end
    carry = seed.carry
    t_P, λ = ep_tangent(A, seed.P, seed.λ; fd=fd_ratio*h)
    points, λs, tangents = [seed.P], [λ], [sign(t_P ⋅ (direction*t0))*t_P]

    status = :max_steps
    while length(points) < nsteps
        if h < hmin
            status = :step_too_small
            break
        end
        P, λ, t = points[end], λs[end], tangents[end]
        # shift: the coalesced eigenvalue extrapolated linearly from the last two points
        σ = length(points) ≥ 2 ? λ + (λ - λs[end - 1])*h/norm(P - points[end - 1]) : λ
        c, t_Q, λQ = try
            c = correct(P + h*t, t, σ, h)
            t_Q, λQ = ep_tangent(A, c.P, c.λ; fd=fd_ratio*h)
            c, sign(t_Q ⋅ t)*t_Q, λQ
        catch err
            failed!(err, "step from $P")
            continue
        end
        θ = acos(clamp(t_Q ⋅ t, -1, 1))
        if θ > θmax
            counts[:turned] += 1
            h /= 2
            continue
        end
        push!(points, c.P); push!(λs, λQ); push!(tangents, t_Q)
        carry = c.carry
        if bounds !== nothing && !all(bounds[1] .≤ c.P .≤ bounds[2])
            status = :left_bounds
            break
        end
        if length(points) ≥ 4 && norm(c.P - points[1]) < h && t_Q ⋅ tangents[1] > 0
            status = :closed
            break
        end
        θ < θmax/3 && (h = min(1.5h, hmax))
    end
    return EPLine(points, λs, tangents, status,
                  (; (k => counts[k] for k ∈ (:newton, :bracket, :failed, :turned, :carried))...))
end

"""
    track_ep_lines(A, seeds; ntasks=4, kwargs...)

Tracks the EP lines through several `seeds` (pairs `(prm_P0, σ0)`, as for `track_ep_line`), each in both directions:
one task per half-line, at most `ntasks` at a time (on as many Julia threads as available: start Julia with
`--threads=ntasks` at least). The half-lines are independent, each with its own solvers; BLAS is set to one thread
meanwhile (restored after), so that the sparse factorisations of the tasks do not oversubscribe the cores. `kwargs`
are passed to `track_ep_line`. Returns, per seed, `(forward, backward)`: the `EPLine`s with direction +1 and -1
(see `join_halves`).
"""
function track_ep_lines(A, seeds; ntasks=4, kwargs...)
    jobs = [(k, direction) for k ∈ eachindex(seeds) for direction ∈ (1, -1)]
    slots = Base.Semaphore(ntasks)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    halves = try
        tasks = map(jobs) do (k, direction)
            Threads.@spawn Base.acquire(slots) do
                prm_P0, σ0 = seeds[k]
                track_ep_line(A, prm_P0, σ0; direction, kwargs...)
            end
        end
        fetch.(tasks)
    finally
        BLAS.set_num_threads(blas_threads)
    end
    return [(forward=halves[2k - 1], backward=halves[2k]) for k ∈ eachindex(seeds)]
end

"""
    join_halves(forward, backward)

The two halves of an EP line (`EPLine`s from the same seed, directions +1 and -1) as one line, `(points, λs,
tangents)`, running from the end of `backward` through the seed to the end of `forward`, the tangents along that
direction.
"""
join_halves(forward::EPLine, backward::EPLine) =
    (points=[reverse(backward.points); forward.points[2:end]], λs=[reverse(backward.λs); forward.λs[2:end]],
     tangents=[-reverse(backward.tangents); forward.tangents[2:end]])
