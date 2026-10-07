# Locating an EP2 by bisection of rectangles, with edge reuse. Both eigenvalues of the pair are tracked along straight
# lines (TrackedLine: one ODE solution per branch). Every side of every rectangle lies on a tracked line; following an
# eigenvalue along a side means finding the branch that matches it at the start and reading that branch at the end.
# A rectangle swaps the pair iff it encloses an odd number of their EPs. Splitting a rectangle only requires tracking
# the new dividing line: the other sides are parts of earlier lines. `ep_bracket` is a greedy variant: a regula falsi
# step on the discriminant D = (λa - λb)², bracketed by cutting the rectangle into 9 cells around the fitted zero.

"""
    TrackedLine(a, b, sols)

The two eigenvalues of a pair tracked along the segment from `a` to `b`: one ODE solution of `track` per branch, with
u ∈ [0, 1] from a to b.
"""
struct TrackedLine
    a::P2
    b::P2
    sols::Vector{Any}
end

"""
    branch(l::TrackedLine, j, s)

The eigenvalue of branch `j` at the parameter `s` along `l`.
"""
branch(l::TrackedLine, j, s) = l.sols[j](s)[end]

"""
    line_param(l::TrackedLine, x; tol=1e-9)

The parameter s of the point `x` along `l`, or `nothing` if x is not on l.
"""
function line_param(l::TrackedLine, x; tol=1e-9)
    d = l.b - l.a
    s = (x - l.a) ⋅ d/(d ⋅ d)
    on = abs(d[1]*(x - l.a)[2] - d[2]*(x - l.a)[1]) ≤ tol*(d ⋅ d) && -tol ≤ s ≤ 1 + tol
    return on ? clamp(s, 0., 1.) : nothing
end

"""
    find_line(lines, x, y)

The tracked line containing both `x` and `y`, with their parameters along it: `(l, sx, sy)`.
"""
function find_line(lines, x, y)
    for l ∈ lines
        sx, sy = line_param(l, x), line_param(l, y)
        sx !== nothing && sy !== nothing && return l, sx, sy
    end
    error("no tracked line contains both $x and $y")
end

"""
    track_line(A, a, b, targets; kwargs...)

Tracks both eigenvalues of the pair from `a` to `b`, starting from shift-invert near the `targets` at a. The shift
offset is at most 1% of the targets' separation: near an EP the pair is close, and a fixed offset would converge both
searches to the same eigenvalue.
"""
function track_line(A, a::P2, b::P2, targets; kwargs...)
    pairs = eigenpairs_near(A(a), targets; δσ=min(1e-3, 0.01*abs(targets[1] - targets[2])))
    sols = Any[]
    for (λ0, r0) ∈ pairs
        sol, _ = track(A, Segment(a, b), λ0, r0; kwargs...)
        tracking_succeeded(sol) || error("tracking along $a → $b failed ($(sol.retcode))")
        push!(sols, sol)
    end
    return TrackedLine(a, b, sols)
end

"""
    pair_midpoint(lines, x, y)

Midpoint of the pair at `x`, read on the tracked line through x and y: the shift for `pair_near` close to the EP.
"""
function pair_midpoint(lines, x, y)
    l, s, _ = find_line(lines, x, y)
    return (branch(l, 1, s) + branch(l, 2, s))/2
end

"""
    transport(lines, λ, x, y; ratio=0.25)

Follows the eigenvalue `λ` from `x` to `y` along the tracked line containing both. The branch must be unambiguous: its
distance to λ at x below `ratio` times the gap between the two branches there.
"""
function transport(lines, λ, x, y; ratio=0.25)
    l, sx, sy = find_line(lines, x, y)
    v = [branch(l, j, sx) for j ∈ eachindex(l.sols)]
    d = abs.(v .- λ)
    j = argmin(d)
    d[j] ≤ ratio*abs(v[1] - v[2]) || error("ambiguous branch at $x (the EP may lie on this line)")
    return branch(l, j, sy)
end

"""
    rect_swaps(lines, (x0, x1, y0, y1))

Whether the rectangle swaps the pair, following one eigenvalue counterclockwise from (x0, y0) on the tracked lines.
"""
function rect_swaps(lines, (x0, x1, y0, y1))
    c = (P2(x0, y0), P2(x1, y0), P2(x1, y1), P2(x0, y1))
    l, s, _ = find_line(lines, c[1], c[2])
    λa, λb = branch(l, 1, s), branch(l, 2, s)
    λ = λa
    for k ∈ 1:4
        λ = transport(lines, λ, c[k], c[mod1(k + 1, 4)])
    end
    return abs(λ - λb) < abs(λ - λa)
end

"""
    ep_bisect(A, rect, corner_targets; depth=12, shift=0.1, kwargs...)

Bisection from a rectangle `(x0, x1, y0, y1)` that swaps the pair: split it across its longer side, track the dividing
line, keep the half that swaps (exactly one must), `depth` times. A dividing line that fails (EP too close to it) is
moved by `shift` of the side, up to three times. `corner_targets(x)` gives the starting targets at the outer corners.
Returns (final rectangle, rectangles at every level, all tracked lines).
"""
function ep_bisect(A, rect, corner_targets; depth=12, shift=0.1, kwargs...)
    x0, x1, y0, y1 = rect
    c = (P2(x0, y0), P2(x1, y0), P2(x1, y1), P2(x0, y1))
    lines = TrackedLine[track_line(A, c[1], c[2], corner_targets(c[1]); kwargs...),
                        track_line(A, c[2], c[3], corner_targets(c[2]); kwargs...),
                        track_line(A, c[4], c[3], corner_targets(c[4]); kwargs...),
                        track_line(A, c[1], c[4], corner_targets(c[1]); kwargs...)]
    rect_swaps(lines, rect) || error("the initial rectangle does not swap the pair: no (or an even number of) EP inside")
    history = [rect]
    for level ∈ 1:depth
        x0, x1, y0, y1 = rect
        vertical = (x1 - x0) ≥ (y1 - y0)                 # split across the longer side
        for attempt ∈ 0:3
            f = 0.5 + shift*attempt*(isodd(attempt) ? 1 : -1)/2
            m = vertical ? x0 + f*(x1 - x0) : y0 + f*(y1 - y0)
            a, b = vertical ? (P2(m, y0), P2(m, y1)) : (P2(x0, m), P2(x1, m))
            halves = vertical ? ((x0, m, y0, y1), (m, x1, y0, y1)) : ((x0, x1, y0, m), (x0, x1, m, y1))
            try
                # starting targets at a: the two branches of the side it lies on
                l, s, _ = find_line(lines, a, vertical ? P2(x1, y0) : P2(x0, y1))
                new = track_line(A, a, b, [branch(l, j, s) for j ∈ 1:2]; kwargs...)
                sw = [rect_swaps([lines; new], h) for h ∈ halves]
                count(sw) == 1 || error("$(count(sw)) halves swap")
                push!(lines, new)
                rect = halves[findfirst(sw)]
                break
            catch err
                attempt == 3 && rethrow()
                @warn "level $level: dividing line at f = $f failed ($(sprint(showerror, err))), shifting it"
            end
        end
        push!(history, rect)
    end
    return rect, history, lines
end

## Greedy bracketing: regula falsi on the discriminant, bracketed by a 3 × 3 split

"""
    disc_fit(lines, (x0, x1, y0, y1); k=4)

Affine fit of the discriminant D = (λa - λb)² of the pair, read on the tracked lines along the sides of the rectangle
(`k` points per side): D ≈ D0 + Dx (x - xc) + Dy (y - yc) around its centre, by least squares. D is smooth and vanishes
linearly at an EP2, so the zero of the fit approximates the EP with an error ~ curvature × size². Returns (zero of the
fit, error estimate: the largest residual of the fit over the smallest singular value of the real Jacobian of
(Re D, Im D)).
"""
function disc_fit(lines, (x0, x1, y0, y1); k=4)
    c = (P2(x0, y0), P2(x1, y0), P2(x1, y1), P2(x0, y1))
    pts, Ds = P2[], ComplexF64[]
    for i ∈ 1:4
        l, sa, sb = find_line(lines, c[i], c[mod1(i + 1, 4)])
        for j ∈ 0:k - 1
            s = sa + (sb - sa)*j/k
            push!(pts, l.a + s*(l.b - l.a))
            push!(Ds, (branch(l, 1, s) - branch(l, 2, s))^2)
        end
    end
    xc = P2((x0 + x1)/2, (y0 + y1)/2)
    M = [ones(length(pts)) [(p - xc)[1] for p ∈ pts] [(p - xc)[2] for p ∈ pts]]
    coeffs = M \ Ds
    D0, Dx, Dy = coeffs
    J = [real(Dx) real(Dy); imag(Dx) imag(Dy)]
    zero = xc + P2(J \ [-real(D0), -imag(D0)])
    return zero, maximum(abs.(M*coeffs .- Ds))/minimum(svdvals(J))
end

# positions of the cuts across [lo, hi] around the fitted zero p: p ± δ, kept if well inside (no sliver cells, no cut on
# a side); with no usable fit (p outside, or δ too large), one cut near the middle (generic offset: not the centre)
function bracket_cuts(lo, hi, p, δ)
    margin = 0.02*(hi - lo)
    (lo < p < hi && 2δ < 0.6*(hi - lo)) || return [lo + 0.4853*(hi - lo)]
    return filter(c -> lo + margin < c < hi - margin, [p - δ, p + δ])
end

"""
    ep_bracket(A, rect, corner_targets; tol, maxiter=10, safety=2.0, threaded=true, kwargs...)

Greedy variant of `ep_bisect`, from a rectangle `(x0, x1, y0, y1)` that swaps the pair, until its larger side is
below `tol` (or `maxiter` iterations). Each iteration: the zero P* of the affine fit of D on the rectangle's sides
(`disc_fit`, free: D is read on the tracked lines), and the lines x = x* ± δ, y = y* ± δ across the whole rectangle,
with δ = `safety` × the fit's error estimate. They cut it into up to 9 cells, the central one a small rectangle around
P*; every cell's sides lie on tracked lines, and exactly one cell swaps (the bracket is kept, as in the bisection).
If the fit is right, the rectangle shrinks from size w to ~ w²; if the EP lies outside the central cell, an outer cell
is kept and the next fit is made there. Without a usable fit (P* outside, or δ too large), the rectangle is
quadrisected. The new lines are tracked in parallel if `threaded`. A cut that fails (EP too close to it) is retried
with δ enlarged by 1.7, up to twice. `corner_targets(x)` gives the starting targets at the outer corners; `kwargs` are
passed to `track`. Returns (final rectangle, rectangles at every iteration, all tracked lines), as `ep_bisect`.
"""
function ep_bracket(A, rect, corner_targets; tol, maxiter=10, safety=2.0, threaded=true, kwargs...)
    track_all(specs) = threaded ? fetch.([Threads.@spawn(track_line(A, a, b, tg; kwargs...)) for (a, b, tg) ∈ specs]) :
                            [track_line(A, a, b, tg; kwargs...) for (a, b, tg) ∈ specs]
    x0, x1, y0, y1 = rect
    c = (P2(x0, y0), P2(x1, y0), P2(x1, y1), P2(x0, y1))
    lines = TrackedLine[track_all([(c[1], c[2], corner_targets(c[1])), (c[2], c[3], corner_targets(c[2])),
                             (c[4], c[3], corner_targets(c[4])), (c[1], c[4], corner_targets(c[1]))])...]
    rect_swaps(lines, rect) || error("the initial rectangle does not swap the pair: no (or an even number of) EP inside")
    history = [rect]
    for it ∈ 1:maxiter
        x0, x1, y0, y1 = rect
        max(x1 - x0, y1 - y0) ≤ tol && break
        P, err = disc_fit(lines, rect)
        δ = max(safety*err, 1e-4*max(x1 - x0, y1 - y0), tol/4)
        for attempt ∈ 0:2
            xs, ys = bracket_cuts(x0, x1, P[1], δ), bracket_cuts(y0, y1, P[2], δ)
            try
                # starting targets of a cut: the two branches of the side it starts on (bottom or left)
                side_branches(a, b_side) = (l_s = find_line(lines, a, b_side); [branch(l_s[1], j, l_s[2]) for j ∈ 1:2])
                specs = [[(P2(m, y0), P2(m, y1), side_branches(P2(m, y0), P2(x1, y0))) for m ∈ xs];
                         [(P2(x0, m), P2(x1, m), side_branches(P2(x0, m), P2(x0, y1))) for m ∈ ys]]
                new = track_all(specs)
                bx, by = [x0; xs; x1], [y0; ys; y1]
                cells = [(bx[i], bx[i + 1], by[j], by[j + 1]) for i ∈ 1:length(bx) - 1 for j ∈ 1:length(by) - 1]
                sw = [rect_swaps([lines; new], cl) for cl ∈ cells]
                count(sw) == 1 || error("$(count(sw)) cells swap")
                append!(lines, new)
                rect = cells[findfirst(sw)]
                break
            catch err
                attempt == 2 && rethrow()
                @warn "iteration $it: cuts at δ = $δ failed ($(sprint(showerror, err))), enlarging δ"
                δ *= 1.7
            end
        end
        push!(history, rect)
    end
    return rect, history, lines
end
