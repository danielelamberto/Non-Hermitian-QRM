# Locating an EP2 by bisection of rectangles, with edge reuse. Both eigenvalues of the pair are tracked along straight
# lines (TrackedLine: one ODE solution per branch). Every side of every rectangle lies on a tracked line; following an
# eigenvalue along a side means finding the branch that matches it at the start and reading that branch at the end.
# A rectangle swaps the pair iff it encloses an odd number of their EPs. Splitting a rectangle only requires tracking
# the new dividing line: the other sides are parts of earlier lines.

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
    s = dot(x - l.a, d)/dot(d, d)
    on = abs(d[1]*(x - l.a)[2] - d[2]*(x - l.a)[1]) ≤ tol*dot(d, d) && -tol ≤ s ≤ 1 + tol
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

Tracks both eigenvalues of the pair from `a` to `b`, starting from shift-invert near the `targets` at a.
"""
function track_line(A, a::P2, b::P2, targets; kwargs...)
    pairs = eigenpairs_near(A(a), targets)
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
