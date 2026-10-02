# Paths in a two-dimensional parameter space. A path p is parametrised by u ∈ [0, 1] and provides position(p, u),
# velocity(p, u) = d position/du, and breakpoints(p), the u where the velocity jumps (passed as tstops to the ODE solver).
# Loops have position(p, 0) == position(p, 1).

"""
    P2

A point (or a velocity) of the two-dimensional parameter plane.
"""
const P2 = SVector{2,Float64}

"""
    ParamPath

A path u ∈ [0, 1] ↦ point of the parameter plane: subtypes implement `position(p, u)`, `velocity(p, u)` and, if the
velocity jumps, `breakpoints(p)`.
"""
abstract type ParamPath end

"""
    breakpoints(p::ParamPath)

The values of u where the velocity of `p` jumps (none by default).
"""
breakpoints(::ParamPath) = Float64[]

"""
    Circle(center, ρ)

Counterclockwise circle of radius `ρ`, starting at its rightmost point.
"""
struct Circle <: ParamPath
    center::P2
    ρ::Float64
end
position(c::Circle, u) = c.center + c.ρ*P2(cospi(2u), sinpi(2u))
velocity(c::Circle, u) = 2π*c.ρ*P2(-sinpi(2u), cospi(2u))

"""
    Polygon(vertices)

Closed polygon through `vertices` in order (a rectangle is 4 vertices), each edge covering an equal share of u.
"""
struct Polygon <: ParamPath
    vertices::Vector{P2}
end
# number of edges, index of the current edge (0-based) and position t ∈ [0, 1] along it
function _edge(p::Polygon, u)
    n = length(p.vertices)
    k = min(floor(Int, n*u), n - 1)
    return n, k, n*u - k
end
function position(p::Polygon, u)
    n, k, t = _edge(p, u)
    return (1 - t)*p.vertices[k + 1] + t*p.vertices[mod1(k + 2, n)]
end
function velocity(p::Polygon, u)
    n, k, _ = _edge(p, u)
    return n*(p.vertices[mod1(k + 2, n)] - p.vertices[k + 1])
end
breakpoints(p::Polygon) = collect(1:length(p.vertices) - 1) ./ length(p.vertices)

"""
    Segment(a, b)

Open straight path from `a` to `b` (not a loop): the grid edges of a scan, the lines of a bisection.
"""
struct Segment <: ParamPath
    a::P2
    b::P2
end
position(p::Segment, u) = p.a + u*(p.b - p.a)
velocity(p::Segment, u) = p.b - p.a
