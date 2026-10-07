# Paths in parameter space, of any dimension N (P2 for a plane, P3 for a volume). A path p is parametrised by
# u ∈ [0, 1] and provides position(p, u), velocity(p, u) = d position/du, and breakpoints(p), the u where the velocity
# jumps (passed as tstops to the ODE solver). Loops have position(p, 0) == position(p, 1).

"""
    P2

A point (or a velocity) of a two-dimensional parameter space.
"""
const P2 = SVector{2,Float64}

"""
    P3

A point (or a velocity) of a three-dimensional parameter space.
"""
const P3 = SVector{3,Float64}

"""
    ParamPath

A path u ∈ [0, 1] ↦ point of the parameter space: subtypes implement `position(p, u)`, `velocity(p, u)` and, if the
velocity jumps, `breakpoints(p)`.
"""
abstract type ParamPath end

"""
    breakpoints(p::ParamPath)

The values of u where the velocity of `p` jumps (none by default).
"""
breakpoints(::ParamPath) = Float64[]

"""
    plane_basis(n::P3)

Orthonormal basis (e1, e2) of the plane normal to `n`, oriented so that e1 × e2 = n/‖n‖: a `Circle` in it runs
counterclockwise seen from the tip of n.
"""
function plane_basis(n::SVector{3})
    n̂ = normalize(P3(n))
    k = argmin(abs.(n̂))                               # the axis furthest from n
    a = P3(ntuple(i -> i == k ? 1.0 : 0.0, 3))
    e1 = normalize(n̂ × a)
    return e1, n̂ × e1
end

"""
    Circle(center::P2, ρ)
    Circle(center::P3, ρ, normal)
    Circle(center, ρ, e1, e2)

Circle of radius `ρ`, center + ρ(cos 2πu e1 + sin 2πu e2), starting at center + ρ e1. In a plane, e1, e2 are the
axes: counterclockwise, starting at the rightmost point. In 3D, the plane normal to `normal` (see `plane_basis`), or
any orthonormal pair (e1, e2).
"""
struct Circle{N} <: ParamPath
    center::SVector{N,Float64}
    ρ::Float64
    e1::SVector{N,Float64}
    e2::SVector{N,Float64}
    function Circle{N}(center, ρ, e1, e2) where N
        (norm(e1) ≈ 1 && norm(e2) ≈ 1 && abs(e1 ⋅ e2) < 1e-12) || throw(ArgumentError("e1, e2 must be orthonormal"))
        return new{N}(center, ρ, e1, e2)
    end
end
Circle(center::SVector{N}, ρ, e1::SVector{N}, e2::SVector{N}) where N = Circle{N}(center, ρ, e1, e2)
Circle(center::SVector{2}, ρ) = Circle(center, ρ, P2(1, 0), P2(0, 1))
Circle(center::SVector{3}, ρ, normal::SVector{3}) = Circle(center, ρ, plane_basis(normal)...)
position(c::Circle, u) = c.center + c.ρ*(cospi(2u)*c.e1 + sinpi(2u)*c.e2)
velocity(c::Circle, u) = 2π*c.ρ*(-sinpi(2u)*c.e1 + cospi(2u)*c.e2)

"""
    Polygon(vertices)

Closed polygon through `vertices` in order (a rectangle is 4 vertices), each edge covering an equal share of u.
"""
struct Polygon{N} <: ParamPath
    vertices::Vector{SVector{N,Float64}}
end
Polygon(vertices::AbstractVector{<:SVector{N}}) where N = Polygon{N}(vertices)
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
struct Segment{N} <: ParamPath
    a::SVector{N,Float64}
    b::SVector{N,Float64}
end
Segment(a::SVector{N}, b::SVector{N}) where N = Segment{N}(a, b)
position(p::Segment, u) = p.a + u*(p.b - p.a)
velocity(p::Segment, u) = p.b - p.a
