# Systematic search for EPs on a rectangular grid of the parameter plane, by eigenpair tracking: the eigenpairs of a
# window are given once, at the lower-left node, and followed along every grid edge with `track` (no diagonalisation
# along the way). Around each cell, the four edges compose into a permutation of the eigenvalues at its lower-left
# corner:
#   - a transposition (a b) ⇔ an odd number of EP2 of that pair inside the cell;
#   - for a pair returning to itself, the winding of D = (λa - λb)² around the cell counts its EP2 with orientation
#     (±1 each) and its DPs (±2): an identity with a nonzero winding is an even number of EP2 or a DP, told apart by
#     refinement (EP2 separate into swapping sub-cells);
#   - longer cycles: EPs involving three eigenvalues or more.
# Cells showing any of these are refined (2 × 2 sub-cells) from the eigenpairs carried to their corner.
#
# Labels. The eigenpairs at the start node are labels 1…n. They are carried to every node along a spanning tree (the
# bottom row, then every column upwards), so tree edges map labels to themselves. Every other (horizontal) edge is
# tracked from its left node, and its end values are matched to the labels of its right node. A tracked eigenvalue that
# ends on none of the labels (an EP with an eigenvalue outside the window, or a failed integration) is "lost": the
# cells around it say nothing about that label.

"""
    NodeLabel

The eigenpair carrying a label at a grid node, or `nothing` if the label was lost on the way there.
"""
const NodeLabel = Union{Nothing,Eigenpair}

"""
    TrackedEdge

The labelled eigenpairs tracked along one grid edge. Per start label: the solver's steps `t` along the edge (empty if
the tracking failed), the eigenvalue `λ` at those steps, and the final eigenpair `ends` (or `nothing`).
"""
struct TrackedEdge
    t::Vector{Vector{Float64}}
    λ::Vector{Vector{ComplexF64}}
    ends::Vector{NodeLabel}
end

"""
    track_edge(A, a, b, pairs; threaded=false, kwargs...)

Tracks the eigenpairs `pairs` (`nothing` for a lost label) from `a` to `b`, in parallel over the eigenpairs if
`threaded`. A tracking that throws or does not reach `b` loses its label.
"""
function track_edge(A, a::P2, b::P2, pairs; threaded=false, kwargs...)
    n = length(pairs)
    t = [Float64[] for _ ∈ 1:n]
    λ = [ComplexF64[] for _ ∈ 1:n]
    ends = Vector{NodeLabel}(nothing, n)
    function track_label(k)
        pairs[k] === nothing && return
        sol = try
            track(A, Segment(a, b), pairs[k]...; kwargs...)[1]
        catch
            nothing
        end
        (sol === nothing || !tracking_succeeded(sol)) && return
        t[k], λ[k], ends[k] = sol.t, [y[end] for y ∈ sol.u], (sol.u[end][end], sol.u[end][1:end-1])
    end
    threaded ? Threads.@threads(for k ∈ 1:n; track_label(k); end) : foreach(track_label, 1:n)
    return TrackedEdge(t, λ, ends)
end

"""
    match_labels(ends, targets; tol=1e-7)

For each tracked end eigenpair, the label (index in `targets`) it lands on: the nearest target eigenvalue, if within
`tol` relative to 1 + |λ|, else 0.
"""
function match_labels(ends, targets; tol=1e-7)
    map(ends) do e
        e === nothing && return 0
        d = [tg === nothing ? Inf : abs(tg[1] - e[1]) for tg ∈ targets]
        j = argmin(d)
        return d[j] ≤ tol*(1 + abs(e[1])) ? j : 0
    end
end

"""
    arg_increment(e::TrackedEdge, a, b)

Change of arg (λa - λb)² along the edge, for its start labels `a` and `b`. Between two points of the union of their
solver steps both eigenvalues are interpolated linearly, so d = λa - λb moves on a straight segment, whose change of
arg seen from 0 is exactly angle(d₁/d₀) ∈ (-π, π): the change for D = d² is twice the sum. (Taking angle(D₁/D₀)
instead aliases when one step turns D by more than π, e.g. the solver crossing a DP's neighbourhood in one step.)
"""
function arg_increment(e::TrackedEdge, a, b)
    function interp(t, v, x)
        k = clamp(searchsortedlast(t, x), 1, length(t) - 1)
        w = (x - t[k])/(t[k + 1] - t[k])
        return (1 - w)*v[k] + w*v[k + 1]
    end
    ss = sort(unique([e.t[a]; e.t[b]]))
    d = [interp(e.t[a], e.λ[a], x) - interp(e.t[b], e.λ[b], x) for x ∈ ss]
    return 2sum(angle(d[k + 1]/d[k]) for k ∈ 1:length(d) - 1)
end

"""
    TrackedGrid

All tracked edges of a grid `xs × ys`, with the labelled eigenpairs at every node.
- `nodes[i, j]`: eigenpairs of the labels at node (xs[i], ys[j]).
- `horizontal[i, j]`: edge from node (i, j) to (i + 1, j); `vertical[i, j]`: from (i, j) to (i, j + 1).
- `to_right[i, j][a]`: label at node (i + 1, j) reached by label `a` of node (i, j) along `horizontal[i, j]`
  (0 if none). Vertical edges are all on the spanning tree: they map labels to themselves.
"""
struct TrackedGrid
    xs::Vector{Float64}
    ys::Vector{Float64}
    nodes::Matrix{Vector{NodeLabel}}
    horizontal::Matrix{TrackedEdge}
    vertical::Matrix{TrackedEdge}
    to_right::Matrix{Vector{Int}}
end

"""
    track_grid(A, xs, ys, start_pairs; kwargs...)

Tracks the eigenpairs `start_pairs`, given at (xs[1], ys[1]), along every edge of the grid `xs × ys`: first the
spanning tree, which carries the labels to every node, then the other edges. `kwargs` are passed to `track`.
"""
function track_grid(A, xs, ys, start_pairs; kwargs...)
    nx, ny = length(xs), length(ys)
    node(i, j) = P2(xs[i], ys[j])
    nodes = Matrix{Vector{NodeLabel}}(undef, nx, ny)
    nodes[1, 1] = NodeLabel[start_pairs...]
    horizontal = Matrix{TrackedEdge}(undef, nx - 1, ny)
    vertical = Matrix{TrackedEdge}(undef, nx, ny - 1)
    for i ∈ 1:nx - 1                                     # tree, bottom row: in sequence, eigenpairs in parallel
        horizontal[i, 1] = track_edge(A, node(i, 1), node(i + 1, 1), nodes[i, 1]; threaded=true, kwargs...)
        nodes[i + 1, 1] = horizontal[i, 1].ends
    end
    Threads.@threads for i ∈ 1:nx                        # tree, every column upwards: columns in parallel
        for j ∈ 1:ny - 1
            vertical[i, j] = track_edge(A, node(i, j), node(i, j + 1), nodes[i, j]; kwargs...)
            nodes[i, j + 1] = vertical[i, j].ends
        end
    end
    others = [(i, j) for i ∈ 1:nx - 1 for j ∈ 2:ny]     # the other horizontal edges, in parallel
    Threads.@threads for k ∈ eachindex(others)
        i, j = others[k]
        horizontal[i, j] = track_edge(A, node(i, j), node(i + 1, j), nodes[i, j]; kwargs...)
    end
    to_right = [match_labels(horizontal[i, j].ends, nodes[i + 1, j]) for i ∈ 1:nx - 1, j ∈ 1:ny]
    return TrackedGrid(collect(xs), collect(ys), nodes, horizontal, vertical, to_right)
end

"""
    cell_permutation(g::TrackedGrid, i, j)

Monodromy of the labels of node (i, j) once around the cell (i, j), counterclockwise: `(σ, lost)`, with σ[a] the label
on which `a` returns (0 if lost) and `lost[a]` whether label `a` was lost on the way. Along the bottom edge `a` reaches
label b at the lower-right corner, carried up the right edge to the upper-right corner; the label at the upper-left
corner whose top edge ends on b is σ[a], carried down the left edge (both vertical edges are on the tree).
"""
function cell_permutation(g::TrackedGrid, i, j)
    n = length(g.nodes[i, j])
    σ, lost = zeros(Int, n), falses(n)
    for a ∈ 1:n
        b = g.to_right[i, j][a]
        l = b == 0 ? nothing : findfirst(==(b), g.to_right[i, j + 1])
        if l === nothing || g.nodes[i + 1, j + 1][b] === nothing || g.nodes[i, j + 1][l] === nothing
            lost[a] = true
        else
            σ[a] = l
        end
    end
    return σ, lost
end

"""
    cell_winding(g::TrackedGrid, i, j, σ, a, k)

Winding number of (λa - λk)² once around the cell (i, j), for two labels `a`, `k` of node (i, j) that return to
themselves: the arg increments along the bottom and right edges, minus those along the top and left edges (traversed
backwards).
"""
function cell_winding(g::TrackedGrid, i, j, σ, a, k)
    right = g.to_right[i, j]                             # labels at the lower-right corner
    total = arg_increment(g.horizontal[i, j], a, k) + arg_increment(g.vertical[i + 1, j], right[a], right[k]) -
            arg_increment(g.horizontal[i, j + 1], σ[a], σ[k]) - arg_increment(g.vertical[i, j], σ[a], σ[k])
    return round(Int, total/2π)
end

"""
    ScanCell

Result of the scan for one grid cell.
- `rect`: `(x0, x1, y0, y1)`.
- `swaps`: the pairs (λa, λb) exchanged around the cell: an odd number of their EP2 inside.
- `windings`: the pairs (λa, λb, w) returning to themselves with a nonzero winding w of (λa - λb)²: an even number of
  EP2, or a DP (w = ±2).
- `cycles`: number of labels in cycles of length ≥ 3.
- `lost`: number of labels lost around the cell.
- `corner`: the labelled eigenpairs at the lower-left corner, to refine the cell without diagonalising.
Eigenvalues are given at the lower-left corner.
"""
struct ScanCell
    rect::NTuple{4,Float64}
    swaps::Vector{NTuple{2,ComplexF64}}
    windings::Vector{Tuple{ComplexF64,ComplexF64,Int}}
    cycles::Int
    lost::Int
    corner::Vector{NodeLabel}
end

"""
    flagged(c::ScanCell)

Whether the cell contains a singularity: a swap, a nonzero winding or a longer cycle.
"""
flagged(c::ScanCell) = !isempty(c.swaps) || !isempty(c.windings) || c.cycles > 0

"""
    scan_cell(g::TrackedGrid, i, j)

The `ScanCell` of the cell (i, j) of the grid.
"""
function scan_cell(g::TrackedGrid, i, j)
    σ, lost = cell_permutation(g, i, j)
    n = length(σ)
    λ = [p === nothing ? NaN + 0im : p[1] for p ∈ g.nodes[i, j]]
    swaps = [(λ[a], λ[σ[a]]) for a ∈ 1:n if !lost[a] && σ[a] > a && σ[σ[a]] == a]
    returns(a) = !lost[a] && σ[a] == a
    windings = Tuple{ComplexF64,ComplexF64,Int}[]
    for a ∈ 1:n, k ∈ a+1:n
        returns(a) && returns(k) || continue
        w = cell_winding(g, i, j, σ, a, k)
        w == 0 || push!(windings, (λ[a], λ[k], w))
    end
    in_cycle(a) = !lost[a] && σ[a] != a && !lost[σ[a]] && σ[σ[a]] != a
    rect = (g.xs[i], g.xs[i + 1], g.ys[j], g.ys[j + 1])
    return ScanCell(rect, swaps, windings, count(in_cycle, 1:n), count(lost), g.nodes[i, j])
end

"""
    tracked_scan(A, xs, ys, start_pairs; kwargs...)

Scan of the grid `xs × ys` from the eigenpairs `start_pairs` at (xs[1], ys[1]): one `ScanCell` per cell, row by row
from the bottom. `kwargs` are passed to `track`.
"""
function tracked_scan(A, xs, ys, start_pairs; kwargs...)
    g = track_grid(A, xs, ys, start_pairs; kwargs...)
    return [scan_cell(g, i, j) for j ∈ 1:length(ys) - 1 for i ∈ 1:length(xs) - 1]
end

"""
    refine_tracked(A, cells, depth; kwargs...)

Recursive refinement of the flagged cells (2 × 2 sub-cells per level, `depth` levels), each sub-grid starting from the
eigenpairs at its cell's corner: no diagonalisation. The flagged cells are refined in parallel (and each sub-scan is
itself threaded). Returns the flagged cells of the last level ("leaves"), in the order of the cells.
"""
function refine_tracked(A, cells, depth; kwargs...)
    depth == 0 && return filter(flagged, cells)
    todo = filter(flagged, cells)
    leaves = Vector{Vector{ScanCell}}(undef, length(todo))
    Threads.@threads for k ∈ eachindex(todo)
        x0, x1, y0, y1 = todo[k].rect
        sub = tracked_scan(A, range(x0, x1, 3), range(y0, y1, 3), todo[k].corner; kwargs...)
        leaves[k] = refine_tracked(A, sub, depth - 1; kwargs...)
    end
    return reduce(vcat, leaves; init=ScanCell[])
end
