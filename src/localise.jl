# Localisation of the EPs found by a scan. A coarse tracked scan (`tracked_scan`, the whole window of eigenvalues)
# finds and classifies every cell; each cell is then handled by orbits of its permutation (the labels of different
# orbits are independent: an EP between them would have merged the orbits):
#   - a transposition (a b) with winding w_ab = ±1: one EP2 of that pair inside (up to an invisible pair of opposite
#     orientations) → `ep_bracket` on that pair alone, from its eigenvalues at the cell's corner;
#   - a pair returning to itself with an even winding w ≠ 0 (a DP, or EP2s with a net index), a transposition with
#     |w| ≥ 3 (several EP2s), a cycle of three labels or more (EP2s sharing an eigenvalue) → the cell is refined (2 × 2
#     sub-cells, the whole window, from the eigenpairs at its corner), and its sub-cells handled the same way; at the
#     depth limit, a cell left with only ±2 windings is classified as a DP, any other as unresolved;
#   - lost labels say nothing about their orbit: the cell is reported, and refined only on request.
# A cell that needs refinement is refined as a whole, its clean transpositions bracketed in the sub-cells (no EP is
# bracketed twice). Refinement should be rare in physical models: the common case goes from the coarse scan straight
# to the bracketing, which converges quadratically (`ep_bracket`).

"""
    LocalisedEP

An EP2 localised by `localise_eps`: the certified `rect` (a rectangle around which the pair swaps: the EP is inside),
the coalesced eigenvalue `λ` (midpoint of the pair at the centre of rect), the `pair` (its two eigenvalues at the
lower-left corner of the scan cell it was found in), and that `cell`.
"""
struct LocalisedEP
    rect::NTuple{4,Float64}
    λ::ComplexF64
    pair::NTuple{2,ComplexF64}
    cell::NTuple{4,Float64}
end

"""
    Localisation

The result of `localise_eps`: the localised EP2s `eps`, the cells classified as `dps` (a winding ±2 persisting down to
the depth limit: a DP, or two EP2s of the same orientation closer than the finest cell), the `unresolved` cells (cycles
or windings |w| ≥ 3 at the depth limit, or whose bracketing failed), and the cells with `lost` labels.
"""
struct Localisation
    eps::Vector{LocalisedEP}
    dps::Vector{ScanCell}
    unresolved::Vector{ScanCell}
    lost::Vector{ScanCell}
end

# what a cell needs: :empty, :bracket (only clean transpositions, |w| = 1), or :refine (an even winding, a
# transposition with |w| ≥ 3, or a cycle)
function cell_action(c::ScanCell)
    flagged(c) || return :empty
    (c.cycles > 0 || !isempty(c.windings) || any(w -> abs(w) != 1, c.swap_windings)) && return :refine
    return :bracket
end

# a cell at the depth limit that still needs refinement: a DP if all it has are windings ±2
is_dp_like(c::ScanCell) = c.cycles == 0 && isempty(c.swaps) && all(w -> abs(w[3]) == 2, c.windings)

"""
    localise_eps(A, cells; max_depth=8, tol=1e-9, refine_lost=false, ntasks=4, kwargs...)

Localises the EP2s in the `cells` of a tracked scan of the family `A` (`tracked_scan`): each cell is handled by orbits
of its permutation (see the head of localise.jl). Clean transpositions (winding ±1) are bracketed on their pair alone
(`ep_bracket` down to a rectangle of size `tol`, from the pair's eigenvalues at the cell's corner); cells with even
windings, |w| ≥ 3 or cycles are refined (2 × 2 sub-cells, the whole window), at most `max_depth` levels. Cells with
lost labels are reported in `lost`, and refined like the others if `refine_lost`. The brackets are independent and run
as parallel tasks, at most `ntasks` at a time. `kwargs` are passed to `track` (scan and bracketing). Returns a
`Localisation`.
"""
function localise_eps(A, cells; max_depth=8, tol=1e-9, refine_lost=false, ntasks=4, kwargs...)
    eps, dps, unresolved, lost = LocalisedEP[], ScanCell[], ScanCell[], ScanCell[]
    todo = [(c, 0) for c ∈ cells if flagged(c) || c.lost > 0]
    while !isempty(todo)
        next = Tuple{ScanCell,Int}[]
        jobs = Tuple{ScanCell,NTuple{2,ComplexF64}}[]           # the pairs to bracket at this level
        for (c, depth) ∈ todo
            c.lost > 0 && depth == 0 && push!(lost, c)
            action = c.lost > 0 && refine_lost ? :refine : cell_action(c)
            if action == :bracket
                append!(jobs, [(c, pair) for pair ∈ c.swaps])
            elseif action == :refine
                if depth < max_depth
                    x0, x1, y0, y1 = c.rect
                    sub = tracked_scan(A, range(x0, x1, 3), range(y0, y1, 3), c.corner; kwargs...)
                    append!(next, [(s, depth + 1) for s ∈ sub if flagged(s) || (refine_lost && s.lost > 0)])
                else
                    push!(is_dp_like(c) ? dps : unresolved, c)
                end
            end
        end
        # the brackets are independent: parallel tasks, at most `ntasks` at a time (each tracks its lines in parallel)
        # BLAS on one thread meanwhile, so that the sparse factorisations of the tasks do not oversubscribe the cores
        slots = Base.Semaphore(ntasks)
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        results = try
            fetch.([Threads.@spawn(Base.acquire(() -> bracket_pair(A, c, pair; tol, kwargs...), slots))
                    for (c, pair) ∈ jobs])
        finally
            BLAS.set_num_threads(blas_threads)
        end
        for ((c, _), ep) ∈ zip(jobs, results)
            ep === nothing ? push!(unresolved, c) : push!(eps, ep)
        end
        todo = next
    end
    return Localisation(eps, dps, unresolved, lost)
end

# `ep_bracket` on one swapped pair of a scan cell, from its eigenvalues at the cell's corner; nothing if it fails
function bracket_pair(A, c::ScanCell, (λa, λb); tol, kwargs...)
    rect, _, lines = try
        ep_bracket(A, c.rect, [λa, λb]; tol, kwargs...)
    catch err
        err isa InterruptException && rethrow()
        @warn "bracketing the pair ($λa, $λb) of the cell $(c.rect) failed: $(sprint(showerror, err))"
        return nothing
    end
    x0, x1, y0, y1 = rect
    # the pair at the final rectangle's corner, read on its bottom side, then its midpoint at the centre
    l, s, _ = find_line(lines, P2(x0, y0), P2(x1, y0))
    _, λm = pair_disc(A, P2((x0 + x1)/2, (y0 + y1)/2), (branch(l, 1, s) + branch(l, 2, s))/2)
    return LocalisedEP(rect, λm, (λa, λb), c.rect)
end
