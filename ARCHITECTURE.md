# Architecture

A map of the package `NonHermitianQRM` (`src/`): what each file does, and how the pieces fit together to find
exceptional points (EPs) of Liouvillians. The methods and their validation are in `notes/monodromy_tracking.typ`; the
docstrings give the details of each function. This file names functions rather than line numbers, so that it does not
go stale.

## The idea in one paragraph

Near an EP of order two, two eigenvalues behave like λ_EP ± α√(z − z_EP). Followed continuously once around a loop
that encloses the EP, they **swap**; around no EP, or two EPs of the same pair, they come back to themselves. So
instead of landing on EPs, the package follows eigenpairs along paths in parameter space ("tracking") and reads the
permutation they undergo around closed loops ("monodromy"). The discriminant D = (λa − λb)², smooth through the EP
and vanishing linearly there, complements this: its winding around a loop counts EPs with their orientation (±1 each,
±2 for a diabolic point), and its zero can be fitted to localise an EP precisely.

## Layers

```
models.jl, master_equations.jl, mock.jl, qrm.jl     physics: models, Liouvillians, test beds
        │
paths.jl, affine.jl                                 parameter space: paths, affine families L(P)
        │
tracking.jl                                         follow one eigenpair along a path
        │
        ├── scan.jl          a grid: monodromy of every cell, for a window of eigenvalues
        ├── bisection.jl     a rectangle: localise one EP of one pair (bisection, greedy bracketing)
        ├── localise.jl      the EPs of a scan: decision tree per cell, then bracketing
        └── epline.jl        lines of EPs in three parameters
```

`NonHermitianQRM.jl` includes the files in that order and lists the exports per file.

## Physics (`models.jl`, `master_equations.jl`, `mock.jl`, `qrm.jl`)

- `models.jl`: the models (`Boson`, `BosonDimer`, `Duffing`, `Optomech`) with their approximations (`RWA_env`,
  `RWA_coupling`, …), and their analytic response: `D_R`, the characteristic polynomial `disc`, analytic `EP`s.
- `master_equations.jl`: `Lindbladian`, `Redfield`, `DressedLiouvillian`, modes and symmetry sectors.
- `mock.jl`: `MockLiouvillian`, a direct sum of small blocks with exactly known spectrum and planted singularities
  (EP2s, a DP, a real-axis EP line, a DP line): the test bed of every algorithm below.
- `qrm.jl`: the quantum Rabi model (`QRM`), its Jaynes–Cummings limit and closed forms (`jc_spectrum`, `jc_EP`),
  excitation-number sectors.

## Step 0: a family of Liouvillians (`paths.jl`, `affine.jl`)

1. A model is turned into a **family** L(P) of sparse matrices over a point P of parameter space.
   `AffineLiouvillian` stores L(P) = L₀ + Σₖ Pₖ Lₖ (built by finite differences of the model, checked to be affine),
   all terms on one sparsity pattern, so that `evaluate!` and `derivative!` write L(P) and its derivative along a
   direction in place. In practice L is first restricted to a symmetry sector (`excitation_sector`, parity).
2. `slice(A, P₀, e1, e2)` restricts a family to a plane; `Reparametrised` changes coordinates (log rates, polar
   coordinates, …); both keep the same interface, so every algorithm below runs on them unchanged.
3. **Paths** (`paths.jl`): `Circle`, `Polygon`, `Segment`, in 2D (`P2`) or 3D (`P3`), each giving `position(p, u)`,
   `velocity(p, u)` for u ∈ [0, 1] and the `breakpoints` where the velocity jumps.

## Step 1: following one eigenpair (`tracking.jl`)

`track(A, path, λ₀, r₀)` follows the eigenpair (λ₀, r₀) along a path:

1. **Starting eigenpairs** come from shift-invert: `eigenpairs_near` (nearest to targets), `pair_near` (the two
   nearest one shift).
2. **Predictor:** the eigenpair, normalised by c†r = 1, obeys an ODE whose right-hand side (`tangent`) is a solve with
   the bordered matrix B = [L − λ, −r; c†, 0], singular only at an EP. Integrated by Tsit5 at reltol 1e-6 by default
   (`alg=Vern7(lazy=false), reltol=1e-10, abstol=1e-12` for an accurate interpolation between steps).
3. **Corrector:** after every step, Newton (`correct!`) puts (λ, r) back on an exact eigenpair, and overwrites the
   saved step: the steps of a solution are exact, only the interpolation between them is approximate. c is reset
   when r rotates too far from it.
4. **Linear solves** (`StaleLU`): B is factorised once and the factorisation reused; each solve is GMRES
   preconditioned by that stale LU (`gmres=false`: iterative refinement), which keeps working near an EP. About two
   factorisations per tracking.

Output: the solution (steps, values, interpolation) and diagnostics. A swap is read by comparing λ at the end of a
loop with the eigenvalues at its start.

## Step 2: scanning a plane (`scan.jl`)

`tracked_scan(A, xs, ys, start_pairs)` scans a grid for a **window** of eigenvalues (the labels):

1. One diagonalisation at the lower-left node gives the labels (`start_pairs`).
2. `track_grid`: the labels are carried to every node along a spanning tree of edges (the bottom row, then every
   column upwards), and every other edge is tracked from its left node and matched to the labels of its right node
   (`track_edge`, `match_labels`). A label landing on no label is **lost**.
3. Per cell (`scan_cell`): the four edges compose into a **permutation** of the labels (`cell_permutation`), and the
   **winding** of D for each pair that closes, fixed or swapped (`cell_winding`, from the arg increments along the
   edges, `arg_increment`). Result: a `ScanCell` (swaps and their windings, windings of fixed pairs, cycles, lost
   labels, the permutation, the corner eigenpairs).
4. `refine_tracked`: flagged cells split into 2 × 2 sub-cells and rescanned from their corner, recursively (still
   available, superseded by step 4 for localisation).

## Step 3: localising one EP from a rectangle (`bisection.jl`)

For a rectangle around which a known pair swaps (exactly one EP of that pair inside), tracking only those two
eigenvalues:

1. `track_line`: both branches tracked along a segment (`TrackedLine`, read anywhere with `branch`; Vern7 at 1e-10
   by default, since the lines are read between steps).
2. `transport` and `rect_swaps`: an eigenvalue is followed around a rectangle by matching branches at the corners,
   every side lying on an earlier line (`find_line`): no side is tracked twice.
3. `ep_bisect`: split across the longer side, track the dividing line, keep the half that swaps.
4. `ep_bracket` (greedy): fit D affinely on the rectangle's sides (`disc_fit`), cut at x* ± δ and y* ± δ around its
   zero (`bracket_cuts`), keep the one of up to nine cells that swaps; the box shrinks from w to about w² per
   iteration while the bracket is never lost. It can start from the pair at the lower-left corner only.

## Step 4: localising the EPs of a scan (`localise.jl`)

`localise_eps(A, cells)` takes the cells of a coarse scan and handles each by orbits of its permutation
(`cell_action`):

1. A transposition with winding ±1 (one EP2 of that pair): `ep_bracket` on that pair alone, from its eigenvalues at
   the cell's corner (`bracket_pair`); the brackets run as parallel tasks.
2. An even winding ≠ 0 (a DP, or EP2s with a net index), a transposition with |w| ≥ 3, a cycle (EP2s sharing an
   eigenvalue): the cell is refined with the whole window until clean; at the depth limit, a DP (windings ±2 only) or
   unresolved.
3. Lost labels: reported (refined only with `refine_lost`).

Output: a `Localisation` (`eps` as `LocalisedEP`s with a certified box and the coalesced eigenvalue, `dps`,
`unresolved`, `lost`). Caveat: two EP2s of the same pair with opposite orientations give no swap and no winding, and
are invisible at a scale that does not separate them.

## Step 5: lines of EPs in three parameters (`epline.jl`)

Complex EP2s form curves in 3D (Re D = 0 ∩ Im D = 0). `track_ep_line(A, P₀, σ₀)` follows one from a seed (e.g. a
`LocalisedEP` of a 2D slice):

1. **Tangent** (`ep_tangent`): ∇Re D × ∇Im D, by central differences of D (`pair_disc`, one shift-invert each).
2. **Predictor:** P + h t.
3. **Corrector** (`ep_correct`), in the `NormalPlane` through the predicted point:
   - Newton on D (`ep_newton`, via `newton_in_plane`), which also gives a local model of the pair (`pair_model`);
   - a **check loop** (`check_loop`) of radius w/16 around Newton's point must swap the pair (`pair_swaps`); its pair
     is carried from the previous step's loop by tracking, so that the line follows the same pair by continuation;
   - otherwise a bracketing of the EP (`bracket_in_plane`: `ep_bracket`, then Newton).
4. **Step control:** h halved when the corrector fails or the tangent turns too much, increased on straight parts;
   stops at `bounds`, `nsteps`, a closed line, or `hmin`. Output: an `EPLine` (points, eigenvalues, tangents, status,
   counts).
5. `track_ep_lines`: several seeds, both directions, the half-lines as parallel tasks (four at a time); `join_halves`
   makes one line per seed.

## Typical workflow (see `notebooks/ep_lines.jl`)

1. Build the family (`AffineLiouvillian` of the model in a symmetry sector).
2. Scan a two-parameter slice (`slice`, `tracked_scan`) from one diagonalisation.
3. Localise its EPs (`localise_eps`).
4. Continue them in the third parameter (`track_ep_lines`), check with dense diagonalisation and a convergence test.

## Where else to look

- `test/runtests.jl`: the checks, each algorithm against closed forms (mock, boson dimer, Jaynes–Cummings).
- `notebooks/`: cell scripts (`#%%`) using the package. `ep_lines.jl` (EP lines, the workflow above),
  `integrators.jl` (integration strategies and scaling), `mock_liouvillian.jl` (scans of the mock, robustness),
  `qrm.jl` (Jaynes–Cummings tower), `linear_bosons.jl`, `nonlinear.jl` (optomechanics).
- `notes/monodromy_tracking.typ`: method, validation, limitations and future work.
- `functions_QRM.jl` (repository root): the coworker's discriminant-based EP search, separate from the package.
