#import "@preview/physica:0.9.8"
#import "@local/byzantine-notes:0.3.2" : setup, cetz-style


#let main(colors: color, cetz-style:()) = [

  = Idea
  Near an EP of order two, two eigenvalues behave as $λ_± ≈ λ_"EP" ± α √(z - z_"EP")$, with $z$ a complex combination of the parameters. Following both eigenvalues continuously around a closed loop enclosing the EP, the square root changes sign: the two eigenvalues _swap_, and each one returns to itself only after two loops. A loop enclosing no EP, or two EPs of the same pair, gives no swap. Detecting a permutation after one loop therefore detects an EP inside it, without having to land on it.

  *Real or complex parameters.* In a basis of Hermitian operators the Liouvillian is a real matrix: its eigenvalues are real or come in complex-conjugate pairs.
  - Two real eigenvalues merging into a conjugate pair (critical damping, $Q = 1\/2$) is codimension 1 in real parameters: these EPs form lines, and a loop must use a complexified parameter, $L(s) = L_0 + s L_1$ with $s ∈ ℂ$.
  - Two complex, non-conjugate eigenvalues merging (the finite-frequency EP of the dimer) is codimension 2: these EPs are isolated points of a real two-parameter plane, and a loop in real parameters encircles them.

  *Diabolic points.* At a DP the two eigenvalues meet without their eigenvectors merging, and $λ_+ - λ_-$ is linear in the distance rather than a square root: a loop around a DP gives no swap, like a loop around no singularity. The two are told apart by the winding of the discriminant $D = (λ_a - λ_b)^2$ around the loop: $± 1$ for each EP2 (with its orientation), $± 2$ for a DP, $0$ for no singularity. An identity with winding $± 2$ is therefore a DP _or_ two EP2 of the same orientation; only a smaller loop, which separates the two EP2, tells them apart (see the systematic scan).

  The code is the package `NonHermitianQRM` (`src/`): paths (`paths.jl`), affine Liouvillians (`affine.jl`), tracking of one eigenpair (`tracking.jl`), bisection and bracketing (`bisection.jl`), systematic scan (`scan.jl`), lines of EPs in three parameters (`epline.jl`), and the mock Liouvillian used as a test bed (`mock.jl`). The notebooks are in `notebooks/` (in particular `integrators.jl` for the integration strategies and `ep_lines.jl` for the EP lines) and the checks in `test/runtests.jl`.

  = Ingredients
  *Paths.* A path in parameter space is given by `position(p, u)`, `velocity(p, u)` $= dif "position"\/dif u$ for $u ∈ [0, 1]$, and the `breakpoints` where the velocity jumps (the corners of a `Polygon`), passed to the ODE solver as `tstops`. Loops are `Circle`s and `Polygon`s; open `Segment`s are the edges of the bisection and of the scan. For the dimer, the plane is $(δ ω, δ γ)$ with $ω_(a,b) = ω_0 ± δ ω$, $γ_(a,b) = γ_0 ± δ γ$, written as a function δ ↦ model with `setproperties` where the loop is defined.

  *Affine Liouvillian.* The Lindblad Liouvillian is linear in the frequencies and the rates, so along the path
  $
    L(u) &= L_0 + δ ω(u) L_ω + δ γ(u) L_γ, \
    L'(u) &= δ ω'(u) L_ω + δ γ'(u) L_γ,
  $
  where primes denote derivatives in $u$,
  with $L_ω = -i[hat(a)^† hat(a) - hat(b)^† hat(b), dot]$ and $L_γ = 2(1 + n_B)(cal(D)[hat(a)] - cal(D)[hat(b)]) + 2 n_B (cal(D)[hat(a)^†] - cal(D)[hat(b)^†])$. `AffineLiouvillian` builds this decomposition from any function δ ↦ L(δ), by $L_k = (L(h e_k) - L_0)\/h$. It checks that the family is affine by comparing $L$ with the decomposition at $δ = 2h(1, …, 1)$, a point not used to build it: at $h(1, …, 1)$ a pure $δ_k^2$ term would be reproduced exactly, and with one parameter the check could never fail. All terms are stored on one sparsity pattern (the union of theirs and the diagonal), so that $L(u)$ and $L'(u)$ are written in place by combining the stored values (`evaluate!`, `derivative!`): nothing is allocated or rebuilt along the path.

  *Starting point.* At $u = 0$ the eigenpairs $(λ, r)$ nearest to target values (for the dimer, $-i ω$ from the roots of `disc`) are obtained by shift-invert (`eigenpairs_near`; `pair_near` for a close pair, from a single shift). The systematic scan starts instead from one dense diagonalisation.

  = Tracking equations
  An eigenpair is defined only up to the scale and phase of $r$. They are fixed by a normalisation $c^† r = 1$ with a fixed vector $c$, initially $c = r_0 \/ (r_0^† r_0)$. The eigenpair then solves the $N + 1$ equations
  $
    F(r, λ) = vec((L - λ) r, c^† r - 1) = 0 .
  $
  Its Jacobian is the bordered matrix
  $
    B = mat(L - λ I, -r; c^†, 0),
  $
  which is non-singular as long as $λ$ is simple and $c^† r ≠ 0$, and becomes singular at an EP, where the two eigenvectors become parallel. Numerically $"cond"(B) ∝ d^(-1\/2)$ at a distance $d$ from the dimer EP, the signature of an EP of order two.

  *Predictor.* Differentiating $F = 0$ along the path,
  $
    B vec(r', λ') = vec(-L' r, 0)
  $
  (`tangent`). This ODE for $[r; λ]$ is integrated over $u ∈ [0, 1]$ with an adaptive solver (`track`: Tsit5, tolerances $10^(-6)$, $10^(-8)$; see Integration strategies). As a check, $λ' = (l^† L' r)\/(l^† r)$ from first-order perturbation theory, with $l$ the left eigenvector.

  *Corrector.* After each accepted step, Newton iterations at fixed $u$ pull the state back onto an exact eigenpair (`correct!`):
  $
    B vec(δ r, δ λ) = vec(-(L - λ) r, 1 - c^† r),
  $
  then $r ← r + δ r$, $λ ← λ + δ λ$.
  The neglected term $-δ λ space δ r$ is second order, so the error is squared at each iteration. Newton converges to the nearest eigenpair: it keeps the state accurate but cannot detect a jump to the other eigenvalue, which only small enough steps prevent. The corrected eigenpair also replaces the value the solver saved at that step, so the steps of a solution are exact eigenpairs (to $10^(-12)$); between steps, the dense output of the solver is accurate to about the integration tolerance only, and less near an EP.

  *Reset of $c$.* Since Newton keeps $c^† r = 1$, a rotation of $r$ away from $c$ shows up as a growth of $‖r‖$, with $cos angle (c, r) = 1\/(‖c‖ ‖r‖)$. When this falls below $0.1$, $c$ is reset to $r\/(r^† r)$, which keeps $c^† r = 1$ and leaves $r$ unchanged. Without it, $‖r‖$ reached $10^7$ around a rectangle and Newton needed eight times more iterations. Since $c$ is the ODE parameter, the solver must not interpolate lazily: Vern7 needs `Vern7(lazy=false)`, as its lazy stages would be evaluated after the solve with the latest $c$; Tsit5's interpolant only uses the stages of the step.

  *Detection.* After one loop, $λ(1)$ is compared with the eigenvalues at $u = 0$: landing on the other one means a swap.

  = Linear solves
  *Stale LU and refinement.* $B$ depends on $(u, λ, r)$ and must be solved at every right-hand-side evaluation (6 per step for Tsit5, 7 to 16 for Vern7) and every Newton iteration. A sparse LU of $B$ dominates the cost. Since $B$ changes little along the path, the factorisation $F$ of an earlier $B_"old"$ is reused, and the solution of $B x = b$ is refined (`StaleLU`, `solve_bordered!`):
  $
    x_0 = F \\ b, quad x_(k+1) = x_k + F \\ (b - B x_k),
  $
  with $B x$ computed without assembling $B$. The error is multiplied by $I - B_"old"^(-1) B$ at each iteration, a factor $≈ 0.07$ over a typical step, at the cost of one sparse product and one triangular solve ($≈ 1\/50$ of a factorisation). The solver refactorises when refinement does not reach $‖b - B x‖ ≤ 10^(-12) ‖b‖$ within 20 iterations or stops decreasing, and after any solve that needed more than 8 iterations. The stale LU divides the tracking times by 2 to 5 compared with a fresh factorisation at every solve, with identical results.

  *GMRES (default).* Near an EP, $B_"old"^(-1)$ grows, the factor $‖I - B_"old"^(-1) B‖$ exceeds one and the refinement diverges: it refactorised every few solves, 50 to 250 factorisations per loop around an EP (770 for a loop of radius $2 × 10^(-4)$ around one of the mock's close EP2). GMRES preconditioned by the same $F$ (right preconditioning, $x = F^(-1) z$, from $x_0 = F^(-1) b$; Givens rotations, workspace in the `StaleLU`) converges in a few iterations where the refinement diverges: about 2 factorisations per tracking, whatever the distance to the EP. `track(…; gmres=false)` returns to the refinement.

  *In-place assembly.* Each tracked eigenpair owns its `StaleLU`, which holds $B$ (a `BorderedPattern`, filled in place) and the work vectors, so that nothing is allocated per step; refactorisations on the same pattern reuse the symbolic analysis of the LU (`lu!`). The pattern of $B$ is that of $L$, bordered by $-r$ and $c^†$ _on their support only_, with no stored corner entry. This matters: the eigenvectors of a Liouvillian with a symmetry vanish exactly outside their sector, and stay so along a path, but a border storing those zeros (or a stored zero in the corner) increases the fill and pivoting of the LU, and made the factorisations 1.4 times slower for the 625-state dimer. The pattern is rebuilt if $r$ or $c$ ever leave the support. Time and allocations for one loop around one EP (one thread, best of five):
  #table(
    columns: 4,
    stroke: white,
    [*System*], [*$N$*], [*Before*], [*In place*],
    [mock Liouvillian], [18], [11.0 ms, 29 MB], [5.7 ms, 4.4 MB],
    [boson dimer, $n_"Fock" = 5$], [625], [1.05 s, 1.5 GB], [0.88 s, 0.45 GB],
    [boson dimer, $n_"Fock" = 8$], [4096], [22.0 s, 14 GB], [19.9 s, 7.3 GB],
  )
  For large $N$ the cost is now the numeric factorisation itself (70 % of the time for $N = 625$), which still allocates its factors (2.5 MB per LU for $N = 625$). (These timings predate the GMRES solves, which removed most factorisations.)

  = Integration strategies
  The corrector puts every step back on an exact eigenpair, so the integrator only has to keep the state in Newton's basin, on the right branch: its tolerance controls the step size, not the accuracy of the result. Variants compared on eight workloads (loops around EPs, a short line near an EP, a long line through the mock crossing its real-axis EP line, a 625-state dimer loop, two scans), with correctness checked against closed forms (`notebooks/integrators.jl`; 23 variants in all):
  #table(
    columns: 5,
    stroke: white,
    [*Variant*], [*QRM check loop*], [*Dimer loop ($N = 625$)*], [*Mock scan*], [*JC scan*],
    [Vern7 $10^(-10)$, refinement (former default)], [0.22 s], [1.30 s], [0.97 s], [3.0 s],
    [Vern7 $10^(-6)$, refinement], [0.14 s], [0.80 s], [0.61 s], [2.0 s],
    [Tsit5 $10^(-6)$, refinement], [0.067 s], [0.35 s], [0.43 s], [1.3 s],
    [Tsit5 $10^(-4)$, refinement], [0.051 s], [0.28 s], [wrong], [wrong],
    [Euler–Newton, refinement], [0.045 s], [0.27 s], [0.43 s], [1.3 s],
    [*Tsit5 $10^(-6)$, GMRES (default)*], [0.032 s], [0.16 s], [0.32 s], [1.0 s],
    [Euler–Newton, GMRES], [0.021 s], [0.11 s], [0.34 s], [1.0 s],
  )
  - The scans became wrong at $10^(-4)$ (Tsit5 and BS3): the labels and windings need steps that resolve the eigenvalues. $10^(-6)$ keeps a decade of margin ($10^(-5)$ still passed).
  - `EulerNewton` is a hand-written continuation, as in AUTO or MATCONT: predictor along the tangent (with a quadratic term from the last two tangents), Newton corrector, and a step accepted if the corrector's displacement is at most $α = 0.1$ times the predictor's. This ratio grows like the curvature of the path times $h$, so it resolves the square-root behaviour near an EP without an error tolerance. It was 1.3 to 1.5 times faster than Tsit5 on loops, but needs special care where a path crosses an EP (the eigenvalue goes like $sqrt(u - u_0)$): it leaps over it and restarts from a shift-invert. It is kept as an option; the default is the library solver.
  - *Scaling.* Full QRM, odd sector, $N = 72$ to $1568$: a shift-invert grows like $N^(1.5)$ at large $N$ (the Liouvillian's connectivity is nearly two-dimensional), and a loop with the default costs about 11 shift-inverts at every size, against 60 to 130 with the former default: the gain grows with $N$, from 6 at $N = 72$ to 12 at $N = 1568$.
  - The lines of the bisection and of the bracketing (`track_line`) are read between steps (at the corners of rectangles inside a line, and by the fit of $D$ below), so they keep an accurate dense output: Vern7 at $10^(-10)$, with GMRES.

  = Validation on the boson dimer
  Dimer with `RWA_env`, $ω_0 = 1$, $γ_0 = 0.1$, $g = 0.01$, EPs at $(δ ω, δ γ) = (0, ± g\/2)$; $n_"Fock" = 5$ per mode, $N = 625$. Both modes are tracked and compared with $-i ω$ from `disc` along the whole loop:
  #table(
    columns: 3,
    stroke: white,
    [*Loop*], [*After one loop*], [*$max|λ - λ_"disc"|$*],
    [circle around one EP], [swap], [$2.6 × 10^(-13)$],
    [rectangle around one EP], [swap], [$7 × 10^(-11)$],
    [circle around no EP], [no swap], [$9 × 10^(-14)$],
    [circle around both EPs], [no swap], [$7 × 10^(-14)$],
    [circle passing $10^(-4)$ from the EP], [swap], [$1.8 × 10^(-12)$],
  )

  = Locating an EP by bisection
  From a rectangle that swaps the pair, `ep_bisect` splits it across its longer side, tracks both eigenvalues along the dividing line, and keeps the half that swaps (exactly one must). Every side of every rectangle lies on a tracked line (`TrackedLine`, one ODE solution per branch): following an eigenvalue along a side means picking the branch that matches it at the start (`transport`, which refuses an ambiguous match, closer than a quarter of the gap) and reading that branch at the end. Splitting a rectangle therefore costs one new line, the other sides being parts of earlier lines. A dividing line passing too close to the EP fails, and is moved by a tenth of the side. The two eigenvalues of a line start from shift-inverts offset by at most 1 % of their separation: a fixed offset made both converge to the same eigenvalue within $10^(-6)$ of an EP.

  *Greedy bracketing.* Bisection only shrinks the area by 4 per two lines. `ep_bracket` uses the discriminant instead: $D = (λ_a - λ_b)^2$ is smooth and vanishes linearly at an EP2, and it is known along every tracked line. An affine fit of $D$ on the sides of the rectangle (`disc_fit`, the sampled eigenpairs polished by Newton) gives its zero $P^*$, with an error estimate $ε$ (largest residual over the smallest singular value of the fit's Jacobian). The lines $x = x^* ± δ$, $y = y^* ± δ$ with $δ = 2 ε$ cut the rectangle into up to nine cells, every side on a tracked line; exactly one swaps, and is kept, so the bracket is never lost. When the fit is right, the rectangle shrinks from $w$ to about $w^2$; otherwise an outer cell is kept and the next fit is made there. On the mock, a box of $5 × 10^(-10)$ in 6 iterations (24 lines), against $4 × 10^(-4)$ after 16 bisection levels (20 lines), in the same time.

  *Driven optomechanics.* In the displaced frame of the linearised steady state ($n_"Fock" = 6$ per mode, $N = 1296$), in the plane $(Δ, F\/F_"EP")$ where $L$ is affine, a loop of radius 0.05 around the linearised EP swaps the pair. Sixteen bisection levels (20 tracked lines, 100 s) locate the EP of the full model at $Δ = -1.02196 ± 0.0002$, $F = 0.77247 ± 0.00015$, shifted by about 1 % from the linearised EP. Direct loops of twice the final box size confirm it (swap around the box, none next to it), and the gap of the pair at the box centres scales as $sqrt("box size")$.

  = Systematic scan
  Without a starting rectangle, a grid of the parameter plane is scanned (`tracked_scan`), and the cells whose loop shows a singularity are refined.

  *One diagonalisation.* The spectrum is computed once, densely, at the lower-left node, and a window of eigenvalues is kept (for the mock below, those with $"Im" λ > 0$, one of each conjugate pair). Each of them is then followed along every grid edge by `track`: no other diagonalisation. The tracks of different eigenpairs, and of different edges, are independent and run in parallel.

  *Labels.* The eigenpairs of the window are labels $1 … n$. They are carried to every node along a spanning tree of the grid (the bottom row, then every column upwards), so tree edges map labels to themselves. Every other (horizontal) edge is tracked from its left node, and the eigenvalues at its end are matched to the labels of its right node (`match_labels`, relative tolerance $10^(-7)$). A tracked eigenvalue that lands on none of the labels is _lost_: an EP with an eigenvalue outside the window, or a failed integration. The cells around it say nothing about that label.

  *Monodromy of a cell.* Around a cell, counterclockwise from its lower-left corner, the four edges compose into a permutation $σ$ of the labels (`cell_permutation`): along the bottom edge label $a$ reaches label $b$ of the lower-right corner, carried up the right edge (a tree edge); the label of the upper-left corner whose top edge ends on $b$ is $σ(a)$, carried down the left edge. A transposition $(a b)$ means an odd number of EP2 of that pair inside the cell. For a pair returning to itself, the winding of $D = (λ_a - λ_b)^2$ is the sum of its arg increments along the four edges (`cell_winding`). Along one edge, both eigenvalues are interpolated linearly between the union of their solver steps, so that $d = λ_a - λ_b$ moves on straight segments, whose change of arg seen from 0 is exactly $"angle"(d_1\/d_0) ∈ (-π, π)$; the change for $D$ is twice the sum (`arg_increment`). Taking $"angle"(D_1\/D_0)$ instead aliases when one step turns $D$ by more than $π$, which happened when the solver crossed the neighbourhood of a DP in one step. Each cell gives a `ScanCell`: its swaps, its pairs with a nonzero winding, the number of labels in longer cycles (EPs involving three eigenvalues or more), and the number of lost labels.

  *Refinement.* The flagged cells (a swap, a nonzero winding or a longer cycle) are split into $2 × 2$ sub-cells, scanned from the eigenpairs carried to their corner, recursively (`refine_tracked`, flagged cells in parallel). An EP2 stays a swap at every level; two EP2 in one cell (identity, winding $± 2$) separate into two swapping sub-cells; a DP stays an identity with winding $± 2$.

  = Test bed: the mock Liouvillian
  To test the scan where the answer is known exactly, `MockLiouvillian` is a direct sum of small independent Lindblad blocks, each with frequencies and rates affine in $(x, y) ∈ [0, 1]^2$, so that $L$ is affine and its spectrum known in closed form:
  - `DimerBlock`: the single-excitation sector of two coupled damped modes, $ω_(a,b) = ω_0 ± s(x - x_0)$, $γ_(a,b) = γ_0 ± s(y - y_0)$, coupling $g\/2$. Its coherences have a pair of EP2 at $(x_0, y_0 ± g\/2s)$, or a DP at $(x_0, y_0)$ if $g = 0$.
  - `DrivenQubit`: resonant drive and decay, a line of real-axis EP2 where $Ω = γ\/4$.
  - `DetunedQubit`: the pair $-γ\/2 ± i Δ$ touches the real axis on the line $Δ = 0$ without coalescing (a DP line).
  Coherences between different blocks are degenerate everywhere, so $L$ is restricted to a sector: the coherences of each block (18 states, default), or all elements within each block (35 states). The default instance has a pair of EP2 $10^(-3)$ apart, a pair $0.2$ apart, an off-axis DP, a real-axis EP2 line and a DP line, at generic (non-round) positions so that nothing falls on the nodes of regular or refined grids.

  *Scan of the default instance.* Window of 7 eigenvalues, $39 × 39$ grid and 7 refinement levels (cells of $2 × 10^(-4)$): each of the four EP2 is found as a swap and the DP as an identity with winding $2$, in leaves within $10^(-4)$ of the planted points, with no other leaf and no lost label (1.3 s for the grid, 0.9 s for the refinement, 12 threads).

  *Loops around several singularities.* Around one EP2 each eigenvalue covers half a closed curve and ends on the other's start. Around two EP2 of the same pair ($0.2$ or $10^(-3)$ apart), both eigenvalues return after one full turn around each other, exactly as around the DP: from a loop much larger than their separation, two EP2 and a DP cannot be told apart, since $sqrt((δ - a)(δ + a)) ≈ δ$. Around a rectangle enclosing the close pair, one EP2 of the other pair and the DP, each block behaves independently (identity, swap, identity), and the monodromy of the whole window is a single transposition.

  *Robustness.* Eight random instances, every feature moved (positions, separations, slopes of the lines), with the rates kept positive and the blocks' frequency bands separated: all 39 planted points inside the square were found with the right type, with no spurious leaf and no lost label (1.6 s per instance).

  *Limitations.*
  - With a window of the upper half-plane, the real-axis EP2 lines are not tested: their two eigenvalues are conjugate, so at most one is tracked, and crossing the line moves it onto the real axis without a visible event. Testing them requires the conjugate eigenvalues in the window.
  - An identity with winding $± 2$ that survives all refinement levels is a DP, or two EP2 closer than the finest cell. The dimension of the kernel of $L - λ$ at the leaf (one for an EP2, two for a DP) would decide.
  - An EP3 has codimension 4 in real parameters: a two-parameter plane generically misses it. What appears instead are two EP2 sharing an eigenvalue ($λ_1 ↔ λ_2$ and $λ_2 ↔ λ_3$): a cell containing both shows a 3-cycle, separated into two swaps by refinement. No mock block has three coupled modes yet.

  = Lines of EPs in three parameters
  Complex EP2 have codimension 2: in a three-parameter space they form curves, the intersection of $"Re" D = 0$ and $"Im" D = 0$. `track_ep_line` follows one by predictor–corrector continuation from a seed found in a two-parameter slice (a scan, or `ep_bracket`):
  - *Predictor:* a step $h$ along the tangent $∇ "Re" D × ∇ "Im" D$, by central differences of $D$ (one `pair_near` per point).
  - *Corrector,* in the plane through the predicted point normal to the tangent (`slice`, the family restricted to it): Newton on $("Re" D, "Im" D) = 0$ from the predicted point, accepted if it converges within $w = h\/2$ and a small loop around the result swaps the pair. The loop's radius is $w\/16$: it only has to enclose Newton's point, and must not enclose another EP involving one of the pair's eigenvalues, as happened with $w\/2$ near the converging lines of the QRM tower (the loop then cycled four eigenvalues instead of swapping two). Its pair is carried from the previous step's loop by tracking along a short segment beside the line, so that the line follows the same pair by continuation. If Newton fails, a bracketing from a square of half-width $w$ (`ep_bracket` to $10^(-2) w$, then Newton).
  - *Step control:* $h$ halved when the corrector fails or the tangent turns by more than $0.15$ rad, increased by 1.5 on nearly straight parts.
  Real-axis EPs of a real Liouvillian have codimension 1 (surfaces in three parameters): the tracker is for complex EP2 only.

  *Validation.* On a synthetic family with a curved EP line ($x y = η\/2$, $z = z_0 + x^2 - y^2$), the points are on the curve to $10^(-16)$ and the tangents within $10^(-8)$. On the Lindblad boson dimer in $(δ ω, δ γ, g)$, the two lines $g = ± 2 δ γ$, $δ ω = 0$ cross at the DP $g = 0$; the tracker passes through it, and the deviations from the analytic lines grow with $|g|$ as the Fock truncation ($10^(-16)$ at small $g$, $10^(-7)$ at $g = 0.17$ with $n_"Fock" = 5$).

  *QRM tower* (`notebooks/ep_lines.jl`), with $ω_(a,b) = ω_0 ± δ ω$, $γ_(a,b) = γ_0 ± δ γ$, coupling $g$:
  - Jaynes–Cummings limit (`RWA_env`, `RWA_coupling`, sector $k = 1$): the closed form gives straight lines $g = δ γ\/(2 sqrt(n))$, $δ ω = 0$, all through the origin. EP#sub[1] to EP#sub[5] are tracked on them to $10^(-16)$.
  - Full coupling (`RWA_env` only, odd parity sector, $N_c = 10$, converged for $n ≤ 5$ while $N_c = 8$ is not for $n = 5$): the lines bend away from $δ ω = 0$ as $δ ω ≈ n g^2\/2$, a Bloch–Siegert-like shift of the resonance, with $δ ω\/g^2 → 0.50, 1.00, 1.50, 2.00, 2.50$ for $n = 1 … 5$. The 3 or 4 Liouvillian pairs that share EP#sub[n] in the JC limit (elements $(n, n-1)$ and $(n+1, n)$) each get their own line, about $10^(-5)$ apart: $L$ is no longer block-triangular. All 19 lines tracked (465 points, 27 s), every point an EP by dense diagonalisation (pair gap $≤ 10^(-7)$).

  = Relation to the discriminant method
  The coworker's `ep_search` (`functions_QRM.jl`) looks for zeros of the discriminant over a window of eigenvalues in a real (Hermitian-operator) basis, localises them by Newton, and classifies them (`check_ep`) by the exponent of the gap ($1\/2$ for an EP2, $1$ for a DP) and by the slope of the off-diagonal element $t$ of a $2 × 2$ Schur compression of the pair (0 for an EP2, 1 for a DP). On the mock it finds the four off-axis EP2 to $10^(-15)$, the real-axis EP2 line, and the DP to $1.4 × 10^(-12)$, but classifies the DP as inconclusive: there $t ≡ 0$ (the DP is normal), and the slope of $t$ is undefined. Normal DPs, and unresolved EP2 pairs, can explain inconclusive verdicts. The two methods are complementary: monodromy detects singularities through closed loops without landing on them, and cannot mistake a DP for an EP2; the discriminant method lands on them with high accuracy.

  = Next steps
  - Kernel dimension at the leaves, to separate DPs from unresolved EP2 pairs.
  - Conjugate eigenvalues in the window, to test the real-axis EP lines; complexified parameter for codimension-1 EPs (single boson at $Q = 1\/2$).
  - A mock block with three coupled modes, to test the cycles.
  - Non-affine Liouvillians (Bloch–Redfield without `RWA_env`), for which $L'$ needs finite differences.
  - A solver that refactorises in place (KLU) for large $N$.
  - The EP-line corrector still evaluates $D$ by shift-invert (about 25 per step, 40 % of a step with the new solves). A compressed $2 × 2$ model of the pair, $M = W^† L V$ on its invariant subspace, gives $D$ and its exact gradient from the affine terms $W^† L_k V$, with about 2 subspace computations per Newton iteration; continuing that subspace along paths would remove the shift-inverts altogether. Both need careful tests in crowded spectra.
  - The loops of the scan applied to the QRM points that the discriminant method leaves inconclusive.

]


#let (template, cetz-style, byz) = setup(theme:"auto")
#show: template.with(
  title: [Monodromy tracking of Liouvillian eigenvalues],
  abstract: [How exceptional points of a Liouvillian are detected by following eigenpairs around closed loops in parameter space: bordered predictor–corrector equations with stale-LU and GMRES solves, the choice of integrator, localisation by bisection and greedy bracketing, a systematic scan of a grid with windings and refinement, the continuation of EP lines in three parameters, and their validation on the boson dimer, on driven optomechanics, on the quantum Rabi model and on a mock Liouvillian with planted singularities.]
)

#main(colors: byz, cetz-style:cetz-style)
