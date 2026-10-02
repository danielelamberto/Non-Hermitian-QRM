#import "@preview/physica:0.9.8"
#import "@local/byzantine-notes:0.3.2" : setup, cetz-style


#let main(colors: color, cetz-style:()) = [

  = Idea
  Near an EP of order two, two eigenvalues behave as $λ_± ≈ λ_"EP" ± α √(z - z_"EP")$, with $z$ a complex combination of the parameters. Following both eigenvalues continuously around a closed loop enclosing the EP, the square root changes sign: the two eigenvalues _swap_, and each one returns to itself only after two loops. A loop enclosing no EP, or two EPs of the same pair, gives no swap. Detecting a permutation after one loop therefore detects an EP inside it, without having to land on it.

  *Real or complex parameters.* In a basis of Hermitian operators the Liouvillian is a real matrix: its eigenvalues are real or come in complex-conjugate pairs.
  - Two real eigenvalues merging into a conjugate pair (critical damping, $Q = 1\/2$) is codimension 1 in real parameters: these EPs form lines, and a loop must use a complexified parameter, $L(s) = L_0 + s L_1$ with $s ∈ ℂ$.
  - Two complex, non-conjugate eigenvalues merging (the finite-frequency EP of the dimer) is codimension 2: these EPs are isolated points of a real two-parameter plane, and a loop in real parameters encircles them.

  = Ingredients
  *Paths.* A closed path in parameter space is given by `position(p, u)`, `velocity(p, u)` $= dif "position"\/dif u$ for $u ∈ [0, 1]$, and the `breakpoints` where the velocity jumps (the corners of a `Polygon`), passed to the ODE solver as `tstops`. For the dimer, the plane is $(δ ω, δ γ)$ with $ω_(a,b) = ω_0 ± δ ω$, $γ_(a,b) = γ_0 ± δ γ$, written as a function δ ↦ model with `setproperties` where the loop is defined.

  *Affine Liouvillian.* The Lindblad Liouvillian is linear in the frequencies and the rates, so along the path
  $
    L(u) &= L_0 + δ ω(u) L_ω + δ γ(u) L_γ, \
    L'(u) &= δ ω'(u) L_ω + δ γ'(u) L_γ,
  $
  where primes denote derivatives in $u$,
  with $L_ω = -i[hat(a)^† hat(a) - hat(b)^† hat(b), dot]$ and $L_γ = 2(1 + n_B)(cal(D)[hat(a)] - cal(D)[hat(b)]) + 2 n_B (cal(D)[hat(a)^†] - cal(D)[hat(b)^†])$ (`AffineLiouvillian`, built from any function δ ↦ L(δ) by $L_k = (L(h e_k) - L_0)\/h$, with a check that the family is affine). Nothing is rebuilt along the path; only sparse sums are evaluated.

  *Starting point.* At $u = 0$ the eigenpairs $(λ, r)$ nearest to target values (here $-i ω$ from the roots of `disc`) are obtained by shift-invert (`eigenpairs_near`).

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
  (`tangent`). This ODE for $[r; λ]$ is integrated over $u ∈ [0, 1]$ with an adaptive solver (`track`, Vern7, tolerances $10^(-10)$, $10^(-12)$). As a check, $λ' = (l^† L' r)\/(l^† r)$ from first-order perturbation theory, with $l$ the left eigenvector.

  *Corrector.* After each accepted step, Newton iterations at fixed $u$ pull the state back onto an exact eigenpair (`correct`):
  $
    B vec(δ r, δ λ) = vec(-(L - λ) r, 1 - c^† r),
  $
  then $r ← r + δ r$, $λ ← λ + δ λ$.
  The neglected term $-δ λ space δ r$ is second order, so the error is squared at each iteration. Newton converges to the nearest eigenpair: it keeps the state accurate but cannot detect a jump to the other eigenvalue, which only small enough steps prevent.

  *Reset of $c$.* Since Newton keeps $c^† r = 1$, a rotation of $r$ away from $c$ shows up as a growth of $‖r‖$, with $cos angle (c, r) = 1\/(‖c‖ ‖r‖)$. When this falls below $0.1$, $c$ is reset to $r\/(r^† r)$, which keeps $c^† r = 1$ and leaves $r$ unchanged. Without it, $‖r‖$ reached $10^7$ around a rectangle and Newton needed eight times more iterations. Since $c$ is the ODE parameter, the solver must not interpolate lazily (`Vern7(lazy=false)`): lazy stages are evaluated after the solve, with the latest $c$.

  *Detection.* After one loop, $λ(1)$ is compared with the eigenvalues at $u = 0$: landing on the other one means a swap.

  = Linear solves: stale LU and refinement
  $B$ depends on $(u, λ, r)$ and must be solved at every right-hand-side evaluation (7 to 16 per step) and every Newton iteration. A sparse LU of $B$ dominated the cost (3–4 ms out of 4 ms for $N = 625$). Since $B$ changes little along the path, the factorisation $F$ of an earlier $B_"old"$ is reused, and the solution of $B x = b$ is refined (`StaleLU`, `solve_bordered!`):
  $
    x_0 = F \\ b, quad x_(k+1) = x_k + F \\ (b - B x_k),
  $
  with $B x$ computed without assembling $B$. The error is multiplied by $I - B_"old"^(-1) B$ at each iteration, a factor $≈ 0.07$ over a typical step, at the cost of one sparse product and one triangular solve ($≈ 1\/50$ of a factorisation). The solver refactorises when refinement does not reach $‖b - B x‖ ≤ 10^(-12) ‖b‖$ within 20 iterations or stops decreasing, and after any solve that needed more than 8 iterations. Near an EP, $B_"old"^(-1)$ grows and refinement slows down; GMRES preconditioned by the same $F$ would be the more robust variant.

  = Validation
  Dimer with `RWA_env`, $ω_0 = 1$, $γ_0 = 0.1$, $g = 0.01$, EPs at $(δ ω, δ γ) = (0, ± g\/2)$; $n_"Fock" = 5$ per mode, $N = 625$. Both modes are tracked and compared with $-i ω$ from `disc` along the whole loop:
  #table(
    columns: 4,
    stroke: white,
    [*Loop*], [*After one loop*], [*$max|λ - λ_"disc"|$*], [*Time*],
    [circle around one EP], [swap], [$2.6 × 10^(-13)$], [1.6–1.9 s],
    [rectangle around one EP], [swap], [$7 × 10^(-11)$], [2.7 s],
    [circle around no EP], [no swap], [$9 × 10^(-14)$], [0.7 s],
    [circle around both EPs], [no swap], [$7 × 10^(-14)$], [1.5 s],
    [circle passing $10^(-4)$ from the EP], [swap], [$1.8 × 10^(-12)$], [0.7–1.0 s],
  )
  The stale LU brought these times down by 2 to 5 compared with a fresh factorisation at every solve, with identical results.

  = Next steps
  - Allocation-free refinement (preallocated buffers, `ldiv!`), lazy interpolation with `saveat`, parallel tasks over modes and loops.
  - Non-affine Liouvillians (Bloch–Redfield without `RWA_env`), for which $L'$ needs finite differences.
  - Complexified parameter $s$ for codimension-1 EPs (single boson at $Q = 1\/2$).
  - Locating the EP, by Newton on the augmented system $(L - λ) r = 0$, $(L - λ)^† l = 0$, $l^† r = 0$, or by following $"cond"(B)$.

]


#let (template, cetz-style, byz) = setup(theme:"auto")
#show: template.with(
  title: [Monodromy tracking of Liouvillian eigenvalues],
  abstract: [How exceptional points of a Liouvillian are detected by following eigenpairs around closed loops in parameter space: path parametrisation, bordered predictor–corrector equations, normalisation resets, stale-LU linear solves, and the validation on the boson dimer.]
)

#main(colors: byz, cetz-style:cetz-style)
