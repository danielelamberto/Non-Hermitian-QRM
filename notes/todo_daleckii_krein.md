# TODO: exact derivative of the Redfield Liouvillian (Daleckii–Krein)

Written 2026-10-30, branch `liouvillian-fd`. Not started: this note is the plan. To do on the dressed-truncation branch
(`notes/todo_truncation.md`), or right after it, not on `liouvillian-fd`.

## What it is

The Daleckii–Krein formula gives the derivative of a matrix function of a Hermitian matrix. For H = U diag(E) U† and a
perturbation Ḣ, with H̃̇ = U† Ḣ U:

    d f(H)[Ḣ] = U (F ∘ H̃̇) U†,    F_ij = f[E_i, E_j] = (f(E_i) - f(E_j))/(E_i - E_j),  F_ii = f'(E_i).

∘ is the entrywise product, and f[a, b] the first divided difference. The point is that it never needs the derivative
of the eigenvectors, dU = U K with K_ij = H̃̇_ij/(E_j - E_i). K is gauge-dependent and singular at degeneracies, while
the divided differences are finite (→ f′) when two levels meet.

Redfield's Λ is not a function of H alone but a two-sided version of the same thing (a "double operator integral"):

    Λ = Σ_ij S(E_i - E_j) |i⟩⟨i| X |j⟩⟨j| = U (S(ω) ∘ X̃) U†,    X̃ = U† X U,  ω_ij = E_i - E_j,

`filtered_operators` in src/master_equations.jl. With X fixed (a + a†, σx: independent of P), differentiating
U, E and S(ω_ij) and collecting the K terms gives, in the eigenbasis (derived 2026-10-30, and checked numerically on
random 8 × 8 Hermitian H, X, Ḣ with an ohmic S at T = 0.3: agreement with central differences of
`filtered_operators` to 6e-7 at h = 1e-3 and 6e-9 at h = 1e-4, so the error is ∝ h². With two levels 1e-7 apart,
Λ̇ stays finite (‖Λ̇‖ ≈ 8) and agrees to 1e-8):

    Λ̃̇_ij = Σ_k H̃̇_ik X̃_kj S[ω_kj, ω_ij] - Σ_k X̃_ik H̃̇_kj S[ω_ij, ω_ik] + (∂_P S · v)(ω_ij) X̃_ij,
    Λ̇ = U Λ̃̇ U†.

- S[a, b] = (S(a) - S(b))/(a - b), and S′(a) when a = b.
- The terms k = i (first sum) and k = j (second sum) carry the eigenvalue shifts Ė_i = H̃̇_ii. The other terms carry
  the eigenvector motion. K has cancelled: no gauge, and no singularity at degenerate levels.
- The last term is the explicit dependence of the spectrum on the parameters. For `ohmic(κ, T)`, S is linear in
  κ = γ/ω: ∂S/∂γ = S/γ, and ωa, ωb enter both H and κ.
- Cost: O(N³) per bath and per direction, with N = dim H (the weights S[·,·] form an N × N × N array). That is
  comparable to one eigendecomposition, and cheap next to the assembly of L.

Then L itself is *linear* in (H, Λ_b, Λ_b†): it is the `RedfieldMap`, L = M x with x = [vec H; vec Λ_1; vec conj Λ_1;
…]. So the exact derivative is

    L̇ = M [vec Ḣ; vec Λ̇_1; vec conj Λ̇_1; …],

one more sparse product with the same M. Ḣ is exact and cheap: the QRM Hamiltonian is affine in g, ωa, ωb, ε.

## Why it matters here

`FDLiouvillian` (src/finite_differences.jl) differentiates by central differences: dL ≈ |v| (L(P + h v̂) -
L(P - h v̂))/2h.
- **Cost.** Two full builds per derivative, each an eigendecomposition plus an assembly, on top of the build of L(P).
  The step check costs 4n more builds per point. The exact derivative reuses the eigendecomposition of L(P): about
  one extra O(N³) pass and one sparse product.
- **Accuracy.** FD is limited to about 1e-6 to 1e-7 relative error: the optimum h ≈ 1e-5 balances h² truncation
  against eps‖L‖/h round-off, and the check's rtol is 1e-6. The exact derivative is accurate to round-off. That
  helps the tracker's predictor (larger steps, fewer corrections) and the gradients of D in `ep_newton`,
  `disc_gradient` and the EP-line corrector, which currently use their own finite differences (`fd`).
- **Robustness: no h to choose.** Near a crossing of two bath-coupled levels at small T, L varies on a scale ~T in the
  level spacing, and FD needs h ≪ T/|d(E_i - E_j)/dP|. The check only sees the points it tests (a known blind spot).
  The exact derivative stays exact there, as long as S is C¹: T > 0. At T = 0, S′ jumps at ω = 0 (the kink), and no
  derivative helps.
- **Speed of scans.** The scans of 2026-10-30 (Fock Nc = 12, about 30 min for 15 × 40 cells) were dominated by builds
  of L. With derivatives no longer costing two builds each, a large part of that goes.

## Plan

1. **Write the derivation down** (paper or a short note), and turn the numerical check above into a test: Λ̇
   against central differences of `filtered_operators` on a small random H, X and S, with error ∝ h², plus a case with
   two nearly degenerate levels, where K blows up but Λ̇ must not. The check used an O(N⁴) double loop: vectorise it
   to O(N³).
2. **Divided-difference weights.** S[a, b] with a guarded limit: when |a - b| is small relative to the scale of S
   (T for `ohmic`), switch to S′ or to a series, to avoid cancellation in (S(a) - S(b))/(a - b). This needs S′ from
   each spectrum: add it next to `ohmic` (analytic), or require spectra to provide it.
3. **A structured family.** A type that knows H(P) (affine pieces: `AffineLiouvillian`-like, for H only), the baths
   (fixed X's, spectra S(P; ω) with ∂_P S), and the `RedfieldMap`. It provides the interface of src/affine.jl
   (`A(P)`, `work_matrix`, `evaluate!`, `derivative!`), so the scans, `slice`, `Reparametrised` and the tracker take it
   unchanged. It caches the eigendecomposition of the last P, because the tracker asks for L and dL at the same point.
   It must be thread-safe (the scans track edges in parallel): one cache per task, or no cache.
4. **`RedfieldMap` with Λ's as input.** Today `R(H, spectra)` computes the Λ's itself. Add a method taking (H, Λs)
   directly, used for both L (from Λ) and L̇ (from Ḣ, Λ̇).
5. **Tests.**
   - Against `FDLiouvillian`: agreement to its own error, ~h², along each coordinate.
   - Along rates only (γa, γb), where Redfield is affine in κ: the exact derivative must equal the difference of two
     builds.
   - Swap around the EP of `notebooks/redfield.jl` (cell "Finite-difference family"), and the EP lines of commit
     607ce6c, with the exact derivative: same EPs and fewer corrector steps.
6. **Gradients of D.** Optionally, give `disc_gradient` / `ep_newton` an exact path: dλ from the left and right
   eigenvectors and L̇ (Hellmann–Feynman for non-Hermitian L, away from the EP) instead of `fd` differences of
   eigenvalues.

## Interaction with the dressed truncation

In the dressed basis (`notes/todo_truncation.md`), L is written in a frame that moves with P. Its derivative then
also contains the frame motion (terms in K, conjugated into the kept levels). K stays finite there because levels of
the same parity do not cross, but it is gauge-dependent. Do the Fock-basis version first (gauge-free, by the formula
above), then decide whether the dressed family differentiates in its own frame or reuses the Fock-basis Λ̇ projected
onto the kept levels. The dressed family knows H(P) and its eigenbasis by construction: that is why this note is
deferred to that branch.
