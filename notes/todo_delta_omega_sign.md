# TODO: sign of the δω shift of the Redfield EP lines

Written 2026-10-30. Open physics question, not a code defect.

**Observation** (commit 607ce6c, `notebooks/redfield.jl`, cells "EP lines of the full QRM with Redfield baths"). In
the parametrisation ωa,b = ω0 ± δω, γa,b = γ0 ± δγ, coupling g, the EP_n lines of the full QRM move off δω = 0 as:
- Lindblad (local, RWA_env): δω = +n g²/2, a Bloch–Siegert-like shift;
- Redfield (T = 0.02): δω = −n g²/2, the same size with the opposite sign.

Both are measured with Nc = 10 at g ≤ 0.03. Nc check done (2026-10-30): points of both Redfield lines, re-localised
at Nc = 12 and 14, agree with Nc = 10 to round-off (|δP| ≤ 6e-14), so the sign is not a truncation effect.

**Candidate explanations to test:**
- The non-secular terms of the x-coupled baths (a + a†, σx): Lindblad keeps only the RWA part of the bath coupling.
- The gauge of the light–matter coupling: H has −ig(a − a†)σx (p-like), while the baths couple through x. Compare
  with an x-like coupling g(a + a†)σx.
- A Lamb-shift-like contribution: our `ohmic` S has no principal-value part, so the shift must come from Λ's
  non-secular structure.

**Checks:**
1. ~~Nc convergence of the shift~~: done, see above.
2. Its dependence on T, and on each bath alone (γa or γb set to 0).
3. Redfield with secular terms only.
4. The dressed master equation (`DressedLiouvillian`).
5. A perturbative estimate of the shift of the EP condition, from second-order Bloch–Siegert terms with each master
   equation.
