# TODO: dressed-basis truncation for the Redfield QRM

Written 2026-10-30, branch `liouvillian-fd`. Not started: this note is the plan.

## Goal

Build the Redfield Liouvillian of the QRM the way the coworker does: diagonalise Ĥ in a large Fock space (Nc_H), keep
only the lowest dressed levels, and assemble L on those. Then use it for scans and tracking (`tracked_scan`,
`localise_eps`, `FDLiouvillian`, the EP-line tracker).

Two truncations, two meanings (coworker's convention, `data/*.jld2`):
- `Nc` (theirs: 60): Fock truncation used to build and diagonalise Ĥ. With Nc large, the kept dressed levels are
  essentially exact.
- `N_trunc` (theirs: 30): the number of dressed levels kept to build L. This is the only truncation that matters in
  practice.
- `maxlevel`: the highest dressed level in the EP's eigenvector content. Their "reliable" table is exactly
  `maxlevel ≤ 15 = N_trunc/2`.

Ours today: `Redfield(â, σ̂, ::QRM)` / `RedfieldMap` build L on all 2Nc levels of the Fock-truncated Ĥ. That is a
single truncation, whose errors sit in the badly wrong top levels of Ĥ and leak into Λ.

## Why

- **Exact coherent part.** The kept levels and their X matrix elements are converged in Nc_H.
- **Local, measurable error.** The truncation error sits in the top kept levels: products like (XΛ)_ik sum over
  intermediate levels j, and those above the cutoff are missing. `maxlevel` measures it directly. In the Fock basis we
  needed the top-Fock-weight check and runs at several Nc.
- **Size, at large g.** The gain is modest for g ≤ 0.5: 30 levels give 450 states in the odd sector, against 512 for
  Fock Nc = 16, which is already converged to about 1e-8 there. It is large for g → 1 and for the high-|Im λ| towers
  (|Im λ| ≈ 3…13), which need many photons in the Fock basis. Redfield's L is fairly dense in either basis, so the cost
  is set by the dimension.

## The catch: a smooth gauge

The tracker needs L(P) as a matrix that is smooth in a fixed basis. Eigenvalues don't depend on the basis, but
tracking carries eigenvectors, and `FDLiouvillian` differentiates L(P ± h). A dressed basis that jumps between nearby
points produces spurious lost labels and failed h-checks. Three things need fixing:

1. **Ordering.** Sorting by energy swaps labels at level crossings, and the QRM has many crossings between opposite
   parities (Juddian points). Fix: at ε = 0, diagonalise each parity block of Ĥ separately and sort by energy within
   each block. Levels of the same parity don't cross, so this ordering is smooth.
2. **Cutoff.** Keep the lowest N levels of *each* parity, not the lowest 2N overall. Otherwise the kept set changes
   whenever a level of one parity crosses the edge against one of the other. Within a parity, near-avoided crossings at
   the edge are smooth but fast: watch the h-check there.
3. **Phases.** Each eigenvector has an arbitrary phase: fix it by a deterministic rule that is smooth over the domain,
   e.g. a chosen component real and positive. The rule must evaluate pointwise (FD and scans call L at arbitrary
   points, not along a path), so check that the chosen component never passes through zero inside the window. Option:
   the gauge change a → i a makes Ĥ real (−ig(a − a†)σx → g(a + a†)σx), which reduces the phase to a sign. X = a + a†
   then becomes imaginary, which is fine for Redfield.

ε ≠ 0 breaks parity: steps 1 and 2 need another ordering then (not needed for the current scans).

## Steps

1. **Family.** Add a `DressedRedfield` (name open) in `src/`:
   - inputs: the model, Nc_H, N per parity;
   - diagonalise each parity block of Ĥ at Nc_H, keep the N lowest levels per parity, fix the gauge;
   - dressed X's = U† X U on the kept levels, H = diag(E);
   - L from `redfield_liouvillian` (it takes any Ĥ and baths, as `QuantumObject`s), restricted to the odd sector of
     the dressed basis (|i⟩⟨j| with i, j of opposite parity);
   - `RedfieldMap` assumes fixed X's, so it doesn't apply as is. A variant taking the X's as inputs, or plain assembly,
     should be fast enough at N ≈ 15–30 per parity.
   - Do not confuse it with `DressedLiouvillian(â, σ̂, ::QRM; N_trunc)` (src/qrm.jl, via
     `liouvillian_dressed_nonsecular`): that is the dressed master equation (RWA in the bath, no Lamb shift), not
     Redfield, and it is not gauge-fixed.
2. **Smoothness.** `FDLiouvillian(L_at, P0; check_at=…)` must pass its h-check at several points of the window,
   including near Juddian points and near-avoided crossings at the cutoff. Repeat the curvature plot of
   `notebooks/redfield.jl` (cell "Smoothness in g").
3. **Validation against the Fock basis.** Re-localise the reference EPs below (small tracked scan + `localise_eps` in a
   box around each, as `nc_boxes.jl` did) as a function of N, and compare with Fock Nc = 20. Also report `maxlevel` for
   each EP, to calibrate a cut of our own.
4. **Validation against the coworker.** Their unfiltered table (`data/ep_prod_complex_eps.jld2`, T = 0) has 96 EPs in
   γa ≤ 0.04, g ≤ 0.5, including the |Im λ| ≈ 3…13 towers. Expect agreement to about 1e-6 for maxlevel well below N.
   Our T = 0.02 has had no visible effect so far.
5. **Use it.** Rescan the (γa, g) plane with a wider eigenvalue window (all |Im λ| towers of the odd sector, and
   Re λ < −0.15 on the |Im λ| ≈ 1 branch, which the first scan left out), then g up to 1, and the even sector.

## Reference data (to validate against)

Setup: Redfield QRM, ωa = ωb = 1, γb = 0.05, Ta = Tb = 0.02, odd sector. Positions converged in Fock Nc: Nc = 16 and
20 agree to about 1e-8 in (γa, g) and 1e-6 in λ. Values at Nc = 20:

| γa | g | λ |
|---|---|---|
| 0.01980852 | 0.3321561 | −0.155262 − 1.003752i |
| 0.01980982 | 0.3321889 | −0.155270 − 1.003752i |
| 0.009814011 | 0.3458568 | −0.081078 − 1.011664i (partner not found: outside its box) |
| 0.01981496 | 0.3571884 | −0.135476 − 1.005038i |
| 0.01981784 | 0.3572180 | −0.135493 − 1.005038i |
| 0.0097826 | 0.3748739 | −0.071197 − 1.016021i |
| 0.0098486 | 0.3749109 | −0.071558 − 1.016022i |
| 0.01981199 | 0.3889993 | −0.115625 − 1.007116i |
| 0.01982100 | 0.3890196 | −0.115670 − 1.007117i |
| 0.006231991 | 0.4004532 | −0.049502 − 1.031382i |
| 0.006233314 | 0.4003615 | −0.049506 − 1.031383i |
| 0.009714485 | 0.4131421 | −0.061264 − 1.023430i |
| 0.009778797 | 0.4131820 | −0.061551 − 1.023430i |
| 0.01978171 | 0.4314170 | −0.095662 − 1.010822i |
| 0.01981347 | 0.4314269 | −0.095787 − 1.010822i |
| 0.005983637 | 0.4526041 | −0.042774 − 1.047958i |
| 0.005986294 | 0.4525244 | −0.042783 − 1.047962i |
| 0.009559154 | 0.4675078 | −0.051265 − 1.037707i |
| 0.009621216 | 0.4675511 | −0.051479 − 1.037707i |
| 0.01969732 | 0.4923674 | −0.075581 − 1.018515i |
| 0.01975142 | 0.4923766 | −0.075740 − 1.018515i |

- The EPs come in split pairs: two EP2s of the same eigenvalue pair, 1e-6 to 1e-4 apart, which a coarse cell sees as
  a winding of +2 and refinement separates.
- The γa ≈ 0.0198 pairs match the coworker's table (T = 0, Nc = 60, N_trunc = 30; their λ is the conjugate of ours) to
  between 1e-7 and 2e-6. That includes g ≈ 0.332, which has maxlevel 17, so their maxlevel ≤ 15 filter is
  conservative.
- The γa ≈ 0.006 and 0.0097 families lie below their γa window (≥ 0.01); they are not in their table.
- Groups found at Nc = 12 near g ≈ 0.113, 0.290, 0.296, 0.307 and 0.359 had no EP in their ±4e-3 box at Nc = 16/20.
  Rescan in wider boxes (±3e-3 in γa, ±0.015 in g), complete at Nc = 16, stopped during Nc = 20:
  - g ≈ 0.113: no EP at Nc = 16 or 20. An artefact of Nc = 12.
  - g ≈ 0.290: it is the γa ≈ 0.0198 family, at (0.01978, 0.29475) and (0.01981, 0.29476), λ ≈ −0.1946 − 1.0023i
    (Nc = 16). That is the coworker's g ≈ 0.2948 pair (maxlevel 20). Nc = 20 moves it by another 1e-5 to 3e-5.
  - g ≈ 0.342: both partners found, at (0.009814, 0.345857) and (0.009882, 0.345892), λ ≈ −0.0811 − 1.0117i.
  - g ≈ 0.296, 0.307 and 0.359 (Nc = 16 only, not confirmed at 20): 9 distinct split pairs in γa ∈ [0.0035, 0.0099],
    g ∈ [0.288, 0.364]:
    - γa ≈ 0.0098–0.0099: g ≈ 0.2877 and 0.3039;
    - γa ≈ 0.0064–0.0065: g ≈ 0.2970, 0.3152 and 0.3642;
    - γa ≈ 0.0046–0.0048: g ≈ 0.3085 and 0.3564;
    - γa ≈ 0.0035–0.0037: g ≈ 0.2983, 0.3238 and 0.3507.

    So below g ≈ 0.37 the γa ≈ 0.006 and 0.0097 families are denser than the Nc = 12 scan showed, and further
    families appear at γa ≈ 0.0035 and 0.0047, partly below the scan's γa = 0.005: a tower structure like the
    coworker's γa ≈ 0.0198 column. Needs a scan at Nc ≥ 16 down to γa ≈ 0.003.
- The γa ≈ 0.026 pair of the Nc = 12 scan (λ ≈ −0.31 − 1.20i) is a truncation artefact: 30% of its weight sits on the
  top Fock levels. Drop it.

## Full scan at Nc = 16 (2026-10-30)

The last cell of `notebooks/redfield.jl`, run on a grid of γa ∈ [0.005, 0.04] (36 points) × g ∈ [0.1, 0.5] (41 points),
with 42 eigenvalues tracked. The scan took 2.1 h and the localisation 1.3 h. It found 58 EP2s, 0 DPs and 0 unresolved
cells; 799 cells have lost labels (window, g ≳ 0.22). Table: `data/redfield_scan_gammaa_g_Nc16.jld2`. Figure:
`figures/redfield_scan_gammaa_g.png`.
- γa ≈ 0.0198 column, g ≥ 0.29: all 14 EPs match the coworker to 2e-7 – 3e-5 in position. At g ≈ 0.265 and 0.280
  (their maxlevel 21–23), the agreement is only 3e-4 – 3e-3.
- The γa ≈ 0.006 and 0.0097 families match the Nc = 20 reference table above and contain the wider-box pairs at
  g ≈ 0.288–0.364.
- New, below g ≈ 0.29, and not checked at a larger Nc: split pairs at g ≈ 0.236–0.241 (γa ≈ 0.0064, 0.0094, 0.0173),
  and at g ≈ 0.271 and 0.278.
- Probably spurious:
  - (0.0193, 0.264) and (0.0193, 0.260), λ ≈ −0.31 − 1.19i: the Nc = 12 γa ≈ 0.026 artefact, moved with Nc. The two
    points are 4e-3 apart and no coworker EP is nearby.
  - (0.034, 0.392), λ ≈ −0.74 − 1.59i: an isolated EP.
- (0.00502, 0.479) sits on the window edge without a partner; the partner is probably just below γa = 0.005.

Localisation of the g < 0.29 pairs and of the suspects is deferred until the dressed truncation and the exact derivative
are in place (step 3 and `notes/todo_daleckii_krein.md`).

## Lessons from the Fock-basis scans

- Nc = 12 is not enough for g ≳ 0.3: EP positions are off by up to about 4e-3, and the γa ≈ 0.026 pair is spurious.
- `ep_newton` with `pair_near` is unreliable in crowded spectra. It picks wrong pairs at higher Nc, sends both
  partners of a split pair to the same point, and diverges. Re-localise with small tracked scans + `localise_eps`
  instead.
- Lost labels in the (γa, g) scan came from the eigenvalue window, not from truncation: 6 edge labels sweep out to
  Im λ ≈ +0.4 or −2 and pass EPs with untracked eigenvalues. A wider window, not a larger Nc, fixes them.

The scan and convergence scripts of 2026-10-30 lived in a temporary scratchpad. The figure they produced is
`figures/redfield_scan_gammaa_g_ext.png` (our scan at Nc = 12, with the coworker's EPs overlaid).
