#import "@preview/physica:0.9.8"
#import "@local/byzantine-notes:0.3.2" : setup, cetz-style


#let main(colors: color, cetz-style:()) = [

  = Models and approximations
  We consider a single damped boson of frequency $ω_0$ and damping $γ$, and a dimer of two bosons $a$, $b$ (frequencies $ω_a$, $ω_b$, dampings $γ_a$, $γ_b$) with coupling
  $
    hat(H)_"int" = g/2 (hat(a) + hat(a)^†)(hat(b) + hat(b)^†) = g hat(x)_a hat(x)_b,
  $
  where $hat(x) = (hat(a) + hat(a)^†)\/√2$.
  Each mode couples to its own Ohmic bath through $hat(x)$, with spectral density normalised so that the local Lindblad jump operator is $√(2γ) hat(a)$. Two approximations can be switched on independently (the `Approximation` enum in `resp_functions.jl`):
  - `RWA_env`: rotating-wave (secular) approximation on the system–bath coupling;
  - `RWA_coupling`: rotating-wave approximation on the coupling, $g/2 (hat(a)^† hat(b) + hat(b)^† hat(a))$.

  = Analytical response
  == Single boson
  The inverse retarded propagator of $hat(x)$ reads, without and with `RWA_env`,
  $
    D_R (ω) &= (ω^2 - ω_0^2 + 2 i γ ω)/ω_0, \
    D_R^"RWA" (ω) &= ((ω + i γ)^2 - ω_0^2)/ω_0 .
  $
  The first one is the velocity-damped oscillator, $dot.double(x) + 2γ dot(x) + ω_0^2 x = 0$. Both have the same structure, since
  $
    ω^2 - ω_0^2 + 2 i γ ω = (ω + i γ)^2 - Ω^2,
  $
  with $Ω^2 = ω_0^2 - γ^2$: the full model is the `RWA_env` one with the bare frequency replaced by the _damped_ frequency $Ω$. The poles are $ω = ± Ω - i γ$ and $ω = ± ω_0 - i γ$ respectively. The full model has an EP at critical damping,
  $
    γ_"EP" = ω_0 quad (Q = ω_0 / (2γ) = 1/2), quad ω_"EP" = - i ω_0,
  $
  where the two poles merge on the imaginary axis. With `RWA_env` there is none. This EP is already discussed for linear oscillators with several Markovian master equations in @tay_2023_LiouvillianExceptional.

  == Coupled bosons
  Without `RWA_coupling`, in the $(x_a, x_b)$ basis,
  $
    D_R (ω) = mat(D_a (ω), g; g, D_b (ω)),
  $
  with $D_i$ the single-boson expressions above. With `RWA_coupling`, in the $(a, b)$ basis, $D_R = mat(ω - ω_a + i γ_a, g\/2; g\/2, ω - ω_b + i γ_b)$. The poles are the roots of the characteristic polynomial $det D_R (ω)$.

  Define $Ω_i^2 = ω_i^2 - γ_i^2$ (or $Ω_i = ω_i$ with `RWA_env`), $overline(γ) = (γ_a + γ_b)\/2$, $δ = (γ_a - γ_b)\/2$ and $s = ω + i overline(γ)$. Up to $1\/(ω_a ω_b)$,
  $
    P(s) = &[(s + i δ)^2 - Ω_a^2] [(s - i δ)^2 - Ω_b^2] \
    &- g^2 ω_a ω_b .
  $
  *EP at finite frequency.* Writing $Δ = Ω_a^2 - Ω_b^2$, $M = (Ω_a^2 + Ω_b^2)\/2$, $k = M - δ^2$ and $G = g^2 ω_a ω_b$, all real, a double root $s_0 = x + i y$ with $x ≠ 0$ requires $P(s_0) = P'(s_0) = 0$:
  - $P' = 0$ gives $Δ = 2 i s_0 (s_0^2 - k)\/δ$, real only if $x^2 = 3 y^2 + k$, and then $Δ = -4 y (4 y^2 + k)\/δ$;
  - $im G = 0$ from $P = 0$ gives $Δ = -4 y^3 \/ δ$;
  - so $y = 0$, $Δ = 0$, $x^2 = k$ and $G = 4 Ω^2 δ^2$.

  An EP at non-zero frequency therefore requires equal _damped_ frequencies $Ω_a = Ω_b = Ω$ (bare ones with `RWA_env`), and sits at
  $
    g_"EP" = (Ω |γ_a - γ_b|)/√(ω_a ω_b), quad ω_"EP" = ± √(Ω^2 - δ^2) - i overline(γ).
  $
  For $Ω_a = Ω_b$, $P$ is even in $s$ and $s^2 = Ω^2 - δ^2 ± √(G - 4 Ω^2 δ^2)$. With `RWA_coupling` and $ω_a = ω_b$: $g_"EP" = |γ_a - γ_b|$, $ω_"EP" = ω_a - i overline(γ)$.

  *Lifted EP.* For equal bare frequencies without `RWA_env`, the damped frequencies differ by $≈ (γ_b^2 - γ_a^2)\/2$: the modes are detuned and no real $g$ gives an EP. The minimal gap scales as $γ^(3\/2)$ (a detuning $∝ γ^2$ opening a square-root branch point $∝ γ$): 0.128 for $(γ_a, γ_b) = (0.3, 0.1)$, then 0.045, 0.016, 0.0056 when both are halved successively.

  *EP on the imaginary axis.* $s^2 = 0$ is a double root when $G = (Ω^2 + δ^2)^2$, i.e. $g^2 ω_a ω_b = (Ω^2 + δ^2)^2$, at $ω = - i overline(γ)$. The positive- and negative-frequency branches of the lower mode merge there, close to the instability $g → √(ω_a ω_b)$ of the $x$–$x$ coupling. Only the model without `RWA_env` has both branches.

  = Master equations
  Three master equations, all Markovian and second order in the bath coupling, reproduce these results to a different extent. In all of them, the modes are the Liouvillian eigenvalues $λ = - i ω$ whose right eigenvectors carry $⟨hat(a)⟩$, $⟨hat(a)^†⟩$: for quadratic models $(hat(a)|$, $(hat(a)^†|$ span an $cal(L)^†$-invariant subspace, and only two eigenvectors per mode have an overlap with it.

  == Local Lindblad form
  Dissipators $cal(D)[√(2γ(1 + n_B)) hat(a)] + cal(D)[√(2γ n_B) hat(a)^†]$. The first moments obey $dif ⟨hat(a)⟩ \/ dif t = -(i ω_0 + γ)⟨hat(a)⟩$: damping acts on both quadratures, and the modes are those of `RWA_env`.

  == Dressed non-secular master equation
  `liouvillian_dressed_nonsecular` in QuantumToolbox @settineri_2018_DissipationThermal. The system Hamiltonian, including counter-rotating coupling terms, is diagonalised, and the bath operator $hat(X) = hat(U)^† hat(x) hat(U)$ is split into its energy-lowering part $hat(X)^+$ (upper triangle) and its adjoint. Rates are Ohmic, $Ω_(i j) |X_(i j)|^2 (1 + n(Ω_(i j)))$, and cross terms between two transitions are weighted by $exp(-(Ω_(i j) - Ω_(k l))^2 \/ 2σ^2)$. The limit $σ → 0$ is the dressed secular master equation @beaudoin_2011_DissipationUltrastrong, $σ = ∞$ keeps all cross terms.

  Because only $hat(X)^+$ is kept, lowering transitions are never paired with raising ones: the equation is non-secular among transitions of the same sign, but still in the rotating-wave approximation on the bath, in the dressed basis. Accordingly it follows `RWA_env`: the EP of the dimer is found when the bare frequencies match, not the damped ones. Two further points:
  - with the $x$–$x$ coupling in $hat(H)$, the normal-mode amplitude of $hat(x)_a$ scales as $1\/√(ω_±)$, which cancels the Ohmic factor $Ω$: both normal modes decay at the same rate and the EP survives. With `RWA_coupling` in $hat(H)$ nothing cancels it and the EP is lifted by $cal(O)(g γ)$, an artefact of mixing approximations;
  - the dressed secular limit ($σ → 0$) has no EP near $g ∼ γ$ at all: both modes keep damping $overline(γ)$ for any $g > 0$.

  == Non-secular Bloch–Redfield
  `bloch_redfield_tensor` with `sec_cutoff = -1` and $S(ω) = κ ω (1 + n(ω))$, $κ = 2γ\/ω_0$. For a single boson, with $Λ = ∑_ω S(ω) hat(x)(ω) \/ 2 = (κ ω_0 \/ 2) hat(a)$ at $T = 0$ and $[hat(a), hat(x)] = 1$,
  $
    (dif ⟨hat(a)⟩)/(dif t) = - i ω_0 ⟨hat(a)⟩ - γ (⟨hat(a)⟩ - ⟨hat(a)^†⟩),
  $
  that is, for the quadratures,
  $
    dot(x) = ω_0 p, quad dot(p) = - ω_0 x - 2 γ p .
  $
  The $⟨hat(a)^†⟩$ term, which pairs the $+ω_0$ and $-ω_0$ transitions and oscillates at $2 ω_0$ in the interaction picture, is what any secular or rotating-wave argument drops. It turns the symmetric damping into velocity damping and produces the shift $Ω = √(ω_0^2 - γ^2)$. Only the odd part $κ ω$ of the spectrum enters the drift, so the modes do not depend on temperature.

  Neither implementation contains a Lamb shift (principal-value part of the bath correlation function), and none is needed: for a strictly Ohmic bath, after the usual frequency counterterm, the retarded self-energy is exactly $-2 i γ ω$. The shift $ω_0 → Ω$ is not a Lamb shift. With a cutoff or a non-Ohmic spectrum, a principal-value shift would add to it.

  == Numerical checks
  With $n_"Fock" = 5$ to $10$ per mode, at $T = 0$:
  - single boson, $γ ∈ [0.02, 2] ω_0$: Redfield matches the poles of $D_R$ to $3 × 10^(-8)$ over the whole sweep, including the overdamped side; the gap at $γ = ω_0$ is $4 × 10^(-8)$. Lindblad and dressed stay on $± ω_0 - i γ$;
  - dimer, $(γ_a, γ_b) = (0.3, 0.1)$: with equal damped frequencies Redfield has its EP at the predicted $g_"EP" = 0.19525$ (gap $3.5 × 10^(-5)$, set by truncation and amplified by the square root), and gives the same avoided crossing as $D_R$ (gap 0.1285) with equal bare frequencies. Lindblad, with or without `RWA_coupling`, has its EP at $g_"EP" = 0.2$ with equal bare frequencies;
  - vacuum response spectrum $⟨hat(E)^+ (τ) hat(E)^- (0)⟩$ of $hat(x)_a$: Redfield reproduces the peak positions of $- im G_(a a)$; the dressed peaks are shifted by $cal(O)(γ^2 \/ ω_0)$.

  = Outlook: non-linear boson and perturbation theory
  Adding a non-linearity (Kerr, $x^4$, two-photon loss), the first moments no longer close and $det D_R$ has no exact counterpart. The EPs involving the lowest levels can still be followed as poles of the retarded response $G_R (ω) = - i ∫ ⟨[hat(a)(t), hat(a)^†]⟩ e^(i ω t) dif t$. These poles are the Liouvillian eigenvalues with non-zero residue $(hat(a)|r_i)(l_i|[hat(a)^†, ρ_"ss"])$. At an EP, $G_R$ has a double pole and the two residues diverge with opposite signs.

  The plan is to compute a self-energy $Σ(ω)$ diagrammatically, then solve $F(ω) = G_0^(-1)(ω) - Σ(ω) = 0$ and $F'(ω) = 0$ non-perturbatively. Points to keep in mind:
  + *Vacuum response at $T = 0$.* With excitation-conserving non-linearities ($(hat(a)^† hat(a))^2$, $hat(a)^2$ loss) and a vacuum steady state, $Σ = 0$ to all orders for the $0 ↔ 1$ pole. The non-linearity enters through finite $n_B$ (Hartree shift $2 K n$ at first order), a drive, or counter-rotating terms, either in the non-linearity ($x^4$ instead of $(hat(a)^† hat(a))^2$) or in the bath (the non-`RWA_env` steady state is not the Fock vacuum).
  + *Expand $Σ$, not the poles.* Near an EP the poles are non-analytic ($∝ √(λ - λ_"EP")$) in the parameters, $Σ$ is not. If $Σ$ is rational in $ω$, the `disc` construction carries over: evaluate $G_0^(-1) - Σ$ on a polynomial variable and look for double roots.
  + *Keep $Σ$ off-shell.* Evaluating the frequency dependence at the real transition frequencies loses the $cal(O)(γ^2 \/ ω_0)$ shift, which is exactly what moves the EP; this is the difference between the dressed and Redfield equations above. At $Q = 1\/2$, $γ = ω_0$ and this is not a small correction. Hartree loops should use bath-broadened propagators; Ohmic $⟨p^2⟩$ diverges logarithmically with the cutoff at $T = 0$ ($⟨x^2⟩$ is finite).
  + *Benchmark.* For a non-linear model, Redfield is exact in the non-linearity but only second order in the bath coupling. Near $Q = 1\/2$ a benchmark non-perturbative in $γ$ is needed: HEOM, or a reaction-coordinate / pseudomode mapping of the Ohmic bath.

  Related work: Liouvillian EPs of a Kerr-cat qubit @han_2026_ExceptionalPhase (driven, presumably Lindblad bath), dissipators beyond the rotating-wave approximation for the squeezed Kerr oscillator @venkatraman_2024_StaticEffective, and spectral theory of Liouvillians and quantum EPs @minganti_2018_SpectralTheory @minganti_2019_QuantumExceptional.

  #bibliography("response_EP.bib", style: "american-physics-society")

]


#let (template, cetz-style, byz) = setup(theme:"auto")
#show: template.with(
  title: [Exceptional points in response functions],
  abstract: [Notes on the exceptional points of damped bosons (single mode and dimer): analytical poles of the retarded response with and without rotating-wave approximations on the bath and on the coupling, which master equations reproduce them, and a plan for non-linear bosons using perturbative self-energies.]
)

#main(colors: byz, cetz-style:cetz-style)
