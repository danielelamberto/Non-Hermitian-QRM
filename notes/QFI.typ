#import "@preview/physica:0.9.8"
#import "@local/byzantine-notes:0.3.2" : setup, cetz-style


#let main(colors: color, cetz-style:()) = [
  
  = Hermitian systems
  The Quantum Fisher Information is defined as the sensitivity of a given state $|ψ(λ)⟩$ on a parameter $λ$. For a small variation, the measurable part is orthogonal to the ray, $|∂ψ^⟂⟩ = |∂ψ⟩ - ⟨ψ|∂ψ⟩|ψ⟩$, and the quantum Fisher Information is 
  $
    F_"Q" &= 4 ⟨∂ψ^⟂|∂ψ^⟂⟩ \
    &= 4 ( ⟨∂ψ|∂ψ⟩ - |⟨∂ψ|ψ⟩|^2).  
  $
  This quantity is important because it appears in the _Quantum Cramér-Rao bound_: the variance when estimating the parameter $λ$ by measuring this state is 
  $
    Δ λ^2 >= 1/(m F_"Q"),
  $
  $m$ the number of repetitions. We often focus on the ground state of the system's Hamiltonian, or sometimes one of the excited states. Writing it as $hat(H)(λ) = hat(H)_0 + λ hat(V)$, we can directly apply perturbation theory. In the eigenbasis of $hat(H)(λ)$, with eigenvalues $E_k (λ)$ and eigenvectors $|ψ_k (λ)⟩$, assuming we choose the state $|ψ_n⟩$ as a probe to estimate $λ$, 
  $
    |∂ψ_n⟩ = ∑_(k≠n) (⟨ψ_k|hat(V)|ψ_n⟩)/(E_n - E_k) |ψ_k⟩,
  $
  and then, since we have by construction $⟨ψ_n|∂ψ_n⟩ = 0 <=> |∂ψ_n⟩ = |∂ψ_n^⟂⟩$, 
  $
    F_"Q" = 4  ∑_(k≠n) (|⟨ψ_k|hat(V)|ψ_n⟩|^2)/(E_k - E_n)^2. 
  $
  This result is often linked to the fidelity susceptibility @you_2007_FidelityDynamic. Considering the fidelity between the $n$#super[th] eigenstate of $hat(H)$ at $λ$ and $λ + δ λ$, 
  $
    F_"id" = |⟨ψ_n (λ)|ψ_n (λ + δ λ)⟩|,
  $
  the exact same term appears in the power expansion of $F_"id"^2$: 
  $
    F_"id"^2 = 1 - δ λ^2 χ_"F" + cal(O)(δ λ^4) 
  $
  and then, $4χ_"F" = F_"Q"$.

  = Non-Hermitian systems 
  Not much to report here, we are still missing a good definition in case the model does not reduce to a quasi-Hermitian case. 

  = Liouvillian systems 
  In this case, the fidelity measure is much harder to manipulate. The expression of the Uhlmann fidelity, 
  $
    F(σ, ρ) = tr(√(√σ ρ √σ)). 
  $
  The square roots are a mess to deal with. A brute force evaluation is always possible, though. 

  === Dissipative QFI
  There is a proposition to use instead a vectorisation of the density matrix, and roll with that : fidelity is again just the scalar product. This requires to normalize $hat(ρ)$, which is suspect in itself. 

  #bibliography("QFI.bib", style: "american-physics-society")

]


#let (template, cetz-style, byz) = setup(theme:"auto")
#show: template.with(
  title: [Quantum Fisher Information],
  abstract: [Quick notes on definitions and derivations for the Quantum Fisher Information in Hermitian, Non-Hermitian, Lindbladian and Liouvillian quantum systems.]
)

#main(colors: byz, cetz-style:cetz-style)