# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Julia research code for the quantum Rabi model (QRM, N = 1) and Dicke model (N two-level systems coupled to one cavity mode). It looks for exceptional points (EPs) in two settings: effective non-Hermitian Hamiltonians, and Liouvillian EPs (LEPs) of a dressed, non-secular master equation. There is no build, lint or test suite. All work happens in Jupyter notebooks that call a single shared module.

- `functions.jl`: module `MyFunctions`, holding all reusable physics code. Every new function must be added to the `export` list.
- `non_hermitian_QRM.ipynb`: covers the Hermitian spectrum, NH-Hamiltonian EP maps and Newton refinement, the effect of γb, the emission spectrum from the generalized Liouvillian, and LEP maps.
- `non_hermitian_Dicke.ipynb`: the same analysis for N > 1.
- `Project.toml` / `Manifest.toml`: Julia environment (Manifest pinned to Julia 1.12.6, QuantumToolbox 0.47.3). The notebook kernel is `julia-1.13`.

## Environment

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -t auto          # the notebooks use Threads.@threads for parameter scans
```

The notebooks load the module with `using Revise; includet(...); using .MyFunctions`. `Revise` is **not** in `Project.toml`, so it has to be available from the global environment. The include path differs between notebooks. `non_hermitian_Dicke.ipynb` uses `includet("functions.jl")`, but `non_hermitian_QRM.ipynb` currently uses `includet("Julia/NHQRM/functions.jl")`, which is one author's parent-directory layout. Adjust the path when running from the repo root.

To check a change to `functions.jl` without a notebook:
```bash
julia --project=. -e 'include("functions.jl"); using .MyFunctions; <quick check>'
```

The notebooks are committed with large embedded PNG outputs. Don't read them raw. Extract the cell sources, for example with `python3 -c "import json; ..."`.

## Conventions in `functions.jl`

- **Parameters** are passed as a `NamedTuple` `vars = (ωa, ωb, Nc, N[, γa, γb])`, where `ωa` is the cavity frequency, `ωb` the qubit splitting, `Nc` the photon Fock cutoff and `N` the number of atoms. The coupling `g` is a separate argument. It is normalized as `g·2/√N·(a+a†)Jx`, so N = 1 gives the QRM `g(a+a†)σx`.
- **Hilbert-space ordering** is `photon ⊗ atom₁ ⊗ … ⊗ atom_N` (`generate_a`, `generate_collective_op`).
- **Three representations of the same Hamiltonian:**
  - the full 2^N tensor product (`H_Dicke`, `H_Dicke_NH`);
  - blocks of conserved total spin j, ordered by decreasing j (`J_separated`, `H_comp_*separated` → `(H_0, H_int)` so that scans over g reuse the components, `H_Dicke_*separated`, `H_Dicke_polaron`);
  - Z₂ parity sectors Π = (-1)^(a†a + Jz + N/2) (`parity_blocks_NH` for the full space, `parity_blocks_NH_coll` for the j = N/2 sector only). EP searches diagonalize each parity block separately so that levels from different sectors, which are allowed to cross, aren't mistaken for EPs.
- **Non-Hermiticity** enters only through `ωa → ωa − iγa/2` and `ωb → ωb − iγb/2` (no quantum jumps). `H_int` stays real.
- **Eigenvalue helpers:** `get_shifted_eigvals*` (Hermitian) return the real spectrum minus E₀. The `_NH` variants keep complex values sorted by real part and subtract only `Re(E₀)`.
- **Liouvillian:** `gen_liouvillian_qrm` wraps QuantumToolbox's `liouvillian_dressed_nonsecular` and returns `(E, U, L)` in the truncated dressed basis (`N_trunc`, `σ_filter`, `tol` pass through as kwargs). It is gauge-sensitive. Because `H_Dicke` is in the rotated frame `a → −ia`, the cavity couples to its bath through `i(a − a†)`, not `a + a†`. The qubit couples through `σx`. The matching detection operator is `qrm_emission_field` = `(a+a†) + 2(g/ωa)σx`. Keep these consistent when adding baths or observables.
- **LEP detection:**
  - `lep_indicators` uses the eigenmatrix overlap inside a window.
  - `lep_min_gap` scans with eigenvalues only, which is preferred in ultrastrong coupling where the eigenvectors are ill-conditioned.
  - A small gap alone does not identify an EP, since diabolic points also give a zero gap. Classify each dip by how the gap scales along a cut: √ scaling means an EP, linear scaling means a DP. Refine off-grid before quoting a location.
