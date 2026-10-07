"""
    NonHermitianQRM

Tools to find exceptional points (EPs) of Liouvillians.
- Models and their analytical response (`models.jl`), master equations (`master_equations.jl`), a mock Liouvillian
  with planted, exactly known singularities as a test bed (`mock.jl`), and the quantum Rabi model with its
  Jaynes–Cummings limit (`qrm.jl`).
- Monodromy: eigenpairs followed continuously around loops in a two-parameter plane swap when the loop encloses an EP.
  Paths (`paths.jl`), Liouvillians affine in the parameters (`affine.jl`), predictor–corrector tracking of one
  eigenpair (`tracking.jl`), localisation of one EP by bisection (`bisection.jl`), and a systematic scan of a grid
  (`scan.jl`). Method and validation: notes/monodromy_tracking.typ.

The notebooks in notebooks/ load it with `using NonHermitianQRM`. The coworker's functions_QRM.jl is separate.
"""
module NonHermitianQRM

using QuantumToolbox
using LinearAlgebra
using SparseArrays
using Polynomials
using Accessors
using OrdinaryDiffEqVerner
import QuantumToolbox: SVector
import Base: position                 # position(path, u): methods for our paths
import Polynomials: derivative        # derivative(A::AffineLiouvillian, v): method for our affine Liouvillians

include("models.jl")
include("master_equations.jl")
include("mock.jl")
include("qrm.jl")
include("paths.jl")
include("affine.jl")
include("tracking.jl")
include("bisection.jl")
include("scan.jl")

# models.jl
export Model, Approximation, RWA_env, RWA_coupling, N_conserving, linearised, Boson, BosonDimer, Duffing, Optomech,
       classical_steady_state, classical_displacement, linearised_dimer, linearised_EP, D_R, disc, EP, roots_sorted
# master_equations.jl
export Hamiltonian, Lindbladian, DressedLiouvillian, Redfield, modes, dressed_modes, parity_sector, min_gap,
       dominant_mode
# mock.jl
export MockBlock, DimerBlock, DrivenQubit, DetunedQubit, MockLiouvillian, default_mock_blocks, block_spectrum,
       mock_sector, mock_matrix, mock_spectrum, mock_singularities
# qrm.jl
export QRM, qrm_operators, excitation_sector, jc_effective_energies, jc_spectrum, jc_EP
# paths.jl, affine.jl
export P2, ParamPath, Circle, Polygon, Segment, velocity, breakpoints, AffineLiouvillian, evaluate!, derivative,
       derivative!
# tracking.jl
export Eigenpair, eigenpairs_near, pair_near, normalisation, StaleLU, tangent, correct!, track, tracking_succeeded
# bisection.jl
export TrackedLine, branch, track_line, pair_midpoint, rect_swaps, ep_bisect
# scan.jl
export NodeLabel, TrackedEdge, TrackedGrid, ScanCell, flagged, track_edge, track_grid, tracked_scan, refine_tracked

end
