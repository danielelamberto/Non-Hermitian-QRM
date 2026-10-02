# Mock Liouvillian with planted, analytically known singularities: the test bed of the monodromy tools.
# It is a direct sum of small independent Lindblad blocks. Each block's frequencies and rates are affine in the two
# parameters (x, y) ∈ [0, 1]², so the Liouvillian is affine in (x, y), and its spectrum is known in closed form:
#   DimerBlock      levels |0⟩, |a⟩, |b⟩: single-excitation sector of two coupled damped modes, ωa,b = ω0 ± s(x - x0),
#                   γa,b = γ0 ± s(y - y0), coupling g/2. Pair of EP2 of the coherences |a⟩⟨0|, |b⟩⟨0| at
#                   (x0, y0 ± g/2s), λ = -γ0 - iω0 (and conjugate); g = 0: DP at (x0, y0).
#   DrivenQubit     resonant drive Ω/2 σx, decay γ (Ω, γ affine): line of real-axis EP2 where Ω = γ/4, λ = -3γ/4.
#   DetunedQubit    H = Δ|e⟩⟨e|, decay γ (Δ affine): the pair -γ/2 ± iΔ touches the real axis on the line Δ = 0
#                   without coalescing (DP line, no EP).
# Coherences between different blocks have eigenvalues repeated once per other block (degenerate everywhere), so
# mock_matrix restricts L to a sector (the block label is conserved):
#   sector = :coherences (default)  dimers: |x⟩⟨0|, |0⟩⟨x| (closed: the excitation difference is conserved); detuned
#                                   qubit: |e⟩⟨g|, |g⟩⟨e|; driven qubit: whole block. 18 states, and the only
#                                   singularities are the planted ones (the blocks' eigenvalues never meet on [0, 1]²).
#   sector = :within                all |i_k⟩⟨j_k| (35 states). The dimers' populations add, on the real axis: a DP line
#                                   x = x0 (outside the EP segment), a crossing of two real eigenvalues on y = y0, a 4-fold
#                                   coalescence at -2γ0 at the EP points, and real crossings between blocks.

"""
    MockBlock

One block of the `MockLiouvillian`: `DimerBlock`, `DrivenQubit` or `DetunedQubit`.
"""
abstract type MockBlock end

"""
    DimerBlock(center, s, ω0, γ0, g)

Two coupled damped modes in their single-excitation sector (levels |0⟩, |a⟩, |b⟩): ωa,b = ω0 ± s(x - x0),
γa,b = γ0 ± s(y - y0) with `center` = (x0, y0), coupling g/2. Pair of EP2 at (x0, y0 ± g/2s); DP at (x0, y0) if g = 0.
"""
struct DimerBlock <: MockBlock
    center::NTuple{2,Float64}
    s::Float64
    ω0::Float64
    γ0::Float64
    g::Float64
end

"""
    DrivenQubit(Ω, γ)

Qubit with resonant drive Ω/2 σx and decay rate γ, both affine: Ω = Ω[1] + Ω[2] x + Ω[3] y, same for γ. Line of
real-axis EP2 where Ω = γ/4.
"""
struct DrivenQubit <: MockBlock
    Ω::NTuple{3,Float64}
    γ::NTuple{3,Float64}
end

"""
    DetunedQubit(Δ, γ)

Qubit with H = Δ|e⟩⟨e|, Δ = Δ[1] + Δ[2] x + Δ[3] y, and decay rate γ: DP line where Δ = 0.
"""
struct DetunedQubit <: MockBlock
    Δ::NTuple{3,Float64}
    γ::Float64
end

# the affine function c[1] + c[2] x + c[3] y
affine3(c, x, y) = c[1] + c[2]*x + c[3]*y

# local Hamiltonian and jump operators of a block at (x, y) (plain matrices; |0⟩ or |g⟩ first)
function block_terms(b::DimerBlock, x, y)
    δω, δγ = b.s*(x - b.center[1]), b.s*(y - b.center[2])
    γa, γb = b.γ0 + δγ, b.γ0 - δγ
    H = ComplexF64[0 0 0; 0 b.ω0+δω b.g/2; 0 b.g/2 b.ω0-δω]
    return H, [ComplexF64[0 √(2γa) 0; 0 0 0; 0 0 0], ComplexF64[0 0 √(2γb); 0 0 0; 0 0 0]]
end
function block_terms(b::DrivenQubit, x, y)
    Ω, γ = affine3(b.Ω, x, y), affine3(b.γ, x, y)
    return ComplexF64[0 Ω/2; Ω/2 0], [ComplexF64[0 √γ; 0 0]]
end
block_terms(b::DetunedQubit, x, y) = ComplexF64[0 0; 0 affine3(b.Δ, x, y)], [ComplexF64[0 √b.γ; 0 0]]
block_dim(b::MockBlock) = size(block_terms(b, 0.5, 0.5)[1], 1)

# Dimer and detuned qubit: the only jumps go to the ground level, so the block's L is triangular and its eigenvalues are
# -i(εᵢ - conj(εⱼ)) over the eigenvalues ε of H_eff = H - (i/2)Σ J†J (ground level ε = 0).
function effective_energies(b::DimerBlock, x, y)
    δω, δγ = b.s*(x - b.center[1]), b.s*(y - b.center[2])
    r = sqrt(complex((δω - im*δγ)^2 + b.g^2/4))
    return [0.0im, b.ω0 - im*b.γ0 + r, b.ω0 - im*b.γ0 - r]
end
effective_energies(b::DetunedQubit, x, y) = [0.0im, affine3(b.Δ, x, y) - im*b.γ/2]

# elements |i⟩⟨j| (local levels, ground first) of a block kept in a sector
sector_elements(b::MockBlock, sector) = (d = block_dim(b); [(i, j) for j ∈ 1:d for i ∈ 1:d])
sector_elements(b::Union{DimerBlock,DetunedQubit}, sector) =
    sector == :coherences ? [(i, j) for j ∈ 1:block_dim(b) for i ∈ 1:block_dim(b) if (i == 1) ⊻ (j == 1)] :
                            [(i, j) for j ∈ 1:block_dim(b) for i ∈ 1:block_dim(b)]

"""
    block_spectrum(b::MockBlock, x, y; sector=:coherences)

Closed-form eigenvalues of the block's Liouvillian in the sector, one per element |i⟩⟨j| kept (in the order of
`sector_elements`). Dimer and detuned qubit: -i(εᵢ - conj(εⱼ)) from the effective energies. Driven qubit (whole block):
the Bloch equations, 0, -γ/2, -3γ/4 ± √(γ²/16 - Ω²).
"""
function block_spectrum(b::MockBlock, x, y; sector=:coherences)
    ε = effective_energies(b, x, y)
    return [-im*(ε[i] - conj(ε[j])) for (i, j) ∈ sector_elements(b, sector)]
end
function block_spectrum(b::DrivenQubit, x, y; sector=:coherences)
    Ω, γ = affine3(b.Ω, x, y), affine3(b.γ, x, y)
    r = sqrt(complex(γ^2/16 - Ω^2))
    return [0.0im, -γ/2 + 0im, -3γ/4 + r, -3γ/4 - r]
end

"""
    MockLiouvillian(; x=0.5, y=0.5, blocks=default_mock_blocks())

The mock system at the point (x, y): the direct sum of its `blocks`. Change the point with
`setproperties(mdl, (x=…, y=…))`.
"""
struct MockLiouvillian <: Model
    x::Float64
    y::Float64
    blocks::Vector{MockBlock}
end
MockLiouvillian(; x=0.5, y=0.5, blocks=default_mock_blocks()) = MockLiouvillian(x, y, blocks)

"""
    default_mock_blocks()

Default test bed: a pair of weak EP2 1e-3 apart, a pair of EP2 0.2 apart, an off-axis DP, a real-axis EP line and a
DP line. Generic (non-round) values, so that no point or line falls on the nodes of regular or refined grids; all rates
stay positive on [0, 1]². Some eigenvalues have a constant real part (-2γ0 for a dimer's population pair, -γ/2 and -γ
for the detuned qubit): these constants are kept distinct, otherwise two such eigenvalues would cross on a line.
"""
default_mock_blocks() = MockBlock[
    DimerBlock((0.3071, 0.2943), 0.1,  1.0, 0.1,  1e-4),    # EP2 pair at (0.3071, 0.2943 ± 5e-4)
    DimerBlock((0.6113, 0.7031), 0.15, 1.3, 0.15, 0.03),    # EP2 pair at (0.6113, 0.6031) and (0.6113, 0.8031)
    DimerBlock((0.7489, 0.2617), 0.2,  0.7, 0.2,  0.0),     # DP at (0.7489, 0.2617)
    DrivenQubit((0.0213, 0.0587, 0.0), (0.1, 0.0, 0.2)),   # real-axis EP line Ω = γ/4: y = 1.174x - 0.074
    DetunedQubit((-0.3071, 0.5, 0.2133), 0.43),            # DP line 0.5x + 0.2133y = 0.3071 (meets the EP line at λ ≠ -3γ/4)
]

# level ranges of the blocks in the whole system
function block_ranges(mdl::MockLiouvillian)
    d = block_dim.(mdl.blocks)
    return [sum(d[1:k-1]) + 1:sum(d[1:k]) for k ∈ eachindex(d)]
end

"""
    Lindbladian(mdl::MockLiouvillian)

Block-diagonal Hamiltonian and jump operators of the whole mock system, (Ĥ, jump operators).
"""
function Lindbladian(mdl::MockLiouvillian)
    rs = block_ranges(mdl)
    D = last(last(rs))
    H = zeros(ComplexF64, D, D)
    jumps = QuantumObject[]
    for (b, r) ∈ zip(mdl.blocks, rs)
        Hb, Jb = block_terms(b, mdl.x, mdl.y)
        H[r, r] .= Hb
        for J ∈ Jb
            Jfull = zeros(ComplexF64, D, D)
            Jfull[r, r] .= J
            push!(jumps, Qobj(sparse(Jfull)))
        end
    end
    return Qobj(sparse(H)), jumps
end

"""
    mock_sector(mdl::MockLiouvillian; sector=:coherences)

Indices of the sector's elements in the vectorised ρ (column-major).
"""
function mock_sector(mdl::MockLiouvillian; sector=:coherences)
    D = last(last(block_ranges(mdl)))
    return sort([r[i] + D*(r[j] - 1) for (b, r) ∈ zip(mdl.blocks, block_ranges(mdl)) for (i, j) ∈ sector_elements(b, sector)])
end

"""
    mock_matrix(mdl::MockLiouvillian; sector=:coherences)

The Liouvillian restricted to the sector (sparse matrix).
"""
mock_matrix(mdl::MockLiouvillian; sector=:coherences) =
    (idx = mock_sector(mdl; sector); liouvillian(Lindbladian(mdl)...).data[idx, idx])

"""
    mock_spectrum(mdl::MockLiouvillian; sector=:coherences)

Exact spectrum of `mock_matrix`, from the blocks' closed forms.
"""
mock_spectrum(mdl::MockLiouvillian; sector=:coherences) =
    reduce(vcat, block_spectrum(b, mdl.x, mdl.y; sector) for b ∈ mdl.blocks)

"""
    mock_singularities(mdl::MockLiouvillian)

The planted singularities, `(points, lines)`:
- points `(block, kind, x, y, λ)`, kind `:EP2` or `:DP`, with the eigenvalue λ there (lower half-plane; its conjugate
  is also an eigenvalue);
- lines `(block, kind, coeffs)`, kind `:real_EP2` or `:DP_line`, the line a x + b y + c = 0 for coeffs = (a, b, c).
"""
function mock_singularities(mdl::MockLiouvillian)
    pts, lines = NamedTuple[], NamedTuple[]
    for (k, b) ∈ enumerate(mdl.blocks)
        if b isa DimerBlock
            λ = -b.γ0 - im*b.ω0
            if b.g > 0
                for σ ∈ (-1, 1)
                    push!(pts, (block=k, kind=:EP2, x=b.center[1], y=b.center[2] + σ*b.g/(2b.s), λ=λ))
                end
            else
                push!(pts, (block=k, kind=:DP, x=b.center[1], y=b.center[2], λ=λ))
            end
        elseif b isa DrivenQubit
            push!(lines, (block=k, kind=:real_EP2, coeffs=(b.Ω[2] - b.γ[2]/4, b.Ω[3] - b.γ[3]/4, b.Ω[1] - b.γ[1]/4)))
        elseif b isa DetunedQubit
            push!(lines, (block=k, kind=:DP_line, coeffs=(b.Δ[2], b.Δ[3], b.Δ[1])))
        end
    end
    return pts, lines
end
