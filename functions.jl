module MyFunctions

using LinearAlgebra
using QuantumToolbox

export get_shifted_eigvals, get_shifted_eigvals_dense, H_Dicke, H_comp_separated, H_Dicke_separated, H_Dicke_polaron, J_separated

# Wrap in a QuantumObject only if needed, so the eigenvalue helpers accept both raw matrices and QuantumObject blocks.
_as_qobj(H::QuantumObject) = H
_as_qobj(H) = Qobj(H)

# Hermitian eigenvalue helpers: return the real part of the eigenvalues, shifted so that the ground state is at zero.

function get_shifted_eigvals(H, nev, sigma, tol)
    vals, _ = eigsolve(H, eigvals=nev, sigma=sigma, tol=tol)
    return real.(vals .- vals[1])
end

function get_shifted_eigvals_dense(H, nev)
    res_vals = eigenstates(Qobj(H))
    vals = res_vals.values
    return real.(vals[1:nev] .- vals[1])
end

# Non-Hermitian eigenvalue helpers: keep complex eigenvalues. The spectrum is ordered by real part; only the real part of the ground state is subtracted, so the imaginary parts stay absolute.

function get_shifted_eigvals_NH(H, nev, sigma, tol)
    vals, _ = eigsolve(_as_qobj(H); eigvals=nev, sigma=sigma, tol=tol)
    e = sort(vals; by=real)
    return e .- real(e[1])
end

function get_shifted_eigvals_dense_NH(H, nev)
    vals = eigvals(to_dense(_as_qobj(H)).data)
    e = sort(vals; by=real)
    return e[1:nev] .- real(e[1])
end

generate_op(O, idx::Int, N::Int) = mapreduce(i -> i == idx ? O : eye(2), kron, 1:N)
generate_a(N::Int, Nc::Int) = kron(destroy(Nc), [eye(2) for i in 1:N]...)
generate_collective_op(O, N::Int, Nc::Int) = mapreduce(i -> kron(eye(Nc), generate_op(O, i, N)), +, 1:N)

# Hermitian Hamiltonians for the Dicke model and its variants.

"""
    H_Dicke(g, vars)

Build the Dicke Hamiltonian for N two-level systems.
"""
function H_Dicke(g::Float64, vars::NamedTuple)
    Nc, N, ωa, ωb = vars.Nc, vars.N, vars.ωa, vars.ωb

    a = generate_a(N, Nc)
    Jx = generate_collective_op(sigmax()/2, N, Nc)
    Jz = generate_collective_op(sigmaz()/2, N, Nc)
    return ωb * Jz + ωa * a' * a + g * 2 / sqrt(N) * (a + a') * Jx
end

"""
    J_separated(N)

Build the total angular momentum operators Jx, Jy, Jz for N two-level systems, separated into blocks of conserved angular momentum. Returns a tuple of three arrays, each containing the corresponding operator for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function J_separated(N::Int)
    js = range(N/2, step=-1, length=Int(floor(N/2 + 1)))
    return ([spin_Jx(j) for j in js], [spin_Jy(j) for j in js], [spin_Jz(j) for j in js])
end

"""
    H_comp_separated(vars)

Build the bare Hamiltonian H_0 and interaction Hamiltonian H_int for N two-level systems, separated into blocks of conserved angular momentum. Returns a tuple of two arrays, H_0 and H_int, each containing the corresponding operator for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function H_comp_separated(vars::NamedTuple)
    Nc, N, ωa, ωb = vars.Nc, vars.N, vars.ωa, vars.ωb
    J_sep = J_separated(N)
    H_0_sep = QuantumObject[]
    H_int_sep = QuantumObject[]

    for i in eachindex(J_sep[1])
        a = kron(destroy(Nc), eye(size(J_sep[1][i], 1)))
        push!(H_0_sep, ωb * kron(eye(Nc), J_sep[3][i]) + ωa * a' * a)
        push!(H_int_sep, 2 / sqrt(N) * (a + a') * kron(eye(Nc), J_sep[1][i]))
    end

    return H_0_sep, H_int_sep
end
    
"""
    H_Dicke_separated(H_0, H_int, g)

Build the Dicke Hamiltonian for N two-level systems, separated into blocks of conserved angular momentum. Returns a tuple containing the corresponding Hamiltonian for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function H_Dicke_separated(H_0, H_int, g::Float64)
    result = QuantumObject[]

    for i in range(1, length(H_0))
        push!(result, H_0[i] + g * H_int[i])
    end

    return result
end

"""
    H_Dicke_polaron(H_0, H_int, g)

Build the Dicke Hamiltonian for N two-level systems in the polaron frame, separated into blocks of conserved angular momentum. Returns a tuple containing the corresponding Hamiltonian for each block, ordered by decreasing angular momentum. The blocks correspond to the irreducible representations of the SU(2) symmetry of the system.
"""
function H_Dicke_polaron(g, vars)
    Nc, N, ωa, ωb = vars.Nc, vars.N, vars.ωa, vars.ωb
    J_sep = J_separated(N)
    a_ph = destroy(Nc)
    n = a_ph' * a_ph

    λ = 2 * g / (ωa * sqrt(N))
    A = to_dense(a_ph' - a_ph).data
    cosh_op = Qobj(cosh(λ * A))
    sinh_op = Qobj(sinh(λ * A))

    result = QuantumObject[]
    for i in eachindex(J_sep[1])
        Is = eye(size(J_sep[1][i], 1))
        Jx = kron(eye(Nc), J_sep[1][i])
        Jy = kron(eye(Nc), J_sep[2][i])
        Jz = kron(eye(Nc), J_sep[3][i])

        push!(result, ωa * kron(n, Is) + ωb * (Jz * kron(cosh_op, Is) - 1im * Jy * kron(sinh_op, Is)) - 4 * g^2 / (ωa * N) * Jx^2)
    end

    return result
end

# Non-Hermitian Hamiltonians for the Dicke model and its variants.

"""
    H_Dicke_NH(g, vars)

Build the effective non-Hermitian Dicke Hamiltonian for N two-level systems, with photon and qubit damping. The dissipation enters through the effective frequencies ωa → ωa - i·γa/2 (photon) and ωb → ωb - i·γb/2 (qubit).
"""
function H_Dicke_NH(g::Real, vars::NamedTuple)
    Nc, N, ωa, ωb, γa, γb = vars.Nc, vars.N, vars.ωa, vars.ωb, vars.γa, vars.γb
    ωa_eff = ωa - 1im * γa / 2
    ωb_eff = ωb - 1im * γb / 2

    a = generate_a(N, Nc)
    Jx = generate_collective_op(sigmax()/2, N, Nc)
    Jz = generate_collective_op(sigmaz()/2, N, Nc)
    return ωb_eff * Jz + ωa_eff * a' * a + g * 2 / sqrt(N) * (a + a') * Jx
end

"""
    H_comp_NH_separated(vars)

Non-Hermitian counterpart of `H_comp_separated`: build the effective bare Hamiltonian H_0 (now complex, carrying the -i·γa/2 and -i·γb/2 damping terms) and the interaction Hamiltonian H_int (unchanged, real) for each block of conserved angular momentum.
Returns (H_0, H_int), each an array of blocks ordered by decreasing angular momentum.
"""
function H_comp_NH_separated(vars::NamedTuple)
    Nc, N, ωa, ωb, γa, γb = vars.Nc, vars.N, vars.ωa, vars.ωb, vars.γa, vars.γb
    ωa_eff = ωa - 1im * γa / 2
    ωb_eff = ωb - 1im * γb / 2
    J_sep = J_separated(N)
    H_0_sep = QuantumObject[]
    H_int_sep = QuantumObject[]

    for i in eachindex(J_sep[1])
        a = kron(destroy(Nc), eye(size(J_sep[1][i], 1)))
        push!(H_0_sep, ωb_eff * kron(eye(Nc), J_sep[3][i]) + ωa_eff * a' * a)
        push!(H_int_sep, 2 / sqrt(N) * (a + a') * kron(eye(Nc), J_sep[1][i]))
    end

    return H_0_sep, H_int_sep
end

"""
    H_Dicke_NH_separated(H_0, H_int, g)

Assemble the effective non-Hermitian Dicke Hamiltonian per block, given the components from `H_comp_NH_separated`. The combination H_0 + g·H_int is identical in form to the Hermitian case: all the non-Hermiticity is already baked into H_0.
"""
function H_Dicke_NH_separated(H_0, H_int, g::Real)
    result = QuantumObject[]

    for i in eachindex(H_0)
        push!(result, H_0[i] + g * H_int[i])
    end

    return result
end


end # module