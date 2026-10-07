using NonHermitianQRM
const nh = NonHermitianQRM
using CairoMakie 
using MakieStyles
using QuantumToolbox
mdl = Boson(ω0 = 2., γ = 0.3, nB = 1.2)
N_fock = 5
â = destroy(N_fock)


#%% 

a = nh.P3([0.3, 1.2, 0])
b = nh.P3([0.3, 1.5, -2])

cross(a, b)
a × b
#%% plane_basis: an orthonormal basis (e1, e2) of the plane normal to n

# plane_basis(n) crosses n̂ with the coordinate axis a furthest from n (smallest |n̂ᵢ|), so that n̂ × a is never small:
# e1 = n̂ × a/‖n̂ × a‖ (in the plane, and ⟂ a), e2 = n̂ × e1, hence e1 × e2 = n̂. A Circle(center, ρ, n) runs in that plane,
# counterclockwise seen from the tip of n (arrow on the circle), starting at center + ρ e1 (where e1 crosses it).
using NonHermitianQRM: Circle               # also exported by Makie: ours is the loop in parameter space
using LinearAlgebra

function plane_basis_panel!(ax, n; ρ=0.75, s=1.0)
    n̂ = normalize(n)
    e1, e2 = plane_basis(n)
    k = argmin(abs.(n̂))
    a = P3(ntuple(i -> i == k ? 1.0 : 0.0, 3))           # the axis plane_basis crosses n̂ with
    o = P3(0, 0, 0)
    corners = [Point3f(s*(σ1*e1 + σ2*e2)) for (σ1, σ2) ∈ ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    mesh!(ax, corners, [1 2 3; 1 3 4]; color=(clrs[:overlay], 0.25), transparency=true, shading=NoShading)
    lines!(ax, [Point3f(-1.2a), Point3f(1.2a)]; color=clrs[:minthe], linestyle=:dash, linewidth=2)
    q = Circle(o, ρ, n)
    lines!(ax, [Point3f(position(q, u)) for u ∈ range(0, 1, 200)]; color=clrs[:helios], linewidth=2)
    arrows3d!(ax, [Point3f(position(q, 0.375))], [Vec3f(0.3*normalize(velocity(q, 0.375)))]; color=clrs[:helios],
              markerscale=1, minshaftlength=0, shaftradius=0.015, tipradius=0.05, tiplength=0.12)
    vecs, names = [n̂, e1, e2], ["n", "e₁", "e₂"]
    arrows3d!(ax, fill(Point3f(o), 3), Vec3f.(vecs); color=[clrs[:text], clrs[:byzantine], clrs[:selene]],
              markerscale=1, minshaftlength=0, shaftradius=0.02, tipradius=0.06, tiplength=0.15)
    text!(ax, Point3f.(1.15 .* vecs); text=names, color=[clrs[:text], clrs[:byzantine], clrs[:selene]], fontsize=20)
    text!(ax, Point3f(1.25a); text="a", color=clrs[:minthe], fontsize=20)
    return ax
end

normals = [P3(0.1, -0.5, 1.0), P3(-1.0, 0.2, 0.5), P3(0.3, -1.0, 0.1)]      # a = x, y, z; ~55° from the view direction
fig_pb = Figure(size=(1500, 560))
for (j, n) ∈ enumerate(normals)
    axis_name = ("x", "y", "z")[argmin(abs.(n))]
    ax = Axis3(fig_pb[1, j]; aspect=:data, title="n = $(Tuple(n)),  a = $axis_name axis",
               limits=((-1.3, 1.3), (-1.3, 1.3), (-1.3, 1.3)))
    plane_basis_panel!(ax, n)
end
fig_pb


#%% Change of variables: around the EP of the boson dimer, in direct and in polar coordinates

# Lindblad dimer (RWA_env) in the plane P = (δω, δγ), as in linear_bosons.jl: ωa,b = 1 ± δω, γa,b = 0.1 ± δγ, g = 0.01,
# EPs at δω = 0, δγ = ±g/2. Two ways around the upper EP c, the same loop parametrised alike (θ = 2πu):
# - direct: the Circle(c, ρ), on the family A, affine in P;
# - polar: the straight Segment (ρ, 0) → (ρ, 2π) in ξ = (r, θ), on Reparametrised(A, φ), φ(ξ) = c + r(cos θ, sin θ).
#   The tracker only sees a line in ξ; L is evaluated at φ(ξ), and its derivative along the line is A's along Jφ(ξ) w.
# Both must swap the two modes, with the same eigenvalues at every u.
using LinearAlgebra, Accessors
using NonHermitianQRM: Circle, SVector

n_fock_d = 5
â_d, b̂_d = destroy(n_fock_d) ⊗ eye(n_fock_d), eye(n_fock_d) ⊗ destroy(n_fock_d)
g_d = 0.01
dimer_d = BosonDimer(g=g_d, approx=[RWA_env])
dimer_at(prm_P) = setproperties(dimer_d, (ωa=1 + prm_P[1], ωb=1 - prm_P[1], γa=0.1 + prm_P[2], γb=0.1 - prm_P[2]))
A_d = AffineLiouvillian(prm_P -> liouvillian(Lindbladian(â_d, b̂_d, dimer_at(prm_P))...).data, 2)
modes_λ(prm_P) = -im .* filter(r -> real(r) > 0, roots_sorted(dimer_at(prm_P)))     # λ = -iω of the two modes

c_ep = P2(0, g_d*0.01/EP(dimer_at(P2(0, 0.01)))[1])      # upper EP: g_EP ∝ δγ at δω = 0, rescaled to g_EP = g
ρ_d = 0.004                                               # < 0.01, the distance to the lower EP
polar(ξ) = c_ep + ξ[1]*SVector(cos(ξ[2]), sin(ξ[2]))     # SVector, not P2: ForwardDiff runs it on dual numbers
R_d = Reparametrised(A_d, polar)
loop_direct = Circle(c_ep, ρ_d)
line_polar = Segment(P2(ρ_d, 0), P2(ρ_d, 2π))

start_d = eigenpairs_near(A_d(position(loop_direct, 0.)), modes_λ(position(loop_direct, 0.)))   # polar(ρ, 0): same point
sols_direct = [first(track(A_d, loop_direct, λ0, r0)) for (λ0, r0) ∈ start_d]
sols_polar = [first(track(R_d, line_polar, λ0, r0)) for (λ0, r0) ∈ start_d]
us_d = range(0, 1, 401)
for k ∈ 1:2
    lands = argmin(abs.(first.(start_d) .- sols_polar[k].u[end][end]))
    dev = maximum(abs(sols_direct[k](u)[end] - sols_polar[k](u)[end]) for u ∈ us_d)
    println("mode $k → mode $lands along the polar line,  max |λ_direct - λ_polar| = ", round(dev, sigdigits=2),
            ",  solver steps: direct ", sols_direct[k].stats.naccept, ", polar ", sols_polar[k].stats.naccept)
end

fig_cv = Figure(size=(1500, 520))
# (r, θ): the path is a straight line; r = 0, where φ is singular, is the EP itself
ax_ξ = Axis(fig_cv[1, 1]; xlabel="r", ylabel="θ", title="polar coordinates ξ = (r, θ)", yticks=(0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
vlines!(ax_ξ, [0]; color=clrs[:overlay], linestyle=:dash, label="r = 0: the EP (φ singular)")
lines!(ax_ξ, [ρ_d, ρ_d], [0, 2π]; color=clrs[:helios], linewidth=3, label="tracked line")
scatter!(ax_ξ, fill(ρ_d, length(sols_polar[1].t)), 2π .* sols_polar[1].t; color=clrs[:minthe], markersize=7,
         label="solver steps (mode 1)")
xlims!(ax_ξ, -0.4ρ_d, 2ρ_d)
# (δω, δγ): the line maps onto the circle
ax_P = Axis(fig_cv[1, 2]; xlabel="δω", ylabel="δγ", title="parameter space P = φ(ξ)", autolimitaspect=1)
lines!(ax_P, [Point2f(position(loop_direct, u)) for u ∈ us_d]; color=clrs[:helios], linewidth=3, label="direct circle")
scatter!(ax_P, [Point2f(polar(position(line_polar, t))) for t ∈ sols_polar[1].t]; color=clrs[:minthe], markersize=7,
         label="polar steps, mapped by φ")
scatter!(ax_P, [0, 0], [c_ep[2], -c_ep[2]]; marker=:star5, markersize=16, color=clrs[:text], label="EPs")
# eigenvalues: lines from the direct circle, markers from the polar line
ax_ω = Axis(fig_cv[1, 3]; xlabel="Re ω", ylabel="Im ω", title="tracked modes, ω = iλ")
for (k, colour) ∈ enumerate((clrs[:byzantine], clrs[:selene]))
    ω_direct = [im*sols_direct[k](u)[end] for u ∈ us_d]
    lines!(ax_ω, real.(ω_direct), imag.(ω_direct); color=colour, linewidth=3, label="mode $k, direct")
    ω_polar = [im*y[end] for y ∈ sols_polar[k].u]
    scatter!(ax_ω, real.(ω_polar), imag.(ω_polar); color=clrs[:base], strokecolor=colour, strokewidth=1.5, markersize=8,
             label="mode $k, polar steps")
    scatter!(ax_ω, [real(im*start_d[k][1])], [imag(im*start_d[k][1])]; marker=:star5, markersize=16, color=colour)
end
for (j, ax) ∈ enumerate((ax_ξ, ax_P, ax_ω))                 # legends below the axes, off the curves
    Legend(fig_cv[2, j], ax; framevisible=false, nbanks=2, labelsize=11, tellheight=true)
end
fig_cv
