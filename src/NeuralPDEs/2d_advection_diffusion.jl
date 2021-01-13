using Printf
import Flux: σ
using ModelingToolkit
using GalacticOptim
using Optim
using DiffEqFlux
using NeuralPDE
using Quadrature, Cubature, Cuba
using Plots

@parameters t,x,y
@variables c(..)
@derivatives Dt'~t
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dx'~x
@derivatives Dy'~y

# Parameters

u = 1.0
v = 1.0
v_vector = (u, v)
R = 0
D = 0 # 0.1 # diffusion
t_max = 2.0
x_min = -1.0
x_max = 1.0
y_min = -1.0
y_max = 1.0

# exp(-(x^2+y^2)/0.1)
# div(v_vector * c) = dx(uc) + dy(vc)
# Equations, initial and boundary conditions
eqs = [ Dt(c(t,x,y)) ~ D * (Dxx(c(t,x,y)) + Dyy(c(t,x,y))) - (u*Dx(c(t,x,y)) + v*Dy(c(t,x,y))) + R]

bcs = [ 
        c(0, x, y) ~ exp(-(x^2+y^2)/0.1), #cos(π*x) * cos(π*y) + 1.0,  
        c(t, x_min, y) ~ c(t, x_max, y),
        c(t, x, y_min) ~ c(t, x, y_max)
]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
        x ∈ IntervalDomain(x_min,x_max),
        y ∈ IntervalDomain(y_min,y_max)
]

# Discretization
nx = 32
# ny = 32
dx = (x_max-x_min) / (nx - 1)
# dy = (y_max-y_min) / (ny -1)
# dt = 0.01

# Neural network
dim = length(domains)
output = length(eqs)
hidden = 32

chain = FastChain( FastDense(dim, hidden, tanh),
                    FastDense(hidden, hidden, tanh),
                    FastDense(hidden, 1))

# strategy = GridTraining(dx=[dt,dx,dy])
strategy = StochasticTraining(900)

discretization = PhysicsInformedNN(chain, strategy=strategy)

pde_system = PDESystem(eqs, bcs, domains, [t,x,y], [c])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob,ADAM();cb=cb,maxiters=10)


# Plots

phi = discretization.phi

initθ = discretization.initθ

acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
ts,xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

# Plot initial condition
c_predict = reshape([ phi([0, x, y], res.minimizer)[1] for x in xs for y in ys], length(xs), length(ys))
plot(xs, ys, c_predict)

# Animate
anim = @animate for (i, t) in enumerate(0:dt:t_max)
    @info "Animating frame $i..."
    c_predict = reshape([phi([t, x, y], res.minimizer)[1] for x in xs for y in ys], length(xs), length(ys))
    title = @sprintf("Advection-diffusion t = %.3f", t)
    heatmap(xs, ys, c_predict, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal)#, clims=(0, 1))
end

gif(anim, "advection_diffusion_2d_pinn.gif", fps=15)


# Plot correct solution
using JLD2
file = jldopen("advection_diffusion/simulation/cosine_advection_diffusion.jld2")
iterations = parse.(Int, keys(file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)
    @info "Animating frame $i..."
    Hx = file["grid/Hx"]
    x = file["grid/xC"][1+Hx:end-Hx]
    t = file["timeseries/t/$iter"]
    c = file["timeseries/c/$iter"][:]
    
    title = @sprintf("Advection-diffusion t = %.3f", t)
    p = plot(x, c .+ 1, linewidth=2, title=title, label="Oceananigans",
             xlabel="x", ylabel="Tracer", xlims=(-1, 1), ylims=(0, 2))

    c_predict = reshape([phi([t, x], res.minimizer)[1] for x in xs], length(xs))
    
    plot!(p, x, c_predict, linewidth=2, label="Neural PDE")
end

gif(anim, "tanh_hidden32_it100_comparison.gif", fps=15)
