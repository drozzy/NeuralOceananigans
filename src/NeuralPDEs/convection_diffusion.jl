using Printf
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba

@parameters t,x
@variables c(..)
@derivatives Dt'~t
@derivatives Dxx''~x
@derivatives Dx'~x

# Parameters

v = 1
R = 0
D = 0.1 # diffusion
t_max = 2.0
x_min = -1.0
x_max = 1.0

# Equations, initial and boundary conditions
eqs = [ Dt(c(t, x)) ~ D * Dxx(c(t,x)) - Dx(c(t,x)) ]

bcs = [ c(t, x_min) ~ c(t, x_max),
        c(0, x) ~ cos(π*x)    
]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(x_min,x_max)
]

# Discretization
nx = 32
dx = (x_max-x_min) / (nx - 1)
dt = 0.01

# Neural network
dim = length(domains)
output = length(eqs)
hidden = 8

chain = FastChain( FastDense(dim, hidden, Flux.σ),
                    FastDense(hidden, hidden, Flux.σ),
                    FastDense(hidden, 1))

strategy = GridTraining(dx=dx)

discretization = PhysicsInformedNN(chain, strategy=strategy)

pde_system = PDESystem(eqs, bcs, domains, [t,x], [c])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=10)


# Plots

phi = discretization.phi

# initθ = discretization.initθ

# acum =  [0;accumulate(+, length.(initθ))]
# sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
# minimizers = [res.minimizer[s] for s in sep]
# ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

anim = @animate for (i, t) in enumerate(0:dt:t_max)
    @info "Animating frame $i..."
    c_predict = reshape([phi([t, x], res.minimizer)[1] for x in xs], length(xs))
    title = @sprintf("Advection-diffusion t = %.3f", t)
    plot(xs, c_predict, label="", title=title) #, ylims=(-1, 1))
end

gif(anim, "advection_diffusion_pinn.gif", fps=15)

# c_predict = reshape([ phi([0, x], res.minimizer)[1] for x in xs], length(xs))
# plot(xs,c_predict)

# c_predict = reshape([ phi([t_max,x], res.minimizer)[1] for x in xs], length(xs))

# using Plots
# plot(xs,c_predict)
