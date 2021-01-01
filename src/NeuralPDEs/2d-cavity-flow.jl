#################################################################################
#   2D Cavity Flow Problem using NeuralPDE
#   Reference: https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/14_Step_11.ipynb
#
#################################################################################
# Andriy: Created by Emmanuel Lujan from Julia Slack
#################################################################################
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba

@parameters t,x,y
@variables u(..),v(..),p(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x
@derivatives Dy'~y
@derivatives Dyy''~y

# Parameters

rho = 1.0 
nu = 0.1
t_max = 0.05 #nit * dt
x_max = 2.0
y_max = 2.0

# Equations, initial and boundary conditions

eqs = [
        ( Dt(u(t,x,y)) + u(t,x,y) * Dx(u(t,x,y)) + v(t,x,y) * Dy(u(t,x,y)) ~ 
          - 1.0 / rho * Dx(p(t,x,y)) + nu * (Dxx(u(t,x,y)) + Dyy(u(t,x,y))) ),
          
        ( Dt(v(t,x,y)) + u(t,x,y) * Dx(v(t,x,y)) + v(t,x,y) * Dy(v(t,x,y)) ~
          - 1.0 / rho * Dy(p(t,x,y)) + nu * (Dxx(v(t,x,y)) + Dyy(v(t,x,y))) ),
          
        ( Dxx(p(t,x,y)) + Dyy(p(t,x,y)) ~ -rho * ( Dx(u(t,x,y)) * Dx(u(t,x,y)) 
          + 2.0 * Dy(u(t,x,y)) * Dx(v(t,x,y)) + Dy(v(t,x,y)) * Dy(v(t,x,y)) ) )
      ]

bcs = [ 
        u(0.0,x,y)~0.0,
        v(0.0,x,y)~0.0,
        p(0.0,x,y)~0.0,

        u(t,0.0,y)~0.0,
        u(t,x_max,y)~0.0,
        u(t,x,0.0)~0.0,
        u(t,x,y_max)~1.0,

        v(t,0.0,y)~0.0,
        v(t,x_max,y)~0.0,
        v(t,x,0.0)~0.0,
        v(t,x,y_max)~0.0,

        p(t,x,y_max)~0.0,
        Dy(p(t,x,0.0))~0.0,
        Dx(p(t,0.0,y))~0.0,
        Dx(p(t,x_max,y))~0.0
      ]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(0.0,x_max),
           y ∈ IntervalDomain(0.0,y_max)]

# Discretization
nx = 41
ny = 41
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
dt = 0.001

# Neural network
dim = length(domains)
output = length(eqs)
neurons = 12
chain1 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))
chain2 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))
chain3 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))

strategy = GridTraining(dx=dx)
#strategy = GridTraining(dx=[dt,dx])

discretization = PhysicsInformedNN([chain1,chain2,chain3],strategy=strategy)

pde_system = PDESystem(eqs,bcs,domains,[t,x,y],[u,v,p])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=1000)
#res = GalacticOptim.solve(prob, ADAM(0.001), cb = cb, maxiters=5000)


# Plots

phi = discretization.phi
initθ = discretization.initθ
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
ts,xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
u_predict = reshape([ phi[1]([t_max,x,y],minimizers[1])[1] for x in xs for y in ys ], length(xs), length(ys))
v_predict = reshape([ phi[2]([t_max,x,y],minimizers[2])[1] for x in xs for y in ys ], length(xs), length(ys))
p_predict = reshape([ phi[3]([t_max,x,y],minimizers[3])[1] for x in xs for y in ys ], length(xs), length(ys))


using Plots, LaTeXStrings
uv(x, y) = (u_predict[Int(round(x/dx))+1,Int(round(y/dy))+1],
            v_predict[Int(round(x/dx))+1,Int(round(y/dy))+1])
xxs = [x for x in xs for y in ys]
yys = [y for x in xs for y in ys]
plot(xs, ys, p_predict, linetype=:contourf, title = L"$Cavity Flow$")
quiver!(xxs, yys, quiver=uv)
savefig("2d-cavity-flow.svg")

