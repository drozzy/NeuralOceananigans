using Printf
import Flux: σ
using ModelingToolkit
using GalacticOptim
using Optim
using DiffEqFlux
using NeuralPDE
using Quadrature, Cubature, Cuba
using Plots

@parameters t,x
@variables c1(..),c2(..),c3(..),c4(..),c5(..),c6(..)
Dt = Differential(t)
Dx = Differential(x)

# Parameters
d_Fei = 1.
d_Cri = 1.
d_Nii = 1.
d_Sii = 1.
d_Fev = 1.
d_Crv = 1.
d_Niv = 1.
d_Siv = 1.
v = 1.
R1 = 0.
R2 = 0.
R3 = 0.
R4 = 0.
R5 = 0.
R6 = 0.

D = 0.1 # diffusion
α = 1.0
t_max = 2.0
x_min = 0.0
x_max = 1.0

f1 = α * c1(t, x) * (d_Fei * Dx(c3(x,t)) + d_Nii * Dx(c4(x,t)) + d_Cri * Dx(c5(x,t)) + d_Sii * Dx(c6(x,t)))+ Dx(c1(x,t)) * (d_Fei * c3(t, x) + d_Nii * c4(t, x) + d_Cri * c5(t, x)+ d_Sii * c6(t, x))
f2 = -α * c2(t, x) * (d_Fev * Dx(c3(x,t)) + d_Niv * Dx(c4(x,t)) + d_Crv * Dx(c5(x,t)) + d_Siv * Dx(c6(x,t)))+ Dx(c2(x,t)) * (d_Fev * c3(t, x) + d_Niv * c4(t, x) + d_Crv * c5(t, x)+ d_Siv * c6(t, x))
f3 = α * Dx(c3(x,t)) * (d_Fei * c1(t, x) + d_Fev * c2(t, x))  - c3(t, x) * (d_Fev * Dx(c2(x,t)) - d_Fei * Dx(c1(x,t)))
f4 = α * Dx(c4(x,t)) * (d_Nii * c1(t, x) + d_Niv * c2(t, x))  - c4(t, x) * (d_Niv * Dx(c2(x,t)) - d_Nii * Dx(c1(x,t)))
f5 = α * Dx(c5(x,t)) * (d_Cri * c1(t, x) + d_Crv * c2(t, x))  - c5(t, x) * (d_Crv * Dx(c2(x,t)) - d_Cri * Dx(c1(x,t)))
f6 = α * Dx(c6(x,t)) * (d_Sii * c1(t, x) + d_Siv * c2(t, x))  - c6(t, x) * (d_Siv * Dx(c2(x,t)) - d_Sii * Dx(c1(x,t)))



# Equations, initial and boundary conditions
eqs = [
Dt(c1(t, x)) ~  Dx(f1*c1(t,x)) + R1, 
Dt(c2(t, x)) ~  Dx(f1*c2(t,x)) + R2, 
Dt(c3(t, x)) ~  Dx(f1*c3(t,x)) + R3, 
Dt(c4(t, x)) ~  Dx(f1*c4(t,x)) + R4, 
Dt(c5(t, x)) ~  Dx(f1*c5(t,x)) + R5, 
Dt(c6(t, x)) ~  Dx(f1*c6(t,x)) + R6 ]
bcs = [ 
        c1(0, x) ~  1.0,  c2(0, x) ~  1.0, c3(0, x) ~  0.68, c4(0, x) ~  0.2, c5(0, x) ~  0.1, c6(0, x) ~  0.02,
        c1(t, x_min) ~ c1(t, x_max),
        c2(t, x_min) ~ c2(t, x_max),
        Dx(c3(t, x_min)) ~ 0., Dx(c3(t, x_max)) ~ 0.,
        Dx(c4(t, x_min)) ~ 0., Dx(c4(t, x_max)) ~ 0.,
        Dx(c5(t, x_min)) ~ 0., Dx(c5(t, x_max)) ~ 0.,
        Dx(c6(t, x_min)) ~ 0., Dx(c6(t, x_max)) ~ 0.,
        
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
hidden = 32

# chain = FastChain( FastDense(dim, hidden, tanh),
#                     FastDense(hidden, hidden, tanh),
#                     FastDense(hidden, 1))

chain1 = FastChain(FastDense(dim,hidden,Flux.σ),FastDense(hidden,hidden,Flux.σ),FastDense(hidden,1))
chain2 = FastChain(FastDense(dim,hidden,Flux.σ),FastDense(hidden,hidden,Flux.σ),FastDense(hidden,1))
chain3 = FastChain(FastDense(dim,hidden,Flux.σ),FastDense(hidden,hidden,Flux.σ),FastDense(hidden,1))
chain4 = FastChain(FastDense(dim,hidden,Flux.σ),FastDense(hidden,hidden,Flux.σ),FastDense(hidden,1))
chain5 = FastChain(FastDense(dim,hidden,Flux.σ),FastDense(hidden,hidden,Flux.σ),FastDense(hidden,1))
chain6 = FastChain(FastDense(dim,hidden,Flux.σ),FastDense(hidden,hidden,Flux.σ),FastDense(hidden,1))
                       
strategy = GridTraining(dx)


discretization = PhysicsInformedNN([chain1, chain2, chain3, chain4, chain5, chain6], strategy)

pde_system = PDESystem(eqs, bcs, domains, [t,x], [c1, c2, c3, c4, c5, c6])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS(),cb=cb,maxiters=100)
