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

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# exp(-(x^2+y^2)/0.1)
# div(v_vector * c) = dx(uc) + dy(vc)
# Equations, initial and boundary conditions
eqs = Dt(c(t,x,y)) ~ D * (Dxx(c(t,x,y)) + Dyy(c(t,x,y))) - (u*Dx(c(t,x,y)) + v*Dy(c(t,x,y))) + R

bcs = [
        c(0, x, y) ~ exp(-(x^2+y^2)/0.1), #cos(π*x) * cos(π*y) + 1.0 #,
        c(t, x_min, y) ~ c(t, x_max, y),
        c(t, x, y_min) ~ c(t, x, y_max)
      ]

#first step
domains = [t ∈ IntervalDomain(0.0,0.001),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)
          ]

dim = length(domains)
hidden = 16
chain = FastChain(FastDense(dim, hidden, Flux.σ),
                  FastDense(hidden, hidden, Flux.σ),
                  FastDense(hidden, 1))

quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubaCuhre(),
                                                   reltol = 1e-4, abstol = 1e-2,
                                                   maxiters = 20)

discretization = NeuralPDE.PhysicsInformedNN(chain,quadrature_strategy)
pde_system = PDESystem(eqs, bcs, domains, [t,x,y], [c])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=1500)



nx = 128
ny = 128
dx = (x_max-x_min) / (nx - 1)
dy = (y_max-y_min) / (ny -1)
dt = 0.01
phi = discretization.phi
ts,xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
analytic_solve(t, x, y) = exp(-((x-t)^2+(y-t)^2)/0.1)
function plots_init()
    c_predict = reshape([ phi([0.0, x, y], res.minimizer)[1] for x in xs for y in ys], length(xs), length(ys))
    extract_f(x, y) = exp(-((x)^2+(y)^2)/0.1)
    c_real = reshape([extract_f(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    p1 = plot(xs, ys, c_predict)
    p2 = plot(xs, ys, c_real)
    plot(p1,p2)
end

function plot_comparison()

    # Animate
    anim = @animate for (i, t) in enumerate(0:dt:t_max)
        @info "Animating frame $i..."
        c_real = reshape([analytic_solve(t,x,y) for x in xs for y in ys], (length(xs),length(ys)))
        c_predict = reshape([phi([t, x, y], res.minimizer)[1] for x in xs for y in ys], length(xs), length(ys))
        error_c = abs.(c_predict .- c_real)
        title = @sprintf("Advection-diffusion predict t = %.3f", t)
        p1 = heatmap(xs, ys, c_predict, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal, clims=(0, 1.))
        title = @sprintf("c_real")
        p2 = heatmap(xs, ys, c_real, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal, clims=(0, 1.))
        title = @sprintf("error_c")
        p3 = heatmap(xs, ys, error_c, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal,clims=(0, 0.3))
        plot(p1,p2,p3)
    end
    gif(anim, "advection_diffusion_2d_pinn_comparison.gif", fps=15)
end

plots_init()

plot_comparison()


#second step
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubaCuhre(),
                                                   reltol = 1e-5, abstol = 1e-3,
                                                   maxiters = 30)
domains = [t ∈ IntervalDomain(0.0,0.01),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)
          ]
initθ_ = res.minimizer
discretization = NeuralPDE.PhysicsInformedNN(chain,quadrature_strategy;init_params = initθ_)
pde_system = PDESystem(eqs, bcs, domains, [t,x,y], [c])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=1000)


plot_comparison()

#3rd step
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubaCuhre(),
                                                   reltol = 1e-8, abstol = 1e-6,
                                                   maxiters = 30)
domains = [t ∈ IntervalDomain(0.0,0.1),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)
          ]
initθ_ = res.minimizer
discretization = NeuralPDE.PhysicsInformedNN(chain,quadrature_strategy;init_params = initθ_)
pde_system = PDESystem(eqs, bcs, domains, [t,x,y], [c])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=200)


plot_comparison()

#4rd step
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubaCuhre(),
                                                   reltol = 1e-8, abstol = 1e-6,
                                                   maxiters = 30)
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)
          ]
initθ_ = res.minimizer
discretization = NeuralPDE.PhysicsInformedNN(chain,quadrature_strategy;init_params = initθ_)
pde_system = PDESystem(eqs, bcs, domains, [t,x,y], [c])
prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=100)


plot_comparison()



# Animate
anim = @animate for (i, t) in enumerate(0:dt:t_max)
    @info "Animating frame $i..."
    c_real = reshape([analytic_solve(t,x,y) for x in xs for y in ys], (length(xs),length(ys)))
    title = @sprintf("Advection-diffusion extract t = %.3f", t)
    heatmap(xs, ys, c_real, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal, clims=(0, 1.))
end
gif(anim, "advection_diffusion_2d_pinn_extract.gif", fps=15)

# Animate
anim = @animate for (i, t) in enumerate(0:dt:t_max)
    @info "Animating frame $i..."
    c_predict = reshape([phi([t, x, y], res.minimizer)[1] for x in xs for y in ys], length(xs), length(ys))
    title = @sprintf("Advection-diffusion predict t = %.3f", t)
    heatmap(xs, ys, c_predict, label="", title=title , xlims=(-1, 1), ylims=(-1, 1), color=:thermal, clims=(0, 1.0))
end
gif(anim, "advection_diffusion_2d_pinn_predict.gif", fps=15)
