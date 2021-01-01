using Printf
using JLD2
using Plots
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.OutputWriters

# Useful for headless plotting with Plots.jl.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = 100

"""
    simulate_advection_diffusion(; Nx, Δt, stop_time, U, κ, initial_condition, output_interval, filename)

Run a 1D advection-diffusion simulation of a passive tracer with `Nx` grid points with a time step
`Δt` value until t = `stop_time` with constant velocity `U` and constant diffusivity `κ`.
The `inital_condition` should be a function of `x` describing the initial profile. Output will be saved to
JLD2 with `filename`.jld2 and animated every `output_interval` units of time.
"""
function simulate_advection_diffusion(; Nx, Δt, stop_time, U, κ, initial_condition, output_interval, filename)
    topo = (Periodic, Periodic, Periodic)
    domain = (x=(-1, 1), y=(0, 1), z=(0, 1))
    grid = RegularCartesianGrid(topology=topo, size=(Nx, 1, 1); domain...)

    model = IncompressibleModel(
        architecture = CPU(),
                grid = grid,
         timestepper = :RungeKutta3,
           advection = WENO5(),
            buoyancy = nothing,
             tracers = :c,
             closure = IsotropicDiffusivity(κ=κ)
    )

    set!(model, u = U, c = (x, y, z) -> initial_condition(x))

    simulation = Simulation(model, Δt=Δt, stop_time=stop_time, iteration_interval=20)

    simulation.output_writers[:tracer] =
        JLD2OutputWriter(model, (c=model.tracers.c,), schedule=TimeInterval(output_interval),
                         dir=@__DIR__, prefix=filename, force=true)

    @info "Running $filename simulation with Nx=$Nx, Δt=$Δt, U=$U, κ=$κ until t=$stop_time..."
    run!(simulation)

    @info "Animating $filename..."
    file = jldopen("advection_diffusion/$filename.jld2")
    iterations = parse.(Int, keys(file["timeseries/t"]))

    anim = @animate for (i, iter) in enumerate(iterations)
        x = xnodes(model.tracers.c)
        t = file["timeseries/t/$iter"]
        c = file["timeseries/c/$iter"][:]

        plot(x, c, linewidth=2, title=@sprintf("t = %.3f", t),
             label="", xlabel="x", ylabel="Tracer", xlims=(-1, 1), ylims=(-1, 1))
    end

    gif(anim, "$filename.gif", fps=15)
    mp4(anim, "$filename.mp4", fps=15)
end

c₀_cosine(x) = cos(π*x)

simulate_advection_diffusion(Nx=32, Δt=0.01, stop_time=2, U=1, κ=0, initial_condition=c₀_cosine,
                             output_interval=0.05, filename="cosine_advection")

simulate_advection_diffusion(Nx=32, Δt=0.01, stop_time=2, U=0, κ=0.1, initial_condition=c₀_cosine,
                             output_interval=0.05, filename="cosine_diffusion")

simulate_advection_diffusion(Nx=32, Δt=0.01, stop_time=2, U=1, κ=0.1, initial_condition=c₀_cosine,
                             output_interval=0.05, filename="cosine_advection_diffusion")
