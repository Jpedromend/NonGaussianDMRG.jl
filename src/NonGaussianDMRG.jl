module NGS

using ITensors, ITensorMPS, Optim, LinearAlgebra, Printf

# Load scripts
include("types.jl")
include("physics.jl")
include("solver.jl")
include("observables.jl")

# --- Public API ---

# Core Types
export SpinBosonSystem, GS, HomogeneousNGS, NGSState, SolverStats

# Builder Mutators
export add_boson!, set_epsilon!, add_spin_coupling!, add_spin_boson_coupling!

# Solver Methods
export solve_ngs

# Observables
export expect_ngs, correlation_matrix_ngs

end