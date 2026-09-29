# src/solver.jl

# --- Top Level Dispatch ---

"""
Executes the optimization loop from scratch (Full Cold Start).
Initializes a random MPS and random variational guesses, then routes to the Warm Start.
"""
function solve_ngs(model::SpinBosonSystem; seed=nothing, maxdim_init=10, kwargs...)
    seed === nothing || Random.seed!(seed)

    sites = siteinds("S=1/2", model.N)
    psi_init = randomMPS(sites; linkdims=maxdim_init)
    
    # Random guesses for the non-Gaussian parameters
    xi_guess = rand(0.0:0.01:0.1)
    lambda_guess = rand(0.0:0.1:0.5)
    
    state_init = NGSState(psi_init, HomogeneousNGS(xi_guess, lambda_guess))

    return solve_ngs(model, state_init; kwargs...)
end

"""
Executes the standard Gaussian State (GS) loop from scratch. 
"""
function solve_ngs(model::SpinBosonSystem, ::GS; seed=nothing, maxdim_init=10, kwargs...)
    seed === nothing || Random.seed!(seed)

    sites = siteinds("S=1/2", model.N)
    psi_init = randomMPS(sites; linkdims=maxdim_init)
    state_init = NGSState(psi_init, GS())

    return solve_ngs(model, state_init; kwargs...)
end

"""
Executes the optimization loop from a pre-existing state (Warm Start).
Guarantees index matching by extracting sites directly from the provided MPS.
"""
function solve_ngs(model::SpinBosonSystem, state_init::NGSState{HomogeneousNGS}; kwargs...)
    # 1. Guarantee index matching
    sites = siteinds(state_init.psi_spin)

    # 2. Build mathematical graph representations (lightweight operation)
    base_ops = build_base_operators(model) 

    return _solve(model, state_init; sites=sites, base_ops=base_ops, kwargs...)
end

"""
Executes the optimization loop from a pre-existing GS state (Warm Start for Parameter Sweeps).
Forces optimize_params=false to prevent accidental non-Gaussian entanglement.
"""
function solve_ngs(model::SpinBosonSystem, state_init::NGSState{GS}; kwargs...)
    sites = siteinds(state_init.psi_spin)
    base_ops = build_base_operators(model) 

    # Create a temporary zeroed context for the internal math engine
    mock_state = NGSState(state_init.psi_spin, HomogeneousNGS(0.0, 0.0))

    return_stats = get(kwargs, :return_stats, false)

    if return_stats
        E0, mock_ngs, stats = _solve(model, mock_state;
                                     optimize_params=false, sites=sites, base_ops=base_ops, kwargs...)
        return E0, NGSState(mock_ngs.psi_spin, GS()), stats
    else
        E0, mock_ngs = _solve(model, mock_state;
                              optimize_params=false, sites=sites, base_ops=base_ops, kwargs...)
        return E0, NGSState(mock_ngs.psi_spin, GS())
    end

end


# --- Internal Engine ---

"""
Engine for the self-consistent protocol using DMRG.
Absorbs standard ITensors arguments (nsweeps, maxdim, observer, etc.) via dmrg_kwargs.
"""
function _solve(model::SpinBosonSystem, state_init::NGSState{HomogeneousNGS};
                sites,
                base_ops,
                optimize_params=true,
                max_iter=250,
                min_iter=1,
                tol=1e-8,
                return_stats=false,
                outputlevel=1,
                mpo_cutoff=1e-12,
                dmrg_kwargs...) 

    # 1. State Validation
    validate(model, state_init.var_params)

    # 2. Setup Loop Variables
    psi = state_init.psi_spin
    vparams_opt = [state_init.var_params.xi, state_init.var_params.lambda]
    
    spin_obs = spin_averages(model, psi)
    
    converged = false
    final_iter = max_iter
    energy_history = return_stats ? Float64[] : nothing
    
    prev_E0 = Inf
    E0 = energy_cost(model, HomogeneousNGS(vparams_opt[1], vparams_opt[2]), spin_obs)
    
    # 3. Self-Consistent Loop
    for iter in 1:max_iter
        if iter > min_iter && abs(E0 - prev_E0) < tol
            if outputlevel >= 1
                @printf("Converged at step %3d\n", iter - 1)
            end
            converged = true
            final_iter = iter - 1
            break
        end
        
        if outputlevel >= 1
            @printf("Step %3d | E = %.8f | dE = %.2e | xi = %.4f | lam = %.4f\n", 
                    iter, E0, abs(E0-prev_E0), vparams_opt[1], vparams_opt[2])
        end
        prev_E0 = E0

        # Reset Observer State
        if haskey(dmrg_kwargs, :observer)
            _obs = dmrg_kwargs[:observer]
            hasproperty(_obs, :energies) && empty!(_obs.energies)
            hasproperty(_obs, :truncerrs) && empty!(_obs.truncerrs)
        end

        # Assemble MPO using dynamically instantiated parameters
        H_eff = effective_hamiltonian(base_ops, model, HomogeneousNGS(vparams_opt[1], vparams_opt[2]), spin_obs.avg_sx_tot, sites; cutoff=mpo_cutoff)        
        
        # Run ITensors DMRG
        E0, psi = dmrg(H_eff, psi; outputlevel=outputlevel, dmrg_kwargs...)
        
        # Optimize Bosonic Parameters
        spin_obs = spin_averages(model, psi)
        if optimize_params
            cost_fn(x) = energy_cost(model, HomogeneousNGS(x[1], x[2]), spin_obs)
            res = optimize(cost_fn, vparams_opt, LBFGS(), Optim.Options(iterations=100))
            vparams_opt = Optim.minimizer(res)
            E0 = Optim.minimum(res)
        else
            E0 = energy_cost(model, HomogeneousNGS(vparams_opt[1], vparams_opt[2]), spin_obs)
        end

        if return_stats
            push!(energy_history, E0)
        end
    end

    # 4. Output Compilation
    psi_ngs = NGSState(psi, HomogeneousNGS(vparams_opt[1], vparams_opt[2]))

    if return_stats
        stats = SolverStats(converged, final_iter, energy_history)
        return E0, psi_ngs, stats
    else
        return E0, psi_ngs
    end
end
