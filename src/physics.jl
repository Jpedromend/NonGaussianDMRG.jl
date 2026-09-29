# src/physics.jl

"""
Precompiles the ITensor OpSums to avoid redundancies.
Returns purely analytical operator sums; no tensor 
networks are instantiated here.
"""
function build_base_operators(model::SpinBosonSystem)
    N = model.N
    
    os_z = OpSum()
    for i in 1:N
        if abs(model.epsilon[i]) > 1e-14
            os_z += model.epsilon[i], "Sz", i
        end
    end
    
    os_x, os_x2 = OpSum(), OpSum()
    for i in 1:N
        os_x += 1.0, "Sx", i
        for j in 1:N
            os_x2 += 1.0, "Sx", i, "Sx", j
        end
    end
    
    os_Jx, os_Jy_y, os_Jy_z, os_Jz_z, os_Jz_y = OpSum(), OpSum(), OpSum(), OpSum(), OpSum()
    
    for c in model.spin_couplings
        if c.axis == :x
            os_Jx += c.val, "Sx", c.i, "Sx", c.j
        elseif c.axis == :y
            os_Jy_y += c.val, "Sy", c.i, "Sy", c.j
            os_Jy_z += c.val, "Sz", c.i, "Sz", c.j
        elseif c.axis == :z
            os_Jz_z += c.val, "Sz", c.i, "Sz", c.j
            os_Jz_y += c.val, "Sy", c.i, "Sy", c.j
        end
    end

    return (
        Hz = os_z, Hx = os_x, Hx2 = os_x2,
        HJx = os_Jx, HJy_y = os_Jy_y, HJy_z = os_Jy_z, HJz_z = os_Jz_z, HJz_y = os_Jz_y
    )
end

"""
Coefficients of the homogeneous ansatz, shared by `energy_cost`, `effective_hamiltonian`
and the observables. Uses the Dicke coupling g_b = 2g/√N.
"""
function coefficients(model::SpinBosonSystem, ansatz::Union{HomogeneousNGS, GS})
    omega = model.omega[1]
    g_b = model.spin_boson_couplings[1].val
    xi, lmd = ansatz.xi, ansatz.lambda

    K = g_b^2 / omega
    C = exp(-2.0 * xi) * (lmd * g_b / omega)^2

    return (
        omega   = omega,
        K       = K,
        C       = C,
        c_field = exp(-C / 2),
        K_mf    = K * (1.0 - lmd)^2,
        c_quad  = K * lmd * (lmd - 2.0),
        c_plus  = 0.5 * (1.0 + exp(-2.0 * C)),
        c_minus = 0.5 * (1.0 - exp(-2.0 * C))
    )
end

"""
Checks that the model has the structure HomogeneousNGS assumes:
one bosonic mode and a uniform x-coupling on every site.
"""
function validate(model::SpinBosonSystem, ::HomogeneousNGS)
    isempty(model.omega) && error("No bosonic mode: call add_boson! first.")
    length(model.omega) > 1 && error("HomogeneousNGS currently supports a single bosonic mode.")

    c = model.spin_boson_couplings
    isempty(c) && error("No spin-boson coupling: use set_dicke_coupling!.")

    uniform = sort([x.site for x in c]) == 1:model.N &&
        all(x -> x.m == 1 && x.axis == :x && x.val == c[1].val, c)
    uniform || error("HomogeneousNGS currently requires a uniform x-coupling on every site " *
        "(use set_dicke_coupling!); site-resolved couplings are currently not supported.")
    return nothing
end


"""
Spin-sector averages ⟨·⟩_s entering the energy functional.
Computes the full N×N correlation matrices.
"""
function spin_averages(model::SpinBosonSystem, psi::MPS)
    exp_sz = expect(psi, "Sz")
    exp_sx = expect(psi, "Sx")

    corr_xx = correlation_matrix(psi, "Sx", "Sx")
    corr_yy = correlation_matrix(complex(psi), "Sy", "Sy")
    corr_zz = correlation_matrix(psi, "Sz", "Sz")

    E_xx, Jy_yy, Jy_zz, Jz_yy, Jz_zz = 0.0, 0.0, 0.0, 0.0, 0.0

    for coup in model.spin_couplings
        if coup.axis == :x
            E_xx  += coup.val * real(corr_xx[coup.i, coup.j])
        elseif coup.axis == :y
            Jy_yy += coup.val * real(corr_yy[coup.i, coup.j])
            Jy_zz += coup.val * real(corr_zz[coup.i, coup.j])
        elseif coup.axis == :z
            Jz_yy += coup.val * real(corr_yy[coup.i, coup.j])
            Jz_zz += coup.val * real(corr_zz[coup.i, coup.j])
        end
    end

    return (
        avg_sx_tot = real(sum(exp_sx)),
        avg_sz_tot = real(sum(exp_sz)),
        sx2_tot    = real(sum(corr_xx)),
        E_field    = real(dot(model.epsilon, exp_sz)),
        E_xx       = E_xx,
        Jy_yy      = Jy_yy,
        Jy_zz      = Jy_zz,
        Jz_yy      = Jz_yy,
        Jz_zz      = Jz_zz
    )
end

"""
Evaluates the analytical variational energy functional for the Homogeneous non-Gaussian state.
"""
function energy_cost(model::SpinBosonSystem, ansatz::HomogeneousNGS, obs)
    co = coefficients(model, ansatz)

    E_boson = co.omega * sinh(ansatz.xi)^2

    E_sb = co.c_field * obs.E_field -
           co.K_mf * obs.avg_sx_tot^2 +
           co.c_quad * obs.sx2_tot

    E_ss = - obs.E_xx -
           co.c_plus  * (obs.Jy_yy + obs.Jz_zz) -
           co.c_minus * (obs.Jy_zz + obs.Jz_yy)

    return E_boson + E_sb + E_ss
end

"""
Assembles the effective spin Hamiltonian by scaling the pre-built OpSums 
with the updated dressing parameters. The tensor network MPO is compiled 
strictly once per optimization step here.
"""
function effective_hamiltonian(base_ops, model::SpinBosonSystem, ansatz::HomogeneousNGS, avg_sx, sites; cutoff=1e-12)
    co = coefficients(model, ansatz)
    c_field, c_quad = co.c_field, co.c_quad
    c_plus, c_minus = co.c_plus, co.c_minus
    c_mf = -2.0 * co.K_mf * avg_sx

    total_os = c_field * base_ops.Hz +
               c_mf * base_ops.Hx +
               c_quad * base_ops.Hx2 -
               1.0 * base_ops.HJx -
               c_plus * base_ops.HJy_y -
               c_minus * base_ops.HJy_z -
               c_plus * base_ops.HJz_z -
               c_minus * base_ops.HJz_y
            
    return MPO(total_os, sites; cutoff=cutoff)
end
