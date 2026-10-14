# src/physics.jl

"""
Builds the ITensor OpSums of the model pieces. Returns purely analytical operator sums;
no tensor networks are instantiated here.
"""
function build_base_operators(model::SpinBosonSystem)
    N = model.N

    os_z = OpSum()
    for i in 1:N
        if abs(model.epsilon[i]) > 1e-14
            os_z += model.epsilon[i], "Sz", i
        end
    end

    os_x = OpSum()
    for i in 1:N
        os_x += 1.0, "Sx", i
    end

    os_Jx, os_Jy_y, os_Jy_z, os_Jz_z, os_Jz_y = OpSum(), OpSum(), OpSum(), OpSum(), OpSum()

    ### addition/change for the optimization test ###
    # Sy_i Sy_j = -(iSy_i)(iSy_j), with the real operator iSy = i Sy, so H_eff and the MPS stay real
    for c in model.spin_couplings
        if c.axis == :x
            os_Jx += c.val, "Sx", c.i, "Sx", c.j
        elseif c.axis == :y
            os_Jy_y += -c.val, "iSy", c.i, "iSy", c.j
            os_Jy_z += c.val, "Sz", c.i, "Sz", c.j
        elseif c.axis == :z
            os_Jz_z += c.val, "Sz", c.i, "Sz", c.j
            os_Jz_y += -c.val, "iSy", c.i, "iSy", c.j
        end
    end
    ##########

    return (
        Hz = os_z, Hx = os_x,
        HJx = os_Jx, HJy_y = os_Jy_y, HJy_z = os_Jy_z, HJz_z = os_Jz_z, HJz_y = os_Jz_y
    )
end

"""
MPOs of the base operator sums, built once per solve; `nothing` for an empty sum.
Hx2 = (S^x)^2 is the square of the S^x MPO.
"""
function build_base_mpos(base_ops, sites)
    mpo(os) = length(os) == 0 ? nothing : MPO(os, sites)
    Hx = mpo(base_ops.Hx)

    return (
        Hz = mpo(base_ops.Hz), Hx = Hx, Hx2 = apply(Hx, Hx; cutoff=1e-15),
        HJx = mpo(base_ops.HJx), HJy_y = mpo(base_ops.HJy_y), HJy_z = mpo(base_ops.HJy_z),
        HJz_z = mpo(base_ops.HJz_z), HJz_y = mpo(base_ops.HJz_y)
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
Spin-sector averages ⟨·⟩_s entering the energy functional, as expectation values of the base MPOs.
"""
function spin_averages(mpos, psi::MPS)
    ev(M) = isnothing(M) ? 0.0 : real(inner(psi', M, psi))

    return (
        avg_sx_tot = ev(mpos.Hx),
        sx2_tot    = ev(mpos.Hx2),
        E_field    = ev(mpos.Hz),
        E_xx       = ev(mpos.HJx),
        Jy_yy      = ev(mpos.HJy_y),
        Jy_zz      = ev(mpos.HJy_z),
        Jz_yy      = ev(mpos.HJz_y),
        Jz_zz      = ev(mpos.HJz_z)
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
Assembles the effective spin Hamiltonian as the sum of the base MPOs scaled by the current
coefficients (direct sum, then SVD compression with `cutoff`). Empty pieces are left out.
The compression leaves the whole operator norm on one tensor; it is spread back as a factor
sqrt(2) per site, the weight of the spin-1/2 identity, so the tensors have the scale of an
OpSum-built MPO (the DMRG noise term depends on this scale).
"""
function effective_hamiltonian(mpos, model::SpinBosonSystem, ansatz::HomogeneousNGS, avg_sx; cutoff=1e-12)
    co = coefficients(model, ansatz)

    terms = (
        (co.c_field,               mpos.Hz),
        (-2.0 * co.K_mf * avg_sx,  mpos.Hx),
        (co.c_quad,                mpos.Hx2),
        (-1.0,                     mpos.HJx),
        (-co.c_plus,               mpos.HJy_y),
        (-co.c_minus,              mpos.HJy_z),
        (-co.c_plus,               mpos.HJz_z),
        (-co.c_minus,              mpos.HJz_y)
    )
    H = [c * M for (c, M) in terms if !isnothing(M)]

    H_eff = truncate!(+(H...; alg="directsum"); cutoff=cutoff)

    oc, f = orthocenter(H_eff), sqrt(2.0)
    for j in eachindex(H_eff)
        j == oc || (H_eff[j] *= f)
    end
    H_eff[oc] /= f^(length(H_eff) - 1)

    return H_eff
end
