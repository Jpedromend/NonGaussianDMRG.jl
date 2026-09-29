# src/observables.jl

"""
Computes expectation values ⟨·⟩_NGS. Site-resolved for "Sx", "Sy", "Sz"; total for "n".
"""
function expect_ngs(op::String, state::NGSState, model::SpinBosonSystem)
    psi = state.psi_spin

    if op == "Sx"
        return expect(psi, "Sx")

    elseif op == "Sy"
        co = coefficients(model, state.var_params)
        return co.c_field .* real.(expect(complex(psi), "Sy"))

    elseif op == "Sz"
        co = coefficients(model, state.var_params)
        return co.c_field .* expect(psi, "Sz")

    elseif op == "n"
        co = coefficients(model, state.var_params)
        xi, lmd = state.var_params.xi, state.var_params.lambda

        sx_tot = real(sum(expect(psi, "Sx")))
        sx2_tot = real(sum(correlation_matrix(psi, "Sx", "Sx")))

        return sinh(xi)^2 + (co.K / co.omega) * ((1.0 - lmd^2) * sx_tot^2 + lmd^2 * sx2_tot)

    else
        error("Observable '$op' is not implemented.")
    end
end

"""
Computes correlation matrices ⟨s^α_i s^α_j⟩_NGS.
"""
function correlation_matrix_ngs(op1::String, op2::String, state::NGSState, model::SpinBosonSystem)
    psi = state.psi_spin

    if op1 == "Sx" && op2 == "Sx"
        return correlation_matrix(psi, "Sx", "Sx")

    elseif (op1, op2) in (("Sy", "Sy"), ("Sz", "Sz"))
        co = coefficients(model, state.var_params)
        corr_yy = real.(correlation_matrix(complex(psi), "Sy", "Sy"))
        corr_zz = correlation_matrix(psi, "Sz", "Sz")

        op1 == "Sy" && return co.c_plus .* corr_yy .+ co.c_minus .* corr_zz
        return co.c_plus .* corr_zz .+ co.c_minus .* corr_yy

    else
        error("Correlation '$op1-$op2' is not implemented.")
    end
end
