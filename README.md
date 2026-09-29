<img src="logo.png" alt="NonGaussianDMRG.jl" width="400">

> [!WARNING]
> This package is in early, active development. It is being made public now for
> early community access alongside our arXiv submission (arXiv:2607.17934). While
> intended for broader research use, this is pre-release software. A comprehensive
> documentation is still pending. Expect breaking changes as the core features are
> developed.
>
> Please use with caution until the official release.

# NonGaussianDMRG.jl: Variational non-Gaussian solutions to interacting spin-boson models

**NonGaussianDMRG.jl** is a Julia package for the ground-state properties of strongly correlated spin-boson systems, such as the Dicke model and its extensions with spin-spin interactions from cavity QED. It combines a non-Gaussian variational ansatz for the bosonic mode with the density matrix renormalization group (DMRG, tensor networks) for the many-body spin state.

This package implements the method described in the following papers:
>  1. **Variational Non-Gaussian Approach to Interacting Spin-Boson Models**,
> JP Mendonça, Y Wang, and K Jachymski,
> https://doi.org/10.48550/arXiv.2607.17934
> 2. **Role of Matter Interactions in Superradiant Phenomena**,
> JP Mendonça, K Jachymski, Y Wang,
> Physical Review Letters 135 (13), 133601 (2025)

If you use this package in your research, please cite the manuscripts above and the software itself (use the "Cite this repository" button on GitHub).

## Overview

The simulation of spin-boson models is often hindered by the infinite-dimensional nature of the bosonic Hilbert space and the presence of strong many-body correlations. Standard approaches typically rely on truncation (limiting photon numbers) or mean-field approximations that neglect entanglement.

This framework addresses these challenges by combining:

1.  **Non-Gaussian Variational Ansatz:**

$$|\psi_{\rm NGS}\rangle = U_{\lambda} \left( U_{\mathrm{GS}} |0\rangle_{\rm b}\otimes|\phi\rangle_{\rm s} \right).$$

The bosonic sector is treated using a variational manifold combining displacement and squeezing, forming a Gaussian state. A (Lang-Firsov-inspired) dressing transformation introduces entanglement between the two subsystems. This replaces explicit photon number truncation.

2.  **Many-Body Solver:** The many-body spin state $|\phi\rangle_{\rm s}$ is obtained, with no further approximations, solving the effective spin Hamiltonian

$$H_{\rm eff} = \langle \psi_{\rm b} | U_\lambda^\dagger H U_\lambda | \psi_{\rm b} \rangle,$$

using DMRG via `ITensors.jl` and `ITensorMPS.jl`, capturing spin-spin correlations.

A self-consistent optimization loop then minimizes the variational energy with respect to both the variational parameters and the spin wavefunction, leading to an accurate approximation of the full spin-boson ground state beyond mean-field theory.

## Features

  * **Truncation-free Bosons:** Effectively handles regimes with macroscopic photon occupancy (e.g., superradiance) without convergence issues related to basis size.
  * **Reduced Computational Cost:** The necessary bond dimension to obtain the ground state is significantly reduced, as the MPS solver only has to deal with an effective spin-only model.
  * **Arbitrary Spin Sector:** Any graph of spin-spin couplings (range, inhomogeneity, disorder) and site-dependent local fields.
  * **Modular Architecture:** The package strictly separates the physical model graph from the variational manifolds (`GS`, `HomogeneousNGS`) and the tensor-network solver.

## Current Scope

The package currently implements the single-mode, homogeneous case: extended Dicke models with an arbitrary spin sector. The general model and ansatz targeted by future versions are given in the [Roadmap](#roadmap).

### Model

$$H = \omega a^\dagger a + \sum_i \varepsilon_i s^z_i + \frac{2g}{\sqrt{N}} \sum_i s^x_i (a + a^\dagger) - \sum_{i\lt j} \sum_{\alpha} J^{\alpha}_{ij} s^\alpha_i s^\alpha_j ,$$

with spin-1/2 operators $s^\alpha = \sigma^\alpha/2$. Each term is set by one function:

| Term | Function |
|---|---|
| $\omega$ | `add_boson!(sys, ω)` |
| $\varepsilon_i$ | `set_epsilon!(sys, i, ε)` |
| $g$ (independent of $N$) | `set_dicke_coupling!(sys, g)`, after `add_boson!` |
| $J^\alpha_{ij}$ | `add_spin_coupling!(sys, α, i, j, J)` with `α` one of `:x`, `:y`, `:z` |

With the minus sign above, $J^\alpha_{ij} > 0$ is ferromagnetic along $\alpha$. In Pauli-matrix notation, a term $-J_\sigma \sigma^\alpha_i \sigma^\alpha_j$ corresponds to $J^\alpha_{ij} = 4 J_\sigma$.

`add_spin_boson_coupling!(sys, m, α, i, val)` stores a general per-site coupling exactly as it enters $H$. The current ansatz requires a uniform $x$-coupling on every site of a single mode, which is what `set_dicke_coupling!` builds; other configurations raise an error at solve time.

### Ansatz

The implemented ansatz (`HomogeneousNGS`) is

$$|\psi_{\rm NGS}\rangle = e^{i\lambda \frac{g'}{\omega} S^x p} e^{-i\Delta_x p} e^{-\frac{i}{2}\xi (xp+px)} |0\rangle_{\rm b}\otimes|\phi\rangle_{\rm s},$$

with $x = (a+a^\dagger)/\sqrt{2}$, $p = i(a^\dagger-a)/\sqrt{2}$, $S^x = \sum_i s^x_i$ and $g' = 2g/\sqrt{N/2}$. The displacement $\Delta_x$ is minimized analytically; the squeezing $\xi$ and the dressing $\lambda$ are optimized numerically. The Gaussian baseline (`GS`) fixes $\lambda = \xi = 0$: the spin and boson sectors form a product state, while the spin state itself remains correlated.

## Installation

Requires Julia 1.12 or newer. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Jpedromend/NonGaussianDMRG.jl")
```

To modify the source, use `Pkg.develop(url="https://github.com/Jpedromend/NonGaussianDMRG.jl")` instead, which gives an editable local clone.

## Usage

### Minimal example: Dicke-Ising chain

```julia
using NonGaussianDMRG
using Printf

N = 20
sys = SpinBosonSystem(N)

add_boson!(sys, 1.0)           # ω = 1
set_dicke_coupling!(sys, 0.8)  # g = 0.8

for i in 1:N
    set_epsilon!(sys, i, 1.0)
    if i < N
        add_spin_coupling!(sys, :z, i, i+1, -2.0)  # antiferromagnetic Ising, J^z = -2
    end
end

# Cold start: random initial MPS and parameters (seed for reproducibility)
E0, state_ngs, stats = solve_ngs(sys;
                                 seed=1234,
                                 return_stats=true,
                                 nsweeps=100,
                                 maxdim=[10, 20, 50, 100],
                                 cutoff=1e-10)

n_tot = expect_ngs("n", state_ngs, sys)
avg_sz_tot = sum(expect_ngs("Sz", state_ngs, sys))

@printf("Energy per spin:        %.8f\n", E0 / N)
@printf("Photon number per spin: %.4f\n", n_tot / N)
@printf("<S^z> per spin:         %.4f\n", avg_sz_tot / N)
println("Converged in $(stats.iterations) iterations.")
```

Keywords not used by the package itself (`nsweeps`, `maxdim`, `cutoff`, `observer`, ...) are passed directly to `ITensorMPS.dmrg`, and `state_ngs.psi_spin` is an ITensorMPS `MPS`; see the [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) documentation.

### Gaussian baseline

```julia
E0_gs, state_gs = solve_ngs(sys, GS(); nsweeps=100, maxdim=[10, 20, 50, 100], cutoff=1e-10)
```

Since $\lambda = \xi = 0$ lies inside the NGS manifold, the converged Gaussian state can seed the NGS optimization, which then only lowers the energy relative to it (up to DMRG truncation):

```julia
init = NGSState(state_gs.psi_spin, HomogeneousNGS(0.0, 0.0))
E0, state_ngs = solve_ngs(sys, init; nsweeps=100, maxdim=[10, 20, 50, 100], cutoff=1e-10)
```

### Parameter sweeps with warm starts

Passing a converged state to `solve_ngs` starts from its MPS and parameters, which is much faster than a cold start and follows a branch through a transition:

```julia
g_values = 0.2:0.05:0.8
set_dicke_coupling!(sys, first(g_values))
E0, state = solve_ngs(sys; seed=1234, nsweeps=50, maxdim=[10, 20, 50])

for g in g_values[2:end]
    set_dicke_coupling!(sys, g)
    E0, state = solve_ngs(sys, state; nsweeps=50, maxdim=[10, 20, 50])
    @printf("g = %.2f   E0/N = %.8f\n", g, E0 / N)
end
```

A warm start from a `GS` result keeps $\lambda = \xi = 0$ fixed; to continue in the NGS manifold instead, wrap its MPS in `HomogeneousNGS(0.0, 0.0)` as in the Gaussian baseline example above.

## Repository Structure

  * `src/`: Source code for the library.
      * `NonGaussianDMRG.jl`: Main module definition and exports.
      * `types.jl`: Core abstract hierarchy, the `SpinBosonSystem` graph builder and its mutators, variational manifolds (`GS`, `HomogeneousNGS`), and the unified `NGSState`.
      * `physics.jl`: Analytical operator sums, ansatz coefficients and model validation, the effective Hamiltonian, and the energy functional.
      * `solver.jl`: The self-consistent optimization loop, cold/warm start dispatch, and the DMRG engine.
      * `observables.jl`: `expect_ngs` and `correlation_matrix_ngs`, with the ansatz dressing applied analytically.
  * `notebooks/`: A standalone, step-by-step implementation of the method for the Dicke model.

## Roadmap

Future versions target the general spin-boson Hamiltonian with $N$ spins and $M$ bosonic modes,

$$H = \sum_{m=1}^{M} \omega_m a_m^\dagger a_m + \sum_{i=1}^{N} \varepsilon_i s_i^z + \sum_{i=1}^{N}\sum_{m=1}^{M} g_{im} s_i^\mu (a_m + a_m^\dagger) - \sum_{i\ltj} \sum_{\gamma\in\lbrace x,y,z \rbrace} J_{ij}^\gamma s_i^\gamma s_j^\gamma ,$$

with a fixed coupling axis $\mu \in \{x, y, z\}$ and couplings $g_{im}$, $J_{ij}^\gamma$ that may be inhomogeneous, disordered, or long-ranged. The corresponding ansatz keeps the structure $|\psi_{\rm NGS}\rangle = U_{\lambda} ( U_{\mathrm{GS}} |0\rangle_{\rm b}\otimes|\phi\rangle_{\rm s} )$, with a multimode Gaussian layer

$$U_{\mathrm{GS}} = \exp\left(i R^T \sigma \Delta_R\right) \exp\left(-\frac{i}{2} R^T \xi R\right), \qquad R = (x_1, \dots, x_M, p_1, \dots, p_M)^T,$$

where $\Delta_R$ collects the displacements of all quadratures, $\sigma$ is the symplectic form and $\xi$ is a real symmetric $2M \times 2M$ squeezing matrix, and a site- and mode-resolved dressing

$$U_\lambda = \exp\left( i \sum_{i=1}^{N} \sum_{m=1}^{M} s_i^\mu \lambda_{im} p_m \right).$$

The current version is the case $M = 1$, $\mu = x$, $g_{i1} = 2g/\sqrt{N}$ and $\lambda_{i1} = \lambda g'/\omega$. The general effective Hamiltonian and energy functional are derived in the appendix of the first paper above. Examples within this class include multimode cavities and spin-Holstein models, where each spin couples to its own local mode ($M = N$, $g_{im} = g_i \delta_{im}$).

## License

Apache License 2.0; see `LICENSE`.
