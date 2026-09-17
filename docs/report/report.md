# Neural Flux-Conservative Solvers for Hyperbolic PDEs

*Jiwon (Keith) Jung — 2026*

## Abstract

Hybrid machine-learning / finite-volume solvers promise to cut the
compute cost of simulating hyperbolic PDEs by learning to take timesteps
past the classical CFL stability limit. We revisit this idea with a single
architectural change: instead of training a network to regress the next
cell-average state directly, we train it to predict the **numerical flux**
at each cell interface, so the state update is the discrete divergence of
those fluxes. This makes mass conservation exact by construction —
independent of training quality — which we verify directly on a randomly
initialized, untrained network. Combined with multi-step rollout training
and noise injection, the resulting `ConservativeFluxNet` takes timesteps
3.6-4.1x past the classical CFL limit on linear advection and shock-forming
Burgers' equation, while matching WENO5's accuracy at 2.5-5.5x lower
wall-clock cost, and holding up over rollouts 4-7.5x longer than it was
trained on.

## 1. Motivation

Explicit finite-volume schemes for hyperbolic conservation laws are
constrained by the CFL condition: the timestep must shrink as the grid is
refined, or as wave speeds grow, to remain stable. This is the main
compute bottleneck in large-scale simulation of fluid dynamics and other
wave-dominated systems. A line of work going back to Bar-Sinai et al.
(2019) asks whether a learned, data-driven discretization can do better —
whether a neural network, trained on high-resolution "truth" data, can
step a coarse simulation forward by a large Δt while remaining accurate
and stable.

This project revisits that question with a specific concern: the original
formulation (and this project's own earlier 2024 iteration, in
TensorFlow/Keras) trained a network to regress the next coarse state
directly. Nothing in that architecture prevents the network from slowly
gaining or losing mass over a long rollout — conservation, if it holds at
all, is a property the network has to *learn* from data, not one the
architecture *guarantees*. For a numerical method whose entire selling
point is "faithfully approximates a conservation law," that is an odd gap.

## 2. Related Work

- **Bar-Sinai, Hoyer, Hickey, Brenner (2019)**, *Learning data-driven
  discretizations for partial differential equations*, PNAS. Trains a
  neural network to produce finite-difference/finite-volume coefficients
  from local data, demonstrating accuracy gains over fixed-coefficient
  schemes at the same resolution. Motivates the "learn the discretization,
  keep the numerical-method structure" approach used here.
- **Li et al. (2020)**, *Fourier Neural Operator for Parametric Partial
  Differential Equations*. A different strategy: learn a resolution-invariant
  operator directly in Fourier space, without an explicit finite-volume
  update. Complementary to this project's approach; a natural head-to-head
  comparison baseline (noted in §7, not completed here).
- **Lu, Jin, Karniadakis (2021)**, *DeepONet: Learning nonlinear operators*.
  Another operator-learning approach, learning a mapping between function
  spaces rather than a local update rule.
- **Brandstetter, Worrall, Welling (2022)**, *Message Passing Neural PDE
  Solvers*, and **Sanchez-Gonzalez et al. (2020)**, *Learning to Simulate
  Complex Physics*. Both identify and address rollout instability in
  learned simulators via training-time noise injection ("the pushforward
  trick") — the technique this project's rollout training directly adopts.
- Recent flux-conservative / entropy-stable neural solver work (e.g.
  conservative-flux neural network formulations for hyperbolic PDEs)
  converges on the same idea used here: parameterize the flux, not the
  state, so the numerical method's own conservation structure is inherited
  by the learned model rather than approximated by it.

## 3. Method

### 3.1 Governing equations and classical baselines

We consider scalar 1D hyperbolic conservation laws `u_t + f(u)_x = 0` on a
periodic domain, discretized into `N` cell averages. Two cases:

- **Linear advection**, `f(u) = a u` — trivial exact solution, used to
  validate the full pipeline before anything is hard.
- **Inviscid Burgers' equation**, `f(u) = u^2/2` — forms shocks in finite
  time from smooth initial data, and is the case that actually exercises
  the "cut compute cost past the CFL limit" claim.

Four classical finite-volume schemes are implemented from scratch as both
training-data generators and comparison baselines, sharing one
reconstruct → Riemann-solve → SSP-RK3 skeleton
(`src/neuralfv/solvers/classical/`):

- **Upwind** (1st order) and **Lax-Friedrichs** — cheap, diffusive.
- **MUSCL** with a minmod limiter (2nd order, the practical "fair fight"
  for the network).
- **WENO5** (5th order in smooth regions) — the high-order ground-truth
  reference solver and the strongest classical baseline.

All four are verified against theoretical convergence order and exact
mass conservation in `tests/test_classical_solvers.py` before being trusted
as training-data generators or baselines.

### 3.2 ConservativeFluxNet

`ConservativeFluxNet` (`src/neuralfv/solvers/neural/flux_net.py`) is a
small stencil CNN: a periodic-padded convolution gathers a local window of
cell averages around each interface, followed by a pointwise MLP producing
one scalar flux per interface. The state update is

```
u_i^{n+1} = u_i^n - (Δt/Δx) (F_{i+1/2} - F_{i-1/2})
```

Every predicted flux is added to one cell and subtracted from its
neighbor, so `sum(u) * Δx` is invariant under this update for *any*
network weights. `tests/test_flux_net_conservation.py` proves this
directly on a freshly initialized, untrained network, decoupling "is the
architecture correct" from "did training work."

### 3.3 Training data

Training pairs are generated the same way the original 2024 project did —
by simulating a traditional numerical method — applied at the specific
resolution/timestep gap the network needs to close
(`src/neuralfv/data/generate_dataset.py`): a fine grid (8x the coarse
resolution) is evolved with WENO5 at its own CFL-stable timestep, then
block-averaged down to the coarse grid at the coarse network's much larger
Δt. Averaging preserves cell averages exactly, so this produces
dynamically-consistent "true coarse trajectory" data without any separate
closure model. Initial conditions are random band-limited Fourier series
(amplitudes decaying as 1/k), giving smooth but diverse data, including
initial slopes that steepen into shocks for Burgers.

### 3.4 Rollout training with noise injection

A network trained on a single accurate one-step prediction can still
diverge once its own small errors start compounding over a long rollout.
Following Brandstetter et al. (2022) and Sanchez-Gonzalez et al. (2020),
training unrolls the model for several steps (`src/neuralfv/solvers/neural/rollout.py`)
and perturbs each intermediate state with noise scaled to the data itself
before feeding it back in — forcing the network to see, and correct for,
inputs that resemble its own compounding error rather than only clean
ground truth. Rollout length is increased on a curriculum (1 step up to 8
steps over training) for stability. This is what makes the "many steps
past CFL" claim in §5 a real, stable statement rather than a one-step
curiosity: both trained models remain stable well past their 8-step
training horizon (empirically verified out to 60 large steps).

## 4. Experimental Setup

Both models: `stencil_half_width=3` (6-cell gather window), 32 hidden
channels, `n_coarse=64`, `downsample_ratio=8` (512-cell fine grid), Adam
(lr 1e-3), trained on 200-320 synthetic trajectories with an 8-step rollout
curriculum and Gaussian noise injection (see
`src/neuralfv/training/configs/`). Evaluation uses 6 held-out test initial
conditions (unseen seed), a 32-large-step rollout, and reports:

1. **Accuracy vs. cost**: relative L2 error against the fine-grid reference
   vs. wall-clock time, for the network and for each classical scheme
   swept across resolutions {16, 32, 64, 128, 256}.
2. **Speedup at matched accuracy**: interpolating the WENO5 resolution
   sweep to the network's error level, then dividing that wall-clock time
   by the network's own.
3. **Effective CFL number**: Δt used by the network, divided by the
   classical CFL-limited Δt at the same resolution.
4. **Conservation drift**: `|mass(t) - mass(0)|` over the rollout.

## 5. Results

| PDE | Effective CFL | Rel. L2 error | Speedup vs. WENO5 (matched accuracy) |
|---|---|---|---|
| Linear advection | 4.07x | 4.77% | 5.52x |
| Burgers' (shock) | 3.63x | 5.05% | 2.55x |

On the accuracy-vs-cost plane (`figures/burgers_pareto.png`,
`figures/advection_pareto.png`), the network's (time, error) point sits to
the left of every classical scheme's curve at comparable accuracy: cheaper
than upwind at any accuracy upwind reaches, and competitive with
MUSCL/WENO5's low-resolution end while taking far fewer, far larger
timesteps.

**Conservation.** The network's mass drift over the full rollout tops out
at `1.4e-7` (advection) and `6.3e-8` (Burgers). The classical WENO5
solver's drift, for comparison, is `~5e-16` — both are "exact" in the
sense that matters (conservation error many orders of magnitude below
solution error), but they differ because the network runs in **float32**
while the classical solvers run in NumPy's default **float64**; the
drift floor scales with each dtype's own machine epsilon, not with the
architecture's correctness. Relative to the ~5% solution error, `6e-8`
conservation drift is off by nearly 6 orders of magnitude — conservation
is not a meaningfully binding constraint on solution quality here, which
is exactly the point of building it into the architecture rather than
penalizing its violation in the loss.

**Rollout stability.** Trained with an 8-step curriculum, both networks
were evaluated to 60 large steps without diverging (see
`figures/*_error_growth.png` for the 32-step reported window): error grows
gradually and stays within roughly one order of magnitude of the
matched-resolution WENO5 solver's own error growth, rather than blowing up
as an untrained-for-robustness rollout typically would.

**Qualitative shock behavior** (`figures/burgers_snapshot.png`): at 64
cells, the network resolves both shocks in the test profile about as
sharply as upwind at the same resolution, while tracking the smooth ramp
regions between shocks visibly more closely — consistent with the
quantitative L2 gap.

## 6. Demo design

`demo/web/` is a static page comparing the network against classical
baselines scenario-by-scenario, with playback over the rollout and a live
accuracy-vs-cost chart. It uses **precomputed trajectories**
(`demo/export_demo_payloads.py`), not live in-browser inference, for three
reasons: (1) it guarantees the numbers shown always exactly match this
report and the README — no risk of a browser runtime (e.g. ONNX.js)
numerically drifting from the reported PyTorch results; (2) it avoids
ONNX.js/TF.js operator-coverage gaps that would otherwise eat implementation
time better spent on the model; (3) a static payload page is trivially
reliable to demo live ("here's the link, it just works"), with zero
runtime ML dependency. A live "draw your own initial condition" mode via
ONNX Runtime Web is a natural follow-up (§7) — `ConservativeFluxNet` is
small enough (a few hundred KB) that export should be straightforward —
but isn't required for the comparisons this report makes.

## 7. Limitations & Future Work

- **1D and scalar only.** Both benchmarks are 1D scalar conservation laws.
  A system of conservation laws (e.g. the 1D shallow-water equations, with
  coupled fluxes across multiple fields) is the natural next benchmark,
  and was scoped out of this iteration to keep the core architectural
  claim (exact conservation + stable long rollouts) tight and well-tested
  rather than spread thin.
- **No neural-operator comparison baseline.** A Fourier Neural Operator
  trained on the same data, using the `neuraloperator` PyTorch-ecosystem
  library, would directly test whether the flux-conservative inductive
  bias used here buys anything over a more generic (but non-conservative)
  operator-learning approach at the same data budget. Deferred rather than
  dropped.
- **Wall-clock numbers are single-CPU, single-trajectory.** Both the
  network and the classical solvers were timed un-batched, in pure
  Python/NumPy/PyTorch, on one CPU core. The network's real advantage
  (batched GPU inference across many trajectories or a much larger grid,
  where fixed Python/kernel-launch overhead amortizes away) is not what
  these numbers measure; they measure realistic single-trajectory
  scripting cost, which is a fair but conservative comparison.
- **Single training Δt per model.** Each network is trained and evaluated
  at one fixed large Δt (chosen at ~4x the coarse CFL limit). Testing
  generalization across a range of Δt values at inference time — and
  training a single network to handle several — is unexplored.
- **A live in-browser demo mode** (ONNX Runtime Web, user-drawn initial
  conditions) would strengthen the interactive demo beyond the precomputed
  scenarios currently shown.

## References

1. Bar-Sinai, Y., Hoyer, S., Hickey, J., & Brenner, M. P. (2019). Learning
   data-driven discretizations for partial differential equations. *PNAS*,
   116(31), 15344-15349.
2. Li, Z., Kovachki, N., Azizzadenesheli, K., et al. (2020). Fourier Neural
   Operator for Parametric Partial Differential Equations. *arXiv:2010.08895*.
3. Lu, L., Jin, P., & Karniadakis, G. E. (2021). DeepONet: Learning
   nonlinear operators for identifying differential equations based on the
   universal approximation theorem of operators. *Nature Machine
   Intelligence*, 3, 218-229.
4. Brandstetter, J., Worrall, D., & Welling, M. (2022). Message Passing
   Neural PDE Solvers. *ICLR 2022*.
5. Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., et al. (2020). Learning to
   Simulate Complex Physics with Graph Networks. *ICML 2020*.
6. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid
   Dynamics* (3rd ed.). Springer.
7. Jiang, G.-S., & Shu, C.-W. (1996). Efficient Implementation of Weighted
   ENO Schemes. *Journal of Computational Physics*, 126(1), 202-228.
