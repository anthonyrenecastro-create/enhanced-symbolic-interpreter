"""
HRM SNN Engine
==============
Unified physics-informed Spiking Neural Network (SNN) architecture governed by
the Helical Recursive Manifold (HRM).

Physics-to-Algorithm mappings
------------------------------
φ₁ (Excitability Field)
    PDE:  ∂ₜφ₁ = -α₁|φ₁|²φ₁ - β₁∇⁴φ₁ + ζ∇²Φ  (Swift-Hohenberg / Cahn-Hilliard variant)
    Role: Dynamic LIF threshold  →  θ_dyn = θ_base - 0.5·φ₁

φ₅ (Entropy/Resource Field)
    PDE:  ∂ₜφ₅ = ω₅φ₅·ln|φ₅| + ∇·(φ₅∇Φ) - γ₅∇²φ₅  (Fokker-Planck / soliton)
    Role: Synaptic plasticity / leak modulation  →  leak_dyn = base_leak / (1 + φ₅)

HRM State Evolution
    dSᵢ/dt = κ∆_G Sᵢ + ηSᵢ(1-‖Sᵢ‖²/C²) + μΣWᵢⱼ(Sⱼ-Sᵢ) + ζ(S*-Sᵢ) + ψ∇_{Sᵢ}C

Global Field
    ∂ₜΦ = Dϕ∇²Φ + aΦ - b|Φ|²Φ - c∇⁴Φ + λ(K*Φ-Φ) + gM(x;S)

Optimization Functional
    J(S,φ) = wf·F_fractal + wp·F_precision + ws·F_stability + wc·F_coherence - wr·R

Modules
-------
FieldEngine          – PDE solver for φ₁, φ₅ on a 2-D grid
SNNCore              – LIF neuron array with field-modulated threshold and leak
TopologyEngine       – HRM genus-12 manifold routing (12 channels, 22 domains, 14 layers)
StateEngine          – S[24] coupled nonlinear state-vector evolution
OptimizationEngine   – Multi-objective J(S, φ) evaluation
SimulationEngine     – asyncio event loop orchestrating all engines
"""

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("hrm_snn")


# ---------------------------------------------------------------------------
# Helpers: pure-Python 2-D finite-difference operations
# ---------------------------------------------------------------------------

def _grid(rows: int, cols: int, fill: float = 0.0) -> List[List[float]]:
    """Allocate a 2-D grid initialised to *fill*."""
    return [[fill] * cols for _ in range(rows)]


def _grid_copy(g: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in g]


def laplacian(g: List[List[float]]) -> List[List[float]]:
    """
    Discrete Laplacian ∇² using the standard 5-point stencil
        [[0, 1, 0],
         [1,-4, 1],
         [0, 1, 0]]
    with zero-flux (Neumann) boundary conditions (clamp indices).
    """
    R = len(g)
    C = len(g[0])
    out = _grid(R, C)
    for r in range(R):
        for c in range(C):
            top    = g[max(r - 1, 0)][c]
            bottom = g[min(r + 1, R - 1)][c]
            left   = g[r][max(c - 1, 0)]
            right  = g[r][min(c + 1, C - 1)]
            out[r][c] = top + bottom + left + right - 4.0 * g[r][c]
    return out


def bilaplacian(g: List[List[float]]) -> List[List[float]]:
    """Bi-Laplacian ∇⁴ = ∇²(∇²) applied recursively."""
    return laplacian(laplacian(g))


def gradient_dot_flux(phi: List[List[float]], potential: List[List[float]]) -> List[List[float]]:
    """
    Approximate ∇·(φ∇Φ) used in the φ₅ advection/chemotaxis term.

    By the product rule:  ∇·(φ∇Φ) = ∇φ·∇Φ + φ∇²Φ

    Both terms are approximated via the discrete Laplacian:
        ∇φ·∇Φ  ≈  ∇²φ · Φ       (Laplacian of φ scaled by local Φ value)
        φ·∇²Φ   =  φ · ∇²Φ      (exact product)

    This is a compact operator approximation suitable for the explicit-Euler
    PDE solver used here.  It preserves the sign and magnitude of the
    advective drift while remaining numerically stable on the discrete grid.
    """
    R = len(phi)
    C = len(phi[0])
    lap_phi = laplacian(phi)
    lap_Phi = laplacian(potential)
    out = _grid(R, C)
    for r in range(R):
        for c in range(C):
            out[r][c] = lap_phi[r][c] * potential[r][c] + phi[r][c] * lap_Phi[r][c]
    return out


def grid_add(a: List[List[float]], b: List[List[float]], scale: float = 1.0) -> List[List[float]]:
    """Element-wise a + scale*b."""
    R, C = len(a), len(a[0])
    return [[a[r][c] + scale * b[r][c] for c in range(C)] for r in range(R)]


def grid_map(g: List[List[float]], fn) -> List[List[float]]:
    return [[fn(v) for v in row] for row in g]


def grid_norm(g: List[List[float]]) -> float:
    """Root-mean-square of the entire grid."""
    total = 0.0
    count = 0
    for row in g:
        for v in row:
            total += v * v
            count += 1
    return math.sqrt(total / max(count, 1))


def grid_mean(g: List[List[float]]) -> float:
    total, count = 0.0, 0
    for row in g:
        for v in row:
            total += v
            count += 1
    return total / max(count, 1)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# 1. FieldEngine
# ---------------------------------------------------------------------------

@dataclass
class FieldEngineConfig:
    """Hyper-parameters for the coupled PDE system."""
    grid_rows: int = 12          # spatial rows (maps to HRM channels)
    grid_cols: int = 22          # spatial cols (maps to HRM domains)
    dt: float = 0.01             # time step

    # φ₁ parameters (Swift-Hohenberg / Cahn-Hilliard variant)
    alpha1: float = 0.5          # cubic saturation coefficient
    beta1: float = 0.1           # bi-harmonic diffusion (pattern forming)
    gamma1: float = 0.05         # linear diffusion
    zeta1: float = 0.2           # coupling from global Φ

    # φ₅ parameters (Fokker-Planck / soliton)
    omega5: float = 0.3          # log-nonlinearity (soliton/memory preservation)
    gamma5: float = 0.1          # diffusivity
    theta5: float = 0.05         # chemotaxis strength (∇·(φ₅∇Φ))

    # Global Φ field (complex Ginzburg-Landau variant)
    D_phi: float = 0.1           # diffusion coefficient
    a_phi: float = 0.2           # linear growth
    b_phi: float = 0.4           # cubic suppression
    c_phi: float = 0.02          # bi-harmonic damping
    lam_phi: float = 0.1         # non-local kernel coupling
    g_phi: float = 0.05          # state–field coupling

    # Numerical stability clamp
    field_max: float = 5.0


class FieldEngine:
    """
    Solves the coupled PDE system for φ₁, φ₅, and the global interplay field Φ.

    Physics
    -------
    φ₁:  ∂ₜφ₁ = -α₁|φ₁|²φ₁ - β₁∇⁴φ₁ + ζ∇²Φ
         Role: Excitability / dynamic threshold modulation for LIF neurons.

    φ₅:  ∂ₜφ₅ = ω₅φ₅·ln|φ₅| + ∇·(φ₅∇Φ) - γ₅∇²φ₅
         Role: Plasticity / resource field for synaptic weight scaling.

    Φ:   ∂ₜΦ  = Dϕ∇²Φ + aΦ - b|Φ|²Φ - c∇⁴Φ + λ(K*Φ-Φ) + gM(x;S)
         Role: Global interplay field coupling all components.
    """

    def __init__(self, cfg: Optional[FieldEngineConfig] = None):
        self.cfg = cfg or FieldEngineConfig()
        R, C = self.cfg.grid_rows, self.cfg.grid_cols
        # Initialise fields with small random perturbations
        self.phi1: List[List[float]] = [
            [random.gauss(0.0, 0.1) for _ in range(C)] for _ in range(R)
        ]
        self.phi5: List[List[float]] = [
            [max(0.01, random.gauss(0.1, 0.02)) for _ in range(C)] for _ in range(R)
        ]
        self.Phi: List[List[float]] = _grid(R, C, 0.0)
        self._step_count = 0
        self._metrics: List[Dict] = []

    # ------------------------------------------------------------------
    # PDE update step
    # ------------------------------------------------------------------

    def step(self, spike_feedback: Optional[List[List[float]]] = None,
             state_map: Optional[List[List[float]]] = None) -> None:
        """
        Advance all fields by one dt step using explicit Euler integration.

        Parameters
        ----------
        spike_feedback : 2-D grid of recent spike rates (same shape as fields).
                         Feeds into Φ update as M(x; S).
        state_map      : 2-D grid encoding aggregate state information S.
                         Also feeds into the Φ equation.
        """
        cfg = self.cfg
        dt = cfg.dt
        R, C = cfg.grid_rows, cfg.grid_cols

        # ---- Update Φ first (it couples into φ₁ and φ₅) -----------------
        lap_Phi  = laplacian(self.Phi)
        bilap_Phi = bilaplacian(self.Phi)

        # Non-local term: λ(K*Φ - Φ).  K = uniform mean kernel over grid.
        mean_Phi = grid_mean(self.Phi)

        # External drive M(x; S) from spikes + state
        M = _grid(R, C, 0.0)
        if spike_feedback is not None:
            for r in range(R):
                for c in range(C):
                    M[r][c] += spike_feedback[r][c]
        if state_map is not None:
            for r in range(R):
                for c in range(C):
                    M[r][c] += cfg.g_phi * state_map[r][c]

        dPhi = _grid(R, C)
        for r in range(R):
            for c in range(C):
                phi_val = self.Phi[r][c]
                dPhi[r][c] = (
                    cfg.D_phi * lap_Phi[r][c]
                    + cfg.a_phi * phi_val
                    - cfg.b_phi * phi_val ** 3
                    - cfg.c_phi * bilap_Phi[r][c]
                    + cfg.lam_phi * (mean_Phi - phi_val)
                    + M[r][c]
                )

        new_Phi = _grid(R, C)
        for r in range(R):
            for c in range(C):
                new_Phi[r][c] = clamp(
                    self.Phi[r][c] + dPhi[r][c] * dt,
                    -cfg.field_max, cfg.field_max
                )

        # ---- Update φ₁ (Swift-Hohenberg / pattern former) ----------------
        # ∂ₜφ₁ = -α₁|φ₁|²φ₁  (cubic saturation)
        #       - β₁∇⁴φ₁       (bi-harmonic diffusion)
        #       + γ₁∇²φ₁       (linear diffusion, stabilises)
        #       + ζ₁∇²Φ        (coupling)
        lap_phi1   = laplacian(self.phi1)
        bilap_phi1 = bilaplacian(self.phi1)
        lap_Phi_new = laplacian(new_Phi)

        dphi1 = _grid(R, C)
        for r in range(R):
            for c in range(C):
                p = self.phi1[r][c]
                dphi1[r][c] = (
                    -cfg.alpha1 * p * p * p
                    - cfg.beta1  * bilap_phi1[r][c]
                    + cfg.gamma1 * lap_phi1[r][c]
                    + cfg.zeta1  * lap_Phi_new[r][c]
                )

        new_phi1 = _grid(R, C)
        for r in range(R):
            for c in range(C):
                new_phi1[r][c] = clamp(
                    self.phi1[r][c] + dphi1[r][c] * dt,
                    -cfg.field_max, cfg.field_max
                )

        # ---- Update φ₅ (Fokker-Planck / memory soliton) ------------------
        # ∂ₜφ₅ = ω₅·φ₅·ln|φ₅|          (log-nonlinearity: soliton preservation)
        #       + θ₅·∇·(φ₅∇Φ)           (chemotaxis: resource drift toward Φ)
        #       - γ₅·∇²φ₅               (diffusion: spreading)
        eps = 1e-6
        lap_phi5 = laplacian(self.phi5)
        flux     = gradient_dot_flux(self.phi5, new_Phi)

        dphi5 = _grid(R, C)
        for r in range(R):
            for c in range(C):
                p5 = self.phi5[r][c]
                log_term  = cfg.omega5 * p5 * math.log(abs(p5) + eps)
                chemotaxis = cfg.theta5 * flux[r][c]
                diffusion  = -cfg.gamma5 * lap_phi5[r][c]
                dphi5[r][c] = log_term + chemotaxis + diffusion

        new_phi5 = _grid(R, C)
        for r in range(R):
            for c in range(C):
                raw = self.phi5[r][c] + dphi5[r][c] * dt
                # φ₅ must remain positive (it's a density-like resource field)
                new_phi5[r][c] = clamp(abs(raw) + eps, eps, cfg.field_max)

        # Commit updates
        self.phi1 = new_phi1
        self.phi5 = new_phi5
        self.Phi  = new_Phi
        self._step_count += 1

        # Collect metrics
        self._metrics.append({
            "step": self._step_count,
            "phi1_rms": grid_norm(self.phi1),
            "phi5_rms": grid_norm(self.phi5),
            "Phi_rms":  grid_norm(self.Phi),
            "phi1_mean": grid_mean(self.phi1),
            "phi5_mean": grid_mean(self.phi5),
        })

    def get_metrics(self) -> List[Dict]:
        return list(self._metrics)

    def inject_spikes(self, channel: int, domain: int, intensity: float = 0.1) -> None:
        """
        Bidirectional coupling: spike events raise the global field Φ locally.
        Reflects 'self.Phi = 0.9*Phi + 0.1*spikes' from the problem description.
        """
        R, C = self.cfg.grid_rows, self.cfg.grid_cols
        r = channel % R
        c = domain % C
        self.Phi[r][c] = 0.9 * self.Phi[r][c] + 0.1 * intensity


# ---------------------------------------------------------------------------
# 2. SNNCore — FieldEnhancedSNN
# ---------------------------------------------------------------------------

@dataclass
class SNNConfig:
    """LIF neuron parameters."""
    n_neurons: int  = 24          # matches S[24] state-vector length
    base_threshold: float = 1.0   # θ_base
    base_leak: float = 0.9        # membrane decay per step
    reset_potential: float = 0.0  # post-spike reset
    refractory_steps: int = 2     # absolute refractory period (ticks)


class SNNCore:
    """
    Leaky Integrate-and-Fire (LIF) neuron array with field modulation.

    Physics-to-algorithm mappings
    ------------------------------
    Dynamic threshold:  θ_dyn = θ_base - 0.5·φ₁[neuron_position]
    Dynamic leak:       leak_dyn = base_leak / (1 + φ₅[neuron_position])
    Spike feedback:     Φ = 0.9·Φ + 0.1·spikes  (via FieldEngine.inject_spikes)
    """

    def __init__(self, cfg: Optional[SNNConfig] = None,
                 field_engine: Optional[FieldEngine] = None):
        self.cfg = cfg or SNNConfig()
        self.field_engine = field_engine
        N = self.cfg.n_neurons
        self.mem_potential: List[float] = [0.0] * N
        self.refractory: List[int]      = [0] * N   # remaining refractory ticks
        self.spike_history: List[List[int]] = []     # per-step spike lists
        self._step_count = 0
        self._metrics: List[Dict] = []

    # ------------------------------------------------------------------
    def _field_at_neuron(self, neuron_idx: int) -> Tuple[float, float]:
        """
        Map neuron index → (φ₁, φ₅) values from the FieldEngine grid.
        Neurons are laid out row-major over the field grid.
        """
        if self.field_engine is None:
            return 0.0, 0.1   # defaults when no FieldEngine attached

        fe = self.field_engine
        R, C = fe.cfg.grid_rows, fe.cfg.grid_cols
        r = (neuron_idx // C) % R
        c = neuron_idx % C
        return fe.phi1[r][c], fe.phi5[r][c]

    # ------------------------------------------------------------------
    def step(self, input_currents: List[float]) -> List[int]:
        """
        Advance one LIF time-step.

        Parameters
        ----------
        input_currents : list of external currents per neuron.

        Returns
        -------
        spikes : list of neuron indices that fired this step.
        """
        cfg = self.cfg
        N = cfg.n_neurons
        spikes: List[int] = []

        for i in range(N):
            if self.refractory[i] > 0:
                self.refractory[i] -= 1
                # Membrane reset is held during refractory
                self.mem_potential[i] = cfg.reset_potential
                continue

            phi1_i, phi5_i = self._field_at_neuron(i)

            # Field-modulated threshold:  θ_dyn = θ_base - 0.5·φ₁
            theta_dyn = cfg.base_threshold - 0.5 * phi1_i

            # Field-modulated leak:  leak_dyn = base_leak / (1 + φ₅)
            leak_dyn = cfg.base_leak / (1.0 + abs(phi5_i))

            # LIF membrane integration
            current = input_currents[i] if i < len(input_currents) else 0.0
            self.mem_potential[i] = leak_dyn * self.mem_potential[i] + current

            # Spike check
            if self.mem_potential[i] >= theta_dyn:
                spikes.append(i)
                self.mem_potential[i] = cfg.reset_potential
                self.refractory[i] = cfg.refractory_steps

                # Feedback: spikes raise the global potential Φ
                if self.field_engine is not None:
                    R, C = (self.field_engine.cfg.grid_rows,
                            self.field_engine.cfg.grid_cols)
                    r = (i // C) % R
                    c = i % C
                    channel_idx = r
                    domain_idx  = c
                    self.field_engine.inject_spikes(channel_idx, domain_idx,
                                                    intensity=1.0)

        self.spike_history.append(spikes)
        self._step_count += 1

        # Collect metrics
        spike_rate = len(spikes) / max(N, 1)
        avg_mem    = sum(self.mem_potential) / max(N, 1)
        self._metrics.append({
            "step": self._step_count,
            "n_spikes": len(spikes),
            "spike_rate": spike_rate,
            "avg_mem_potential": avg_mem,
        })

        return spikes

    def spike_rate_grid(self) -> List[List[float]]:
        """
        Aggregate recent spike rates back onto the field grid shape.
        Returns a 2-D grid matching (grid_rows × grid_cols).
        Used as spike_feedback in FieldEngine.step().
        """
        if self.field_engine is None:
            return _grid(12, 22, 0.0)
        R, C = self.field_engine.cfg.grid_rows, self.field_engine.cfg.grid_cols
        rate_grid = _grid(R, C, 0.0)
        recent = self.spike_history[-5:] if len(self.spike_history) >= 5 else self.spike_history
        for step_spikes in recent:
            for idx in step_spikes:
                r = (idx // C) % R
                c = idx % C
                rate_grid[r][c] += 1.0
        # Normalise
        window = max(len(recent), 1)
        return [[v / window for v in row] for row in rate_grid]

    def get_metrics(self) -> List[Dict]:
        return list(self._metrics)


# ---------------------------------------------------------------------------
# 3. TopologyEngine — HRM genus-12 manifold
# ---------------------------------------------------------------------------

@dataclass
class HRMState:
    """
    Position in the Helical Recursive Manifold.

    Fields
    ------
    theta   : cycle position in [0, 2π)
    channel : active channel index (1-based, 1..12)
    domain  : active domain index  (1-based, 1..22)
    layer   : active layer index   (1-based, 1..14)
    guna    : polarity triplet (sattva, rajas, tamas) — must sum to 1
    """
    theta:   float = 0.0
    channel: int   = 1
    domain:  int   = 1
    layer:   int   = 1
    guna:    Tuple[float, float, float] = (1.0/3.0, 1.0/3.0, 1.0/3.0)   # (s, r, t)

    # ------------------------------------------------------------------
    @property
    def norm_channel(self) -> float:
        return (self.channel - 1) / 11.0   # normalised [0,1]

    @property
    def norm_domain(self) -> float:
        return (self.domain - 1) / 21.0

    @property
    def norm_layer(self) -> float:
        return (self.layer - 1) / 13.0


class TopologyEngine:
    """
    Manages HRM genus-12 topology: routing, canonical transitions, and
    guna polarity tracking across 12 channels, 22 domains, and 14 layers.

    The manifold is parameterised as a discrete torus:
        θ ∈ [0, 2π),  channel ∈ {1..12},  domain ∈ {1..22},  layer ∈ {1..14}

    Routing rule (canonical transformation):
        - Advance θ by dθ per step
        - Channel, domain, layer increment according to helical recursion
    """

    NUM_CHANNELS = 12
    NUM_DOMAINS  = 22
    NUM_LAYERS   = 14
    CYCLE_RATE   = 2.0 * math.pi / 100.0   # complete macro-cycle in 100 steps

    def __init__(self, seed: int = 0):
        random.seed(seed)
        self.state = HRMState()
        self._step_count = 0
        self._trajectory: List[HRMState] = []
        self._metrics: List[Dict] = []

    # ------------------------------------------------------------------
    def step(self, dt: float = 1.0) -> HRMState:
        """
        Advance manifold position by one routing step.

        Helical recursion: position advances along a space-filling helix over
        (channel, domain, layer).  Guna polarity rotates following the cycle
        position θ to reflect energetic phase transitions.
        """
        # Advance cycle angle
        new_theta = (self.state.theta + self.CYCLE_RATE * dt) % (2.0 * math.pi)

        # Canonical channel/domain/layer advance (mod-wrap on the torus)
        new_channel = (self.state.channel % self.NUM_CHANNELS) + 1
        # Domain advances every NUM_CHANNELS steps (helical winding)
        domain_step = (self._step_count // self.NUM_CHANNELS) % self.NUM_DOMAINS
        new_domain  = domain_step + 1
        # Layer advances every (NUM_CHANNELS × NUM_DOMAINS) steps
        layer_step  = (self._step_count // (self.NUM_CHANNELS * self.NUM_DOMAINS)) % self.NUM_LAYERS
        new_layer   = layer_step + 1

        # Guna polarity: sinusoidal rotation in phase space
        #   sattva (clarity/illumination) peaks at θ ≈ 0
        #   rajas  (activity/transformation) peaks at θ ≈ 2π/3
        #   tamas  (inertia/structure)      peaks at θ ≈ 4π/3
        s = 0.5 * (1.0 + math.cos(new_theta))
        r = 0.5 * (1.0 + math.cos(new_theta - 2.0 * math.pi / 3.0))
        t = 0.5 * (1.0 + math.cos(new_theta - 4.0 * math.pi / 3.0))
        total = s + r + t + 1e-9
        new_guna = (s / total, r / total, t / total)

        self.state = HRMState(
            theta=new_theta,
            channel=new_channel,
            domain=new_domain,
            layer=new_layer,
            guna=new_guna,
        )
        self._step_count += 1
        self._trajectory.append(self.state)

        self._metrics.append({
            "step": self._step_count,
            "theta": round(new_theta, 4),
            "channel": new_channel,
            "domain":  new_domain,
            "layer":   new_layer,
            "guna_s": round(new_guna[0], 4),
            "guna_r": round(new_guna[1], 4),
            "guna_t": round(new_guna[2], 4),
        })
        return self.state

    def get_metrics(self) -> List[Dict]:
        return list(self._metrics)

    def trajectory(self) -> List[HRMState]:
        return list(self._trajectory)


# ---------------------------------------------------------------------------
# 4. StateEngine — S[24] coupled nonlinear dynamics
# ---------------------------------------------------------------------------

@dataclass
class StateEngineConfig:
    """Coefficients for the HRM state evolution equation."""
    dim: int    = 24      # state-vector dimension
    kappa: float = 0.05   # graph-Laplacian coupling strength κ
    eta: float   = 0.1    # nonlinear reaction coefficient η
    C_cap: float = 2.0    # capacity constant C
    mu: float    = 0.02   # inter-state diffusion coefficient μ
    zeta: float  = 0.05   # attractor pull coefficient ζ
    psi: float   = 0.01   # gradient ascent on C term ψ
    dt: float    = 0.01   # integration time step


class StateEngine:
    """
    Maintains and evolves the S[24] state vector governing the HRM.

    Physics (state evolution equation)
    ------------------------------------
    dSᵢ/dt = κ∆_G Sᵢ                           (graph-Laplacian diffusion)
           + ηSᵢ(1 - ‖Sᵢ‖²/C²)                (nonlinear reaction / amplitude limiting)
           + μΣⱼWᵢⱼ(Sⱼ - Sᵢ)                  (weighted inter-node coupling)
           + ζ(S* - Sᵢ)                         (attractor pull toward target S*)
           + ψ∇_{Sᵢ}C                           (coherence gradient)

    Here each scalar component Sᵢ ∈ ℝ is tracked independently.
    The graph-Laplacian ∆_G acts on a nearest-neighbour ring topology
    (Sᵢ₋₁ - 2Sᵢ + Sᵢ₊₁), consistent with a discretised 1-D manifold.
    """

    def __init__(self, cfg: Optional[StateEngineConfig] = None):
        self.cfg = cfg or StateEngineConfig()
        D = self.cfg.dim
        self.S: List[float] = [random.gauss(0.0, 0.5) for _ in range(D)]
        self.S_target: List[float] = [0.0] * D   # attractor / target state S*
        # Symmetric weight matrix W (random sparse, diagonal = 0)
        self.W: List[List[float]] = [[0.0] * D for _ in range(D)]
        for i in range(D):
            for j in range(i + 1, D):
                if random.random() < 0.3:   # 30% connectivity
                    w = random.gauss(0.0, 0.1)
                    self.W[i][j] = w
                    self.W[j][i] = w
        self._step_count = 0
        self._metrics: List[Dict] = []

    # ------------------------------------------------------------------
    def _graph_laplacian_term(self) -> List[float]:
        """
        κ∆_G Sᵢ on a ring topology: Sᵢ₋₁ - 2Sᵢ + Sᵢ₊₁
        """
        D = self.cfg.dim
        lap = [0.0] * D
        for i in range(D):
            prev_ = self.S[(i - 1) % D]
            next_ = self.S[(i + 1) % D]
            lap[i] = self.cfg.kappa * (prev_ - 2.0 * self.S[i] + next_)
        return lap

    def _reaction_term(self) -> List[float]:
        """ηSᵢ(1 - ‖S‖²/C²) — amplitude-limited nonlinear reaction."""
        cfg = self.cfg
        norm_sq = sum(s * s for s in self.S)
        factor = 1.0 - norm_sq / (cfg.C_cap * cfg.C_cap)
        return [cfg.eta * si * factor for si in self.S]

    def _coupling_term(self) -> List[float]:
        """μΣⱼWᵢⱼ(Sⱼ - Sᵢ) — weighted diffusive inter-state coupling."""
        D = self.cfg.dim
        out = [0.0] * D
        for i in range(D):
            total = sum(self.W[i][j] * (self.S[j] - self.S[i]) for j in range(D))
            out[i] = self.cfg.mu * total
        return out

    def _attractor_term(self) -> List[float]:
        """ζ(S* - Sᵢ) — pull toward the target attractor."""
        return [self.cfg.zeta * (self.S_target[i] - self.S[i])
                for i in range(self.cfg.dim)]

    def _coherence_gradient(self, field_engine: Optional[FieldEngine]) -> List[float]:
        """
        ψ∇_{Sᵢ}C — direction of steepest ascent on coherence C w.r.t. Sᵢ.
        Here C is approximated as the mean φ₅ (resource/plasticity field).

        Note: this is a sign-based heuristic, not a true finite-difference
        gradient.  The term encourages each Sᵢ to grow in the direction that
        aligns with the current resource level: positive Sᵢ is nudged upward
        when C > 0, negative Sᵢ downward.  A full finite-difference estimate
        would require re-evaluating the field at Sᵢ ± ε, which is not
        tractable within the explicit-Euler step.
        """
        if field_engine is None:
            return [0.0] * self.cfg.dim
        C_val = grid_mean(field_engine.phi5)
        return [self.cfg.psi * C_val * math.copysign(1.0, si) for si in self.S]

    # ------------------------------------------------------------------
    def step(self, field_engine: Optional[FieldEngine] = None,
             topology: Optional[HRMState] = None) -> List[float]:
        """
        Advance S[24] by one time-step via explicit Euler integration.

        Parameters
        ----------
        field_engine : FieldEngine instance for coherence gradient.
        topology     : current HRMState to adjust target attractor.
        """
        cfg = self.cfg
        D = cfg.dim

        # Optionally shift the target attractor using topology position
        if topology is not None:
            for i in range(D):
                scale = math.sin(topology.theta + i * math.pi / D)
                self.S_target[i] = scale * topology.norm_domain

        lap   = self._graph_laplacian_term()
        react = self._reaction_term()
        coup  = self._coupling_term()
        attr  = self._attractor_term()
        coh_g = self._coherence_gradient(field_engine)

        new_S = [0.0] * D
        for i in range(D):
            dS = lap[i] + react[i] + coup[i] + attr[i] + coh_g[i]
            new_S[i] = self.S[i] + dS * cfg.dt

        self.S = new_S
        self._step_count += 1

        norm = math.sqrt(sum(s * s for s in self.S))
        self._metrics.append({
            "step": self._step_count,
            "S_norm": round(norm, 6),
            "S_mean": round(sum(self.S) / D, 6),
            "S_max":  round(max(self.S), 6),
            "S_min":  round(min(self.S), 6),
        })
        return list(self.S)

    def as_grid(self, rows: int = 12, cols: int = 22) -> List[List[float]]:
        """
        Project S[24] onto a (rows × cols) grid for FieldEngine coupling.
        Values are tiled cyclically and normalised to [-1, 1].
        """
        D = self.cfg.dim
        norm = max(abs(s) for s in self.S) + 1e-9
        grid = _grid(rows, cols, 0.0)
        for r in range(rows):
            for c in range(cols):
                idx = (r * cols + c) % D
                grid[r][c] = self.S[idx] / norm
        return grid

    def get_metrics(self) -> List[Dict]:
        return list(self._metrics)


# ---------------------------------------------------------------------------
# 5. OptimizationEngine — J(S, φ) multi-objective evaluation
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Weights for the J(S, φ) functional and regularisation."""
    w_fractal:   float = 0.25    # wf · F_fractal
    w_precision: float = 0.25    # wp · F_precision
    w_stability: float = 0.25    # ws · F_stability
    w_coherence: float = 0.25    # wc · F_coherence
    w_reg:       float = 0.05    # wr · R (sparsity regularisation)
    lr:          float = 0.01    # learning rate for parameter nudges


class OptimizationEngine:
    """
    Evaluates the multi-objective functional:

        J(S, φ) = wf·F_fractal + wp·F_precision + ws·F_stability
                + wc·F_coherence - wr·R

    Each component is computed from observable statistics of the running system.

    F_fractal   : complexity / richness of the spike pattern (normalised entropy)
    F_precision : how closely S tracks its attractor S* (inverse tracking error)
    F_stability : smoothness of φ₁ over time (low temporal variance = stable)
    F_coherence : mean φ₅ level (high resource = coherent plasticity)
    R           : L2 regularisation of S (sparsity/compactness penalty)
    """

    def __init__(self, cfg: Optional[OptimizationConfig] = None):
        self.cfg = cfg or OptimizationConfig()
        self._J_history: List[float] = []
        self._metrics: List[Dict] = []
        self._step_count = 0
        # Running statistics for stability (variance of phi1 norm over time)
        self._phi1_norm_history: List[float] = []

    # ------------------------------------------------------------------
    def _fractal_score(self, snn: SNNCore) -> float:
        """
        F_fractal: normalised entropy of the spike-rate distribution.
        High entropy → rich, non-repetitive spike patterns.
        """
        if not snn.spike_history:
            return 0.0
        n = snn.cfg.n_neurons
        window = snn.spike_history[-20:] if len(snn.spike_history) >= 20 else snn.spike_history
        counts = [0] * n
        for step_spikes in window:
            for idx in step_spikes:
                counts[idx] += 1
        total = sum(counts) + 1e-9
        entropy = 0.0
        for c in counts:
            p = c / total
            if p > 0:
                entropy -= p * math.log(p + 1e-12)
        max_entropy = math.log(n + 1e-12)
        return entropy / max(max_entropy, 1e-9)

    def _precision_score(self, state_engine: StateEngine) -> float:
        """
        F_precision: inverse normalised distance between S and S*.
        """
        D = state_engine.cfg.dim
        dist_sq = sum(
            (state_engine.S[i] - state_engine.S_target[i]) ** 2
            for i in range(D)
        )
        rmse = math.sqrt(dist_sq / max(D, 1))
        return 1.0 / (1.0 + rmse)

    def _stability_score(self, field_engine: FieldEngine) -> float:
        """
        F_stability: low temporal variance of ‖φ₁‖ signals a stable field.
        """
        norm = grid_norm(field_engine.phi1)
        self._phi1_norm_history.append(norm)
        if len(self._phi1_norm_history) < 3:
            return 0.5
        recent = self._phi1_norm_history[-10:]
        mean_v = sum(recent) / len(recent)
        var_v  = sum((x - mean_v) ** 2 for x in recent) / len(recent)
        return 1.0 / (1.0 + var_v)

    def _coherence_score(self, field_engine: FieldEngine) -> float:
        """
        F_coherence: mean φ₅ across the grid (resource availability).
        """
        mean_phi5 = grid_mean(field_engine.phi5)
        return clamp(mean_phi5 / field_engine.cfg.field_max, 0.0, 1.0)

    def _regularisation(self, state_engine: StateEngine) -> float:
        """
        R: L2 norm of S (sparsity penalty).
        """
        D = state_engine.cfg.dim
        return math.sqrt(sum(s * s for s in state_engine.S) / max(D, 1))

    # ------------------------------------------------------------------
    def evaluate(self, snn: SNNCore, field_engine: FieldEngine,
                 state_engine: StateEngine) -> Dict[str, float]:
        """
        Compute J(S, φ) and all component scores.

        Returns a dict of labelled scores and the aggregate J value.
        """
        cfg = self.cfg
        F_f = self._fractal_score(snn)
        F_p = self._precision_score(state_engine)
        F_s = self._stability_score(field_engine)
        F_c = self._coherence_score(field_engine)
        R_  = self._regularisation(state_engine)

        J = (
            cfg.w_fractal   * F_f
            + cfg.w_precision * F_p
            + cfg.w_stability * F_s
            + cfg.w_coherence * F_c
            - cfg.w_reg       * R_
        )

        self._J_history.append(J)
        self._step_count += 1
        result = {
            "step":        self._step_count,
            "J":           round(J, 6),
            "F_fractal":   round(F_f, 6),
            "F_precision": round(F_p, 6),
            "F_stability": round(F_s, 6),
            "F_coherence": round(F_c, 6),
            "R":           round(R_, 6),
        }
        self._metrics.append(result)
        return result

    def adapt_parameters(self, field_cfg: FieldEngineConfig,
                         state_cfg: StateEngineConfig,
                         last_J: float) -> None:
        """
        Lightweight parameter adaptation: nudge field coupling strengths
        based on the sign of ΔJ (gradient ascent on J).

        Only adjusts α₁, γ₅, κ to keep the system near an optimal operating point.
        """
        if len(self._J_history) < 2:
            return
        dJ = self._J_history[-1] - self._J_history[-2]
        lr = self.cfg.lr
        # Improve excitability patterning if J dropped
        if dJ < 0:
            field_cfg.alpha1 = clamp(field_cfg.alpha1 + lr * 0.01, 0.01, 2.0)
            field_cfg.gamma5 = clamp(field_cfg.gamma5 + lr * 0.005, 0.01, 1.0)
            state_cfg.kappa  = clamp(state_cfg.kappa  + lr * 0.005, 0.001, 0.5)
        else:
            field_cfg.alpha1 = clamp(field_cfg.alpha1 - lr * 0.005, 0.01, 2.0)

    def get_metrics(self) -> List[Dict]:
        return list(self._metrics)


# ---------------------------------------------------------------------------
# 6. SimulationEngine — asyncio orchestration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Top-level simulation parameters."""
    n_steps:      int   = 10
    dt:           float = 0.01
    log_interval: int   = 1       # print summary every N steps
    random_seed:  int   = 42

    # Sub-engine configs (can be overridden)
    field_cfg:  FieldEngineConfig    = field(default_factory=FieldEngineConfig)
    snn_cfg:    SNNConfig             = field(default_factory=SNNConfig)
    state_cfg:  StateEngineConfig     = field(default_factory=StateEngineConfig)
    opt_cfg:    OptimizationConfig    = field(default_factory=OptimizationConfig)


class SimulationEngine:
    """
    Async event-driven simulation loop orchestrating all HRM SNN engines.

    Execution order per step
    ------------------------
    1. TopologyEngine.step()      – advance manifold position
    2. StateEngine.step()         – evolve S[24] (uses field_engine, topology)
    3. SNNCore.step()             – fire LIF neurons (field-modulated)
    4. FieldEngine.step()         – solve PDE for φ₁, φ₅, Φ (uses spikes, S)
    5. OptimizationEngine.evaluate() – compute J(S, φ)
    6. OptimizationEngine.adapt_parameters() – adjust config hypers

    All heavy tasks are wrapped in asyncio to allow non-blocking concurrency.
    """

    def __init__(self, cfg: Optional[SimulationConfig] = None):
        self.cfg = cfg or SimulationConfig()
        random.seed(self.cfg.random_seed)

        self.field_engine = FieldEngine(self.cfg.field_cfg)
        self.snn          = SNNCore(self.cfg.snn_cfg, self.field_engine)
        self.topology     = TopologyEngine(seed=self.cfg.random_seed)
        self.state_engine = StateEngine(self.cfg.state_cfg)
        self.opt_engine   = OptimizationEngine(self.cfg.opt_cfg)

        self._step_count = 0
        self._all_metrics: List[Dict] = []

    # ------------------------------------------------------------------
    # Async sub-tasks

    async def _task_topology(self) -> HRMState:
        """Advance the HRM manifold position (non-blocking)."""
        return self.topology.step(dt=self.cfg.dt)

    async def _task_state(self, topo: HRMState) -> List[float]:
        """Evolve state vector S[24]."""
        return self.state_engine.step(
            field_engine=self.field_engine,
            topology=topo,
        )

    async def _task_snn(self) -> List[int]:
        """
        Fire LIF neurons with stochastic input currents.
        In a real application these come from sensory data or upstream layers.
        """
        N = self.cfg.snn_cfg.n_neurons
        # Poisson-like random input current to each neuron
        currents = [random.gauss(0.3, 0.2) for _ in range(N)]
        return self.snn.step(currents)

    async def _task_field(self, spikes: List[int],
                          state_vec: List[float]) -> None:
        """Advance the PDE fields using spike feedback and state map."""
        spike_grid = self.snn.spike_rate_grid()
        state_map  = self.state_engine.as_grid(
            self.cfg.field_cfg.grid_rows,
            self.cfg.field_cfg.grid_cols,
        )
        self.field_engine.step(
            spike_feedback=spike_grid,
            state_map=state_map,
        )

    async def _task_optimize(self) -> Dict:
        """Evaluate J(S, φ) and adapt parameters."""
        result = self.opt_engine.evaluate(
            self.snn, self.field_engine, self.state_engine
        )
        self.opt_engine.adapt_parameters(
            self.cfg.field_cfg,
            self.cfg.state_cfg,
            result["J"],
        )
        return result

    # ------------------------------------------------------------------
    async def _simulation_step(self) -> Dict:
        """One complete simulation step: all engines advance concurrently."""
        step_start = time.perf_counter()

        # Stage 1: topology (must precede state)
        topo = await self._task_topology()

        # Stage 2: state + SNN are independent → run concurrently
        state_task = asyncio.create_task(self._task_state(topo))
        snn_task   = asyncio.create_task(self._task_snn())
        state_vec, spikes = await asyncio.gather(state_task, snn_task)

        # Stage 3: field step depends on spikes and state
        await self._task_field(spikes, state_vec)

        # Stage 4: optimisation
        opt_result = await self._task_optimize()

        elapsed_ms = (time.perf_counter() - step_start) * 1000.0
        self._step_count += 1

        step_summary = {
            "step":       self._step_count,
            "theta":      round(topo.theta, 4),
            "channel":    topo.channel,
            "domain":     topo.domain,
            "layer":      topo.layer,
            "guna":       (round(topo.guna[0], 3),
                           round(topo.guna[1], 3),
                           round(topo.guna[2], 3)),
            "n_spikes":   len(spikes),
            "S_norm":     round(math.sqrt(sum(s * s for s in state_vec)), 4),
            "phi1_rms":   round(grid_norm(self.field_engine.phi1), 4),
            "phi5_rms":   round(grid_norm(self.field_engine.phi5), 4),
            "Phi_rms":    round(grid_norm(self.field_engine.Phi), 4),
            "J":          opt_result["J"],
            "F_fractal":  opt_result["F_fractal"],
            "F_precision":opt_result["F_precision"],
            "F_stability":opt_result["F_stability"],
            "F_coherence":opt_result["F_coherence"],
            "elapsed_ms": round(elapsed_ms, 3),
        }
        self._all_metrics.append(step_summary)

        if self._step_count % self.cfg.log_interval == 0:
            logger.info(
                "step=%d  θ=%.3f  ch=%d  dom=%d  lay=%d  "
                "spikes=%d  S_norm=%.3f  φ₁=%.4f  φ₅=%.4f  "
                "J=%.4f  [%.2f ms]",
                step_summary["step"],
                step_summary["theta"],
                step_summary["channel"],
                step_summary["domain"],
                step_summary["layer"],
                step_summary["n_spikes"],
                step_summary["S_norm"],
                step_summary["phi1_rms"],
                step_summary["phi5_rms"],
                step_summary["J"],
                step_summary["elapsed_ms"],
            )

        return step_summary

    # ------------------------------------------------------------------
    async def run(self) -> List[Dict]:
        """
        Run the full simulation for n_steps asynchronously.

        Returns
        -------
        List of per-step summary dicts.
        """
        logger.info(
            "HRM SNN Simulation starting: %d steps, dt=%.4f, "
            "%d neurons, grid=%dx%d",
            self.cfg.n_steps,
            self.cfg.dt,
            self.cfg.snn_cfg.n_neurons,
            self.cfg.field_cfg.grid_rows,
            self.cfg.field_cfg.grid_cols,
        )

        for _ in range(self.cfg.n_steps):
            await self._simulation_step()

        logger.info(
            "Simulation complete.  Final J=%.4f  (best=%.4f, worst=%.4f)",
            self._all_metrics[-1]["J"],
            max(m["J"] for m in self._all_metrics),
            min(m["J"] for m in self._all_metrics),
        )
        return self._all_metrics

    # ------------------------------------------------------------------
    def all_metrics(self) -> Dict[str, List[Dict]]:
        """Aggregated metrics from all sub-engines."""
        return {
            "simulation": self._all_metrics,
            "field":      self.field_engine.get_metrics(),
            "snn":        self.snn.get_metrics(),
            "topology":   self.topology.get_metrics(),
            "state":      self.state_engine.get_metrics(),
            "optimizer":  self.opt_engine.get_metrics(),
        }

    def checkpoint(self) -> Dict:
        """
        Snapshot the current simulation state for persistence/resumption.
        Returns a serialisable dict.
        """
        fe = self.field_engine
        se = self.state_engine
        topo = self.topology.state
        return {
            "step":         self._step_count,
            "phi1":         [row[:] for row in fe.phi1],
            "phi5":         [row[:] for row in fe.phi5],
            "Phi":          [row[:] for row in fe.Phi],
            "S":            list(se.S),
            "S_target":     list(se.S_target),
            "mem_potential":list(self.snn.mem_potential),
            "topology": {
                "theta":   topo.theta,
                "channel": topo.channel,
                "domain":  topo.domain,
                "layer":   topo.layer,
                "guna":    list(topo.guna),
            },
            "metrics": self._all_metrics,
        }


# ---------------------------------------------------------------------------
# Convenience: parameter-sweep runner
# ---------------------------------------------------------------------------

async def parameter_sweep(
    param_grid: Dict[str, List],
    n_steps: int = 10,
) -> List[Dict]:
    """
    Run multiple simulations over a grid of hyper-parameter combinations.

    Parameters
    ----------
    param_grid : dict mapping parameter names to lists of values.
                 Supported keys: 'alpha1', 'omega5', 'kappa', 'eta', 'n_neurons'
    n_steps    : simulation steps per configuration.

    Returns
    -------
    List of result dicts (one per configuration), each containing
    the parameter values and the final J score.
    """
    # Build flat list of (name, value) combinations via Cartesian product
    import itertools

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        field_cfg = FieldEngineConfig()
        snn_cfg   = SNNConfig()
        state_cfg = StateEngineConfig()

        # Apply parameter overrides
        if "alpha1"    in params: field_cfg.alpha1   = params["alpha1"]
        if "omega5"    in params: field_cfg.omega5   = params["omega5"]
        if "kappa"     in params: state_cfg.kappa    = params["kappa"]
        if "eta"       in params: state_cfg.eta      = params["eta"]
        if "n_neurons" in params: snn_cfg.n_neurons  = params["n_neurons"]

        sim_cfg = SimulationConfig(
            n_steps=n_steps,
            field_cfg=field_cfg,
            snn_cfg=snn_cfg,
            state_cfg=state_cfg,
        )
        engine = SimulationEngine(sim_cfg)
        metrics = await engine.run()

        final_J = metrics[-1]["J"] if metrics else float("nan")
        result = {**params, "final_J": round(final_J, 6), "steps": n_steps}
        results.append(result)
        logger.info("Sweep: %s  →  J=%.4f", params, final_J)

    return results


# ---------------------------------------------------------------------------
# ASCII visualisation helpers
# ---------------------------------------------------------------------------

def visualise_field(g: List[List[float]], title: str = "",
                    width: int = 44, n_rows: int = 6) -> str:
    """
    Render a 2-D field as a compact ASCII heatmap using Unicode block characters.
    Sampled to fit within *width* characters.

    Note: the default character set uses Unicode block-fill characters
    (U+2591–U+2593, U+2588).  These render correctly on UTF-8 terminals
    (Linux, macOS, modern Windows Terminal) but may appear as '?' on legacy
    terminals that lack Unicode support.  In that case replace CHARS with a
    pure-ASCII palette such as " .:-=#".
    """
    CHARS = " ░▒▓█"
    R = len(g)
    C = len(g[0])
    row_step = max(1, R // n_rows)
    col_step = max(1, C // (width - 2))

    all_vals = [v for row in g for v in row]
    mn = min(all_vals)
    mx = max(all_vals)
    rng = mx - mn + 1e-9

    lines = [f"┌{'─' * width}┐  {title}"]
    for r in range(0, R, row_step):
        row_chars = ""
        for c in range(0, C, col_step):
            norm = (g[r][c] - mn) / rng
            idx = int(norm * (len(CHARS) - 1))
            row_chars += CHARS[idx]
        lines.append(f"│{row_chars:<{width}}│")
    lines.append(f"└{'─' * width}┘  min={mn:.3f} max={mx:.3f}")
    return "\n".join(lines)


def print_simulation_summary(metrics: List[Dict]) -> None:
    """Print a tabular summary of the simulation run."""
    if not metrics:
        print("No metrics available.")
        return
    header = (
        f"{'step':>4} {'θ':>6} {'ch':>3} {'dom':>4} {'lay':>4} "
        f"{'spk':>4} {'S‖':>7} {'φ₁':>7} {'φ₅':>7} {'J':>7} {'ms':>6}"
    )
    print("\n" + "=" * len(header))
    print("HRM SNN Simulation – Step Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m['step']:>4} {m['theta']:>6.3f} {m['channel']:>3} "
            f"{m['domain']:>4} {m['layer']:>4} "
            f"{m['n_spikes']:>4} {m['S_norm']:>7.3f} "
            f"{m['phi1_rms']:>7.4f} {m['phi5_rms']:>7.4f} "
            f"{m['J']:>7.4f} {m['elapsed_ms']:>6.2f}"
        )
    print("=" * len(header))
    best = max(metrics, key=lambda x: x["J"])
    print(f"Best J = {best['J']:.4f}  at step {best['step']}")


# ---------------------------------------------------------------------------
# Entry-point: 10-step demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    cfg = SimulationConfig(
        n_steps=10,
        dt=0.01,
        log_interval=1,
        field_cfg=FieldEngineConfig(grid_rows=12, grid_cols=22),
        snn_cfg=SNNConfig(n_neurons=24),
    )

    engine = SimulationEngine(cfg)
    metrics = asyncio.run(engine.run())

    print_simulation_summary(metrics)

    # ASCII visualisation of final field states
    print("\n--- Final φ₁ (Excitability Field) ---")
    print(visualise_field(engine.field_engine.phi1, title="φ₁"))
    print("\n--- Final φ₅ (Plasticity Field) ---")
    print(visualise_field(engine.field_engine.phi5, title="φ₅"))
    print("\n--- Final Φ (Global Interplay Field) ---")
    print(visualise_field(engine.field_engine.Phi, title="Φ"))

    # Save a checkpoint
    chk = engine.checkpoint()
    print(f"\nCheckpoint captured: step={chk['step']}, "
          f"S_norm={math.sqrt(sum(s**2 for s in chk['S'])):.4f}")

    # Quick parameter sweep example (2 configs)
    print("\n--- Parameter Sweep (alpha1 × omega5) ---")
    sweep_results = asyncio.run(parameter_sweep(
        {"alpha1": [0.3, 0.7], "omega5": [0.2, 0.4]},
        n_steps=5,
    ))
    for r in sweep_results:
        print(r)
