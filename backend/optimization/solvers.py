"""Solvers for :class:`SchedulingProblem`: QAOA, greedy, simulated annealing, MILP (and brute force).

Every solver returns the same structure, evaluated with the *original* cost model (not the QUBO
energy), so the comparison is like-for-like. Nothing here asserts that QAOA is better; the report
gives measured cost, feasibility, runtime, and the gap to the MILP optimum.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Callable

import numpy as np

from .problem import SchedulingProblem, qubo_energy

log = logging.getLogger("qadt.optimization")
Progress = Callable[[float, str], None]


def _result(problem: SchedulingProblem, name: str, x: np.ndarray, seconds: float, **extra) -> dict:
    ev = problem.evaluate(x)
    return {"solver": name, "runtime_seconds": round(seconds, 4), **ev, **extra}


def solve_greedy(problem: SchedulingProblem) -> dict:
    t0 = time.time()
    x = np.zeros(problem.n_decision, int)
    load = np.zeros(problem.n_slots, int)
    benefit = problem.cost_none[:, None] - problem.cost_maint
    for i, t in sorted(itertools.product(range(problem.n_assets), range(problem.n_slots)),
                       key=lambda it: -benefit[it]):
        if benefit[i, t] <= 0:
            break
        if x[i * problem.n_slots:(i + 1) * problem.n_slots].any() or load[t] >= problem.capacity:
            continue
        x[problem.var(i, t)] = 1
        load[t] += 1
    return _result(problem, "greedy", x, time.time() - t0)


def solve_milp(problem: SchedulingProblem) -> dict:
    from scipy.optimize import Bounds, LinearConstraint, milp

    t0 = time.time()
    n, T = problem.n_assets, problem.n_slots
    c = (problem.cost_maint - problem.cost_none[:, None]).ravel()
    rows, ub = [], []
    for i in range(n):
        r = np.zeros(n * T); r[i * T:(i + 1) * T] = 1; rows.append(r); ub.append(1)
    for t in range(T):
        r = np.zeros(n * T); r[t::T] = 1; rows.append(r); ub.append(problem.capacity)
    res = milp(c, constraints=LinearConstraint(np.array(rows), -np.inf, np.array(ub)),
               integrality=np.ones(n * T), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return _result(problem, "milp_highs", np.round(res.x).astype(int), time.time() - t0, optimal=True)


def solve_brute_force(problem: SchedulingProblem, max_vars: int = 20) -> dict:
    t0 = time.time()
    nd = problem.n_decision
    if nd > max_vars:
        raise ValueError(f"brute force limited to {max_vars} decision variables")
    best, best_x = np.inf, None
    for bits in itertools.product((0, 1), repeat=nd):
        x = np.array(bits)
        ev = problem.evaluate(x)
        if ev["feasible"] and ev["total_expected_cost"] < best:
            best, best_x = ev["total_expected_cost"], x
    return _result(problem, "brute_force", best_x, time.time() - t0, optimal=True)


def solve_simulated_annealing(problem: SchedulingProblem, sweeps: int = 3000, restarts: int = 8,
                              seed: int = 0) -> dict:
    t0 = time.time()
    Q, offset, nv = problem.to_qubo()
    S = Q + Q.T - np.diag(np.diag(Q))   # symmetric couplings; diag holds linear terms
    rng = np.random.default_rng(seed)
    scale = max(np.abs(np.diag(Q)).max(), 1.0)
    best_e, best_x = np.inf, None
    for _ in range(restarts):
        x = rng.integers(0, 2, nv)
        e = qubo_energy(Q, offset, x)
        for k in range(sweeps):
            temp = scale * (1.0 - k / sweeps) + 1e-9
            j = rng.integers(nv)
            # energy change of flipping bit j
            delta = (1 - 2 * x[j]) * (S[j, j] + S[j] @ x - S[j, j] * x[j])
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                x[j] ^= 1
                e += delta
                if e < best_e:
                    best_e, best_x = e, x.copy()
    return _result(problem, "simulated_annealing", best_x, time.time() - t0, qubo_energy=float(best_e),
                   sweeps=sweeps, restarts=restarts)


def _ising(Q: np.ndarray, offset: float):
    """x = (1 − z)/2 → H = Σ h_i Z_i + Σ J_ij Z_i Z_j + const."""
    n = len(Q)
    h, J, const = np.zeros(n), {}, offset
    for i in range(n):
        h[i] -= Q[i, i] / 2
        const += Q[i, i] / 2
        for j in range(i + 1, n):
            if Q[i, j]:
                J[(i, j)] = Q[i, j] / 4
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                const += Q[i, j] / 4
    return h, J, const


def solve_qaoa(problem: SchedulingProblem, layers: int = 2, steps: int = 80, seed: int = 0,
               device: str = "lightning.qubit", top_k: int = 256, max_qubits: int = 16,
               progress: Progress | None = None) -> dict:
    import pennylane as qml
    from scipy.optimize import minimize

    progress = progress or (lambda f, m: None)
    t0 = time.time()
    Q, offset, nv = problem.to_qubo()
    if nv > max_qubits:
        raise ValueError(f"QAOA needs {nv} qubits; the simulator limit is {max_qubits}. Reduce assets/slots/capacity.")
    h, J, const = _ising(Q, offset)
    norm = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=0), 1e-12)
    coeffs = [h[i] / norm for i in range(nv) if h[i]] + [v / norm for v in J.values()]
    ops = [qml.PauliZ(i) for i in range(nv) if h[i]] + [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in J]
    cost_h = qml.Hamiltonian(coeffs, ops)
    mixer_h = qml.qaoa.x_mixer(range(nv))
    try:
        dev = qml.device(device, wires=nv)
    except Exception:
        device = "default.qubit"
        dev = qml.device(device, wires=nv)

    def layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    def ansatz(params):
        for w in range(nv):
            qml.Hadamard(w)
        qml.layer(layer, layers, params[0], params[1])

    @qml.qnode(dev)
    def expectation(params):
        ansatz(params)
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def probabilities(params):
        ansatz(params)
        return qml.probs(wires=range(nv))

    rng = np.random.default_rng(seed)
    history = []

    def objective(flat):
        val = float(expectation(flat.reshape(2, layers)))
        history.append(val)
        if len(history) % 10 == 0:
            progress(min(0.9, len(history) / steps), f"QAOA evaluation {len(history)}: ⟨H⟩={val:.4f}")
        return val

    # Linear-ramp initialisation (annealing-inspired schedule), then one random restart.
    frac = (np.arange(layers) + 0.5) / layers
    inits = [np.concatenate([0.8 * frac, 0.8 * (1 - frac)]), rng.uniform(0, np.pi / 2, 2 * layers)]
    best = None
    for init in inits:
        res = minimize(objective, init, method="COBYLA", options={"maxiter": steps // 2, "rhobeg": 0.3})
        if best is None or res.fun < best.fun:
            best = res
    params = best.x.reshape(2, layers)
    probs = np.asarray(probabilities(params))
    order = np.argsort(probs)[::-1][:top_k]
    candidates = []
    for idx in order:
        bits = np.array([int(b) for b in np.binary_repr(idx, nv)])
        ev = problem.evaluate(bits)
        candidates.append((ev["feasible"], ev["total_expected_cost"], idx, bits))
    feasible = [c for c in candidates if c[0]]
    chosen = min(feasible, key=lambda c: c[1]) if feasible else candidates[0]
    specs = qml.specs(expectation, level="device")(params).resources
    feasible_mass = float(sum(probs[c[2]] for c in candidates if c[0]))
    return _result(
        problem, "qaoa", chosen[3], time.time() - t0, qubits=nv, layers=layers,
        circuit_depth=int(specs.depth), gate_count=int(specs.num_gates), device=device,
        function_evaluations=len(history), final_expectation=float(best.fun),
        expectation_history=[round(v, 5) for v in history[:: max(1, len(history) // 100)]],
        probability_of_selected=float(probs[chosen[2]]), feasible_probability_top_k=feasible_mass,
        params={"gamma": params[0].tolist(), "alpha": params[1].tolist()})


def compare_solvers(problem: SchedulingProblem, qaoa_layers: int = 2, qaoa_steps: int = 80,
                    max_qubits: int = 16, progress: Progress | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    Q, _, nv = problem.to_qubo()
    out = {"problem": problem.describe(), "qubo_variables": nv,
           "decision_variables": problem.n_decision, "results": []}
    progress(0.02, "MILP (exact)")
    out["results"].append(solve_milp(problem))
    progress(0.05, "greedy")
    out["results"].append(solve_greedy(problem))
    progress(0.08, "simulated annealing")
    out["results"].append(solve_simulated_annealing(problem))
    try:
        out["results"].append(solve_qaoa(problem, qaoa_layers, qaoa_steps, max_qubits=max_qubits,
                                         progress=lambda f, m: progress(0.1 + 0.85 * f, m)))
    except ValueError as exc:
        out["results"].append({"solver": "qaoa", "error": str(exc)})
    opt = out["results"][0]["total_expected_cost"]
    for r in out["results"]:
        if "total_expected_cost" in r:
            r["gap_to_optimum"] = float(r["total_expected_cost"] - opt)
            r["gap_percent"] = float(100 * (r["total_expected_cost"] - opt) / opt) if opt else 0.0
    feasible = [r for r in out["results"] if r.get("feasible")]
    out["best_solver"] = min(feasible, key=lambda r: (r["total_expected_cost"], r["runtime_seconds"]))["solver"]
    out["note"] = ("MILP is exact for this problem size; QAOA and simulated annealing are heuristics. "
                   "Gaps are measured, not assumed. QAOA's plan is the cheapest feasible bitstring among its "
                   "most probable outcomes (classical post-selection); 'probability_of_selected' shows how much "
                   "amplitude the circuit itself placed on that plan.")
    progress(1.0, "done")
    return out
