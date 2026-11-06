"""Qiskit execution of the VQC blocks used by the quantum sequence models.

PennyLane trains the circuits (autodiff through ``default.qubit``). This module rebuilds the *same*
circuit gate by gate in Qiskit —

    AngleEmbedding(RY)  →  StronglyEntanglingLayers: Rot(φ,θ,ω) = RZ(ω)·RY(θ)·RZ(φ) on every wire,
                           then CNOT(i, (i + r) mod n) with range r = (l mod (n−1)) + 1 in layer l

— and evaluates ⟨Z_i⟩ with Qiskit's exact ``Statevector``. :func:`qiskit_execution` swaps every circuit of
a trained model to Qiskit for inference, so predictions can be reproduced by an independent quantum
SDK (and the circuits exported, e.g. to OpenQASM, for hardware runs). Qiskit uses little-endian qubit
order; wire ``i`` here is Qiskit qubit ``i`` and expectation values are read per qubit accordingly.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


def vqc_circuit(inputs, weights):
    """Qiskit ``QuantumCircuit`` equal to the PennyLane VQC block for one input vector."""
    from qiskit import QuantumCircuit

    x = np.asarray(inputs, float)
    w = np.asarray(weights, float)            # (layers, n_qubits, 3)
    n = len(x)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(x[i]), i)
    for layer in range(w.shape[0]):
        for i in range(n):
            phi, theta, omega = w[layer, i]
            qc.rz(float(phi), i)
            qc.ry(float(theta), i)
            qc.rz(float(omega), i)
        if n > 1:
            r = (layer % (n - 1)) + 1
            for i in range(n):
                qc.cx(i, (i + r) % n)
    return qc


def expvals(inputs, weights) -> np.ndarray:
    """⟨Z_0⟩ … ⟨Z_{n−1}⟩ of the circuit, from Qiskit's exact statevector."""
    from qiskit.quantum_info import Statevector

    qc = vqc_circuit(inputs, weights)
    p = Statevector(qc).probabilities()
    idx = np.arange(len(p))
    return np.array([np.sum(p * (1 - 2 * ((idx >> i) & 1))) for i in range(qc.num_qubits)])


def to_qasm(inputs, weights) -> str:
    from qiskit import qasm3

    return qasm3.dumps(vqc_circuit(inputs, weights))


class QiskitLayer(nn.Module):
    """Drop-in replacement for a block's PennyLane ``TorchLayer`` (inference only)."""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.w = weights.detach().cpu().numpy()
        self.evaluations = 0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rows = z.detach().cpu().numpy()
        self.evaluations += len(rows)
        return torch.as_tensor(np.stack([expvals(r, self.w) for r in rows]), dtype=z.dtype)


@contextmanager
def qiskit_execution(model):
    """Within the block, every VQC block of ``model`` runs its circuit on Qiskit instead of PennyLane."""
    from .qseq import VQCBlock

    swapped = []
    for block in model.modules():
        if isinstance(block, VQCBlock):
            original = block.q
            block.q = QiskitLayer(next(iter(original.parameters())))
            swapped.append((block, original))
    try:
        yield [b.q for b, _ in swapped]
    finally:
        for block, original in swapped:
            block.q = original


def compare_backends(model, X: np.ndarray, batch: int = 64) -> dict:
    """Run ``model`` on ``X`` with PennyLane and with Qiskit; report the largest output difference."""
    import time

    Xt = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        t0 = time.time()
        a = torch.cat([model(Xt[i:i + batch]) for i in range(0, len(Xt), batch)]).numpy()
        t_pl = time.time() - t0
        with qiskit_execution(model) as layers:
            t1 = time.time()
            b = torch.cat([model(Xt[i:i + batch]) for i in range(0, len(Xt), batch)]).numpy()
            t_qk = time.time() - t1
            n_eval = sum(l.evaluations for l in layers)
    return {"max_abs_diff": float(np.max(np.abs(a - b))), "mean_abs_diff": float(np.mean(np.abs(a - b))),
            "n_inputs": int(len(X)),
            "qiskit_circuit_evaluations": int(n_eval), "pennylane_seconds": round(t_pl, 3),
            "qiskit_seconds": round(t_qk, 3), "outputs_pennylane": a, "outputs_qiskit": b}


# ---------------------------------------------------------------- circuit-level verification


def _pennylane_block_circuit(n_qubits: int):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


def parameter_shift_gradients(inputs, weights, coeffs) -> tuple[np.ndarray, np.ndarray]:
    """Exact gradients of f = Σᵢ cᵢ⟨Zᵢ⟩ w.r.t. inputs and weights, from Qiskit statevectors with the parameter-shift
    rule (every angle enters exactly one Pauli rotation, so ±π/2 shifts are exact)."""
    x = np.asarray(inputs, float)
    w = np.asarray(weights, float)
    c = np.asarray(coeffs, float)

    def f(xx, ww):
        return float(c @ expvals(xx, ww))

    gx = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = np.pi / 2
        gx[i] = (f(x + e, w) - f(x - e, w)) / 2
    gw = np.zeros_like(w)
    for idx in np.ndindex(w.shape):
        e = np.zeros_like(w)
        e[idx] = np.pi / 2
        gw[idx] = (f(x, w + e) - f(x, w - e)) / 2
    return gx, gw


def verify_circuit_equivalence(n_qubits: int = 4, layers: int = 2, n_cases: int = 12, seed: int = 0) -> dict:
    """Forward values, gradients (inputs and weights) and parameter mapping: Qiskit vs PennyLane, incl. edge cases."""
    rng = np.random.default_rng(seed)
    circuit = _pennylane_block_circuit(n_qubits)
    shape = (layers, n_qubits, 3)
    cases = [("random", rng.uniform(-np.pi / 2, np.pi / 2, n_qubits), rng.normal(0, 1, shape)) for _ in range(n_cases)]
    cases += [("zeros", np.zeros(n_qubits), np.zeros(shape)),
              ("input ±π/2 bounds", np.array([np.pi / 2, -np.pi / 2] * (n_qubits // 2) + [0.0] * (n_qubits % 2)),
               rng.normal(0, 1, shape)),
              ("large weights", rng.uniform(-np.pi / 2, np.pi / 2, n_qubits), rng.normal(0, 20, shape)),
              ("weights = π", rng.uniform(-1, 1, n_qubits), np.full(shape, np.pi))]
    fwd, gx_err, gw_err, map_err = [], [], [], []
    for _, x, w in cases:
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
        pl = torch.stack(list(circuit(xt, wt)))
        qk = expvals(x, w)
        fwd.append(np.abs(pl.detach().numpy() - qk))
        coeffs = rng.normal(size=n_qubits)
        (pl * torch.tensor(coeffs)).sum().backward()
        gx_q, gw_q = parameter_shift_gradients(x, w, coeffs)
        gx_err.append(np.abs(xt.grad.numpy() - gx_q))
        gw_err.append(np.abs(wt.grad.numpy() - gw_q).ravel())
        # parameter mapping: moving one weight must move both simulators' outputs identically
        idx = tuple(rng.integers(0, s) for s in shape)
        w2 = w.copy()
        w2[idx] += 0.37
        with torch.no_grad():
            d_pl = torch.stack(list(circuit(torch.tensor(x), torch.tensor(w2)))).numpy() - pl.detach().numpy()
        map_err.append(np.abs(d_pl - (expvals(x, w2) - qk)))

    def summary(errs):
        a = np.concatenate([np.ravel(e) for e in errs])
        return {"max_abs": float(a.max()), "mean_abs": float(a.mean()), "n_values": int(a.size)}

    return {"n_qubits": n_qubits, "layers": layers, "cases": len(cases),
            "edge_cases": [c[0] for c in cases if c[0] != "random"],
            "forward": summary(fwd), "grad_inputs": summary(gx_err), "grad_weights": summary(gw_err),
            "parameter_mapping": summary(map_err),
            "note": "Both are exact statevector simulations; equivalence here says nothing about noisy hardware."}
