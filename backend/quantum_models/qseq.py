"""Quantum sequence models (PennyLane + PyTorch): QLSTM, Quantum Transformer, Quantum CNN-LSTM.

Every quantum layer is the same *VQC block*::

    x ∈ Rⁿⁱⁿ → Linear(nᵢₙ, q) → tanh·π/2 → AngleEmbedding(RY) → StronglyEntanglingLayers(L) → ⟨Z₀..Z_q⟩ → Linear(q, nₒᵤₜ)

* **QLSTM** (after Chen, Yoo & Fang, 2020): the four LSTM gate transforms (forget, input, candidate,
  output) are separate VQC blocks over ``[x_t, h_{t-1}]``; the cell/hidden-state algebra is the
  standard LSTM recurrence.
* **Quantum Transformer** (after Di Sipio et al., 2022): the query, key and value projections and the
  feed-forward sub-layer are VQC blocks; attention weights are computed classically as
  ``softmax(QKᵀ/√d)``, with residual connections and layer norm.
* **Quantum CNN-LSTM**: a 1-D *quanvolution* filter — one VQC block applied to every ``k``-step patch
  of all channels (causal padding, configurable stride) — produces quantum feature maps that a QLSTM
  then models over time.

Each architecture has a **classical twin** (``quantum=False``): identical except that the circuit is
replaced by a classical ``Linear(q, q) + tanh`` bottleneck of the same width. Comparing a model with
its twin isolates the contribution of the circuit itself. All circuits run on a statevector simulator
(``default.qubit``); nothing here claims a quantum advantage.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

ARCHS = ("lstm", "transformer", "cnn_lstm")
SEQ_MODELS = tuple(f"{p}_{a}" for p in ("q", "c") for a in ARCHS)   # q_lstm, q_transformer, ..., c_cnn_lstm
QUANTUM_SEQ_MODELS = tuple(m for m in SEQ_MODELS if m.startswith("q_"))


@dataclass
class SeqModelConfig:
    name: str = "q_lstm"            # one of SEQ_MODELS
    n_features: int = 14
    n_out: int = 1                  # 1 for regression; n_features for imputation
    task: str = "regression"        # "regression" (last step → scalar) or "sequence" (one output per step)
    window: int = 24
    n_qubits: int = 4
    layers: int = 2                 # variational layers per circuit
    hidden: int = 8                 # recurrent hidden size / transformer model width
    kernel: int = 4                 # quanvolution patch length (cnn_lstm)
    stride: int = 2                 # quanvolution stride (forced to 1 for task="sequence")
    device: str = "default.qubit"

    @property
    def arch(self) -> str:
        return self.name.split("_", 1)[1]

    @property
    def quantum(self) -> bool:
        return self.name.startswith("q_")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- building blocks


class VQCBlock(nn.Module):
    """Classical in-projection → variational quantum circuit → classical read-out (works on any leading dims)."""

    def __init__(self, n_in: int, n_out: int, n_qubits: int, layers: int, device: str = "default.qubit"):
        super().__init__()
        import pennylane as qml

        self.n_qubits, self.layers = n_qubits, layers
        self.inp = nn.Linear(n_in, n_qubits)
        dev = qml.device(device, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit
        shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_qubits)
        self.q = qml.qnn.TorchLayer(circuit, {"weights": shape},
                                    init_method=lambda t: nn.init.normal_(t, 0.0, 0.1))
        self.out = nn.Linear(n_qubits, n_out)
        self.calls = 0   # circuit evaluations (batched) since the last reset — for cost reporting
        self.ablate = False   # diagnostics only: replace circuit outputs by their batch mean (see ablate_circuits)

    def forward(self, x):
        lead = x.shape[:-1]
        z = torch.tanh(self.inp(x)).reshape(-1, self.n_qubits) * (math.pi / 2)
        self.calls += 1
        e = self.q(z)
        if e.dtype != z.dtype:
            e = e.to(z.dtype)
        if self.ablate:
            e = e.mean(0, keepdim=True).expand_as(e)
        return self.out(e.reshape(*lead, self.n_qubits))

    def circuit_specs(self) -> dict:
        import pennylane as qml

        try:
            w = next(iter(self.q.parameters())).detach()
            res = qml.specs(self.circuit, level="device")(torch.zeros(self.n_qubits, dtype=w.dtype), w).resources
            return {"depth": int(res.depth), "gates": int(res.num_gates), "gate_types": dict(res.gate_types)}
        except Exception:  # informational only
            return {"depth": None, "gates": None, "gate_types": {}}


class ClassicalBlock(nn.Module):
    """Classical twin of :class:`VQCBlock`: the circuit is replaced by Linear(q, q) + tanh of the same width."""

    def __init__(self, n_in: int, n_out: int, n_qubits: int, layers: int, device: str = ""):
        super().__init__()
        self.inp = nn.Linear(n_in, n_qubits)
        self.mid = nn.Linear(n_qubits, n_qubits)
        self.out = nn.Linear(n_qubits, n_out)
        self.calls = 0
        self.ablate = False

    def forward(self, x):
        self.calls += 1
        e = torch.tanh(self.mid(torch.tanh(self.inp(x))))
        if self.ablate:
            flat = e.reshape(-1, e.shape[-1])
            e = flat.mean(0, keepdim=True).expand_as(flat).reshape(e.shape)
        return self.out(e)


class QLSTM(nn.Module):
    """LSTM recurrence whose four gate transforms are (quantum or classical-twin) blocks."""

    def __init__(self, n_in: int, hidden: int, block):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.ModuleList([block(n_in + hidden, hidden) for _ in range(4)])   # f, i, g, o

    def forward(self, x):                       # x: (B, T, F) → (B, T, H)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        c = x.new_zeros(B, self.hidden)
        outs = []
        for t in range(T):
            v = torch.cat([x[:, t], h], dim=-1)
            f, i, g, o = (m(v) for m in self.gates)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1)


class QTransformerLayer(nn.Module):
    """Encoder layer with block-based Q/K/V projections and feed-forward; classical softmax attention."""

    def __init__(self, d: int, block):
        super().__init__()
        self.d = d
        self.q, self.k, self.v, self.ffn = (block(d, d) for _ in range(4))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):                       # (B, T, d)
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(Q @ K.transpose(1, 2) / math.sqrt(self.d), dim=-1)
        x = self.n1(x + attn @ V)
        return self.n2(x + self.ffn(x))


class Quanv1d(nn.Module):
    """1-D quanvolution: one block applied to each causal patch of ``kernel`` steps × all channels."""

    def __init__(self, n_in: int, n_filters: int, kernel: int, stride: int, block):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.filter = block(n_in * kernel, n_filters)

    def forward(self, x):                       # (B, T, F) → (B, T', n_filters)
        pad = x[:, :1].expand(-1, self.kernel - 1, -1)                     # replicate-pad on the left
        xp = torch.cat([pad, x], dim=1)
        patches = xp.unfold(1, self.kernel, self.stride)                    # (B, T', F, k)
        patches = patches.transpose(2, 3).reshape(x.shape[0], patches.shape[1], -1)
        return torch.relu(self.filter(patches))


# ---------------------------------------------------------------- models


class SeqModel(nn.Module):
    def __init__(self, cfg: SeqModelConfig):
        super().__init__()
        if cfg.name not in SEQ_MODELS:
            raise ValueError(f"model must be one of {SEQ_MODELS}")
        self.cfg = cfg
        Block = VQCBlock if cfg.quantum else ClassicalBlock

        def block(n_in, n_out):
            return Block(n_in, n_out, cfg.n_qubits, cfg.layers, cfg.device)

        H = cfg.hidden
        if cfg.arch == "lstm":
            self.body = QLSTM(cfg.n_features, H, block)
        elif cfg.arch == "transformer":
            self.proj = nn.Linear(cfg.n_features, H)
            self.pos = nn.Parameter(torch.randn(1, cfg.window, H) * 0.02)
            self.body = QTransformerLayer(H, block)
        else:  # cnn_lstm
            stride = 1 if cfg.task == "sequence" else cfg.stride
            self.conv = Quanv1d(cfg.n_features, cfg.n_qubits, cfg.kernel, stride, block)
            self.body = QLSTM(cfg.n_qubits, H, block)
        self.head = nn.Linear(H, cfg.n_out)

    def forward(self, x):                       # x: (B, T, F)
        a = self.cfg.arch
        if a == "lstm":
            h = self.body(x)
        elif a == "transformer":
            h = self.body(self.proj(x) + self.pos[:, :x.shape[1]])
        else:
            h = self.body(self.conv(x))
        if self.cfg.task == "sequence":
            return self.head(h)                                     # (B, T, n_out)
        pooled = h.mean(1) if a == "transformer" else h[:, -1]
        return self.head(pooled).squeeze(-1)                        # (B,)

    # -- reporting
    def blocks(self) -> list[nn.Module]:
        return [m for m in self.modules() if isinstance(m, (VQCBlock, ClassicalBlock))]

    def describe(self) -> dict:
        blocks = self.blocks()
        q_params = sum(p.numel() for b in blocks if isinstance(b, VQCBlock) for p in b.q.parameters())
        out = {"model": self.cfg.name, "arch": self.cfg.arch, "quantum": self.cfg.quantum,
               "trainable_params": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
               "quantum_params": int(q_params), "blocks": len(blocks)}
        if self.cfg.quantum:
            out.update(n_qubits=self.cfg.n_qubits, circuit_layers=self.cfg.layers, circuits=len(blocks),
                       circuit=blocks[0].circuit_specs(), device=self.cfg.device)
        return out

    def reset_calls(self) -> None:
        for b in self.blocks():
            b.calls = 0

    def calls(self) -> int:
        return sum(b.calls for b in self.blocks())


class ablate_circuits:
    """Context manager: inside it, every block's circuit (or the twin's same-width classical layer) outputs only its
    batch mean, removing all input-dependent information that passes through it. Comparing errors with and
    without ablation shows how much the model relies on the circuits (and the twin on its counterpart)."""

    def __init__(self, model: "SeqModel"):
        self.blocks = model.blocks()

    def __enter__(self):
        for b in self.blocks:
            b.ablate = True
        return self

    def __exit__(self, *exc):
        for b in self.blocks:
            b.ablate = False
        return False


def block_gradient_norms(model: "SeqModel") -> dict:
    """L2 norm of the gradients of the circuit weights (quantum) or of the counterpart layer (twin), after backward."""
    norms = []
    for b in model.blocks():
        params = list(b.q.parameters()) if isinstance(b, VQCBlock) else list(b.mid.parameters())
        g = [p.grad.detach().flatten() for p in params if p.grad is not None]
        if g:
            norms.append(float(torch.cat(g).norm()))
    return {"mean": float(np.mean(norms)) if norms else 0.0, "min": float(np.min(norms)) if norms else 0.0,
            "max": float(np.max(norms)) if norms else 0.0, "blocks": len(norms)}


def build_model(cfg: SeqModelConfig, seed: int | None = None) -> SeqModel:
    if seed is not None:
        torch.manual_seed(seed)
    return SeqModel(cfg).double()


def state_to_numpy(model: nn.Module) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def model_from_state(config: dict, state: dict) -> SeqModel:
    m = build_model(SeqModelConfig(**config))
    m.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in state.items()})
    m.eval()
    return m
