# src/qpe.py
import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

_BACKEND = None
_BACKEND_NAME = None
_BACKEND_OK = False

def _select_backend():
    global _BACKEND, _BACKEND_NAME, _BACKEND_OK
    try:
        from qiskit_aer import AerSimulator
        _BACKEND = AerSimulator()
        _BACKEND_NAME = "AerSimulator (qiskit-aer)"
        _BACKEND_OK = True
        return
    except Exception:
        pass
    try:
        from qiskit.providers.basic_provider import BasicSimulator
        _BACKEND = BasicSimulator()
        _BACKEND_NAME = "BasicSimulator (Terra)"
        _BACKEND_OK = True
        return
    except Exception:
        _BACKEND = None
        _BACKEND_NAME = "None"
        _BACKEND_OK = False

_select_backend()

def backend_info():
    return _BACKEND_NAME, _BACKEND_OK

def pad_and_rescale(L: np.ndarray, eta: float = 0.49) -> np.ndarray:
    n = L.shape[0]
    N = 1 << (n - 1).bit_length()
    if N != n:
        P = np.zeros((N, N), dtype=float)
        P[:n, :n] = L
        Lp = P
    else:
        Lp = L.astype(float, copy=True)
    lam = np.max(np.sum(np.abs(Lp), axis=1)) 
    scale = (np.pi * eta) / max(lam, 1e-8)
    return Lp * scale

def unitary_from_laplacian(Lp: np.ndarray, t: float = 1.0) -> UnitaryGate:
    U = expm(1j * Lp * t)
    return UnitaryGate(U)

def _inverse_qft(qc: QuantumCircuit, regs: list[int]) -> None:
    n = len(regs)
    for i in range(n // 2):
        qc.swap(regs[i], regs[n - 1 - i])
    for j in range(n):
        qc.h(regs[j])
        for m in range(j + 1, n):
            qc.cp(-np.pi / (1 << (m - j)), regs[m], regs[j])

def qpe_zero_prob(U_gate: UnitaryGate, n_target: int, shots: int = 2048, verbose: bool=False) -> float:
    if not _BACKEND_OK:
        raise ImportError(
            "No simulator backend available. Install qiskit-aer or a Terra build with BasicSimulator.\n"
            "Try: pip install --upgrade qiskit qiskit-aer"
        )
    n_phase = max(3, n_target.bit_length()) 
    qc = QuantumCircuit(n_phase + n_target, n_phase)
    phase = list(range(n_phase))
    target = list(range(n_phase, n_phase + n_target))
    for q in phase:
        qc.h(q)
    for k, ctrl in enumerate(phase):
        for _ in range(1 << k):
            qc.append(U_gate.control(1), [ctrl] + target)
    _inverse_qft(qc, phase)
    qc.measure(phase, list(range(n_phase)))
    compiled = transpile(qc, backend=_BACKEND, optimization_level=1)
    result = _BACKEND.run(compiled, shots=shots).result()
    counts = result.get_counts()
    p0 = counts.get("0"*n_phase, 0) / shots
    if verbose:
        print(f"[qpe] backend={_BACKEND_NAME}, shots={shots}, phase_qubits={n_phase}, target_qubits={n_target}")
        print(f"[qpe] counts head: {dict(list(counts.items())[:5])}")
    return p0
