import numpy as np
import numpy.linalg as LA
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator
from qiskit.circuit import QuantumCircuit
from scipy.linalg import expm

def generate_active_states(N_qubit, n_states=10, time=5.0, config="neel", seed=None):
    if seed is not None:
        np.random.seed(seed)

    def random_hamiltonian():
        paulis = []
        coeffs = []
        for _ in range(N_qubit * 2):
            # Genera fino a quando il termine ha almeno 3 non-identità
            while True:
                pstr = ""
                for _ in range(N_qubit):
                    pstr += np.random.choice(["I", "X", "Y", "Z"], p=[0.3, 0.2333, 0.2333, 0.2334])
                if sum(1 for c in pstr if c != "I") >= 3:
                    break
            paulis.append(pstr)
            coeffs.append(np.random.uniform(-1, 1))
        return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

    def get_initial_state():
        qc = QuantumCircuit(N_qubit)
        if config == "neel":
            qc.x(range(0, N_qubit, 2))
        elif config == "domain_wall":
            qc.x(range(N_qubit // 2, N_qubit))
        elif config == "up":
            qc.x(range(N_qubit))
        elif config == "random":
            for i in range(N_qubit):
                if np.random.rand() > 0.5:
                    qc.x(i)
        elif config == "superpos":
            qc.h(range(N_qubit))
        return Statevector.from_instruction(qc)

    states = []
    H = random_hamiltonian()
    
    for _ in range(n_states):
        psi = get_initial_state()
        
        U = expm(-1j * H.to_matrix() * time)
        evolved = Operator(U) @ psi.data
        states.append(Statevector(evolved))

    return states

class QuantumEvolver:
    def __init__(self, n_qubit, max_time, seed, initial_config="neel"):
        self.n_qubit = n_qubit
        self.max_time = max_time
        self.initial_config = initial_config
        self.states = generate_active_states(n_qubit, n_states=max_time, config=initial_config, seed=seed)

    def get_states(self):
        return self.states

if __name__ == "__main__":
    q = QuantumEvolver(n_qubit=4, max_time=4, seed=None, initial_config="neel")
    