import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
import numpy as np
import gensim.downloader as api
import tensorflow as tf
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from functools import reduce
from sentence_transformers import SentenceTransformer
from PIL import Image

class ansatzBuilder:
    def get_ansatz(self):
        return self.unitary_circ
    
    def __init__(self, num_qubits, params, num_layers):
        
        self._num_qubits = int(num_qubits)
        self.qubits = QuantumRegister(self._num_qubits)
        self.unitary_circ = QuantumCircuit(self.qubits)
        
        for l in range(num_layers):
            self.layer(params[l][0], params[l][1])
        
        
    


    def get_params(self, _num_qubits, _num_layer):
        
        self.num_qubits = _num_qubits
        self.num_layers = _num_layer
        x = self.get_param_resolver(self.num_qubits, self.num_layers)
        params = self.min_to_vqsd(x, self.num_qubits, self.num_layers)
        return params

    def get_param_resolver(self, num_qubits, num_layers):
        num_angles = 12 * num_qubits * num_layers
        angs = np.pi * (2 * np.random.rand(num_angles) - 1)
        params = ParameterVector('θ', num_angles)
        param_dict = dict(zip(params, angs))
        return param_dict

    def num_angles_required_for_layer(self):
        return 12 * (self._num_qubits)

    def min_to_vqsd(self, param_list, num_qubits, num_layer):
       
        param_values = np.array(list(param_list.values())) 
        x = param_values.reshape(num_layer, 2, num_qubits // 2, 12)
        x_reshaped = x.reshape(num_layer, 2, num_qubits // 2, 4, 3)
        return x_reshaped

    def layer(self, params, shifted_params):
        n = self._num_qubits
        if params.size != self.num_angles_required_for_layer() / 2:
            raise ValueError("Params.size: ", params.size, " angoli richiesti: ", self.num_angles_required_for_layer() / 2)

        shift = 0  # non ho copie, lavoro con un singolo stato
        for ii in range(0, n - 1, 2):
            qubits = [self.qubits[ii + shift], self.qubits[ii + 1 + shift]]
            gate_params = params[ii // 2]
            self._apply_gate(qubits, gate_params)

        shift = self._num_qubits * 0  # 0 = COPY

        if n >= 2:
            for ii in range(1, n, 2):
                self._apply_gate(
                    [self.qubits[ii], self.qubits[(ii + 1) % n]],
                    shifted_params[ii // 2]
                )

    def _apply_gate(self, qubits, params):
        q = qubits[0]
        p = params[0]
        self._rot(q, p)
        q = qubits[1]
        p = params[1]
        self._rot(q, p)
        self.unitary_circ.cx(qubits[0], qubits[1])
        q = qubits[0]
        p = params[2]
        self._rot(q, p)
        q = qubits[1]
        p = params[3]
        self._rot(q, p)
        self.unitary_circ.cx(qubits[0], qubits[1])

    def _rot(self, qubit, params):
        self.unitary_circ.rx(params[0], qubit)
        self.unitary_circ.ry(params[1], qubit)
        self.unitary_circ.rz(params[2], qubit)

    def get_unitary(self,circuit_name):
        combined_circuit = QuantumCircuit(self._num_qubits, name=circuit_name)
        combined_circuit.compose(self.unitary_circ, inplace=True)
        return combined_circuit.to_instruction()
