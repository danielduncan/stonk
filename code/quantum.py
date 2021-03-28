# testground for hybrid classical-quantum neural network

import numpy as np

import torch

import qiskit

class QuantumCircuit:
    def __init__(self, backend, qubits, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        allQubits = [i for i in range(qubits)]
        self.theta = qiskit.circuit.Parameter('theta')

        self._circuit.h(allQubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, allQubits)

        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        transpileCircuit = transpile(self._circuit, self.backend)

        quantumObject = assemble(transpileCircuit, shots = self.shots, parameterBinds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(quantumObject)
        result = job.result().getCounts(self._circuit)

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        probabilities = counts / self.shots

        expectation = np.sum(states * probabilities)

        return np.array([expectation])