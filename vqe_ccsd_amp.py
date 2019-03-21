import os

from numpy import array, concatenate, zeros
from numpy.random import randn
from scipy.optimize import minimize

from openfermion.config import *
from openfermionprojectq import *

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, bravyi_kitaev_tree
from openfermion.utils import (uccsd_singlet_paramsize, trotter_operator_grouping,
                               uccsd_singlet_get_packed_amplitudes)

from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer

# Load the molecule.
# from molecule import molecule

# from uccsd2 import calculated_molecule
# filename = os.path.join(DATA_DIRECTORY, 'H2_sto-3g_singlet_0.7414')

bond_len = 1.45
atom_1 = 'H'
atom_2 = 'Li'
basis = 'sto-3g'
multiplicity = 1
charge = 0

coordinate_1 = (0.0, 0.0, 0.0)
coordinate_2 = (0.0, 0.0, bond_len)
geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]

molecule = MolecularData(geometry, basis, multiplicity,
                         charge, description='1.45')

molecule.load()

# Use a Jordan-Wigner encoding, and compress to remove 0 imaginary components
qubit_hamiltonian = jordan_wigner(molecule.get_molecular_hamiltonian())
qubit_hamiltonian.compress()
compiler_engine = uccsd_trotter_engine()


def energy_objective(packed_amplitudes):
    """Evaluate the energy of a UCCSD singlet wavefunction with packed_amplitudes
    Args:
        packed_amplitudes(ndarray): Compact array that stores the unique
            amplitudes for a UCCSD singlet wavefunction.

    Returns:
        energy(float): Energy corresponding to the given amplitudes
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Set Jordan-Wigner initial state with correct number of electrons
    wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
    for i in range(molecule.n_electrons):
        X | wavefunction[i]

    # Build the circuit and act it on the wavefunction
    evolution_operator = uccsd_singlet_evolution(packed_amplitudes, molecule.n_qubits,
                                                 molecule.n_electrons)

    evolution_operator | wavefunction
    compiler_engine.flush()

    # Evaluate the energy and reset wavefunction
    energy = compiler_engine.backend.get_expectation_value(qubit_hamiltonian, wavefunction)
    All(Measure) | wavefunction
    compiler_engine.flush()
    return energy


n_amplitudes = uccsd_singlet_paramsize(molecule.n_qubits, molecule.n_electrons)

initial_amplitudes = uccsd_singlet_get_packed_amplitudes(molecule.ccsd_single_amps, molecule.ccsd_double_amps
                                                         , molecule.n_qubits, molecule.n_electrons)

initial_energy = energy_objective(initial_amplitudes)

# Run VQE Optimization to find new CCSD parameters


opt_result = minimize(energy_objective, initial_amplitudes,
                      method='BFGS', options={'disp': True})

opt_energy, opt_amplitudes = opt_result.fun, opt_result.x

print("\nOptimal UCCSD Singlet Energy: {}".format(opt_energy))
print("Optimal UCCSD Singlet Amplitudes: {}".format(opt_amplitudes))
print("Classical CCSD Energy: {} Hartrees".format(molecule.ccsd_energy))
print("Exact FCI Energy: {} Hartrees".format(molecule.fci_energy))
print("Initial amplitudes: {}".format(initial_amplitudes))
print("Initial Energy of UCCSD with CCSD amplitudes: {} Hartrees".format(initial_energy))

'''
compiler_engine = uccsd_trotter_engine(CommandPrinter())
wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
for i in range(molecule.n_electrons):
    X | wavefunction[i]

# Build the circuit and act it on the wavefunction


evolution_operator = uccsd_singlet_evolution(initial_amplitudes, molecule.n_qubits, 
                                             molecule.n_electrons)
evolution_operator | wavefunction
compiler_engine.flush()
'''

