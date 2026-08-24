import pytest
import numpy as np
from numpy.testing import assert_allclose
from ase.build import bulk

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum
from pylocfield.utils import add_equivalent_muon_sites, real_imag_phase_to_fc


# Magnetic constant
MU0 = 4 * np.pi * 1e-7  # T·m/A
Ang = 1e-10

def dipole_field(m, r, dipole_position=np.zeros(3)):
    """
    Calculate the magnetic field B created by a point magnetic dipole.

    Parameters
    ----------
    m : array-like, shape (3,)
        Magnetic dipole moment in A·m².
    r : array-like, shape (3,)
        Observation point in meters.
    dipole_position : array-like, shape (3,)
        Position of the dipole in meters.

    Returns
    -------
    B : ndarray, shape (3,)
        Magnetic field in teslas.
    """
    m = np.asarray(m, dtype=float)
    r = np.asarray(r, dtype=float)
    dipole_position = np.asarray(dipole_position, dtype=float)

    displacement = r - dipole_position
    distance = np.linalg.norm(displacement)

    if distance == 0:
        raise ValueError("The observation point cannot be at the dipole position.")

    r_hat = displacement / distance

    # B = μ0/(4πr³) [3(m·r_hat)r_hat - m]
    B = (MU0 / (4 * np.pi * distance**3)) * (
        3 * np.dot(m, r_hat) * r_hat - m
    )

    return B


def test_one_dipole_in_cubic_cell():

    # Create a cubic lattice with 100 Ang. edge
    # This is done to approximate an isolated dipole
    a = 100
    atoms = bulk("H", crystalstructure="sc", a=a)
 
    # The muon position (random) in fractional coordinates
    mu = np.array(
        [
            [0.01, 0.02, 0.03],
        ]
    )
    
    # AF order, to avoid spurious contributions
    k = np.array(
        [
            [0.5, 0.5, 0.5],  # Propagation vector 1
        ]
    )
    
    
    ri = np.array([[[0,0,1]]], dtype=complex)
    

    atoms.info["mu"] = mu

    atoms.set_array("ri", ri)
    atoms.set_array("phi", np.zeros([len(atoms), 1]))
    atoms.info["q"] = k
    atoms.set_array("fc", real_imag_phase_to_fc(atoms))


    # Compute dipolar tensors with Ewald method
    compute_dipolar_tensors(atoms, r_c=2., g_c=2.)

    # Compute local fields using dipolar tensors
    B_e = ewald_sum(atoms)
    # Do the same in real space
    B_r = direct_sum(atoms, r_c=22.0)


    m = ri[0,0].real* 9.2740101e-24 #A·m²

    # Dipole located at the origin
    dipole_position = np.array([0.0, 0.0, 0.0])

    # Calculate the magnetic field for a single dipole
    B = dipole_field(m, mu[0]*a*Ang, dipole_position)

    assert_allclose(np.squeeze(B_e), B, atol=1e-4)   
    assert_allclose(np.squeeze(B_r), B, atol=1e-4)
