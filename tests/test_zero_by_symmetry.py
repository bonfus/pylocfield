import pytest
import numpy as np
from numpy.testing import assert_allclose
from ase.build import bulk

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum
from pylocfield.utils import add_equivalent_muon_sites, real_imag_phase_to_fc

def check_zero(atoms, muons, k, ri):
    atoms.info["mu"] = muons

    atoms.set_array("ri", ri)
    atoms.set_array("phi", np.zeros([len(atoms), 3]))
    atoms.info["q"] = k
    atoms.set_array("fc", real_imag_phase_to_fc(atoms))
    
    
    # Compute dipolar tensors with Ewald method
    compute_dipolar_tensors(atoms)
    
    # Compute local fields using dipolar tensors
    B_e = ewald_sum(atoms)
    
    assert_allclose(B_e, np.zeros_like(B_e), atol=1e-10)
    
    B_r = direct_sum(atoms, r_c=220.0)
    
    assert_allclose(B_r, np.zeros_like(B_r), atol=1e-10)

def test_zero_sc():

    # Create a SC lattice
    a = 5
    atoms = bulk("H", crystalstructure="sc", a=a)
    
    # The muon position in fractional coordinates
    mu = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    
    # AF order
    k = np.array(
        [
            [0.5, 0.5, 0.5],  # Propagation vector 1
            [0.5, 0.5, 0.0],  # Propagation vector 2
            [0.5, 0.0, 0.0],  # Propagation vector 3
        ]
    )
    
    
    ri = np.array([[[0,0,1],[0,0,1],[0,0,1]]], dtype=complex)
    
    check_zero(atoms, mu, k, ri)


def test_zero_tet():

    # Create a Tetragonal lattice
    a = 5
    c = 11
    atoms = bulk("H", crystalstructure="orthorhombic", a=a, b=a, c=c, orthorhombic=True)
 
    # The muon position in fractional coordinates
    mu = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    
    # AF order
    k = np.array(
        [
            [0.5, 0.5, 0.5],  # Propagation vector 1
            [0.5, 0.5, 0.0],  # Propagation vector 2
            [0.5, 0.0, 0.0],  # Propagation vector 3
        ]
    )
    
    
    ri = np.array([[[0,0,1],[0,0,1],[0,0,1]]], dtype=complex)
    
    check_zero(atoms, mu, k, ri)


