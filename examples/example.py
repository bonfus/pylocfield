import numpy as np
from ase.build import bulk

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum
from pylocfield.utils import add_equivalent_muon_sites

np.set_printoptions(suppress=True, precision=5)

# Create simple cubic lattice
a = 2.0
atoms = bulk("Fe", crystalstructure="sc", a=a, basis=[[0, 0, 0]])

# A copuple of muon positions in fractional coordinates
mupos = np.array(
    [
        [0.25, 0.0, 0.0],
        [0.35, 0.0, 0.0]
    ]
)


# A magnetic structure, we define two propagation vectors in
# reciprocal lattice units (i.e. these are fractional coordinates)
k = np.array(
    [
        [0.0, 0.0, 0.1],  # Propagation vector 1
        [0.0, 0.0, 0.2],  # Propagation vector 2
    ]
)

fc = np.array(
    [
        [  # first (and only) atom
            [  # first k vector
                [1, 0, 0]  # fourier components for first k
            ],
            [  # second k vector
                [2, 0, 0]  # fourier components for second k
            ],
        ]
    ],
    dtype=complex,
)

# Here we set magnetic properties inside the ASE object.
# The fourier componensts are introduced as the array "fc"
# while q vectors and muon positions are stored as "info".
#
# The definition of fourier components is given in Eq. 34
# of EPJ Web of Conferences 22, 00010 (2012) DOI: 10.1051/epjconf/20122200010
#
# It is also possible to write it as a real part, a complex part and a phase.
# This is shown explicitly in the expression just before Eq. 37.
# Notice the factor 1/2 in front of the expression.
#
# It is possible to specify the magnetic order with real imaginary and phase by doing
#
# atoms.set_array("ri", complex_array[na, nq, 3])
# atoms.set_array("phi", real array[na, nq, 1] )
# real_imag_phase_to_fc(atoms)
#
# The last function returns the fourier components to be used for `atoms.set_array("fc", fc)`.
# This is how fourier constants were specified in muesr.
#
# For more detail see Section 5 of https://www.epj-conferences.org/articles/epjconf/pdf/2012/04/epjconf_cscm2012_00010.pdf
#
atoms.set_array("fc", fc)
atoms.info["q"] = k
atoms.info["mu"] = mupos


# add equivalent muon sites
add_equivalent_muon_sites(atoms)

# Compute dipolar tensors with Ewald method
compute_dipolar_tensors(atoms)

# Compute local fields using dipolar tensors
B_e = ewald_sum(atoms)

# Compute the same fields in real space
B_r = direct_sum(atoms)


n_mu = len(atoms.info["mu"])
for i in range(n_mu):
    print(f"Muon site {i},\n")
    for j in range(2):  # 2 k vectors
        print(
            f"  Direct sum method for k{j+1}: {B_r[i, j]},\n  Ewald sum method for k{j+1}: {B_e[i, j]}"
        )
    print('\n')
