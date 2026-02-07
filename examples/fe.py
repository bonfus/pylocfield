import numpy as np
from ase.build import bulk

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum
from pylocfield.utils import add_equivalent_muon_sites
from visualization import show_ase

np.set_printoptions(suppress=True, precision=5)


#| # BCC Iron
#|
#| BCC Iron is an interesting case for musr.
#| Here we compute.
#|

#| ## Step 1: define the sample

# Create a BCC lattice
a = 2.866
atoms = bulk("Fe", crystalstructure="bcc", a=a)

#| ## Step 2: add the muon
#|
#| Add the muon in the tetrahedral interstitial site.
#| Notice that the position is specified in fractional coordinates.

# The muon position in fractional coordinates
atoms.info['mu'] = np.array(
    [
        [0.25, 0.5, 0.75], # these are fractional coordinates
                           # for the tetrahedral interstitial
                           # in BCC lattice
    ]
)

# Generate equivalent positions
add_equivalent_muon_sites(atoms)

#| ## Step 3: the magnetic structure
#|
#| The magnetic structure is specified as
#|
#| * a k-point (representative of the (+k, -k) pair)
#|   This must be specified in reciprocal lattice units, i.e.
#|   in fractional coordinates of the reciprocal cell.
#| * Fourier components. These are specified in _Cartesian coordinates_.
#|
#| In this case everything is fairly simple: the propagation vector is (0, 0, 0)
#| and the Fourier constant is just 0.5 * m_Fe along the z direction.

# A magnetic structure, we define two propagation vectors in
# reciprocal lattice units (i.e. these are fractional coordinates)
k = np.array(
    [
        [0.0, 0.0, 0.0],  # Propagation vector 1
    ]
)

fc = 0.5 * np.array(
    [  # first (and only) atom
        [  # first k vector
            [0, 0, 2.22]  # fourier components for first k
        ],
    ],
    dtype=complex,
)

atoms.set_array("fc", fc)
atoms.info["q"] = k

#| ## Step 4: check the magnetic structure
#|
#| A few exaples showing how to visualize structures
#| are given in visualization.py
#| You can either use ASE (click on visualize magmoms)
#| or generate a mcif file that can be shown by VESTA,
#| Jmol and others.

show_ase(atoms,
            # BCC to conventional SC
            np.array([[0,1,1],
                      [1,0,1],
                      [1,1,0]]),
            # change this to true for the interactive visualization
            gui=False
        )

#from visualization import save_mcif
#save_mcif('tetrahedral-in-primitive-BCC',
#            atoms,
#            # BCC to conventional SC
#            np.array([[0,1,1],
#                      [1,0,1],
#                      [1,1,0]])
#        )


#| ## Step 5: compute
#|
#| Compute the dipolar field with both real space and reciprocal space approach.

# Compute dipolar tensors with Ewald method
compute_dipolar_tensors(atoms)

# Compute local fields using dipolar tensors
B_e = ewald_sum(atoms)

# Compute the same fields in real space
B_r = direct_sum(atoms, r_c=220.)


n_mu = len(atoms.info["mu"])
for i in range(n_mu):
    print(f"Muon site {i},\n")
    for j in range(1):  # 1 k vectors
        print(
            f"  Direct sum method for k{j+1}: {B_r[i, j]},\n  Ewald sum method for k{j+1}: {B_e[i, j]}"
        )
    print('\n')

print('---')


#| ## Be direct
#|
#| Working with non-orthogonal lattices can be tricky,
#| let's get the same with a conventional simple cubic lattice.

a = 2.866
atoms_conv = bulk("Fe", crystalstructure="sc", a=a)
atoms_conv.append('Fe')
atoms_conv.set_scaled_positions([[0,0,0],[0.5,0.5,0.5]])


# The muon position in fractional coordinates
atoms_conv.info['mu'] = np.array(
    [
        [0.25, 0.5, 0.], # these are fractional coordinates
                           # for the tetrahedral interstitial
                           # in the conventional cell of a BCC crystal
    ]
)

# Now there are 12 equivalent muon sites
add_equivalent_muon_sites(atoms_conv)


# A magnetic structure, we define two propagation vectors in
# reciprocal lattice units (i.e. these are fractional coordinates)
k = np.array(
    [
        [0.0, 0.0, 0.0],  # Propagation vector 1
    ]
)

# Now there are two atoms in the unit cell
fc = 0.5 * np.array(
    [   # first atom
        [   # first (and only) k vector
            [0, 0, 2.22]
        ],
        # second atom
        [   # first (and only) k vector
            [0, 0, 2.22]
        ],
    ],
    dtype=complex,
)

atoms_conv.set_array("fc", fc)
atoms_conv.info["q"] = k

#| ## Check the results

# To save an mcif file uncomment the lines below.
# you'll need pymatgen.

#save_mcif('tetrahedral-in-conventional-SC',
#            atoms_conv,
#            # BCC to conventional SC
#            np.diag(np.ones(3))
#        )

## Compute

# Compute dipolar tensors with Ewald method
compute_dipolar_tensors(atoms_conv)

# Compute local fields using dipolar tensors
B_e = ewald_sum(atoms_conv)

# Compute the same fields in real space
B_r = direct_sum(atoms_conv, r_c=200.)

n_mu = len(atoms_conv.info["mu"])
for i in range(n_mu):
    print(f"Muon site {i},\n")
    for j in range(1):  # 1 k vectors
        print(
            f"  Direct sum method for k{j+1}: {B_r[i, j]},\n  Ewald sum method for k{j+1}: {B_e[i, j]}"
        )
    print('\n')

print('---')
