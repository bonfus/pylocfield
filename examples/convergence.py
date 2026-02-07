import numpy as np
from pymatgen.core import Structure
from visualization import show_ase

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum
from pylocfield.utils import add_equivalent_muon_sites, real_imag_phase_to_fc

np.set_printoptions(suppress=True, precision=5)


#| # Convergence
#|
#| Here we consider CoF2 to check convergence of real space and Ewald
#| approaches.

#| ## Step 1: define structure

# Here we load all information from an mCIF file
st = Structure.from_file("0.178_CoF2.mcif")
atoms = st.to_ase_atoms()

#| ## Step 2: add the muon
#|
#| Add the muon in the tetrahedral interstitial site.
#| Notice that the position is specified in fractional coordinates.

# The muon position in fractional coordinates
atoms.info["mu"] = np.array(
    [
        [0.0, 0.5, 0.0],
    ]
)

# Generate equivalent positions
add_equivalent_muon_sites(atoms)

#| ## Step 3: the magnetic structure
#|

# A magnetic structure, we define two propagation vectors in
# reciprocal lattice units (i.e. these are fractional coordinates)
k = np.array(
    [
        [0.0, 0.0, 0.0],  # Propagation vector 1
    ]
)

ri = np.array([x.moment for x in st.site_properties["magmom"]], dtype=complex)

atoms.set_array("ri", ri)
atoms.set_array("phi", np.zeros(len(atoms)))
atoms.info["q"] = k
atoms.set_array("fc", real_imag_phase_to_fc(atoms))

#| ## Step 4: check the magnetic structure
#|
#| A few exaples showing how to visualize structures
#| are given in visualization.py
#| You can either use ASE (click on visualize magmoms)
#| or generate a mcif file that can be shown by VESTA,
#| Jmol and others.

show_ase(
    atoms,
    # BCC to conventional SC
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    gui=False,
)

# from visualization import save_mcif
# save_mcif('tetrahedral-in-primitive-BCC',
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
B_r = direct_sum(atoms, r_c=220.0)


n_mu = len(atoms.info["mu"])
for i in range(n_mu):
    print(f"Muon site {i},\n")
    for j in range(1):  # 1 k vectors
        print(
            f"  Direct sum method for k{j + 1}: {B_r[i, j]},\n  Ewald sum method for k{j + 1}: {B_e[i, j]}"
        )
    print("\n")

print("---")


#| ## Step 6: convergence
#|
#|
import matplotlib.pyplot as plt

plt.loglog()
# Compute dipolar tensors with Ewald method
for r in np.logspace(0.1, 0.8):
    compute_dipolar_tensors(atoms, r_c=r, g_c=r)
    b = ewald_sum(atoms)
    plt.scatter(np.ones(4) * r, np.linalg.norm(b, axis=2), c="k")

for r in np.logspace(1, 2, 10):
    b = direct_sum(atoms, r_c=r)
    plt.scatter(np.ones(4) * r, np.linalg.norm(b, axis=2), c="r")
plt.show()

#| ## Step 7: ferromagnetic convergence
#|
#| Let's consider instead a ferromagnetic order.
#|
atoms.set_array("fc", np.abs(atoms.get_array("fc")).astype(complex))

plt.loglog()
# Compute dipolar tensors with Ewald method
for r in np.logspace(0.1, 0.8):
    compute_dipolar_tensors(atoms, r_c=r, g_c=r)
    b = ewald_sum(atoms)
    plt.scatter(np.ones(4) * r, np.linalg.norm(b, axis=2), c="k")

for r in np.logspace(1, 2, 10):
    b = direct_sum(atoms, r_c=r)
    plt.scatter(np.ones(4) * r, np.linalg.norm(b, axis=2), c="r")
plt.show()
