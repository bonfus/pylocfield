import numpy as np
from ase.spacegroup import crystal

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum
from pylocfield.utils import add_equivalent_muon_sites, real_imag_phase_to_fc

import matplotlib.pyplot as plt

#| # MnSi example
#|
#| This example shows how to reproduce the dipolar field distribution at muon
#| sites in MnSi. The details are described in A. Amato et al. PRB 93 144419 (2016).
#|
#| Here's an extract from manuscript:
#|     This is remarkable as
#|     the position of Bmin,II,III,IV, with respect to BI, depends on the
#|     helicity of the incommensurate helix and is located above BI
#|     only for left-handed helicity
#|
#| Below we reproduce this interesting result.

np.set_printoptions(suppress=True, precision=5)

def uv(vec):
    # a simple function to obtain unit vectors
    return vec/np.linalg.norm(vec)

#| ## Step 1: Lattice and muons
#|
#| Here we define the lattice structure and add the known
#| muon position.
#| The function `add_equivalent_muon_sites` is used to
#| identify the symmetry of the lattice and to add the
#| equivalent muon sites in the system.

# Define the lattice structure
a = 4.558 # Å
atoms = crystal("MnSi",
                basis = [
                            [0.138,0.138,0.138],
                            [0.845,0.845,0.845]
                        ],
                spacegroup=198,
                cellpar=[a,a,a,90,90,90])

# Add the known muon site.
# Other equivalent positions are automatically added below.
atoms.info['mu'] = np.array(
    [
        [0.532, 0.532, 0.532],
    ]
)

# Find and add equivalent muon sites.
# Notice that the position
# inserted above may not be the first one.
add_equivalent_muon_sites(atoms)

#| ## Step 2: Magnetic order(s)
#|
#| Here we define the magnetic order in terms of propagation vector
#| and Fourier components.

# Norm of the propagation vector in MnSi (in reciprocal space)
norm_k = 0.036 # Å −1

# Define the propagation vector in reciprocal space along
# the [111] direction.
k = norm_k * uv(np.ones(3))

# Same as above in reduced coordinates.
# Notice that ASE gives reciprocal space without the 2π factor.
k_rlu = atoms.cell.reciprocal().scaled_positions(k) / (2 * np.pi)

# A magnetic structure, we define two propagation vectors in
# reciprocal lattice units (i.e. these are fractional coordinates)
# These are the same for MnSi since we just want to define two magnetic
# orders that differ by the handness of the helix.
atoms.info['q'] = np.array(
    [
        k_rlu,  # first k
        k_rlu,  # second k, equal to the first one.
    ]
)

# Now the Fourier components, we first define to vectors
# in the plane perpendicular to k.

k_u = uv(k) # unit vector along k
a = uv([1,-1,0]) # a unit vector orthogonal to k
b = np.cross(k_u,a) # a third vector, orthogonal to both previous ones.

# Here we define the real and imaginary parts of the Fourier components (FC).
# We assume the local moment on Mn to be 0.385 mu_B and we generate the
# FC with the two unit vectors defined above.

ri = 0.385 * np.array(
    [
        [  # first atom
            a+1.j*b  # fourier components for first k vector, right handed helix
            ,
            a-1.j*b  # fourier components for second k vector, left handed helix
        ],
        [  # second atom
            a+1.j*b  # first k
            ,
            a-1.j*b  # second k
        ],
        [  # third atom
            a+1.j*b  # ffirst k
            ,
            a-1.j*b  # second k
        ],
        [  # fourth atom
            a+1.j*b  # first k
            ,
            a-1.j*b  # second k
        ],
        [np.zeros(3), np.zeros(3),], # these are the values for Si.
        [np.zeros(3), np.zeros(3),], # They are set to 0 but a better solution is probably
        [np.zeros(3), np.zeros(3),], # to only add magnetic atoms to the structure.
        [np.zeros(3), np.zeros(3),], # This causes no harm though.
    ],
    dtype=complex
)

# Now define the phase. We assume here a perfect helix.
phi = np.zeros([8, 2]).reshape(8,2,1)

phi[0] = np.dot(k, atoms.positions[0]) # k cdot Mn_1 pos, in Cartesian space
phi[1] = np.dot(k, atoms.positions[1])
phi[2] = np.dot(k, atoms.positions[2])
phi[3] = np.dot(k, atoms.positions[3])
phi /= 2*np.pi

# Set FCs using real, imaginary and phase components.
atoms.set_array("ri", ri)
atoms.set_array("phi", phi)
atoms.set_array("fc", real_imag_phase_to_fc(atoms))

#| ## Step 3: Computation
#| #
#| # Here we compute the local fields at the muon site with both
#| # real space and Ewals approaches.
#| #
#| # In the Ewald method we compute the dipolar tensors first.
#| # Notice that the dipolar tensors do not depend on Fourier components so
#| # they can be reused with different values for this parameter.
#| # However, to speedup the computation, the function will skip
#| # atoms that have zero FC (Si in this case).
#
# This step is required to later compute the local fields.
compute_dipolar_tensors(atoms)

# Compute local fields using dipolar tensors for the FC
# previously defined.
B_e = ewald_sum(atoms)

# Now repeat the same computation with the direct sum approach.
# A large cutoff radius is required (the wavelength is about 18 nm i.e.).a
# Below we use 200 Ang = 20 nm which is barely enough.
B_r = direct_sum(atoms, r_c=200)

# Print the results to compare the two approaches.
n_mu = len(atoms.info["mu"])
for i in range(n_mu):
    print(f"Muon site {i},\n")
    for j in range(2):  # 2 k vectors
        print(
            f"  Direct sum method for k{j+1}: {B_r[i, j]},\n  Ewald sum method for k{j+1}: {B_e[i, j]}"
        )
    print('\n')

#| ## Step 5: Incommensurate orders
#|
#| Here we change the phase of the FC. This is an incommensurate order
#| and the muon will probe all possible values from 0 to 1 (the values are
#| in units of 2π).

phase_points = 300
phases = np.linspace(0, 1, phase_points)
# Here we store results for Ewald sums.
# Order is number of phase points, number of muon sites, number of q vectors
fields_esum = np.zeros((phase_points, 4, 2))
# same for direct sums, but since real space sum
# is time consuming, we do it once every 10 phases and use a much smaller (too small!)
# cutoff radius.
fields_dsum = np.zeros((phase_points//10, 4, 2))

# Here we estimate the contact term.
# From the manuscript, A=-0.9276(20) mol/emu
# the factor 0.071884019 is documented here: https://muesr.readthedocs.io/en/latest/ContactTerm.html
A = -0.9276 * 0.071884019

# (2 magnetic_constant/3)⋅1bohr_magneton   = ((2 ⋅ magnetic_constant) ∕ 3) ⋅ (1 ⋅ bohr_magneton)
# ≈ 7.769376E-27((g⋅m^3) ∕ (A⋅s^2))
# ≈ 7.769376 T⋅Å^3

A *= 7.769376 * atoms.cell.volume / 4

for i, p in enumerate(phases):
    phi = np.zeros((8, 2, 1)) + p
    atoms.set_array("phi", phi)
    atoms.set_array("fc", real_imag_phase_to_fc(atoms))

    # Compute contact field
    M = direct_sum(atoms, r_c=2., only_magnetization=True)
    B_c = M * A

    # Compute dipolar field
    B_e = ewald_sum(atoms)

    fields_esum[i] = np.linalg.norm(B_e + B_c, axis=-1)

    if (i%10) == 0:
        # same as above with real space sum
        B_r = direct_sum(atoms, r_c=100)
        fields_dsum[i//10] = np.linalg.norm(B_r + B_c, axis=-1)


#| ## Step 6: Plot
#|
#| Finally we make a beautiful picture!
markers = ["o", "s", "^", "D"]
for i in range(4):
    plt.scatter(
        phases,
        fields_esum[:, i, 0],
        c="orange",
        s=12,
        marker=markers[i],
        label=f"$k_1 ^{{(RH)}}, \\mu_{i + 1}$",
    )
for i in range(4):
    plt.scatter(
        phases,
        fields_esum[:, i, 1],
        c="green",
        s=4,
        marker=markers[i],
        label=f"$k_1 ^{{(LH)}}, \\mu_{i + 1}$",
    )

for i in range(4):
    plt.plot(
        phases[::10],
        fields_dsum[:, i, 0],
        c="orange",
        marker=markers[i],
        label=f"$k_1 ^{{(RH)}}, \\mu_{i + 1}$",
        alpha=0.5,
    )
for i in range(4):
    plt.plot(
        phases[::10],
        fields_dsum[:, i, 1],
        c="green",
        marker=markers[i],
        label=f"$k_1 ^{{(LF)}}, \\mu_{i + 1}$",
        alpha=0.5
    )

plt.legend(title='Ewald (left) and Direct(right) dipolar sums', ncols=2)
plt.show()