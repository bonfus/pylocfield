import numpy as np
from ase.spacegroup import crystal

from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum
from pylocfield.realspace import compute_field as direct_sum
from pylocfield.utils import add_equivalent_muon_sites, real_imag_phase_to_fc

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=5)

def uv(vec):
    return vec/np.linalg.norm(vec)

# Create simple cubic lattice
a = 4.558 # Å
atoms = crystal("MnSi", 
                basis = [
                            [0.138,0.138,0.138],
                            [0.845,0.845,0.845] 
                        ],
                spacegroup=198,
                cellpar=[a,a,a,90,90,90])

# A copuple of muon positions in fractional coordinates
atoms.info['mu'] = np.array(
    [
        [0.45,0.45,0.45],
    ]
)

# add equivalent muon sites
add_equivalent_muon_sites(atoms)

# norm of propagation vector (in reciprocal space)
norm_k = 0.036 # Å −1

# reduced coordinates for propagation vector
k = norm_k * uv(np.ones(3))

k_rlu = atoms.cell.reciprocal().scaled_positions(k) / (2 * np.pi)

# A magnetic structure, we define two propagation vectors in
# reciprocal lattice units (i.e. these are fractional coordinates)
atoms.info['q'] = np.array(
    [
        k_rlu,  # Propagation vector 1
        k_rlu,  # Propagation vector 2
    ]
)



k_u = uv([1,1,1])
a = uv([1,-1,0])
b = np.cross(k_u,a)

ri = 0.385 * np.array(
    [
        [  # first atom
                a+1.j*b  # fourier components for first k
            ,
          # second k vector
                a-1.j*b  # fourier components for second k
            ,
        ],
        [  # first atom
                a+1.j*b  # fourier components for first k
            ,
          # second k vector
                a-1.j*b  # fourier components for second k
            ,
        ],
        [  # first atom
                a+1.j*b  # fourier components for first k
            ,
          # second k vector
                a-1.j*b  # fourier components for second k
            ,
        ],
        [  # first atom
                a+1.j*b  # fourier components for first k
            ,
          # second k vector
                a-1.j*b  # fourier components for second k
            ,
        ],
        [np.zeros(3), np.zeros(3),],
        [np.zeros(3), np.zeros(3),],
        [np.zeros(3), np.zeros(3),],
        [np.zeros(3), np.zeros(3),],
    ],
    dtype=complex
)

phi = np.zeros([8, 2]).reshape(8,2,1)

pos_Mn1 = atoms.get_scaled_positions()[1]

phi[0] = np.dot(k, atoms.positions[0])
phi[1] = np.dot(k, atoms.positions[1])
phi[2] = np.dot(k, atoms.positions[2])
phi[3] = np.dot(k, atoms.positions[3])

# Set Fourier components using real, imaginary and phase parts
atoms.set_array("ri", ri)
atoms.set_array("phi", phi)

atoms.set_array("fc", real_imag_phase_to_fc(atoms))


# Compute dipolar tensors with Ewald method
compute_dipolar_tensors(atoms)

# Compute local fields using dipolar tensors
B_e = ewald_sum(atoms)

# Compute the same fields in real space
B_r = direct_sum(atoms, Rc=200)


n_mu = len(atoms.info["mu"])
for i in range(n_mu):
    print(f"Muon site {i},\n")
    for j in range(2):  # 2 k vectors
        print(
            f"  Direct sum method for k{j+1}: {B_r[i, j]},\n  Ewald sum method for k{j+1}: {B_e[i, j]}"
        )
    print('\n')

phase_points = 300
phases = np.linspace(0, 1, phase_points)
# number of phase points, number of muon sites, number of q vectors
fields = np.empty((phase_points, 4, 2))

for i, p in enumerate(phases):
    phi = np.zeros((8, 2, 1)) + p
    atoms.set_array("phi", phi)
    atoms.set_array("fc", real_imag_phase_to_fc(atoms))

    fields[i] = np.linalg.norm(ewald_sum(atoms), axis=-1)

markers = ["o", "s", "^", "D"]
for i in range(4):
    plt.scatter(
        phases,
        fields[:, i, 0],
        c="orange",
        s=12,
        marker=markers[i],
        label=f"$k_1, \\mu_{i + 1}$",
    )
for i in range(4):
    plt.scatter(
        phases,
        fields[:, i, 1],
        c="green",
        s=4,
        marker=markers[i],
        label=f"$k_2, \\mu_{i + 1}$",
    )

plt.legend()
plt.show()