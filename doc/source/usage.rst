Usage
=====

We give below a short introduction to the features of the library.
It's probably easier to start with the examples.

Defining the system
-------------------

The library uses ASE to define a periodic crystal. A few examples are shown below, for more details refer to their documentation.

.. code-block:: python
    
    # read any format supported by ASE
    from ase.io import read
    atoms = read('example.cif')

    # define a simple cubic structure
    from ase.build import bulk
    atoms = bulk("Po", crystalstructure="sc", a=3.35, basis=[[0, 0, 0]])


Defining muon position
----------------------

All additional information are stored inside the ASE Atoms object, using the `info` dictionary and the `get_array`/`set_array` methods
using numpy arrays.

For muon positions, a `(N,3)` array specifies the positions in reduced coordinates.

.. code-block:: python

    atoms.info['mu'] = np.array([[0,0,0]]) # a muon at the origin of the unit cell.



Defining magnetic orders
------------------------

To specify a magnetic order, one needs both the propagation vector(s) and fourier components.
For propagation vectors one generally specifies only a representative of the (k,-k) pair.

This is done as by setting `atoms.info['q']` with a (N,3) array in reduced reciprocal space coordinates.

.. code-block:: python

    atoms.info['q'] = np.array(
        [
            [0.0, 0.0, 0.5],
        ]
    )


Multiple propagation vectors can be specified with

.. code-block:: python

    atoms.info['q'] = np.array(
        [
            [0.0, 0.0, 0.5],  # Propagation vector 1
            [0.5, 0.5, 0.0],  # Propagation vector 2
        ]
    )


Fourier components are specified in Cartesian space with a numpy array having the dimension (N_a, N_q, 3) in units of Bohr magnetons.

An example is given below:

.. code-block:: python

    atoms.set_array('fc') = \
    np.array(
            [  # first atom
                [  # first k vector
                    [1, 0, 0]  # fourier components for first atom, first k
                ],
                [  # second k vector
                    [2, 0, 0]  # fourier components for first atom, second k
                ],
            ],
            [  # second atom
                [  # first k vector
                    [1, 0, 0]  # fourier components for second atom, first k
                ],
                [  # second k vector
                    [2, 0, 0]  # fourier components for second atom, second k
                ],
            ],
        dtype=complex,
    )


Alternatively, the function `real_imag_phase_to_fc` inside `pylocfield.utils` can be used to
define Fourier components in terms of real, imaginary and phase parts.
An example is given below.


Computing the local field
-------------------------

The local field at the muon site can be computed with both a direct sum in real space or the
Ewald approach. The latter is much faster.

An example is given below.

.. code-block:: python

    from pylocfield.ewald import compute_dipolar_tensors, compute_field as ewald_sum

    # Compute dipolar tensors with Ewald method
    compute_dipolar_tensors(atoms)

    # Compute local fields using dipolar tensors
    B_e = ewald_sum(atoms)

The results are reported in Tesla per muon site.