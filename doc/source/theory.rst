Theory
======

A short description of the key concepts used in this code are provided below.

Magnetic orders
---------------

Magnetic orders should be specified using the propagation vector formalism.

A generic magnetic structure is defined by

.. math::
    :label: moment

    m_{lj} = \sum _{\mathbf{k}} = \mathbf{S}_{\mathbf{k}j} \exp (-2 \pi i \mathbf{k} \cdot \mathbf{R}_l)

where :math:`\mathbf{S}_{\mathbf{k}j}` are the Fourier coefficients for for atom :math:`j` and propagation
vector :math:`\mathbf{k}`. The vectors :math:`\mathbf{R}_{l}` identify the origin of cell :math:`l`, in which atoms
have positions :math:`\mathbf{R}_{lj}=\mathbf{R}_{l}+\mathbf{r}_{j}`.
The sum extends to one or more pairs of :math:`(\mathbf{k}, -\mathbf{k})` vectors.

Propagation vectors are defined in the first Brillouin zone and :math:`\mathbf{S}_{\mathbf{k}j} = \mathbf{S}^* _{-\mathbf{k}j}`.

For convenience, the expression of the Fourier components on the atom :math:`j` may be reported as:

.. math::
    :label: realimag

    \mathbf{S}_{\mathbf{k}j} = \frac{1}{2} \left\{ \mathbf{R}_{\mathbf{k}j} + i \mathbf{I}_{\mathbf{k}j}  \right\} \exp (-2 \pi \phi_{\mathbf{R}_{\mathbf{k}j}})

where :math:`\mathbf{R}_{\mathbf{k}j}, \mathbf{I}_{\mathbf{k}j}` 
are real vectors defining, together with the phase :math:`\phi_{\mathbf{k}j}`, the 
Fourier components for each propagation vector. When :math:`\phi_{\mathbf{k}j}=0`, the two vectors are just the real
and imaginary parts of the Fourier components.
This leads to

.. math::
    :label: moms

    \mathbf{m}_{lj} = \sum_{\langle \mathbf{k} \rangle} \left\{ \mathbf{R}_{\mathbf{k}j}  \cos \left( 2 \pi \mathbf{k} \cdot \mathbf{R}_l + \phi_{\mathbf{k}j} \right) + \mathbf{I}_{\mathbf{k}j} \sin \left( 2 \pi \mathbf{k} \cdot \mathbf{R}_l + \phi_{\mathbf{k}j} \right) \right\}

where the sum extends over pairs of :math:`(\mathbf{k}, -\mathbf{k})` vectors.


Dimensions and units
--------------------

Fourier components are defined in Cartesian space in units of Bohr magntons.
Magnetic fields are reported in Tesla.