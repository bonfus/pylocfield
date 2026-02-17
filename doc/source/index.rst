.. pylocfield documentation master file, created by
   sphinx-quickstart on Thu Feb  5 15:02:33 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pylocfield
==========

Pylocfield is a small library to compute magnetic dipolar sums.
It's mostly intended for muSR spectroscopy.
It implements both real space and Ewald based approaches.

A good introduction to the formalism of propagation vectors is given in
`Symmetry and magnetic structures, J. Rodríguez-Carvajal and F. Bourée, EPJ Web of Conferences 22, 00010 (2012) <https://www.epj-conferences.org/articles/epjconf/pdf/2012/04/epjconf_cscm2012_00010.pdf>`__.

.. toctree::
   theory
   usage


.. nbgallery::
    :caption: Examples
    :name: Examples
    :glob:
    :reversed:

    convergence
    fe
    mnsi
