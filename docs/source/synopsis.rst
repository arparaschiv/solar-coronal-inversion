Synopsis and Motivation
=======================

Synopsis
--------

**CLEDB** aims to invert coronal magnetic field information from observations of polarized light. The algorithm takes arrays of one or two sets of Spectro-polarimetric Stokes IQUV observations ``sobs\_in`` along with their header information. The data and metadata are pre-processed, and optimal corresponding sets of databases resulting from forward calculations are selected and read from disk storage. 

The data processing is split into two branches, based on the available polarized coronal Stokes observations: 

* **1-line branch:** 4 input IQUV observations *(one coronal emission line)*\ .
* **2-line branch:** 8 input I\ :sub:`1`\ Q\ :sub:`1`\ U\ :sub:`1`\ V\ :sub:`1`\ I\ :sub:`2`\ Q\ :sub:`2`\ U\ :sub:`2`\ V\ :sub:`2` observations *(two coronal emission lines)*\ .


Spectroscopic analysis products are computed for each line for both 1-line or 2-line branches.

The 1-line branch employs analytical approximations to calculate line of sight (LOS) integrated magnetic field products, while the 2-line branch offers access to additional magnetic field products. This second case benefits from more degrees of freedom allowing us to break degeneracies intrinsic in the inversion. Thus, the two-line algorithm branch performs a :math:`{\chi}^2` fitting  between the observation and a forward-modeled database to recover full 3D vector magnetic field components.

The databases are generated via forward modeling of combinations of input magnetic field and geometric parameters. Databases are used as a static input with respect to the inversion scheme and *are not* computed dynamically for each observation.

Motivation for the CLEDB approach
---------------------------------

By utilizing 2-line observations, we recover the 3D magnetic field information for single point voxels using a :math:`{\chi}^2` fitting approach. We employ the CLE spectral synthesis code to  generate forward-model calculations in the order of 10\ :sup:`7`\ -\ 10\ :sup:`9` atomic plasma and magnetic configurations are needed in order to satisfy a solution resolution criteria. Forward modeling such solutions in a dynamic fashion is time consuming. Such a calculation (e.g. iterating for 1 pixel) has execution times in the order of 10 hours, on a single thread, with a fast implementation of the FORTRAN CLE code. 

Building a separate database (the ``CLEDB\_BUILD`` module in our algorithm) to store a vast set of synthetic Stokes observations, along with the input plasma and magnetic field configurations responsible for producing polarized emission, proved to be a significantly more feasible approach. 

Additionally, the database theoretical calculations gain intrinsic access to otherwise un-observable input parameters (e.g. atomic alignment :math:`{\sigma}_0^2`, intrinsic magnetic field angles :math:`{\vartheta}_B`, :math:`{\varphi}_B` etc.) that can be used to break inherent degeneracies encountered when attempting analytical inversions (as for example occurring in the 1-line branch implementation). The dimensionality of the problem at hand can be further reduced by 1-2 orders of magnitude by using native symmetries when building and querying databases. Detailed discussions on the physics aspects of dimensionality reduction and degeneracy breaking effects can be found in the sources below.


List of resources
-----------------

Academic journal publications that helped build CLEDB:

1. `Paraschiv & Judge, SolPhys, 2022 <https://ui.adsabs.harvard.edu/abs/2022SoPh..297...63P/abstract>`_ covered the scientific justification of the algorithm, and the setup of the CLEDB inversion.
2. `Judge, Casini, & Paraschiv, ApJ, 2021 <https://ui.adsabs.harvard.edu/abs/2021ApJ...912...18J/abstract>`_ discussed the importance of scattering geometry when solving for coronal magnetic fields.
3. `Ali, Paraschiv, Reardon, & Judge, ApJ, 2022 <https://ui.adsabs.harvard.edu/abs/2022ApJ...932...22A/abstract>`_ performed a spectroscopic exploration of the infrared regions of the emission lines available for inversion with CLEDB.   
4. `Dima & Schad, ApJ, 2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...889..109D/abstract>`_ discussed potential degeneracies in using certain line combinations. The one-line CLEDB inversion directly utilizes the methods and results described in this work.
5. `Schiffmann, Brage, Judge, Paraschiv & Wang, ApJ, 2021 <https://ui.adsabs.harvard.edu/abs/2021ApJ...923..186S/abstract>`_ performed large-scale Lande g factor calculations for ions of interest and discusses degeneracies in context of their results.
6. `Casini & Judge, ApJ, 1999 <https://ui.adsabs.harvard.edu/abs/1999ApJ...522..524C/abstract>`_ and `Judge & Casini, ASP proc., 2001 <https://ui.adsabs.harvard.edu/abs/2001ASPC..236..503J/abstract>`_ described the theoretical line formation process implemented by CLE, the coronal forward synthesis Fortran code that is currently utilized by CLEDB. 
