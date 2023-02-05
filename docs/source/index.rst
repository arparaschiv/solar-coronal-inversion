.. solar-coronal-inversion documentation master file, created by
   sphinx-quickstart on Thu Jan 26 18:38:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the **CLEDB** Distribution
============================================
.. image:: https://img.shields.io/badge/GitHub-arparaschiv%2Fsolar--coronal--inversion-blue.svg?style=flat
   :target: https://github.com/arparaschiv/solar-coronal-inversion
   :alt: CLEDB repository

------------------------------------------------------

**Purpose:**


The **C**\ oronal **L**\ ine **E**\ mission **D**\ ata\ **B**\ ase inversion is a **public** Python based tool and pipeline that is used to infer magnetic field information from SpectroPolariemtric observations of the Solar Corona.
This document describes the main concepts, functions, and variables, comprising the CLEDB inversion algorithm for inverted coronal magnetic fields.

-----------------------------------------------------------------------------

**Authors and Contact:** 

| Alin Paraschiv and Philip Judge 
| -- High Altitude Observatory NCAR|UCAR 
| -- National Solar Observatory, AURA

| arparaschiv *at* ucar edu  or paraschiv.alinrazvan+cledb *at* gmail

-----------------------------------------------------------------------

.. Check out the :doc:`usage` section for further information.

.. caution::
   Vector coronal magnetometry is not yet advanced enough as to recover HMI-like magnetograms. 
   A user should not expect such a product. 
   Solution degeneracies exist, and the user 
   is required to make a decision on how to interpret the inversion outputs and 
   which products to utilize for their science. 

.. Danger::
   This setup should be considered at best beta-level release. 
   This setup tackles a problem for which currently no observation data exists and where 
   a complex forward-model + inversion is required. 
   Bugs or unintentional bad outcomes will still exist at this point in time. 
   Please get in touch with us about any "feature" you discover.

.. note::
   Last updated at the :ref:`*update-readthedocs*<todo-label>` commit tag.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   synopsis
   module_overview
   install
   inputvars
   cledb_build
   cledb_prepinv
   cledb_proc
   outputs
   readmes
   changelog
   glossary 



..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`