.. _outputs-label:

Output Products
===============

The main CLEDB inversion algorithm outputs are stored in the following variables:

* **specout** 
	12 ``SPECTRO_PROC`` output products. These are described :ref:`here <specout-label>`\ .

* **blosout**  
	4 ``BLOS_PROC`` output products. These are described :ref:`here <blos-label>`\ .

* **invout**  
	11 ``CLEDB_INVPROC`` output products. These are described :ref:`here <invout-label>`\ .

* **issuemask**  
	Records any issues that arise in processing for each voxel (to be implemented). The issuemask will be updated by both modules.

.. note::
	The global process followed to produce the outputs is sketched in in :ref:`module_flow-label`.


.. _issuemask-label:

Tentative Issuemask Implementation
----------------------------------

The inversion will implement a confidence/issue map of size [nx,ny] for all spatial pixels in an input observation that will be returned along with the main output products. 

.. Important::
	Issuemask encoding not currently active. Final form to be decided and implemented. 


Example of issuemask coding:

Code 0
	No apparent problem in voxel.

Code 1
	One or more of Stokes I, Q, U are lower than noise RMS threshold.

Code 2
    Stokes V is lower than noise RMS threshold.

Code 4
    Linear polarization is close to Van-Vleck ambiguity (warning).  

Code 8
    B\ :math:`_{LOS}` or :math:`\Phi_B` is lower than noise threshold (for 1-line observations).

Code 16
    Database fit failed to converge reliably (for 2-line obs).

Code 32
    One or more of B, :math:`\Phi_B`, :math:`\Theta_B` are lower than noise threshold (for 2-line observations).

Code 64
	\-----------------------------------------    

Code 128
	\-----------------------------------------

Encoding the information sequentially when processing through the different modules will be done by using powers of 2. The issuemask values are thus cumulative. Following the map coding from above, we take for example a pixel from a 1-line observation with unreliable Stokes V signal. The uncertainty in Stokes V will also lead to compromised B\ :math:`_{LOS}` information. Thus, the *issuemask* will encode a value of 10 for that respective pixel.
