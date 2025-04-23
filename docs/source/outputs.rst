.. _outputs-label:

Output Products
===============

Output Variable Overview
------------------------

The main CLEDB inversion algorithm outputs are stored in the following variables:

* **specout**
	12 **SPECTRO_PROC** output products. These are described :ref:`here <specout-label>`\ .

* **blosout**
	4 **BLOS_PROC** output products. These are described :ref:`here <blos-label>`\ .

* **invout**
	11 **CLEDB_INVPROC** output products. These are described :ref:`here <invout-label>`\ .

* **sfound**
	11 **CLEDB_INVPROC** matched profile list. These are described :ref:`here <sfound-label>`\ .

* **issuemask**
	Records any issues that arise in processing for each pixel (to be implemented). The issuemask will be updated by both modules.

.. note::
	The global process followed to produce these outputs is sketched in :ref:`module_flow-label`.


.. _issuemask-label:

Current Issuemask Implementation
----------------------------------

The inversion will implement a confidence and/or potential issue map of size [nx,ny,nline] for all spatial pixels in an input observation that will be returned along with the main output products. This mask flags potential interpretation issues in the case of successful inversion but not inversion errors. These can be printed through extended verbosity.
The issuemask variable is created by the sobs_preprocess function and dinamycally updated by the downstream processing.

.. Important::
	Issuemask encoding advises the user of one or more potential issues in a specific observation location. It does not directly mean that the observation is compromised or that an error occurred. Individual judgment is advised.

Current implementation of issuemask coding:

Code 0
    No apparent problem in specified pixel or area.

Code 1
    WARNING: One or more of Stokes I, Q, U are lower than noise SNR threshold.

Code 2
    WARNING: Stokes V is lower than noise SNR threshold.

Code 4
    WARNING: Observed height is greater than the maximum height extent of the density look-up table.


Code 8 & Code 16
    \----------------TBD - reserved for prepinv codes----------------------

Code 32
    WARNING: :math:`\Phi_B` is lower than noise threshold (for 1-line observations).  Theta_B possibly influenced by the Stokes QU noise threshold or arctangent asymptote (for 1-line observations).

.. note::
	We check the behavior of the U/Q ratio around the asymptotic behavior of the arctangent function. If either component is close to 0, this will lead to uncertainties in the retrieved angle. As either Q or U can cross through 0, this does not necessarily represent a measurement error, but rather higher uncertainties in evaluating the arctangent.

Code 64
    WARNING: Linear polarization azimuth might be close to Van-Vleck ambiguity (for 1-line observations).

.. note::
	In this case, there  is no access to the LVS magnetic variables. We only check for the simultaneous vanishing of the Stokes Q and U parameters.

Code 128
    WARNING: Initial Gaussian fit guess parameters not accurately found and/or signal is not following a Gaussian distribution (1-line observations).

Code 256
    \----------------TBD - reserved for 1-line processing codes----------

Code 512

    WARNING: Linear polarization azimuth on at least one solution is close to Van-Vleck ambiguity (for 2-line observations).

.. note::
	If ANY of the returned 2-line solutions are close to the Van-Vleck ambiguity, this flag gets enabled.

Code 1024
    WARNING: Database fit failed to converge reliably (for 2-line obs).

Code 2048
    WARNING: One or more of B components, are lower than noise threshold (for 2-line observations).

Code 4096
    \----------------TBD - reserved for 2-line processing codes----------


Encoding the information is done sequentially when progressing through the different modules. This will be done by using a decimal to binary conversion to map the codes. The issuemask values thus become cumulative. Following the sketch map encoding from above, we take for example a pixel from a 1-line observation with unreliable Stokes V signal. The uncertainty in Stokes V will also lead to compromised B\ :math:`_{LOS}` information. Thus, the *issuemask* will encode a value of 130 for that respective pixel.
