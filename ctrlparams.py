# -*- coding: utf-8 -*-
# """
# @author: Alin Paraschiv
# """

class ctrlparams:
    """
    A set of inversion controlling parameters grouped in a convenient class that is passed along to inversion modules.

    To load the class:
    $par=ctrlparams()
    $print(vars(par))
    $print(a.__dict__)
    """
    def __init__(self):
        """
        Class only performs a main initialization for all variables.
        """
        ## general params
        self.dbdir        = './CLEDB_BUILD/'                                                   ## directory for database
        self.lookuptb     = self.dbdir + 'chianti_v10.1_pycelp_fe13_h99_d120_ratio.npz'        ## CHIANTI density look-up table
        self.verbose      = 2                                                                  ## verbosity parameter
        self.ncpu         = 100                                                                ## Number of cpu cores to use. If more than the available cores are requested, n-2 available cores are used.

        ## Used in CLEDB_PREPINV
        self.integrated   = False                                                              ## Boolean; parameter for switching to line-integrated data such as CoMP/uCoMP/COSMO
        self.dblinpolref  = 0                                                                  ## Parameter for changing the database calculation linear reference. Should not need changing in normal situations. radian units.
        self.instwidth    = 0                                                                  ## Parameter for fine-correcting non-thermal widths if instrument widths are known or computed by user. nm units.
        self.atmred       = False                                                              ## Parameter that controls whether to reduce photospheric and atmospheric contributions using spectral atlases. Useful for Cryo-NIRSP L1 data.

        ## Used in CLEDB_PROC
        self.nsearch      = 8                                                                  ## number of closest solutions to compute
        self.maxchisq     = 1000                                                               ## Stop computing solutions above this chi^2 threshold
        self.gaussfit     = 2                                                                  ## Gaussian parametric fitting to use instead of the standard CDF fitting
        self.bcalc        = 0                                                                  ## control parameter for computing the B magnitude for two line observations.
        self.reduced      = False                                                              ## Boolean; parameter for reduced database search using linear polarization azimuth
        self.iqud         = False                                                              ## Boolean; parameter for IQU + Doppler data matching when Stokes V is not measurable

        ##numba jit flags
        self.jitparallel  = False                                                              ## Boolean; Enable or disable numba jit parralel interpreter. Parallel runs
        self.jitcache     = False                                                              ## Boolean; Jit caching for slightly faster repeated execution. Enable only after no changes to @jit functions are required. Otrherwise kernel restarts are needed to clear caches.
        self.jitdisable   = True                                                               ## Boolean; enable or disable numba jit entirely; Requires python kernel restart!

        # import yaml                                                                           ## Workaround to save the jitdisable keyword to a separate config file to be read by numba.
        # names={'DISABLE_JIT' : self.jitdisable}                                               ## Working kernel needs to be reset for numba to pick up the change
        # with open('.numba_config.yaml', 'w') as file:                                         ## more info on numba flags can be found here: https://numba.readthedocs.io/en/stable/reference/envvars.html
        #     yaml.dump(names, file)