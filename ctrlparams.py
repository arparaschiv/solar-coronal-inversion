"""
@author: Alin Paraschiv paraschiv.alinrazvan+cledb@gmail.com
"""

## To load the class: 
#par=ctrlparams()
#print(vars(par))
#print(a.__dict__)

class ctrlparams:
    def __init__(self):
        self.dbdir    = '/home/alin/Documents/physics_prog/cle/test_cle_degeneracy/db202_R0500/'   ## directory for database
        self.verbose  = 1                                                                          ## verbosity parameter
        self.reduced  = 1                                                                          ## boolean parameter for reduced search
        self.nsearch  = 4                                                                          ## number of closest solutions to compute
        self.bcalc    = 0                                                                          ## control parameter for computing the B magnitude for two line observations.
        self.maxchisq = 10                                                                         ## Stop computing solutions above this chi^2 threshold
        self.gaussfit = 2                                                                          ## Gaussian parametric fitting to use instead of the standard CDF fitting
        ##numba jit flags
        self.numbajitflag = 1                                                                      ## Enable or disable numba jit XXXX
        self.jitparallel = True
        