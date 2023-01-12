# """
# @author: Alin Paraschiv paraschiv.alinrazvan+cledb@gmail.com
# """

## To load the class: 
#par=ctrlparams()
#print(vars(par))
#print(a.__dict__)

class ctrlparams:
    def __init__(self):
        ## general params
        self.dbdir        = '/home/noxpara/Documents/physics_prog/cle/db204_R0500/'                       ## directory for database
        self.verbose      = 1                                                                             ## verbosity parameter
        
        ## Used in CLEDB_PREPINV
        self.integrated = False                                                                           ## Boolean; parameter for switching to line-integrated data such as CoMP/uCoMP/COSMO        
        
        ## Used in CLEDB_PROC
        self.nsearch      = 8                                                                             ## number of closest solutions to compute
        self.maxchisq     = 100                                                                           ## Stop computing solutions above this chi^2 threshold
        self.gaussfit     = 2                                                                             ## Gaussian parametric fitting to use instead of the standard CDF fitting
        self.bcalc        = 0                                                                             ## control parameter for computing the B magnitude for two line observations.
        self.reduced      = True                                                                          ## Boolean; parameter for reduced database search 
        self.iqud         = False                                                                         ## Boolean; parameter for IQU + Doppled data matching when Stokes V is not measurable
    
        ##numba jit flags
        self.jitparallel  = True                                                                          ## Boolean; Enable or disable numba jit parralel interpreter
        self.jitcache     = False                                                                         ## Boolean; Jit caching for slightly faster repeated execution. Enable only after no changes to @jit functions are required. Otrherwise kernel restarts are needed to clear caches. 
        self.jitdisable   = False                                                                         ## Boolean; enable or disable numba jit entirely; Requires python kernel restart!
        
        import yaml                                                                                       ## Workaround to save the jitdisable keyword to a separate config file.
        names={'DISABLE_JIT' : self.jitdisable}                                                           ## Working kernel needs to be reset for numba to pick up the change
        with open('.numba_config.yaml', 'w') as file:                                                     ## more info on numba flags can be found here: https://numba.readthedocs.io/en/stable/reference/envvars.html
            yaml.dump(names, file)

