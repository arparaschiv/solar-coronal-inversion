## -*- coding: utf-8 -*-
## """
## @author: Alin Paraschiv paraschiv.alinrazvan+cledb@gmail.com
##
## """
## TODO: update final form of constants and units

## To load the class: 
#consts=Constants()
#print(vars(consts))
#print(consts.__dict__)

class Constants:
    def __init__(self,ion):
        ## Solar units in different projections
        #self.solar_diam_arc = 1919
        #self.solar_diam_deg = self.solar_diam_arc/3600.
        #self.solar_diam_rad= np.deg2rad(0.0174533self.solar_diam_deg)
        #self.solar_diam_st = 2.*np.pi*(1.-np.cos(self.solar_diam_rad/2.))

        ##Physical constants
        self.l_speed       = 2.9979e+8                     ## speed of light [m s^-1]
        self.kb            = 1.3806488e-23                 ## Boltzmann constant SI [m^2 kg s^-2 K^-1];
        self.e_mass        = 9.10938356e-31                ## Electron mass SI [Kg]
        self.e_charge      = 1.602176634e-19               ## Electron charge SI [C]
        self.bohrmagneton  = 9.274009994e-24*1.e-4         ## Bohr magneton [kg⋅m^2⋅s^−2 G^-1]; Mostly SI; T converted to G;
        self.planckconst   = 6.62607004e-34                ## Planck's constant SI [m^2 kg s^-1];

        # ion/line specific constants
        if (ion == "fe-xiii_1074"):
            self.ion_temp = 6.25                      ## Ion temperature SI [K]; li+2017<--Chianti
            self.ion_mass = 55.847*1.672621E-27       ## Ion mass SI [Kg] ## for ion XIII this needs to be computed for all 
            #self.line_ref = 1074.62686                ## CLE Ion referential wavelength [nm]
            self.line_ref = 1074.68                   ## Ion referential wavelength [nm]
            self.width_th = self.line_ref/self.l_speed*(4.*0.69314718*self.kb*(10.**self.ion_temp)/self.ion_mass)**0.5   ## Line thermal width 
            self.F_factor= 0.0                             ## Dima & Schad 2020 Eq. 9
            self.gu = 1.5                                  ## upper level g factor
            self.gl = 1                                    ## lower level g factor
            self.ju = 1                                    ## upper level angular momentum
            self.jl = 0                                    ## lower level angular momentum
            self.g_eff=0.5*(self.gu+self.gl)+0.25*(self.gu-self.gl)*(self.ju*(self.ju+1)-self.jl*(self.jl+1))    ## LS coupling effective Lande factor; e.g. Landi& Landofi 2004 eg 3.44; Casini & judge 99 eq 34

        elif (ion == "fe-xiii_1079"):
            self.ion_temp = 6.25                      ## Ion temperature SI [K]; li+2017<--Chianti
            self.ion_mass = 55.847*1.672621E-27       ## Ion mass SI [Kg] ## for ion XIII this needs to be computed for all 
            #self.line_ref = 1079.78047                ## CLE Ion referential wavelength [nm]
            self.line_ref = 1079.79                   ## Ion referential wavelength [nm]
            self.width_th = self.line_ref/self.l_speed*(4.*0.69314718*self.kb*(10.**self.ion_temp)/self.ion_mass)**0.5  ## Line thermal width 
            self.F_factor= 0.0                        ## Dima & Schad 2020 Eq. 9
            self.gu = 1.5                             ## upper level g factor
            self.gl = 1.5                             ## lower level g factor
            self.ju = 2                               ## upper level angular momentum
            self.jl = 1                               ## lower level angular momentum
            self.g_eff=0.5*(self.gu+self.gl)+0.25*(self.gu-self.gl)*(self.ju*(self.ju+1)-self.jl*(self.jl+1))    ## LS coupling effective Lande factor; e.g. Landi& Landofi 2004 eg 3.44; Casini & judge 99 eq 34

        elif (ion == "si-x_1430"):
            self.ion_temp = 6.15                      ## Ion temperature SI [K]; li+2017<--Chianti
            self.ion_mass = 28.0855*1.672621E-27      ## Ion mass SI [Kg] ## for ion XIII this needs to be computed for all 
            self.line_ref = 1430.2231                 ## CLE Ion referential wavelength [nm] ;;needs to be double-checked with most current ATOM
            #self.line_ref = 1430.10                   ## Ion referential wavelength [nm]
            self.width_th = self.line_ref/self.l_speed*(4.*0.69314718*self.kb*(10.**self.ion_temp)/self.ion_mass)**0.5  ## Line thermal width 
            self.F_factor= 0.5                        ## Dima & Schad 2020 Eq. 9
            self.gu = 1.334                           ## upper level g factor
            self.gl = 0.665                           ## lower level g factor
            self.ju = 1.5                             ## upper level angular momentum
            self.jl = 0.5                             ## lower level angular momentum
            self.g_eff=0.5*(self.gu+self.gl)+0.25*(self.gu-self.gl)*(self.ju*(self.ju+1)-self.jl*(self.jl+1))    ## LS coupling effective Lande factor; e.g. Landi& Landofi 2004 eg 3.44; Casini & judge 99 eq 34

        elif (ion == "si-ix_3934"):
            self.ion_temp = 6.05                      ## Ion temperature SI [K]; li+2017<--Chianti
            self.ion_mass = 28.0855*1.672621E-27      ## Ion mass SI [Kg] ## for ion XIII this needs to be computed for all
            self.line_ref = 3926.6551                 ## CLE Ion referential wavelength [nm] ;;needs to be double-checked with most current ATOM
            #self.line_ref = 3934.34                   ## Ion referential wavelength [nm]
            self.width_th = self.line_ref/self.l_speed*(4.*0.69314718*self.kb*(10.**self.ion_temp)/self.ion_mass)**0.5  ## Line thermal width 
            self.F_factor= 0.0                        ## Dima & Schad 2020 Eq. 9
            self.gu = 1.5                             ## upper level g factor
            self.gl = 1                               ## lower level g factor
            self.ju = 1                               ## upper level angular momentum
            self.jl = 0                               ## lower level angular momentum
            self.g_eff=0.5*(self.gu+self.gl)+0.25*(self.gu-self.gl)*(self.ju*(self.ju+1)-self.jl*(self.jl+1))    ## LS coupling effective Lande factor; e.g. Landi& Landofi 2004 eg 3.44; Casini & judge 99 eq 34
        else:
            print("Not supported ion or wrong string. Ion not Fe fe-xiii_1074, fe-xiii_1079, si-x_1430 or si-ix_3934.\nIon specific constants not returned!")
