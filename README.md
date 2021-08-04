# **CLEDB CORONAL FIELD DATABASE INVERSION**
### [solar-coronal-inversion repository on github](https://github.com/arparaschiv/solar-coronal-inversion/)

Main repository for the **CLEDB** (Coronal Line Emission DataBase) code distribution.

**Authors:** Alin Paraschiv & Philip Judge. National Solar Observatory & High Altitude Observatory

**Contact:** arparaschiv "at" nso.edu; paraschiv.alinrazvan+cledb "at" gmail.com

#### **Main aim:** 
Invert coronal vector magnetic field products from observations of polarized light. 
The algorithm takes arrays of one or two sets of spectroscopic Stokes IQUV observations
to derive line of sight or full vector magnetic field products 

#### **Applications:** 
Inverting magnetic field information from spectro-polarimetric solar coronal observations from instruments like DKIST Cryo-NIRSP; DL-NIRSP; COMP/UCOMP; 

### **Documentation**

1. Extensive documentation, **including instalation instruction, dependencies, algorithm scematics and much more** is available in a dedicated documentation writeup. [README-CODEDOC.pdf](./codedoc-latex/README-CODEDOC.pdf).
2. Additional in-depth documentation for the bash/fortran parallel database generation module is also provided; [README-RUNDB.md](./CLEDB_BUILD/README-RUNDB.md)
3. This is a alpha-level release. Not all functionality is implemented. [TODO.md](./TODO.md) documents current issues and functions to be implemented in the near future.
### **System platform compatibility**

1. Debian+derivatives Linux x64       -- all inversion modules are fully working.
2. OSX (Darwin) Catalina and Big Sur  -- all inversion modules are fully working; One additional homebrew package required. See README-CODEDOC.pdf.
3. Windows platform                   -- not tested.

### **Examples**
After installing the package and generating databases, as describes in the README-CODEDOC,
both 1-line and 2-line implementations of CLEDB can be tested with synthetic data using the two provided Jupyter notebook examples   
1. [test_1line](./test_1line.ipynb)
2. [test_2line](./test_2line.ipynb)

The synthetic test data is [hosted separately here.](https://drive.google.com/file/d/1XpBxEwUUyaqYy1NjbVKyCHJhMUKzoV_m/view?usp=sharing).
Both examples are expected to fully execute in a correct installation.

### **Scholarly works supporting the CLEDB inversion**
1. [Judge, Casini, & Paraschiv, ApJ, 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...912...18J/abstract) 
discusses the importance of scattering geometry when solving for coronal magnetic fields.
2. [Paraschiv & Judge, in prep A, 2021](No link yet) covers the scientific justification of the algorithm, and the setup of the CLEDB inversion.
3. [Paraschiv & Judge, in prep B, 2021](No link yet) performs synthetic observation benchmarks of the CLEDB algorithm.
4. [Ali, Paraschiv, & Reardon, in prep, 2021](no link yet) spectroscopically explored the infrared regions of 
    the emission lines available for inversion with CLEDB. 
5. The CLEDB inversion evolved from the CLE fortran code written by Philip Judge and Roberto Casini. 
The theoretical formulation of the coronal inversion problem is best described in [Casini & Judge, ApJ, 1999](https://ui.adsabs.harvard.edu/abs/1999ApJ...522..524C/abstract)


