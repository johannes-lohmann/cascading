# cascading

This repository contains code that underlies the publication "Abrupt climate change as a rate-dependent cascading tipping point" by Johannes Lohmann et al.
The following scripts have been used to perform all simulations presented in the paper.

Calculation and plotting of bifurcation diagrams of the Stommel model:
plot_bifdiagrams.py

Simulations with the coupled model:
seaice_ocean.py 

Cython code for the simulation of the SDEs of the coupled model: 
seaice_ocean_cy_noise.pyx

Simulations with the Stommel-only model:
stommel_rtip.py

Cython code for the simulation of the SDEs of the Stommel model: 
stommel_cy_noise.pyx

Cython functions for fast calculation of early-warning signals:
ews_cy.pyx

Further functions to calculate early-warning signals:
ews_functions.py
