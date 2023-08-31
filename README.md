`SubGen2` is a subhalo population generator which works for both CDM and WDM of arbitrary DM particle mass. It can be used to generate a population of subhaloes according to the joint distribution of subhalo bound mass, infall mass and halo-centric distance in a halo of a given mass. 

`SubGen2` is an extension to the [`SubGen`](https://github.com/Kambrian/SubGen) which works only for CDM subhaloes.
### Prerequisites

You need a python installation with the core scientific packages: `numpy`, `matplotlib` and `scipy`.
You also need the `emcee` package(http://dan.iel.fm/emcee) for MCMC sampling. Try

     easy_install numpy matplotlib scipy emcee

or

     pip install numpy matplotlib scipy emcee

to install these dependences if you miss them.

## Usage
The usage of this code is almost the same as the `SubGen` code, except for an additional parameter specifying the WDM particle mass.

For example, to sample subhaloes inside a host halo of mass 1e14 Msun/h, for WDM with a thermal relic particle mass of 1.3 keV
```
from subgen2 import *
sample=SubhaloSample(M=1e4, WDMmass=1.3) 
```
Please consult the original `SubGen` documentation for more instructions and examples at: https://github.com/Kambrian/SubGen 

## References
If you make use of this code, please cite the following papers:
- `SubGen2`: He, Han, Gao & Zhang, 2023 (in prep.), [Extending the unified subhalo model to warm dark matter](TBA) 
- `SubGen`: Han, Cole, Frenk & Jing, 2016, MNRAS, [A unified model for the spatial and mass distribution of subhaloes](http://arxiv.org/abs/1509.02175) 
