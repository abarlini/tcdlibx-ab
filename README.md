# tcdlibx
A small library to handle volumetric datasets containing  vectorial fields

## Required Packages ##
 - `numpy`,
 - [`estampes`](https://github.com/jbloino/estampes)

### Optional ###
 - `vtk`,
 - `PySide6`
 - `numba`

## Install ###
In your virtual env

Install manually the `estampes` dependency with:
`pip install -e git+https://github.com/mast-theolab/estampes.git#egg=estampes`

To install only the library:
`pip install -e .`

to include GUI and visualization capabilities
`pip install -e .[gui]`

numba should be installed independently 

## TODO ##
 - [ ] control the units
 - [ ] a lot more... 



