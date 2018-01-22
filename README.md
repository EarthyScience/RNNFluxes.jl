# RNNFluxes

| **Documentation**                                                                                                        | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://bgi-jena.github.io/RNNFluxes.jl/latest) | [![Build Status](https://travis-ci.org/bgi-jena/RNNFluxes.jl.svg?branch=master)](https://travis-ci.org/bgi-jena/RNNFluxes.jl)|

An RNN and LSTM implementation to solve regression problems. Initially implemented to estimate carbon exchange fluxes including history.

# Installation

In Julia, run

````julia
Pkg.clone("RNNFluxes")
````

To plot the progress during training, run your session inside a Jupyter Notebook and add the following lines somewhere to enable inline plotting:

````julia
using Plots
default(show=:inline)
````
