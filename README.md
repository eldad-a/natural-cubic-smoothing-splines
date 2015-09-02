# natural-cubic-smoothing-splines
#### Cubic smoothing splines with natural boundary conditions and automated choice of the smoothing parameter

A natural cubic smoothing splines module to smooth-out noise and obtain an estimate of the first two derivatives 
(velocity and acceleration in the case of a particle trajectory). 

Various methods have been introduced for the automatic choice of the smoothing parameter.

It can also be used to get an interpolating natural cubic spline.

This code was adapted to python from the Octave splines package created by 
N.Y. Krakauer (see Refs. below), with algorithmic modifications, mainly to gain performance.

##### Related projects and publications:
+ Krakauer, N. Y. & Fekete, B. M. [Are climate model simulations useful for forecasting precipitation trends? hindcast and synthetic-data experiments](http://dx.doi.org/10.1088/1748-9326/9/2/024009). Environ. Res. Lett. 9, 024009 (2014).
+ Krakauer's original octave splines code http://octave.sourceforge.net/splines/
+ E. Afik and V. Steinberg. [Pair dispersion in a chaotic flow reveals the role of the memory of
initial velocity](http://arxiv.org/abs/1502.02818). _ArXiv e-prints arXiv:1502.02818_. submitted.
+ E. Afik. [Robust and highly performant ring detection algorithm for 3d particle tracking using 2d microscope imaging](http://www.nature.com/articles/srep13584). Sci. Rep. 5, 13584; doi: 10.1038/srep13584 (2015)
