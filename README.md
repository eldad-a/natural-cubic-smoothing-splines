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
+ E. Afik and V. Steinberg. [On the role of initial velocities in pair dispersion in a microfluidic chaotic flow](https://www.nature.com/articles/s41467-017-00389-8). _Nature Communications_ __8__, Article number: 468 (2017) [doi: 10.1038/s41467-017-00389-8](http://dx.doi.org/10.1038/s41467-017-00389-8).
+ E. Afik and V. Steinberg. A Lagrangian approach to elastic turbulence in a curvilinear microfluidic channel. figshare [doi: 10.6084/m9.figshare.5112991](http://dx.doi.org/10.6084/m9.figshare.5112991) (2017).
+ E. Afik. [Robust and highly performant ring detection algorithm for 3d particle tracking using 2d microscope imaging](http://www.nature.com/articles/srep13584). Sci. Rep. 5, 13584; doi: 10.1038/srep13584 (2015)
