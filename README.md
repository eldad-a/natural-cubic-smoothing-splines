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

* * *

## Theoretical background

### When the smoothing parameter is provided:
    
`Carl de Boor (1978), A Practical Guide to Splines, Springer, Chapter XIV`

Given noisy data y parametrised by x (with weights $w_i = 1/\epsilon _i^2 $),
assuming some underlying $$ \qquad y_i = g(x_i) + \epsilon_i $$
find the natural cubic spline $ f_p(x) $, minimizer of the functional:

   $$ \qquad p \sum \{ w_i | y_i - f_p(x_i) |^2 \}  +  (1-p) \int f_p '' (x)^2 dx $$

** Some notations (following [`de Boor`]):**

$N$ length of $x$
$$ i = 0,...,N-1 $$
$$ a_i = f_p(x_i) $$
$$ c_i = f_p''(x_i)/2 $$
$$ u = c / 3p $$
$ S[f_p] $ weighted sum of squared errors (first term in the functional above)
$$ \Delta x_{i} = x_{i+1}-x_{i} $$
$R$ symmetric tridiagonal matrix of order $N-2$:
$$ \qquad \Delta x_{i-1} , 2(\Delta x_{i-1} + \Delta x_i) , \Delta x_i $$
$Q^T$ tridiagonal matrix of order $(N-2) \times N$:
$$ \qquad 1/\Delta x_{i-1} , -1/\Delta x_{i-1} - 1/\Delta x_i, 1/\Delta x_i $$
$D$ is the diagonal matrix of the uncertainties on y:
$$ \qquad D_{i,i} = 1/w_i $$

**The following relations result from the continuity of the first derivative and minimisation the regularised LSQ**

expressing $c$ in terms of the $a$ :  $$ \qquad R c = 3 Q^T a$$

It should be noted that for natural spline $ c_{0} = c_{N-1} = 0 $, 
hence in the above equation the first and last elements of $c$ are omitted.

$$ \Rightarrow $$
$$ \qquad \{ 6(1-p) Q^T D^2 Q + pR \} u = Q^T y $$
$$ \qquad a = y - 6(1-p) D^2 Q u $$
$$ \qquad S[f_p] = (y-a)^T D^{-2} (y-a) =|| 6(1-p) D Q u ||^2_2 $$


**The ppform (in terms of $u$ and $a$):**

$$ f_p(x_i) = a_i $$
$$ f_p'(x_i) = \Delta a_i / \Delta x_i - 3p \Delta x_i ( 2 u_i  + u_{i+1} ) $$
$$ f_p''(x_i) = 6p u_i $$
$$ f_p'''(x_i^+) = 6p \Delta u_i / \Delta x_i $$

### For automatic selection of the smoothing parameter:

`C.M. Hurvich, J.S. Simonoff, C-L Tsai (1998), Smoothing parameter
selection in nonparametric regression using an improved Akaike
information criterion, J. Royal Statistical Society, 60B:271-293`

`P. Craven and G. Wahba (1978), Smoothing noisy data with spline
functions, Numerische Mathematik, 31:377-403`

`M. F. Hutchinson and F. R. de Hoog (1985), Smoothing noisy data with
spline functions, Numerische Mathematik, 47:99-106`

`V. Cherkassky and F. Mulier (2007), Learning from Data: Concepts,
Theory, and Methods. Wiley`

`Nir Krakauer, octave-splines 1.2.4`

`L. Wasserman (2004), All of Nonparametric Statistics`

Following `Hurvich et al. (1998)`:
"Classical" methods to choose the smoothing parameter are based on the
minimisation of an approximately unbiased estimator of either the mean
average squared error: $$ \qquad MASE = \frac{1}{N} E \{ || g(x_i) -
f_p(x_i) ||^2 \} $$
(as the case of GCV) or the expected Kullback-Leibler discrepency (as
the case for the AIC).

The smoothing parameter is the the minimizer of $$ \qquad \log \hat
\sigma ^2 + \psi (H_p / N) $$
where $$ \qquad \hat \sigma ^2 = \frac{1}{N} \sum \{ y_i - f_p(x_i)
\}^2 = \frac{1}{N} ||  (I-H_p)y ||^2 $$
and $\psi$ is the penalty function designed to decrease with increasing
smoothness of $f_p$.

The Hat matrix $H_p$ (also referred to as the Smoothing matrix $L$
[`Wasserman (2004)`] 
or the Influence matrix $A_p$ [`Hutchinson & de Hoog (1985)`]),
is the matrix transforming the (noisy) data to the estimator:
$$ \qquad a = H_p y $$

Denote $h=tr(H_p)/N$; some possiblities for the penalty
function:
$$ \qquad \psi_{GCV}(H) = -2\log [1-h] $$
$$ \qquad \psi_{AIC}(H) = 2 h $$
$$ \qquad \psi_{AIC_C}(H) = \frac{1+h}{1-(h+2/N)} = 1 + 2
\frac{tr(H) +1}{N- tr(H)-2} 
 \qquad if \quad  tr(H)+2>N \quad
\psi_{AIC_C}=\infty$$
$$ \qquad \psi_{T}(H) = -\log [1 - 2 h] $$

`Cherkassky and Mulier (2007) [p.129]`

Vapnik-Chervonenkis penalization factor or Vapnik's measure:
$$ \qquad \psi_{VM} = -\log \left[ 1 - \sqrt{ h-h \log h +\log
N / 2N } \right] $$

All require the estimation of $I-H_p$ and $Tr{H_p}$. Using de
Boor's notation
$$\qquad I - H_p = \frac{1}{6(1-p)} D^2 Q \{ Q^T D^2 Q +
\frac{p}{6(1-p)} R \}^{-1} Q^T = \dots$$
Following `Craven & Wahba (1978)`, denote $F = DQR^{-1/2}$ ,
for $R^{-1/2}$ is the (symmetric?) square-root of the inverse
of $R$. 
Rewritte in terms of $F$: $$ \qquad \ldots = \frac{1}{6(1-p)}
DF \{F^T F +\frac{p}{6(1-p)} I \}^{-1} F^T D^{-1} = \ldots$$
Let the signular value decomposition of $F$ be $ F = U \Sigma
V^T $ with non-zero singluar values $\sigma_i$ and 
then: $$\qquad \ldots =  D U \left(
\frac{\sigma_i^2}{6(1-p)\sigma_i^2 + p} \right) U^T D^{-1}$$
and so finally `Craven & Wahba (1978)` suggest to choose by 
$$ \qquad \min_p \frac{\hat \sigma^2} { [Tr(I-H_p)/N]^2} =
\min_p N \frac{ || \Lambda U^T y ||^2}{\left(Tr(\Lambda)
\right)^2}$$
where $ \Lambda $ denotes the diagonal matrix with the elements
$\frac{\sigma_i^2}{6(1-p)\sigma_i^2 + p} $ on the diagonal.


