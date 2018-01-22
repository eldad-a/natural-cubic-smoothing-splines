#! python

################################################################################
#  filename: splines.py
#  first online: https://github.com/eldad-a/natural-cubic-smoothing-splines
#  
#  Copyright (c) 2014, Eldad Afik
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  
#  * Neither the name of this software nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
################################################################################

"""
    # Smoothing Natural Cubic Spline - Algorithm
    `L. Wasserman (2004), All of Nonparametric Statistics`
    
    A module to smooth data by its least-squares natural cubic spline
    approximation.
    See Carl de Boor - A Practical Guide to Splines (Chp. XIV)
    
     References: 
    
     Larry Wasserman (2004), All of Nonparametric Statistics
    
     Vladimir Cherkassky and Filip Mulier (2007), Learning from Data: Concepts,
     Theory, and Methods. Wiley
    
     Carl de Boor (1978), A Practical Guide to Splines, Springer, Chapter XIV
    
     Clifford M. Hurvich, Jeffrey S. Simonoff, Chih-Ling Tsai (1998), Smoothing
     parameter selection in nonparametric regression using an improved Akaike
     information criterion, J. Royal Statistical Society, 60B:271-293
    
     M. F. Hutchinson and F. R. de Hoog (1985), Smoothing noisy data with
     spline functions, Numerische Mathematik, 47:99-106
    
     M. F. Hutchinson (1986), Algorithm 642: A fast procedure for calculating
     minimum cross-validation cubic smoothing splines, ACM Transactions on
     Mathematical Software, 12:150-153
    
     Grace Wahba (1983), Bayesian ``confidence intervals'' for the
     cross-validated smoothing spline, J Royal Statistical Society,
     45B:133-150
    
     Nir Krakauer <nkrakauer@ccny.cuny.edu>, octave-splines
    
    TODO:
        * convert a and u (rescaled second deriv) into ppform
        * get smoothing parameter automatically (based on GCV, Vapnik measure,
         ...)
        * consider other boundary conditions (other than natural, such as
        * constant
          acceleration).
        * Performance: which one is more efficient - sparse or dense operations
        * Performance: consider forking the w==None/1 case to remove D (and
         powers)
"""
from __future__ import print_function # Python 2 and 3
## dependency: conda install future
from past.builtins import xrange # Python 2 and 3: backward-compatible 
from future.utils import iteritems # Python 2 and 3

import numpy as np
from scipy import sparse, optimize
from scipy.sparse import linalg as spla
from scipy import linalg as la #import solve_banded, svd, inv, sqrtm
from scipy.linalg import LinAlgError
from scipy.interpolate import PPoly

import sys


class NaturalCubicSpline:
    """
    ## When the smoothing parameter is provided:
    
    `Carl de Boor (1978), A Practical Guide to Splines, Springer, Chapter XIV`
    
    Given noisy data y parametrised by x (with weights $w_i = 1/\epsilon _i^2 $),
    assuming some underlying $$ \qquad y_i = g(x_i) + \epsilon_i $$
    find the natural cubic spline $ f_p(x) $, minimizer of the functional:
                                                           
       $$ \qquad p \sum \\{ w_i | y_i - f_p(x_i) |^2 \\}  +  (1-p) \int f_p '' (x) dx $$
                                                           
    ** Some notations (following [`de Boor`]):**
    
    $N$ length of $x$
    $$ i = 0,...,N-1 $$
    $$ a_i = f_p(x_i) $$
    $$ c_i = f_p''(x_i)/2 $$
    $$ u = c / 3p $$
    $ S[f_p] $ weighted sum of squared errors (first term in the functional above)
    $$ \Delta x_{i} = x_{i+1}-x{i} $$
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
    $$ \qquad \\{ 6(1-p) Q^T D^2 Q + pR \\} u = Q^T y $$
    $$ \qquad a = y - 6(1-p) D^2 Q u $$
    $$ \qquad S[f_p] = (y-a)^T D^{-2} (y-a) =|| 6(1-p) D Q u ||^2_2 $$
    
    
    **The ppform (in terms of $u$ and $a$):**
    
    $$ f_p(x_i) = a_i $$
    $$ f_p'(x_i) = \Delta a_i / \Delta x_i - 3p \Delta x_i ( 2 u_i  + u_{i+1} ) $$
    $$ f_p''(x_i) = 6p u_i $$
    $$ f_p'''(x_i^+) = 6p \Delta u_i / \Delta x_i $$

    ## For automatic selection of the smoothing parameter:

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
    average squared error: $$ \qquad MASE = \frac{1}{N} E \\{ || g(x_i) -
    f_p(x_i) ||^2 \\} $$
    (as the case of GCV) or the expected Kullback-Leibler discrepency (as
    the case for the AIC).

    The smoothing parameter is the the minimizer of $$ \qquad \log \hat
    \sigma ^2 + \psi (H_p / N) $$
    where $$ \\qquad \hat \sigma ^2 = \frac{1}{N} \sum \\{ y_i - f_p(x_i)
    \\}^2 = \frac{1}{N} ||  (I-H_p)y ||^2 $$
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
    \frac{tr(H) +1}{N- tr(H)-2} \\qquad if \quad  tr(H)+2>N \quad
    \psi_{AIC_C}=\infty$$
    $$ \qquad \psi_{T}(H) = -\log [1 - 2 h] $$

    `Cherkassky and Mulier (2007) [p.129]`

    Vapnik-Chervonenkis penalization factor or Vapnik's measure:
    $$ \qquad \psi_{VM} = -\log \left[ 1 - \sqrt{ h-h \log h +\log
    N / 2N } \right] $$

    All require the estimation of $I-H_p$ and $Tr{H_p}$. Using de
    Boor's notation
    $$\qquad I - H_p = \frac{1}{6(1-p)} D^2 Q \\{ Q^T D^2 Q +
    \frac{p}{6(1-p)} R \\}^{-1} Q^T = \dots$$
    Following `Craven & Wahba (1978)`, denote $F = DQR^{-1/2}$ ,
    for $R^{-1/2}$ is the (symmetric?) square-root of the inverse
    of $R$. 
    Rewritte in terms of $F$: $$\\qquad \ldots = \frac{1}{6(1-p)}
    DF \\{F^T F +\frac{p}{6(1-p)} I \\}^{-1} F^T D^{-1} = \ldots$$
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

    """
    def __init__(self, x=None, y=None, w=None, p='GCV', LOG=True, 
                                        #boundary_condition='natural',
                                        return_spline=True):
        """
        generate the (sparse) matrices R,Q and D
        y are the (noisy) data points parameterised by x
        w are the weights (possibly inverse squared relative uncertainty?)
        p is the smoothing criterion (or paramater)
        """
        if x is not None: 
            self.__take_inputs__(x, y, w, p, LOG)
        else:
            print( 'No data were provided')
            return

        if return_spline:
            f0, f2 = self.compute_smoothing_spline()
            #boundary_condition=boundary_condition)
            self.coefs = self.compute_coefficients(f0,f2) # TODO: knots_coefs?
            #self.ppoly = self
            #return self.ppoly, self.p, self.sigma2, self.unc_y

    def __take_inputs__(self, x, y, w, p, LOG):
        #TODO: 
        #    * make sure that y and x are properly aligned
        #    * handle the case of non-increasing x
        N = len(x) # Number of data points
        x, y = np.atleast_1d(x,y) # in case y in a list of y1,...,yd
        if y.ndim>1 and y.shape[1]>y.shape[0]:
            y = np.transpose(y)         
        self.w = w
        assert y.shape[0]==len(x), 'shape of y %s is not in accord with x %s' %\
                (y.shape, x.shape)
        self.x,self.y = x,y
        dx = np.diff(x)
        self.dx = dx
        # verify that x is in increasing order:
        assert np.all(dx>=0), 'x is not in increasing order'
        if w==None:
            self.D = 1
            self.D2 = 1
        else:
            self.D = sparse.diags(1./np.sqrt(w), 0, shape=(N,N))
            self.D2 = sparse.diags(1./w, 0, shape=(N,N))
        self.R = sparse.diags(
                    [2*(dx[:-1]+dx[1:]), dx[1:-1], dx[1:-1]],
                    [0,1,-1], 
                    shape=(N-2,N-2)
                    )
        self.Q = sparse.diags(
                    [1./dx[:-1], np.r_[-1./dx[:-1]-1./dx[1:],0], 1./dx[1:]],
                    # in the previous line the second diag is appended with
                    # and extra term (which will not be used), otherwise an
                    # error is raised
                    [0,1,2],
                    shape=(N-2,N) 
                    ).T
        
        try:
            if p in ['GCV', 'AIC', 'AICC', 'VM']:
                self.crit = p
                self.LOG = LOG # True => log of penatly terms instead of ratio
                self.p = self.parameter_selection()
            elif 0<=p<=1:
                self.p = p
            elif p<0:
                # In case p is not provided, attempt equal weight for the LSQ and
                # the penalty (Nir Krakauer's choice for octave-splines):
                #              Tr { 6(1-p) Q.T D^2 Q } = Tr { pR }
                R,Q,D2 = self.R, self.Q, self.D2
                lamda = R.diagonal().sum() / (Q.T * D2 * Q).diagonal().sum() / 6. 
                self.p = 1. / (1+lamda)
                self.crit = 'equal_traces'
        except:
            print( "Unexpected error:", sys.exc_info()[0])
            print( 'p=%s is not supported' % p)
            raise

    def compute_smoothing_spline(self, boundary_condition='natural'):
        """
        """
        y, Q, R, D2, p = self.y, self.Q, self.R, self.D2, self.p
        ## TODO: need to provide a calculation for p = 0 or 1
        #assert p!=1 and p!=0, 'p=0 and p=1 are not implemented yet'
        # Solve for u: [ 6(1-p) Q.T D^2 Q + pR ] u = Q.T y
        Bp = 6*(1-p) * Q.T * D2 * Q + p*R
        b = Q.T*y
        # u = la.solve(Bp.todense(), b ) # non-sparse solver, slower
        # u = spla.spsolve(Bp, b ) # sparse solver, ~15x faster for 500
        # datapoints
        Bp = Bp.todia()
        # A is a banded matrix while b is non sparse, hence use solve_banded, in
        # principle could try solveh_banded as it is symmetric, but the help
        # says it is requires positive definite matrices.
        # u = la.solveh_banded(Bp.data[Bp.offsets >= 0][::-1], b) # shows larger
        # discrepancy with Nir's csaps.m
        # consider setting check_finite=Flase
        # solve banded shows results which are closest to Nir's csaps.m (in
        # terms of a and u)
        d = Bp.offsets[-1]
        u = la.solve_banded((d,d), Bp.data[::-1], b ) # needs inversion of order
        # get a = y - 6(1-p) D^2 Q u
        a = y - 6*(1-p)* D2 * Q * u
        # TODO: cosider leaving this step to the construction of coefs
        # natural boundary conditions: 
        if boundary_condition=='natural':
            if len(u.shape)==1:
                f2 = np.pad(6*p*u,1,'constant', constant_values=0 )
            elif len(u.shape)==2:
                f2 = np.pad(6*p*u, ((1,1),(0,0)), 'constant', 
                                                    constant_values=0)
        #elif boundary_condition=='zero_jerk':
        #    f2 = 6*p*np.r_[ u[0], u, u[-1] ]

        f0 = a
        return f0, f2

    def compute_coefficients(self,f0,f2):
        """
        constructs the coefficients array where
            0th axis is for the intervals (defined by the knots)
            1st axis is the spline order (currently only cubic)
            2nd axis is for dimensionality

        [the equations are available in de Boor Chap. XIV Eqns 9]
        """
        order = 3
        if f0.ndim>1:
            n_intervals, ndim = f0.shape
        else: 
            ndim = 1
            n_intervals = f0.shape[0]
            f0 = f0[:, np.newaxis]
            f2 = f2[:, np.newaxis]
        coefs = np.empty((n_intervals ,order+1, ndim))
        dx = self.dx[:,np.newaxis]
        # initialise the first and last intervals coefs to zero
        coefs[[0,-1], ...] = 0 
        coefs[:, -1] = f0
        coefs[:, -3] = f2
        coefs[:-1, -4] = np.diff(f2,axis=0) / dx
        coefs[:-1, -2] = np.diff(f0,axis=0) / dx - dx/3. * (f2[:-1] + f2[1:]/2.)
        # last first derivative calculated by the last interval derivative:
        # f'(x_N) = f'(x_N-1) + f''(x_N-1) dx_N-1 + 1/2 f'''(x_N-1) dx_N-1**2
        coefs[-1, -2] = coefs[-2,-2] + \
                           coefs[-2,-3]*dx[-1] + \
                           coefs[-2,-4]*dx[-1]**2
        return coefs

#    def piecewise_polynomial(self):
#        piecewise_pol = [SP.PiecewisePolynomial(A.x, coefs.T[:,::-1], orders=3,
#            direction=1, axis=0) for coefs in A.coefs.T]
#        pass

    def get_ppoly_from_coefs(self, return_ppoly=False):
        self.ppoly = []

        for coefs in self.coefs.T: 
            _ppoly = coefs[:,:-1].copy()
            _ppoly[0] /= 6
            _ppoly[1] /= 2
            self.ppoly.append(PPoly(_ppoly, self.x, extrapolate=False))
        
        if len(self.ppoly)==1:
            self.ppoly = self.ppoly[0]

        if return_ppoly:
            return self.ppoly

    def Lambda(self,p):
        _Lambda = lambda p: self.S**2/(self.S**2 + p/(1-p)/6)
        try:
            return _Lambda(p)
        except AttributeError:
            self.U,self.S = self.svd_eye_minus_hat_matrix()
            return _Lambda(p)

    def estimate_uncertainty(self, p):
        Lambda = self.Lambda
        data_infidelity, trI_H = self.get_penalty_terms(p)
        ## estimated noise variance [`wahba83`]
        sigma2 = np.mean(data_infidelity) / trI_H
        ## standard error estimates of fitted values [`hutchinson86`]
        w = self.w if self.w!=None else 1
        diagH = 1 - np.dot(self.U**2, Lambda(p))
        unc_y = np.sqrt(sigma2 * diagH  / w)
        return sigma2, unc_y

    def parameter_selection(self):
        """
        TODO: doc
        """
        crit = self.crit.lower()
        #penalty_function = lambda p: penalty_compute(p, self.U, self.S**2,
        #                                             self.y, self.w,
        #                                             len(self.x))
        penalty_function = lambda p: getattr(self, crit)(p, self.LOG)
        p = optimize.fminbound(penalty_function, 0, 1)
        return p

    def svd_eye_minus_hat_matrix(self):
        """
        Following Craven & Wahba find the singular value decomposition of 
            F = D Q R^{-1/2} 
        This function returns the non-zero singular values S and the
        corresponding left singular vectors in U, satisfying
            I - Hp = D U [ si**2 / ( 6(1-p) si**2 + p ) ] U.T D**(-1)
        where si is the ith singular value.
        """
        # TODO: is it indeed faster to take the non-sparse inverse?!
        method = 4
        if method==0:
            sqrt_invR = sqrtm(spla.inv(self.R).A)
        elif method==1:
            sqrt_invR = sqrtm(la.inv(self.R.todense()))
        elif method==2:
            invR = la.inv(self.R.todense())
            eR, oR = la.eigh(invR)
            sqrt_invR = oR.dot(np.diag(np.sqrt(eR))).dot(oR.T)
        elif method==3:
            eR, oR = la.eigh(self.R.todense())
            sqrt_invR = oR.dot(np.diag(1./np.sqrt(eR))).dot(oR.T)
        elif method==4:
            # TODO: 
            #     deal with the error
            #     File "splines.py", line 378, in svd_eye_minus_hat_matrix
            #     eR, oR = la.eig_banded(self.R.data[self.R.offsets>=0][::-1])
            #     File "/ph2users/eldada/lib/anaconda/lib/python2.7/site-packages/scipy/linalg/decomp.py",
            #     line 563, in eig_banded
            #     raise LinAlgError("eig algorithm did not converge")
            try:
                eR, oR = la.eig_banded(self.R.data[self.R.offsets>=0][::-1])
                sqrt_invR = oR.dot(np.diag(1./np.sqrt(eR))).dot(oR.T)
            except LinAlgError:
                # if eig_banded fails try the eigh
                eR, oR = la.eigh(self.R.todense())
                sqrt_invR = oR.dot(np.diag(1./np.sqrt(eR))).dot(oR.T)
        U, S, VT = la.svd(self.D * self.Q * sqrt_invR, full_matrices=False)
        return U,S
    
    def get_penalty_terms(self, p):
        """
        TODO: doc
        """
        Lambda = self.Lambda
        # TODO: verify that this is the correct way to introduce the weights!
        if self.w==None:
            L = Lambda(p)**2  
            z = (np.dot(self.U.T, self.y))**2 
            data_infidelity = L.dot(z)  
        else:
            ## || U Lambda U.T (sqrt(w)*y) ||**2
            sqrt_w = np.sqrt(self.w)
            L = Lambda(p)
            if self.y.ndim>1:
                sqrt_w = sqrt_w[:,np.newaxis]
                L = L[:,np.newaxis]
            y_weighted = sqrt_w * self.y
            z = L * self.U.T.dot(y_weighted) 
            data_infidelity = self.U.dot(z)
            data_infidelity = np.sum(data_infidelity**2,0)

        trace_eye_minus_hat = Lambda(p).sum()
        return data_infidelity, trace_eye_minus_hat
        
    def gcv(self, p, LOG=True):
        data_infidelity, trI_H = self.get_penalty_terms(p)
        if LOG:
            return np.mean(np.log(data_infidelity)) - \
                                                2*np.log(trI_H)
        else:
            return np.mean(data_infidelity) / trI_H**2

    def aic(self, p, LOG=True):
        data_infidelity, trI_H = self.get_penalty_terms(p)
        N = len(self.x)
        Y = 2*(N-trI_H)/N
        if LOG:
            return np.mean(np.log(data_infidelity)) + Y
                                                
        else:
            return np.mean(data_infidelity) * np.exp(Y)

    def aicc(self, p, LOG=True):
        data_infidelity, trI_H = self.get_penalty_terms(p)
        N = len(self.x)
        Y = (N-trI_H+1) / (trI_H/2 - 1) if trI_H/2 - 1 > 0 else np.inf
        if LOG:
            return np.mean(np.log(data_infidelity)) + Y
        else:
            return np.mean(data_infidelity) * np.exp(Y)

    def vm(self, p, LOG=True):
        """
        Vapnik-Chervonenkis penalization factor / Vapnik's measure
        """
        data_infidelity, trI_H = self.get_penalty_terms(p)
        N = len(self.x)
        h = (1.-trI_H/N)
        if h==0:
            Y = 1 - np.sqrt(np.log(N)/(2*N))
        else:
            Y = h - h*np.log(h) + np.log(N)/(2*N)
            if Y >= 1: return np.inf
            #otherwise:
            Y = 1 - np.sqrt(Y)
        if LOG:
            return np.mean(np.log(data_infidelity)) - np.log(Y)
        else:
            return np.mean(data_infidelity) / Y
       
def sqrtm(A, disp=True):
    """
    Symmetric Matrix square root.

    modified version of the scipy.linalg sqrtm function for performance:
    (i) introduced a dot product [based on https://groups.google.com/forum/#!topic/scipy-user/iNzZzkHjlgA]
    (ii) avoid rsf2csf as the input is expected to be symmetric

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose square root to evaluate
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    sgrtm : (N, N) ndarray
        Value of the sign function at `A`

    errest : float
        (if disp == False)

        Frobenius norm of the estimated error, ||err||_F / ||A||_F

    Notes
    -----
    Uses algorithm by Nicholas J. Higham

    """
    A = np.asarray(A)
    if len(A.shape)!=2:
        raise ValueError("Non-matrix input to matrix function.")
    T, Z = la.schur(A)
    # if the matrix is real and symmetric can skip the complex part
    if not (np.allclose(A,A.T,rtol=0,atol=1e-8) and np.all(A.imag==0)):
        T, Z = la.rsf2csf(T,Z)
    n,n = T.shape

    R = np.zeros((n,n),T.dtype.char)
    for j in xrange(n):
        R[j,j] = np.sqrt(T[j,j])
        for i in xrange(j-1,-1,-1):
            #s = 0
            #for k in range(i+1,j):
            #    s = s + R[i,k]*R[k,j]
            s = R[i,(i+1):j].dot(R[(i+1):j,j])
            R[i,j] = (T[i,j] - s)/(R[i,i] + R[j,j])

    R, Z = la.all_mat(R,Z)
    X = (Z * R * Z.H)

    if disp:
        nzeig = np.any(np.diag(T)==0)
        if nzeig:
            print("Matrix is singular and may not have a square root.")
        return X.A
    else:
        arg2 = la.norm(X*X - A,'fro')**2 / la.norm(A,'fro')
        return X.A, arg2

### Translations from Nir

def gcv(MSR, Ht, N):
    return np.mean(np.log(MSR)) - 2 * np.log(1 - Ht / N)

def influence_matrix(p, U, S2, N, w): #returns influence matrix for given p
    H = np.eye(N,N) - U.dot( np.diag(S2 / (S2 + (p / (6*(1-p))))).dot(U.T))
    #H = diag(1.  / sqrt(w)).dot(H.dot(diag(sqrt(w)))); #rescale to original units	
    return H	

def penalty_terms(H, y, w):
    #MSR = mean(w * (y - (H.dot(y)))**2); #mean square residual
    MSR = np.mean( (y - (H.dot(y)))**2); # not weighted mean square residual
    Ht = np.trace(H); #effective number of fitted parameters
    return MSR, Ht

def penalty_compute(p, U, S2, y, w, N): #evaluates a user-supplied penalty function crit at given p
  H = influence_matrix(p, U, S2, N, w);
  MSR, Ht = penalty_terms(H, y, w);
  J = gcv(MSR, Ht, N);
  #if ~isfinite(J)
  #  J = Inf;
  return J

##########################################################################
#
### @deftypefn{Function File}{[@var{yi} @var{p} @var{sigma2},@var{unc_y}] =} csaps_sel(@var{x}, @var{y}, @var{xi}, @var{w}=[], @var{crit}=[])
### @deftypefnx{Function File}{[@var{pp} @var{p} @var{sigma2},@var{unc_y}] =} csaps_sel(@var{x}, @var{y}, [], @var{w}=[], @var{crit}=[])
###
### Cubic spline approximation with smoothing parameter estimation @*
### Approximates [@var{x},@var{y}], weighted by @var{w} (inverse variance; if not given, equal weighting is assumed), at @var{xi}.
###
### The chosen cubic spline with natural boundary conditions @var{pp}(@var{x}) minimizes @var{p} Sum_i @var{w}_i*(@var{y}_i - @var{pp}(@var{x}_i))^2  +  (1-@var{p}) Int @var{pp}''(@var{x}) d@var{x}.
###
### A selection criterion @var{crit} is used to find a suitable value for @var{p} (between 0 and 1); possible values for @var{crit} are  `vm' (Vapnik's measure [Cherkassky and Mulier 2007] from statistical learning theory); `aicc' (corrected Akaike information criterion, the default); `aic' (original Akaike information criterion); `gcv' (generalized cross validation). If @var{crit} is a scalar instead of a string, then @var{p} is chosen to so that the mean square scaled residual Mean_i (@var{w}_i*(@var{y}_i - @var{pp}(@var{x}_i))^2) is approximately equal to @var{crit}.
###
### @var{x} and @var{w} should be @var{n} by 1 in size; @var{y} should be @var{n} by @var{m}; @var{xi} should be @var{k} by 1; the values in @var{x} should be distinct and in ascending order; the values in @var{w} should be nonzero.
###
### Returns the selected @var{p}, the estimated data scatter (variance from the smooth trend) @var{sigma2}, and the estimated uncertainty (SD) of the smoothing spline fit at each @var{x} value, @var{unc_y}.
###
### For small @var{n}, the optimization uses singular value decomposition of an @var{n} by @var{n} matrix  in order to quickly compute the residual size and model degrees of freedom for many @var{p} values for the optimization (Craven and Wahba 1979). For large @var{n} (currently >300), an asymptotically more computation and storage efficient method that takes advantage of the sparsity of the problem's coefficient matrices is used (Hutchinson and de Hoog 1985).
###
### References: 
###
### Vladimir Cherkassky and Filip Mulier (2007), Learning from Data: Concepts, Theory, and Methods. Wiley
###
### Carl de Boor (1978), A Practical Guide to Splines, Springer, Chapter XIV
###
### Clifford M. Hurvich, Jeffrey S. Simonoff, Chih-Ling Tsai (1998), Smoothing parameter selection in nonparametric regression using an improved Akaike information criterion, J. Royal Statistical Society, 60B:271-293
###
### M. F. Hutchinson and F. R. de Hoog (1985), Smoothing noisy data with spline functions, Numerische Mathematik, 47:99-106
###
### M. F. Hutchinson (1986), Algorithm 642: A fast procedure for calculating minimum cross-validation cubic smoothing splines, ACM Transactions on Mathematical Software, 12:150-153
###
### Grace Wahba (1983), Bayesian ``confidence intervals'' for the cross-validated smoothing spline, J Royal Statistical Society, 45B:133-150
###
### @end deftypefn
### @seealso{csaps, spline, csapi, ppval, dedup, gcvspl}
#
### Author: Nir Krakauer <nkrakauer@ccny.cuny.edu>
#
#function [ret,p,sigma2,unc_y]=csaps_sel(x,y,xi,w,crit)
#
#  if (nargin < 5)
#    crit = [];
#    if(nargin < 4)
#      w = [];
#      if(nargin < 3)
#        xi = [];
#      endif
#    endif
#  endif
#
#  if(columns(x) > 1)
#    x = x';
#    y = y';
#    w = w';
#  endif
#
#  if any (isnan ([x y w](:)) )
#    error('NaN values in inputs; use dedup to remove them')
#  endif
#
#  h = diff(x);
#  if any(h <= 0)
#	  error('x must be strictly increasing; use dedup to achieve this')
#  endif
#
#  n = numel(x);
#  
#  if isempty(w)
#    w = ones(n, 1);
#  end
#
#  if isscalar(crit)
#    if crit <= 0 #return an exact cubic spline interpolation
#        [ret,p]=csaps(x,y,1,xi,w);
#        sigma2 = 0; unc_y = zeros(size(x));
#        return
#      end
#    w = w ./ crit; #adjust the sample weights so that the target mean square scaled residual is 1
#    crit = 'msr_bound';
#  end	
#
#  if isempty(crit)
#    crit = 'aicc';
#  end
#
#  #R = spdiags([h(1:end-1) 2*(h(1:end-1) + h(2:end)) h(2:end)], [-1 0 1], n-2, n-2);
#  R = spdiags([h(2:end) 2*(h(1:end-1) + h(2:end)) h(1:end-1)], [-1 0 1], n-2, n-2);
#
#  QT = spdiags([1 ./ h(1:end-1) -(1 ./ h(1:end-1) + 1 ./ h(2:end)) 1 ./ h(2:end)], [0 1 2], n-2, n);
#
#
#chol_method = (n > 300); #use a sparse Cholesky decomposition followed by solving for only the central bands of the inverse to solve for large n (faster), and singular value decomposition for small n (less prone to numerical error if data values are spaced close together)
#
###choose p by minimizing the penalty function
#  
#if chol_method
#  penalty_function = @(p) penalty_compute_chol(p, QT, R, y, w, n, crit);
#else
#  ##determine influence matrix for different p without repeated inversion
#  [U D V] = svd(diag(1 ./ sqrt(w))*QT'*sqrtm(inv(R)), 0); D = diag(D).^2;
#  penalty_function = @(p) penalty_compute(p, U, D, y, w, n, crit);
#end
#
#  p = fminbnd(penalty_function, 0, 1);
#
### estimate the trend uncertainty
#if chol_method
#  [MSR, Ht] = penalty_terms_chol(p, QT, R, y, w, n);
#  Hd = influence_matrix_diag_chol(p, QT, R, y, w, n);
#else
#  H = influence_matrix(p, U, D, n, w);
#  [MSR, Ht] = penalty_terms(H, y, w);
#  Hd = diag(H);
#end
#
#  sigma2 = mean(MSR(:)) * (n / (n-Ht)); #estimated data error variance (wahba83)
#  unc_y = sqrt(sigma2 * Hd ./ w); #uncertainty (SD) of fitted curve at each input x-value (hutchinson86)
#
### construct the fitted smoothing spline 
#  [ret,p]=csaps(x,y,p,xi,w);
#
#endfunction
#
#
#
#function H = influence_matrix(p, U, D, n, w) #returns influence matrix for given p
#  H = speye(n) - U * diag(D ./ (D + (p / (6*(1-p))))) * U';
#  H = diag(1 ./ sqrt(w)) * H * diag(sqrt(w)); #rescale to original units	
def influence_matrix(p, U, S2, N, w):
    '''
    returns the Hat (or influence) matrix for a given smoothing parameter p
    '''
    tmp = sparse.eye(N,N) - U.dot( S2 / (S2 + p/(1-p)/6.)).dot(U.T)
    D = sparse.diags(1./np.sqrt(w), 0, shape=(N,N))
    invD = sparse.diags(np.sqrt(w), 0, shape=(N,N))
    return D * tmp * invD

def influence_matrix_diag_chol(p, Q, R, y, w, N):
    ## Following Hutchinson & de Hoog 1985:
    #LDL factorization of Bp = 6*(1-p)*QT*D2*QT' + p*R
    D2 = sparse.diags(1./np.sqrt(w), 0, shape=(N,N))
    Bp = 6*(1-p) * Q.T*D2*Q + p*R
    Bp = Bp.todia()
    u = la.cholesky_banded(Bp.data[Bp.offsets >= 0][::-1])
    U = sparse.dia_matrix((u, Bp.offsets[Bp.offsets >= 0][::-1]), Bp.shape)
    ## may fail as the input is sparse (if so use todense), need testing
    #U = la.cholesky(A , lower=False)
    ## To get from ut u -> Ut D U:
    ## U you are looking for is the above matrix with diagonal elements
    ## replaced by 1, and D is the diagonal matrix whose diagonal elements are 
    ## the squares of the diagonal elements in the above matrix.
    d = 1./u[-1]
    U = sparse.diags(d, 0, shape=U.shape)*U
    d = d**2
    ## TODO: this should probably change...
    #5 central bands in the inverse of 6*(1-p)*QT*diag(1 ./ w)*QT' + p*R
    Binv = banded_matrix_inverse(d, U, 2)
    Hd = np.diag( sparse.eye(N,N) - (6*(1-p))*D2*Q*Binv*Q.T )
    return Hd
#endfunction	
#
#function [MSR, Ht] = penalty_terms(H, y, w)
#  MSR = mean(w .* (y - (H*y)) .^ 2); #mean square residual
#  Ht = trace(H); #effective number of fitted parameters
#endfunction
#
#function Hd = influence_matrix_diag_chol(p, QT, R, y, w, n)
#  #LDL factorization of 6*(1-p)*QT*diag(1 ./ w)*QT' + p*R
#  U = chol(6*(1-p)*QT*diag(1 ./ w)*QT' + p*R, 'upper');
#  d = 1 ./ diag(U);
#  U = diag(d)*U; 
#  d = d .^ 2;
#  #5 central bands in the inverse of 6*(1-p)*QT*diag(1 ./ w)*QT' + p*R
#  Binv = banded_matrix_inverse(d, U, 2);
#  Hd = diag(speye(n) - (6*(1-p))*diag(1 ./ w)*QT'*Binv*QT);	
#endfunction
#
#function [MSR, Ht] = penalty_terms_chol(p, QT, R, y, w, n)
#  debug_on_error(true, "local")
#  #LDL factorization of 6*(1-p)*QT*diag(1 ./ w)*QT' + p*R
#  U = chol(6*(1-p)*QT*diag(1 ./ w)*QT' + p*R, 'upper');
#  d = 1 ./ diag(U);
#  U = diag(d)*U; 
#  d = d .^ 2;
#  Binv = banded_matrix_inverse(d, U, 2); #5 central bands in the inverse of 6*(1-p)*QT*diag(1 ./ w)*QT' + p*R
#  Ht = 2 + trace(speye(n-2) - (6*(1-p))*QT*diag(1 ./ w)*QT'*Binv);
#  MSR = mean(w .* ((6*(1-p)*diag(1 ./ w)*QT'*((6*(1-p)*QT*diag(1 ./ w)*QT' + p*R) \ (QT*y)))) .^ 2);
#endfunction
#
#function J = vm(MSR, Ht, n)
##Vapnik-Chervonenkis penalization factor or Vapnik's measure in cherkassky07, p. 129
#  p = Ht/n;
#  if p == 0
#    J = mean(log(MSR)(:)) - log(1 - sqrt(log(n)/(2*n)));
#  elseif n == 0 || (p*(1 - log(p)) + log(n)/(2*n)) >= 1
#    J = Inf;
#  else
#    J = mean(log(MSR)(:)) - log(1 - sqrt(p*(1 - log(p)) + log(n)/(2*n)));
#  endif
#endfunction
#
#function J = aicc(MSR, Ht, n)
#  J = mean(log(MSR)(:)) + 2 * (Ht + 1) / max(n - Ht - 2, 0); #hurvich98, taking the average if there are multiple data sets as in woltring86 
#endfunction
#
#function J = aic(MSR, Ht, n)
#  J = mean(log(MSR)(:)) + 2 * Ht / n;
#endfunction
#
#function J = gcv(MSR, Ht, n)
#  J = mean(log(MSR)(:)) - 2 * log(1 - Ht / n);
#endfunction
#
#function J = msr_bound(MSR, Ht, n)
#  J = mean(MSR(:) - 1) .^ 2;
#endfunction
#
#function J = penalty_compute(p, U, D, y, w, n, crit) #evaluates a user-supplied penalty function crit at given p
#  H = influence_matrix(p, U, D, n, w);
#  [MSR, Ht] = penalty_terms(H, y, w);
#  J = feval(crit, MSR, Ht, n);
#  if ~isfinite(J)
#    J = Inf;
#  endif
#endfunction
#
#function J = penalty_compute_chol(p, QT, R, y, w, n, crit) #evaluates a user-supplied penalty function crit at given p
#  [MSR, Ht] = penalty_terms_chol(p, QT, R, y, w, n);
#  J = feval(crit, MSR, Ht, n);
#  if ~isfinite(J)
#    J = Inf;
#  endif
#endfunction
#
#function Binv = banded_matrix_inverse(d, U, m) #given a (2m+1)-banded, symmetric n x n matrix B = U'*inv(diag(d))*U, where U is unit upper triangular with bandwidth (m+1), returns Binv, a sparse symmetric matrix containing the central 2m+1 bands of the inverse of B
##Reference: Hutchinson and de Hoog 1985
#  Binv = sparse(diag(d));
#  n = rows(U);
#  for i = n:(-1):1
#    p = min(m, n - i);
#    for l = 1:p
#      for k = 1:p
#        Binv(i, i+l) -= U(i, i+k)*Binv(i + k, i + l);
#      end
#      Binv(i, i) -= U(i, i+l)*Binv(i, i+l);
#    end
#    Binv(i+(1:p), i) = Binv(i, i+(1:p))'; #add the lower triangular elements
#  end
#endfunction
#
#%!shared x,y,ret,p,sigma2,unc_y
#%! x = [0:0.01:1]'; y = sin(x);
#%! [ret,p,sigma2,unc_y] = csaps_sel(x,y,x);
#%!assert (1 - p, 0, 1E-6);
#%!assert (sigma2, 0, 1E-10);
#%!assert (ret - y, zeros(size(y)), 1E-4);
#%!assert (unc_y, zeros(size(unc_y)), 1E-5);
#
#%{
##experiments comparing different selection criteria for recovering a function sampled with standard normal noise -- aicc was consistently better than aic, but otherwise which method does best is problem-specific
#m = 1; #number of replicates available
#ni = 200; #number of evaluation points
#ns = [5 10 20 40]; #number of given sample points
#nk = 100; #number of trials to average over
#f = @(x) sin(2*pi*x); #function generating the synthetic data
#mse = nan(4, numel(ns), nk);
#for i = 1:numel(ns)
#  for k = 1:nk
#    n = ns(i);
#    x = linspace(0, 1, n)(:);
#    y = f(x) + randn(n, m);
#    xi = rand(ni, 1);
#    yt = f(xi);
#    yi = csaps_sel(x,y,xi,[],'vm');
#    mse(1, i, k) = meansq((yi - yt')(:));
#    yi = csaps_sel(x,y,xi,[],'aicc');
#    mse(2, i, k) = meansq((yi - yt')(:));
#    yi = csaps_sel(x,y,xi,[],'aic');
#    mse(3, i, k) = meansq((yi - yt')(:)); 
#    yi = csaps_sel(x,y,xi,[],'gcv');
#    mse(4, i, k) = meansq((yi - yt')(:));   
#  endfor
#endfor
#msem = mean(mse, 3);
#%}
#
