#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-12-22 16:00
Last Edited on 2020-12-22 16:00
@author: adtzlr
                           .__  __  .__  _____ 
_____  ______   ___________|__|/  |_|__|/ ____\
\__  \ \____ \_/ __ \_  __ \  \   __\  \   __\ 
 / __ \|  |_> >  ___/|  | \/  ||  | |  ||  |   
(____  /   __/ \___  >__|  |__||__| |__||__|   
     \/|__|        \/                          
APERITIF - Nonlinear Finite Elements Code for Structural Mechanics
Copyright (C) 2020  Dutzler A.
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
THIS FILE IS WORK-IN-PROGRESS!!!
"""

from autograd import numpy as np
from autograd.numpy.linalg import det
from autograd.numpy.linalg import eigh
from autograd import jacobian as dF
from autograd import jacobian as dJ

def udamage(ψ, material, statevars_old, full_output=True):
    r'''Isotropic strain energy density - based damage function.
    
    Parameters
    ----------
    ψ : float
        hyperelastic strain energy density functional
    material : dict
        dict containing material parameters
    statevars_old : array
        old state variables
    full_output : bool, optional
        flag for new state variables output (default is True)
        
    Returns
    -------
    η : float
        damage factor
    statevars : array, optional
        array of new state variables (only returned if optional input 
        full_output is True)
    
    Theory
    ------
    Ogden-Roxburgh mullins-softening:
        
    :math:`\eta=1-\frac{1}{r}\tanh\left(\frac{\psi_{max}-\psi}{m+\beta \psi_{max}}\right)`
    
    '''
    
    # extract material parameters from dict
    r, m, β = [material[key] for key in ('r','m','β')] # mullins
    
    # update max. strain energy density
    ψmax_old = statevars_old[0]
    ψmax = max(ψ, ψmax_old)
    
    # update state variables
    statevars = np.array([ψmax])
    
    # mullins-damage (ogden-roxburgh) isotropic evolution law
    η = 1-1/r*np.tanh((ψmax-ψ)/(m+β*ψmax))
    
    if full_output:
        return η, statevars
    else:
        return η

def uhyperstretch(λ, material, statevars_old, full_output=True):
    r'''Principal stretch based isotropic hyperelastic material model.
    
    Parameters
    ----------
    λ : array
        principal stretches
    material : dict
        dict containing material parameters
    statevars_old : array
        old state variables
    full_output : bool, optional
        flag for new state variables output (default is True)
        
    Returns
    -------
    ψ : float
        strain energy density per unit undeformed volume
    statevars : array, optional
        array of new state variables (only returned if optional input 
        full_output is True)
    
    Theory
    ------
    (k)-generalized neo-hookean hyperelastic material model:
        
    :math:`\psi = \frac{\mu}{k} (\lambda_1^k+\lambda_2^k+\lambda_3^k-3)`
    
    '''
    
    # extract material parameters from dict
    μ, k = [material[key] for key in ('μ','k')]
    
    # hyperelastic strain energy density
    ψ = μ/k*(np.sum(λ**k)-3)
    
    if full_output:
        return ψ, statevars_old
    else:
        return ψ
    
def uhypervol(J, material, statevars_old=None, full_output=True):
    r'''Volumetric isotropic strain energy density functional 
    per unit undeformed volume for hyperelastic materials.
    
    Parameters
    ----------
    J : array
        determinant of deformation gradient
    material : dict
        dict containing material parameters
    statevars_old : array
        old state variables
    full_output : bool, optional
        flag for new state variables output (default is True)
        
    Returns
    -------
    U : float
        volumetric part of isotropic strain energy density
    statevars : array, optional
        array of new state variables (only returned if optional input 
        full_output is True)
        
    Theory
    ------
    The deformation gradient is decomposed by the distortional-volumetric 
    split. The volumetric part of the deformation gradient is described by
    the volume ratio J.
    
    :math:`J=\det(\boldsymbol{F})`
    
    **Constitution**
    The volumetric part of an isotropic hyperelastic material model takes the
    determinant of the deformation gradient as input variable.
    
    :math:`U(J)=\frac{K}{2}(J-1)^2`
    '''
    
    K = material['K']
    
    U = K/2*(J-1)**2
    
    # output volumetric strain energy density
    if full_output:
        return U, statevars_old
    else:
        return U
    
def uhyperiso(F, material, statevars_old=None, full_output=True):
    r'''(Pseudo-elastic) isotropic strain energy density functional 
    per unit undeformed volume for hyperelastic materials with optional
    isotropic damage evolution.
    
    Parameters
    ----------
    F : array
        deformation gradient tensor
    material : dict
        dict containing material parameters
    statevars_old : array
        old state variables
    full_output : bool, optional
        flag for new state variables output (default is True)
        
    Returns
    -------
    η*ψ : float
        mullins-damaged strain energy density
    statevars : array, optional
        array of new state variables (only returned if optional input 
        full_output is True)
        
    Theory
    ------
    The deformation gradient is decomposed by the distortional-volumetric 
    split. This leads to the distortional part of the right cauchy-green
    deformation tensor.
    
    :math:`J=\det(\boldsymbol{F})`
    
    :math:`\boldsymbol{C}=\boldsymbol{F}^T \boldsymbol{F}`
    
    :math:`\hat{\boldsymbol{C}}=J^{-2/3} \boldsymbol{C}`
    
    **Constitution**
    The isotropic hyperelastic material model takes the principal stretches
    of the isochoric part of the deformation as input variables.
    
    :math:`\lambda_\alpha, N_\alpha = \text{eig}(\boldsymbol{C})`
    
    :math:`\hat{\psi} = \f(\hat{\lambda}_1,\hat{\lambda}_2,\hat{\lambda}_3)`
    
    The damage is calculated by an isotropic damage evolution law.
    
    :math:`\eta=f(\hat{\psi},\xi)`
    
    The "softened" or "damaged" strain energy density is the product of
    the damage variable multplied by the pure hyperelastic 
    strain energy density functional.
    
    :math:`\tilde{\psi} = \eta \hat{\psi}`
    '''
    
    # distortional part of right cauchy-green deformation tensor
    J = det(F)
    C = F.T@F
    
    λC, v = np.linalg.eigh(C)
    λiso = J**(-1/3)*np.sqrt(λC)
    
    # hyperelastic strain energy density
    ψ, statevars = uhyperstretch(λiso, material, statevars_old)
    
    if material['damage']:
        η, statevars = udamage(ψ, material, statevars)
    else:
        η = 1
    
    # output pseudo-elastic (damaged) strain energy density
    if full_output:
        return η*ψ, statevars
    else:
        return η*ψ  

if __name__ == '__main__':
    # test script for material database

    # material parameters
    mat = dict(μ=1, k=1.3, K=5000, 
               r=3, m=1, β=0, 
               damage=True)
    
    # (fixed) random deformation gradient
    np.random.seed(1056)
    F = np.random.rand(3,3)
    
    # isochoric strain energy density function and partial derivatives
    ψ = uhyperiso
    dψdF    =    dF(ψ)
    d2ψdFdF = dF(dF(ψ))
    
    # strain energy density
    ψ, sv = uhyperiso(F, statevars_old=[0], material=mat)
    
    # pk1-stress and elasticity tensor (automatic differentiation)
    P =    dψdF(F, statevars_old=sv, material=mat, full_output=False)
    A = d2ψdFdF(F, statevars_old=sv, material=mat, full_output=False)
    
    # cauchy stress and "total" elasticity tensor
    J = np.linalg.det(F)
    s = np.einsum('jJ,iJ',F,P)/J
    a = np.einsum('jJ,lL,iJkL',F,F,A)/J
    
    # volumetric strain energy density function
    # and partial derivatives
    U = uhypervol
    dUdJ = dJ(U)
    d2UdJdJ = dJ(dJ(U))
    
    # hydrostatic stress and derivative
    U    = uhypervol(J, material=mat, full_output=False)
    p    =      dUdJ(J, material=mat, full_output=False)
    dpdJ =   d2UdJdJ(J, material=mat, full_output=False)