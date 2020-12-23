#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-10-27 21:00
Last Edited on 2020-12-20 22:00
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

import numpy as np
from numpy import trace as tr
from numpy import tensordot as tdot

import mathlib
from mathlib import deviator as dev
import geometry

import constitution

from types import SimpleNamespace

from numpy.linalg import det,inv

def kinematics(x0,u,v0,element):
    "Create kinematic quantities with displacements."
    
    # Element dimension and rank 2 identity tensor
    ndim = element.ndim
    I    = mathlib.I[ndim]

    # shape function derivative and gradients at int.points
    # get displacemnt gradient w.r.t. deformed coordinates
    dhdr = element.shape.dhdr
    dxdr = (x0+u).T@dhdr
    drdx = np.array([inv(dxdr_ip) for dxdr_ip in dxdr])
    dhdx = dhdr@drdx
    dudx = u.T@dhdx
    
    def extend(F):
        'Extend deformation gradient to shape (3,3).'
        F3D = np.eye(3)
        n,m = F.shape
        F3D[:n,:m] = F
        return F3D
    
    # init kinematics struct and evalutate jacobian Jr, 
    # deformation gradient F as well as J=det(F) at integration points
    # F has dimensions (n,n)!
    kin      = SimpleNamespace()
    kin.dhdx = dhdr@drdx
    kin.Jr   = np.array([det(dxdr_ip) for dxdr_ip in dxdr])
    kin.F    = np.array([inv(extend(I-dudx_ip)) for dudx_ip in dudx])
    kin.J    = np.array([det(F) for F in kin.F])
    
    # get element volume ratio
    kin.v = geometry.volume(x0+u)
    kin.Jm = kin.v/v0
    
    # element-based mean shape function derivative
    kin.M = 1/v0 * np.sum(
            np.array([dhdx*Jr*w
                      for dhdx,Jr,w in zip(kin.dhdx,
                                           kin.Jr,
                                           element.gauss.weights)]),
                          axis=0)

    return kin

def stiffness_force(u,x0,v0,kinematics,element,material):
    
    ndim = element.ndim

    # identity tensors and deviatoric projection tensor
    I   = mathlib.I[3]
    II  = mathlib.II[3]
    IIt = mathlib.IIt[3]
    p4  = mathlib.P[3]
    
    # element-based volume ratio
    Jm = kinematics.Jm
    
    # TODO
    state_vars_old = None
    
    #sigma          = np.zeros((element.npgauss,6))
    internal_force = np.zeros((element.nnodes,element.ndim))
    
    # Evaluate pre-integrated part of Element Stiffness matrix
    M = kinematics.M
    stiffness  = 2*dpdJ*np.tensordot(M,M.T,0)
                           
    U    = constitution.U(   Jm,material,full_output=False)
    p    = constitution.dUdJ(Jm,material,full_output=False)
    dpdJ = constitution.dUdJ(Jm,material,full_output=False)
    
    II  = np.einsum('ij,kl',I,I)
    IIt = np.einsum('il,jk',I,I)
    
    s_ = []

    for ip, (J,F,dhdx,Jr,w) in enumerate(zip(
            kinematics.J,
            kinematics.F,
            kinematics.dhdx,
            kinematics.Jr,
            element.gauss.weights)):
        
        ψ, state_vars = constitution.ψ(      F,material,state_vars_old)
        P             = constitution.dψdF(   F,material,state_vars,full_output=False)
        A             = constitution.d2ψdFdF(F,material,state_vars,full_output=False)
        
        sdev = np.einsum('jJ,iJ',F,P)/J
        adev = np.einsum('jJ,lL,iJkL',F,F,A)/J
        
        s = sdev+p*I
        s_.append(mathlib.tovoigt(s))
        
        a = adev+p*(II-IIt)
        
        internal_force += tdot(dhdx,s[:ndim,:ndim],1)*Jr*w
        
        at = a[:ndim,:ndim,:ndim,:ndim].transpose([1,0,2,3])
        stiffness += tdot( tdot(dhdx,at,1), dhdx.T, 1) * Jr*w
        
    stress = np.array([mathlib.tovoigt(s) for s in s_])
        
    return internal_force, stiffness.transpose([0,1,3,2]), stress