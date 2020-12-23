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


def stiffness_force(u,x0,v0,kinematics,element,material):
    
    ndim = element.ndim

    # identity tensors and deviatoric projection tensor
    I = np.eye(3)
    
    # element-based volume ratio
    Jm = kinematics.Jm
    
    # TODO
    state_vars_old = None
    
    #sigma          = np.zeros((element.npgauss,6))
    internal_force = np.zeros((element.nnodes,element.ndim))
    
    # volumetric strain energy density
    U    = constitution.U(   Jm,material,full_output=False)
    p    = constitution.dUdJ(Jm,material,full_output=False)
    dpdJ = constitution.dUdJ(Jm,material,full_output=False)
    
    # pre-integrated part of element stiffness matrix
    M = kinematics.M
    stiffness  = 2*dpdJ*v0*np.tensordot(M,M.T,0)
    
    II  = np.einsum('ij,kl',I,I)
    IIt = np.einsum('il,jk',I,I)
    
    stress = []

    for ip, (J,F,dhdx,Jr,w) in enumerate(zip(
            kinematics.J,
            kinematics.F,
            kinematics.dhdx,
            kinematics.Jr,
            element.gauss.weights)):
        
        ψ, state_vars = constitution.ψ(      F,material,state_vars_old)
        P             = constitution.dψdF(   F,material,state_vars,full_output=False)
        A4            = constitution.d2ψdFdF(F,material,state_vars,full_output=False)
        
        sdev  = np.einsum('jJ,iJ',F,P)/J
        a4dev = np.einsum('jJ,lL,iJkL',F,F,A4)/J
        
        s = sdev+p*I
        stress.append(mathlib.tovoigt(s))
        
        a4 = a4dev+p*(II-IIt)
        
        internal_force += tdot(dhdx,s[:ndim,:ndim],1)*Jr*w
        
        at4 = a4[:ndim,:ndim,:ndim,:ndim].transpose([1,0,2,3])
        stiffness += tdot( tdot(dhdx,at4,1), dhdx.T, 1) * Jr*w
        
    return internal_force, stiffness.transpose([0,1,3,2]), np.array(stress)