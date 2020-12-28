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
from numpy import tensordot as tdot

from types import SimpleNamespace

from aperitif import mathlib
from aperitif import geometry
from aperitif import constitution

def stiffness_force(u,x0,v0,kinematics,element,material):
    
    ndim = element.ndim

    # element-based volume ratio
    Jm = kinematics.Jm
    v = kinematics.v
    
    # TODO
    state_vars_old = None
    
    # volumetric strain energy density
    #U    = constitution.U(   Jm,material,full_output=False)
    p    = constitution.dUdJ(   Jm,material,full_output=False)
    dpdJ = constitution.d2UdJdJ(Jm,material,full_output=False)
    
    # pre-integrated helper functions (mean shape function derivative)
    M, MM  = kinematics.M, kinematics.MM

    a4p = kinematics.H4-np.einsum('ijkl->ilkj',kinematics.H4)

    # pre-integrated (hydrostatic) part of cauchy stress
    # and its contribution to the (nodal) internal forces
    s6 = np.tile(p*mathlib.I6,(element.npgauss,1))
    
    # internal force and stiffness (pre-evaluated parts)
    internal_force = p*v*M
    stiffness_p    = dpdJ*Jm*v*MM
    stiffness_d    = p*v*a4p

    for ip, (J,F,dhdx,Jr,w) in enumerate(zip(
            kinematics.J,
            kinematics.F,
            kinematics.dhdx,
            kinematics.Jr,
            element.gauss.weights)):

        ψ, state_vars = constitution.ψ(      F,material,state_vars_old)
        PDev          = constitution.dψdF(   F,material,state_vars,
                                             full_output=False)
        A4Dev         = constitution.d2ψdFdF(F,material,state_vars,
                                             full_output=False)

        sdev  = np.einsum('jJ,iJ->ij',F,PDev)/J
        a4dev = np.einsum('jJ,lL,iJkL->ijkl',F,F,A4Dev)/J
        
        s6[ip] += mathlib.tovoigt(sdev)
        
        internal_force += tdot(dhdx,sdev[:ndim,:ndim],1)*Jr*w
        
        #at4 = a4dev[:ndim,:ndim,:ndim,:ndim].transpose([1,0,2,3])
        #stiffness += tdot( tdot(dhdx,at4,1), dhdx.T, 1) * Jr*w
        
        stiffness_d += np.einsum('aj,ijkl,bl->aibk',
                               dhdx,a4dev[:ndim,:ndim,:ndim,:ndim],dhdx)*Jr*w

    return internal_force, stiffness_d, stiffness_p, s6 #.transpose([0,1,3,2])