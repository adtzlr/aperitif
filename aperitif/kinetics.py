#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-10-27 21:00
Last Edited on 2020-10-27 21:00
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

from umats_modified import umat

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
    
    # init kinematics struct and evalutate jacobian Jr, 
    # deformation gradient F as well as J=det(F) at integration points
    # F has dimensions (n,n)!
    kin      = SimpleNamespace()
    kin.dhdx = dhdr@drdx
    kin.Jr   = np.array([det(dxdr_ip) for dxdr_ip in dxdr])
    kin.F    = np.array([inv(I-dudx_ip) for dudx_ip in dudx])
    kin.J    = np.array([det(F) for F in kin.F])
    
    if element.shape.unimodular:
        # total potential energy (internal force contribution)
        # Π_int = ∫ψ dV + ∫mP(J^m-θ) dV
        
        # evaluate isochoric/unimodular part of deformation gradient
        # naming convention is "unimodular part of deformation gradient": Fu
        # Fu always has dimensions (3,3) - even in 2D analysis
        def function_Fu(F,J,ndim):
            F3D = mathlib.I[3].copy()
            n,m = F.shape
            F3D[:n,:m] = F
            Fu = J**(-1/3)*F3D
            #Fu = abs(J)**(2/3)/J*F3D
            return Fu
            
        kin.Fu = np.array([function_Fu(F,J,ndim) for F,J in zip(kin.F,kin.J)])
        kin.bu = np.array([tdot(Fu,Fu.T,1) for Fu in kin.Fu])
        
        # exponent on volume ratio for interpolation
        kin.m = 1
        
        # get element volume ratio
        kin.v = geometry.volume(x0+u)
        kin.Jm = kin.v/v0
        
        # get modified deformation gradient
        kin.Fm = kin.Jm**(1/3)*kin.Fu
        kin.bm = kin.Jm**(2/3)*kin.bu
        
        # element-based mean shape function derivative
        kin.M = 1/v0 * np.sum(
                np.array([dhdx*Jr*w
                          for dhdx,Jr,w in zip(kin.dhdx,kin.Jr,
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
    
    Jm = kinematics.Jm
    
    # TODO
    state_vars = None
    
    sigma          = np.zeros((element.npgauss,3,3))
    internal_force = np.zeros((element.nnodes,element.ndim))
    
    # Evaluate pre-integrated part of Element Stiffness matrix
    stiffness  = np.zeros((element.nnodes,element.ndim,
                           element.ndim,  element.nnodes))
    
    Tm = []
    C4m = []
    for ip, (s,J,Fm,bm,Fu,bu,dhdx,Jr,w) in enumerate(zip(
            sigma,
            kinematics.J,
            kinematics.Fm,
            kinematics.bm,
            kinematics.Fu,
            kinematics.bu)):
        
        tm,c4m,state_vars = umat(Fm,bm,Fu,bu,state_vars,material)
        Tm.append(tm)
        C4m.append(c4m)
        
    p = 1/kinematics.v*np.array([tdot(tm,I,2)/3/J*Jr*w for tm,J,Jr,w in
                                 zip(Tm,
                                     kinematics.J,
                                     kinematics.Jr,
                                     element.gauss.weights)])

    # loop over integration points
    for ip, (s,J,Fm,bm,Fu,bu,dhdx,Jr,w) in enumerate(zip(
            sigma,
            kinematics.J,
            kinematics.Fm,
            kinematics.bm,
            kinematics.Fu,
            kinematics.bu,
            kinematics.dhdx,
            kinematics.Jr,
            element.gauss.weights)):
        
        # quantities at integration points
        # --------------------------------
        # tu  ... unimodular part of kirchhoff stress
        # c4u ... unimodular part of spatial rank-4 elasticity tensor
        # J   ... determinant of deformation gradient
        # Jr  ... determinant of jacobian deformed-->natural configuration
        # w   ... integration point weight
        
        # WORKAROUND
        if abs(np.trace(bm)-3) < 1e-3:
            fp = 1
        else:
            fp = 1#/material.K
        
        tm,c4m,state_vars = umat(Fm,bm,Fu,bu,state_vars,material)

        tr_tm = tdot(tm,I,2)

        s = dev(tm)/J+p*I

        sigma[ip] = s

        # internal force vector
        internal_force += tdot(dhdx,s[:ndim,:ndim],1)*Jr*w
        
        # helper tensor products of `tm` and `I`
        tI = tdot(tm,I,0)
        It = tI.transpose([2,3,0,1])
        tIt = (tI.transpose([0,2,1,3]) +
               tI.transpose([0,3,1,2]) )/2

        # deviatoric part of constitutive elasticity matrix
        c4_dev = 2/3 * (tr_tm*IIt - (tI+It) + tr_tm/3*II) / J
        c4_hyd = J/Jm * 2/3 * (It+tdot(I,tdot(I,c4m,2),0)+c4m-1/3*tI+tdot(c4m,II,4)*II)

        if np.any(c4m):
            # calculate only if necessary
            c4_dev += tdot( tdot(p4, c4m, 2), p4, 2)/J

        # geometric stiffness matrix
        c4_geo = (tIt.transpose([1,0,3,2])/2-fp*p*IIt/2
               + fp*p*(II-3/2*IIt))
                  
        # absolute elasticity matrix
        a4 = c4_dev + c4_hyd + c4_geo
        a4t = a4[:ndim,:ndim,:ndim,:ndim].transpose([1,0,2,3])
        stiffness += tdot( tdot(dhdx,a4t,1), dhdx.T, 1) * Jr*w# * fr
        
    stress = np.array([mathlib.tovoigt(s) for s in sigma])
        
    return internal_force, stiffness.transpose([0,1,3,2]), stress