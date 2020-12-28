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
from numpy.linalg import det,inv

from types import SimpleNamespace

from aperitif import geometry



def kinematics(x0,u,v0,element):
    "Create kinematic quantities with displacements."
    
    # Element dimension and rank 2 identity tensor
    ndim = element.ndim
    I    = np.eye(ndim)

    # shape function derivative and gradients at int.points
    # get displacemnt gradient w.r.t. deformed coordinates
    h    = element.shape.h
    dhdr = element.shape.dhdr
    dxdr = (x0+u).T@dhdr
    drdx = np.array([inv(dxdr_ip) for dxdr_ip in dxdr])
    dhdx = dhdr@drdx
    dudx = u.T@dhdx
    
    if element.axisymmetric.flag:
        # get radius ratio at integration points - coordinates (z,r,theta)
        R = np.array([x0_ip[1] for x0_ip in (x0  ).T@h])
        r = np.array([ x_ip[1] for  x_ip in (x0+u).T@h])
        Rr = R/r
    
    def extend(F,rR=1.0):
        'Extend deformation gradient to shape (3,3).'
        F3D = np.diag(np.array([1,1,rR]))
        n,m = F.shape
        F3D[:n,:m] = F
        return F3D
    
    # init kinematics struct and evalutate jacobian Jr, 
    # deformation gradient F as well as J=det(F) at integration points
    # F has dimensions (n,n)!
    kin      = SimpleNamespace()
    kin.dhdx = dhdr@drdx
    kin.Jr   = np.array([det(dxdr_ip) for dxdr_ip in dxdr])
    
    if element.axisymmetric.flag:
        kin.F = np.array([inv(extend(I-dudx_ip,Rr_ip))
                          for dudx_ip, Rr_ip in zip(dudx,Rr)])
    else:
        kin.F = np.array([inv(extend(I-dudx_ip,1.0)) 
                          for dudx_ip in dudx])
                          
    kin.J = np.array([det(F) for F in kin.F])

    # get element volume ratio
    kin.v = geometry.volume(x0+u)
    kin.Jm = kin.v/v0
    
    # element-based mean shape function derivative
    kin.M = 1/kin.v * np.sum(
            np.array([dhdx*Jr*w
                      for dhdx,Jr,w in zip(kin.dhdx,
                                           kin.Jr,
                                           element.gauss.weights)]),
                          axis=0)
    kin.MM = np.tensordot(kin.M,kin.M,0)
    
    kin.H4 = 1/kin.v * np.sum(
            np.array([np.tensordot(dhdx,dhdx,0)*Jr*w
                      for dhdx,Jr,w in zip(kin.dhdx,
                                           kin.Jr,
                                           element.gauss.weights)]),
                          axis=0)

    return kin