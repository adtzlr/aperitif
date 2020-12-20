#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-12-20 22:00
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

def identity_tensors(ndim):
    '''Helper Function for the generation of 
    identity and deviatoric projection tensors.'''
    I   = np.eye(ndim)
    II  = np.tensordot(I,I,0)
    IIt = (II.transpose([0,2,1,3]) + II.transpose([0,3,2,1]))/2
    P   = IIt-1/ndim*II
    return I, II, IIt, P

I   = {}
II  = {}
IIt = {}
P   = {}
I[2], II[2], IIt[2], P[2] = identity_tensors(ndim=2)
I[3], II[3], IIt[3], P[3] = identity_tensors(ndim=3)


def deviator(A):
    ndim = len(A)
    #ndim = 3
    return A-np.trace(A)/ndim*np.eye(ndim)

def tovoigt(A,scaleshear=1):
    ndim = len(A)
    B = np.zeros((ndim*(1+ndim))//2)
    
    for i in range(ndim):
        B[i] = A[i,i]
        for j in range(ndim):
            if j > i:
                B[ndim-1+i+j] = scaleshear*A[i,j]
    return B