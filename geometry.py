#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-10-26 20:00
Last Edited on 2020-10-26 20:00

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
"""

import numpy as np
    
def _length_line(nodes):
    # A B
    # 0 1
    
    p = nodes
    
    AB = p[1]-p[0]
    
    L = np.sqrt(np.dot(AB,AB))
    
    return L

def _area_quad(nodes):
    # A B C D
    # 0 1 2 3
    
    #permutation = [0,2,3,1]
    p = nodes#[permutation]
    
    DB = p[3]-p[1]
    AC = p[0]-p[2]
    
    S = 1/2*np.cross(DB,AC)
    
    return S

def _volume_hexa(nodes):
    # A B C D E F G H
    # 0 1 2 3 4 5 6 7
    
    #permutation = [0,4,6,2,1,5,7,3]
    p = nodes#[permutation]

    AC = (p[0]-p[2])
    AF = (p[0]-p[5])
    AH = (p[0]-p[7])
    BE = (p[1]-p[4])
    CH = (p[2]-p[7])
    DB = (p[3]-p[1])
    ED = (p[4]-p[3])
    FC = (p[5]-p[2])
    GA = (p[6]-p[0])
    GB = (p[6]-p[1])
    GC = (p[6]-p[2])
    GD = (p[6]-p[3])
    GE = (p[6]-p[4])
    GF = (p[6]-p[5])
    GH = (p[6]-p[7])
    HF = (p[7]-p[5])

    V12 = np.dot(GA,(np.cross(DB,AC)+np.cross(BE,AF)+np.cross(ED,AH))) \
        + np.dot(GB,(np.cross(DB,AC)+np.cross(GC,FC))) \
        + np.dot(GE,(np.cross(BE,AF)+np.cross(GF,HF))) \
        + np.dot(GD,(np.cross(ED,AH)+np.cross(GH,CH)))
        
    V = V12/12
    
    return V

def volume(nodes):
    dim = nodes.shape[1]
        
    lav = [None,
           _length_line, 
           _area_quad, 
           _volume_hexa]
    
    return lav[dim](nodes)