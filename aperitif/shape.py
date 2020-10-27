#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-10-27 10:00
Last Edited on 2020-10-27 10:00

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

def shape_p1_1d(rv):
    '''Return array of shape functions for 
    given natural coordinate vector "rv" in case of
    - 1D and
    - 1st oder interpolation.'''
    r = rv
    return np.array([(1-r), (1+r)])/2

def shape_p1_2d(rv):
    '''Return array of shape functions for 
    given natural coordinate vector "rv" in case of
    - 2D and
    - 1st oder interpolation.'''
    r,s = rv
    return np.array([(1-r)*(1-s),
                     (1+r)*(1-s),
                     (1+r)*(1+s),
                     (1-r)*(1+s)])/4
    
def shape_p1_3d(rv):
    '''Return array of shape functions for 
    given natural coordinate vector "rv" in case of
    - 3D and
    - 1st oder interpolation.'''
    r,s,t = rv
    return np.array([(1-r)*(1-s)*(1-t),
                     (1+r)*(1-s)*(1-t),
                     (1+r)*(1+s)*(1-t),
                     (1-r)*(1+s)*(1-t),
                     (1-r)*(1-s)*(1+t),
                     (1+r)*(1-s)*(1+t),
                     (1+r)*(1+s)*(1+t),
                     (1-r)*(1+s)*(1+t)])/8
    
def dshape_p1_1d(rv):
    '''Return array of shape functions derivatives for 
    given natural coordinate vector "rv" in case of
    - 1D and
    - 1st oder interpolation.'''
    r = rv
    return np.array([-1, 1])/2

def dshape_p1_2d(rv):
    '''Return array of shape functions derivatives for 
    given natural coordinate vector "rv" in case of
    - 2D and
    - 1st oder interpolation.'''
    r,s = rv
    return np.array([[-(1-s),-(1-r)],
                     [ (1-s),-(1+r)],
                     [ (1+s), (1+r)],
                     [-(1+s), (1-r)]])/4
    
def dshape_p1_3d(rv):
    '''Return array of shape functions derivatives for 
    given natural coordinate vector "rv" in case of
    - 3D and
    - 1st oder interpolation.'''
    r,s,t = rv
    return np.array([[-(1-s)*(1-t),-(1-r)*(1-t),-(1-r)*(1-s)],
                     [ (1-s)*(1-t),-(1+r)*(1-t),-(1+r)*(1-s)],
                     [ (1+s)*(1-t), (1+r)*(1-t),-(1+r)*(1+s)],
                     [-(1+s)*(1-t), (1-r)*(1-t),-(1-r)*(1+s)],
                     [-(1-s)*(1+t),-(1-r)*(1+t), (1-r)*(1-s)],
                     [ (1-s)*(1+t),-(1+r)*(1+t), (1+r)*(1-s)],
                     [ (1+s)*(1+t), (1+r)*(1+t), (1+r)*(1+s)],
                     [-(1+s)*(1+t), (1-r)*(1+t), (1-r)*(1+s)],])/8
    