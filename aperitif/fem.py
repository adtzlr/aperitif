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

from functools import partial

import numpy as np
from aperitif import quadrature
from aperitif import shape
#import geometry

from types import SimpleNamespace

def system_ai(connectivity,nnodes,ndim):
    '''System (global) indices for sparse matrix.
    
    Example
    -------
    
    Planar Quad4-Element
    >>> nnodes = 4
    >>> ndim = 2
    >>> connectivity = np.array([7,8,5,4])
    
    repeat each entry of connectivity by `ndim`
    repeat `np.arange(ndim)` by number of nodes `nnodes`
    
    >>> ai0 = np.tile(connectivity,ndim)
    >>> ai0
    array([ 7, 7, 8, 8, 5, 5, 4, 4])
    
    >>> ai1 = np.repeat(np.arange(ndim),nnodes)
    >>> ai1
    array([ 0, 1, 0, 1, 0, 1, 0, 1])
    
    >>> np.vstack((ai0,ai1))
    array([[ 7, 7, 8, 8, 5, 5, 4, 4],
           [ 0, 1, 0, 1, 0, 1, 0, 1]])
    '''
    
    a = np.repeat(connectivity,ndim)
    i = np.tile(np.arange(ndim),nnodes)
    
    return a*ndim+i

def system_aibj(connectivity,nnodes,ndim):
    '''System (global) indices for sparse matrix.'''
    
    ai = system_ai(connectivity,nnodes,ndim)
    
    return np.repeat(ai,nnodes*ndim), np.tile(ai,nnodes*ndim)

def element_properties(label,ndim,nnodes,gaussdata,
                       shapefun,shapefunder,
                       shapemean=False,
                       axisymmetric=None):
    '''Element Properties with global parameters and shape functions.'''
    elemN = SimpleNamespace()
    elemN.label  = label
    elemN.ndim   = ndim
    elemN.nnodes = nnodes
    #elemN.nvoigt = (ndim*(1+ndim))//2
    elemN.gauss  = gaussdata
    elemN.shape  = SimpleNamespace()
    elemN.shape.mean = shapemean
    elemN.shape.h  = np.array([shapefun(ra) for ra in gaussdata.coords])
    elemN.shape.dhdr = np.array([shapefunder(ra) for ra in gaussdata.coords])
    elemN.shape.dhdr_center = shapefunder(gaussdata.center)
    elemN.npgauss = len(elemN.shape.h)
    #elemN.geometry = SimpleNamespace()
    #elemN.geometry.f_volume = geometry.volume_nd
    
    elemN.axisymmetric = SimpleNamespace()
    if axisymmetric is not None:
        elemN.axisymmetric.flag = True
        elemN.axisymmetric.axis = axisymmetric
    else:
        elemN.axisymmetric.flag = False
    
    #_ai = np.indices((nnodes,ndim)).reshape(2,-1)
    #ai = _ai.copy()
    
    elemN.ai   = partial(system_ai,  nnodes=nnodes,ndim=ndim)
    elemN.aibj = partial(system_aibj,nnodes=nnodes,ndim=ndim)

    return elemN

line2 = element_properties(label='line2', ndim=1, nnodes=2,
                           gaussdata   = quadrature.gauss_p1_1d,
                           shapefun    = shape.shape_p1_1d,
                           shapefunder = shape.dshape_p1_1d)

quad4 = element_properties(label='quad4', ndim=2, nnodes=4,
                           gaussdata   = quadrature.gauss_p1_2d,
                           shapefun    = shape.shape_p1_2d,
                           shapefunder = shape.dshape_p1_2d)
    
quad4p = element_properties(label='quad4p', ndim=2, nnodes=4,
                            gaussdata   = quadrature.gauss_p1_2d,
                            shapefun    = shape.shape_p1_2d,
                            shapefunder = shape.dshape_p1_2d,
                            shapemean  = True)

axix4p = element_properties(label='axix4p', ndim=2, nnodes=4,
                            gaussdata   = quadrature.gauss_p1_2d,
                            shapefun    = shape.shape_p1_2d,
                            shapefunder = shape.dshape_p1_2d,
                            shapemean   = True,
                            axisymmetric = 0)

axiy4p = element_properties(label='axiy4p', ndim=2, nnodes=4,
                            gaussdata   = quadrature.gauss_p1_2d,
                            shapefun    = shape.shape_p1_2d,
                            shapefunder = shape.dshape_p1_2d,
                            shapemean   = True,
                            axisymmetric = 1)

hexa8 = element_properties(label='hexa8', ndim=3, nnodes=8,
                           gaussdata   = quadrature.gauss_p1_3d,
                           shapefun    = shape.shape_p1_3d,
                           shapefunder = shape.dshape_p1_3d)

hexa8p = element_properties(label='hexa8p', ndim=3, nnodes=8,
                            gaussdata   = quadrature.gauss_p1_3d,
                            shapefun    = shape.shape_p1_3d,
                            shapefunder = shape.dshape_p1_3d,
                            shapemean   = True)

element = {}
element['line2' ]  = line2
element['quad4' ]  = quad4
element['quad4p']  = quad4p
element['axix4p']  = axix4p
element['axiy4p']  = axiy4p
element['hexa8' ]  = hexa8
element['hexa8p']  = hexa8p