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
from numpy import sqrt

from types import SimpleNamespace

# Library with integration (gauss-) point weights and coordinates

gauss_p1_1d = SimpleNamespace()
gauss_p1_1d.coords = np.array([-1, 1])/sqrt(3)
gauss_p1_1d.weights = np.array([1,1])
gauss_p1_1d.center = np.array([0])

gauss_p1_2d = SimpleNamespace()
gauss_p1_2d.coords = np.array([[-1,-1],
                               [ 1,-1],
                               [ 1, 1],
                               [-1, 1]])/np.sqrt(3)
gauss_p1_2d.weights = np.array([1,1,1,1])
gauss_p1_2d.center = np.array([0,0])

gauss_p1_3d = SimpleNamespace()
gauss_p1_3d.coords = np.array([[-1,-1,-1],
                               [ 1,-1,-1],
                               [ 1, 1,-1],
                               [-1, 1,-1],
                               [-1,-1, 1],
                               [ 1,-1, 1],
                               [ 1, 1, 1],
                               [-1, 1, 1]])/sqrt(3)
gauss_p1_3d.weights = np.array([1,1,1,1,1,1,1,1])
gauss_p1_3d.center = np.array([0,0,0])