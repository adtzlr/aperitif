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
from types import SimpleNamespace

import multiprocessing
from joblib import Parallel, delayed
from joblib import parallel_backend
from joblib import wrap_non_picklable_objects

import fem
import force

from scipy import sparse

def system_force_stiffness(u,args):
    
    model,state = args
    
    state.iteration = assemblage(u,
                                 x0=model.nodes,
                                 model=model,
                                 iteration=state.iteration)
    r = state.iteration.T.toarray().reshape(model.nnodes,model.ndim)
    K = state.iteration.K
    
    return r, K, state
    

def assemblage(u,x0,model,iteration=None):
    
    # reshape from flattened u to 'a,i'
    u = u.reshape(model.nnodes,model.ndim)
    
    # struct for iteration data
    # contains pre-calculated kinematics
    if iteration is None:
        iteration = SimpleNamespace()
        iteration.u = u
        iteration.kinematics = np.empty(model.nelements,
                                        dtype=object)
            
    #@wrap_non_picklable_objects #activate if error occurs
    @delayed
    def elem_results(zipped_input,parameters):
        '''Calculate nodal forces and tangent stiffnes per element'''
        
        # unzip element-specific input
        elabel,conn,mlabel,v0 = zipped_input
        
        # unzip global paramters
        model,u,x0 = parameters #,fem_e,force_e
        
        # get element and material informations
        element = fem.element[elabel]
        material = model.materials[mlabel].parameters
        
        # slice displacements and initial coordinates 
        # for current elemental nodes
        u_e = u[conn]
        x0_e = x0[conn]

        # calculate kinematics
        kin = force.kinematics(x0_e,u_e,v0,element)
            
        # calculate force, stiffness (and stress, strain, state var.)
        Tij_e, Kij_e, stress_e = force.stiffness_force(u_e,x0_e,v0,kin,element,material)
        return Tij_e.flatten(), Kij_e.flatten()
    
    num_cores = multiprocessing.cpu_count()#//2
    #num_cores = 1
    inputs = zip(model.elements.types,
                 model.elements.connectivities,
                 model.elements.materiallabels,
                 model.elements.v0)

    with parallel_backend('loky',n_jobs=num_cores):
        results = Parallel(verbose=0,max_nbytes=None)(elem_results(inp,(model,u,x0)) for inp in inputs)
    
    Tai = model.elements.Tai
    Kai = model.elements.Kai
    Kbj = model.elements.Kbj
    
    Tij = np.array([item[0] for item in results]).flatten()
    Kij = np.array([item[1] for item in results]).flatten()

    T = sparse.csr_matrix((Tij, (model.elements.Tai,np.zeros_like(Tai))),
                           shape=(model.ndof,1))
    
    K = sparse.csr_matrix((Kij, (Kai,Kbj)),
                           shape=(model.ndof,model.ndof))
        
    iteration.T = T
    iteration.K = K
    
    return iteration