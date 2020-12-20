#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-10-26 20:00
Last Edited on 2020-11-22 12:00

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

from types import SimpleNamespace
from functools import partial

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d

import fem
import geometry

import readertools

def read(filename):
    """
    Read an input file and return modeldata.
    
    Parameters
    ----------
    filename : str
        The full path and filename of the inputfile.
        
    Returns
    -------
    model : SimpleNamespace
        The generated model data as struct.
    """
    
    print('''
                               .__  __  .__  _____ 
    _____  ______   ___________|__|/  |_|__|/ ____\
    \__  \ \____ \_/ __ \_  __ \  \   __\  \   __\ 
     / __ \|  |_> >  ___/|  | \/  ||  | |  ||  |   
    (____  /   __/ \___  >__|  |__||__| |__||__|   
         \/|__|        \/                          
    _______________________________________________________
    Nonlinear Finite Elements Code for Structural Mechanics
    
    ''')
    print('    Version:   2020.12')
    print('    Author:    Dutzler A.')
    
    print('')
    print('READ INPUT FILE')
    print('===============\n')
    
    # read xlsx-inputfile to pandas dataframe
    df = pd.read_excel(filename,sheet_name=None)
    
    # get data of sheets from xlsx-file
    elementdata    = df['Elements']
    materialdata   = df['Materials']
    tabledata      = df['Tables']
    #tableptsdata   = df['Tablepoints']
    forcedata      = df['Forces']
    dofdata        = df['ControlledDOF']
    bcdata         = df['BoundaryConditions']
    lcdata         = df['Loadcases']
    
    # init model struct
    model = SimpleNamespace()
    
    model.filename  = filename
    print('Inputfile:', model.filename)

    # create material, table, boundary, dof and loadcase library from inputfile
    model.elements   = readertools.create_elements(elementdata)
    model.materials  = readertools.create_materials(materialdata)
    model.tables     = readertools.create_tables(tabledata)
    model.boundaries = readertools.create_boundary_conditions(bcdata)
    model.dof        = readertools.create_dofs(dofdata)
    model.loadcases  = readertools.create_loadcases(lcdata)
        
    # get nodal coordinates
    model.nodes = df['Nodes'].drop(columns=['Id']).values.astype(float)
    
    # global model data
    model.nnodes, model.ndim = model.nodes.shape
    model.nelements = len(model.elements.connectivities)
    model.ndof = model.nnodes*model.ndim
    
    print('\n')
    print('Global Model Data')
    print('-----------------')
    print('No. of Elements: %d' % model.nelements)
    print('No. of Nodes:    %d' % model.nnodes)
    print('No. of DOF:      %d' % model.ndof)
    print('\n')
          
    # postprocess table library
    # -------------------------
    
    # init dummy Table for constant-valued boundary Conditions
    tbl0 = readertools.init_const_table()
    
    # add dummy Tables for "None,Fixed,Constant"-labeled bc's
    keys = ['None','Fixed','Constant']
    for key in keys:
        model.tables[key] = tbl0

    # create external displacements and forces functions from inputfile
    # -----------------------------------------------------------------
    
    # get array with displacement boundary condition label for each dof
    model._ext_displacements_b = np.array(dofdata.drop(columns=['Id']).values, dtype=object)
    model._ext_displacements_0 = np.zeros_like(model.nodes)
    # get array with nodal force boundary label for each dof
    model._ext_forces_b = np.array(forcedata.drop(columns=['Id']).values, dtype=object)
    model._ext_forces_0 = np.zeros_like(model.nodes)
    
    # loop over boundary conditions (displacement and force) 
    # and replace labels with table multiplier values
    for blabel, bc in model.boundaries.items():
        model._ext_displacements_0[model._ext_displacements_b == blabel] = bc.value
        model._ext_forces_0[model._ext_forces_b == blabel] = bc.value
    
    # replace all non-prescribed dof's with NaN multiplier vaLue
    model._ext_displacements_0[model.dof.active] = np.nan
    
    # save external forces and disp. to model
    ext_forces        = partial(readertools.ext_forces,        model=model)
    ext_displacements = partial(readertools.ext_displacements, model=model)
    model.ext_forces        = ext_forces
    model.ext_displacements = ext_displacements
    
    # init elements
    model.elements.v0  = np.zeros(model.nelements)
    model.elements.Tai = np.zeros(model.nelements,dtype=object)
    model.elements.Kai = np.zeros(model.nelements,dtype=object)
    model.elements.Kbj = np.zeros(model.nelements,dtype=object)
    
    # calculate initial element volumes
    # and indices of the elemental internal force and tangent stiffness
    # components in the global system matrices
    for a,(etype,conn) in enumerate(zip(model.elements.types,
                                        model.elements.connectivities)):
                                        
        model.elements.v0[a]  = geometry.volume(model.nodes[conn])
        model.elements.Tai[a] = fem.element[etype.lower()].indices_ai(conn)
        
        aibj = fem.element[etype.lower()].indices_aibj(conn)
        
        model.elements.Kai[a] = aibj[0]
        model.elements.Kbj[a] = aibj[1]
        
    model.elements.Tai = np.concatenate(model.elements.Tai)
    model.elements.Kai = np.concatenate(model.elements.Kai)
    model.elements.Kbj = np.concatenate(model.elements.Kbj)

    return model