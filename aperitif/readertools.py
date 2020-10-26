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

def convert_material_parameters(mtype,mparam,mat):

    p = mat.parameters
    p.K = np.array([mparam[0]]).reshape(1,1)

    if mtype.lower() == 'invariants':
        # [bulk k ---- m01k m02k m03k 
        #         m10k m11k m12k m13k 
        #         m20k m21k m22k m23k
        #         m30k m31k m32k m33k]
        p.k  = np.array([mparam[1]]).reshape(1,1)
        p.mu = np.array(mparam[2:18]).reshape(1,4,4)
        
    if mtype.lower() == 'multi-k':
        # [K n mu_1 ... mu_n
        #       k_1 ...  k_n]
        p.n = int(mparam[1])
        
        # init k,mu
        p.k = np.zeros(p.n)
        p.mu = np.zeros((p.n,4,4))
        
        for n in range(p.n):
            p.mu[n,1,0] = mparam[2+n    ]
            p.k[n]      = mparam[2+n+p.n]
        
    return mat
    
def create_elements(elementdata):
    elements  = SimpleNamespace()
    
    # store element type "Quad4" as lowercase string "quad4"
    # store material labels (case sensitive)
    # store element connectivity matrix
    elements.types          = [etype.lower() for etype in elementdata.Type.values]
    elements.materiallabels = elementdata.Material.values
    elements.connectivities = elementdata.drop(columns=['Type', 'Material', 'Id']).values.astype(int)
    return elements
    
def create_materials(materialdata):
    # create material library from inputfile
    # init an empty dictionary, loop over materials in inputfile
    materials = {}
    for mlabel,mtype,mparam in zip(materialdata.Label.values, 
                                   materialdata.Type.values, 
                                   materialdata.drop(columns=['Label', 
                                                              'Type']).values):

        # create material as struct and save it in the materials-dictionary
        mat = SimpleNamespace()
        mat.label = mlabel
        mat.type = mtype
        mat.parameters = SimpleNamespace()
        mat.parameters.type  = mtype.lower()
        
        mat = convert_material_parameters(mtype,mparam,mat)
        materials[mlabel] = mat
    return materials
    
def create_tables(tabledata):
    # create table library from inputfile
    # init an empty dictionary, loop over tables in inputfile
    tables = {}
    
    for tid,tlabel,ttype in zip(tabledata.Id,tabledata.Label,tabledata.Type):

        if not np.isnan(tid):
            # create table as struct and save it in the tbl-dictionary
            tbl = SimpleNamespace()
            tbl.id = tid
            tbl.label = tlabel
            tbl.type = ttype.lower()
            x = tabledata['x'+str(int(tid))].values
            f = tabledata['f'+str(int(tid))].values
            tbl.points = (np.vstack((x,f)).T)[~np.isnan(x)]
            tbl.eval = interp1d(x[~np.isnan(x)],f[~np.isnan(x)])
            
            tables[tbl.label] = tbl
    return tables
    
def create_boundary_conditions(bcdata):
    # create boundary conditions library from inputfile
    # init an empty dictionary, loop over boundaries in inputfile
    boundaries = {}
    for blabel,bvalue,btable in zip(bcdata.Label,bcdata.Value,bcdata.Table):

        # create bc as struct and save it in the bc-dictionary
        bc = SimpleNamespace()
        bc.label = blabel
        bc.value = bvalue
        bc.table = btable
        
        boundaries[blabel] = bc
    return boundaries
    
def create_loadcases(lcdata):
    # create loadcase library from inputfile
    # init an empty list, loop over loadcases in inputfile
    loadcases = []
    _t0 = 0.0
    for llabel,ldur,lincs,liter,ltol in zip(lcdata.Label,
                                            lcdata.Duration,
                                            lcdata.Increments,
                                            lcdata.Iterations,
                                            lcdata.Tolerance):

        # create lc as struct and save it in the lc-dictionary
        lcase = SimpleNamespace()
        lcase.label = llabel
        lcase.duration = ldur
        lcase.nincs = lincs
        lcase.nmax = liter
        lcase.gtol = ltol
        lcase.start = _t0
        
        _t0 += lcase.duration
        
        loadcases.append(lcase)
    return loadcases
    
def create_dofs(dofdata):
    # create active (=free) / inactive (=prescribed, controlled) dof mask
    dof = SimpleNamespace()
    dof.inactive = dofdata.drop(columns=['Id']).values.astype(bool)
    dof.active = ~model.dof.inactive
    return dof
    
def init_const_table():
    tbl0 = SimpleNamespace() 
    tbl0.id = -1
    tbl0.label = 'None'
    tbl0.type = 'time'
    x = np.array([0])
    f = np.array([0])
    tbl0.points = np.vstack((x,f)).T
    tbl0.eval = lambda x: 1
    return tbl0
    
# create time dependent function for external displacements
def ext_displacements(t,model):
    ed = model._ext_displacements_0.copy()
    # evaluate boundary conditions at time t and save values
    for blabel, bc in model.boundaries.items():
        tbl_value = model.tables[bc.table].eval(t)
        value = ed[model._ext_displacements_b==blabel]
        ed[model._ext_displacements_b==blabel] = value*tbl_value
    return ed

def ext_forces(t,model):
    ef = model._ext_forces_0.copy()
    # evaluate boundary conditions at time t and save values
    for blabel, bc in model.boundaries.items():
        tbl_value = model.tables[bc.table].eval(t)
        value = ef[model._ext_forces_b==blabel]
        ef[model._ext_forces_b==blabel] = value*tbl_value
    return ef