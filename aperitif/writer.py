#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created     on 2020-12-26 22:00
Last Edited on 2020-12-26 22:00
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

import meshio
import numpy as np

def xdmf(filename,history,model):
    celldict = {'quad4' : 'quad', 
                'quad4p': 'quad',
                'axix4p': 'quad',
                'axiy4p': 'quad',
                'hexa8' : 'hexahedron',
                'hexa8p': 'hexahedron'}
    celltype = celldict[model.elements.types[0].lower()]
    cells = {
             celltype: model.elements.connectivities
            }
    
    print('\n\n\n')
    print('BEGIN Output-File')
    print('=================')
    print('')
    print('backend: aperitif XDMF writer')
    print('         using meshio - "https://github.com/nschloe/meshio"\n')
    
    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(model.nodes, cells)
        
        for inc,state in enumerate(history):
            displacements = state.uh
            int_forces = state.rh
            ext_forces = state.fh
            ext_displacements = state.uGh
            
            if displacements.shape[1] < 3:
                displacements = np.hstack((displacements, 
                                           np.zeros((displacements.shape[0],1))))
            #    stress = np.hstack((stress, 
            #                              np.zeros((displacements.shape[0],5))))
            cell_data = {celltype: 
                        {'Materials': (model.elements.materiallabels=='Rubber').astype(float).reshape(-1,1)}}
            
            point_data = {'External Force': ext_forces,
                          'External Displacement': ext_displacements,
                          'Reaction Force': int_forces,
                          #'Cauchy Stress': stress,
                          'Displacement': displacements}
            
            writer.write_data(state.t, 
                              point_data=point_data, 
                              cell_data=cell_data)
            print('    save Inc. (%d/%d)' % (inc, len(history)-1))
            
    print('\n\nJob finished.\n\n')

    cite_a = '''
    "Education is the most powerful weapon which you can 
     use to change the world." - Nelson Mandela'''
            
    cite_b = '''
    "It always seems impossible until 
     it's done." - Nelson Mandela'''
    citations = [cite_a,cite_b]
                
    print(citations[np.random.randint(0,2)])
          
    #input('\n\nPress ENTER to QUIT.\n\n')