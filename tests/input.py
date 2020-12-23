# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:44:40 2019

@author: adutz

  _____   _____   ____    ___    ____   _   _   _____ 
 |  ___| | ____| / ___|  |_ _|  / ___| | | | | |_   _|
 | |_    |  _|   \___ \   | |  | |  _  | |_| |   | |  
 |  _|   | |___   ___) |  | |  | |_| | |  _  |   | |  
 |_|     |_____| |____/  |___|  \____| |_| |_|   |_|  
                                                      

FESIGHT - (F)inite (E)lements in(SIGHT)
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

import aperitif

# Inputfile
filename = 'block_2d_101'

# Model
model = aperitif.reader.load('inputfiles/'+filename+'.xlsx')

# Solver
history = aperitif.solver.job(model,state=None)

# write output file
aperitif.writer.xdmf('outputfiles/'+filename+'.xdmf',history,model)