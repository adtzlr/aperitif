# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:44:40 2019

@author: adutz
"""

import aperitif

# Inputfile
filename = 'UX_3D'

# Model
model = aperitif.reader.load('inputfiles/'+filename+'.xlsx')
model.materials['Rubber'].parameters.K = 1000
model.loadcases[0].nincs = 200

# Solver
history = aperitif.solver.job(model,state=None)

# write output file
aperitif.writer.xdmf('outputfiles/'+filename+'.xdmf',history,model)

