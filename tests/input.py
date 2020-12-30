# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:44:40 2019

@author: adutz
"""

import aperitif

# BEGIN USER MATERIAL
from autograd import jacobian as dF
from autograd import numpy as np

def upsi(F,material,statevars_old=None,full_output=True):
    """St.Venant-Kirchhoff Material"""
    young, nu = material.parameters.all[:2]
    mu, gamma = young/(2*(1+nu)), young*nu/(1+nu)/(1-2*nu)

    E = (F.T@F-np.eye(3))/2
    psi = mu*np.trace(E@E) + gamma/2*np.trace(E)**2
    if full_output:
        return psi, statevars_old
    else:
        return psi 

umatdb = aperitif.constitution.init()
umatdb.ψ       =       upsi
umatdb.dψdF    =    dF(upsi)
umatdb.d2ψdFdF = dF(dF(upsi))
# END USER MATERIAL

# Inputfile
filename = 'block_2d_101'

# Model
model = aperitif.reader.load('inputfiles/'+filename+'.xlsx')
model.materials['rubber'].parameters.K = 20
model.loadcases[0].nincs = 10
model.constdb['user-rubber'] = umatdb

# Solver
history = aperitif.solver.job(model,state=None)

# write output file
aperitif.writer.xdmf('outputfiles/'+filename+'.xdmf',history,model)

