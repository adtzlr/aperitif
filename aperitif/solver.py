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

import copy
import numpy as np
#import pandas as pd
from types import SimpleNamespace as Struct

from numpy.linalg import norm
from scipy.sparse.linalg import spsolve as solve

from aperitif.assembly import system_force_stiffness as force

def init_state(model):
    state = Struct()
    state.lcase = None
    state.converged = True
    state.t   = 0
    state.uh  = np.zeros_like(model.nodes)
    state.uGh = state.uh.copy()
    state.rh  = state.uh.copy()
    state.fh  = state.uh.copy()
    state.iteration = None
    
    return state

def job(model,state=None):
    
    print('INIT JOB')
    print('--------')
    
    if state is None:
        state = init_state(model)
        
    history = [copy.copy(state)]
    
    print('\nLIST OF LOADCASES:')
    [print('* %s' % lcase.label) for lcase in model.loadcases]
    
    dtmin = 1e-4
    for lcase in model.loadcases:
        print('\n\n')
        print('\nBEGIN LOADCASE %s \n' % lcase.label)

        t  = lcase.start
        dt = abs(lcase.duration/lcase.nincs)
        
        print('Start      = %1.4e' % lcase.start)
        print('Duration   = %1.4e' % lcase.duration)
        if lcase.nincs < 0:
            print('Increments = %d (automatic ramp)'    % abs(lcase.nincs))
        else:
            print('Increments = %d'    % lcase.nincs)
        print('Timestep   = %1.4e' % dt)
        
        dof1 = model.dof.active
        dof0 = model.dof.inactive
        
        stop = False
        
        # INCREMENT LOOP
        while t < (lcase.start+lcase.duration):
            t = np.round(t+dt,15)
            if lcase.nincs < 0:
                dt = dt*1.2
            
            if t > lcase.duration:
                t = lcase.duration
                stop = True
                
            print('\nTime=%1.4e ' % t)
            
            f_k  = model.ext_forces(t)
            uG_k = model.ext_displacements(t)
            
            # init NR-Iteration
            uh = state.uh.copy()
            rh = state.rh.copy()
            
            u, uG = uh[dof1], uh[dof0]
            
            state.converged = False
            rhd, rhp, KTd, KTp, state = force(uh, (model,state))
            
            rh = rhd+rhp
            r, rG = rh[dof1], rh[dof0]

            # NEWTON ITERATIONS
            for n in range(lcase.nmax):
                #KT = KT.toarray()
                
                KT_11 = (KTd+KTp)[dof1.flatten(),:][:,dof1.flatten()]
                KT_10 = (KTd+KTp)[dof1.flatten(),:][:,dof0.flatten()]
                #KT_01 = KT[dof0.flatten(),:][:,dof1.flatten()]
                #KT_00 = KT[dof0.flatten(),:][:,dof0.flatten()]
                
                #KT_11i = np.linalg.inv(KT_11.toarray())
                #print(np.linalg.det(KT_11.toarray()))
                    
                duG = (uG_k[dof0] - uG)
                #du = KT_11i@(-r+f_k[dof1]-KT_10.dot(duG))
                du  = solve(KT_11, -(rhd+rhp)[dof1]+f_k[dof1]-KT_10.dot(duG),
                            )#use_umfpack=True
                
                duh = np.zeros_like(uh)
                duh[dof1] = du
                duh[dof0] = duG

                uh = uh + duh
                
                state.iteration = None
                rhd, rhp, KTd, KTp, state = force(uh, (model,state))
                
                rh = rhd+rhp
                u, uG = uh[dof1], uh[dof0]
                r, rG = rh[dof1], rh[dof0]
                
                gref = norm(f_k[dof1]) + norm(rG)
                g    = -r+f_k[dof1]
                
                ganorm = norm(g)
                grnorm = ganorm/gref
                print('   NR-IT. #%d | A-Res.=%1.2e | R-Res.=%1.2e ( Ref.F.=%1.2e )' % (n,ganorm,grnorm,gref))
                      
                if np.isnan(grnorm):
                    break
                
                if grnorm < lcase.gtol or ganorm < 1e-12:
                    state.converged = True
                    state.gnorm = grnorm
                    state.n = n
                    state.uh = uh
                    state.rh = rh
                    state.fh = f_k
                    state.uGh = uG_k
                    state.t = t
                    state.lcase = lcase
                    state.KT = KTd+KTp
                    break
            
            if state.converged:
                history.append(copy.copy(state))
            else:
                print('   NR-IT. process failed.')
                if lcase.nincs < 0:
                    t = t-dt
                    dt = dt/2
                    stop = False
                    if dt < dtmin:
                        break
                else:
                    break
            
            if stop:
                break
    return history