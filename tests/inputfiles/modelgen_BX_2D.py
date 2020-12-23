# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:27:33 2020

@author: dutzi
"""

import numpy as np
import pandas as pd

import meshgen as mesh

file = 'block_2d_101'

d = 16
N,E = mesh.block_2d(d,d)

N = mesh.scale([0.5,0.5],N)
N = mesh.move([0.5,0.5],N)

nodes = N
connectivities = E

ControlledDOF = np.zeros_like(nodes,dtype=object)
Forces = np.zeros_like(nodes,dtype=object)

ControlledDOF[nodes[:,0]==1,0] = 'Displ-x'
ControlledDOF[nodes[:,0]==1,1] = 'Fixed'
#ControlledDOF[nodes[:,1]==1,1] = 'Force1'

# Symmetry Planes XY
ControlledDOF[nodes[:,0]==0,0] = 'Fixed'
ControlledDOF[nodes[:,1]==0,1] = 'Fixed'

BC_Labels = ['Fixed','Displ-x']
BC_Values = [0, 4]
BC_Tables = ['None','Table1']

TableData = [np.array([[0,0],
                       [1,1]])]

ndim = nodes.shape[1]
EType = [None, None, 'Quad4p', 'Hexa8p']

writer = pd.ExcelWriter(file+'.xlsx')

df0 = pd.DataFrame(nodes,columns=['x','y','z'][:ndim],dtype=float)
df0.index.name = 'Id'
df0.to_excel(writer,sheet_name='Nodes',index=True)

df1 = pd.DataFrame(connectivities,dtype=int)
df1.insert(0, 'Type', [EType[ndim]]*len(connectivities))
df1.insert(1, 'Material', ['Rubber']*len(connectivities))
df1.index.name = 'Id'
df1.to_excel(writer,sheet_name='Elements',index=True)

df3 = pd.DataFrame(TableData[0],columns=['x0','f0'],dtype=float)
df3.insert(0, 'Id', [0,None])
df3.insert(1, 'Label', ['Table1',''])
df3.insert(2, 'Type', ['Time',''])
df3.to_excel(writer,sheet_name='Tables',index=False)

df4 = pd.DataFrame(Forces,columns=['Fx','Fy','Fz'][:ndim]) #,dtype=float
df4.index.name = 'Id'
df4.to_excel(writer,sheet_name='Forces',index=True)

df5 = pd.DataFrame(ControlledDOF,columns=['u','v','w'][:ndim]) #,dtype=float
df5.index.name = 'Id'
df5.to_excel(writer,sheet_name='ControlledDOF',index=True)

df6 = pd.DataFrame(BC_Values,columns=['Value'])
df6.insert(0, 'Label', BC_Labels)
df6.insert(2, 'Table', BC_Tables)
df6.to_excel(writer,sheet_name='BoundaryConditions',index=False)

df7 = pd.DataFrame(np.array([[1,5000]]),dtype=float)
df7.insert(0, 'Label', ['Rubber'])
df7.insert(1, 'Type', ['Neo-Hooke'])
df7.to_excel(writer,sheet_name='Materials',index=False)

df8 = pd.DataFrame(np.array([[]]),dtype=float)
df8.insert(0, 'Label', ['Loadcase1'])
df8.insert(1, 'Duration', [1])
df8.insert(1, 'Increments', [10])
df8.insert(1, 'Iterations', [25])
df8.insert(1, 'Tolerance', [0.1])
df8.to_excel(writer,sheet_name='Loadcases',index=True)

writer.save()
writer.close()