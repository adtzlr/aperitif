# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:15:39 2020

@author: dutzi
"""

import numpy as np

def block_2d(ne,me,a=0):
    
    n = ne+1
    m = me+1
    
    A = (a+np.arange(n*m)).reshape(n,m)
    
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,m)
    X,Y = np.meshgrid(x,y,indexing='ij')
    #print(A,X.flatten(),Y.flatten())
    
    nodes = np.vstack((X.flatten(), Y.flatten())).T
    connectivity = np.zeros((0,4),dtype=int)

    for i in range(n-1):
        for j in range(m-1):
            conn = A[i:i+2,j:j+2].take([0,2,3,1])
            #print('n',nodes[conn])
            connectivity = np.append(connectivity, conn.reshape(1,-1), axis=0)
            
    return nodes, connectivity

def transform(F,x0):
    return (F@x0.T).T

def scale(c,x0):
    x = np.zeros_like(x0)
    for i,ci in enumerate(c):
        x[:,i] = ci*x0[:,i]
    return x

def move(u,x0):
    x = np.zeros_like(x0)
    for i,ui in enumerate(u):
        x[:,i] = ui+x0[:,i]
    return x

def rotate2(phi_deg,x0,p=(0,0),axis=None):
    phi = np.deg2rad(phi_deg)
    R2 = np.array([[np.cos(phi),-np.sin(phi)],
                   [np.sin(phi), np.cos(phi)]])
    #R3 = np.pad(R2,(1,1),'constant',constant_values=(0,0))
    #R3[2,2] = 1
    xt = np.zeros_like(x0)
    
    for i,pi in enumerate(p):
        xt[:,i] = -pi+x0[:,i]
    xr = (R2@xt.T).T
    for i,pi in enumerate(p):
        xr[:,i] = pi+xr[:,i]
    return xr

def rotate3(phi_deg,x0,p=(0,0,0),axis=2):
    phi = np.deg2rad(phi_deg)
    R2 = np.array([[np.cos(phi),-np.sin(phi)],
                   [np.sin(phi), np.cos(phi)]])
    R3 = np.insert(R2,axis,0, axis=0)
    R3 = np.insert(R3,axis,0, axis=1)
    R3[axis,axis] = 1
    xt = np.zeros_like(x0)
    
    for i,pi in enumerate(p):
        xt[:,i] = -pi+x0[:,i]
    xr = (R3@xt.T).T
    for i,pi in enumerate(p):
        xr[:,i] = pi+xr[:,i]
    return xr

def expand(nodes_2d, connectivity_2d, dz=1, repeat=1):
    #nodes = np.array([np.pad(nodes_2d, (0,1), 'constant', constant_values=(0,dzi))[:-1] for dzi in np.linspace(0,dz,1+repeat)]).reshape((1+repeat)*nodes_2d.shape[0],-1)
    nodes = np.array([np.insert(nodes_2d,2,dzi,axis=1) for dzi in np.linspace(0,dz,1+repeat)]).reshape((1+repeat)*nodes_2d.shape[0],-1)
    a = np.max(connectivity_2d)+1
    conns = [connectivity_2d+b for b in a*np.arange(repeat+1)]
    connectivity = np.zeros((0,conns[0].shape[1]*2),dtype=int)
    for b in range(repeat):
        connectivity = np.append(connectivity, np.hstack((conns[b],conns[b+1])), axis=0)
    return nodes, connectivity

def revolve(nodes_2d, connectivity_2d, dphi=10, repeat=18):
    nodes_plane = np.insert(nodes_2d,2,0, axis=1)
    
    nodes = np.array([rotate3(phi,nodes_plane,axis=0) for phi in np.linspace(0,dphi*repeat,1+repeat)]).reshape((1+repeat)*nodes_plane.shape[0],-1)
    a = np.max(connectivity_2d)+1
    conns = [connectivity_2d+b for b in a*np.arange(repeat+1)]
    connectivity = np.zeros((0,conns[0].shape[1]*2),dtype=int)
    for b in range(repeat):
        connectivity = np.append(connectivity, np.hstack((conns[b],conns[b+1])), axis=0)
    return nodes, connectivity

if __name__ == '__main__':

    N,E = block_2d(2,3)
    
    F = np.array([[2,0.1],
                  [0,1]])
        
    n = scale([2,1],N)
    nn = move([0,1],n)
    
    n3,e3 = expand(N,E)
    
    nr = rotate2(10,N,[1,1])
    na,ea = revolve(N,E)
    
    #print(np.vstack((X[0:2,0:2].flatten(), Y[0:2,0:2].flatten())).T)