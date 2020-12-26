# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:23:23 2020

@author: adutz
"""

import sympy as sym

lam = sym.symbols(r'\lambda')

F = sym.MatrixSymbol('F',3,3).as_explicit()

C = F.T*F
w = C.eigenvals()