import sympy as sym
from autograd import numpy as np
from autograd import jacobian
from autograd import elementwise_grad

from autograd import make_jvp
from autograd.differential_operators import make_jvp_reversemode

C = sym.MatrixSymbol('C',6,1).as_explicit()
D = sym.Matrix([[C[0,0],C[1,0],C[2,0]],
                [C[1,0],C[3,0],C[4,0]],
                [C[2,0],C[4,0],C[5,0]]])

J = sym.det(D)
Ciso = J**(-sym.Rational(1,3))*D

P4 = Ciso.diff(C)
#P4 = Ciso.reshape(9,1).jacobian(F.reshape(9,1))

pCiso = sym.lambdify(C,P4)

def fCiso(C):
    J = np.linalg.det(C)
    Ciso = J**(-1/3)*C
    return Ciso

f = jacobian(fCiso)

def gCiso(C):
    I = np.eye(3)
    I4 = (np.einsum('ij,kl->ikjl',I,I)+np.einsum('ij,kl->ilkj',I,I))/2
    #I4 = (np.tensordot(I,I,0).transpose([0,2,1,3])
    #     +np.tensordot(I,I,0).transpose([0,3,2,1]))/2
    return np.linalg.det(C)**(-1/3)*(I4-1/3*np.tensordot(C,np.linalg.inv(C),0))

np.random.seed(10346)
F = np.random.rand(3,3)
C = F.T@F

f_jvp_fast = make_jvp_reversemode(fCiso)(np.eye(2))

for basis in (np.array([1, 0]), np.array([0, 1])):
    val_of_f, col_of_jacobian = f_jvp_fast(basis)
    print(col_of_jacobian)