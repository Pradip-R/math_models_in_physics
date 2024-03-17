import numpy as np
import sympy as sp
import math as mth
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from numpy.linalg import eig

func_innprd=lambda f,g: integrate.quad(lambda t: f(t)*g(t),0,1)[0]
func_norm=lambda f: np.sqrt(func_innprd(f,f))

#defining original basis (the homogeneous polynomials) and 
#initializing the dictionaries containing original basis and their 
#inner products with each other
no_basis= 5 #number of basis vectors considered
def orig_basis_vec(i): return lambda t: t**i
orig_bis=[orig_basis_vec(i) for i in range(no_basis+1)]
orth_bis={}
ips={} 

#computing the matrix with inner products of the homogeneous 
#polynomials with each other
def compute_inner_prods(k=no_basis,degree=no_basis):
    for i in range(k,degree+1):
        xi=orig_bis[i]
        ek=orth_bis[k]
        ips[(i,k)]=func_innprd(xi,ek)
    return

#defining function for graham schmidt orthogonalization
def gram_schmidt(k=no_basis,degree=no_basis):
    fk=lambda t: orig_bis[k](t)-np.sum([ips[(k,i)] * orth_bis[i](t) for i in range(k)])
    nfk=func_norm(fk)
    ek=lambda t: (1/nfk) * fk(t)
    orth_bis[k]=ek
    compute_inner_prods(k=k,degree=degree)
    return ek

#computing the orthonormal basis functions
for i in range(no_basis+1): gram_schmidt(k=i,degree=no_basis)

#defining function that converts a given function v into coordinates in the orthonormal basis 
#with 'degree' number of basis vector 
def compute_PUv_coeffs(v,degree):
    return [func_innprd(v,orth_bis[i]) for i in range(degree)]

x=sp.symbols('x')

g=compute_PUv_coeffs(lambda x: np.sin(np.pi*x),no_basis+1) #calculating the non homogeneous function in the orthonormal basis
#print(np.round(g,decimals =4)) 

#Calculating the differential, integral and functional operator in the orthonormal basis
A_d=np.zeros([no_basis+1,no_basis+1],float)
A_i=np.zeros([no_basis+1,no_basis+1],float)
A_f=np.zeros([no_basis+1,no_basis+1],float)
for i in range(0,no_basis+1):
    for j in range(0,no_basis+1): 
        A_d[(i,j)]=sp.lambdify(x,sp.integrate(orth_bis[i](x)*((x**2-x)*(sp.diff(orth_bis[j](x),x,x))-(1+x/2)*(sp.diff(orth_bis[j](x)))),x),"numpy")(1)
        A_i[(i,j)]=sp.lambdify(x,sp.integrate(orth_bis[i](x)*(-3/x*sp.integrate(orth_bis[j](x),x)),x),"numpy")(1)
        A_f[(i,j)]=sp.lambdify(x,sp.integrate(orth_bis[i](x)*(orth_bis[j](2*x)-orth_bis[j](1-x)),x),"numpy")(1)

A=A_d+A_i+A_f #matrix for total operator
inv_A=np.linalg.inv(A) #inverse of matrix A
#print(np.round(inv_A,decimals = 5))

f_orth=(np.dot(inv_A,g)) #function f in orthonormal basis
f_x=sum(f_orth[i]*orth_bis[i](x) for i in range(no_basis+1)) #f in homogeneous polynomial basis
#print(np.round(f_orth,decimals=4))
#print(f_x)
  
g_x=sum(g[i]*orth_bis[i](x) for i in range(no_basis+1)) #approximation for g in homogeneous basis
#print(g_x)

#Checking if function f satisfies the original equation by directly plugging it in
f_2x=sum((np.dot(inv_A,g))[i]*orth_bis[i](2*x) for i in range(no_basis+1)) #f(2x) to plug into the checking equation
f_1_x=sum((np.dot(inv_A,g))[i]*orth_bis[i](1-x) for i in range(no_basis+1))#f(1-x) to plug into the checking equation

check_g=sp.lambdify(x,x*(x-1)*sp.diff(f_x,x,x)-(1+x/2)*sp.diff(f_x,x)-3/x*sp.integrate(f_x,x)+f_2x-f_1_x,"numpy")
#print(sp.simplify(check_g(x)))


#Graphing the functions to check the validity of the basis approximation given the number of basis considered
def graph(funct, x_range, cl='r--'):
    y_range=[]
    for x in x_range:
        y_range.append(funct(x))
    plt.plot(x_range,y_range,cl)
    return

r=np.linspace(0.0001,1,80)
graph(check_g,r,cl='r-')
graph(lambda x: np.sin(np.pi*x),r,cl='b-')

w,v=eig(A)
#print('E-value:', w)
#print('E-vector', v)