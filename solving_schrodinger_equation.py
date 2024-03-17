import numpy as np
import sympy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math as mth
from numpy.linalg import eig

func_innprd=lambda f,g: integrate.quad(lambda t: f(t)*g(t)*np.exp(-t**4),-30,30)[0]
func_norm=lambda f: np.sqrt(func_innprd(f,f))

x=sp.symbols('x')

#defining original basis (the homogeneous polynomials) and 
#initializing the dictionaries containing original basis and their 
#inner products with each other
no_basis= 8 #highest degree of basis vectors considered 
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
    
#print(orth_bis[n](x)) # prints nth orthonormal basis function 

#defining function that converts a given function v into coordinates in the orthonormal basis 
#with 'degree' number of basis vector 
def compute_PUv_coeffs(v,degree):
    return [func_innprd(v,orth_bis[i]) for i in range(degree)]

g=compute_PUv_coeffs(lambda x: np.sin(np.pi*x),no_basis+1) #calculating the non homogeneous function in the orthonormal basis
#print(np.round(g,decimals =4))

#Constructing hamiltonian operator in the orthonormal basis
H=np.zeros([no_basis+1,no_basis+1],float) # initilizing hamiltonain matrix

#assembling lambda functions involved in calculating matrix elements by numerical integral
def bis(i): return sp.lambdify(x,orth_bis[i](x),"numpy")
b=[bis(i) for i in range(no_basis+1)] #array of orthonormal basis functions
def df1_bis(i): return sp.lambdify(x,sp.diff(orth_bis[i](x),x),"numpy")
d1=[df1_bis(i) for i in range(no_basis+1)] #array with first derrivative of the orthonormal basis functions
def df2_bis(i): return sp.lambdify(x,sp.diff(orth_bis[i](x),x,x),"numpy")
d2=[df2_bis(i) for i in range(no_basis+1)]#array with second derrivative of the orthonormal basis functions
def df_bis(i): return sp.lambdify(x,df2_bis(i)(x)-4*x**3*df1_bis(i)(x)-6*x**2*bis(i)(x),"numpy")
d=[df_bis(i) for i in range(no_basis+1)]#array withj functions obtained when differential operator acts on basis functions

#Calculating the matrix
for i in range(0,no_basis+1):
    for j in range(0,no_basis+1): 
        H[i,j]=func_innprd(b[i],d[j])
        
#print(np.round(H,decimals = 5)) #Hamiltonian to 5 decimal places

w,v=eig(-H) # Calculating eigenvalue and eigenvector of the hamiltonian

#print(np.round(w,decimals=5))
#print(np.round(v,decimals=5))

#Wavefunctions in homogeneous polynomial basis
#psi_0=lambda x: np.exp((-x**4)/2)*sum(v.T[5,j]*b[j](x) for j in range(no_basis+1))
psi_1=lambda x: np.exp((-x**4)/2)*sum(v.T[6,j]*b[j](x) for j in range(no_basis+1))
#psi_1=lambda x: np.exp((-x**4)/2)*sum(v.T[4,j]*b[j](x) for j in range(no_basis+1))
#psi_1=lambda x: np.exp((-x**4)/2)*sum(v.T[6,j]*b[j](x) for j in range(no_basis+1))
#psi_1=lambda x: np.exp((-x**4)/2)*sum(v.T[3,j]*b[j](x) for j in range(no_basis+1))



#Normalizing the wavefunctions
N_0=func_norm(psi_0)

#Graphing the wavefunction
r=np.linspace(-2.5,2.5,800)
#plt.plot(r,w[2]+(psi_0(r)**2)/N_0,'r',label="\psi_0")
plt.plot(r,w[3]+(psi_1(r)**2)/N_0,'b',label="\psi_0")
plt.xlabel("x")
plt.ylabel("\psi")
#plt.plot(r,(w[4]-psi_1(r))/N_0,'g',label="\psi_0")
#plt.plot(r,(w[6]-psi_1(r))/N_0,'c',label="\psi_0")
#plt.plot(r,(w[3]-psi_1(r))/N_0,'y',label="\psi_0")
#plt.plot(r,w[5]+(psi_1(r))**2/N_1,'b')

#plt.plot(r,psi_2(r)/N_2)


