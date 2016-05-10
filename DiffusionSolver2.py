
# coding: utf-8

# In[7]:

import numpy as np
import operator
from numpy import linalg as LA
from numpy.linalg import inv
from scipy import linalg
import time


# In[8]:

def Gausseidel(A,x,b,tol=0.000001,sc=0.000001,iteration = 1000):
    # A,x,b - matrices defining the problem
    # tol - absolute tolerance for convergence (default set to 10^-6)
    # sc - stopping criterion (relative error) for convergence (default set to 10^-6)
    # iteration - (optional) choose how many iterations to run
    # iteration will stop at the first condition that is satisfied whether it be 
    # absolute tolerance or stopping criterion. 
    # If you want to pick a specific criteria for convergence,set the others to -1
    xnew = np.copy(x)
    conv = [1]*np.size(x)
    rel = [1]*np.size(x)
    iterations = 0
    while np.amax(conv) > tol and np.amax(rel) > sc and iterations != iteration:
        xold = np.copy(xnew)
        iterations = iterations + 1
        for i in range(np.size(x)):
            firstsum = 0
            secondsum = 0
            for j in range(0, i):
                firstsum = firstsum + A[i][j]*xnew[j]
            for j in range(i+1, np.size(x)):
                secondsum = secondsum + A[i][j]*xold[j]
            xnew[i] = 1.0/A[i][i]*(b[i] - firstsum - secondsum)
        for i in range(0, np.size(x)):
            conv[i] = abs(xnew[i] - xold[i])
            if xnew[i] != 0:
                rel[i] = abs(xnew[i] - xold[i])/(xnew[i])
            else:
                rel[i] = 0
    return xnew, iterations


# In[9]:

def differencematrix(V):
    # V - matrix defining the grid
    # Solves for the grid lengths in each cell, both y and x
    dif = [[0]*(np.size(V[0])-1), [0]*(np.size(V[1])-1)]
    for i in range(np.size(V[0])-1):
        dif[0][i] = abs(V[0][i+1] - V[0][i])
    for i in range(np.size(V[1])-1):
        dif[1][i] = abs(V[1][i+1] - V[1][i])
    return dif
    


# In[10]:

def finitevolume(V,D,S,absorption,T=0,B=0,L=0,R=0):
    # V,D,S, absorption - matrices defining the problem. V sets the grid, D sets cell-centered diffusion constants
    # S sets cell-centered sources, absorption sets cell-centered macroscopic absorption constants
    # L,R,T,B set boundary conditions on the problem, 0 is set for vacuum boundary on that side while 1 is for reflecting boundary
    eps = differencematrix(V)[1]
    sig = differencematrix(V)[0]
    sources = [[0.0]*np.size(V[0]) for i in range(np.size(V[1]))]
    fluxeq = [None]*np.size(V[0])*np.size(V[1])
    for j in range(np.size(V[1])):
        for i in range(np.size(V[0])):
            if i == 0 or j == 0 or i == np.size(V[0])-1 or j == np.size(V[1])-1:
                #applying boundary conditions for multiple cases
                if i == 0 and L==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif i == np.size(V[0])-1 and R==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif j == 0 and B==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif j == np.size(V[1])-1 and T==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif i == 0 and j==0 and L==0 and B==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif i == np.size(V[0])-1 and j==0 and R==0 and B==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif j == np.size(V[1])-1 and i==0 and L==0 and T==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif j == np.size(V[1])-1 and i==np.size(V[0])-1 and T==0 and R==0:
                    aL = aR = aB = aT = 0
                    aC = 1
                    sources[j][i] = 0.0
                elif i == 0  and j == 0 and B==1 and L==1:
                    aL = 0
                    aR = -1.0*(D[j][i]*sig[i])/(2.0*eps[j])
                    aB = 0
                    aT = -1.0*(D[j][i]*eps[j])/(2.0*sig[i])
                    aC = (absorption[j][i]*0.25*eps[j]*sig[i]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j][i]*0.25*eps[j]*sig[i]
                elif i == np.size(V[0])-1  and j == 0 and B==1 and R==1:
                    aL = -1.0*(D[j][i-1]*sig[i-1])/(2.0*eps[j])
                    aR = 0
                    aB = 0
                    aT = -1.0*(D[j][i-1]*eps[j])/(2.0*sig[i-1])
                    aC = (absorption[j][i-1]*0.25*eps[j]*sig[i-1]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j][i-1]*0.25*eps[j]*sig[i-1]
                elif i == np.size(V[0])-1  and j == np.size(V[1])-1 and T==1 and R==1:
                    aL = -1.0*(D[j-1][i-1]*sig[i-1])/(2.0*eps[j-1])
                    aR = 0
                    aB = -1.0*(D[j-1][i-1]*eps[j-1])/(2.0*sig[i-1])
                    aT = 0
                    aC = (absorption[j-1][i-1]*0.25*eps[j-1]*sig[i-1]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j-1][i-1]*0.25*eps[j-1]*sig[i-1]
                elif i == 0  and j == np.size(V[1])-1 and T==1 and L==1:
                    aL = 0
                    aR = -1.0*(D[j-1][i]*sig[i])/(2.0*eps[j-1])
                    aB = -1.0*(D[j-1][i]*eps[j-1])/(2.0*sig[i])
                    aT = 0
                    aC = (absorption[j-1][i]*0.25*eps[j-1]*sig[i]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j-1][i]*0.25*eps[j-1]*sig[i]    
                elif i == 0 and L==1 and j < np.size(V[1])-1:
                    aL = 0
                    aR = -1.0*(D[j-1][i]*eps[j-1] + D[j][i]*eps[j])/(2.0*sig[i])
                    aB = -1.0*(D[j-1][i]*sig[i])/(2.0*eps[j-1])
                    aT = -1.0*(D[j][i]*sig[i])/(2.0*eps[j])
                    aC = (absorption[j][i]*0.25*eps[j]*sig[i] + absorption[j-1][i]*0.25*eps[j-1]*sig[i]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j][i]*0.25*eps[j]*sig[i] + S[j-1][i]*0.25*eps[j-1]*sig[i]
                elif i == np.size(V[0])-1 and R==1 and j < np.size(V[1])-1:
                    aL = -1.0*(D[j-1][i-1]*eps[j-1] + D[j][i-1]*eps[j])/(2.0*sig[i-1])
                    aR = 0
                    aB = -1.0*(D[j-1][i-1]*sig[i-1])/(2.0*eps[j-1])
                    aT = -1.0*(D[j][i-1]*sig[i-1])/(2.0*eps[j])
                    aC = (absorption[j][i-1]*0.25*eps[j]*sig[i-1] + absorption[j-1][i-1]*0.25*eps[j-1]*sig[i-1]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j][i-1]*0.25*eps[j]*sig[i-1] + S[j-1][i-1]*0.25*eps[j-1]*sig[i-1]
                elif j == 0 and B==1 and i < np.size(V[0])-1:
                    aL = -1.0*(D[j][i-1]*eps[j])/(2.0*sig[i-1])
                    aR = -1.0*(D[j][i]*eps[j])/(2.0*sig[i])
                    aB = 0
                    aT = -1.0*(D[j][i-1]*sig[i-1] + D[j][i]*sig[i])/(2.0*eps[j])
                    aC = (absorption[j][i-1]*0.25*eps[j]*sig[i-1] + absorption[j][i]*0.25*eps[j]*sig[i]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j][i-1]*0.25*eps[j]*sig[i-1] + S[j][i]*0.25*eps[j]*sig[i]
                elif j == np.size(V[1])-1 and T==1 and i < np.size(V[0])-1:
                    aL = -1.0*(D[j-1][i-1]*eps[j-1])/(2.0*sig[i-1])
                    aR = -1.0*(D[j-1][i]*eps[j-1])/(2.0*sig[i])
                    aB = -1.0*(D[j-1][i-1]*sig[i-1] + D[j-1][i]*sig[i])/(2.0*eps[j-1])
                    aT = 0
                    aC = (absorption[j-1][i-1]*0.25*eps[j-1]*sig[i-1] + absorption[j-1][i]*0.25*eps[j-1]*sig[i]) - (aL + aR + aB + aT)
                    sources[j][i] = S[j-1][i-1]*0.25*eps[j-1]*sig[i-1] + S[j-1][i]*0.25*eps[j-1]*sig[i]
            else:
                # solving for cases outside of the boundary conditions
                aL = -1.0*(D[j-1][i-1]*eps[j-1] + D[j][i-1]*eps[j])/(2.0*sig[i-1])
                aR = -1.0*(D[j-1][i]*eps[j-1] + D[j][i]*eps[j])/(2.0*sig[i])
                aB = -1.0*(D[j-1][i-1]*sig[i-1] + D[j-1][i]*sig[i])/(2.0*eps[j-1])
                aT = -1.0*(D[j][i-1]*sig[i-1] + D[j][i]*sig[i])/(2.0*eps[j])
                aC = (absorption[j-1][i-1]*0.25*eps[j-1]*sig[i-1] + absorption[j][i-1]*0.25*eps[j]*sig[i-1] + absorption[j][i]*0.25*eps[j]*sig[i] + absorption[j-1][i]*0.25*eps[j-1]*sig[i]) - (aL + aR + aB + aT)
                sources[j][i] = S[j-1][i-1]*0.25*eps[j-1]*sig[i-1] + S[j][i-1]*0.25*eps[j]*sig[i-1] + S[j][i]*0.25*eps[j]*sig[i] + S[j-1][i]*0.25*eps[j-1]*sig[i]
            source = np.asarray(sources).flatten()
            flux = [0]*np.size(source)
            flux[j*np.size(V[0]) + i] = aC
            if j*np.size(V[0]) + i + 1 < np.size(flux):
                flux[j*np.size(V[0]) + i + 1] = aR
            if j*np.size(V[0]) + i - 1 >= 0:
                flux[j*np.size(V[0]) + i - 1] = aL
            if j*np.size(V[0]) + i + np.size(V[0]) < np.size(flux):
                flux[j*np.size(V[0]) + i + np.size(V[0])] = aT
            if j*np.size(V[0]) + i - np.size(V[0]) >= 0:
                flux[j*np.size(V[0]) + i - np.size(V[0])] = aB
            fluxeq[j*np.size(V[0]) + i] = flux
    test = source
    finalflux, iterations = Gausseidel(fluxeq,test,source)
    return finalflux, iterations
    
            


# In[11]:

def inputFile(fileName):
    inputfile = open(fileName,'r')

    while True:
        c = inputfile.readline()
        if c == 'grid (x and y)\n':
            c = inputfile.readline()
            column = int(c.split()[0])
            row = int(c.split()[1])
            D = [[0.0]*(column-1) for i in range(row-1)] 
            absorption = [[0.0]*(column-1) for i in range(row-1)] 
            S = [[0.0]*(column-1) for i in range(row-1)] 
            if column < 0 or row < 0:
                raise ValueError('Invalid grid size '.format())
        if c == 'Boundary Top (Reflecting or Vacuum)\n':
            c = inputfile.readline()
            if c == 'Reflecting\n':
                T = 1
            elif c == 'Vacuum\n':
                T = 0
            else:
                raise ValueError('Invalid boundary condition: Only "Reflecting and Vacuum allowed'.format())
        if c == 'Boundary Bottom (Reflecting or Vacuum)\n':
            c = inputfile.readline()
            if c == 'Reflecting\n':
                B = 1
            elif c == 'Vacuum\n':
                B = 0
            else:
                raise ValueError('Invalid boundary condition: Only "Reflecting and Vacuum allowed'.format())
        if c == 'Boundary Right (Reflecting or Vacuum)\n':
            c = inputfile.readline()
            if c == 'Reflecting\n':
                R = 1
            elif c == 'Vacuum\n':
                R = 0
            else:
                raise ValueError('Invalid boundary condition: Only "Reflecting and Vacuum allowed'.format())
        if c == 'Boundary Left (Reflecting or Vacuum)\n':
            c = inputfile.readline()
            if c == 'Reflecting\n':
                L = 1
            elif c == 'Vacuum\n':
                L = 0
            else:
                raise ValueError('Invalid boundary condition: Only "Reflecting and Vacuum allowed'.format())
        if c == 'xstart xend\n':
            c = inputfile.readline()
            xstart = float(c.split()[0])
            xend = float(c.split()[1])
            x = np.linspace(xstart,xend,column).tolist()
            if xstart >= xend:
                raise ValueError('Invalid x interval'.format())
        if c == 'ystart yend\n':
            c = inputfile.readline()
            ystart = float(c.split()[0])
            yend = float(c.split()[1])
            y = np.linspace(ystart,yend,row).tolist()
            if ystart >= yend:
                raise ValueError('Invalid y interval '.format())
        if c == 'diffusion coefficient (Use cell-centered definition: Matrix of size y-1 by x-1 -- with row of y-1 and column of x-1)\n':
#             print row-1, column-1
            for i in range(row-1):
                c = inputfile.readline()
                for j in range(column-1):
                    try:
                        D[row -2- i][j] = float(c.split()[j])
                    except:
                        raise Exception("Diffusion Coefficient matrix dimension doesn't match the grid size")
                    if D[row -2- i][j] < 0:
                        raise ValueError('Diffusion Coefficient matrix is invalid (No negative value!))'.format())
        if c == 'absorption (Use cell-centered definition: Matrix of size y-1 by x-1 -- with row of y-1 and column of x-1)\n':
            for i in range(row-1):
                c = inputfile.readline()
                for j in range(column-1):
                    try:
                        absorption[row -2 - i][j] = float(c.split()[j])
                    except:
                        raise Excepetion("Absorption matrix dimension doesn't match the grid size")
                    if absorption[row -2 - i][j] < 0.0:
                        raise ValueError('Absorption matrix is invalid (No negative value!))'.format())
        if c == 'source (Use cell-centered definition: Matrix of size y-1 by x-1 -- with row of y-1 and column of x-1)\n':
            for i in range(row-1):
                c = inputfile.readline()
                for j in range(column-1):
                    try:
                        S[row -2 - i][j] = float(c.split()[j])
                    except:
                        raise Excepetion("Source matrix dimension doesn't match the grid size")
                    if S[row - 2 - i][j] < 0:
                        raise ValueError('Source matrix is invalid (No negative value!))'.format())
        if not c:
#          print "End of file"
          break
#     print np.size(D)
#     print np.size(S)
#     print np.size(absorption)
#     print np.size(x)
#     print np.size(y)
#     print S
#     print D
#     print absorption
#     print T,B,R,L
        
    return D, absorption, S, x, y, T, B, L, R


# In[16]:

result = inputFile('InputFile.txt')
D3 = result[0]
# print np.size(D3)
# print "\n"
# print "\n"
absorption3 = result[1]
# print np.size(absorption3)
# print "\n"
# print "\n"
S3 = result[2]
# print np.size(S3)
# print "\n"
# print "\n"
V3 = [result[3], result[4]]
TBC = result[5]
BBC = result[6]
LBC = result[7]
RBC = result[8]

# print np.size(V3[1])
# print "\n"
# print "\n"
start_time = time.time()
sol3 = finitevolume(V3,D3,S3,absorption3,TBC,BBC,LBC,RBC)
runtime = (time.time() - start_time)
plotsol3 = np.reshape(sol3[0], (np.size(V3[1]), np.size(V3[0])))
runtimestr = str(runtime)

target = open('2DDifussionSolverOutput.txt', 'w+')
target.write('2D-Diffusion Solver Output File\nVersion 2.0\nAuthors               : David Smith, Joshua Cahyadi, Eduardo Zagal\n')
target.write("Current date & time   : " + time.strftime("%c") + '\n')
target.write('Execution time        : ' + runtimestr + 's\n')
target.write('Input file            : InputFile.txt\n')
target.write('Output file           : 2DDifussionSolverOutput.txt\n')
target.write('================================================================================\n')
target.write('Number of Iterations: ' + str(sol3[1]) +'\n\n')
target.write('Flux Solution Matrix:\n' + str(plotsol3) +'\n\n')
target.write('X vector:\n' + str(V3[0]) +'\n\n')
target.write('Y vector:\n' + str(V3[1]) +'\n\n')
target.close()


# In[17]:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')
X = V3[0]
Y = V3[1]
X, Y = np.meshgrid(X, Y)
Z = plotsol3
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('flux')
plt.show()


# In[ ]:



