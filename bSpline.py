#AUTHORS:
#Alexander Israelsson
#Edvard Johansson
#David Petersson
#Linnéa Støvring-Nielsen

import scipy
import scipy.linalg
import numpy
import matplotlib.pyplot as plt
class BSpline(object):
    def __init__(self,dvalues,grid=None):
        #Grid should be same length as dvalues
        if(grid is None):
            grid = numpy.linspace(0,1,numpy.size(dvalues,1)-4)
            self.grid = numpy.insert(grid,[0,0,-1,-1],[grid[0],grid[0],grid[-1],grid[-1]])
        else:
            self.grid = grid #these are the u values, such that grid[i] = u_i
        self.d = dvalues #these are the corresponding d values, so that dvalues[i] = d_i, 

    def __call__(self, u):
        return self.findS(u)

    def findHotInterval(self, t):
        return (self.grid > t).argmax()-1
    
    @classmethod
    def basisFunction(cls,u,j):
        return lambda x: cls.N(u,3,j,x)
    @classmethod
    def N(cls, u, k, j, x):
        L = numpy.size(u) 
        if k == 0:
            j1 = j-1
            j2 = j
            if (j1 < 0):
                j1 = 0
            if (j1 >= L):
                j1 = L-1
            if (j2 < 0):
                j2 = 0
            if (j2 >= L):
                j2 = L-1
            return (x>=u[j1] and x<u[j2])
        else: 
            return cls.alpha(u, (j+k-1), (j-1),x)*cls.N(u, (k-1), j, x) + cls.alpha(u, j, (j+k),x)*cls.N(u, k-1, (j+1),x)
            
    def findS(self, t):
        i = self.findHotInterval(t)
        #intressanta d: dvalues[i-2:i+1]
        second1 = self.theD(self.d[:,i-2],self.d[:,i-1], i-2, i+1, t)
        second2 = self.theD(self.d[:,i-1],self.d[:,i], i-1, i+2, t)
        second3 = self.theD(self.d[:,i], self.d[:,i+1], i, i+3, t)
        last1 = self.theD(second1, second2, i-1, i+1, t)
        last2 = self.theD(second2, second3, i, i+2, t)
        return self.theD(last1, last2, i, i +1, t)

    def theD(self, d1, d2, i_l, i_r, t):
        alpha = self.alpha(self.grid, i_l, i_r,t)
        return (alpha*d1 + (1-alpha)*d2)

    @classmethod
    def alpha(cls, u, i_l, i_r, x):
        L = numpy.size(u)
        if (i_l < 0):
            i_l = 0
        if (i_l >= L):
            i_l = L-1

        if (i_r < 0):
            i_r = 0
        if (i_r >= L):
            i_r = L-1
        if (u[i_r]==u[i_l]):
            return 0
        return (u[i_r] - x)/(u[i_r] - u[i_l])

    def plot(self):
        points = 1000
        xy = numpy.zeros((2, points-1))
        for j, t in enumerate(numpy.linspace(self.grid[0], self.grid[-1],points)[0:-1]):
            xy[:,j] = self.findS(t)
        plt.plot(xy[0,:],xy[1,:])
        plt.plot(self.d[0,:],self.d[1,:])
        plt.scatter(self.d[0,:],self.d[1,:])

        plt.show()
    
    @classmethod
    def interpolation(cls, grid, xy): #We didn't think this was necessary, and this doesn't work.
        xi  = numpy.array((grid[:-2]+grid[1:-1]+grid[2:])/3)
        xi[-1] = xi[-1]-(1e-8)
        L = numpy.size(xi)
        # utvärdera splinevärdena på alla xchi (men det kanske inte behövs med alla xchi)?
        #print(xi)
        #print(xi[1])
        #print(BSpline.basisFunction(grid, 0, 2)(xi[1]))
        #print(BSpline.basisFunction(grid, 3, 2)(xi[0])) 
        #print(grid)
        N = numpy.array([[BSpline.basisFunction(grid, j)(xi[i]) for i in range(0,L)] for j in range(0,L)]).T
        # skapa ekvationssystemet
        f = (BSpline.basisFunction(grid, L))
       # for t in numpy.linspace(0,1):
        #        plt.scatter(t, f(t))
        #plt.show()
        #print(numpy.shape(N))
        numpy.set_printoptions(precision=3)
        #print(N)
        #print(xi)
        x = xy[:,0]
        print("xy shape @interpolation: " + str(numpy.shape(xy)))
        print("xy @interpolation: " + str(xy))
        print("N shape @interpolation: " + str(numpy.shape(N)))
        print("N @interpolation: " + str(N))
        dx = scipy.linalg.solve(N, xy[:,0])  #detta bör bytas mot solve_banded när vi fattar hur saker funkar. 	
        dy = scipy.linalg.solve(N, xy[:,1])  #detta också.
        d = numpy.array([dx,dy])
        return (xi,d)
